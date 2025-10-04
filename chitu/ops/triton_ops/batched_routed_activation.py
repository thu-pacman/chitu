# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import triton
import triton.language as tl

from chitu.utils import ceil_div

# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 SGLang Team
# SPDX-SnippetCopyrightText: 2025 vLLM Team
# SPDX-SnippetCopyrightText: 2025 Qingcheng.AI
# SDPX—SnippetName: The Triton implementation of fused MoE kernel
#
# The Triton implementation of fused MoE kernel is originally from vLLM
# (https://github.com/vllm-project/vllm/blob/ed6e9075d31e32c8548b480a47d1ffb77da1f54c/vllm/model_executor/layers/fused_moe/fused_moe.py),
# licensed under Apache 2.0, which is further based on SGLang
# (https://github.com/sgl-project/sglang/commit/ba5112ff691d791a9e38c6c71f59324a5fcb49d0),
# licensed under Apache 2.0. Qingcheng.AI added soft fp4/fp8 support to this implementation.


@triton.jit
def batched_routed_activation_indexed_to_expert_block_indexed_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)

    start_idx = pid * tokens_per_thread

    off_c = (pid + 1) * num_experts

    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


@triton.jit
def batched_routed_activation_indexed_to_expert_block_indexed_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)

    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def batched_routed_activation_indexed_to_expert_block_indexed_stage3(
    total_blocks_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    last_n_blocks_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_n_blocks_cumsum += tl.cdiv(token_cnt, block_size)
        tl.store(cumsum_ptr + i, last_n_blocks_cumsum * block_size)
    tl.store(total_blocks_post_pad_ptr, last_n_blocks_cumsum)


@triton.jit
def batched_routed_activation_indexed_to_expert_block_indexed_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)


def batched_routed_activation_indexed_to_expert_block_indexed_triton_detail(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_block_post_pad: torch.Tensor,
    cumsum_buffer: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    grid = (num_experts,)
    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    tokens_per_thread = ceil_div(numel, num_experts)

    batched_routed_activation_indexed_to_expert_block_indexed_stage1[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
    )
    batched_routed_activation_indexed_to_expert_block_indexed_stage2[grid](
        tokens_cnts,
        num_experts,
    )
    batched_routed_activation_indexed_to_expert_block_indexed_stage3[(1,)](
        num_block_post_pad,
        tokens_cnts,
        cumsum_buffer,
        num_experts,
        block_size,
    )
    batched_routed_activation_indexed_to_expert_block_indexed_stage4[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum_buffer,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
    )


def batched_routed_activation_indexed_to_expert_block_indexed_triton(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # The case of max_num_m_blocks: Suppose the first `num_experts` tokens each
    # routed to a different expert, each occupying one block. For the reset
    # `topk_ids.numel() - num_experts` tokens, every `block_size` tokens contributes
    # to one block
    max_num_m_blocks = (
        max(num_experts, topk_ids.numel())
        + max(topk_ids.numel() - num_experts, 0) // block_size
    )
    sorted_ids = torch.empty(
        (max_num_m_blocks * block_size,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    # Expert ids must be zeroed out to prevent index out of bounds error while
    # mapping global expert ids to local expert ids in expert parallelism.
    expert_ids = torch.zeros(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_block_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    cumsum_buffer = torch.zeros(
        (num_experts + 1,), dtype=torch.int32, device=topk_ids.device
    )
    batched_routed_activation_indexed_to_expert_block_indexed_triton_detail(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_block_post_pad,
        cumsum_buffer,
    )
    return sorted_ids.view(max_num_m_blocks, block_size), expert_ids, num_block_post_pad


# SPDX-SnippetEnd


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 ModelTC
# SDPX—SnippetName: lightllm deepep_scatter_gather
#
# Parts of this file are adapted from lightllm (https://github.com/ModelTC/lightllm/blob/main/lightllm/common/fused_moe/deepep_scatter_gather.py)
# Modified and adjusted to fit the requirements of this project
@triton.jit
def _fwd_kernel_ep_scatter_1(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)

    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + offset_cumsum,
        mask=offset_cumsum < num_experts,
        other=0,
    )
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
    tl.store(expert_start_loc + offset_cumsum, cumsum, mask=offset_cumsum < num_experts)

    cur_expert_start = tl.load(expert_start_loc + cur_expert)
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)

    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)

    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E, num_stages=4):
        tl.store(
            m_indices_start_ptr + start_m + off_expert,
            cur_expert,
        )


@triton.jit
def _fwd_kernel_ep_scatter_2(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    topk_num: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE_PAD: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)

    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE

    offset_in_s = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)
    mask_s = offset_in_s < SCALE_HIDDEN_SIZE

    for token_id in range(start_token_id, total_token_num, grid_num):
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)
        to_copy_s = tl.load(
            recv_x_scale + token_id * recv_x_scale_stride0 + offset_in_s, mask=mask_s
        )

        for topk_index in tl.range(0, topk_num, 1, num_stages=4):
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)
            if expert_id >= 0:
                dest_token_index = tl.atomic_add(expert_start_loc + expert_id, 1)
                tl.store(
                    output_index + token_id * output_index_stride0 + topk_index,
                    dest_token_index,
                )
                output_tensor_ptr = (
                    output_tensor + dest_token_index * output_tensor_stride0
                )
                output_tensor_scale_ptr = (
                    output_tensor_scale + dest_token_index * output_tensor_scale_stride0
                )
                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)
                tl.store(output_tensor_scale_ptr + offset_in_s, to_copy_s, mask=mask_s)


@torch.no_grad()
def ep_scatter(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
):
    BLOCK_E = 128  # token num of per expert is aligned to 128
    BLOCK_D = 128  # block size of quantization
    num_warps = 8
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    # grid = (triton.cdiv(hidden_size, BLOCK_D), num_experts)
    grid = num_experts

    assert m_indices.shape[0] % BLOCK_E == 0

    _fwd_kernel_ep_scatter_1[(grid,)](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=num_warps,
        BLOCK_E=BLOCK_E,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )

    grid = min(recv_topk.shape[0], 1024 * 8)

    _fwd_kernel_ep_scatter_2[(grid,)](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        recv_x_scale.stride(0),
        recv_x_scale.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor_scale,
        output_tensor_scale.stride(0),
        output_tensor_scale.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=recv_topk.shape[1],
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
        SCALE_HIDDEN_SIZE=hidden_size // BLOCK_D,
        SCALE_HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size // BLOCK_D),
    )
    return


# SPDX-SnippetEnd


def batched_routed_activation_indexed_to_expert_block_permuted_blockfp8_triton(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    token_to_expert_indices: torch.Tensor,
    *,
    block_size: int,
    n_tokens_padded: int,
    n_tokens_per_expert_padded: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert n_tokens_padded % block_size == 0
    n_blocks = n_tokens_padded // block_size

    expert_start_loc = torch.empty_like(n_tokens_per_expert_padded)
    blocked_activation = torch.empty(
        n_tokens_padded,
        *activation.shape[1:],
        dtype=activation.dtype,
        device=activation.device,
    )
    blocked_activation_scale = torch.empty(
        n_tokens_padded,
        *activation_scale.shape[1:],
        dtype=activation_scale.dtype,
        device=activation_scale.device,
    )
    block_to_expert_indices = torch.empty(
        n_tokens_padded, device=activation.device, dtype=torch.int32
    )
    token_comma_topk_to_block_x_item_indices = token_to_expert_indices.clone()
    ep_scatter(
        activation,
        activation_scale,
        token_to_expert_indices,
        n_tokens_per_expert_padded,
        expert_start_loc,
        blocked_activation,
        blocked_activation_scale,
        block_to_expert_indices,
        token_comma_topk_to_block_x_item_indices,
    )
    return (
        blocked_activation.view(n_blocks, block_size, *activation.shape[1:]),
        blocked_activation_scale.view(
            n_blocks, block_size, *activation_scale.shape[1:]
        ),
        token_comma_topk_to_block_x_item_indices,
        block_to_expert_indices.view(n_blocks, block_size),
    )
