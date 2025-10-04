# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch
import triton
import triton.language as tl


def moe_sum_per_token_triton(
    x: torch.Tensor, topk_weights: torch.Tensor, *, out: Optional[torch.Tensor] = None
):
    # NOTE: Although this function accept inplace `out` parameter, but `out` cannot
    # have the same address as `x` or `topk_weights`, due to Triton limitations.

    M, topk, N = x.shape

    # SPDX-SnippetBegin
    # SPDX-License-Identifier: Apache-2.0
    # SPDX-SnippetCopyrightText: 2025 unslothai
    # SDPX—SnippetName: calculate_settings from unsloth
    def calculate_settings(n):
        # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

        MAX_FUSED_SIZE = 65536
        BLOCK_SIZE = triton.next_power_of_2(n)
        if BLOCK_SIZE > MAX_FUSED_SIZE:
            raise RuntimeError(
                f"Cannot launch Triton kernel since n = {n} exceeds "
                f"the recommended Triton blocksize = {MAX_FUSED_SIZE}."
            )

        num_warps = 4
        if BLOCK_SIZE >= 32768:
            num_warps = 32
        elif BLOCK_SIZE >= 8192:
            num_warps = 16
        elif BLOCK_SIZE >= 1024:
            num_warps = 8
        return BLOCK_SIZE, num_warps

    # SPDX-SnippetEnd

    BLOCK_SIZE_N, num_warps = calculate_settings(N)
    # Determine grid and block sizes

    if out is None:
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    assert x.is_contiguous()
    assert topk_weights.is_contiguous()
    assert out.is_contiguous()

    moe_sum_per_token_triton_kernel[M,](
        x,
        topk_weights,
        out,
        M,
        topk,
        N,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=num_warps,
    )

    return out


@triton.jit
def moe_sum_per_token_triton_kernel(
    # Pointers to matrices
    x_ptr,  # (M, topk, N)
    topk_weights_ptr,  # (M, topk)
    output_ptr,  # (M, N)
    # Matrix dimensions
    M,
    topk,
    N,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID
    row_index = tl.program_id(axis=0)
    # Create offsets for m and n dimensions
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    # Create a mask to handle the case where the block extends beyond the matrix
    n_mask = offs_n < N

    # Initialize the output sum to zero
    output_sum = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    # Loop over the topk dimension
    for k in range(topk):
        # Load the input for the current slice
        x_offset = row_index * (topk * N) + k * N + offs_n
        x = tl.load(x_ptr + x_offset, n_mask, other=0.0)

        # Load the weight for the current slice
        topk_weight_offset = row_index * topk + k
        topk_weight = tl.load(topk_weights_ptr + topk_weight_offset)

        # Add to the running sum
        output_sum += x * topk_weight

    # Store the final sum to the output tensor
    output_offset = row_index * N + offs_n
    tl.store(output_ptr + output_offset, output_sum, mask=n_mask)


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 ModelTC
# SDPX—SnippetName: lightllm deepep_scatter_gather
#
# Parts of this file are adapted from lightllm (https://github.com/ModelTC/lightllm/blob/main/lightllm/common/fused_moe/deepep_scatter_gather.py)
# Modified and adjusted to fit the requirements of this project


@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    token_comma_topk_to_row_indices,
    token_comma_topk_to_row_indices_stride0,
    token_comma_topk_to_row_indices_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_block = tl.program_id(0)
    start_cur_token = tl.program_id(1)
    grid_num = tl.num_programs(1)

    for cur_token in range(start_cur_token, total_token_num, grid_num):
        off_d = tl.arange(0, BLOCK_D)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for topk_index in range(0, topk_num):
            source_token_index = tl.load(
                token_comma_topk_to_row_indices
                + cur_token * token_comma_topk_to_row_indices_stride0
                + topk_index
            )
            if source_token_index >= 0:
                acc_weight = tl.load(
                    recv_topk_weight + cur_token * recv_topk_weight_stride0 + topk_index
                )
                tmp = tl.load(
                    input_tensor
                    + source_token_index * input_tensor_stride0
                    + cur_block * BLOCK_D
                    + off_d
                )
                accumulator += tmp.to(tl.float32) * acc_weight

        tl.store(
            output_tensor
            + cur_token * output_tensor_stride0
            + cur_block * BLOCK_D
            + off_d,
            accumulator.to(output_tensor.dtype.element_ty),
        )


@torch.no_grad()
def moe_sum_expert_block_permuted_triton(
    x: torch.Tensor,
    token_comma_topk_to_row_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
):
    x = x.view(-1, x.shape[-1])

    num_tokens, topk = topk_weights.shape
    _, hidden_size = x.shape

    if out is None:
        out = torch.empty(num_tokens, hidden_size, dtype=x.dtype, device=x.device)

    BLOCK_D = 1024  # No longer needed (FIXME)
    num_warps = 2
    assert hidden_size % BLOCK_D == 0
    grid = (triton.cdiv(hidden_size, BLOCK_D), min(num_tokens, 1024))
    _fwd_kernel_ep_gather[grid](
        num_tokens,
        x,
        x.stride(0),
        x.stride(1),
        topk_weights,
        topk_weights.stride(0),
        topk_weights.stride(1),
        token_comma_topk_to_row_indices,
        token_comma_topk_to_row_indices.stride(0),
        token_comma_topk_to_row_indices.stride(1),
        out,
        out.stride(0),
        out.stride(1),
        topk_num=topk,
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
    )

    return out


# SPDX-SnippetEnd
