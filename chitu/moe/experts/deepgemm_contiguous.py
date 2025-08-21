# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import List, Optional

from deep_gemm import m_grouped_gemm_fp8_fp8_bf16_nt_contiguous

from chitu.ops import silu_and_mul
from chitu.ops.quant import blockfp8_act_quant
from chitu.ops.triton_ops.permutation import ep_gather, ep_scatter
from chitu.ops.triton_ops.quant_gemm import tma_align_input_scale


def deepgemm_contiguous_fused_expert(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    use_fp8_w8a8: bool = False,
    use_fp4_w4a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_scale_2: Optional[torch.Tensor] = None,
    w2_scale_2: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    soft_fp8: bool = False,
    tokens_per_expert: Optional[torch.Tensor] = None,
    experts_start_idx: int = 0,
):
    assert use_fp8_w8a8
    assert block_shape is not None
    assert not use_int8_w8a16
    assert not use_int4_w4a16
    assert not soft_fp8
    assert activation == "silu"
    assert w1_zp is None
    assert w2_zp is None

    M = hidden_states.shape[0]
    E, N, K = w1.shape
    all_tokens = sum(tokens_per_expert)

    intermediate_cache1 = torch.empty(
        (all_tokens, N), device=hidden_states.device, dtype=hidden_states.dtype
    )

    if inplace:
        gather_out = hidden_states
    else:
        gather_out = torch.empty_like(hidden_states)

    is_fp8_input = isinstance(hidden_states, tuple)
    if not is_fp8_input:
        hidden_states, a1_scale = blockfp8_act_quant(
            x=hidden_states,
        )
    else:
        hidden_states, a1_scale = hidden_states

    input_tensor = [
        torch.empty(
            (all_tokens, K),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ),
        torch.empty(
            (all_tokens, K // 128),
            device=a1_scale.device,
            dtype=a1_scale.dtype,
        ),
    ]
    m_indices = torch.empty(all_tokens, device=hidden_states.device, dtype=torch.int32)
    output_index = topk_ids.clone()

    num_recv_tokens_per_expert_gpu = torch.tensor(
        tokens_per_expert,
        dtype=torch.int32,
        pin_memory=True,
        device="cpu",
    ).cuda(non_blocking=True)
    expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)
    ep_scatter(
        hidden_states,
        a1_scale,
        topk_ids,
        num_recv_tokens_per_expert_gpu,
        expert_start_loc,
        input_tensor[0],
        input_tensor[1],
        m_indices,
        output_index,
    )
    input_tensor[1] = tma_align_input_scale(input_tensor[1])
    m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
        (input_tensor[0], input_tensor[1]),
        (w1, w1_scale),
        intermediate_cache1,
        m_indices,
    )

    intermediate_cache2 = silu_and_mul(intermediate_cache1.view(-1, N), impl="triton")
    # silu_and_mul(intermediate_cache1.view(-1, N), y=intermediate_cache2)

    qintermediate_cache2, a2q_scale = blockfp8_act_quant(
        x=intermediate_cache2,
    )
    intermediate_cache3 = torch.empty(
        (all_tokens, K),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
        (qintermediate_cache2, a2q_scale),
        (w2, w2_scale),
        intermediate_cache3,
        m_indices,
    )
    ep_gather(
        intermediate_cache3,
        topk_ids,
        topk_weights,
        output_index,
        gather_out,
    )
    return gather_out
