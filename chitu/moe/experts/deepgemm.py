# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import torch
from deep_gemm import m_grouped_gemm_fp8_fp8_bf16_nt_masked

from chitu.ops.quant import act_quant_deepseek_v3
from chitu.ops.triton_ops.activation import silu_and_mul_masked_post_quant_fwd


def deep_gemm_fused_expert(
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
    # dtype check
    assert use_fp8_w8a8
    assert block_shape is not None
    assert not use_int8_w8a16
    assert not use_int4_w4a16
    assert not soft_fp8
    assert activation == "silu"
    assert w1_zp is None
    assert w2_zp is None
    assert not inplace

    is_fp8_input = isinstance(hidden_states, tuple)
    if not is_fp8_input:
        hidden_states_fp8, a1_scale = act_quant_deepseek_v3(
            x=hidden_states,
        )
    else:
        hidden_states_fp8, a1_scale = hidden_states

    M = hidden_states_fp8.shape[1]
    E, N, _ = w1.shape
    if global_num_experts == -1:
        global_num_experts = E

    intermediate_cache1 = torch.empty(
        (E, M, N), device=hidden_states_fp8.device, dtype=torch.bfloat16
    )

    intermediate_cache3 = torch.empty(
        hidden_states_fp8.shape,
        device=hidden_states_fp8.device,
        dtype=torch.bfloat16,
    )

    m_grouped_gemm_fp8_fp8_bf16_nt_masked(
        (hidden_states_fp8, a1_scale),
        (w1, w1_scale),
        intermediate_cache1,
        tokens_per_expert,
        M,
    )

    qintermediate_cache2 = torch.empty(
        (
            intermediate_cache1.shape[0],
            intermediate_cache1.shape[1],
            intermediate_cache1.shape[2] // 2,
        ),
        device=intermediate_cache1.device,
        dtype=torch.float8_e4m3fn,
    )
    scale_block_size = 128
    a2q_scale = torch.empty(
        (
            intermediate_cache1.shape[0],
            intermediate_cache1.shape[1],
            intermediate_cache1.shape[2] // 2 // scale_block_size,
        ),
        device=intermediate_cache1.device,
        dtype=torch.float32,
    )

    silu_and_mul_masked_post_quant_fwd(
        intermediate_cache1,
        qintermediate_cache2,
        a2q_scale,
        scale_block_size,
        tokens_per_expert,
    )

    m_grouped_gemm_fp8_fp8_bf16_nt_masked(
        (qintermediate_cache2, a2q_scale),
        (w2, w2_scale),
        intermediate_cache3,
        tokens_per_expert,
        M,
    )

    return intermediate_cache3
