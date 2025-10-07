# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from chitu.moe.batched_routed_activation import (
    PerExpertDenseBatchedRoutedActivation,
    PerExpertDenseBatchedRoutedActivationBlockfp8,
)
from chitu.ops.quant import blockfp8_act_quant
from chitu.ops.triton_ops.quant.blockfp8.convert import (
    silu_and_mul_and_blockfp8_act_quant_with_expert_mask,
)
from chitu.utils import try_import_opt_dep

deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")


def deepgemm_masked_fused_expert(
    hidden_states: PerExpertDenseBatchedRoutedActivation,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    use_fp8_w8a8: bool = False,
    use_fp4_w4a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_scale_2: Optional[torch.Tensor] = None,
    w2_scale_2: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
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

    if isinstance(hidden_states, PerExpertDenseBatchedRoutedActivationBlockfp8):
        hidden_states_fp8 = hidden_states.activation_per_expert
        a1_scale = hidden_states.activation_scale_per_expert
    else:
        hidden_states_fp8, a1_scale = blockfp8_act_quant(
            hidden_states.activation_per_expert
        )

    M = hidden_states_fp8.shape[1]
    E, N, _ = w1.shape

    intermediate_cache1 = torch.empty(
        (E, M, N), device=hidden_states_fp8.device, dtype=torch.bfloat16
    )

    intermediate_cache3 = torch.empty(
        hidden_states_fp8.shape,
        device=hidden_states_fp8.device,
        dtype=torch.bfloat16,
    )

    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        (hidden_states_fp8, a1_scale),
        (w1, w1_scale),
        intermediate_cache1,
        hidden_states.n_tokens_per_expert,
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

    silu_and_mul_and_blockfp8_act_quant_with_expert_mask(
        intermediate_cache1,
        qintermediate_cache2,
        a2q_scale,
        scale_block_size,
        hidden_states.n_tokens_per_expert,
    )

    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        (qintermediate_cache2, a2q_scale),
        (w2, w2_scale),
        intermediate_cache3,
        hidden_states.n_tokens_per_expert,
        M,
    )

    return intermediate_cache3
