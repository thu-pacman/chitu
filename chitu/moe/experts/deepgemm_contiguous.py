# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import torch
from typing import Optional

from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivationWithPaddedPerExpertCnt,
    IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt,
    ExpertBlockPermutedBatchedRoutedActivationBlockfp8,
)
from chitu.moe.batched_expert_result import ExpertBlockPermutedBatchedExpertResult
from chitu.ops import silu_and_mul
from chitu.ops.quant import blockfp8_act_quant
from chitu.ops.triton_ops.quant_gemm import tma_align_input_scale
from chitu.utils import try_import_opt_dep

deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")


@functools.singledispatch
def deepgemm_contiguous_fused_expert(
    hidden_states: BatchedRoutedActivation,
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
    out: Optional[torch.Tensor] = None,
):
    raise ValueError(f"Unsupported hidden_states type: {type(hidden_states)}")


@deepgemm_contiguous_fused_expert.register
def _(
    hidden_states: IndexedBatchedRoutedActivationWithPaddedPerExpertCnt,
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
    out: Optional[torch.Tensor] = None,
):
    if out is None and inplace:
        out = hidden_states.activation

    quant_hidden_states, hidden_states_scale = blockfp8_act_quant(
        hidden_states.activation
    )
    return deepgemm_contiguous_fused_expert(
        IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt(
            activation=quant_hidden_states,
            activation_scale=hidden_states_scale,
            token_to_expert_indices=hidden_states.token_to_expert_indices,
            n_tokens_per_expert_padded=hidden_states.n_tokens_per_expert_padded,
        ),
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        inplace=inplace,
        activation=activation,
        use_fp8_w8a8=use_fp8_w8a8,
        use_fp4_w4a8=use_fp4_w4a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        global_num_experts=global_num_experts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_scale_2=w1_scale_2,
        w2_scale_2=w2_scale_2,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        soft_fp8=soft_fp8,
        out=out,
    )


@deepgemm_contiguous_fused_expert.register
def _(
    hidden_states: IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt,
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
    out: Optional[torch.Tensor] = None,
):
    return deepgemm_contiguous_fused_expert(
        ExpertBlockPermutedBatchedRoutedActivationBlockfp8.convert_from(
            hidden_states, block_size=128
        ),
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        inplace=inplace,
        activation=activation,
        use_fp8_w8a8=use_fp8_w8a8,
        use_fp4_w4a8=use_fp4_w4a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        global_num_experts=global_num_experts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_scale_2=w1_scale_2,
        w2_scale_2=w2_scale_2,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        soft_fp8=soft_fp8,
        out=out,
    )


@deepgemm_contiguous_fused_expert.register
def _(
    hidden_states: ExpertBlockPermutedBatchedRoutedActivationBlockfp8,
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
    out: Optional[torch.Tensor] = None,
):
    assert use_fp8_w8a8
    assert block_shape is not None
    assert not use_int8_w8a16
    assert not use_int4_w4a16
    assert not soft_fp8
    assert activation == "silu"
    assert w1_zp is None
    assert w2_zp is None

    E, N, K = w1.shape
    n_tokens_padded = (
        hidden_states.blocked_activation.shape[0]
        * hidden_states.blocked_activation.shape[1]
    )

    if out is None:
        out = torch.empty(
            topk_weights.shape[0],
            K,
            device=hidden_states.activation.device,
            dtype=torch.bfloat16,
        )

    intermediate_cache1 = torch.empty(
        (n_tokens_padded, N),
        device=hidden_states.blocked_activation.device,
        dtype=torch.bfloat16,
    )
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (
            hidden_states.blocked_activation.view(
                n_tokens_padded, hidden_states.blocked_activation.shape[-1]
            ),
            tma_align_input_scale(
                hidden_states.blocked_activation_scale.view(
                    n_tokens_padded, hidden_states.blocked_activation_scale.shape[-1]
                )
            ),
        ),
        (w1, w1_scale),
        intermediate_cache1,
        hidden_states.block_to_expert_indices.flatten(),
    )

    intermediate_cache2 = silu_and_mul(intermediate_cache1.view(-1, N), impl="triton")

    qintermediate_cache2, a2q_scale = blockfp8_act_quant(
        x=intermediate_cache2,
    )
    intermediate_cache3 = torch.empty(
        (n_tokens_padded, K),
        device=hidden_states.blocked_activation.device,
        dtype=torch.bfloat16,
    )
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (qintermediate_cache2, a2q_scale),
        (w2, w2_scale),
        intermediate_cache3,
        hidden_states.block_to_expert_indices.flatten(),
    )

    return ExpertBlockPermutedBatchedExpertResult(
        intermediate_cache3, hidden_states.token_comma_topk_to_block_x_item_indices
    ).weighted_sum(topk_weights, out=out)
