# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import torch
from typing import Optional

from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    ExpertBlockPermutedBatchedRoutedActivationNormal,
    IndexedBatchedRoutedActivation,
    IndexedBatchedRoutedActivationBlockfp8,
    IndexedBatchedRoutedActivationWithPaddedPerExpertCnt,
    IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt,
    ExpertBlockPermutedBatchedRoutedActivationBlockfp8,
)
from chitu.moe.batched_expert_result import ExpertBlockPermutedBatchedExpertResult
from chitu.ops import silu_and_mul
from chitu.ops.quant import blockfp8_act_quant
from chitu.ops.triton_ops.quant_gemm import tma_align_input_scale
from chitu.utils import try_import_opt_dep
from chitu.lazy import eval_lazy

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

    new_hidden_states = ExpertBlockPermutedBatchedRoutedActivationNormal.convert_from(
        hidden_states, block_size=128, num_experts=w1.shape[0]
    )
    del hidden_states
    return deepgemm_contiguous_fused_expert(
        new_hidden_states,
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
    hidden_states: ExpertBlockPermutedBatchedRoutedActivationNormal,
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

    blocked_activation = hidden_states.blocked_activation
    block_to_expert_indices = hidden_states.block_to_expert_indices
    token_comma_topk_to_block_x_item_indices = (
        hidden_states.token_comma_topk_to_block_x_item_indices
    )
    del hidden_states

    device = blocked_activation.device

    E, N, K = w1.shape
    n_tokens_padded = blocked_activation.shape[0] * blocked_activation.shape[1]

    intermediate_cache1 = torch.empty(
        (n_tokens_padded, N), device=device, dtype=torch.bfloat16
    )

    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
        blocked_activation.view(n_tokens_padded, blocked_activation.shape[-1]),
        w1,
        intermediate_cache1,
        block_to_expert_indices.flatten(),
    )
    del blocked_activation

    intermediate_cache2 = silu_and_mul(x=intermediate_cache1.view(-1, N), impl="triton")
    del intermediate_cache1
    intermediate_cache2 = eval_lazy(intermediate_cache2)

    intermediate_cache3 = torch.empty(
        (n_tokens_padded, K), device=device, dtype=torch.bfloat16
    )
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        block_to_expert_indices.flatten(),
    )
    del intermediate_cache2, block_to_expert_indices
    results = ExpertBlockPermutedBatchedExpertResult(
        intermediate_cache3, token_comma_topk_to_block_x_item_indices
    )
    del intermediate_cache3, token_comma_topk_to_block_x_item_indices

    if out is None:
        out = torch.empty(topk_weights.shape[0], K, device=device, dtype=torch.bfloat16)
    return results.weighted_sum(topk_weights, out=out)


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
    temp_hidden_states = (
        ExpertBlockPermutedBatchedRoutedActivationBlockfp8.convert_from(
            hidden_states, block_size=128, num_experts=w1.shape[0]
        )
    )
    del hidden_states
    return deepgemm_contiguous_fused_expert(
        temp_hidden_states,
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

    blocked_activation = hidden_states.blocked_activation
    blocked_activation_scale = hidden_states.blocked_activation_scale
    block_to_expert_indices = hidden_states.block_to_expert_indices
    token_comma_topk_to_block_x_item_indices = (
        hidden_states.token_comma_topk_to_block_x_item_indices
    )
    del hidden_states

    device = blocked_activation.device

    E, N, K = w1.shape
    n_tokens_padded = blocked_activation.shape[0] * blocked_activation.shape[1]

    intermediate_cache1 = torch.empty(
        (n_tokens_padded, N), device=device, dtype=torch.bfloat16
    )
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (
            blocked_activation.view(n_tokens_padded, blocked_activation.shape[-1]),
            tma_align_input_scale(
                blocked_activation_scale.view(
                    n_tokens_padded, blocked_activation_scale.shape[-1]
                )
            ),
        ),
        (w1, w1_scale),
        intermediate_cache1,
        block_to_expert_indices.flatten(),
    )
    del blocked_activation
    del blocked_activation_scale

    intermediate_cache2 = silu_and_mul(intermediate_cache1.view(-1, N), impl="triton")
    del intermediate_cache1

    qintermediate_cache2, a2q_scale = blockfp8_act_quant(x=intermediate_cache2)
    del intermediate_cache2

    intermediate_cache3 = torch.empty(
        (n_tokens_padded, K), device=device, dtype=torch.bfloat16
    )
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (qintermediate_cache2, a2q_scale),
        (w2, w2_scale),
        intermediate_cache3,
        block_to_expert_indices.flatten(),
    )
    del qintermediate_cache2
    del a2q_scale

    expert_result = ExpertBlockPermutedBatchedExpertResult(
        intermediate_cache3, token_comma_topk_to_block_x_item_indices
    )
    del intermediate_cache3

    if out is None:
        out = torch.empty(topk_weights.shape[0], K, device=device, dtype=torch.bfloat16)
    return expert_result.weighted_sum(topk_weights, out=out)


@deepgemm_contiguous_fused_expert.register
def _(
    hidden_states: IndexedBatchedRoutedActivation,
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

    assert not use_fp4_w4a8
    assert not use_int8_w8a16
    assert not use_int4_w4a16

    pad_block_size = 128
    quant_block_size = 128
    n_experts = w1.shape[0]

    if use_fp8_w8a8:
        hidden_states_fp8, scale = blockfp8_act_quant(
            hidden_states.activation, block_size=quant_block_size
        )
        hidden_states = IndexedBatchedRoutedActivationBlockfp8(
            activation=hidden_states_fp8,
            activation_scale=scale,
            token_to_expert_indices=hidden_states.token_to_expert_indices,
        )
        hidden_states = (
            IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt.convert_from(
                hidden_states, pad_block_size=pad_block_size, n_experts=n_experts
            )
        )
    else:
        hidden_states = (
            IndexedBatchedRoutedActivationWithPaddedPerExpertCnt.convert_from(
                hidden_states, pad_block_size=pad_block_size, n_experts=n_experts
            )
        )

    return deepgemm_contiguous_fused_expert(
        hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        inplace=inplace,
        activation=activation,
        use_fp8_w8a8=use_fp8_w8a8,
        use_fp4_w4a8=use_fp4_w4a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        global_num_experts=n_experts,
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
        experts_start_idx=experts_start_idx,
        out=out,
    )
