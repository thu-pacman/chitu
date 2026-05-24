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
from chitu.moe.batched_expert_result import (
    BatchedExpertResult,
    ExpertBlockPermutedBatchedExpertResult,
)
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
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    round_scale_to_pow2: bool = False,
    swiglu_limit: Optional[float] = None,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
) -> BatchedExpertResult:
    raise ValueError(f"Unsupported hidden_states type: {type(hidden_states)}")


@deepgemm_contiguous_fused_expert.register
def _(
    hidden_states: IndexedBatchedRoutedActivationWithPaddedPerExpertCnt,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    round_scale_to_pow2: bool = False,
    swiglu_limit: Optional[float] = None,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
) -> BatchedExpertResult:
    assert not soft_fp8
    if w1.dtype == torch.float8_e4m3fn:
        assert len(block_shape) == 2
        assert block_shape[0] == block_shape[1]
        quant_block_size = block_shape[0]
        hidden_states_fp8, scale = blockfp8_act_quant(
            hidden_states.activation,
            block_size=quant_block_size,
            round_scale_to_pow2=round_scale_to_pow2,
        )
        return deepgemm_contiguous_fused_expert(
            IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt(
                activation=hidden_states_fp8,
                activation_scale=scale,
                token_to_expert_indices=hidden_states.token_to_expert_indices,
                n_tokens_per_expert_padded=hidden_states.n_tokens_per_expert_padded,
                pad_block_size=hidden_states.pad_block_size,
                n_tokens_padded=hidden_states.n_tokens_padded,
                expert_ids_are_local=hidden_states.expert_ids_are_local,
                expected_n_tokens_per_expert=hidden_states.expected_n_tokens_per_expert,
            ),
            w1=w1,
            w2=w2,
            activation=activation,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            block_shape=block_shape,
            round_scale_to_pow2=round_scale_to_pow2,
            swiglu_limit=swiglu_limit,
            experts_start_idx=experts_start_idx,
        )

    hidden_states = hidden_states.as_local_expert_ids(
        experts_start_idx, experts_start_idx + w1.shape[0]
    )
    new_hidden_states = ExpertBlockPermutedBatchedRoutedActivationNormal.convert_from(
        hidden_states, block_size=128, num_experts=w1.shape[0]
    )
    del hidden_states
    return deepgemm_contiguous_fused_expert(
        new_hidden_states,
        w1=w1,
        w2=w2,
        activation=activation,
        swiglu_limit=swiglu_limit,
        experts_start_idx=experts_start_idx,
    )


@deepgemm_contiguous_fused_expert.register
def _(
    hidden_states: ExpertBlockPermutedBatchedRoutedActivationNormal,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    round_scale_to_pow2: bool = False,
    swiglu_limit: Optional[float] = None,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
) -> ExpertBlockPermutedBatchedExpertResult:
    assert not soft_fp8
    assert w1.dtype in {torch.bfloat16, torch.float16}
    assert w2.dtype in {torch.bfloat16, torch.float16}
    assert w1_scale is None
    assert w2_scale is None

    hidden_states = hidden_states.as_local_expert_ids(
        experts_start_idx, experts_start_idx + w1.shape[0]
    )

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

    intermediate_cache2 = silu_and_mul(
        x=intermediate_cache1.view(-1, N),
        swiglu_limit=swiglu_limit,
        impl="triton",
    )
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
    return ExpertBlockPermutedBatchedExpertResult(
        intermediate_cache3, token_comma_topk_to_block_x_item_indices
    )


@deepgemm_contiguous_fused_expert.register
def _(
    hidden_states: IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    round_scale_to_pow2: bool = False,
    swiglu_limit: Optional[float] = None,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
) -> BatchedExpertResult:
    assert not soft_fp8
    hidden_states = hidden_states.as_local_expert_ids(
        experts_start_idx, experts_start_idx + w1.shape[0]
    )
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
        activation=activation,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
        round_scale_to_pow2=round_scale_to_pow2,
        swiglu_limit=swiglu_limit,
        experts_start_idx=experts_start_idx,
    )


@deepgemm_contiguous_fused_expert.register
def _(
    hidden_states: ExpertBlockPermutedBatchedRoutedActivationBlockfp8,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    round_scale_to_pow2: bool = False,
    swiglu_limit: Optional[float] = None,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
) -> ExpertBlockPermutedBatchedExpertResult:
    if tuple(block_shape) != (128, 128):
        raise NotImplementedError(
            f"deep_gemm only supports 128x128 quantization block, but got {block_shape}"
        )
    if torch.cuda.get_device_capability()[0] == 10 and not round_scale_to_pow2:
        raise NotImplementedError(
            "deep_gemm does not support round_scale_to_pow2==False on sm_10x"
        )
    if torch.get_default_dtype() != torch.bfloat16:
        raise NotImplementedError(
            f"deep_gemm only supports bfloat16 activation output, but got {torch.get_default_dtype()}"
        )

    hidden_states = hidden_states.as_local_expert_ids(
        experts_start_idx, experts_start_idx + w1.shape[0]
    )

    assert not soft_fp8
    assert block_shape is not None
    assert activation == "silu"

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

    intermediate_cache2 = silu_and_mul(
        intermediate_cache1.view(-1, N),
        swiglu_limit=swiglu_limit,
        impl="triton",
    )
    del intermediate_cache1

    qintermediate_cache2, a2q_scale = blockfp8_act_quant(
        intermediate_cache2,
        block_size=block_shape[0],
        round_scale_to_pow2=round_scale_to_pow2,
    )
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

    return ExpertBlockPermutedBatchedExpertResult(
        intermediate_cache3, token_comma_topk_to_block_x_item_indices
    )


@deepgemm_contiguous_fused_expert.register
def _(
    hidden_states: IndexedBatchedRoutedActivation,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    round_scale_to_pow2: bool = False,
    swiglu_limit: Optional[float] = None,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
) -> BatchedExpertResult:
    assert not soft_fp8
    assert len(block_shape) == 2
    assert block_shape[0] == block_shape[1]
    quant_block_size = block_shape[0]
    pad_block_size = 128
    n_experts = w1.shape[0]
    hidden_states = hidden_states.as_local_expert_ids(
        experts_start_idx, experts_start_idx + n_experts
    )

    if w1.dtype == torch.float8_e4m3fn:
        if isinstance(hidden_states, IndexedBatchedRoutedActivationBlockfp8):
            hidden_states_fp8 = hidden_states.activation
            scale = hidden_states.activation_scale
        else:
            hidden_states_fp8, scale = blockfp8_act_quant(
                hidden_states.activation,
                block_size=quant_block_size,
                round_scale_to_pow2=round_scale_to_pow2,
            )
        hidden_states = IndexedBatchedRoutedActivationBlockfp8(
            activation=hidden_states_fp8,
            activation_scale=scale,
            token_to_expert_indices=hidden_states.token_to_expert_indices,
            expected_n_tokens_per_expert=hidden_states.expected_n_tokens_per_expert,
            expert_ids_are_local=hidden_states.expert_ids_are_local,
        )
        hidden_states = (
            IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt.convert_from(
                hidden_states, pad_block_size=pad_block_size, n_experts=n_experts
            )
        )
        return deepgemm_contiguous_fused_expert(
            hidden_states,
            w1=w1,
            w2=w2,
            activation=activation,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            block_shape=block_shape,
            swiglu_limit=swiglu_limit,
            experts_start_idx=experts_start_idx,
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
            activation=activation,
            swiglu_limit=swiglu_limit,
            experts_start_idx=experts_start_idx,
        )
