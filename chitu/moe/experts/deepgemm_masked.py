# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Optional

import torch

from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    IndexedBatchedRoutedActivationWithScale,
    PerExpertDenseBatchedRoutedActivation,
    PerExpertDenseBatchedRoutedActivationMinimal,
    PerExpertDenseBatchedRoutedActivationWithScale,
    PerExpertDenseBatchedRoutedActivationWithScaleMinimal,
)
from chitu.moe.batched_expert_result import (
    BatchedExpertResult,
    PerExpertDenseBatchedExpertResultMinimal,
    PerExpertDenseBatchedExpertResult,
)
from chitu.ops.quant import blockfp8_act_quant, silu_and_mul_and_blockfp8_act_quant
from chitu.utils import try_import_opt_dep
from chitu.ops import silu_and_mul
from chitu.lazy import eval_lazy

deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")


@functools.singledispatch
def deepgemm_masked_fused_expert(
    hidden_states: BatchedRoutedActivation,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation: str = "silu",
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    round_scale_to_pow2: bool = False,
    swiglu_limit: Optional[float] = None,
    experts_start_idx: int = 0,
) -> BatchedExpertResult:
    raise NotImplementedError(
        f"deepgemm_masked_fused_expert not implemented for type {type(hidden_states)}"
    )


@deepgemm_masked_fused_expert.register
def _(
    hidden_states: IndexedBatchedRoutedActivation,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation: str = "silu",
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    round_scale_to_pow2: bool = False,
    swiglu_limit: Optional[float] = None,
    experts_start_idx: int = 0,
) -> PerExpertDenseBatchedExpertResult:
    hidden_states = hidden_states.as_local_expert_ids(
        experts_start_idx, experts_start_idx + w1.shape[0]
    )

    if w1.dtype == torch.float8_e4m3fn:
        assert not isinstance(hidden_states, IndexedBatchedRoutedActivationWithScale)
        assert len(block_shape) == 2
        assert block_shape[0] == block_shape[1]
        activation_fp8, activation_scale = blockfp8_act_quant(
            hidden_states.activation,
            scale_block_shape=block_shape,
            round_scale_to_pow2=round_scale_to_pow2,
        )
        return deepgemm_masked_fused_expert(
            IndexedBatchedRoutedActivationWithScale(
                activation=activation_fp8,
                activation_scale=activation_scale,
                token_to_expert_indices=hidden_states.token_to_expert_indices,
                quant_method="blockfp8",
                expected_n_tokens_per_expert=hidden_states.expected_n_tokens_per_expert,
                expert_ids_are_local=True,
            ),
            w1=w1,
            w2=w2,
            activation=activation,
            global_num_experts=global_num_experts,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            block_shape=block_shape,
            round_scale_to_pow2=round_scale_to_pow2,
            swiglu_limit=swiglu_limit,
            experts_start_idx=experts_start_idx,
        )

    return deepgemm_masked_fused_expert(
        PerExpertDenseBatchedRoutedActivation.convert_from(
            hidden_states, num_experts=w1.shape[0]
        ),
        w1=w1,
        w2=w2,
        activation=activation,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
        swiglu_limit=swiglu_limit,
        experts_start_idx=experts_start_idx,
    )


@deepgemm_masked_fused_expert.register
def _(
    hidden_states: IndexedBatchedRoutedActivationWithScale,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation: str = "silu",
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    round_scale_to_pow2: bool = False,
    swiglu_limit: Optional[float] = None,
    experts_start_idx: int = 0,
) -> PerExpertDenseBatchedExpertResult:
    hidden_states = hidden_states.as_local_expert_ids(
        experts_start_idx, experts_start_idx + w1.shape[0]
    )

    return deepgemm_masked_fused_expert(
        PerExpertDenseBatchedRoutedActivationWithScale.convert_from(
            hidden_states, num_experts=w1.shape[0]
        ),
        w1=w1,
        w2=w2,
        activation=activation,
        global_num_experts=global_num_experts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        round_scale_to_pow2=round_scale_to_pow2,
        swiglu_limit=swiglu_limit,
        experts_start_idx=experts_start_idx,
    )


@deepgemm_masked_fused_expert.register
def _(
    hidden_states: PerExpertDenseBatchedRoutedActivationMinimal,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    round_scale_to_pow2: bool = False,
    swiglu_limit: Optional[float] = None,
    experts_start_idx: int = 0,
) -> PerExpertDenseBatchedExpertResultMinimal:
    # dtype check
    assert activation == "silu"

    hidden_states = hidden_states.as_local_expert_ids(
        experts_start_idx, experts_start_idx + w1.shape[0]
    )

    device = hidden_states.activation_per_expert.device
    E, M, K = hidden_states.activation_per_expert.shape
    E2, N, K2 = w1.shape
    assert E == E2
    assert K == K2

    if M < 64:
        # M<64 triggers a DeepGEMM bug: https://github.com/deepseek-ai/DeepGEMM/issues/268
        hidden_states.activation_per_expert = torch.nn.functional.pad(
            hidden_states.activation_per_expert, (0, 0, 0, 64 - M), "constant", 0
        )

    if M == 0:
        intermediate_cache3 = torch.empty(
            (E, M, K), device=device, dtype=torch.bfloat16
        )
    elif w1.dtype != torch.float8_e4m3fn:
        assert has_deep_gemm, "BF16 masked path requires deep_gemm backend"

        # For bf16, DeepGEMM requires reduction dimensions % 64 == 0. Search `DG_HOST_ASSERT(k % 64 == 0)`
        # in `third_party/DeepGEMM/csrc/jit_kernels/impls/sm90_bf16_gemm.hpp` for details.
        assert K % 64 == 0
        assert N % 64 == 0

        intermediate_cache1 = torch.empty(
            (E, M, N), device=device, dtype=torch.bfloat16
        )
        deep_gemm.m_grouped_bf16_gemm_nt_masked(
            hidden_states.activation_per_expert,
            w1,
            intermediate_cache1,
            hidden_states.n_tokens_per_expert,
            expected_m=hidden_states.expected_n_tokens_per_expert,
        )

        intermediate_cache2 = eval_lazy(
            silu_and_mul(
                intermediate_cache1,
                expert_n_tokens=hidden_states.n_tokens_per_expert,
                swiglu_limit=swiglu_limit,
            )
        )
        del intermediate_cache1

        intermediate_cache3 = torch.empty(
            (E, M, K), device=device, dtype=torch.bfloat16
        )
        deep_gemm.m_grouped_bf16_gemm_nt_masked(
            intermediate_cache2,
            w2,
            intermediate_cache3,
            hidden_states.n_tokens_per_expert,
            expected_m=hidden_states.expected_n_tokens_per_expert,
        )
        del intermediate_cache2
    else:

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

        if isinstance(
            hidden_states, PerExpertDenseBatchedRoutedActivationWithScaleMinimal
        ):
            hidden_states_fp8 = hidden_states.activation_per_expert
            a1_scale = hidden_states.activation_scale_per_expert
        else:
            hidden_states_fp8, a1_scale = blockfp8_act_quant(
                hidden_states.activation_per_expert,
                scale_block_shape=block_shape,
                round_scale_to_pow2=round_scale_to_pow2,
            )

        intermediate_cache1 = torch.empty(
            (E, M, N), device=device, dtype=torch.bfloat16
        )
        deep_gemm.m_grouped_fp8_gemm_nt_masked(
            (hidden_states_fp8, a1_scale),
            (w1, w1_scale),
            intermediate_cache1,
            hidden_states.n_tokens_per_expert,
            expected_m=hidden_states.expected_n_tokens_per_expert,
        )
        del hidden_states_fp8
        del a1_scale

        qintermediate_cache2, a2q_scale = silu_and_mul_and_blockfp8_act_quant(
            intermediate_cache1,
            expert_n_tokens=hidden_states.n_tokens_per_expert,
            swiglu_limit=swiglu_limit,
            scale_block_shape=block_shape,
            round_scale_to_pow2=round_scale_to_pow2,
        )
        del intermediate_cache1

        intermediate_cache3 = torch.empty(
            (E, M, K), device=device, dtype=torch.bfloat16
        )
        deep_gemm.m_grouped_fp8_gemm_nt_masked(
            (qintermediate_cache2, a2q_scale),
            (w2, w2_scale),
            intermediate_cache3,
            hidden_states.n_tokens_per_expert,
            expected_m=hidden_states.expected_n_tokens_per_expert,
        )

    if isinstance(
        hidden_states,
        (
            PerExpertDenseBatchedRoutedActivation,
            PerExpertDenseBatchedRoutedActivationWithScale,
        ),
    ):
        # PerExpertDenseBatchedRoutedActivation and PerExpertDenseBatchedRoutedActivationWithScale
        # are subclasses of PerExpertDenseBatchedRoutedActivationMinimal
        return PerExpertDenseBatchedExpertResult(
            intermediate_cache3,
            token_to_expert_indices=hidden_states.token_to_expert_indices,
            token_pos_in_expert=hidden_states.token_pos_in_expert,
        )
    else:
        return PerExpertDenseBatchedExpertResultMinimal(intermediate_cache3)
