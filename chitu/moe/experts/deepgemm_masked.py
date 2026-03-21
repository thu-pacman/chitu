# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Optional

import torch

from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    IndexedBatchedRoutedActivationBlockfp8,
    PerExpertDenseBatchedRoutedActivation,
    PerExpertDenseBatchedRoutedActivationMinimal,
    PerExpertDenseBatchedRoutedActivationBlockfp8,
    PerExpertDenseBatchedRoutedActivationBlockfp8Minimal,
)
from chitu.moe.batched_expert_result import (
    BatchedExpertResult,
    PerExpertDenseBatchedExpertResultMinimal,
    PerExpertDenseBatchedExpertResult,
)
from chitu.ops.quant import blockfp8_act_quant, silu_and_mul_and_blockfp8_act_quant
from chitu.utils import try_import_opt_dep
from chitu.ops import silu_and_mul

deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")


@functools.singledispatch
def deepgemm_masked_fused_expert(
    hidden_states: BatchedRoutedActivation,
    w1: torch.Tensor,
    w2: torch.Tensor,
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
    round_scale_to_pow2: bool = False,
    soft_fp8: bool = False,
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
    round_scale_to_pow2: bool = False,
    soft_fp8: bool = False,
    experts_start_idx: int = 0,
) -> PerExpertDenseBatchedExpertResult:
    # first compute the n_tokens_per_expert (Tensor) based on token_to_expert_indices
    if global_num_experts > 0:
        n_experts = global_num_experts
    else:
        from chitu.global_vars import get_global_args

        n_experts = get_global_args().infer.num_experts_slots

    hidden_states = hidden_states.as_local_expert_ids(
        experts_start_idx, experts_start_idx + w1.shape[0]
    )

    if use_fp8_w8a8:
        assert not isinstance(hidden_states, IndexedBatchedRoutedActivationBlockfp8)
        assert len(block_shape) == 2
        assert block_shape[0] == block_shape[1]
        activation_fp8, activation_scale = blockfp8_act_quant(
            hidden_states.activation,
            block_size=block_shape[0],
            round_scale_to_pow2=round_scale_to_pow2,
        )
        return deepgemm_masked_fused_expert(
            IndexedBatchedRoutedActivationBlockfp8(
                activation=activation_fp8,
                activation_scale=activation_scale,
                token_to_expert_indices=hidden_states.token_to_expert_indices,
            ),
            w1=w1,
            w2=w2,
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
            round_scale_to_pow2=round_scale_to_pow2,
            soft_fp8=soft_fp8,
            experts_start_idx=experts_start_idx,
        )

    return deepgemm_masked_fused_expert(
        PerExpertDenseBatchedRoutedActivation.convert_from(hidden_states),
        w1=w1,
        w2=w2,
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
        round_scale_to_pow2=round_scale_to_pow2,
        soft_fp8=soft_fp8,
        experts_start_idx=experts_start_idx,
    )


@deepgemm_masked_fused_expert.register
def _(
    hidden_states: IndexedBatchedRoutedActivationBlockfp8,
    w1: torch.Tensor,
    w2: torch.Tensor,
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
    round_scale_to_pow2: bool = False,
    soft_fp8: bool = False,
    experts_start_idx: int = 0,
) -> PerExpertDenseBatchedExpertResult:
    # first compute the n_tokens_per_expert (Tensor) based on token_to_expert_indices
    if global_num_experts > 0:
        n_experts = global_num_experts
    else:
        from chitu.global_vars import get_global_args

        n_experts = get_global_args().infer.num_experts_slots

    hidden_states = hidden_states.as_local_expert_ids(
        experts_start_idx, experts_start_idx + w1.shape[0]
    )

    return deepgemm_masked_fused_expert(
        PerExpertDenseBatchedRoutedActivationBlockfp8.convert_from(hidden_states),
        w1=w1,
        w2=w2,
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
        round_scale_to_pow2=round_scale_to_pow2,
        soft_fp8=soft_fp8,
        experts_start_idx=experts_start_idx,
    )


@deepgemm_masked_fused_expert.register
def _(
    hidden_states: PerExpertDenseBatchedRoutedActivationMinimal,
    w1: torch.Tensor,
    w2: torch.Tensor,
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
    round_scale_to_pow2: bool = False,
    soft_fp8: bool = False,
    experts_start_idx: int = 0,
) -> PerExpertDenseBatchedExpertResultMinimal:
    # dtype check
    assert activation == "silu"
    assert not soft_fp8
    assert w1_zp is None
    assert w2_zp is None
    assert not use_int8_w8a16
    assert not use_int4_w4a16

    hidden_states = hidden_states.as_local_expert_ids(
        experts_start_idx, experts_start_idx + w1.shape[0]
    )

    if not use_fp8_w8a8:
        assert isinstance(
            hidden_states, PerExpertDenseBatchedRoutedActivationBlockfp8Minimal
        )
        assert has_deep_gemm, "BF16 masked path requires deep_gemm backend"

        hidden_states_bf16 = hidden_states.activation_per_expert
        M = hidden_states_bf16.shape[1]
        E, N, _ = w1.shape

        intermediate_cache1 = torch.empty(
            (E, M, N), device=hidden_states_bf16.device, dtype=torch.bfloat16
        )
        intermediate_cache3 = torch.empty(
            hidden_states_bf16.shape,
            device=hidden_states_bf16.device,
            dtype=torch.bfloat16,
        )

        deep_gemm.m_grouped_bf16_gemm_nt_masked(
            hidden_states_bf16,
            w1,
            intermediate_cache1,
            hidden_states.n_tokens_per_expert,
            M,
        )

        intermediate_cache2 = silu_and_mul(
            intermediate_cache1.view(-1, N), impl="triton"
        ).view_as(intermediate_cache1)

        deep_gemm.m_grouped_bf16_gemm_nt_masked(
            intermediate_cache2,
            w2,
            intermediate_cache3,
            hidden_states.n_tokens_per_expert,
            M,
        )

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

        assert use_fp8_w8a8
        if isinstance(
            hidden_states, PerExpertDenseBatchedRoutedActivationBlockfp8Minimal
        ):
            hidden_states_fp8 = hidden_states.activation_per_expert
            a1_scale = hidden_states.activation_scale_per_expert
        else:
            hidden_states_fp8, a1_scale = blockfp8_act_quant(
                hidden_states.activation_per_expert,
                block_size=block_shape[0],
                round_scale_to_pow2=round_scale_to_pow2,
            )

        device = hidden_states_fp8.device
        E, M, K = hidden_states_fp8.shape
        E2, N, K2 = w1.shape
        assert E == E2
        assert K == K2

        intermediate_cache1 = torch.empty(
            (E, M, N), device=device, dtype=torch.bfloat16
        )
        deep_gemm.m_grouped_fp8_gemm_nt_masked(
            (hidden_states_fp8, a1_scale),
            (w1, w1_scale),
            intermediate_cache1,
            hidden_states.n_tokens_per_expert,
            M,
        )
        del hidden_states_fp8
        del a1_scale

        qintermediate_cache2, a2q_scale = silu_and_mul_and_blockfp8_act_quant(
            intermediate_cache1,
            expert_n_tokens=hidden_states.n_tokens_per_expert,
            block_size=block_shape[0],
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
            M,
        )

    if isinstance(hidden_states, PerExpertDenseBatchedRoutedActivation):
        # PerExpertDenseBatchedRoutedActivation is a subclass of PerExpertDenseBatchedRoutedActivationMinimal
        return PerExpertDenseBatchedExpertResult(
            intermediate_cache3,
            token_to_expert_indices=hidden_states.token_to_expert_indices,
            token_pos_in_expert=hidden_states.token_pos_in_expert,
        )
    else:
        return PerExpertDenseBatchedExpertResultMinimal(intermediate_cache3)
