# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Optional

import torch

from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    PerExpertDenseBatchedRoutedActivation,
    PerExpertDenseBatchedRoutedActivationBlockfp8,
)
from chitu.moe.batched_expert_result import PerExpertDenseBatchedExpertResult
from chitu.ops.quant import blockfp8_act_quant, silu_and_mul_and_blockfp8_act_quant
from chitu.utils import try_import_opt_dep
from chitu.ops import silu_and_mul

deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")


@functools.singledispatch
def deepgemm_masked_fused_expert(
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
):
    raise NotImplementedError(
        f"deepgemm_masked_fused_expert not implemented for type {type(hidden_states)}"
    )


@deepgemm_masked_fused_expert.register
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
):
    # first compute the n_tokens_per_expert (Tensor) based on token_to_expert_indices
    if global_num_experts > 0:
        n_experts = global_num_experts
    else:
        from chitu.global_vars import get_global_args

        n_experts = get_global_args().infer.num_experts_slots

    hidden_states = hidden_states.as_local_expert_ids(
        experts_start_idx, experts_start_idx + w1.shape[0]
    )

    token_to_expert = hidden_states.token_to_expert_indices  # [B, topk]

    token_cnt_per_expert = torch.zeros(
        n_experts,
        device=token_to_expert.device,
        dtype=torch.int32,
    )
    flat_expert = token_to_expert.view(-1)
    ones = torch.ones_like(flat_expert, dtype=torch.int32)
    token_cnt_per_expert.index_add_(0, flat_expert, ones)

    B, H = hidden_states.activation.shape
    topk = token_to_expert.shape[1]

    max_n_tokens_per_expert = B * topk
    activation_per_expert = hidden_states.activation.new_zeros(
        (n_experts, max_n_tokens_per_expert, H)
    )

    # # Reference semantics (not used in graph):
    # E = n_experts
    # token_pos_in_expert = torch.empty(
    #     (B, topk), dtype=torch.int32, device=hidden_states.activation.device
    # )
    # write_pos = torch.zeros(
    #     E, dtype=torch.int32, device=hidden_states.activation.device
    # )
    # for token_id in range(B):
    #     for k in range(topk):
    #         expert_id = token_to_expert[token_id, k].item()
    #         pos = write_pos[expert_id].item()
    #         activation_per_expert[expert_id, pos] = hidden_states.activation[token_id]
    #         token_pos_in_expert[token_id, k] = pos
    #         write_pos[expert_id] += 1

    T = B * topk
    device = hidden_states.activation.device

    flat_expert = token_to_expert.view(-1).to(torch.int64)
    flat_token_ids = (
        torch.arange(B, device=device, dtype=torch.int64)
        .unsqueeze(1)
        .expand(B, topk)
        .reshape(-1)
    )

    sorted_expert, sort_idx = torch.sort(flat_expert)

    diff = torch.ones_like(sorted_expert, dtype=torch.bool)
    if T > 1:
        diff[1:] = sorted_expert[1:] != sorted_expert[:-1]

    pos_in_expert_sorted = torch.arange(
        T, device=device, dtype=torch.int64
    ) - torch.searchsorted(sorted_expert, sorted_expert, side="left")

    unsort_idx = torch.empty_like(sort_idx)
    unsort_idx[sort_idx] = torch.arange(T, device=device, dtype=torch.int64)
    flat_token_pos_in_expert = pos_in_expert_sorted[unsort_idx]
    token_pos_in_expert = flat_token_pos_in_expert.view(B, topk).to(torch.int32)

    expert_idx = flat_expert
    pos_idx = flat_token_pos_in_expert
    token_idx = flat_token_ids

    feat_idx = torch.arange(H, device=device, dtype=torch.int64)
    expert_idx_b = expert_idx.unsqueeze(1).expand(T, H)
    pos_idx_b = pos_idx.unsqueeze(1).expand(T, H)
    feat_idx_b = feat_idx.unsqueeze(0).expand(T, H)

    activation_per_expert.index_put_(
        (expert_idx_b, pos_idx_b, feat_idx_b),
        hidden_states.activation[token_idx],
        accumulate=False,
    )

    densed_hidden_states = PerExpertDenseBatchedRoutedActivation(
        activation_per_expert=activation_per_expert,
        n_tokens_per_expert=token_cnt_per_expert,
        expert_ids_are_local=hidden_states.expert_ids_are_local,
    )

    del (
        flat_expert,
        ones,
        sorted_expert,
        sort_idx,
        diff,
        pos_in_expert_sorted,
        unsort_idx,
        flat_token_pos_in_expert,
        expert_idx,
        pos_idx,
        token_idx,
        feat_idx,
        expert_idx_b,
        pos_idx_b,
        feat_idx_b,
        flat_token_ids,
        hidden_states,
    )

    intermediate_cache3 = deepgemm_masked_fused_expert(
        densed_hidden_states,
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
        experts_start_idx=experts_start_idx,
    )

    batch_size = topk_weights.shape[0]
    _, _, K = intermediate_cache3.shape
    out = torch.empty(
        batch_size,
        K,
        device=intermediate_cache3.device,
        dtype=intermediate_cache3.dtype,
    )

    return PerExpertDenseBatchedExpertResult(
        activation_per_expert=intermediate_cache3,
        token_to_expert_indices=token_to_expert,
        token_pos_in_expert=token_pos_in_expert,
    ).weighted_sum(topk_weights, out=out)


@deepgemm_masked_fused_expert.register
def _(
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
        assert isinstance(hidden_states, PerExpertDenseBatchedRoutedActivationBlockfp8)
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

        return intermediate_cache3

    assert use_fp8_w8a8
    assert block_shape is not None
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
    qintermediate_cache2, a2q_scale = silu_and_mul_and_blockfp8_act_quant(
        intermediate_cache1,
        expert_n_tokens=hidden_states.n_tokens_per_expert,
        block_size=128,
    )
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        (qintermediate_cache2, a2q_scale),
        (w2, w2_scale),
        intermediate_cache3,
        hidden_states.n_tokens_per_expert,
        M,
    )

    return intermediate_cache3
