# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.ops.utils import make_op_dispatcher
from chitu.utils import try_import_platform_dep
from chitu.batched_seq_len import BatchedSeqLenDelta

triton, has_triton = try_import_platform_dep("triton")

if has_triton:
    from chitu.ops.triton_ops import (
        bf16_index_score_ragged_q_paged_k_dsv32_triton,
        bf16_index_score_ragged_qk_dsv32_triton,
    )


@make_op_dispatcher
def bf16_index_score_ragged_q_paged_k_dsv32(
    q: torch.Tensor,  # [s_q, H, D] bf16
    weights: torch.Tensor,  # [s_q, H] fp32
    k_cache: torch.Tensor,  # [n_pages, page_size, D] bf16
    seq_len_delta: BatchedSeqLenDelta,
    k_page_table: torch.Tensor,  # [b, n_pages_per_seq]
    max_seq_len: int,
    impl: str = "auto",
) -> torch.Tensor:  # [s_q, max_seq_len]
    """Decode-stage bf16 paged MQA index score."""
    raise NotImplementedError


# q, weights, k_full, seq_len_delta, is_causal, ke, ks, impl="triton"
@make_op_dispatcher
def bf16_index_score_ragged_qk_dsv32(
    q: torch.Tensor,  # [s_q, H, D] bf16
    weights: torch.Tensor,  # [s_q, H] fp32
    k_cache: torch.Tensor,  # [n_pages, page_size, D] bf16
    seq_len_delta: BatchedSeqLenDelta,
    is_casual: bool,
    ke,  # Optional[torch.Tensor] = None
    ks,  # Optional[torch.Tensor] = None
    impl: str = "auto",
) -> torch.Tensor:  # [s_q, max_seq_len]
    """Decode-stage bf16 paged MQA index score."""
    raise NotImplementedError


@bf16_index_score_ragged_q_paged_k_dsv32.register_auto
def _auto_bf16_index_score_ragged_q_paged_k_dsv32():
    if has_triton:
        return "triton"
    raise NotImplementedError("bf16_index_score_ragged_q_paged_k_dsv32 requires triton")


@bf16_index_score_ragged_qk_dsv32.register_auto
def _auto_bf16_index_score_ragged_qk_dsv32():
    if has_triton:
        return "triton"
    raise NotImplementedError("bf16_index_score_ragged_qk_dsv32 requires triton")


bf16_index_score_ragged_q_paged_k_dsv32.register_candidate("triton")
if has_triton:
    bf16_index_score_ragged_q_paged_k_dsv32.register("triton")(
        bf16_index_score_ragged_q_paged_k_dsv32_triton
    )

bf16_index_score_ragged_qk_dsv32.register_candidate("triton")
if has_triton:
    bf16_index_score_ragged_qk_dsv32.register("triton")(
        bf16_index_score_ragged_qk_dsv32_triton
    )
