# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

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


# q, weights, k_full, seq_len_delta, is_causal, ke, ks, impl
@make_op_dispatcher
def bf16_index_score_ragged_qk_dsv32(
    q: torch.Tensor,  # [s_q, H, D] bf16
    weights: torch.Tensor,  # [s_q, H] fp32
    k_cache: torch.Tensor,  # [s_k, D] bf16 ragged concat
    seq_len_delta: BatchedSeqLenDelta,
    is_casual: bool,
    ke: Optional[torch.Tensor] = None,
    ks: Optional[torch.Tensor] = None,
    impl: str = "auto",
) -> torch.Tensor:  # [s_q, max(ke - ks)]
    """Prefill-stage bf16 ragged-QK MQA index score."""
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
    return "torch"


bf16_index_score_ragged_q_paged_k_dsv32.register_candidate("triton")
if has_triton:
    bf16_index_score_ragged_q_paged_k_dsv32.register("triton")(
        bf16_index_score_ragged_q_paged_k_dsv32_triton
    )


@bf16_index_score_ragged_qk_dsv32.register("torch")
def bf16_index_score_ragged_qk_dsv32_torch(
    q: torch.Tensor,  # [s_q, H, D] bf16
    weights: torch.Tensor,  # [s_q, H] fp32
    k_cache: torch.Tensor,  # [s_k, D] bf16 ragged concat
    seq_len_delta: BatchedSeqLenDelta,
    is_casual: bool,
    ke: Optional[torch.Tensor] = None,  # [s_q] int32, global end offsets
    ks: Optional[torch.Tensor] = None,  # [s_q] int32, global start offsets
) -> torch.Tensor:  # [s_q, max(ke - ks)]
    """Pure-torch bf16 ragged-qk indexer score.

    The output columns are request-local key offsets. ``ks``/``ke`` are per-query
    global ragged-K bounds, so CP-local query metadata can be passed directly.
    """
    s_q, h, _ = q.shape
    weights = weights.reshape(s_q, h)
    if s_q == 0:
        return torch.empty((0, 0), dtype=torch.float32, device=q.device)

    if ks is None:
        ks = seq_len_delta.new.prefix_lens_tensor_device[
            seq_len_delta.delta_seq_ids_tensor_device
        ]
    if ke is None:
        if is_casual:
            ke = seq_len_delta.delta_position_ids_tensor_device + ks + 1
        else:
            ke = (
                seq_len_delta.new.lens_tensor_device[
                    seq_len_delta.delta_seq_ids_tensor_device
                ]
                + ks
            )

    ks = ks.to(device=q.device, dtype=torch.int32).contiguous()
    ke = ke.to(device=q.device, dtype=torch.int32).contiguous()
    actual_max_n = int((ke - ks).max().item())
    if actual_max_n <= 0:
        return torch.full((s_q, 0), float("-inf"), dtype=torch.float32, device=q.device)

    qk = torch.matmul(q, k_cache.transpose(0, 1))
    qk = torch.relu(qk)
    score_global = (
        (qk * weights.to(qk.dtype).unsqueeze(-1)).sum(dim=1).to(torch.float32)
    )

    s_k = k_cache.shape[0]
    j_local = torch.arange(actual_max_n, device=q.device, dtype=ks.dtype)
    j_global = ks.unsqueeze(1) + j_local.unsqueeze(0)
    in_range = j_global < ke.unsqueeze(1)
    j_clamped = j_global.clamp(min=0, max=max(s_k - 1, 0)).to(torch.long)
    gathered = score_global.gather(1, j_clamped)
    return torch.where(in_range, gathered, torch.full_like(gathered, float("-inf")))


bf16_index_score_ragged_qk_dsv32.register_candidate("triton")
if has_triton:
    bf16_index_score_ragged_qk_dsv32.register("triton")(
        bf16_index_score_ragged_qk_dsv32_triton
    )
