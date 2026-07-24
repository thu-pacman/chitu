# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Decode-time compression plan for DeepSeek-V4.

The plan stores per-request facts and exposes row views on demand.  Callers pick
full rows or compact rows according to the kernel/backend they are about to run;
the plan itself does not bake in that dispatch decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DeepSeekV4DecodeCompressPlan:
    """Per-request facts needed by decode compression and cache writeback."""

    start_positions: torch.Tensor
    cache_slots: torch.Tensor
    cache_seq_ids: Optional[torch.Tensor]
    should_compress: torch.Tensor
    ratio: int
    q_len: int
    head_dim: int
    is_csa: bool

    @property
    def batch_size(self) -> int:
        return int(self.start_positions.numel())

    @property
    def full_request_rows(self) -> torch.Tensor:
        return torch.arange(
            self.batch_size,
            device=self.start_positions.device,
            dtype=torch.long,
        )

    @property
    def full_row_is_valid(self) -> torch.Tensor:
        return self.should_compress

    @property
    def compact_request_rows(self) -> torch.Tensor:
        return torch.nonzero(self.should_compress, as_tuple=False).flatten()


def decode_should_compress(
    start_positions: torch.Tensor,
    *,
    ratio: int,
    q_len: int,
) -> torch.Tensor:
    """Per-request boundary test for DeepSeek-V4 compressed KV production."""
    return start_positions.remainder(ratio) + q_len >= ratio


def build_decode_compress_plan(
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    cache_seq_ids: Optional[torch.Tensor] = None,
    *,
    ratio: int,
    q_len: int,
    head_dim: int,
    is_csa: bool,
    use_cuda_graph: bool,
) -> DeepSeekV4DecodeCompressPlan:
    """Build per-request decode compression facts.

    ``use_cuda_graph`` remains in the signature so call sites can pass the same
    runtime context while row selection moves to the operator dispatch layer.
    """
    if start_positions.dtype != torch.long:
        start_positions = start_positions.to(dtype=torch.long)
    if cache_slots.dtype != torch.long:
        cache_slots = cache_slots.to(dtype=torch.long)
    if cache_seq_ids is not None and cache_seq_ids.dtype != torch.long:
        cache_seq_ids = cache_seq_ids.to(dtype=torch.long)

    should_compress = decode_should_compress(
        start_positions,
        ratio=ratio,
        q_len=q_len,
    )

    return DeepSeekV4DecodeCompressPlan(
        start_positions=start_positions,
        cache_slots=cache_slots,
        cache_seq_ids=cache_seq_ids,
        should_compress=should_compress,
        ratio=int(ratio),
        q_len=int(q_len),
        head_dim=int(head_dim),
        is_csa=bool(is_csa),
    )
