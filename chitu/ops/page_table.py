# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for mapping sequence-local top-k ids to physical KV pages."""

import torch


def topk_ids_to_page_ids(
    topk_ids: torch.Tensor,
    block_table: torch.Tensor,
    *,
    ids_per_page: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map sequence-local ids to physical page ids and offsets within each page.

    ``ids_per_page`` is expressed in the input id space. Pass ``1`` when the
    input already contains logical block/page ids, or the KV page size when the
    input contains token ids.

    Validity is intentionally left to callers because causal bounds and padding
    semantics differ between attention backends. Indices are clamped only to
    keep the gather in bounds.
    """
    if topk_ids.ndim != 2 or block_table.ndim != 2:
        raise ValueError("topk_ids and block_table must both be 2D")
    if topk_ids.shape[0] > block_table.shape[0]:
        raise ValueError(
            "topk_ids batch dimension cannot exceed block_table batch dimension"
        )
    if ids_per_page <= 0:
        raise ValueError(f"ids_per_page must be positive, got {ids_per_page}")
    if block_table.shape[1] == 0:
        raise ValueError("block_table has zero width")

    block_table = block_table[: topk_ids.shape[0]]
    safe_ids = topk_ids.clamp(min=0)
    logical_page_ids = (safe_ids // ids_per_page).clamp(max=block_table.shape[1] - 1)
    page_ids = block_table.gather(1, logical_page_ids.to(torch.long))
    offsets = safe_ids % ids_per_page
    return page_ids, offsets
