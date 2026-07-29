# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Top-k block index layout helpers for MiniMax M3 sparse attention."""

from __future__ import annotations

import torch


def topk_index_reduce(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Union top-k indices along ``dim`` (dedupe, left-align, pad with -1)."""
    tensor_permuted = torch.movedim(tensor, source=dim, destination=-2)
    combined = tensor_permuted.flatten(start_dim=-2)

    sorted_vals, _ = combined.sort(dim=-1)
    is_new_element = sorted_vals[..., 1:] != sorted_vals[..., :-1]
    first_col_true = torch.ones_like(sorted_vals[..., :1], dtype=torch.bool)
    non_duplicate_mask = torch.cat([first_col_true, is_new_element], dim=-1)
    valid_mask = non_duplicate_mask & (sorted_vals != -1)
    sort_idx = torch.argsort((~valid_mask).int(), dim=-1, stable=True)
    result = torch.gather(sorted_vals, -1, sort_idx)
    valid_count = valid_mask.sum(dim=-1, keepdim=True)
    total_cols = result.size(-1)
    col_idx = torch.arange(total_cols, device=result.device).view(
        *([1] * (result.dim() - 1)), total_cols
    )
    return result.masked_fill(col_idx >= valid_count, -1)


def block_indices_to_kv_topk(
    block_indices: torch.Tensor,
    n_kv_heads: int,
) -> torch.Tensor:
    """Convert indexer output to kernel layout ``[n_kv_heads, total_q, topk]``."""
    if block_indices.ndim != 3:
        raise ValueError(
            f"block_indices must be [total_q, n_idx_heads, topk], got {block_indices.shape}"
        )
    total_q, n_idx_heads, topk = block_indices.shape
    if n_idx_heads == n_kv_heads:
        return block_indices.transpose(0, 1).contiguous()
    if n_idx_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_idx_heads ({n_idx_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )
    group = n_idx_heads // n_kv_heads
    grouped = block_indices.view(total_q, n_kv_heads, group, topk).permute(1, 2, 0, 3)
    return topk_index_reduce(grouped, dim=1)


def stack_paged_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Stack separate paged K/V into ``[num_blocks, 2, page_size, n_kv, dim]``."""
    return torch.stack([k, v], dim=1)
