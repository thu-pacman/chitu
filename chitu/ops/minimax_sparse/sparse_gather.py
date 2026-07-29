# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Gather block-sparse KV into a contiguous layout for Flash decode."""

from __future__ import annotations

import torch

from chitu.ops.page_table import topk_ids_to_page_ids

_GATHER_K_WORK: dict[tuple, torch.Tensor] = {}
_GATHER_V_WORK: dict[tuple, torch.Tensor] = {}


def _get_gather_work_buffers(
    k_cache: torch.Tensor,
    *,
    bsz: int,
    work_len: int,
    n_kv: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (k_cache.device, bsz, work_len, n_kv, head_dim, k_cache.dtype)
    k_work = _GATHER_K_WORK.get(key)
    v_work = _GATHER_V_WORK.get(key)
    if k_work is None:
        k_work = k_cache.new_zeros(bsz, work_len, n_kv, head_dim)
        v_work = k_cache.new_zeros(bsz, work_len, n_kv, head_dim)
        _GATHER_K_WORK[key] = k_work
        _GATHER_V_WORK[key] = v_work
    else:
        k_work.zero_()
        v_work.zero_()
    return k_work, v_work


def compute_sparse_decode_block_layout(
    block_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sorted logical blocks and per-block token counts for sparse decode."""
    if block_indices.ndim != 3 or block_indices.shape[1] != 1:
        raise ValueError(
            "sparse decode block layout expects one local index head, got "
            f"shape={tuple(block_indices.shape)}"
        )
    logical_blks = block_indices[:, 0, :].to(torch.long)
    valid = logical_blks >= 0
    num_logical_blocks = (seq_lens + page_size - 1) // page_size
    valid = valid & (logical_blks < num_logical_blocks.unsqueeze(1))
    block_starts = logical_blks.clamp(min=0) * page_size
    valid = valid & (block_starts < seq_lens.unsqueeze(1))

    sort_key = torch.where(
        valid,
        logical_blks,
        torch.iinfo(logical_blks.dtype).max,
    )
    sort_order = sort_key.argsort(dim=-1, stable=True)
    sorted_blks = logical_blks.gather(1, sort_order)
    sorted_valid = valid.gather(1, sort_order)

    sorted_block_starts = sorted_blks.clamp(min=0) * page_size
    token_end = seq_lens.unsqueeze(1)
    contrib = (
        torch.minimum(token_end, sorted_block_starts + page_size) - sorted_block_starts
    ).clamp(min=0)
    contrib = contrib * sorted_valid.to(contrib.dtype)
    sparse_lens = contrib.sum(dim=1).to(torch.int32)
    return sorted_blks, contrib, sorted_valid, sparse_lens


@torch.no_grad()
def gather_block_sparse_kv_for_decode(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    sorted_blks: torch.Tensor,
    contrib: torch.Tensor,
    sorted_valid: torch.Tensor,
    *,
    page_size: int = 128,
    max_sparse_len: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack selected logical blocks into ``[batch, sparse_len, n_kv, dim]`` tensors."""
    bsz, num_block_slots = sorted_blks.shape
    n_kv, head_dim = k_cache.shape[2], k_cache.shape[3]
    device = k_cache.device
    tokens_per_batch = num_block_slots * page_size
    max_sparse_len = min(max_sparse_len, num_block_slots * page_size)
    # Extra slot absorbs invalid scatter writes without colliding with valid tokens.
    work_len = max_sparse_len + 1

    k_work, v_work = _get_gather_work_buffers(
        k_cache,
        bsz=bsz,
        work_len=work_len,
        n_kv=n_kv,
        head_dim=head_dim,
    )

    prefix = torch.zeros(bsz, num_block_slots + 1, dtype=torch.int32, device=device)
    prefix[:, 1:] = contrib.cumsum(dim=1).to(torch.int32)
    sparse_lens = prefix[:, -1]

    in_block = torch.arange(page_size, device=device, dtype=torch.int32)
    out_idx = prefix[:, :num_block_slots].unsqueeze(-1) + in_block.view(1, 1, -1)
    slot_mask = sorted_valid.unsqueeze(-1) & (
        in_block.view(1, 1, -1) < contrib.unsqueeze(-1).to(torch.int32)
    )

    out_flat = out_idx.reshape(bsz, tokens_per_batch)
    mask_flat = slot_mask.reshape(bsz, tokens_per_batch)

    page_ids, _ = topk_ids_to_page_ids(
        sorted_blks,
        block_table,
        ids_per_page=1,
    )
    page_ids = page_ids.clamp(min=0, max=k_cache.shape[0] - 1)
    page_ids = (
        page_ids.unsqueeze(-1).expand(-1, -1, page_size).reshape(bsz, tokens_per_batch)
    )
    in_page = torch.arange(page_size, device=device).view(1, 1, -1)
    in_page = in_page.expand(bsz, num_block_slots, -1).reshape(bsz, tokens_per_batch)
    k_tokens = k_cache[page_ids.long(), in_page.long()]
    v_tokens = v_cache[page_ids.long(), in_page.long()]

    pad_idx = max_sparse_len
    safe_out = torch.where(
        mask_flat,
        out_flat.clamp(max=max_sparse_len - 1),
        torch.full_like(out_flat, pad_idx),
    ).long()
    k_vals = k_tokens * mask_flat.unsqueeze(-1).unsqueeze(-1).to(k_tokens.dtype)
    v_vals = v_tokens * mask_flat.unsqueeze(-1).unsqueeze(-1).to(v_tokens.dtype)
    scatter_idx = safe_out.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_kv, head_dim)
    k_work.scatter_(1, scatter_idx, k_vals)
    v_work.scatter_(1, scatter_idx, v_vals)

    sparse_lens = prefix[:, -1]
    return (
        k_work[:, :max_sparse_len].contiguous(),
        v_work[:, :max_sparse_len].contiguous(),
        sparse_lens,
    )
