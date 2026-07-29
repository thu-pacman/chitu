# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Graph-safe MiniMax M3 indexer for classic decode (batched GPU path)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.global_vars import get_global_args
from chitu.kv_cache import KVCacheAccessor, PagedKVCacheAccessor


def _read_idx_k_tokens(
    kv: torch.Tensor,
    block_table: torch.Tensor,
    positions_2d: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Gather indexer K tokens from paged cache with bounds-safe indexing."""
    page_size = kv.shape[1]
    page_idx = (positions_2d // page_size).clamp(min=0, max=block_table.shape[1] - 1)
    phys = block_table.gather(1, page_idx.to(torch.long)).clamp(0, kv.shape[0] - 1)
    off = (positions_2d % page_size).to(torch.long)
    out = kv[phys, off]
    return out * valid.unsqueeze(-1).to(out.dtype)


def _compute_block_scores_vectorized(
    idx_q_f: torch.Tensor,
    kv: torch.Tensor,
    block_table: torch.Tensor,
    *,
    block_size: int,
    max_num_blocks: int,
    k_lens: torch.Tensor,
    position_ids: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Score all logical key blocks in one batched pass (CUDA-graph friendly)."""
    bsz = idx_q_f.shape[0]
    device = idx_q_f.device
    in_block_pos = torch.arange(block_size, device=device, dtype=torch.int32)
    block_starts = (
        torch.arange(max_num_blocks, device=device, dtype=torch.int32) * block_size
    )

    positions_3d = (
        block_starts.view(1, max_num_blocks, 1) + in_block_pos.view(1, 1, block_size)
    ).expand(bsz, -1, -1)

    page_idx_3d = positions_3d // page_size
    num_blocks_per_seq = (k_lens + page_size - 1) // page_size
    valid = positions_3d < k_lens.view(bsz, 1, 1)
    valid = valid & (page_idx_3d < num_blocks_per_seq.view(bsz, 1, 1))
    valid = valid & (page_idx_3d < block_table.shape[1])

    positions_2d = positions_3d.reshape(bsz, max_num_blocks * block_size)
    valid_2d = valid.reshape(bsz, max_num_blocks * block_size)
    idx_k_all = _read_idx_k_tokens(kv, block_table, positions_2d, valid_2d)
    idx_k_all = idx_k_all.view(bsz, max_num_blocks, block_size, -1)

    pos = position_ids.view(bsz, 1, 1, 1)
    k_lens_v = k_lens.view(bsz, 1, 1, 1)
    k_pos = positions_3d.unsqueeze(1)
    valid_4d = valid.reshape(bsz, max_num_blocks, block_size).unsqueeze(1)
    scores = torch.einsum("bhd,blsd->bhls", idx_q_f, idx_k_all.float())
    scores = scores.masked_fill(~valid_4d, float("-inf"))
    scores = scores.masked_fill(
        (k_pos > pos) | (k_pos >= k_lens_v),
        float("-inf"),
    )
    return scores.amax(dim=-1)


def compute_block_indices_classic_decode_batched(
    *,
    idx_q: torch.Tensor,
    block_scores: torch.Tensor,
    position_ids: torch.Tensor,
    k_lens: torch.Tensor,
    block_size: int,
    topk_blocks: int,
    local_blocks: int,
    n_local_index_heads: int,
) -> torch.Tensor:
    """Top-k block selection from precomputed block scores.

    Args:
        idx_q: ``[batch, n_local_index_heads, head_dim]`` (for device/dtype)
        block_scores: ``[batch, n_local_index_heads, max_num_blocks]``
        position_ids: ``[batch]``
        k_lens: ``[batch]`` device int tensor
    Returns:
        ``[batch, n_local_index_heads, topk_blocks]``
    """
    bsz = idx_q.shape[0]
    max_num_blocks = block_scores.shape[-1]

    num_key_blocks = -(-k_lens // block_size)
    q_block = position_ids // block_size
    if local_blocks > 0:
        local = torch.arange(local_blocks, device=idx_q.device)
        local_idx = (q_block.view(bsz, 1) - local.view(1, -1)).clamp(min=0)
        local_idx = local_idx.unsqueeze(1).expand(-1, n_local_index_heads, -1)
        valid_local = local_idx < num_key_blocks.view(bsz, 1, 1)
        safe_local_idx = local_idx.clamp(max=max_num_blocks - 1)
        local_updates = torch.full_like(block_scores, float("-inf"))
        local_values = torch.where(
            valid_local,
            torch.full_like(local_idx, float("inf"), dtype=block_scores.dtype),
            torch.full_like(local_idx, float("-inf"), dtype=block_scores.dtype),
        )
        local_updates.scatter_(
            -1,
            safe_local_idx,
            local_values,
        )
        block_scores = torch.maximum(block_scores, local_updates)

    pick = min(topk_blocks, max_num_blocks)
    topk_scores, topk_indices = block_scores.topk(pick, dim=-1)
    topk_indices = topk_indices.masked_fill(topk_scores == float("-inf"), -1)
    if topk_indices.shape[-1] < topk_blocks:
        topk_indices = F.pad(
            topk_indices,
            (0, topk_blocks - topk_indices.shape[-1]),
            value=-1,
        )
    return topk_indices


def indexer_classic_decode_after_append(
    indexer,
    idx_q: torch.Tensor,
    seq_len_delta: BatchedSeqLenDelta,
    cache_accessor: PagedKVCacheAccessor,
) -> torch.Tensor:
    """Compute block indices after ``_project_qk`` and ``_append_idx_k``."""
    bsz = seq_len_delta.batch_size
    block_size = indexer.block_size
    # Static upper bound for CUDA graph (dynamic max_len is frozen at capture).
    max_seq_len = get_global_args().infer.max_seq_len
    max_num_blocks = -(-max_seq_len // block_size)

    idx_q = idx_q.reshape(-1, indexer.n_local_index_heads, indexer.index_head_dim)[:bsz]
    position_ids = seq_len_delta.delta_position_ids_tensor_device.view(bsz)
    k_lens = seq_len_delta.new.lens_tensor_device

    kv = cache_accessor.kv["idx_k"]
    block_table = cache_accessor.block_table[:bsz]
    page_size = kv.shape[1]

    idx_q_f = idx_q.float()
    block_scores = _compute_block_scores_vectorized(
        idx_q_f,
        kv,
        block_table,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        k_lens=k_lens,
        position_ids=position_ids,
        page_size=page_size,
    )

    return compute_block_indices_classic_decode_batched(
        idx_q=idx_q,
        block_scores=block_scores,
        position_ids=position_ids,
        k_lens=k_lens,
        block_size=block_size,
        topk_blocks=indexer.topk_blocks,
        local_blocks=indexer.local_blocks,
        n_local_index_heads=indexer.n_local_index_heads,
    )
