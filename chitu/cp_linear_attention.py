# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Contiguous context-parallel support for linear attention.

This module provides the building blocks for splitting linear-attention prefill
contiguously across PCP ranks:

- contiguous rank intervals and cross-rank state-chain metadata;
- interleaved <-> contiguous all-to-all reshuffles;
- per-segment affine transitions and prefix-state merging.

Segment transitions reuse the existing chunk operator by running it with an
augmented value tensor, so no separate kernel is required. When a local segment
does not start on the global chunk grid, chunk boundaries differ from the
replicated all-gather path and results can differ by roughly 1e-4 relative;
short-convolution outputs remain bit-exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import torch

__all__ = [
    "ContiguousCPPlan",
    "build_contiguous_cp_plan",
    "interleaved_rank_indices",
    "contiguous_interval",
    "contiguous_rank_of_position",
    "contiguous_rank_indices",
    "build_layout_route",
    "build_reshuffle_route",
    "all_to_all_reshape",
    "segment_affine_transition",
    "merge_prefix_states",
]


# ---------------------------------------------------------------------------
# 1. Contiguous interval metadata
# ---------------------------------------------------------------------------


@dataclass
class ContiguousCPPlan:
    """Contiguous-CP interval and cross-rank chain metadata for one rank.

    The global token axis concatenates each request's delta tokens. ``cu_seqlens``
    is its prefix sum. Rank ``r`` owns the half-open interval
    ``[r * T // W, (r + 1) * T // W)``; segment lengths differ by at most one
    when ``T`` is not divisible by ``W``.
    """

    rank: int
    world_size: int
    total_tokens: int
    part_len: int  # Local segment length (interval_end - interval_start).
    interval_start: int
    interval_end: int  # Exclusive.

    # Global sequences with tokens in this interval, in global order, and their
    # local lengths.
    global_seq_indices: List[int] = field(default_factory=list)
    local_lens: List[int] = field(default_factory=list)
    # Number of upstream rank transitions to merge before each local segment.
    # Only the first local segment can be non-zero.
    chain_len: List[int] = field(default_factory=list)

    # Start of the leftmost global sequence, used for the short-convolution
    # left boundary.
    first_seq_global_start: int = 0
    # Tokens belonging to the first local sequence before this interval starts.
    pre_conv_tokens: int = 0

    @property
    def n_local(self) -> int:
        return self.interval_end - self.interval_start

    @property
    def n_local_seqs(self) -> int:
        return len(self.local_lens)

    @property
    def local_cu_seqlens(self) -> List[int]:
        """Prefix sums of the local segment lengths."""
        out = [0]
        acc = 0
        for n in self.local_lens:
            acc += n
            out.append(acc)
        return out

    @property
    def continues_from_prev_rank(self) -> bool:
        """Whether the first local sequence continues from an earlier rank."""
        return len(self.chain_len) > 0 and self.chain_len[0] > 0

    @property
    def is_first_rank(self) -> bool:
        return not self.continues_from_prev_rank

    @property
    def has_local_tokens(self) -> bool:
        return self.n_local > 0


def build_contiguous_cp_plan(
    cu_seqlens: Sequence[int],
    rank: int,
    world_size: int,
) -> ContiguousCPPlan:
    """Build the contiguous-CP plan for one rank from global ``cu_seqlens``.

    Global sequence boundaries are clamped to this rank's interval to form the
    local ``cu_seqlens``. ``chain_len`` records how many upstream segments must
    be merged before the first local sequence.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not (0 <= rank < world_size):
        raise ValueError(f"rank {rank} out of range [0, {world_size})")

    cu = [int(x) for x in cu_seqlens]
    if len(cu) < 2:
        raise ValueError(f"cu_seqlens must have at least 2 entries, got {cu}")
    total = cu[-1]
    # Uneven split: use floor(r * T / W) boundaries, so lengths differ by at
    # most one (see :func:`contiguous_interval`).
    start, end = contiguous_interval(rank, world_size, total)
    part_len = end - start

    plan = ContiguousCPPlan(
        rank=rank,
        world_size=world_size,
        total_tokens=total,
        part_len=part_len,
        interval_start=start,
        interval_end=end,
    )

    for seq_idx in range(len(cu) - 1):
        seq_start, seq_end = cu[seq_idx], cu[seq_idx + 1]
        lo = max(seq_start, start)
        hi = min(seq_end, end)
        if hi <= lo:
            continue
        plan.global_seq_indices.append(seq_idx)
        plan.local_lens.append(hi - lo)

    if plan.global_seq_indices:
        first_seq = plan.global_seq_indices[0]
        plan.first_seq_global_start = cu[first_seq]
        plan.pre_conv_tokens = max(0, start - cu[first_seq])
        # Only the first local sequence can continue from earlier segments.
        # Use the owner of its global start to handle uneven splits.
        first_slot = contiguous_rank_of_position(cu[first_seq], world_size, total)
        chain0 = rank - first_slot if cu[first_seq] < start else 0
        plan.chain_len = [chain0] + [0] * (len(plan.local_lens) - 1)

    return plan


# ---------------------------------------------------------------------------
# 2. Interleaved <-> contiguous reshuffle
# ---------------------------------------------------------------------------


def interleaved_rank_indices(
    rank: int, world_size: int, total_tokens: int
) -> torch.Tensor:
    """Global token indices owned by a rank under interleaved CP."""
    return torch.arange(rank, total_tokens, world_size, dtype=torch.long)


def contiguous_interval(
    rank: int, world_size: int, total_tokens: int
) -> Tuple[int, int]:
    """Return the half-open global interval ``[start, end)`` for a rank.

    ``start = rank * T // W`` and ``end = (rank + 1) * T // W``. Segment lengths
    are ``T // W`` or ``T // W + 1`` and do not require ``T % W == 0``.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not (0 <= rank < world_size):
        raise ValueError(f"rank {rank} out of range [0, {world_size})")
    if total_tokens < 0:
        raise ValueError(f"total_tokens must be >= 0, got {total_tokens}")
    return (rank * total_tokens) // world_size, (
        (rank + 1) * total_tokens
    ) // world_size


def contiguous_rank_of_position(pos: int, world_size: int, total_tokens: int) -> int:
    """Return the rank owning global position ``pos`` under contiguous CP.

    This is the inverse of :func:`contiguous_interval`.
    """
    if not (0 <= pos < total_tokens):
        raise ValueError(f"pos {pos} out of range [0, {total_tokens})")
    return (world_size * (pos + 1) - 1) // total_tokens


def contiguous_rank_indices(
    rank: int, world_size: int, total_tokens: int
) -> torch.Tensor:
    """Global token indices owned by a rank under contiguous CP."""
    start, end = contiguous_interval(rank, world_size, total_tokens)
    return torch.arange(start, end, dtype=torch.long)


def build_reshuffle_route(
    src_positions_by_rank: Sequence[torch.Tensor],
    dst_positions_by_rank: Sequence[torch.Tensor],
    total_tokens: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[int]], List[List[int]]]:
    """Build the all-to-all route between two per-rank token layouts.

    ``src_positions_by_rank[r]`` and ``dst_positions_by_rank[r]`` are the
    ascending global token indices held by rank ``r`` before and after the
    reshuffle. Both lists must partition ``[0, total_tokens)``.

    Returns ``(send_index, recv_index, send_counts, recv_counts)``:

    - ``send_index[r]``: local row order to send from rank ``r``, grouped by
      destination rank and stable within each group;
    - ``send_counts[r][d]``: rows sent from rank ``r`` to rank ``d``;
    - ``recv_counts[r]``: rows received by rank ``r`` from each source rank;
    - ``recv_index[r]``: permutation restoring the received buffer to ascending
      target order.
    """
    world_size = len(src_positions_by_rank)
    if len(dst_positions_by_rank) != world_size:
        raise ValueError("src/dst route lists must have the same length")

    total = sum(int(dst.numel()) for dst in dst_positions_by_rank)
    if total != total_tokens:
        raise ValueError(f"dst route covers {total} tokens, expected {total_tokens}")
    if sum(int(p.numel()) for p in src_positions_by_rank) != total_tokens:
        raise ValueError("src route does not cover every token exactly once")

    # Destination rank for every global token.
    dst_rank_of_pos = torch.full((total_tokens,), -1, dtype=torch.long)
    for r, dst in enumerate(dst_positions_by_rank):
        dst_rank_of_pos[dst] = r
    if bool((dst_rank_of_pos < 0).any()):
        raise ValueError("dst route does not cover every token exactly once")
    # Source rank for every global token, used to restore received order.
    src_rank_of_pos = torch.full((total_tokens,), -1, dtype=torch.long)
    for r, src in enumerate(src_positions_by_rank):
        src_rank_of_pos[src] = r
    if bool((src_rank_of_pos < 0).any()):
        raise ValueError("src route does not cover every token exactly once")

    send_index: List[torch.Tensor] = []
    recv_index: List[torch.Tensor] = []
    send_counts: List[List[int]] = []
    recv_counts: List[List[int]] = []

    for r in range(world_size):
        pos = src_positions_by_rank[r]
        owners = dst_rank_of_pos[pos]
        send_index.append(torch.argsort(owners, stable=True))
        counts = torch.bincount(owners, minlength=world_size)
        send_counts.append([int(c) for c in counts.tolist()])

    for r in range(world_size):
        counts = [send_counts[s][r] for s in range(world_size)]
        recv_counts.append(counts)
        dst = dst_positions_by_rank[r]
        if dst.numel() == 0:
            recv_index.append(torch.empty(0, dtype=torch.long))
            continue
        # The receive buffer is concatenated by source rank with ascending
        # global positions inside each block; invert that order here.
        src_of_pos = src_rank_of_pos[dst]
        order = torch.argsort(src_of_pos, stable=True)
        slot = torch.empty_like(order)
        slot[order] = torch.arange(order.numel(), dtype=torch.long)
        recv_index.append(slot)

    return send_index, recv_index, send_counts, recv_counts


def build_layout_route(
    world_size: int, total_tokens: int, *, to_contiguous: bool
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[List[int]],
    List[List[int]],
]:
    """Build the interleaved <-> contiguous reshuffle route.

    Returns:
        ``(src_positions_by_rank, dst_positions_by_rank, send_index, recv_index,
        send_counts, recv_counts)``.
    """
    inter = [
        interleaved_rank_indices(r, world_size, total_tokens) for r in range(world_size)
    ]
    cont = [
        contiguous_rank_indices(r, world_size, total_tokens) for r in range(world_size)
    ]
    if to_contiguous:
        src, dst = inter, cont
    else:
        src, dst = cont, inter
    send_index, recv_index, send_counts, recv_counts = build_reshuffle_route(
        src, dst, total_tokens
    )
    return src, dst, send_index, recv_index, send_counts, recv_counts


def all_to_all_reshape(
    x: torch.Tensor,
    rank: int,
    send_index: Sequence[torch.Tensor],
    recv_index: Sequence[torch.Tensor],
    send_counts: Sequence[Sequence[int]],
    recv_counts: Sequence[Sequence[int]],
    group,
) -> torch.Tensor:
    """Move local rows ``x`` to the target layout using a precomputed route.

    ``x`` has shape ``[n_local, ...]``; the result has shape ``[n_local_dst, ...]``.
    """
    import torch.distributed as dist

    if group is None:
        raise ValueError("all_to_all_reshape requires a valid process group")
    if x.shape[0] != len(send_index[rank]):
        raise ValueError(
            f"rank {rank}: local rows {x.shape[0]} != routed send rows "
            f"{len(send_index[rank])}"
        )
    send_buf = x.index_select(0, send_index[rank].to(x.device)).contiguous()
    recv_buf = x.new_empty((sum(recv_counts[rank]),) + tuple(x.shape[1:]))
    dist.all_to_all_single(
        recv_buf,
        send_buf,
        output_split_sizes=list(recv_counts[rank]),
        input_split_sizes=list(send_counts[rank]),
        group=group,
    )
    return recv_buf.index_select(0, recv_index[rank].to(x.device))


# ---------------------------------------------------------------------------
# 3. Segment affine transitions and prefix-state merging
# ---------------------------------------------------------------------------


def segment_affine_transition(
    chunk_op,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_len_list: Sequence[int],
    *,
    use_qk_l2norm_in_kernel: bool,
    impl: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute each local sequence's affine transition ``S_out = M @ S_in + S_ext``.

    The recurrence is affine in ``(S_in, v)``. Run the existing chunk operator
    once with ``v`` augmented to ``[0 | v]`` and an initial state ``[I | 0]``;
    the final state contains ``M`` in its first ``K`` columns and ``S_ext`` in
    the remaining columns. This avoids a separate transition kernel.

    Returns:
        ``(M, S_ext)`` with shapes ``[N, H, K, K]`` and ``[N, H, K, V]`` in fp32.
    """
    n_seqs = len(seq_len_list)
    n_heads = key.shape[-2]
    k_dim = key.shape[-1]
    v_dim = value.shape[-1]

    # Augment v to [0 | v]; only the trailing V columns carry external input.
    zero_cols = value.new_zeros((*value.shape[:-1], k_dim))
    value_aug = torch.cat([zero_cols, value], dim=-1)

    identity = torch.eye(k_dim, dtype=torch.float32, device=query.device)
    identity = identity.view(1, 1, k_dim, k_dim).expand(n_seqs, n_heads, k_dim, k_dim)
    zero_block = torch.zeros(
        (n_seqs, n_heads, k_dim, v_dim), dtype=torch.float32, device=query.device
    )
    initial_aug = torch.cat([identity, zero_block], dim=-1).contiguous()

    _, last_state = chunk_op(
        query,
        key,
        value_aug,
        g=g,
        beta=beta,
        initial_state=initial_aug,
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        seq_len_list=list(seq_len_list),
        impl=impl,
    )
    m = last_state[..., :k_dim].to(torch.float32)
    s_ext = last_state[..., k_dim:].to(torch.float32)
    return m, s_ext


def merge_prefix_states(
    m_by_slot: torch.Tensor,
    ext_by_slot: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Merge ordered segment transitions into ``initial_state``.

    ``m_by_slot`` and ``ext_by_slot`` have shapes ``[n_slots, H, K, K]`` and
    ``[n_slots, H, K, V]`` in chronological order; ``initial_state`` has shape
    ``[H, K, V]``. The chain is evaluated in fp32:

        S = S0
        for j in slots: S = M_j @ S + S_ext_j
    """
    if m_by_slot.shape[0] != ext_by_slot.shape[0]:
        raise ValueError("m_by_slot and ext_by_slot must have the same slot count")
    s = initial_state.to(torch.float32)
    for j in range(m_by_slot.shape[0]):
        s = torch.matmul(m_by_slot[j], s) + ext_by_slot[j]
    return s
