# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
CP (Context Parallelism) utilities.

Provides CPContext / NoOpCPContext that encapsulate all CP state and operations.
Access via get_cp_context() singleton. When pcp_size == 1, returns NoOpCPContext
(identity/no-op). When pcp_size > 1, create_cp_context() selects a layout-specific
implementation: CPContextInterleaved or CPContextContiguous.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from typing import Optional, Sequence, Tuple

from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.batched_seq_len import BatchedSeqLenDelta, BatchedSeqLenDeltaView
from chitu.cp_linear_attention import (
    all_to_all_reshape,
    build_layout_route,
    build_zigzag_route,
    contiguous_interval,
    zigzag_rank_indices,
)
from chitu.distributed.comm_group import CommGroup
from chitu.utils import ceil_div

LayoutRoute = Tuple[
    Sequence[torch.Tensor],
    Sequence[torch.Tensor],
    Sequence[Sequence[int]],
    Sequence[Sequence[int]],
]

# ---- Global singleton ----

_CP_CONTEXT_INSTANCE: Optional[CPContext | NoOpCPContext] = None


def create_cp_context(
    pcp_size: int, cp_group: CommGroup, layout: str = "interleaved"
) -> CPContext | NoOpCPContext:
    """Create the CP context for the selected stream layout."""
    if pcp_size <= 1:
        return NoOpCPContext()
    if layout == "interleaved":
        return CPContextInterleaved(pcp_size, cp_group)
    if layout == "contiguous":
        return CPContextContiguous(pcp_size, cp_group)
    raise ValueError(f"Unsupported pcp_layout={layout!r}")


def get_cp_context() -> CPContext | NoOpCPContext:
    """Get the global CPContext singleton."""
    global _CP_CONTEXT_INSTANCE
    if _CP_CONTEXT_INSTANCE is None:
        # Lazy init from parallel_state (only if parallel groups not yet set up)
        from chitu.distributed.parallel_state import get_pcp_size, get_pcp_group

        pcp_size = get_pcp_size()
        if pcp_size <= 1:
            _CP_CONTEXT_INSTANCE = NoOpCPContext()
        else:
            from chitu.global_vars import get_global_args

            layout = getattr(get_global_args().infer, "pcp_layout", "interleaved")
            _CP_CONTEXT_INSTANCE = create_cp_context(pcp_size, get_pcp_group(), layout)
    return _CP_CONTEXT_INSTANCE


def _set_cp_context(ctx: CPContext | NoOpCPContext):
    """Set the global CPContext singleton (called during parallel group init)."""
    global _CP_CONTEXT_INSTANCE
    _CP_CONTEXT_INSTANCE = ctx


def _reset_cp_context():
    """Reset singleton (for tests or re-init)."""
    global _CP_CONTEXT_INSTANCE
    _CP_CONTEXT_INSTANCE = None


@contextmanager
def use_cp_context(ctx: "CPContext | NoOpCPContext"):
    """Temporarily use another CP context for a nested operation."""
    global _CP_CONTEXT_INSTANCE
    old = _CP_CONTEXT_INSTANCE
    _CP_CONTEXT_INSTANCE = ctx
    try:
        yield
    finally:
        _CP_CONTEXT_INSTANCE = old


def build_cp_reorder_idx(pcp_size: int, n_local: int, device) -> torch.Tensor:
    """
    Build reorder index for CP allgather transpose.

    allgather output is rank-concatenated: [rank0_n, rank1_n, ..., rank7_n].
    Need to reorder to token-interleaved: [token0, token1, ..., token{N-1}].

    token_i is on rank i % pcp_size, at local position i // pcp_size,
    which sits at allgather position = rank * n_local + local_pos.
    """
    N = pcp_size * n_local
    ranks = torch.arange(pcp_size, device=device).repeat(n_local)
    local_offsets = torch.arange(n_local, device=device).repeat_interleave(pcp_size)
    return ranks * n_local + local_offsets


def build_cp_local_indices(
    cp_rank: int, pcp_size: int, total_tokens: int, device
) -> torch.Tensor:
    """Build interleaved global token indices owned by one CP rank."""
    if cp_rank >= total_tokens:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.arange(
        cp_rank,
        total_tokens,
        pcp_size,
        device=device,
        dtype=torch.long,
    )


def contiguous_interval(
    rank: int, world_size: int, total_tokens: int
) -> Tuple[int, int]:
    """Return one rank's balanced contiguous slice of the global token axis.

    The slice is ``[rank * T // W, (rank + 1) * T // W)``. Segment lengths
    differ by at most one token, and ``T`` does not need to be divisible by
    ``W``.
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


def build_contiguous_chunk_lengths(total_tokens: int, pcp_size: int) -> list[int]:
    """Per-rank lengths for :func:`contiguous_interval`."""
    return [
        contiguous_interval(rank, pcp_size, total_tokens)[1]
        - contiguous_interval(rank, pcp_size, total_tokens)[0]
        for rank in range(pcp_size)
    ]


def build_cp_contiguous_reorder_idx(
    chunk_lengths: Sequence[int], expected_n_local: int, device
) -> torch.Tensor:
    """Remove per-rank padding from a contiguous-chunk allgather result.

    ``all_gather_into_tensor`` requires equal input shapes. Each rank pads its
    contiguous chunk to ``expected_n_local`` rows, so the gathered layout is
    ``[rank0_real, rank0_pad, rank1_real, rank1_pad, ...]``. This index selects
    only the real rows while preserving rank order, which is already global token
    order for contiguous chunks.
    """
    expected_n_local = int(expected_n_local)
    if expected_n_local <= 0:
        raise ValueError(f"expected_n_local must be positive, got {expected_n_local}")
    parts = []
    for rank, length in enumerate(chunk_lengths):
        if length < 0 or length > expected_n_local:
            raise ValueError(
                f"chunk length {length} for rank {rank} is outside [0, "
                f"{expected_n_local}]"
            )
        parts.append(
            torch.arange(length, device=device, dtype=torch.long)
            + rank * expected_n_local
        )
    if not parts:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.cat(parts, dim=0)


def _build_layout_route(
    cp_ctx: "CPContext",
    total_tokens: int,
    source_layout: str,
    target_layout: str,
) -> LayoutRoute:
    """Build the one-way all-to-all route between two layouts."""
    if {source_layout, target_layout} == {"interleaved", "contiguous"}:
        _, _, send_index, recv_index, send_counts, recv_counts = build_layout_route(
            cp_ctx.pcp_size,
            int(total_tokens),
            to_contiguous=target_layout == "contiguous",
        )
        return send_index, recv_index, send_counts, recv_counts
    if {source_layout, target_layout} == {"contiguous", "zigzag"}:
        return build_zigzag_route(
            cp_ctx.pcp_size,
            int(total_tokens),
            to_zigzag=target_layout == "zigzag",
        )
    raise NotImplementedError(
        f"Unsupported layout transform {source_layout!r} -> {target_layout!r}"
    )


def _get_layout_routes(
    cp_ctx: "CPContext",
    total_tokens: int,
    source_layout: str,
    target_layout: str,
) -> tuple[LayoutRoute, LayoutRoute]:
    """Cached forward and reverse routes for one layout transform."""
    key = (cp_ctx.pcp_size, int(total_tokens), source_layout, target_layout)
    cached = getattr(cp_ctx, "_layout_route_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    fwd = _build_layout_route(cp_ctx, total_tokens, source_layout, target_layout)
    bwd = _build_layout_route(cp_ctx, total_tokens, target_layout, source_layout)
    routes = (fwd, bwd)
    cp_ctx._layout_route_cache = (key, routes)
    return routes


def to_layout(
    x: torch.Tensor,
    cp_ctx: "CPContext",
    total_tokens: int,
    target_layout: str,
) -> tuple[torch.Tensor, Optional[LayoutRoute]]:
    """Move a local token chunk to ``target_layout``.

    Supported transforms are interleaved <-> contiguous residual layouts and
    contiguous <-> zigzag MLA chunk layouts. Returns ``(x_target,
    restore_route)``. If the context already uses the target layout, this is a
    no-op and ``restore_route`` is ``None``. The route is consumed by
    :func:`from_layout`.
    """
    source_layout = getattr(cp_ctx, "layout", "interleaved")
    if source_layout == target_layout:
        return x, None

    fwd_route, bwd_route = _get_layout_routes(
        cp_ctx, total_tokens, source_layout, target_layout
    )
    group = cp_ctx.cp_group.gpu_group if cp_ctx.cp_group is not None else None
    x_target = all_to_all_reshape(x, cp_ctx.cp_rank, *fwd_route, group=group)
    return x_target, bwd_route


def from_layout(
    x: torch.Tensor,
    restore_route: Optional[LayoutRoute],
    cp_ctx: "CPContext",
) -> torch.Tensor:
    """Restore a transformed tensor to the context's residual layout."""
    if restore_route is None:
        return x

    group = cp_ctx.cp_group.gpu_group if cp_ctx.cp_group is not None else None
    return all_to_all_reshape(x, cp_ctx.cp_rank, *restore_route, group=group)


@dataclass
class CPChunkLayout:
    """Per-step metadata for a contiguous MLA/KV chunk."""

    start: int
    end: int
    chunk_lengths: list[int]
    cache_positions: torch.Tensor
    cache_seq_ids: torch.Tensor
    seq_len_delta: BatchedSeqLenDelta

    @property
    def local_rows(self) -> int:
        return self.end - self.start


def _build_mla_interval_layout(
    seq_len_delta: BatchedSeqLenDelta,
    start: int,
    end: int,
    *,
    chunk_lengths: Optional[Sequence[int]] = None,
) -> CPChunkLayout:
    """Build MLA chunk metadata for one global token interval.

    ``chunk_lengths`` is the per-rank length list used by ordered allgather. It
    defaults to this interval's local length for layer-local chunks.
    """
    total_tokens = int(seq_len_delta.delta_total_len)
    old_lens = seq_len_delta.old.lens_list
    delta_lens = seq_len_delta.delta_lens_list
    if len(old_lens) != len(delta_lens):
        raise RuntimeError("CP MLA chunk received inconsistent request lengths")
    cu = [0]
    for length in delta_lens:
        cu.append(cu[-1] + int(length))
    if cu[-1] != total_tokens:
        raise RuntimeError(
            "CP MLA chunk delta lengths do not sum to the global token count: "
            f"got {cu[-1]}, expected {total_tokens}"
        )

    present_seq_ids = []
    k_lens = []
    chunk_old_lens = []
    chunk_new_lens = []
    for seq_id in range(len(delta_lens)):
        seq_start, seq_end = cu[seq_id], cu[seq_id + 1]
        a = max(start, seq_start)
        b = min(end, seq_end)
        if b <= a:
            continue
        present_seq_ids.append(seq_id)
        old_for_q = int(old_lens[seq_id]) + (a - seq_start)
        new_for_q = int(old_lens[seq_id]) + (b - seq_start)
        chunk_old_lens.append(old_for_q)
        chunk_new_lens.append(new_for_q)
        k_lens.append(new_for_q)

    device = seq_len_delta.device
    cache_positions = []
    cache_seq_ids = []
    for seq_id, k_len in zip(present_seq_ids, k_lens):
        cache_positions.append(torch.arange(k_len, device=device, dtype=torch.int32))
        cache_seq_ids.append(
            torch.full((k_len,), seq_id, device=device, dtype=torch.int32)
        )
    if cache_positions:
        cache_positions_tensor = torch.cat(cache_positions, dim=0)
        cache_seq_ids_tensor = torch.cat(cache_seq_ids, dim=0)
    else:
        cache_positions_tensor = torch.empty(0, device=device, dtype=torch.int32)
        cache_seq_ids_tensor = torch.empty(0, device=device, dtype=torch.int32)

    chunk_delta = BatchedSeqLenDelta(
        chunk_old_lens,
        chunk_new_lens,
        device=device,
        use_prefix_lens_static_tensor=False,
        use_position_ids_static_tensor=False,
        use_seq_ids_static_tensor=False,
        use_delta_prefix_lens_static_tensor=False,
        use_delta_position_ids_static_tensor=False,
        use_delta_seq_ids_static_tensor=False,
    )
    return CPChunkLayout(
        start=start,
        end=end,
        chunk_lengths=(
            list(chunk_lengths) if chunk_lengths is not None else [end - start]
        ),
        cache_positions=cache_positions_tensor,
        cache_seq_ids=cache_seq_ids_tensor,
        seq_len_delta=chunk_delta,
    )


def pad_rows_to_count(
    tensors: Sequence[torch.Tensor], target_rows: int
) -> list[torch.Tensor]:
    """Zero-pad tensors along dim 0 to exactly ``target_rows`` rows.

    Used to equalize per-rank row counts before equal-shape collectives (e.g.
    the CP allgather and the ETP reduce-scatter in the MoE CP-ETP dispatcher).
    Real rows always come first; trailing pad rows are all zeros. No-op for
    tensors that already have ``target_rows`` rows.

    Raises if a tensor has more rows than the target: silently feeding an
    oversized tensor to an equal-shape collective would desynchronize (hang)
    the whole group, so fail fast instead.
    """
    padded = []
    for tensor in tensors:
        rows = tensor.shape[0]
        if rows == target_rows:
            padded.append(tensor)
        elif rows < target_rows:
            pad = torch.zeros(
                target_rows - rows,
                *tensor.shape[1:],
                device=tensor.device,
                dtype=tensor.dtype,
            )
            padded.append(torch.cat([tensor, pad], dim=0))
        else:
            raise ValueError(
                f"Cannot pad tensor with {rows} rows down to target_rows={target_rows}"
            )
    return padded


def trim_rows(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    """Keep the first ``rows`` rows of dim 0 (no-op if already shorter)."""
    return tensor[:rows] if tensor.shape[0] > rows else tensor


class CPContext:
    """Shared Prefill Context Parallel (PCP) state and helpers.

    Layout-specific splitting, gathering and local-index mapping are provided
    by ``CPContextInterleaved``.
    """

    layout = None

    def __init__(self, pcp_size: int, cp_group: CommGroup):
        self.pcp_size = pcp_size
        self.cp_group = cp_group
        self.cp_rank = cp_group.rank_in_group
        self.is_active = True
        self.is_first_rank = cp_group.is_first_rank

        # Internal state for tracking split/allgather across stages.
        self._orig_num_tokens: int = 0
        self._n_local: int = 0
        # Equal-shape collectives still need this padded per-rank length at
        # communication boundaries; model/attention paths use _n_local rows.
        self._expected_n_local: int = 0
        self._step_active: bool = False

        # Per-step cached query view (set by prepare_local_lengths, read by layers)
        self._seq_len_delta_view: Optional[BatchedSeqLenDeltaView] = None
        self._interleaved_ctx: Optional["CPContextInterleaved"] = None
        self._layout_route_cache: Optional[
            tuple[tuple[int, int, str, str], tuple[LayoutRoute, LayoutRoute]]
        ] = None

    def set_step_active(self, active: bool) -> None:
        self._step_active = bool(active)
        self.clear_step_cache()

    @property
    def step_active(self) -> bool:
        return self._step_active

    def prepare_prefill(self, total_tokens: int) -> int:
        """Prepare CP metadata before MoE dispatch, without slicing tensors."""
        self.set_step_active(self._should_split_prefill(total_tokens))
        self._orig_num_tokens = total_tokens
        self._n_local = self.compute_pp_num_tokens(total_tokens)
        self._expected_n_local = (
            ceil_div(total_tokens, self.pcp_size) if self.step_active else total_tokens
        )
        return self._n_local

    def split_prefill(self, tokens, hiddens, freqs_cis, total_tokens):
        raise NotImplementedError

    def prepare_local_lengths(self, seq_len_delta, n_tokens, is_decode_stage):
        raise NotImplementedError

    def local_flat_indices(self, n_local, total_tokens, device):
        raise NotImplementedError

    def position_owner_and_local_index(self, positions, total_tokens):
        """Map global token positions to (owner rank, local row index)."""
        raise NotImplementedError

    def slice_for_local(self, tensor, n_local, total_tokens):
        raise NotImplementedError

    def compute_pp_num_tokens(self, num_tokens: int) -> int:
        raise NotImplementedError

    def get_seq_len_delta_view(self):
        if self.step_active:
            return self._seq_len_delta_view
        return None

    def as_interleaved(self) -> "CPContextInterleaved":
        """Return a cached interleaved view over the same CP group and token step."""
        if self.layout == "interleaved":
            return self
        ctx = self._interleaved_ctx
        if ctx is None:
            ctx = CPContextInterleaved(self.pcp_size, self.cp_group)
            self._interleaved_ctx = ctx
        ctx._step_active = self.step_active
        ctx._orig_num_tokens = self._orig_num_tokens
        ctx._n_local = ctx.compute_pp_num_tokens(self._orig_num_tokens)
        ctx._expected_n_local = (
            ceil_div(self._orig_num_tokens, self.pcp_size)
            if self.step_active
            else self._orig_num_tokens
        )
        return ctx

    def _allgather_global_payload(self, local_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _allgather_padded_rows(
        self, local_tensor: torch.Tensor, expected_n_local: int
    ) -> torch.Tensor:
        if local_tensor.shape[0] > expected_n_local:
            raise AssertionError(
                f"CP{self.cp_rank}: local rows {local_tensor.shape[0]} > "
                f"{expected_n_local}"
            )
        if local_tensor.shape[0] < expected_n_local:
            pad = torch.zeros(
                expected_n_local - local_tensor.shape[0],
                *local_tensor.shape[1:],
                device=local_tensor.device,
                dtype=local_tensor.dtype,
            )
            local_tensor = torch.cat([local_tensor, pad], dim=0)
        gathered = torch.empty(
            expected_n_local * self.pcp_size,
            *local_tensor.shape[1:],
            device=local_tensor.device,
            dtype=local_tensor.dtype,
        )
        self.cp_group.all_gather_into_tensor(gathered, local_tensor.contiguous())
        return gathered

    def allgather_hidden_states(self, h, output_token_offsets, post_layers_fn):
        """CP allgather hidden states, reorder, trim, select output, post_layers.

        Args:
            h: local hidden states [n_local, dim]
            output_token_offsets: indices of output token positions in global
                order, or None to return all gathered rows
            post_layers_fn: callable for post_layers (lm_head etc.), or None to
                return hidden states without projection

        Returns:
            Logits tensor [num_outputs, dim], or gathered hidden states when
            post_layers_fn is None.
        """
        if not self.step_active:
            if output_token_offsets is not None:
                h = h[output_token_offsets]
            if post_layers_fn is None:
                return h
            return post_layers_fn(h).float()

        if h.shape[0] != self._n_local:
            raise AssertionError(
                f"CP{self.cp_rank}: gather rows {h.shape[0]} != {self._n_local}"
            )

        h = self._allgather_global_payload(h)
        if output_token_offsets is not None:
            h = h[output_token_offsets]
        if post_layers_fn is None:
            return h
        return post_layers_fn(h).float()

    def allgather_kv(self, kv, indexer_k_local):
        """CP allgather KV + optional indexer K, reorder to global token order.

        Args:
            kv: local KV tensor [n_local, 1, kv_dim]
            indexer_k_local: local indexer K [n_local, index_head_dim] or None

        Returns:
            (kv_global, indexer_k_global):
              kv_global: [total_len] flat KV tensor (unsqueezed later by caller)
              indexer_k_global: [total_len, index_head_dim] or None
        """
        kv_flat = kv.squeeze(1)  # [n_local, kv_dim]

        if indexer_k_local is not None:
            payload = torch.cat([indexer_k_local, kv_flat], dim=-1)
            index_head_dim = indexer_k_local.shape[-1]
        else:
            payload = kv_flat
            index_head_dim = None

        global_payload = self._allgather_global_payload(payload)

        if indexer_k_local is not None:
            indexer_k_global = global_payload[:, :index_head_dim]
            kv_global = global_payload[:, index_head_dim:]
        else:
            indexer_k_global = None
            kv_global = global_payload

        return kv_global, indexer_k_global

    def barrier(self):
        """Barrier on CP group."""
        self.cp_group.barrier()

    def barrier_if_pp(self, pp_size: int):
        """CP+PP barrier: all CP ranks must synchronize before CP collectives."""
        if pp_size > 1:
            self.cp_group.barrier()

    def _should_split_prefill(self, delta_total: int = 0) -> bool:
        """Determine whether CP split should be applied for prefill.

        Args:
            delta_total: global number of tokens computed in this prefill step
        """
        return delta_total >= self.pcp_size

    def should_recv_directly(self, tp_size: int) -> bool:
        """In CP mode (or no TP), every rank receives from its PP pair directly."""
        return True  # pcp_size > 1 always true for CPContext

    def should_send_directly(self) -> bool:
        """In CP mode, every rank has its own PP pair and sends independently."""
        return True  # pcp_size > 1 always true for CPContext

    def clear_step_cache(self):
        """Clear per-step cached values."""
        self._seq_len_delta_view = None
        self._interleaved_ctx = None

    @property
    def n_local(self) -> int:
        """Get the local token count from the most recent split."""
        return self._n_local

    @property
    def expected_n_local(self) -> int:
        """Equal per-rank row count required by equal-shape collectives.

        Real per-rank row counts may differ after PCP removes token padding;
        collective boundaries pad every rank up to this count. Only meaningful
        while the CP context is active (``is_active``).
        """
        return self._expected_n_local

    @property
    def orig_num_tokens(self) -> int:
        """Get the original (global) token count from the most recent split."""
        return self._orig_num_tokens


class CPContextInterleaved(CPContext):
    """Interleaved CP parallelism state and operations.

    Token ``p`` belongs to rank ``p % pcp_size``. This is the original PCP
    layout; contiguous and zigzag layouts are provided by later subclasses.
    """

    layout = "interleaved"

    def split_prefill(
        self,
        tokens: Optional[torch.Tensor],
        hiddens: Optional[torch.Tensor],
        freqs_cis: Optional[BatchedFreqsCis],
        total_tokens: int,
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[BatchedFreqsCis]
    ]:
        """Split CP prefill inputs to this rank's real interleaved local rows.

        Stage 0 provides token IDs and splits them here. Later PP stages receive
        CP-local hidden states from the previous PP stage; when MTP is enabled,
        the last PP stage may also have local token IDs. In that case this method
        slices tokens and validates hiddens against the same real local rows.
        Collectives pad at their own boundary when equal per-rank shapes are
        required.
        """
        self.prepare_prefill(total_tokens)
        if not self.step_active:
            return tokens, hiddens, freqs_cis

        if tokens is None and hiddens is None:
            raise ValueError(f"CP{self.cp_rank}: missing prefill payload")
        device = tokens.device if tokens is not None else hiddens.device
        local_indices = build_cp_local_indices(
            self.cp_rank, self.pcp_size, self._orig_num_tokens, device=device
        )
        if hiddens is not None and hiddens.shape[0] != self._n_local:
            raise RuntimeError(
                f"CP{self.cp_rank}: hidden rows {hiddens.shape[0]} != {self._n_local}"
            )
        if tokens is not None:
            tokens = tokens[local_indices]
        if freqs_cis is not None:
            freqs_cis = BatchedFreqsCis(
                freqs_cis.cos[local_indices],
                freqs_cis.sin[local_indices],
            )
        return tokens, hiddens, freqs_cis

    def prepare_local_lengths(
        self,
        seq_len_delta: BatchedSeqLenDelta,
        n_tokens: int,
        is_decode_stage: bool,
    ):
        """Compute and cache the CP-local query delta view for this step.

        Called once per CP prefill step. The returned view represents the
        q-axis rows owned by this CP rank while preserving the global k-axis via
        ``seq_len_delta.new``.
        """
        if is_decode_stage:
            self._seq_len_delta_view = None
            return seq_len_delta
        else:
            self._seq_len_delta_view = self._build_local_delta_view(
                seq_len_delta, n_tokens
            )
            return self._seq_len_delta_view

    def _build_local_delta_view(
        self,
        seq_len_delta: BatchedSeqLenDelta,
        n_local: int,
    ) -> BatchedSeqLenDeltaView:
        total_tokens = seq_len_delta.delta_position_ids_tensor_device.shape[0]
        position_ids = seq_len_delta.delta_position_ids_tensor_device
        local_len_idx = build_cp_local_indices(
            self.cp_rank, self.pcp_size, total_tokens, position_ids.device
        )
        if local_len_idx.shape[0] != n_local:
            raise RuntimeError(
                "CP local delta length does not match real local tokens: "
                f"got {local_len_idx.shape[0]}, expected {n_local}"
            )
        return BatchedSeqLenDeltaView(seq_len_delta, indices=local_len_idx)

    def compute_pp_num_tokens(self, num_tokens: int) -> int:
        """Compute the number of tokens for PP hidden state transfer.

        In PCP mode, each CP rank only processes its real interleaved local tokens.
        """
        if not self._should_split_prefill(num_tokens):
            return num_tokens
        return (
            num_tokens - 1 - self.cp_rank
        ) // self.pcp_size + 1  # 即: pcp_size*k+cp_rank <= num_tokens-1

    def local_flat_indices(
        self, n_local: int, total_tokens: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build interleaved global indices and validity mask for this CP rank.

        Returns:
            (flat_indices, valid_mask) where flat_indices[i] is the global token
            index that local position i maps to, clamped to valid range.
            valid_mask[i] is True if the position is a real token (not padding).
        """
        local_offsets = torch.arange(n_local, device=device, dtype=torch.long)
        flat_indices = local_offsets * self.pcp_size + self.cp_rank
        valid_mask = flat_indices < total_tokens
        if total_tokens > 0:
            flat_indices = torch.clamp(flat_indices, max=total_tokens - 1)
        return flat_indices, valid_mask

    def position_owner_and_local_index(self, positions, total_tokens):
        """Map global positions to interleaved owner rank and local row."""
        owner = torch.remainder(positions, self.pcp_size)
        local = torch.div(positions, self.pcp_size, rounding_mode="floor")
        return owner.long(), local.long()

    def slice_for_local(
        self, tensor: torch.Tensor, n_local: int, total_tokens: int
    ) -> torch.Tensor:
        """Slice a global tensor to this CP rank's local tokens (interleaved)."""
        flat_indices, _ = self.local_flat_indices(n_local, total_tokens, tensor.device)
        return tensor[flat_indices]

    def _allgather_global_payload(self, local_tensor: torch.Tensor) -> torch.Tensor:
        """Allgather a CP-local payload using the current split_prefill state."""
        n_local = self._n_local
        expected_n_local = self._expected_n_local
        if local_tensor.shape[0] != n_local:
            raise AssertionError(
                f"CP{self.cp_rank}: local rows {local_tensor.shape[0]} != {n_local}"
            )
        if n_local > expected_n_local:
            raise AssertionError(
                f"CP{self.cp_rank}: local rows {n_local} > {expected_n_local}"
            )
        if self._orig_num_tokens <= 0 or expected_n_local != ceil_div(
            self._orig_num_tokens, self.pcp_size
        ):
            raise AssertionError(f"CP{self.cp_rank}: invalid allgather state")

        gathered = self._allgather_padded_rows(local_tensor, expected_n_local)
        reorder_idx = build_cp_reorder_idx(
            self.pcp_size, expected_n_local, device=local_tensor.device
        )
        return gathered[reorder_idx][: self._orig_num_tokens]


class CPContextContiguous(CPContext):
    """Contiguous CP parallelism state and operations.

    Rank ``r`` owns the balanced interval
    ``[r * T // W, (r + 1) * T // W)`` of the global delta-token axis. Unlike
    ``CPContextInterleaved``, the residual-stream layout itself is contiguous,
    so layers do not need an interleaved-to-contiguous all-to-all around each
    local computation.
    """

    layout = "contiguous"

    def __init__(self, pcp_size: int, cp_group: CommGroup):
        super().__init__(pcp_size, cp_group)
        self._mla_chunk: Optional[CPChunkLayout] = None
        self._mla_chunk_source: Optional[BatchedSeqLenDelta] = None
        self._mla_zigzag_source = None
        self._mla_zigzag_chunks = None
        self._zigzag_indices_key = None
        self._zigzag_indices = None

    def clear_step_cache(self):
        super().clear_step_cache()
        self._mla_chunk = None
        self._mla_chunk_source = None
        self._mla_zigzag_source = None
        self._mla_zigzag_chunks = None

    def prepare_mla_chunk(self, seq_len_delta: BatchedSeqLenDelta) -> CPChunkLayout:
        """Build this step's contiguous MLA chunk metadata."""
        if not self.step_active:
            raise RuntimeError("CP MLA chunk metadata requires an active CP step")

        total_tokens = int(seq_len_delta.delta_total_len)
        if total_tokens != self._orig_num_tokens:
            raise RuntimeError(
                "CP MLA chunk total does not match split_prefill: "
                f"got {total_tokens}, expected {self._orig_num_tokens}"
            )
        if self._mla_chunk is not None and self._mla_chunk_source is seq_len_delta:
            return self._mla_chunk

        start, end = contiguous_interval(self.cp_rank, self.pcp_size, total_tokens)
        layout = _build_mla_interval_layout(
            seq_len_delta,
            start,
            end,
            chunk_lengths=build_contiguous_chunk_lengths(total_tokens, self.pcp_size),
        )
        self._mla_chunk = layout
        self._mla_chunk_source = seq_len_delta
        return layout

    def allgather_ordered_payload(self, local_tensor: torch.Tensor) -> torch.Tensor:
        """Allgather a contiguous chunk into global token order."""
        layout = self._mla_chunk
        if layout is None:
            raise RuntimeError("CP MLA chunk metadata has not been prepared")
        if local_tensor.shape[0] != layout.local_rows:
            raise AssertionError(
                f"CP{self.cp_rank}: chunk rows {local_tensor.shape[0]} != "
                f"{layout.local_rows}"
            )

        expected_n_local = max(layout.chunk_lengths)
        gathered = self._allgather_padded_rows(local_tensor, expected_n_local)
        reorder_idx = build_cp_contiguous_reorder_idx(
            layout.chunk_lengths, expected_n_local, device=local_tensor.device
        )
        return gathered[reorder_idx]

    def _zigzag_indices_for(self, total_tokens: int):
        key = (self.pcp_size, int(total_tokens))
        if self._zigzag_indices_key != key:
            self._zigzag_indices = [
                zigzag_rank_indices(r, self.pcp_size, int(total_tokens))
                for r in range(self.pcp_size)
            ]
            self._zigzag_indices_key = key
        return self._zigzag_indices

    def prepare_mla_zigzag_chunks(
        self, seq_len_delta: BatchedSeqLenDelta
    ) -> list[CPChunkLayout]:
        if not self.step_active:
            raise RuntimeError("CP MLA zigzag metadata requires an active CP step")
        total_tokens = int(seq_len_delta.delta_total_len)
        if total_tokens != self._orig_num_tokens:
            raise RuntimeError(
                "CP MLA zigzag total does not match split_prefill: "
                f"got {total_tokens}, expected {self._orig_num_tokens}"
            )
        if self._mla_zigzag_source is seq_len_delta:
            assert self._mla_zigzag_chunks is not None
            return self._mla_zigzag_chunks

        n_blocks = 2 * self.pcp_size
        intervals = [
            contiguous_interval(self.cp_rank, n_blocks, total_tokens),
            contiguous_interval(n_blocks - 1 - self.cp_rank, n_blocks, total_tokens),
        ]
        chunks = [
            _build_mla_interval_layout(seq_len_delta, start, end)
            for start, end in intervals
        ]
        self._mla_zigzag_source = seq_len_delta
        self._mla_zigzag_chunks = chunks
        return chunks

    def allgather_zigzag_payload(self, local_tensor: torch.Tensor) -> torch.Tensor:
        indices = self._zigzag_indices_for(self._orig_num_tokens)
        local_len = len(indices[self.cp_rank])
        if local_tensor.shape[0] != local_len:
            raise AssertionError(
                f"CP{self.cp_rank}: zigzag rows {local_tensor.shape[0]} != "
                f"{local_len}"
            )
        expected = max(len(x) for x in indices)
        gathered = self._allgather_padded_rows(local_tensor, expected)
        slots = torch.full(
            (self._orig_num_tokens,), -1, device=local_tensor.device, dtype=torch.long
        )
        for r, idx in enumerate(indices):
            slots[idx.to(local_tensor.device)] = r * expected + torch.arange(
                len(idx), device=local_tensor.device, dtype=torch.long
            )
        if bool((slots < 0).any()):
            raise RuntimeError("zigzag allgather did not cover every global token")
        return gathered[slots]

    def split_prefill(
        self,
        tokens: Optional[torch.Tensor],
        hiddens: Optional[torch.Tensor],
        freqs_cis: Optional[BatchedFreqsCis],
        total_tokens: int,
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[BatchedFreqsCis]
    ]:
        """Split CP prefill inputs to this rank's contiguous local rows."""
        self.prepare_prefill(total_tokens)
        if not self.step_active:
            return tokens, hiddens, freqs_cis

        if tokens is None and hiddens is None:
            raise ValueError(f"CP{self.cp_rank}: missing prefill payload")
        device = tokens.device if tokens is not None else hiddens.device
        start, end = contiguous_interval(
            self.cp_rank, self.pcp_size, self._orig_num_tokens
        )
        local_indices = torch.arange(start, end, device=device, dtype=torch.long)
        if hiddens is not None and hiddens.shape[0] != self._n_local:
            raise RuntimeError(
                f"CP{self.cp_rank}: hidden rows {hiddens.shape[0]} != {self._n_local}"
            )
        if tokens is not None:
            tokens = tokens[local_indices]
        if freqs_cis is not None:
            freqs_cis = BatchedFreqsCis(
                freqs_cis.cos[local_indices],
                freqs_cis.sin[local_indices],
            )
        return tokens, hiddens, freqs_cis

    def prepare_local_lengths(
        self,
        seq_len_delta: BatchedSeqLenDelta,
        n_tokens: int,
        is_decode_stage: bool,
    ):
        """Compute and cache the CP-local query delta view for this step."""
        if is_decode_stage:
            self._seq_len_delta_view = None
            return seq_len_delta
        else:
            self._seq_len_delta_view = self._build_local_delta_view(
                seq_len_delta, n_tokens
            )
            return self._seq_len_delta_view

    def _build_local_delta_view(
        self,
        seq_len_delta: BatchedSeqLenDelta,
        n_local: int,
    ) -> BatchedSeqLenDeltaView:
        total_tokens = seq_len_delta.delta_position_ids_tensor_device.shape[0]
        start, end = contiguous_interval(self.cp_rank, self.pcp_size, total_tokens)
        if end - start != n_local:
            raise RuntimeError(
                "CP local delta length does not match real local tokens: "
                f"got {end - start}, expected {n_local}"
            )
        return BatchedSeqLenDeltaView(seq_len_delta, start, end)

    def compute_pp_num_tokens(self, num_tokens: int) -> int:
        """Compute the number of tokens for PP hidden state transfer."""
        if not self._should_split_prefill(num_tokens):
            return num_tokens
        start, end = contiguous_interval(self.cp_rank, self.pcp_size, num_tokens)
        return end - start

    def local_flat_indices(
        self, n_local: int, total_tokens: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build contiguous global indices and validity mask for this rank."""
        start, end = contiguous_interval(self.cp_rank, self.pcp_size, total_tokens)
        flat_indices = torch.arange(
            start, start + n_local, device=device, dtype=torch.long
        )
        valid_mask = flat_indices < end
        if total_tokens > 0:
            flat_indices = torch.clamp(flat_indices, max=total_tokens - 1)
        return flat_indices, valid_mask

    def position_owner_and_local_index(self, positions, total_tokens):
        """Map global positions to contiguous owner rank and local row."""
        total_tokens = int(total_tokens)
        if total_tokens <= 0:
            return torch.zeros_like(positions), positions.clone()
        owner = ((positions + 1) * self.pcp_size - 1) // total_tokens
        start = owner * total_tokens // self.pcp_size
        return owner.long(), (positions - start).long()

    def slice_for_local(
        self, tensor: torch.Tensor, n_local: int, total_tokens: int
    ) -> torch.Tensor:
        """Slice a global tensor to this CP rank's local tokens (contiguous)."""
        start, _ = contiguous_interval(self.cp_rank, self.pcp_size, total_tokens)
        return tensor[start : start + n_local]

    def _allgather_global_payload(self, local_tensor: torch.Tensor) -> torch.Tensor:
        """Allgather a contiguous CP-local payload into global token order."""
        if local_tensor.shape[0] != self._n_local:
            raise AssertionError(
                f"CP{self.cp_rank}: local rows {local_tensor.shape[0]} != "
                f"{self._n_local}"
            )
        chunk_lengths = build_contiguous_chunk_lengths(
            self._orig_num_tokens, self.pcp_size
        )
        expected_n_local = max(chunk_lengths)
        gathered = self._allgather_padded_rows(local_tensor, expected_n_local)
        reorder_idx = build_cp_contiguous_reorder_idx(
            chunk_lengths, expected_n_local, device=local_tensor.device
        )
        return gathered[reorder_idx]


class NoOpCPContext:
    """No-op CP context for pcp_size == 1.

    All methods return inputs unchanged, making non-CP path zero-overhead.
    """

    pcp_size: int = 1
    cp_rank: int = 0
    is_active: bool = False
    is_first_rank: bool = True
    cp_group: None = None
    step_active: bool = False

    def set_step_active(self, active: bool):
        self.step_active = False

    def prepare_prefill(self, total_tokens: int) -> int:
        return total_tokens

    def split_prefill(self, tokens, hiddens, freqs_cis, total_tokens):
        return tokens, hiddens, freqs_cis

    def allgather_hidden_states(
        self,
        h,
        output_token_offsets,
        post_layers_fn,
    ):
        if output_token_offsets is not None:
            h = h[output_token_offsets]
        if post_layers_fn is None:
            return h
        h = post_layers_fn(h)
        return h.float()

    def allgather_kv(self, kv, indexer_k_local):
        return kv.squeeze(1), indexer_k_local

    def prepare_local_lengths(self, seq_len_delta, n_tokens, is_decode_stage):
        return seq_len_delta

    def get_seq_len_delta_view(self):
        return None

    def clear_step_cache(self):
        pass

    def barrier(self):
        pass

    def barrier_if_pp(self, pp_size):
        pass

    def _should_split_prefill(self, delta_total: int = 0) -> bool:
        return False

    def compute_pp_num_tokens(self, num_tokens):
        return num_tokens

    def should_recv_directly(self, tp_size):
        return tp_size <= 1

    def should_send_directly(self):
        return False

    def local_flat_indices(self, n_local, total_tokens, device):
        indices = torch.arange(n_local, device=device, dtype=torch.long)
        mask = torch.ones(n_local, device=device, dtype=torch.bool)
        return indices, mask

    def position_owner_and_local_index(self, positions, total_tokens):
        return torch.zeros_like(positions), positions

    def slice_for_local(self, tensor, n_local, total_tokens):
        return tensor

    @property
    def n_local(self) -> int:
        return 0

    @property
    def expected_n_local(self) -> int:
        """Undefined when CP is inactive; callers must check ``is_active`` first."""
        return 0

    @property
    def orig_num_tokens(self) -> int:
        return 0
