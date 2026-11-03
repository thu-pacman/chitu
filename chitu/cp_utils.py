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

import torch
from typing import Optional, Sequence, Tuple

from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.batched_seq_len import BatchedSeqLenDelta, BatchedSeqLenDeltaView
from chitu.distributed.comm_group import CommGroup
from chitu.utils import ceil_div

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
        freqs_cis: BatchedFreqsCis,
        total_tokens: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], BatchedFreqsCis]:
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

    def split_prefill(
        self,
        tokens: Optional[torch.Tensor],
        hiddens: Optional[torch.Tensor],
        freqs_cis: BatchedFreqsCis,
        total_tokens: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], BatchedFreqsCis]:
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
