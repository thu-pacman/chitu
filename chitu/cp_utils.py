# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
CP (Context Parallelism) utilities.

Provides CPContext / NoOpCPContext that encapsulate all CP state and operations.
Access via get_cp_context() singleton. When pcp_size == 1, returns NoOpCPContext
(identity/no-op). When pcp_size > 1, returns CPContext with full CP operations.
"""

from __future__ import annotations

import torch
from typing import Optional, Tuple

from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.distributed.comm_group import CommGroup

# ---- Global singleton ----

_CP_CONTEXT_INSTANCE: Optional[CPContext | NoOpCPContext] = None


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
            _CP_CONTEXT_INSTANCE = CPContext(pcp_size, get_pcp_group())
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


class CPContext:
    """Encapsulates CP parallelism state and operations.

    When pcp_size > 1, provides full CP split/gather/allgather operations.
    When pcp_size == 1, use NoOpCPContext instead (all methods are identity).
    """

    def __init__(self, pcp_size: int, cp_group: CommGroup):
        self.pcp_size = pcp_size
        self.cp_group = cp_group
        self.cp_rank = cp_group.rank_in_group
        self.is_active = True
        self.is_first_rank = cp_group.is_first_rank

        # Internal state for tracking split/gather across stages
        self._orig_num_tokens: int = 0
        self._n_local: int = 0
        self._step_active: bool = False

        # Per-step cached values (set by prepare_local_lengths, read by layers)
        self._local_lengths: Optional[torch.Tensor] = None
        self._local_seq_ids: Optional[torch.Tensor] = None

    def set_step_active(self, active: bool) -> None:
        self._step_active = bool(active)
        self.clear_step_cache()

    @property
    def step_active(self) -> bool:
        return self._step_active

    def split_stage0(
        self, tokens: torch.Tensor, freqs_cis: BatchedFreqsCis
    ) -> Tuple[torch.Tensor, BatchedFreqsCis]:
        """CP split on PP stage 0 or no-PP: interleave token IDs across CP ranks.

        Pads to ceil(num_tokens/pcp_size) so all ranks have equal local length,
        which is required by CP allgather collectives.
        """
        self.set_step_active(True)
        local_indices = torch.arange(
            self.cp_rank, tokens.shape[0], self.pcp_size, device=tokens.device
        )
        self._orig_num_tokens = tokens.shape[0]
        self._n_local = (self._orig_num_tokens + self.pcp_size - 1) // self.pcp_size
        n_local = local_indices.shape[0]
        if n_local < self._n_local:
            local_indices = torch.cat(
                [
                    local_indices,
                    local_indices[-1].repeat(self._n_local - n_local),
                ]
            )
        tokens = tokens[local_indices]
        freqs_cis = BatchedFreqsCis(
            freqs_cis.cos[local_indices],
            freqs_cis.sin[local_indices],
        )
        return tokens, freqs_cis

    def split_stage1(
        self, h: torch.Tensor, freqs_cis: BatchedFreqsCis
    ) -> BatchedFreqsCis:
        """CP split on PP stage 1+: hidden states already local from PP transfer.

        Only splits freqs_cis; pads freqs_cis if stage 0 padded tokens.
        """
        self.set_step_active(True)
        n_local = h.shape[0]
        bs_seq_full = n_local * self.pcp_size
        orig_len = freqs_cis.cos.shape[0]
        if bs_seq_full > orig_len:
            pad_len = bs_seq_full - orig_len
            pad_cos = freqs_cis.cos[-1:].repeat(pad_len, 1)
            pad_sin = freqs_cis.sin[-1:].repeat(pad_len, 1)
            freqs_cis = BatchedFreqsCis(
                torch.cat([freqs_cis.cos, pad_cos]),
                torch.cat([freqs_cis.sin, pad_sin]),
            )
        local_indices = torch.arange(
            self.cp_rank, bs_seq_full, self.pcp_size, device=h.device
        )
        self._orig_num_tokens = orig_len
        self._n_local = n_local
        return BatchedFreqsCis(
            freqs_cis.cos[local_indices],
            freqs_cis.sin[local_indices],
        )

    def split_prefill(
        self,
        tokens: Optional[torch.Tensor],
        freqs_cis: BatchedFreqsCis,
        hiddens: Optional[torch.Tensor],
        pp_stage: int,
        delta_total: int,
    ) -> Tuple[Optional[torch.Tensor], BatchedFreqsCis]:
        """Split stage-0 token IDs or later-stage freqs_cis for CP prefill."""
        if not self.should_split_prefill(delta_total):
            self.set_step_active(False)
            return tokens, freqs_cis
        if pp_stage == 0:
            tokens, freqs_cis = self.split_stage0(tokens, freqs_cis)
        else:
            assert hiddens is not None, "hiddens must be provided at PP stage > 0"
            freqs_cis = self.split_stage1(hiddens, freqs_cis)
        return tokens, freqs_cis

    def gather(
        self,
        h: torch.Tensor,
        output_token_offsets: torch.Tensor,
        post_layers_fn,
        cp_active: Optional[bool] = None,
        pp_size: int = 1,
        pp_stage: int = 0,
        seq_len_delta: Optional[BatchedSeqLenDelta] = None,
    ) -> torch.Tensor:
        """CP allgather hidden states, reorder, trim, select output, post_layers.

        Args:
            h: local hidden states [n_local, dim]
            output_token_offsets: indices of output token positions in global order
            post_layers_fn: callable for post_layers (lm_head etc.)
            cp_active: whether CP split was active on this stage
            pp_size: pipeline parallel size
            pp_stage: current pipeline stage
            seq_len_delta: for delta_total_len at PP stage 1+

        Returns:
            Logits tensor [num_outputs, dim]
        """
        need_gather = self.step_active if cp_active is None else cp_active
        # Later PP stages decide by global delta, not local hidden length.
        if not need_gather and pp_size > 1 and pp_stage != 0:
            if seq_len_delta is not None:
                delta_total = seq_len_delta.delta_total_len
                need_gather = delta_total >= self.pcp_size
        if not need_gather:
            h = h[output_token_offsets]
            h = post_layers_fn(h)
            return h.float()

        orig = self._orig_num_tokens
        if orig <= 0 and pp_size > 1:
            if seq_len_delta is not None:
                orig = seq_len_delta.delta_total_len

        h = self.allgather_interleaved(
            h, orig if orig > 0 else h.shape[0] * self.pcp_size
        )
        h = h[output_token_offsets]
        h = post_layers_fn(h)
        return h.float()

    def allgather_kv(
        self,
        n_local: int,
        kv: torch.Tensor,
        indexer_k_local: Optional[torch.Tensor],
        index_head_dim: Optional[int],
        seq_len_delta: BatchedSeqLenDelta,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """CP allgather KV + optional indexer K, reorder to global token order.

        Args:
            n_local: local token count (padded to ceil(total/pcp_size))
            kv: local KV tensor [n_local, 1, kv_dim]
            indexer_k_local: local indexer K [n_local, index_head_dim] or None
            index_head_dim: indexer head dimension
            seq_len_delta: BatchedSeqLenDelta with delta_total_len

        Returns:
            (kv_global, indexer_k_global):
              kv_global: [total_len] flat KV tensor (unsqueezed later by caller)
              indexer_k_global: [total_len, index_head_dim] or None
        """
        expected_n_local = n_local
        bs_seq_global = expected_n_local * self.pcp_size
        kv_flat = kv.squeeze(1)  # [n_local, kv_dim]

        if indexer_k_local is not None:
            allgather_payload = torch.cat([indexer_k_local, kv_flat], dim=-1)
            payload_dim = allgather_payload.shape[-1]
        else:
            allgather_payload = kv_flat
            payload_dim = kv_flat.shape[-1]

        if n_local < expected_n_local:
            pad = torch.zeros(
                expected_n_local - n_local,
                payload_dim,
                device=kv.device,
                dtype=kv.dtype,
            )
            allgather_payload = torch.cat([allgather_payload, pad], dim=0)

        global_payload = torch.empty(
            bs_seq_global, payload_dim, device=kv.device, dtype=kv.dtype
        )
        self.cp_group.all_gather_into_tensor(
            global_payload, allgather_payload.contiguous()
        )

        reorder_idx = build_cp_reorder_idx(
            self.pcp_size, expected_n_local, device=global_payload.device
        )
        global_payload = global_payload[reorder_idx]
        delta_len = seq_len_delta.delta_total_len
        if global_payload.shape[0] > delta_len:
            global_payload = global_payload[:delta_len]

        if indexer_k_local is not None:
            indexer_k_global = global_payload[:, :index_head_dim]
            kv_global = global_payload[:, index_head_dim:]
        else:
            indexer_k_global = None
            kv_global = global_payload

        return kv_global, indexer_k_global

    def build_local_lengths(
        self,
        seq_len_delta: BatchedSeqLenDelta,
        n_local: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute per-query local_lengths and local_seq_ids for CP sparse attn.

        Returns (local_lengths, local_seq_ids):
          local_lengths: per-query causal upper bounds (padded to n_local)
          local_seq_ids: per-query sequence IDs for page table isolation, or None
        """
        total_tokens = seq_len_delta.delta_position_ids_tensor_device.shape[0]
        position_ids = seq_len_delta.delta_position_ids_tensor_device
        local_len_idx = build_cp_local_indices(
            self.cp_rank, self.pcp_size, total_tokens, position_ids.device
        )
        local_lengths = (
            torch.index_select(position_ids, 0, local_len_idx) + 1
        ).contiguous()
        local_seq_ids = None
        if seq_len_delta.batch_size > 1:
            local_seq_ids = torch.index_select(
                seq_len_delta.delta_seq_ids_tensor_device, 0, local_len_idx
            )
        if local_lengths.shape[0] < n_local:
            pad_len = n_local - local_lengths.shape[0]
            pad = torch.full(
                (pad_len,),
                seq_len_delta.delta_total_len,
                dtype=local_lengths.dtype,
                device=local_lengths.device,
            )
            local_lengths = torch.cat([local_lengths, pad])
            # Also pad local_seq_ids so its length matches n_local.
            # Use seq_id=0 for padding tokens; they will be assigned to
            # batch 0 in the batch-grouped sort, which is harmless since
            # padding tokens produce zero-contribution attention output.
            if local_seq_ids is not None:
                seq_pad = torch.zeros(
                    pad_len, dtype=local_seq_ids.dtype, device=local_seq_ids.device
                )
                local_seq_ids = torch.cat([local_seq_ids, seq_pad])
        return local_lengths, local_seq_ids

    def prepare_local_lengths(
        self,
        seq_len_delta: BatchedSeqLenDelta,
        n_tokens: int,
        is_decode_stage: bool,
    ):
        """Compute and cache local_lengths/local_seq_ids for this step.

        Called once per attention forward, reused by subsequent layers and
        attn_backend/indexer via the local_lengths/local_seq_ids properties.
        """
        if is_decode_stage:
            self._local_lengths = (
                seq_len_delta.delta_position_ids_tensor_device + 1
            ).contiguous()
            self._local_seq_ids = None
        else:
            self._local_lengths, self._local_seq_ids = self.build_local_lengths(
                seq_len_delta, n_tokens
            )

    @property
    def local_lengths(self) -> Optional[torch.Tensor]:
        """Cached local_lengths for the current step (None if pcp_size==1 or not prepared)."""
        return self._local_lengths

    @property
    def local_seq_ids(self) -> Optional[torch.Tensor]:
        """Cached local_seq_ids for the current step (None if pcp_size==1 or not prepared)."""
        return self._local_seq_ids

    def clear_step_cache(self):
        """Clear per-step cached values."""
        self._local_lengths = None
        self._local_seq_ids = None

    def barrier(self):
        """Barrier on CP group."""
        self.cp_group.barrier()

    def barrier_if_pp(self, pp_size: int):
        """CP+PP barrier: all CP ranks must synchronize before CP collectives."""
        if pp_size > 1:
            self.cp_group.barrier()

    def should_split_prefill(
        self,
        delta_total: int = 0,
    ) -> bool:
        """Determine whether CP split should be applied for prefill.

        Args:
            delta_total: global number of tokens computed in this prefill step
        """
        return delta_total >= self.pcp_size

    def compute_pp_num_tokens(self, num_tokens: int) -> int:
        """Compute the number of tokens for PP hidden state transfer.

        In PCP mode, each CP rank only processes ceil(num_tokens/pcp_size) local tokens.
        """
        if num_tokens >= self.pcp_size:
            return (num_tokens + self.pcp_size - 1) // self.pcp_size
        return num_tokens

    def should_recv_directly(self, tp_size: int) -> bool:
        """In CP mode (or no TP), every rank receives from its PP pair directly."""
        return True  # pcp_size > 1 always true for CPContext

    def should_send_directly(self) -> bool:
        """In CP mode, every rank has its own PP pair and sends independently."""
        return True  # pcp_size > 1 always true for CPContext

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

    def slice_for_local(
        self, tensor: torch.Tensor, n_local: int, total_tokens: int
    ) -> torch.Tensor:
        """Slice a global tensor to this CP rank's local tokens (interleaved)."""
        flat_indices, _ = self.local_flat_indices(n_local, total_tokens, tensor.device)
        return tensor[flat_indices]

    def allgather_interleaved(
        self, local_tensor: torch.Tensor, total_tokens: int
    ) -> torch.Tensor:
        """Allgather CP-local tensor, reorder to global token order, trim padding.

        Unlike ``gather``, this does *not* apply output_token_offsets or
        post_layers_fn — it returns the reordered hidden states directly.
        """
        n_local = local_tensor.shape[0]
        gathered = torch.empty(
            n_local * self.pcp_size,
            *local_tensor.shape[1:],
            device=local_tensor.device,
            dtype=local_tensor.dtype,
        )
        self.cp_group.all_gather_into_tensor(gathered, local_tensor.contiguous())
        reorder_idx = build_cp_reorder_idx(
            self.pcp_size, n_local, device=local_tensor.device
        )
        return gathered[reorder_idx][:total_tokens]

    @property
    def n_local(self) -> int:
        """Get the local token count from the most recent split."""
        return self._n_local

    @property
    def orig_num_tokens(self) -> int:
        """Get the original (global) token count from the most recent split."""
        return self._orig_num_tokens


class NoOpCPContext:
    """No-op CP context for pcp_size == 1.

    All methods return inputs unchanged, making non-CP path zero-overhead.
    """

    pcp_size: int = 1
    cp_rank: int = 0
    is_active: bool = False
    is_first_rank: bool = True
    cp_group: None = None
    local_lengths = None
    local_seq_ids = None
    step_active: bool = False

    def set_step_active(self, active: bool):
        self.step_active = False

    def split_stage0(self, tokens, freqs_cis):
        return tokens, freqs_cis

    def split_stage1(self, h, freqs_cis):
        return freqs_cis

    def split_prefill(self, tokens, freqs_cis, hiddens, pp_stage, delta_total):
        return tokens, freqs_cis

    def gather(
        self,
        h,
        output_token_offsets,
        post_layers_fn,
        cp_active=None,
        pp_size=1,
        pp_stage=0,
        seq_len_delta=None,
    ):
        h = h[output_token_offsets]
        h = post_layers_fn(h)
        return h.float()

    def allgather_kv(self, n_local, kv, indexer_k_local, index_head_dim, seq_len_delta):
        return kv.squeeze(1), indexer_k_local

    def build_local_lengths(self, seq_len_delta, n_local):
        return None, None

    def prepare_local_lengths(self, seq_len_delta, n_tokens, is_decode_stage):
        pass

    def clear_step_cache(self):
        pass

    def barrier(self):
        pass

    def barrier_if_pp(self, pp_size):
        pass

    def should_split_prefill(self, delta_total: int = 0) -> bool:
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

    def slice_for_local(self, tensor, n_local, total_tokens):
        return tensor

    def allgather_interleaved(self, local_tensor, total_tokens):
        return local_tensor

    @property
    def n_local(self) -> int:
        return 0

    @property
    def orig_num_tokens(self) -> int:
        return 0
