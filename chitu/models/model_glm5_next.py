# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3-Flash text model support."""

from collections import OrderedDict
from typing import Any, Optional
from typing_extensions import override

import torch
import torch.nn.functional as F
from torch import nn

from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.checkpoint_prefix import as_checkpoint_prefix
from chitu.cp_utils import get_cp_context
from chitu.global_vars import get_global_args
from chitu.kv_cache import DenseKVCacheAccessor, KVCacheBase, PagedKVCacheAccessor
from chitu.models.model import (
    RMSNorm,
    RMSNormResidual,
    TransformerBlock,
    get_linear_layout_contig_y,
)
from chitu.models.model_deepseek_v3 import (
    TransformerDeepSeekV3,
)
from chitu.models.model_deepseek_v4 import mHCSubLayer
from chitu.models.model_glm52 import (
    AttentionGLM52,
    IndexerGLM52,
    TransformerBlockGLM52MTP,
    _IndexerBuffer,
    make_mlp_for_layer,
)
from chitu.models.registry import ModelType, register_model
from chitu.ops import (
    append_to_dense_kv_cache,
    append_to_paged_kv_cache,
    blockfp8_weight_dequant,
    causal_conv1d_prefill,
    causal_conv1d_update,
    chunk_kimi_delta_attention,
    read_from_dense_kv_cache,
    read_from_paged_kv_cache,
    read_from_singleton_paged_kv_cache,
    recurrent_kimi_delta_attention,
    rms_norm_gate,
    silu_and_mul,
    soft_fp8_blockfp8_weight_dequant,
    update_singleton_paged_kv_cache,
)
from chitu.quantization import QuantizationRegistry
from chitu.task_type import TaskType
from chitu.tensor_parallel import ColumnParallelLinear, LocalLinear, RowParallelLinear
from chitu.distributed.parallel_state import get_tp_size


def get_mtp_accept_indices():
    from chitu.backend import Backend

    return Backend.model.mtp_accept_indices.get()


class IndexerGLM5Next(IndexerGLM52):
    """K-pool DSA indexer used by GLM-5.3-Flash sparse layers."""

    def __init__(
        self,
        args,
        *,
        checkpoint_prefix,
        indexer_impl,
        buffer_mode=None,
        indexer_buffer=None,
    ):
        super().__init__(
            args,
            checkpoint_prefix=checkpoint_prefix,
            indexer_impl=indexer_impl,
            buffer_mode=buffer_mode,
            indexer_buffer=indexer_buffer,
        )
        checkpoint_prefix = as_checkpoint_prefix(checkpoint_prefix)
        self.index_kpool = int(getattr(args, "index_kpool", 16))
        self.index_kpool_always_select_tail = bool(
            getattr(args, "index_kpool_always_select_tail", True)
        )
        self.index_kpool_compress_ape = nn.Parameter(
            torch.empty(self.index_kpool, self.head_dim)
        )
        self.index_kpool_compress_gate = LocalLinear(
            self.dim,
            self.head_dim,
            has_bias=False,
            checkpoint_prefix=checkpoint_prefix / "index_kpool_compress_gate",
        )

    @override
    def must_materialize_topk_indices(self) -> bool:
        return True

    def _append_packed_states(self, packed_states, seq_len_delta, cache_accessor):
        assert packed_states.shape[0] == seq_len_delta.delta_total_len, (
            f"packed states ({packed_states.shape[0]}) must cover exactly the "
            f"delta rows ({seq_len_delta.delta_total_len})"
        )
        if isinstance(cache_accessor, PagedKVCacheAccessor):
            append_to_paged_kv_cache(
                cache_accessor.kv["indexer_packed"],
                cache_accessor.block_table,
                packed_states.contiguous(),
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                get_page_ids=cache_accessor.get_page_ids,
                get_offs_in_page=cache_accessor.get_offs_in_page,
                use_i64_offsets=cache_accessor.use_i64_offsets,
            )
            return
        if isinstance(cache_accessor, DenseKVCacheAccessor):
            append_to_dense_kv_cache(
                cache_accessor.kv["indexer_packed"],
                packed_states.contiguous(),
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                use_i64_offsets=cache_accessor.use_i64_offsets,
            )
            return
        raise TypeError(f"Unsupported indexer cache accessor {type(cache_accessor)!r}")

    def _read_full_packed_states(self, seq_len_delta, cache_accessor):
        positions = seq_len_delta.new.position_ids_tensor_device
        seq_ids = seq_len_delta.new.seq_ids_tensor_device
        if isinstance(cache_accessor, PagedKVCacheAccessor):
            return read_from_paged_kv_cache(
                cache_accessor.kv["indexer_packed"],
                cache_accessor.block_table,
                positions,
                seq_ids,
                use_i64_offsets=cache_accessor.use_i64_offsets,
            )
        if isinstance(cache_accessor, DenseKVCacheAccessor):
            return read_from_dense_kv_cache(
                cache_accessor.kv["indexer_packed"], positions, seq_ids
            )
        raise TypeError(f"Unsupported indexer cache accessor {type(cache_accessor)!r}")

    # Upper bound of the float32 elements used by the per-head pool scores of
    # one query tile, i.e. `(rows, n_heads, num_pools)`. Only limits how many
    # queries are scored at once, so prefilling a long sequence stays in memory.
    _SCORE_BUDGET_ELEMS = 1 << 24

    def _pool_sequence_states(self, packed_seq):
        """Compress the packed states of one sequence into complete k-pools.

        Pool `p` holds the tokens `[p * index_kpool, (p + 1) * index_kpool)` of
        the sequence. A pool does not depend on the query, so all the queries
        of a sequence share the same pools.
        """
        num_pools = packed_seq.shape[0] // self.index_kpool
        if num_pools == 0:
            return packed_seq.new_zeros((0, self.head_dim))
        pooled = packed_seq[: num_pools * self.index_kpool].view(
            num_pools, self.index_kpool, -1
        )
        keys = pooled[..., : self.head_dim]
        gates = pooled[..., self.head_dim : self.head_dim * 2]
        logits = gates.float() + self.index_kpool_compress_ape.float().unsqueeze(0)
        probs = logits.softmax(dim=1).to(keys.dtype)
        return (probs * keys).sum(dim=1)

    def _score_pools_topk(self, q, weights, query_pos, pool_keys, out):
        """Select the k-pools of the queries of one sequence, filling `out`.

        `q`, `weights`, `query_pos` and `out` are the rows of that sequence
        only, so that `out` is written in place. The layout matches the
        per-query implementation: the tokens of the selected complete pools
        come first, then the incomplete tail, then `-1`.
        """
        num_pools = pool_keys.shape[0]
        select_k = min(int(self.index_topk) // self.index_kpool, num_pools)
        num_selected = select_k * self.index_kpool
        # A pool is visible to a query only if the query is not before the last
        # token of that pool.
        pool_counts = torch.clamp((query_pos + 1) // self.index_kpool, max=num_pools)
        if select_k > 0:
            tile = max(
                1, self._SCORE_BUDGET_ELEMS // (self.n_heads * max(num_pools, 1))
            )
            pool_ids = torch.arange(num_pools, device=q.device)
            pool_shifts = torch.arange(self.index_kpool, device=q.device)
            for start in range(0, q.shape[0], tile):
                stop = min(start + tile, q.shape[0])
                visible = pool_counts[start:stop].unsqueeze(1)
                scores = torch.matmul(q[start:stop].float(), pool_keys.float().T)
                scores = F.relu(scores * self.softmax_scale)
                scores = torch.matmul(
                    weights[start:stop].float().unsqueeze(-2), scores
                ).squeeze(-2)
                scores = scores.masked_fill(
                    pool_ids.unsqueeze(0) >= visible, float("-inf")
                )
                selected = scores.topk(select_k, dim=-1).indices
                # Pool `p` covers the tokens `[p * kpool, (p + 1) * kpool)`,
                # selected pools out of the causal range are dropped.
                selected_pos = selected.unsqueeze(-1) * self.index_kpool + pool_shifts
                selected_pos = selected_pos.masked_fill(
                    (selected >= visible).unsqueeze(-1), -1
                ).flatten(-2)
                out[start:stop, :num_selected] = selected_pos.to(torch.int32)

        if self.index_kpool_always_select_tail and self.index_kpool > 1:
            tail_offsets = torch.arange(self.index_kpool - 1, device=q.device)
            pool_widths = pool_counts * self.index_kpool
            # The incomplete tail of a query is written right after the pools
            # it selected, and holds the last tokens of its visible prefix.
            selected_widths = torch.clamp(pool_counts, max=select_k) * self.index_kpool
            tail_slots = selected_widths.unsqueeze(1) + tail_offsets.unsqueeze(0)
            tail_pos = pool_widths.unsqueeze(1) + tail_offsets.unsqueeze(0)
            tail = torch.where(
                tail_offsets.unsqueeze(0) < (query_pos + 1 - pool_widths).unsqueeze(1),
                tail_pos,
                -1,
            )
            out.scatter_(1, tail_slots, tail.to(torch.int32))
        return out

    def _build_kpool_topk(self, x, q, packed_states, seq_len_delta):
        q = q.view(q.shape[0], self.n_heads, self.head_dim)
        weights = self.weights_proj(x) * (self.n_heads**-0.5)
        query_pos = seq_len_delta.delta_position_ids_tensor_device
        output_width = int(self.index_topk) + (
            self.index_kpool - 1 if self.index_kpool_always_select_tail else 0
        )
        if seq_len_delta.is_decode_stage:
            if int(seq_len_delta.new.max_len) <= int(self.index_topk):
                offsets = torch.arange(
                    output_width, dtype=query_pos.dtype, device=x.device
                )
                tail = query_pos.unsqueeze(1) - (output_width - 1 - offsets).unsqueeze(
                    0
                )
                return torch.where(tail >= 0, tail, torch.full_like(tail, -1)).to(
                    torch.int32
                )
        new_lens = seq_len_delta.new.lens_list
        delta_lens = seq_len_delta.delta_lens_list
        # The packed states are read in the order of the sequences and of the
        # positions inside each sequence, so the states of a sequence are a
        # contiguous slice instead of a per-query mask over the whole context.
        assert len(new_lens) == len(delta_lens)
        assert sum(new_lens) == packed_states.shape[0]
        assert sum(delta_lens) == q.shape[0]
        out = torch.full(
            (q.shape[0], output_width), -1, dtype=torch.int32, device=x.device
        )
        row_start = 0
        kv_start = 0
        for num_tokens, num_queries in zip(new_lens, delta_lens):
            rows = slice(row_start, row_start + num_queries)
            packed_seq = packed_states[kv_start : kv_start + num_tokens]
            row_start += num_queries
            kv_start += num_tokens
            if num_queries == 0:
                continue
            self._score_pools_topk(
                q[rows],
                weights[rows],
                query_pos[rows],
                self._pool_sequence_states(packed_seq),
                out[rows],
            )
        return out

    @override
    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len_delta,
        freqs_cis: BatchedFreqsCis,
        is_causal: bool,
        cache_accessor,
        freqs_cis_k: Optional[BatchedFreqsCis] = None,
        k_pre_normed: bool = False,
    ) -> torch.Tensor:
        mode = self._indexer_buffer_mode
        buffer = self._indexer_buffer
        if mode == "read":
            assert buffer is not None and buffer.topk is not None
            return buffer.topk

        # CP (pcp) prefill hands us the all-gathered global indexer K and the
        # CP-local q-axis view, while `x` — and therefore the compress gate —
        # stays local. The gate is all-gathered as well, so that every rank
        # appends the identical global packs on the global token axis. This is
        # the same "every rank writes the whole context" scheme the MLA path
        # uses for its KV cache; it keeps reads local to the rank and needs no
        # cross-rank ordering assumption.
        cp_active = freqs_cis_k is not None

        if cp_active:
            cp_ctx = get_cp_context()
            assert cp_ctx.step_active and cp_ctx.pcp_size > 1, (
                "GLM5Next indexer received global K (CP path) without an "
                "active CP prefill step"
            )
            assert (
                not seq_len_delta.is_decode_stage
            ), "GLM5Next indexer only supports CP during prefill"
            gate_scores = cp_ctx.allgather_hidden_states(
                self.index_kpool_compress_gate(x), None, None
            )
            assert gate_scores.shape[0] == k.shape[0], (
                f"global compress gate rows ({gate_scores.shape[0]}) must match "
                f"global indexer K rows ({k.shape[0]})"
            )
            # The packs cover the global token axis, so they are appended with
            # the unsliced delta rather than the CP-local query view.
            append_delta = seq_len_delta.base_delta
        else:
            gate_scores = self.index_kpool_compress_gate(x)
            append_delta = seq_len_delta

        k = k if k_pre_normed else self.k_norm(k)
        valid = torch.ones((k.shape[0], 1), dtype=k.dtype, device=k.device)
        self._append_packed_states(
            torch.cat([k, gate_scores, valid], dim=-1), append_delta, cache_accessor
        )
        out = self._build_kpool_topk(
            x,
            q,
            self._read_full_packed_states(seq_len_delta, cache_accessor),
            seq_len_delta,
        )
        if mode == "write":
            assert buffer is not None
            buffer.topk = out
            buffer.topk_page_table = None
        return out


class AttentionGLM5Next(AttentionGLM52):
    def __init__(self, *args, indexer_role: str, **kwargs):
        old_allowed_merge_qkv = QuantizationRegistry.allowed_merge_qkv
        QuantizationRegistry.allowed_merge_qkv = classmethod(
            lambda cls, checkpoint, can_use_mla_prologue_int8=False: False
        )
        try:
            super().__init__(*args, indexer_role=indexer_role, **kwargs)
        finally:
            QuantizationRegistry.allowed_merge_qkv = old_allowed_merge_qkv
        if getattr(self, "wqkv_a_indexer_k", None) is not None:
            self.indexer.wqkv_a_indexer_k = self.wqkv_a_indexer_k
            del self.wqkv_a_indexer_k
            self.indexer.merge_qkv = self.merge_qkv
            self.merge_qkv = False
        if indexer_role == "shared":
            del self.indexer.index_kpool_compress_ape
            del self.indexer.index_kpool_compress_gate

    @override
    def make_indexer(self, args, *, checkpoint_prefix, indexer_impl):
        mode = None
        if self._indexer_role_for_make == "shared":
            mode = "read"
        elif self._indexer_buffer_for_make is not None:
            mode = "write"
        return IndexerGLM5Next(
            args,
            checkpoint_prefix=checkpoint_prefix,
            indexer_impl=indexer_impl,
            buffer_mode=mode,
            indexer_buffer=self._indexer_buffer_for_make,
        )


class Glm5NextForgetGate(nn.Module):
    def __init__(self, args, checkpoint_prefix):
        super().__init__()
        checkpoint_prefix = as_checkpoint_prefix(checkpoint_prefix)
        self.head_dim = int(args.linear_head_dim)
        self.num_heads = int(args.linear_num_heads)
        self.qkv_dim = self.head_dim * self.num_heads
        self.safe_gate_lower_bound = getattr(args, "linear_lower_bound", -5.0)
        self.f_a_proj = LocalLinear(
            args.dim,
            self.head_dim,
            has_bias=False,
            checkpoint_prefix=checkpoint_prefix / "f_a_proj",
        )
        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.qkv_dim,
            has_bias=False,
            gather_output=False,
            checkpoint_prefix=checkpoint_prefix / "f_b_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.qkv_dim // get_tp_size(), dtype=torch.float32)
        )
        self.A_log = nn.Parameter(
            torch.empty(self.num_heads // get_tp_size(), dtype=torch.float32)
        )

    def forward(self, x, f_a: Optional[torch.Tensor] = None):
        if f_a is None:
            f_a = self.f_a_proj(x)
        g = self.f_b_proj(f_a).float() + self.dt_bias.float().view(1, -1)
        # `x` may hold fewer rows than `f_a` under CP, where the gate is also
        # computed from all-gathered projections.
        g = g.view(g.shape[0], -1, self.head_dim)
        decay_rate = torch.exp(self.A_log.float()).view(1, -1, 1)
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * torch.sigmoid(decay_rate * g.float())
        return -decay_rate * F.softplus(g.float())


class Glm5NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate):
        return rms_norm_gate(
            hidden_states,
            gate,
            self.weight,
            self.variance_epsilon,
            compute_dtype=torch.float32,
            activation="sigmoid",
        )


class Glm5NextLinearAttention(nn.Module):
    def __init__(self, args, layer_id, cache, *, checkpoint_prefix):
        super().__init__()
        checkpoint_prefix = as_checkpoint_prefix(checkpoint_prefix)
        self.num_heads = int(args.linear_num_heads)
        self.head_dim = int(args.linear_head_dim)
        self.qkv_dim = self.num_heads * self.head_dim
        self.n_local_heads = self.num_heads // get_tp_size()
        self.local_qkv_dim = self.n_local_heads * self.head_dim
        self.conv_kernel_size = int(args.linear_conv_kernel_dim)
        self.conv_impl = getattr(args, "linear_conv_impl", "auto")
        self.layer_id = layer_id
        self.cache = cache
        self.impl = getattr(args, "linear_attention_impl", "auto")
        self.mtp_size = get_global_args().infer.mtp_size
        self.forget_gate = Glm5NextForgetGate(args, checkpoint_prefix / "forget_gate")
        self.merge_qkv = QuantizationRegistry.allowed_merge_qkv(
            checkpoint_prefix / "in_proj_qkvbfg_a"
        )
        if self.merge_qkv:
            del self.forget_gate.f_a_proj
            self.in_proj_qkvbfg_a = ColumnParallelLinear(
                args.dim,
                self.qkv_dim * 3 + self.num_heads + self.head_dim * 2 * get_tp_size(),
                has_bias=False,
                gather_output=False,
                checkpoint_prefix=checkpoint_prefix / "in_proj_qkvbfg_a",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                args.dim,
                self.qkv_dim,
                has_bias=False,
                gather_output=False,
                checkpoint_prefix=checkpoint_prefix / "q_proj",
            )
            self.k_proj = ColumnParallelLinear(
                args.dim,
                self.qkv_dim,
                has_bias=False,
                gather_output=False,
                checkpoint_prefix=checkpoint_prefix / "k_proj",
            )
            self.v_proj = ColumnParallelLinear(
                args.dim,
                self.qkv_dim,
                has_bias=False,
                gather_output=False,
                checkpoint_prefix=checkpoint_prefix / "v_proj",
            )
        self.conv1d = nn.Conv1d(
            self.local_qkv_dim * 3,
            self.local_qkv_dim * 3,
            self.conv_kernel_size,
            groups=self.local_qkv_dim * 3,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        if not self.merge_qkv:
            self.b_proj = ColumnParallelLinear(
                args.dim,
                self.num_heads,
                has_bias=False,
                gather_output=False,
                checkpoint_prefix=checkpoint_prefix / "b_proj",
            )
            self.g_a_proj = LocalLinear(
                args.dim,
                self.head_dim,
                has_bias=False,
                checkpoint_prefix=checkpoint_prefix / "g_a_proj",
            )
        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.qkv_dim,
            has_bias=False,
            gather_output=False,
            checkpoint_prefix=checkpoint_prefix / "g_b_proj",
        )
        self.o_norm = Glm5NextRMSNormGated(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            self.qkv_dim,
            args.dim,
            has_bias=False,
            input_is_parallel=True,
            base_linear_class=get_linear_layout_contig_y(
                "torch", checkpoint_prefix=checkpoint_prefix / "o_proj"
            ),
            checkpoint_prefix=checkpoint_prefix / "o_proj",
        )

    def forward(self, x: torch.Tensor):
        seq_len_delta = self.cache.seq_len_delta
        cp_ctx = get_cp_context()
        cp_active = cp_ctx.step_active and not seq_len_delta.is_decode_stage
        is_mtp_decode_stage = (
            self.cache.is_mtp_decode_stage if self.mtp_size > 1 else False
        )
        mtp_accept_indices = get_mtp_accept_indices() if is_mtp_decode_stage else None
        cache_accessor = self.cache.get_accessor(self.layer_id)
        conv_state = read_from_singleton_paged_kv_cache(
            cache_accessor.kv["conv_state"],
            cache_accessor.block_table,
            mtp_accept_indices=mtp_accept_indices,
        )
        recurrent_state = read_from_singleton_paged_kv_cache(
            cache_accessor.kv["recurrent_state"],
            cache_accessor.block_table,
            mtp_accept_indices=mtp_accept_indices,
        )
        if self.merge_qkv:
            projected = self.in_proj_qkvbfg_a(x)
            qkv, beta_raw, f_a, g_a = torch.split(
                projected,
                [
                    self.local_qkv_dim * 3,
                    self.n_local_heads,
                    self.head_dim,
                    self.head_dim,
                ],
                dim=-1,
            )
        else:
            qkv = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)
            beta_raw = self.b_proj(x)
            f_a = self.forget_gate.f_a_proj(x)
            g_a = self.g_a_proj(x)
        if cp_active:
            # Both the causal conv and the gated-delta recurrence couple tokens
            # along the sequence, but a CP rank only holds every pcp-th token.
            # All-gather the projected states and run them over the full global
            # sequence on every rank (the same "whole context on every rank"
            # scheme as the MLA path), then slice the per-token result back to
            # this rank's rows below.
            n_local = x.shape[0]
            total_tokens = seq_len_delta.delta_total_len

            qkv, beta_raw, f_a, g_a = torch.split(
                cp_ctx.allgather_hidden_states(
                    torch.cat([qkv, beta_raw, f_a, g_a], dim=-1), None, None
                ),
                [qkv.shape[-1], beta_raw.shape[-1], f_a.shape[-1], g_a.shape[-1]],
                dim=-1,
            )

        if seq_len_delta.is_classic_decoding:
            qkv, conv_state = causal_conv1d_update(
                qkv, conv_state, self.conv1d.weight, impl=self.conv_impl
            )
        elif is_mtp_decode_stage:
            qkv = qkv.view(-1, self.mtp_size, qkv.shape[-1])
            qkv_out = torch.empty_like(qkv)
            conv_state_out = torch.empty(
                (qkv.shape[0], self.mtp_size, *conv_state.shape[1:]),
                device=qkv.device,
                dtype=conv_state.dtype,
            )
            for step in range(self.mtp_size):
                qkv_out[:, step], conv_state = causal_conv1d_update(
                    qkv[:, step], conv_state, self.conv1d.weight, impl=self.conv_impl
                )
                conv_state_out[:, step] = conv_state
            qkv = qkv_out.view(-1, qkv_out.shape[-1])
            conv_state = conv_state_out
        else:
            qkv, conv_state = causal_conv1d_prefill(
                qkv,
                conv_state,
                self.conv1d.weight,
                seq_len_delta.delta_prefix_lens_tensor_device,
                impl=self.conv_impl,
            )
            if self.mtp_size > 1:
                conv_state = conv_state.unsqueeze(1).expand(
                    -1, self.mtp_size, *([-1] * (conv_state.dim() - 1))
                )
        q, k, v = torch.split(qkv, [self.local_qkv_dim] * 3, dim=-1)
        q, k, v = map(
            lambda h: h.reshape(h.size(0), -1, self.head_dim), (q, k, v)
        )  # (total_len, n_heads, head_dim)

        beta = beta_raw.sigmoid()
        g = self.forget_gate(x, f_a=f_a)
        if seq_len_delta.is_classic_decoding:
            out, last_state = recurrent_kimi_delta_attention(
                q.unsqueeze(1),
                k.unsqueeze(1),
                v.unsqueeze(1),
                g=g.unsqueeze(1),
                beta=beta.unsqueeze(1),
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                impl=self.impl,
            )
        elif is_mtp_decode_stage:
            q_mtp, k_mtp, v_mtp, beta_mtp, g_mtp = map(
                lambda x: x.view(-1, self.mtp_size, *x.shape[1:]).contiguous(),
                [q, k, v, beta, g],
            )

            out_mtp = torch.empty_like(q_mtp)
            last_state = torch.empty(
                (q_mtp.shape[0], self.mtp_size, *recurrent_state.shape[1:]),
                device=q_mtp.device,
                dtype=recurrent_state.dtype,
            )
            for step in range(self.mtp_size):
                out_step, recurrent_state = recurrent_kimi_delta_attention(
                    q_mtp[:, step : step + 1],
                    k_mtp[:, step : step + 1],
                    v_mtp[:, step : step + 1],
                    g=g_mtp[:, step : step + 1],
                    beta=beta_mtp[:, step : step + 1],
                    initial_state=recurrent_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    impl=self.impl,
                )
                out_mtp[:, step] = out_step.squeeze(1)
                last_state[:, step] = recurrent_state
            out = out_mtp.view(-1, self.n_local_heads, self.head_dim)
        else:
            out, last_state = chunk_kimi_delta_attention(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                g=g.unsqueeze(0),
                beta=beta.unsqueeze(0),
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=seq_len_delta.delta_prefix_lens_tensor_device,
                seq_len_list=seq_len_delta.delta_lens_list,
                impl=self.impl,
            )
            if self.mtp_size > 1:
                last_state = last_state.unsqueeze(1).expand(
                    -1, self.mtp_size, *([-1] * (last_state.dim() - 1))
                )
        if cp_active:
            # The recurrence above ran on the all-gathered global sequence, so
            # `out` and the gate source cover every token; keep only this
            # rank's interleaved rows. `out` still carries the leading batch dim
            # of `chunk_kimi_delta_attention`, while `g_a` is [rows, head_dim].
            # The conv/recurrent states written below are global on every rank,
            # which is what the next step (or the non-CP decode) reads.
            assert (
                out.dim() == 4 and out.shape[0] == 1
            ), f"unexpected linear attention output shape {tuple(out.shape)}"
            local_rows, _ = cp_ctx.local_flat_indices(n_local, total_tokens, out.device)
            out = out.index_select(1, local_rows)
            g_a = g_a.index_select(0, local_rows)
        update_singleton_paged_kv_cache(
            cache_accessor.kv["conv_state"],
            cache_accessor.block_table,
            conv_state,
            mtp_size=self.mtp_size,
        )
        update_singleton_paged_kv_cache(
            cache_accessor.kv["recurrent_state"],
            cache_accessor.block_table,
            last_state.to(x.dtype),
            mtp_size=self.mtp_size,
        )
        gate = self.g_b_proj(g_a).view(x.shape[0], self.n_local_heads, self.head_dim)
        out = self.o_norm(
            out.reshape(-1, self.head_dim), gate.reshape(-1, self.head_dim)
        ).view(x.shape[0], -1)
        return self.o_proj(out)


class TransformerBlockGLM5Next(TransformerBlock):
    def __init__(
        self,
        layer_id,
        args,
        cache_dict,
        attn_backend,
        op_impl,
        mla_absorb,
        *,
        checkpoint_prefix,
        indexer_impl,
        indexer_role,
        indexer_buffer=None,
    ):
        super().__init__(
            layer_id, args, cache_dict, attn_backend=attn_backend, op_impl=op_impl
        )
        self.block_type = args.layer_types[layer_id]
        if self.block_type == "linear_attention":
            self.self_attn = Glm5NextLinearAttention(
                args,
                layer_id,
                cache_dict["linear"],
                checkpoint_prefix=f"{checkpoint_prefix}.self_attn",
            )
        else:
            self.self_attn = AttentionGLM5Next(
                args,
                layer_id,
                cache_dict["main"],
                attn_backend,
                op_impl=op_impl,
                mla_absorb=mla_absorb,
                checkpoint_prefix=f"{checkpoint_prefix}.self_attn",
                indexer_cache=cache_dict.get("indexer"),
                indexer_impl=indexer_impl,
                indexer_role=indexer_role,
                indexer_buffer=indexer_buffer,
            )
        self.mlp = make_mlp_for_layer(
            args,
            layer_id,
            op_impl,
            checkpoint_prefix=checkpoint_prefix,
            use_layer_mlp_type=True,
        )
        self.input_layernorm = RMSNorm(
            args.dim, dtype=torch.float32, eps=args.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            args.dim, dtype=torch.float32, eps=args.rms_norm_eps
        )
        hc_args = dict(
            rms_eps=args.rms_norm_eps,
            hc_pre_eps=args.hc_eps,
            hc_sinkhorn_eps=args.hc_eps,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=args.hc_sinkhorn_iters,
        )
        self.attn_hc = mHCSubLayer(args.hc_mult, args.dim, **hc_args)
        self.ffn_hc = mHCSubLayer(args.hc_mult, args.dim, **hc_args)

    def forward(
        self, x: torch.Tensor, freqs_cis: BatchedFreqsCis, is_mtp: bool = False
    ):
        if self.block_type == "linear_attention":
            x = self.attn_hc(
                x,
                lambda h: self.self_attn(self.input_layernorm(h)),
            )
        else:
            x = self.attn_hc(
                x,
                lambda h: self.self_attn(self.input_layernorm(h), freqs_cis, is_mtp),
            )
        x = self.ffn_hc(x, lambda h: self.mlp(self.post_attention_layernorm(h)))
        return x


class TransformerBlockGLM5NextMTP(TransformerBlockGLM52MTP):
    def __init__(
        self,
        layer_id,
        args,
        cache_dict,
        attn_backend,
        op_impl,
        mla_absorb,
        *,
        checkpoint_prefix,
        indexer_impl,
    ):
        # The MTP layer does not use mHC; it reuses the GLM-5.2 residual block.
        super().__init__(
            layer_id,
            args,
            cache_dict,
            attn_backend,
            op_impl,
            mla_absorb,
            checkpoint_prefix=checkpoint_prefix,
            indexer_impl=indexer_impl,
        )

    def _make_attention(
        self,
        layer_id,
        args,
        cache_dict,
        attn_backend,
        op_impl: str,
        mla_absorb: str,
        checkpoint_prefix,
        indexer_impl,
        indexer_role: str,
        indexer_buffer=None,
    ):
        return AttentionGLM5Next(
            args,
            layer_id,
            cache_dict["main"],
            attn_backend,
            op_impl=op_impl,
            mla_absorb=mla_absorb,
            checkpoint_prefix=f"{checkpoint_prefix}.self_attn",
            indexer_cache=cache_dict.get("indexer"),
            indexer_impl=indexer_impl,
            indexer_role=indexer_role,
            indexer_buffer=indexer_buffer,
        )


@register_model(ModelType.GLM_5_NEXT)
class TransformerGLM5Next(TransformerDeepSeekV3):
    @override
    def __init__(
        self,
        params,
        cache_dict: dict[str, KVCacheBase],
        *,
        max_position_embeddings: int,
        attn_backend,
        op_impl: str,
        mla_absorb: str,
    ):
        super().__init__(
            params,
            cache_dict,
            max_position_embeddings=max_position_embeddings,
            attn_backend=attn_backend,
            op_impl=op_impl,
            mla_absorb=mla_absorb,
        )

    @override
    def _init_layers(self, cache_dict: dict[str, KVCacheBase], attn_backend, op_impl):
        self.layers = torch.nn.ModuleList()
        self._backbone_buf = _IndexerBuffer()
        self._mtp_buf = _IndexerBuffer()
        self._mtp_skip: bool = False
        for layer_id in range(self.local_begin_layer_id, self.local_end_layer_id):
            if layer_id >= self.params.n_layers:
                self.layers.append(
                    TransformerBlockGLM5NextMTP(
                        layer_id,
                        self.params,
                        cache_dict,
                        attn_backend,
                        op_impl,
                        self.mla_absorb,
                        checkpoint_prefix=f"layers.{layer_id}",
                        indexer_impl=self.indexer_backend,
                    )
                )
            else:
                role = (
                    self.params.indexer_types[layer_id]
                    if self.params.layer_types[layer_id] == "deepseek_sparse_attention"
                    else "linear"
                )
                self.layers.append(
                    TransformerBlockGLM5Next(
                        layer_id,
                        self.params,
                        cache_dict,
                        attn_backend,
                        op_impl,
                        self.mla_absorb,
                        checkpoint_prefix=f"layers.{layer_id}",
                        indexer_impl=self.indexer_backend,
                        indexer_role=role,
                        indexer_buffer=self._backbone_buf,
                    )
                )

    def get_pipeline_payload_shape(self, num_tokens: int) -> list[int]:
        return [num_tokens, self.params.hc_mult, self.params.dim]

    def get_pipeline_payload_dtype(self) -> torch.dtype:
        return torch.get_default_dtype()

    def _clear_backbone_indexer_buffer(self) -> None:
        self._backbone_buf.topk = None
        self._backbone_buf.topk_page_table = None

    # def set_mtp_skip_topk(self, skip: bool) -> None:
    #     self._mtp_skip = skip

    def _reduce_mhc(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 3 and h.shape[1] == self.params.hc_mult:
            return h.mean(dim=1)
        return h

    @override
    def _pre_layers(self, h, **args):
        return (
            super()
            ._pre_layers(h, **args)
            .unsqueeze(1)
            .repeat(1, self.params.hc_mult, 1)
        )

    def _run_non_mtp_layers(
        self, h: torch.Tensor, freqs_cis: BatchedFreqsCis
    ) -> torch.Tensor:
        for layer in self.non_mtp_layers:
            h = layer(h, freqs_cis)
        return h

    @override
    def _post_layers(self, h):
        h = h.mean(dim=1)
        h = self.norm(h)
        return self.lm_head(h)

    @override
    @torch.inference_mode()
    def prefill_no_pipeline(
        self, tokens: torch.Tensor, output_token_offsets: torch.Tensor, **args
    ) -> torch.Tensor:
        self._clear_backbone_indexer_buffer()
        freqs_cis = self.prepare_freqs_cis()
        delta_total = self.cache_dict["main"].seq_len_delta.delta_total_len
        tokens, _, freqs_cis = self.cp_context.split_prefill(
            tokens=tokens, hiddens=None, freqs_cis=freqs_cis, total_tokens=delta_total
        )

        if self.cp_context.step_active:
            self.cp_context.prepare_local_lengths(
                self.cache_dict["main"].seq_len_delta,
                int(tokens.shape[0]),
                is_decode_stage=False,
            )

        if self.moe_impl is not None:
            self.moe_impl.prepare(TaskType.Prefill, int(tokens.shape[0]))

        h = self._pre_layers(tokens, **args)
        h = self._run_non_mtp_layers(h, freqs_cis)
        if self.mtp_size > 1:
            self.mtp_prefill(
                x=self._pre_layers_mtp(tokens, **args),
                h=h,
                freqs_cis=freqs_cis,
            )
        return self.cp_context.allgather_hidden_states(
            h, output_token_offsets, self._post_layers
        )

    @override
    @torch.inference_mode()
    def decode_no_pipeline(self, tokens, freqs_cis: BatchedFreqsCis):
        self._clear_backbone_indexer_buffer()
        h = self._pre_layers(tokens)
        h = self._run_non_mtp_layers(h, freqs_cis)
        if self.mtp_size > 1:
            h_for_cache = self._reduce_mhc(h)
            self.update_mtp_hidden_states(
                self.norm(h_for_cache, compute_dtype=h_for_cache.dtype),
                is_mtp=True,
            )
        return self._post_layers(h).float()

    @override
    @torch.inference_mode()
    def prefill_pipeline(
        self,
        tokens: torch.Tensor | None,
        hiddens: torch.Tensor | None,
        output_token_offsets: torch.Tensor,
        **args,
    ) -> torch.Tensor:
        self._clear_backbone_indexer_buffer()
        freqs_cis = self.prepare_freqs_cis()
        delta_total = self.cache_dict["main"].seq_len_delta.delta_total_len

        if self.pp_stage == 0:
            assert tokens is not None and hiddens is None
            tokens, _, freqs_cis = self.cp_context.split_prefill(
                tokens=tokens,
                hiddens=None,
                freqs_cis=freqs_cis,
                total_tokens=delta_total,
            )
            batch_size = tokens.shape[0]
            h = self._pre_layers(tokens, **args)
        else:
            assert hiddens is not None and hiddens.ndim == 3
            _, hiddens, freqs_cis = self.cp_context.split_prefill(
                tokens=None,
                hiddens=hiddens,
                freqs_cis=freqs_cis,
                total_tokens=delta_total,
            )
            batch_size = hiddens.shape[0]
            h = hiddens
            del hiddens

        if self.cp_context.step_active:
            self.cp_context.prepare_local_lengths(
                self.cache_dict["main"].seq_len_delta,
                int(batch_size),
                is_decode_stage=False,
            )

        if self.moe_impl is not None:
            self.moe_impl.prepare(TaskType.Prefill, batch_size)

        h = self._run_non_mtp_layers(h, freqs_cis)

        if self.pp_stage == self.pp_end_stage:
            return self.cp_context.allgather_hidden_states(
                h,
                output_token_offsets,
                self._post_layers,
            )
        return h

    @override
    @torch.inference_mode()
    def decode_pipeline(self, middle_state, freqs_cis: BatchedFreqsCis):
        self._clear_backbone_indexer_buffer()
        if self.pp_stage == 0:
            h = self._pre_layers(middle_state)
        else:
            assert middle_state.ndim == 3
            h = middle_state
        h = self._run_non_mtp_layers(h, freqs_cis)
        if self.pp_stage == self.pp_end_stage:
            return self._post_layers(h).float()
        return h

    def mtp_decode_no_pipeline(self, tokens, freqs_cis: BatchedFreqsCis):
        h = self._pre_layers_mtp(tokens)
        mtp_layer = self.layers[-1]
        mtp_layer.set_indexer_buffer(
            "read" if self._mtp_skip else "write",
            self._mtp_buf,
        )
        try:
            h = mtp_layer(
                h,
                freqs_cis,
                self.read_mtp_hidden_states(),
                is_mtp=True,
            )
        finally:
            mtp_layer.set_indexer_buffer(None, None)
        h = self._reduce_mhc(h)
        self.update_mtp_hidden_states(h)
        return self._post_layers_mtp(h).float()

    @override
    def _post_layers_mtp(self, h: torch.Tensor) -> torch.Tensor:
        return super()._post_layers_mtp(h)

    @override
    def mtp_prefill(self, x, h, freqs_cis):
        super().mtp_prefill(x, self._reduce_mhc(h), freqs_cis)

    @override
    def _get_non_layer_prefix_mappings(self) -> list[tuple[str, str]]:
        prefix_mappings = []
        if self.pp_stage == 0:
            prefix_mappings.append(
                ("model.language_model.embed_tokens.", "embed_tokens.")
            )
        if self.pp_stage == self.pp_end_stage:
            prefix_mappings.extend(
                [
                    ("model.language_model.norm.", "norm."),
                    ("lm_head.", "lm_head."),
                ]
            )
            if self.mtp_size > 1 and self.mtp_tie_word_embeddings:
                prefix_mappings.append(
                    ("model.language_model.embed_tokens.", "embed_tokens.")
                )
        return prefix_mappings

    @override
    def _get_layer_i_prefix_mapping(self, i: int) -> tuple[str, str]:
        return (f"model.language_model.layers.{i}.", f"layers.{i}.")

    @override
    def _get_tensor_column_parallel_layer_names(self) -> list[str]:
        ret = super()._get_tensor_column_parallel_layer_names()
        ret += [
            "in_proj_qkvbfg_a",
            "q_proj",
            "k_proj",
            "v_proj",
            "f_b_proj",
            "b_proj",
            "g_b_proj",
        ]
        return ret

    @override
    def _get_tensor_row_parallel_layer_names(self) -> list[str]:
        return super()._get_tensor_row_parallel_layer_names() + ["self_attn.o_proj"]

    @override
    def process_state_dict_for_splitting_qkv(self, checkpoint: dict[str, Any]):
        projection_size = self.params.linear_num_heads * self.params.linear_head_dim
        return self.process_state_dict_for_splitting_tensors(
            checkpoint,
            src_layer="in_proj_qkvbfg_a",
            tgt_layer_to_proportion=OrderedDict(
                [
                    ("q_proj", projection_size),
                    ("k_proj", projection_size),
                    ("v_proj", projection_size),
                    ("b_proj", self.params.linear_num_heads),
                    ("forget_gate.f_a_proj", self.params.linear_head_dim),
                    ("g_a_proj", self.params.linear_head_dim),
                ]
            ),
        )

    @override
    def process_state_dict_for_merging_qkv(self, checkpoint: dict[str, Any]):
        return self.process_state_dict_for_merging_tensors(
            checkpoint,
            tgt_layer="in_proj_qkvbfg_a",
            src_layers=[
                "q_proj",
                "k_proj",
                "v_proj",
                "b_proj",
                "forget_gate.f_a_proj",
                "g_a_proj",
            ],
            enable_callback=QuantizationRegistry.allowed_merge_qkv,
        )

    def chunk_checkpoint_for_tensor_parallelize_attn_weights(
        self, checkpoint, rank, world_size
    ):
        for k in list(checkpoint.keys()):
            if any(
                k.endswith(name) for name in [".self_attn.A_log", ".self_attn.dt_bias"]
            ):
                if checkpoint[k].shape[0] % world_size == 0:
                    checkpoint[k] = torch.chunk(checkpoint[k], world_size, dim=0)[rank]

        for src_layer in ("q_conv1d", "k_conv1d", "v_conv1d"):
            for k in list(checkpoint.keys()):
                if not k.endswith(f".self_attn.{src_layer}.weight"):
                    continue
                assert checkpoint[k].shape[0] % world_size == 0
                checkpoint[k] = torch.chunk(checkpoint[k], world_size, dim=0)[rank]

        checkpoint = self.process_state_dict_for_merging_tensors(
            checkpoint,
            tgt_layer="conv1d",
            src_layers=["q_conv1d", "k_conv1d", "v_conv1d"],
            dim_type=0,
        )
        return checkpoint

    def _dequantize_self_attn_fp8_weights(self, state_dict: dict[str, Any]) -> None:
        weight_dequant_fn = (
            soft_fp8_blockfp8_weight_dequant
            if get_global_args().infer.raise_lower_bit_float_to == "bfloat16"
            else blockfp8_weight_dequant
        )
        for k in list(state_dict.keys()):
            if ".self_attn." not in k or not k.endswith(".weight"):
                continue
            weight = state_dict[k]
            if weight.dtype != torch.float8_e4m3fn:
                continue
            prefix = k[: -len(".weight")]
            scale_key = (
                f"{prefix}.weight_scale_inv"
                if f"{prefix}.weight_scale_inv" in state_dict
                else f"{prefix}.scale"
            )
            if scale_key not in state_dict:
                continue
            scale = state_dict.pop(scale_key)
            old_device = weight.device
            state_dict[k] = weight_dequant_fn(
                weight.cuda(),
                scale.cuda(),
                scale_block_shape=[128, 128],
            ).to(old_device)

    @override
    def preprocess_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *,
        skip_preprocess: bool = False,
        replace: bool = True,
    ) -> dict[str, Any]:
        if not skip_preprocess:
            state_dict = self.chunk_checkpoint_for_tensor_parallelize_attn_weights(
                state_dict, self.rank % self.tp_size, self.tp_size
            )
        if not skip_preprocess:
            self._dequantize_self_attn_fp8_weights(state_dict)
        if not skip_preprocess and replace:
            for k in list(state_dict.keys()):
                value = state_dict.pop(k)
                name = k.replace(".weight_scale_inv", ".scale")
                if name.endswith(".scale") and ".self_attn." in name:
                    continue
                name = name.replace(".indexer.wq_b", ".indexer_wq_b")
                name = name.replace(".indexer.wk", ".indexer_wk")
                if name.endswith(".indexer.index_kpool_compress_gate"):
                    name = f"{name}.weight"
                name = name.replace(".self_attn.A_log", ".self_attn.forget_gate.A_log")
                name = name.replace(
                    ".self_attn.dt_bias", ".self_attn.forget_gate.dt_bias"
                )
                name = name.replace(
                    ".self_attn.f_a_proj.", ".self_attn.forget_gate.f_a_proj."
                )
                name = name.replace(
                    ".self_attn.f_b_proj.", ".self_attn.forget_gate.f_b_proj."
                )
                name = name.replace(".hc_attn_fn", ".attn_hc.fn")
                name = name.replace(".hc_attn_base", ".attn_hc.hc_base")
                name = name.replace(".hc_attn_scale", ".attn_hc.hc_scale")
                name = name.replace(".hc_ffn_fn", ".ffn_hc.fn")
                name = name.replace(".hc_ffn_base", ".ffn_hc.hc_base")
                name = name.replace(".hc_ffn_scale", ".ffn_hc.hc_scale")
                name = name.replace(".attn_hc.scale", ".attn_hc.hc_scale")
                name = name.replace(".attn_hc.base", ".attn_hc.hc_base")
                name = name.replace(".ffn_hc.scale", ".ffn_hc.hc_scale")
                name = name.replace(".ffn_hc.base", ".ffn_hc.hc_base")
                state_dict[name] = value
        return super().preprocess_state_dict_parallel(
            state_dict, skip_preprocess=skip_preprocess, replace=replace
        )
