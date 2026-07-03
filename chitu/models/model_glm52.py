# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.2 model — extends DeepSeek-V3 with a shared top-k indexer schedule.

Backbone layers are tagged "full" or "shared" via config.json
``indexer_types``.  A "full" layer computes top-k and writes the result into a
per-step ``_IndexerBuffer``; "shared" layers read from that buffer without
recomputation.  Backbone buffer routing is bound once during layer
initialization; MTP decode temporarily switches the same indexer hook at call
time because its skip state is runtime-controlled.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import torch
from typing_extensions import override

from chitu.kv_cache import KVCacheBase
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.dsa_indexer import DSAIndexer
from chitu.task_type import TaskType
from chitu.models.registry import ModelType, register_model
from chitu.quantization import QuantizationRegistry
from chitu.quantization.utils import get_layer_id_from_checkpoint_prefix
from chitu.models.model import TransformerBlock, RMSNorm
from chitu.models.model_deepseek_v3 import (
    Indexer,
    AttentionDeepSeekV3,
    TransformerDeepSeekV3,
    MLPDeepSeekV3,
    ParallelMoeBlockDeepSeekV3,
    SharedHeadDeepSeekV3,
)
from chitu.quantization import get_quant_from_checkpoint_prefix
from chitu.tensor_parallel import VocabParallelEmbedding
from chitu.utils import parse_dtype, ceil_div
from chitu.muxi_utils import NormalMoeExpertsMuxiLayout, Blockfp8MoeExpertsMuxiLayout
from chitu.distributed.parallel_state import get_dp_size
from chitu.distributed.partition import compute_layer_dist_in_pp
from chitu.global_vars import get_global_args

# ---------------------------------------------------------------------------
# Shared buffer
# ---------------------------------------------------------------------------


class _IndexerBuffer:
    """Two-slot cell shared between a "full" layer and its downstream "shared" layers.

    ``topk``            — filled by :py:meth:`IndexerGLM52.forward`.
    ``topk_page_table`` — filled by
    :py:meth:`IndexerGLM52.build_decode_topk_page_table`.

    A full layer writes the slot used by the current path and clears the other;
    shared layers read back the most recently produced value.
    """

    __slots__ = ("topk", "topk_page_table")

    def __init__(self) -> None:
        self.topk: Optional[torch.Tensor] = None
        self.topk_page_table: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# IndexerGLM52
# ---------------------------------------------------------------------------


class IndexerGLM52(Indexer):
    """Indexer subclass with GLM-5.2 top-k buffer routing."""

    def __init__(
        self,
        args,
        *,
        checkpoint_prefix: str,
        indexer_impl: DSAIndexer,
        buffer_mode: Optional[str] = None,
        indexer_buffer: Optional[_IndexerBuffer] = None,
    ):
        super().__init__(
            args,
            checkpoint_prefix=checkpoint_prefix,
            indexer_impl=indexer_impl,
        )
        self.set_indexer_buffer(buffer_mode, indexer_buffer)

    def set_indexer_buffer(
        self,
        mode: Optional[str],
        buffer: Optional[_IndexerBuffer],
    ) -> None:
        assert mode in (None, "write", "read")
        assert (mode is None) == (buffer is None)
        self._indexer_buffer_mode = mode
        self._indexer_buffer = buffer

    def must_materialize_topk_indices(self) -> bool:
        return self._indexer_buffer_mode is not None

    def read_reused_topk_for_mla(
        self, *, use_page_table: bool
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert self._indexer_buffer_mode == "read"
        buffer = self._indexer_buffer
        assert buffer is not None
        if use_page_table and buffer.topk_page_table is not None:
            return None, buffer.topk_page_table
        assert buffer.topk is not None
        return buffer.topk, None

    def build_decode_topk_page_table(
        self,
        x: torch.Tensor,
        q,
        k,
        seq_len_delta,
        freqs_cis: BatchedFreqsCis,
        is_causal: bool,
        cache_accessor,
        source_page_table: torch.Tensor,
        freqs_cis_k: Optional[BatchedFreqsCis] = None,
        k_pre_normed: bool = False,
    ) -> torch.Tensor:
        mode = self._indexer_buffer_mode
        buffer = self._indexer_buffer
        if mode == "read":
            assert buffer is not None and buffer.topk_page_table is not None
            return buffer.topk_page_table
        out = super().build_decode_topk_page_table(
            x,
            q,
            k,
            seq_len_delta,
            freqs_cis,
            is_causal,
            cache_accessor,
            source_page_table,
            freqs_cis_k=freqs_cis_k,
            k_pre_normed=k_pre_normed,
        )
        if mode == "write":
            assert buffer is not None
            buffer.topk_page_table = out
            buffer.topk = None
        return out

    def forward(
        self,
        x: torch.Tensor,
        q,
        k,
        seq_len_delta,
        freqs_cis: BatchedFreqsCis,
        is_causal: bool,
        cache_accessor,
        freqs_cis_k: Optional[BatchedFreqsCis] = None,
        k_pre_normed: bool = False,
    ):
        mode = self._indexer_buffer_mode
        buffer = self._indexer_buffer
        if mode == "read":
            assert buffer is not None and buffer.topk is not None
            return buffer.topk
        out = super().forward(
            x,
            q,
            k,
            seq_len_delta,
            freqs_cis,
            is_causal,
            cache_accessor,
            freqs_cis_k=freqs_cis_k,
            k_pre_normed=k_pre_normed,
        )
        if mode == "write":
            assert buffer is not None
            buffer.topk = out
            buffer.topk_page_table = None
        return out


# ---------------------------------------------------------------------------
# AttentionGLM52
# ---------------------------------------------------------------------------


class AttentionGLM52(AttentionDeepSeekV3):
    """MLA attention with GLM-5.2 shared-indexer buffer routing."""

    def __init__(
        self,
        args,
        layer_id,
        cache,
        attn_backend,
        op_impl: str,
        mla_absorb,
        *,
        checkpoint_prefix: str,
        indexer_cache,
        indexer_impl,
        indexer_role: str,
        indexer_buffer: Optional[_IndexerBuffer] = None,
    ):
        has_local_indexer = indexer_role == "full"
        self._indexer_role_for_make = indexer_role
        self._indexer_buffer_for_make = indexer_buffer
        super().__init__(
            args,
            layer_id,
            cache,
            attn_backend,
            op_impl,
            mla_absorb,
            checkpoint_prefix=checkpoint_prefix,
            indexer_cache=indexer_cache,
            indexer_impl=indexer_impl,
            has_local_indexer=has_local_indexer,
        )
        if not has_local_indexer:
            # Shared layer: checkpoint has no k_norm/weights_proj weights.
            # Remove them so strict load_state_dict passes.
            del self.indexer.k_norm
            del self.indexer.weights_proj
        del self._indexer_role_for_make
        del self._indexer_buffer_for_make

    @override
    def make_indexer(
        self,
        args,
        *,
        checkpoint_prefix: str,
        indexer_impl: DSAIndexer,
    ) -> IndexerGLM52:
        indexer_role = self._indexer_role_for_make
        indexer_buffer = self._indexer_buffer_for_make
        buffer_mode = None
        if indexer_role == "shared":
            buffer_mode = "read"
        elif indexer_buffer is not None:
            buffer_mode = "write"
        return IndexerGLM52(
            args,
            checkpoint_prefix=checkpoint_prefix,
            indexer_impl=indexer_impl,
            buffer_mode=buffer_mode,
            indexer_buffer=indexer_buffer,
        )

    def set_indexer_buffer(
        self, mode: Optional[str], buffer: Optional[_IndexerBuffer]
    ) -> None:
        self.indexer.set_indexer_buffer(mode, buffer)


# ---------------------------------------------------------------------------
# TransformerBlockGLM52
# ---------------------------------------------------------------------------


class TransformerBlockGLM52(TransformerBlock):
    """Transformer block that constructs AttentionGLM52 directly.

    Inherits directly from TransformerBlock (not TransformerBlockDeepSeekV3)
    so that AttentionGLM52 is constructed exactly once — avoiding the double
    hook registration that would occur if we called super().__init__() on
    TransformerBlockDeepSeekV3 and then replaced self.self_attn.
    """

    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        mla_absorb,
        *,
        checkpoint_prefix,
        indexer_impl,
        indexer_role: str,
        indexer_buffer: Optional[_IndexerBuffer] = None,
    ):
        super().__init__(
            layer_id, args, cache_dict, attn_backend=attn_backend, op_impl=op_impl
        )
        self.layer_id = layer_id
        self.self_attn = AttentionGLM52(
            args,
            layer_id,
            cache_dict["main"],
            attn_backend,
            op_impl=op_impl,
            mla_absorb=mla_absorb,
            checkpoint_prefix=f"{checkpoint_prefix}.self_attn",
            indexer_cache=cache_dict["indexer"],
            indexer_impl=indexer_impl,
            indexer_role=indexer_role,
            indexer_buffer=indexer_buffer,
        )
        base_moe_experts_class = None
        if op_impl == "muxi_custom_kernel":
            quant = get_quant_from_checkpoint_prefix(
                f"{checkpoint_prefix}.mlp", args.quant_config.rules
            )
            if quant is None:
                base_moe_experts_class = NormalMoeExpertsMuxiLayout
            elif quant == "blockfp8":
                base_moe_experts_class = Blockfp8MoeExpertsMuxiLayout
            else:
                raise NotImplementedError(
                    "Unsupported quantization type for muxi_custom_kernel"
                )
        self.mlp = (
            MLPDeepSeekV3(
                args,
                role="standalone",
                op_impl=op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.mlp",
            )
            if layer_id < args.n_dense_layers
            else (
                ParallelMoeBlockDeepSeekV3(
                    args,
                    op_impl=op_impl,
                    base_moe_experts_class=base_moe_experts_class,
                    checkpoint_prefix=f"{checkpoint_prefix}.mlp",
                    layer_id=layer_id,
                )
            )
        )
        self.input_layernorm = RMSNorm(
            args.dim,
            dtype=parse_dtype(args.rms_norm_dtype),
            eps=args.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            args.dim,
            dtype=parse_dtype(args.rms_norm_dtype),
            eps=args.rms_norm_eps,
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        is_mtp: bool = False,
    ):
        x = x + self.self_attn(
            self.input_layernorm(x, compute_dtype=x.dtype),
            freqs_cis,
            is_mtp,
        )
        x = x + self.mlp(self.post_attention_layernorm(x, compute_dtype=x.dtype))
        return x

    def set_indexer_buffer(
        self, mode: Optional[str], buffer: Optional[_IndexerBuffer]
    ) -> None:
        self.self_attn.set_indexer_buffer(mode, buffer)


# ---------------------------------------------------------------------------
# TransformerBlockGLM52MTP
# ---------------------------------------------------------------------------


class TransformerBlockGLM52MTP(TransformerBlockGLM52):
    """MTP block: TransformerBlockGLM52 + MTP-specific enorm/hnorm/eh_proj/shared_head."""

    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        mla_absorb,
        *,
        checkpoint_prefix,
        indexer_impl,
    ):
        # MTP layers always own their indexer weights (full role).
        super().__init__(
            layer_id,
            args,
            cache_dict,
            attn_backend,
            op_impl,
            mla_absorb,
            checkpoint_prefix=checkpoint_prefix,
            indexer_impl=indexer_impl,
            indexer_role="full",
        )
        self.enorm = RMSNorm(
            args.dim,
            dtype=parse_dtype(args.rms_norm_dtype),
            eps=args.rms_norm_eps,
        )
        self.hnorm = RMSNorm(
            args.dim,
            dtype=parse_dtype(args.rms_norm_dtype),
            eps=args.rms_norm_eps,
        )
        self.eh_proj = torch.nn.Linear(args.dim * 2, args.dim, bias=False)
        self.max_batch_size_per_dp = ceil_div(
            int(get_global_args().infer.max_batch_size), get_dp_size()
        )
        self.shared_head = SharedHeadDeepSeekV3(args, self.max_batch_size_per_dp)
        if not args.mtp_tie_word_embeddings:
            self.embed_tokens = VocabParallelEmbedding(
                args.vocab_size,
                args.dim,
                decode_max_num_tokens=self.max_batch_size_per_dp,
            )

    @override
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        previous_hidden_states: torch.Tensor,
        is_mtp: bool = False,
    ):
        inputs_embeds = self.enorm(x)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        x = self.eh_proj(torch.cat([inputs_embeds, previous_hidden_states], dim=-1))
        x = x + self.self_attn(
            self.input_layernorm(x, compute_dtype=x.dtype),
            freqs_cis,
            is_mtp,
        )
        x = x + self.mlp(self.post_attention_layernorm(x, compute_dtype=x.dtype))
        return x


# ---------------------------------------------------------------------------
# TransformerGLM52
# ---------------------------------------------------------------------------


@register_model(ModelType.GLM_5_2)
class TransformerGLM52(TransformerDeepSeekV3):
    """GLM-5.2 transformer: DeepSeek-V3 backbone + shared indexer schedule."""

    @override
    def _init_layers(self, cache_dict: dict[str, KVCacheBase], attn_backend, op_impl):
        import logging
        import resource

        logger = logging.getLogger(__name__)
        self.layers = torch.nn.ModuleList()
        memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        indexer_types = self.params.indexer_types

        # Backbone buffer (written by "full" layers, read by "shared" layers).
        self._backbone_buf = _IndexerBuffer()
        # MTP buffer (reused across draft steps in spec-decode).
        self._mtp_buf = _IndexerBuffer()
        # When True, the MTP layer reads from the MTP buffer instead of computing.
        self._mtp_skip: bool = False

        for layer_id in range(self.local_begin_layer_id, self.local_end_layer_id):
            logger.debug(
                f"initing layer : {layer_id}  cpu memory usage: "
                f"{memory_usage / 1024**2} GB  gpu memory usage : RANK : "
                f"{torch.cuda.current_device()} "
                f"{torch.cuda.memory_allocated()/(1024**3)} GB"
            )
            if layer_id < self.params.n_layers:
                block = TransformerBlockGLM52(
                    layer_id,
                    self.params,
                    cache_dict,
                    attn_backend,
                    self.op_impl,
                    mla_absorb=self.mla_absorb,
                    checkpoint_prefix=f"layers.{layer_id}",
                    indexer_impl=self.indexer_backend,
                    indexer_role=indexer_types[layer_id],
                    indexer_buffer=self._backbone_buf,
                )
            else:
                block = TransformerBlockGLM52MTP(
                    layer_id,
                    self.params,
                    cache_dict,
                    attn_backend,
                    self.op_impl,
                    mla_absorb=self.mla_absorb,
                    checkpoint_prefix=f"layers.{layer_id}",
                    indexer_impl=self.indexer_backend,
                )

            self.layers.append(block)

        # Determine whether topk must be transmitted across adjacent PP stage
        # boundaries. Payload shape is declared by the receiving stage, so the
        # sending stage must pack topk exactly when the next stage will unpack it.
        self._cross_stage_send_topk: bool = False
        self._cross_stage_recv_topk: bool = False
        if self.pp_size > 1:
            global_indexer_types = indexer_types  # complete global role array
            n_backbone_layers = self.params.n_layers

            pp_layer_dist = compute_layer_dist_in_pp(self.pp_size)
            first_layer_id_of_each_stage = [0]
            for num_layers in pp_layer_dist:
                first_layer_id_of_each_stage.append(
                    first_layer_id_of_each_stage[-1] + num_layers
                )

            def stage_backbone_range(stage: int) -> tuple[int, int]:
                begin = first_layer_id_of_each_stage[stage]
                end = first_layer_id_of_each_stage[stage + 1]
                return begin, min(end, n_backbone_layers)

            def stage_needs_initial_topk(stage: int) -> bool:
                begin, end = stage_backbone_range(stage)
                return begin < end and global_indexer_types[begin] == "shared"

            self._cross_stage_recv_topk = (
                self.pp_stage > 0 and stage_needs_initial_topk(self.pp_stage)
            )
            self._cross_stage_send_topk = (
                self.pp_stage < self.pp_end_stage
                and stage_needs_initial_topk(self.pp_stage + 1)
            )

    def _clear_backbone_indexer_buffer(self) -> None:
        self._backbone_buf.topk = None
        self._backbone_buf.topk_page_table = None

    # ------------------------------------------------------------------
    # PP cross-stage topk transport: pack into / unpack from hidden state
    # ------------------------------------------------------------------

    def _payload_index_topk(self) -> int:
        """Return the top-k width actually produced by the Indexer."""
        return min(
            int(self.params.index_topk),
            int(get_global_args().infer.max_seq_len),
        )

    def get_pipeline_payload_shape(self, num_tokens: int) -> list[int]:
        """Declare hidden payload cols, including incoming topk when needed."""
        topk_bf16_cols = (
            self._payload_index_topk() * 2 if self._cross_stage_recv_topk else 0
        )
        return [num_tokens, self.params.dim + topk_bf16_cols]

    def get_pipeline_payload_dtype(self) -> torch.dtype:
        return torch.get_default_dtype()

    def _pack_topk(self, h: torch.Tensor, topk: torch.Tensor) -> torch.Tensor:
        # topk: [T, effective_index_topk] int32 — reinterpret as BF16 columns.
        # view() is zero-copy; contiguous() ensures the layout is compatible.
        assert topk.dtype == torch.int32
        target_topk = self._payload_index_topk()
        if topk.shape[-1] < target_topk:
            pad = topk.new_full((*topk.shape[:-1], target_topk - topk.shape[-1]), -1)
            topk = torch.cat([topk, pad], dim=-1)
        topk_bf16 = topk.contiguous().view(torch.bfloat16)
        return torch.cat([h, topk_bf16], dim=-1)

    def _unpack_topk(self, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        topk_bf16_cols = self._payload_index_topk() * 2
        h = packed[:, :-topk_bf16_cols].contiguous()
        topk = packed[:, -topk_bf16_cols:].contiguous().view(torch.int32)
        return h, topk

    # ------------------------------------------------------------------
    # Override layer loops to support CP split/gather and PP topk payloads.
    # ------------------------------------------------------------------

    @override
    @torch.inference_mode()
    def prefill_no_pipeline(
        self, tokens: torch.Tensor, output_token_offsets: torch.Tensor, **args
    ) -> torch.Tensor:
        self._clear_backbone_indexer_buffer()
        freqs_cis = self.prepare_freqs_cis()
        delta_total = self.cache_dict["main"].seq_len_delta.delta_total_len
        tokens, freqs_cis = self.cp_context.split_prefill(
            tokens,
            freqs_cis,
            hiddens=None,
            pp_stage=0,
            delta_total=delta_total,
        )

        if self.moe_impl is not None:
            self.moe_impl.prepare(TaskType.Prefill, int(tokens.shape[0]))
        h = self._pre_layers(tokens, **args)

        for layer in self.non_mtp_layers:
            h = layer(h, freqs_cis)

        if self.mtp_size > 1:
            self.mtp_prefill(
                x=self._pre_layers_mtp(tokens, **args),
                h=h,
                freqs_cis=freqs_cis,
            )
        return self.cp_context.gather(
            h,
            output_token_offsets,
            self._post_layers,
        )

    @override
    @torch.inference_mode()
    def decode_no_pipeline(self, tokens, freqs_cis: BatchedFreqsCis):
        self._clear_backbone_indexer_buffer()
        h = self._pre_layers(tokens)
        for layer in self.non_mtp_layers:
            h = layer(h, freqs_cis)
        if self.mtp_size > 1:
            self.update_mtp_hidden_states(
                self.norm(h, compute_dtype=h.dtype), is_mtp=True
            )
        h = self._post_layers(h)
        h = h.float()
        return h

    @override
    @torch.inference_mode()
    def mtp_decode_no_pipeline(self, tokens, freqs_cis: BatchedFreqsCis):
        h = self._pre_layers_mtp(tokens)
        mtp_layer = self.layers[-1]
        mtp_layer.set_indexer_buffer(
            "read" if self._mtp_skip else "write", self._mtp_buf
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
        self.update_mtp_hidden_states(h)
        h = self._post_layers_mtp(h)
        h = h.float()
        return h

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
            assert tokens is not None
            assert hiddens is None
            tokens, freqs_cis = self.cp_context.split_prefill(
                tokens,
                freqs_cis,
                hiddens,
                self.pp_stage,
                delta_total=delta_total,
            )
            batch_size = tokens.shape[0]
            h = self._pre_layers(tokens, **args)
        else:
            assert hiddens is not None
            # Incoming payload may include topk for this stage.
            if self._cross_stage_recv_topk:
                hiddens, topk = self._unpack_topk(hiddens)
                self._backbone_buf.topk = topk
            tokens, freqs_cis = self.cp_context.split_prefill(
                tokens,
                freqs_cis,
                hiddens,
                self.pp_stage,
                delta_total=delta_total,
            )
            batch_size = hiddens.shape[0]
            h = hiddens
            del hiddens

        if self.moe_impl is not None:
            self.moe_impl.prepare(TaskType.Prefill, batch_size)

        for layer in self.non_mtp_layers:
            h = layer(h, freqs_cis)

        if self.pp_stage == self.pp_end_stage:
            if self.mtp_size > 1:
                assert tokens is not None
                self.mtp_prefill(
                    x=self._pre_layers_mtp(tokens, **args),
                    h=h,
                    freqs_cis=freqs_cis,
                )
            seq_len_delta = (
                self.cache_dict["main"].seq_len_delta if self.pp_size > 1 else None
            )
            h = self.cp_context.gather(
                h,
                output_token_offsets,
                self._post_layers,
                pp_size=self.pp_size,
                pp_stage=self.pp_stage,
                seq_len_delta=seq_len_delta,
            )
        else:
            # Pack topk for the next PP stage.
            if self._cross_stage_send_topk:
                assert self._backbone_buf.topk is not None
                assert self._backbone_buf.topk.shape[0] == h.shape[0]
                h = self._pack_topk(h, self._backbone_buf.topk)
        return h

    @override
    @torch.inference_mode()
    def decode_pipeline(self, middle_state, freqs_cis: BatchedFreqsCis):
        self._clear_backbone_indexer_buffer()
        if self.pp_stage == 0:
            h = self._pre_layers(middle_state)
        else:
            # middle_state is the pipeline payload from the previous stage
            # (hidden states, optionally with packed topk appended).
            if self._cross_stage_recv_topk:
                middle_state, topk = self._unpack_topk(middle_state)
                self._backbone_buf.topk = topk
            h = middle_state
        for layer in self.non_mtp_layers:
            h = layer(h, freqs_cis)
        if self.pp_stage == self.pp_end_stage:
            if self.mtp_size > 1:
                self.update_mtp_hidden_states(
                    self.norm(h, compute_dtype=h.dtype), is_mtp=True
                )
            h = self._post_layers(h)
            h = h.float()
        else:
            # Pack topk for the next PP stage.
            if self._cross_stage_send_topk:
                assert self._backbone_buf.topk is not None
                assert self._backbone_buf.topk.shape[0] == h.shape[0]
                h = self._pack_topk(h, self._backbone_buf.topk)
        return h

    # ------------------------------------------------------------------
    # MTP skip toggle (used by spec-decode driver)
    # ------------------------------------------------------------------

    def set_mtp_skip_topk(self, skip: bool) -> None:
        self._mtp_skip = skip

    # ------------------------------------------------------------------
    # Checkpoint merging — route by has_local_indexer
    # ------------------------------------------------------------------

    @override
    def process_state_dict_for_merging_qkv(self, checkpoint: dict[str, Any]):
        def enable_callback(k: str):
            layer_id = get_layer_id_from_checkpoint_prefix(
                k, self.params.quant_config.rules
            )
            return QuantizationRegistry.allowed_merge_qkv(
                k,
                (
                    (
                        self.layers[layer_id].self_attn.mla_prologue_int8_partial
                        or self.layers[layer_id].self_attn.mla_prologue_int8_full
                    )
                    if layer_id > -1
                    else False
                ),
            )

        def _layer_id_from_key(k: str) -> int:
            match = re.search(r"layers\.(\d+)\.", k)
            return int(match.group(1)) if match else -1

        def enable_for_local_indexer_layer(k: str) -> bool:
            if not enable_callback(k):
                return False
            layer_id = _layer_id_from_key(k)
            if layer_id < 0 or layer_id >= len(self.layers):
                return True
            return self.layers[layer_id].self_attn.has_local_indexer

        def enable_for_shared_indexer_layer(k: str) -> bool:
            if not enable_callback(k):
                return False
            layer_id = _layer_id_from_key(k)
            if layer_id < 0 or layer_id >= len(self.layers):
                return False
            return not self.layers[layer_id].self_attn.has_local_indexer

        checkpoint = self.process_state_dict_for_merging_tensors(
            checkpoint,
            tgt_layer="wqkv_a_indexer_k",
            src_layers=["indexer_wk", "q_a_proj", "kv_a_proj_with_mqa"],
            enable_callback=enable_for_local_indexer_layer,
        )
        checkpoint = self.process_state_dict_for_merging_tensors(
            checkpoint,
            tgt_layer="wq_b_indexer_q_b",
            src_layers=["indexer_wq_b", "q_b_proj"],
            enable_callback=enable_for_local_indexer_layer,
        )
        checkpoint = self.process_state_dict_for_merging_tensors(
            checkpoint,
            tgt_layer="wqkv_a",
            src_layers=["q_a_proj", "kv_a_proj_with_mqa"],
            enable_callback=enable_for_shared_indexer_layer,
        )
        return checkpoint
