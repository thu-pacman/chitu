# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.2 model — extends DeepSeek-V3 with a shared top-k indexer schedule.

Backbone layers are tagged "full" or "shared" via config.json
``indexer_types``.  A "full" layer computes top-k and writes the result into a
per-request ``_IndexerBuffer``; "shared" layers read from that buffer without
recomputation.  Buffer routing is passed through forward kwargs
``share_buffer_to`` / ``read_buffer_from`` so no routing state sits on the
attention or indexer modules themselves.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import torch
from typing_extensions import override

from chitu.kv_cache import KVCacheBase
from chitu.kv_cache import PagedKVCacheAccessor
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.native_layout import NativeLayoutTensor
from chitu.ops import apply_rotary_pos_emb_partial, mla_prologue, a8_per_token_act_quant
from chitu.task_type import TaskType
from chitu.models.registry import ModelType, register_model
from chitu.quantization import QuantizationRegistry
from chitu.quantization.utils import get_layer_id_from_checkpoint_prefix
from chitu.models.model import TransformerBlock, RMSNorm
from chitu.models.model_deepseek_v3 import (
    Indexer,
    AttentionDeepSeekV3,
    TransformerBlockDeepSeekV3,
    TransformerBlockDeepSeekV3MTP,
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
from chitu.global_vars import get_global_args

# ---------------------------------------------------------------------------
# Shared buffer
# ---------------------------------------------------------------------------


class _IndexerBuffer:
    """Two-slot cell shared between a "full" layer and its downstream "shared" layers.

    ``topk``            — filled by :py:meth:`IndexerGLM52.forward`.
    ``topk_page_table`` — filled by
    :py:meth:`IndexerGLM52.build_decode_topk_page_table`.

    Both slots are overwritten on every full-layer call; shared layers read
    back the most recently produced value.
    """

    __slots__ = ("topk", "topk_page_table")

    def __init__(self) -> None:
        self.topk: Optional[torch.Tensor] = None
        self.topk_page_table: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# IndexerGLM52
# ---------------------------------------------------------------------------


class IndexerGLM52(Indexer):
    """Indexer subclass with buffer-routing kwargs passed explicitly per call.

    Buffer state is never stored on the module itself; it flows purely as
    function arguments so the module remains stateless between calls.
    """

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
        *,
        share_buffer_to: Optional[_IndexerBuffer] = None,
        read_buffer_from: Optional[_IndexerBuffer] = None,
    ) -> torch.Tensor:
        if read_buffer_from is not None:
            assert read_buffer_from.topk_page_table is not None
            return read_buffer_from.topk_page_table
        out = super().build_decode_topk_page_table(
            x,
            q,
            k,
            seq_len_delta,
            freqs_cis,
            is_causal,
            cache_accessor,
            source_page_table,
        )
        if share_buffer_to is not None:
            share_buffer_to.topk_page_table = out
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
        *,
        share_buffer_to: Optional[_IndexerBuffer] = None,
        read_buffer_from: Optional[_IndexerBuffer] = None,
    ):
        if read_buffer_from is not None:
            assert read_buffer_from.topk is not None
            return read_buffer_from.topk
        out = super().forward(
            x, q, k, seq_len_delta, freqs_cis, is_causal, cache_accessor
        )
        if share_buffer_to is not None:
            share_buffer_to.topk = out
        return out


# ---------------------------------------------------------------------------
# AttentionGLM52
# ---------------------------------------------------------------------------


class AttentionGLM52(AttentionDeepSeekV3):
    """MLA attention with GLM-5.2 shared-indexer buffer routing.

    Buffer kwargs ``share_buffer_to`` / ``read_buffer_from`` are accepted in
    ``forward`` and passed directly to the two ``IndexerGLM52`` call sites.
    No buffer state is stored on the module itself.
    """

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
        indexer_cache=None,
        indexer_impl=None,
        has_local_indexer: bool = True,
    ):
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
        # Upgrade the Indexer class in-place so it accepts buffer kwargs.
        # __class__ mutation is safe: all parameters, buffers, and sub-modules
        # are preserved; only the method resolution changes.
        if hasattr(self, "indexer") and self.indexer is not None:
            self.indexer.__class__ = IndexerGLM52
            if not has_local_indexer:
                # Shared layer: checkpoint has no k_norm/weights_proj weights.
                # Remove them so strict load_state_dict passes.
                del self.indexer.k_norm
                del self.indexer.weights_proj

    @override
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        is_mtp: bool = False,
        *,
        share_buffer_to: Optional[_IndexerBuffer] = None,
        read_buffer_from: Optional[_IndexerBuffer] = None,
    ):
        """Full override of AttentionDeepSeekV3.forward.

        Identical logic except the two indexer call sites pass
        share_buffer_to / read_buffer_from directly as kwargs.
        Shared layers (has_local_indexer=False) skip the indexer branch and
        run dense attention; the buffer topk is not yet wired into attn_backend.
        """
        seq_len_delta = self.cache.get_seq_len_delta(is_mtp)
        bs_seq, _ = x.size()

        if self.can_use_mla_prologue_torch_npu:

            def try_get_scale(module):
                if hasattr(module, "weight_scale"):
                    return module.weight_scale.view(module.out_features)
                return None

            if self.mla_prologue_int8_full:
                x_int8, scale_w_x = a8_per_token_act_quant(x.view(-1, x.shape[-1]))
                q_nope, q_pe, kv = mla_prologue(
                    x_int8,
                    self.q_a_proj.get_native_layout_weight(),
                    self.q_b_proj.get_native_layout_weight(),
                    self.kv_b_proj_absorb_1.get_native_layout_weight(),
                    self.kv_a_proj_with_mqa.get_native_layout_weight(),
                    self.q_a_layernorm.weight,
                    self.kv_a_layernorm.weight,
                    freqs_cis,
                    self.q_a_layernorm.eps,
                    self.kv_a_layernorm.eps,
                    dequant_scale_x=scale_w_x,
                    dequant_scale_q_a_proj=try_get_scale(self.q_a_proj),
                    dequant_scale_q_b_proj=try_get_scale(self.q_b_proj),
                    dequant_scale_kv_a_proj_with_mqa=try_get_scale(
                        self.kv_a_proj_with_mqa
                    ),
                    smooth_scales=None,
                    impl="torch_npu",
                )
            else:
                q_nope, q_pe, kv = mla_prologue(
                    x,
                    self.q_a_proj.get_native_layout_weight(),
                    self.q_b_proj.get_native_layout_weight(),
                    self.kv_b_proj_absorb_1.get_native_layout_weight(),
                    self.kv_a_proj_with_mqa.get_native_layout_weight(),
                    self.q_a_layernorm.weight,
                    self.kv_a_layernorm.weight,
                    freqs_cis,
                    self.q_a_layernorm.eps,
                    self.kv_a_layernorm.eps,
                    dequant_scale_q_b_proj=try_get_scale(self.q_b_proj),
                    smooth_scales=None,
                    impl="torch_npu",
                )
            x = self.attn_backend.mla(
                q_nope,
                q_pe,
                self.cache.get_accessor(self.layer_id, is_mtp),
                kv,
                seq_len_delta=seq_len_delta,
                causal=True,
                softmax_scale=self.softmax_scale,
            )
            x = self.kv_b_proj_absorb_2(x)

        else:
            assert self.q_lora_rank > 0
            indexer_k = None
            has_indexer_weights = self.index_topk is not None and self.has_local_indexer
            if self.merge_qkv:
                if not has_indexer_weights:
                    q_a_kv = self.wqkv_a(x)
                    q_a, kv = torch.split(
                        q_a_kv,
                        [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                        dim=-1,
                    )
                else:
                    q_a_kv_indexer_k = self.wqkv_a_indexer_k(x)
                    indexer_k, q_a, kv = torch.split(
                        q_a_kv_indexer_k,
                        [
                            self.index_head_dim,
                            self.q_lora_rank,
                            self.kv_lora_rank + self.qk_rope_head_dim,
                        ],
                        dim=-1,
                    )
            else:
                q_a = self.q_a_proj(x)
                kv = self.kv_a_proj_with_mqa(x)
                if has_indexer_weights:
                    indexer_k = self.indexer_wk(x)

            qr = self.q_a_layernorm(q_a, compute_dtype=q_a.dtype)

            indexer_q = None
            if self.merge_qkv and has_indexer_weights:
                q_indexer_q = self.wq_b_indexer_q_b(qr)
                indexer_q, q = torch.split(
                    q_indexer_q,
                    [
                        self.index_n_heads * self.index_head_dim,
                        q_indexer_q.shape[-1]
                        - self.index_n_heads * self.index_head_dim,
                    ],
                    dim=-1,
                )
            else:
                q = self.q_b_proj(qr)
                if has_indexer_weights:
                    indexer_q = self.indexer_wq_b(qr)

            q = q.view(bs_seq, self.n_local_heads, -1)
            kv = kv.view(bs_seq, 1, -1)

            q, kv, q_nope, q_pe, _, kv_lora, k_pe, _ = apply_rotary_pos_emb_partial(
                q,
                kv,
                freqs_cis,
                q_rotary_begin=q.shape[-1] - self.qk_rope_head_dim,
                k_rotary_begin=self.kv_lora_rank,
                rotary_type="interleaved",
            )

            if self.mla_absorb == "none":
                if isinstance(k_pe, NativeLayoutTensor):
                    k_pe = k_pe.convert_to_plain()

                kv = self.kv_b_proj(self.kv_a_layernorm(kv_lora))
                kv = kv.view(
                    bs_seq, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim
                )
                k_nope, v = torch.split(
                    kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
                )
                k = torch.cat(
                    [
                        k_nope.view(bs_seq, self.n_local_heads, self.qk_nope_head_dim),
                        k_pe.view(bs_seq, 1, self.qk_rope_head_dim).expand(
                            -1, self.n_local_heads, -1
                        ),
                    ],
                    dim=-1,
                )

                if has_indexer_weights:
                    # Full layer: compute and optionally write buffer.
                    assert self.indexer_cache is not None
                    topk_indices = self.indexer(
                        x,
                        indexer_q,
                        indexer_k,
                        seq_len_delta,
                        freqs_cis,
                        is_causal=True,
                        cache_accessor=self.indexer_cache.get_accessor(self.layer_id),
                        share_buffer_to=share_buffer_to,
                        read_buffer_from=read_buffer_from,
                    )
                else:
                    topk_indices = None

                x = self.attn_backend(
                    q,
                    self.cache.get_accessor(self.layer_id),
                    k,
                    v,
                    seq_len_delta=seq_len_delta,
                    causal=True,
                    softmax_scale=self.softmax_scale,
                )

            elif self.mla_absorb in ["absorb-without-precomp", "absorb"]:
                if self.mla_absorb == "absorb-without-precomp":
                    q_nope = self.kv_b_proj_absorb_1(q_nope)

                self.kv_a_layernorm(kv_lora, compute_dtype=kv.dtype, out=kv_lora)

                main_cache_accessor = self.cache.get_accessor(self.layer_id, is_mtp)
                topk_indices = None
                topk_page_table = None
                if has_indexer_weights:
                    # Full layer: compute and optionally write buffer.
                    assert self.indexer_cache is not None
                    indexer_cache_accessor = self.indexer_cache.get_accessor(
                        self.layer_id
                    )
                    if (
                        self.attn_backend.requires_sparse_decode_page_table()
                        and seq_len_delta.is_classic_decoding
                        and isinstance(main_cache_accessor, PagedKVCacheAccessor)
                    ):
                        topk_page_table = self.indexer.build_decode_topk_page_table(
                            x,
                            indexer_q,
                            indexer_k,
                            seq_len_delta,
                            freqs_cis,
                            is_causal=True,
                            cache_accessor=indexer_cache_accessor,
                            source_page_table=main_cache_accessor.block_table,
                            share_buffer_to=share_buffer_to,
                            read_buffer_from=read_buffer_from,
                        )
                    else:
                        topk_indices = self.indexer(
                            x,
                            indexer_q,
                            indexer_k,
                            seq_len_delta,
                            freqs_cis,
                            is_causal=True,
                            cache_accessor=indexer_cache_accessor,
                            share_buffer_to=share_buffer_to,
                            read_buffer_from=read_buffer_from,
                        )

                x = self.attn_backend.mla(
                    q_nope,
                    q_pe,
                    main_cache_accessor,
                    kv,
                    seq_len_delta=seq_len_delta,
                    causal=True,
                    softmax_scale=self.softmax_scale,
                    topk_indices=topk_indices,
                    topk_page_table=topk_page_table,
                )

                if self.mla_absorb == "absorb-without-precomp":
                    x = self.kv_b_proj_absorb_2(x)

            else:
                raise NotImplementedError(
                    f"MLA absorb mode {self.mla_absorb} not supported"
                )

        return self.o_proj(x.flatten(-2)).view(bs_seq, -1)


# ---------------------------------------------------------------------------
# TransformerBlockGLM52
# ---------------------------------------------------------------------------


class TransformerBlockGLM52(TransformerBlock):
    """Transformer block that passes buffer kwargs to AttentionGLM52.

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
        indexer_role: str = "full",
    ):
        super().__init__(
            layer_id, args, cache_dict, attn_backend=attn_backend, op_impl=op_impl
        )
        self.layer_id = layer_id
        has_local_indexer = indexer_role != "shared"
        self.self_attn = AttentionGLM52(
            args,
            layer_id,
            cache_dict["main"],
            attn_backend,
            op_impl=op_impl,
            mla_absorb=mla_absorb,
            checkpoint_prefix=f"{checkpoint_prefix}.self_attn",
            indexer_cache=cache_dict.get("indexer", None),
            indexer_impl=indexer_impl,
            has_local_indexer=has_local_indexer,
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
            dtype=(
                parse_dtype(args.rms_norm_dtype)
                if hasattr(args, "rms_norm_dtype")
                else None
            ),
            eps=getattr(args, "rms_norm_eps", 1e-6),
        )
        self.post_attention_layernorm = RMSNorm(
            args.dim,
            dtype=(
                parse_dtype(args.rms_norm_dtype)
                if hasattr(args, "rms_norm_dtype")
                else None
            ),
            eps=getattr(args, "rms_norm_eps", 1e-6),
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        is_mtp: bool = False,
        *,
        share_buffer_to: Optional[_IndexerBuffer] = None,
        read_buffer_from: Optional[_IndexerBuffer] = None,
    ):
        x = x + self.self_attn(
            self.input_layernorm(x, compute_dtype=x.dtype),
            freqs_cis,
            is_mtp,
            share_buffer_to=share_buffer_to,
            read_buffer_from=read_buffer_from,
        )
        x = x + self.mlp(self.post_attention_layernorm(x, compute_dtype=x.dtype))
        return x


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
            dtype=(
                parse_dtype(args.rms_norm_dtype)
                if hasattr(args, "rms_norm_dtype")
                else None
            ),
            eps=getattr(args, "rms_norm_eps", 1e-6),
        )
        self.hnorm = RMSNorm(
            args.dim,
            dtype=(
                parse_dtype(args.rms_norm_dtype)
                if hasattr(args, "rms_norm_dtype")
                else None
            ),
            eps=getattr(args, "rms_norm_eps", 1e-6),
        )
        self.eh_proj = torch.nn.Linear(args.dim * 2, args.dim, bias=False)
        self.max_batch_size_per_dp = ceil_div(
            int(getattr(get_global_args().infer, "max_batch_size", 1)), get_dp_size()
        )
        self.shared_head = SharedHeadDeepSeekV3(args, self.max_batch_size_per_dp)
        if not getattr(args, "mtp_tie_word_embeddings", False):
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
        *,
        share_buffer_to: Optional[_IndexerBuffer] = None,
        read_buffer_from: Optional[_IndexerBuffer] = None,
    ):
        inputs_embeds = self.enorm(x)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        x = self.eh_proj(torch.cat([inputs_embeds, previous_hidden_states], dim=-1))
        x = x + self.self_attn(
            self.input_layernorm(x, compute_dtype=x.dtype),
            freqs_cis,
            is_mtp,
            share_buffer_to=share_buffer_to,
            read_buffer_from=read_buffer_from,
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

        share_index_cache = bool(getattr(self.params, "share_index_cache", False))
        self._share_index_cache = share_index_cache

        # Map layer_id → role ("full" / "shared") for the buffer-routing helper.
        self._layer_indexer_roles: dict[int, str] = {}

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
            is_mtp_layer = self.mtp_size > 1 and layer_id >= self.params.n_layers

            if share_index_cache and not is_mtp_layer:
                indexer_role = self.params.indexer_types[layer_id]
            else:
                indexer_role = "full"

            self._layer_indexer_roles[layer_id] = indexer_role

            if not is_mtp_layer:
                block = TransformerBlockGLM52(
                    layer_id,
                    self.params,
                    cache_dict,
                    attn_backend,
                    self.op_impl,
                    mla_absorb=self.mla_absorb,
                    checkpoint_prefix=f"layers.{layer_id}",
                    indexer_impl=self.indexer_backend,
                    indexer_role=indexer_role,
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

    def _indexer_buffers_for(self, layer_id: int):
        """Return (share_buffer_to, read_buffer_from) for the given layer."""
        if not self._share_index_cache:
            return None, None
        role = self._layer_indexer_roles.get(layer_id, "full")
        if role == "full":
            return self._backbone_buf, None
        else:
            return None, self._backbone_buf

    # ------------------------------------------------------------------
    # Override all five layer-loop methods to inject buffer kwargs
    # ------------------------------------------------------------------

    @override
    @torch.inference_mode()
    def prefill_no_pipeline(
        self, tokens: torch.Tensor, output_token_offsets: torch.Tensor, **args
    ) -> torch.Tensor:
        freqs_cis = self.prepare_freqs_cis()
        if self.moe_impl is not None:
            self.moe_impl.prepare(TaskType.Prefill, int(tokens.shape[0]))
        h = self._pre_layers(tokens, **args)

        for it, layer in enumerate(self.non_mtp_layers):
            layer_id = self.local_begin_layer_id + it
            share_buf, read_buf = self._indexer_buffers_for(layer_id)
            h = layer(
                h, freqs_cis, share_buffer_to=share_buf, read_buffer_from=read_buf
            )

        if self.mtp_size > 1:
            self.mtp_prefill(
                x=self._pre_layers_mtp(tokens, **args),
                h=h,
                freqs_cis=freqs_cis,
            )
        h = h[output_token_offsets]
        h = self._post_layers(h)
        h = h.float()
        return h

    @override
    @torch.inference_mode()
    def decode_no_pipeline(self, tokens, freqs_cis: BatchedFreqsCis):
        h = self._pre_layers(tokens)
        for it, layer in enumerate(self.non_mtp_layers):
            layer_id = self.local_begin_layer_id + it
            share_buf, read_buf = self._indexer_buffers_for(layer_id)
            h = layer(
                h, freqs_cis, share_buffer_to=share_buf, read_buffer_from=read_buf
            )
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
        share_buf = None if self._mtp_skip else self._mtp_buf
        read_buf = self._mtp_buf if self._mtp_skip else None
        h = self.layers[-1](
            h,
            freqs_cis,
            self.read_mtp_hidden_states(),
            is_mtp=True,
            share_buffer_to=share_buf,
            read_buffer_from=read_buf,
        )
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
        freqs_cis = self.prepare_freqs_cis()

        if self.pp_stage == 0:
            batch_size = tokens.shape[0]
            assert hiddens is None
            h = self._pre_layers(tokens, **args)
        else:
            batch_size = hiddens.shape[0]
            h = hiddens
            del hiddens

        if self.moe_impl is not None:
            self.moe_impl.prepare(TaskType.Prefill, batch_size)

        for it, layer in enumerate(self.non_mtp_layers):
            layer_id = self.local_begin_layer_id + it
            share_buf, read_buf = self._indexer_buffers_for(layer_id)
            h = layer(
                h, freqs_cis, share_buffer_to=share_buf, read_buffer_from=read_buf
            )

        if self.pp_stage == self.pp_end_stage:
            if self.mtp_size > 1:
                assert tokens is not None
                self.mtp_prefill(
                    x=self._pre_layers_mtp(tokens, **args),
                    h=h,
                    freqs_cis=freqs_cis,
                )
            h = h[output_token_offsets]
            h = self._post_layers(h)
            h = h.float()
        return h

    @override
    @torch.inference_mode()
    def decode_pipeline(self, tokens, freqs_cis: BatchedFreqsCis):
        if self.pp_stage == 0:
            h = self._pre_layers(tokens)
        else:
            h = tokens
        for it, layer in enumerate(self.non_mtp_layers):
            layer_id = self.local_begin_layer_id + it
            share_buf, read_buf = self._indexer_buffers_for(layer_id)
            h = layer(
                h, freqs_cis, share_buffer_to=share_buf, read_buffer_from=read_buf
            )
        if self.pp_stage == self.pp_end_stage:
            if self.mtp_size > 1:
                self.update_mtp_hidden_states(
                    self.norm(h, compute_dtype=h.dtype), is_mtp=True
                )
            h = self._post_layers(h)
            h = h.float()
        return h

    # ------------------------------------------------------------------
    # MTP skip toggle (used by spec-decode driver)
    # ------------------------------------------------------------------

    def set_mtp_skip_topk(self, skip: bool) -> None:
        if self.mtp_size <= 1:
            return
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
            return getattr(self.layers[layer_id].self_attn, "has_local_indexer", True)

        def enable_for_shared_indexer_layer(k: str) -> bool:
            if not enable_callback(k):
                return False
            layer_id = _layer_id_from_key(k)
            if layer_id < 0 or layer_id >= len(self.layers):
                return False
            return not getattr(
                self.layers[layer_id].self_attn, "has_local_indexer", True
            )

        if not hasattr(self.params, "index_topk"):
            return self.process_state_dict_for_merging_tensors(
                checkpoint,
                tgt_layer="wqkv_a",
                src_layers=["q_a_proj", "kv_a_proj_with_mqa"],
                enable_callback=enable_callback,
            )
        else:
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
