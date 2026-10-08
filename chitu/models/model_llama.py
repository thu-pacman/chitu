# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from chitu.attn_backend import AttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.kv_cache import KVCacheBase
from chitu.models.model import Attention, RMSNorm, Transformer, TransformerBlock
from chitu.models.registry import ModelType, register_model
from chitu.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    get_tp_size,
)

# Fuse the block's residual-add + the following rms_norm into the vendor GCU single-kernel
# `_C.fused_add_rms_norm` (residual += x; x = rms_norm(residual), in place). Self-gating: the
# op only exists on the enflame GCU stack, so `_fused_add_rms_norm` returns None elsewhere and
# the plain add+norm path runs.
#
# Default OFF: measured 0 wall-clock gain (fusing 1 add+norm pair/layer ~= 1ms/token, under the
# run noise). Under cuda graph these small ops are nearly free — the decode cost is the FP8 GEMV
# (weight reads, near the HBM-bandwidth floor), not norm/add. Set CHITU_GCU_FUSED_NORM=1 to try.
_GCU_FUSED_NORM = os.environ.get("CHITU_GCU_FUSED_NORM", "0") == "1"
_gcu_fused_norm_op = None


def _fused_add_rms_norm(inp, residual, norm):
    """If the vendor fused kernel is available, do `residual += inp; inp = rms_norm(residual)`
    in place and return (normed_view=inp, sum_view=residual). Returns None to signal the caller
    to use the unfused path. `inp` and `residual` MUST be contiguous (mutated in place)."""
    global _gcu_fused_norm_op
    if not _GCU_FUSED_NORM:
        return None
    if _gcu_fused_norm_op is None:
        try:
            from chitu.import_utils import ensure_gcu_c_ops

            ensure_gcu_c_ops()
            _gcu_fused_norm_op = getattr(torch.ops._C, "fused_add_rms_norm", None)
        except Exception:
            _gcu_fused_norm_op = False
    if not _gcu_fused_norm_op or inp.device.type == "meta":
        return None
    _gcu_fused_norm_op(inp, residual, norm.weight, norm.eps)
    return inp, residual


class AttentionLlama(Attention):
    def __init__(self, args, layer_id, cache, attn_backend, checkpoint_prefix):
        super().__init__(layer_id, cache, attn_backend)
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        tp_size = get_tp_size()
        self.n_local_heads = args.n_heads // tp_size
        self.n_local_kv_heads = self.n_kv_heads // tp_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            has_bias=False,
            gather_output=False,
            checkpoint_prefix=f"{checkpoint_prefix}.wq",
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            has_bias=False,
            gather_output=False,
            checkpoint_prefix=f"{checkpoint_prefix}.wk",
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            has_bias=False,
            gather_output=False,
            checkpoint_prefix=f"{checkpoint_prefix}.wv",
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            has_bias=False,
            input_is_parallel=True,
            checkpoint_prefix=f"{checkpoint_prefix}.wo",
        )

    def _run_linear(self, x):
        return self.wq(x), self.wk(x), self.wv(x)

    def _run_output_linear(self, x):
        return self.wo(x)


@register_model(ModelType.LLAMA)
class TransformerLlama(Transformer):
    def __init__(
        self,
        params,
        cache_dict: dict[str, KVCacheBase],
        *,
        max_position_embeddings: int,
        attn_backend: AttnBackend,
        op_impl: str,
        merge_qkv_gate_up: bool = False,
        **kvargs,
    ):
        super().__init__(
            params,
            cache_dict,
            max_position_embeddings=max_position_embeddings,
            attn_backend=attn_backend,
            op_impl=op_impl,
            **kvargs,
        )
        self.op_impl = op_impl
        if merge_qkv_gate_up:
            raise NotImplementedError("merge_qkv_gate_up is not supported in llama")

    def _get_tensor_column_parallel_layer_names(self) -> list[str]:
        return ["wq", "wk", "wv", "w1", "w3", "output", "embed"]

    def _get_tensor_row_parallel_layer_names(self) -> list[str]:
        return ["wo", "w2"]

    def _get_pre_layer_prefixes(self) -> list[str]:
        return ["tok_embeddings."]

    def _get_post_layer_prefixes(self) -> list[str]:
        return ["output.", "norm."]

    def _get_layer_i_prefixes(self, i: int) -> list[str]:
        return [f"layers.{i}."]

    def _init_pre_layers(self):
        self.tok_embeddings = VocabParallelEmbedding(
            self.params.vocab_size, self.params.dim
        )

    def _init_layers(self, cache_dict: dict[str, KVCacheBase], attn_backend, op_impl):
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.local_begin_layer_id, self.local_end_layer_id):
            self.layers.append(
                TransformerBlockLlama(
                    layer_id,
                    self.params,
                    cache_dict,
                    attn_backend,
                    self.op_impl,
                    checkpoint_prefix=f"layers.{layer_id}",
                )
            )

    def _init_post_layers(self):
        self.norm = RMSNorm(self.params.dim, eps=self.params.norm_eps)
        self.output = ColumnParallelLinear(
            self.params.dim,
            self.params.vocab_size,
            has_bias=False,
            checkpoint_prefix=f"output",
        )

    def _pre_layers(self, h, **args):
        return self.tok_embeddings(h)

    def _post_layers(self, h):
        """NOTE: _post_layers is assumed to be a token-wise computation"""
        h = self.norm(h)
        h = self.output(h)
        return h


class FeedForwardLlama(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        checkpoint_prefix: str,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim,
            hidden_dim,
            has_bias=False,
            gather_output=False,
            checkpoint_prefix=f"{checkpoint_prefix}.w1",
        )
        self.w2 = RowParallelLinear(
            hidden_dim,
            dim,
            has_bias=False,
            input_is_parallel=True,
            checkpoint_prefix=f"{checkpoint_prefix}.w2",
        )
        self.w3 = ColumnParallelLinear(
            dim,
            hidden_dim,
            has_bias=False,
            gather_output=False,
            checkpoint_prefix=f"{checkpoint_prefix}.w3",
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlockLlama(TransformerBlock):
    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        checkpoint_prefix,
    ):
        super().__init__(layer_id, args, cache_dict, attn_backend, op_impl)
        self.attention = AttentionLlama(
            args,
            layer_id,
            cache_dict["main"],
            attn_backend,
            f"{checkpoint_prefix}.attention",
        )
        self.feed_forward = FeedForwardLlama(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            checkpoint_prefix=f"{checkpoint_prefix}.feed_forward",
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: BatchedFreqsCis):
        h_attn = self.attention(self.attention_norm(x), freqs_cis)
        # Fuse the post-attention residual add with ffn_norm: one vendor kernel does
        # `h = h_attn + x; f = ffn_norm(h)` instead of an add plus a separate norm. The unfused
        # path (below) is used when the fused kernel is unavailable.
        fused = _fused_add_rms_norm(h_attn.contiguous(), x.contiguous(), self.ffn_norm)
        if fused is not None:
            f, h = fused  # f = ffn_norm(h_attn + x), h = h_attn + x
            return h + self.feed_forward(f)
        h = h_attn + x
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
