from .model import Attention, Transformer, TransformerBlock, RMSNorm
from torch import nn
from typing import Optional, List
import torch
import torch.nn.functional as F
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)


class AttentionLlama(Attention):
    def __init__(self, args, layer_id, cache, attn_backend):
        super().__init__(layer_id, cache, attn_backend)
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

    def _run_linear(self, x):
        return self.wq(x), self.wk(x), self.wv(x)

    def _run_output_linear(self, x):
        return self.wo(x)


class TransformerLlama(Transformer):
    def __init__(
        self,
        params,
        cache,
        pipeline_parallel_size,
        model_parallel_size,
        attn_backend,
        op_impl,
        merge_qkv_gate_up=False,
    ):
        super().__init__(
            params,
            cache,
            pipeline_parallel_size,
            model_parallel_size,
            attn_backend,
            op_impl,
        )
        self.op_impl = op_impl
        if merge_qkv_gate_up:
            raise NotImplementedError("merge_qkv_gate_up is not supported in llama")

    def _get_tensor_column_parallel_layer_names(self) -> List[str]:
        return ["wq", "wk", "wv", "w1", "w3", "output", "embed"]

    def _get_tensor_row_parallel_layer_names(self) -> List[str]:
        return ["wo", "w2"]

    def _get_pre_layer_prefixes(self) -> List[str]:
        return ["tok_embeddings."]

    def _get_post_layer_prefixes(self) -> List[str]:
        return ["output.", "norm."]

    def _get_layer_i_prefixes(self, i: int) -> List[str]:
        return [f"layers.{i}."]

    def _init_pre_layers(self):
        self.tok_embeddings = VocabParallelEmbedding(
            self.params.vocab_size, self.params.dim, init_method=lambda x: x
        )

    def _init_layers(self, cache, attn_backend, op_impl):
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(
                TransformerBlockLlama(
                    layer_id, self.params, cache, attn_backend, self.op_impl
                )
            )

    def _init_post_layers(self):
        self.norm = RMSNorm(self.params.dim, eps=self.params.norm_eps)
        self.output = ColumnParallelLinear(
            self.params.dim, self.params.vocab_size, bias=False, init_method=lambda x: x
        )

    def _pre_layers(self, h):
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
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlockLlama(TransformerBlock):
    def __init__(self, layer_id: int, args, cache, attn_backend, op_impl):
        super().__init__(layer_id, args, cache, attn_backend, op_impl)
        self.attention = AttentionLlama(args, layer_id, cache, attn_backend)
        self.feed_forward = FeedForwardLlama(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, varlens=None):
        h = self.attention(self.attention_norm(x), freqs_cis, varlens)
        h += x
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
