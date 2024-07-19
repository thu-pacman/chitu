from .model import Attention, Transformer, TransformerBlock, RMSNorm
from torch import nn
from typing import Optional
import torch
import torch.nn.functional as F
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

from logging import getLogger

logger = getLogger(__name__)


class AttentionQwen(Attention):
    def __init__(self, args, layer_id, cache):
        super().__init__(layer_id, cache)
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.q_proj = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=True,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.k_proj = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=True,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.v_proj = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=True,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.o_proj = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

    def _run_linear(self, x):
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def _run_output_linear(self, x):
        return self.o_proj(x)


class FeedForwardQwen(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.down_proj = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.up_proj = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlockQwen(TransformerBlock):
    def __init__(self, layer_id: int, args, cache):
        super().__init__(layer_id, args)
        self.self_attn = AttentionQwen(args, layer_id, cache)
        self.mlp = FeedForwardQwen(dim=args.dim, hidden_dim=args.intermediate_dim)
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, varlens=None):
        # h = self.input_layernorm(x)
        h = self.self_attn(self.input_layernorm(x), freqs_cis, varlens)
        h += x
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class TransformerQwen(Transformer):
    def __init__(self, params, cache, pipeline_parallel_size, model_parallel_size):
        super().__init__(params, cache, pipeline_parallel_size, model_parallel_size)

    def _init_pre_layers(self):
        self.embed_tokens = VocabParallelEmbedding(
            self.params.vocab_size, self.params.dim, init_method=lambda x: x
        )

    def _init_layers(self, cache):
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlockQwen(layer_id, self.params, cache))

    def _init_post_layers(self):
        self.norm = RMSNorm(self.params.dim, eps=self.params.norm_eps)
        self.lm_head = ColumnParallelLinear(
            self.params.dim, self.params.vocab_size, bias=False, init_method=lambda x: x
        )

    def _pre_layers(self, h):
        return self.embed_tokens(h)

    def _post_layers(self, h):
        h = self.norm(h)
        h = self.lm_head(h)
        return h
