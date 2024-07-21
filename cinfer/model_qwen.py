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
import flash_attn
from xformers.ops import fmha
from logging import getLogger

from .ops import apply_rotary_pos_emb_triton

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

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=2048,
            base=10000.0,
        )

    def _run_linear(self, x):
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def _run_output_linear(self, x):
        return self.o_proj(x)

    def prefill_forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        varlens,
    ):
        bs_seq, _ = x.shape
        xq, xk, xv = self._run_linear(x)
        xq = xq.view(bs_seq, self.n_local_heads, self.head_dim)
        xk = xk.view(bs_seq, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs_seq, self.n_local_kv_heads, self.head_dim)
        # torch.cuda.synchronize()
        cos, sin = self.rotary_emb(xv, seq_len=bs_seq)
        # torch.cuda.synchronize()
        xq, xk = apply_rotary_pos_emb_torch(
            xq, xk, cos, sin, position_ids=torch.arange(bs_seq, device=x.device)
        )
        # torch.cuda.synchronize()
        self.cache.finalize_cache_bylayer_prefill(
            xk, xv, self.cache.curr_req_ids, self.cache.curr_varlens, self.layer_id
        )
        output = flash_attn.flash_attn_varlen_func(
            xq,
            xk,
            xv,
            varlens.prefix_lens,
            varlens.prefix_lens,
            varlens.max_len,
            varlens.max_len,
            causal=True,
        ).view(bs_seq, -1)
        return self._run_output_linear(output)

    def decode_forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        bsz, seqlen, _ = x.shape
        assert seqlen == 1, "decode_forward only supports single token decoding"
        xq, xk, xv = self._run_linear(x)

        xq = xq.view(-1, self.n_local_heads, self.head_dim)
        xk = xk.view(-1, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(-1, self.n_local_kv_heads, self.head_dim)

        # torch.cuda.synchronize()
        cos, sin = self.rotary_emb(xv, seq_len=max(self.cache.curr_seq_lens) + 1)
        # torch.cuda.synchronize()
        xq, xk = apply_rotary_pos_emb(
            xq,
            xk,
            cos,
            sin,
            position_ids=self.cache.curr_seq_lens_gpu,
        )
        # logger.warning(f"cos shape {cos.shape} {self.cache.curr_seq_lens} {cos.dtype}")
        # torch.cuda.synchronize()

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        cache = self.cache.update_cache_decode(xk, xv, self.layer_id)
        cache_k = cache[0]
        cache_v = cache[1]
        max_seq_len = cache.shape[2]

        if self.n_local_heads != self.n_local_kv_heads:
            group_size = self.n_local_heads // self.n_local_kv_heads
            assert group_size > 1
            xq = xq.view(bsz, seqlen, self.n_local_kv_heads, group_size, self.head_dim)
            cache_k = cache_k.view(
                bsz, max_seq_len, self.n_local_kv_heads, 1, self.head_dim
            ).expand(bsz, max_seq_len, self.n_local_kv_heads, group_size, self.head_dim)
            cache_v = cache_v.view(
                bsz, max_seq_len, self.n_local_kv_heads, 1, self.head_dim
            ).expand(bsz, max_seq_len, self.n_local_kv_heads, group_size, self.head_dim)
        output = fmha.memory_efficient_attention_forward(xq, cache_k, cache_v).view(
            bsz, seqlen, -1
        )
        return self._run_output_linear(output)

    def decode_forward_paged(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        bsz, seqlen, _ = x.shape
        assert seqlen == 1, "decode_forward only supports single token decoding"
        xq, xk, xv = self._run_linear(x)

        xq = xq.view(-1, self.n_local_heads, self.head_dim)
        xk = xk.view(-1, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(-1, self.n_local_kv_heads, self.head_dim)

        # xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        torch.cuda.synchronize()
        cos, sin = self.rotary_emb(xv, seq_len=max(self.cache.curr_seq_lens) + 1)
        # logger.warning(f"cos shape {cos.shape} {self.cache.curr_seq_lens}")
        xq, xk = apply_rotary_pos_emb(
            xq,
            xk,
            cos,
            sin,
            position_ids=self.cache.curr_seq_lens_gpu,
        )
        torch.cuda.synchronize()

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        self.cache.prepare_block_table_for_decode(
            self.cache.curr_req_ids, self.layer_id
        )
        block_table = self.cache.get_gpu_block_table(
            self.cache.curr_req_ids, self.layer_id
        )
        cache_seqlens = self.cache.get_gpu_seq_lens(self.cache.curr_req_ids)
        paged_k_cache, paged_v_cache = self.cache.get_paged_kv_cache()
        output = flash_attn.flash_attn_with_kvcache(
            xq,
            paged_k_cache,
            paged_v_cache,
            xk,
            xv,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
        ).view(bsz, seqlen, -1)
        return self._run_output_linear(output)


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


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_torch(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    # cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    # sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)

    # cossin = torch.stack((cos, sin)).contiguous()
    # torch.cuda.synchronize()
    # ops.rotary_embedding(
    #     positions=position_ids,
    #     query=q,
    #     key=k,
    #     head_size=q.shape[-1],
    #     cos_sin_cache=cossin,
    #     is_neox=False,
    # )
    q_embed = apply_rotary_pos_emb_triton(q, cos, sin, position_ids)
    k_embed = apply_rotary_pos_emb_triton(k, cos, sin, position_ids)
    # torch.cuda.synchronize()

    return q_embed, k_embed
    # return q, k
