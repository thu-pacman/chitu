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
from logging import getLogger

from .ops import apply_rotary_pos_emb_triton

logger = getLogger(__name__)


class AttentionHFLlama(Attention):
    def __init__(
        self,
        args,
        layer_id,
        cache,
        attn_backend,
        rotary_type="default",
        op_impl: str = "torch",
        merge_qkv: bool = True,
    ):
        super().__init__(layer_id, cache, attn_backend)
        self.rotary_type = rotary_type
        self.op_impl = op_impl
        self.merge_qkv = merge_qkv

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # Do a parallel + fused linear projection. Goals:
        # - Parallelization should be among the kv_heads dim, so there is no communication.
        # - Outputs from q_proj, k_proj, v_proj should be contiguous in memory.
        #
        # Therefore, the projected shape should be [model_parallel_size, self.n_rep + 2, self.n_local_kv_heads, self.head_dim]

        qkv_has_bias = args.qkv_has_bias if hasattr(args, "qkv_has_bias") else True
        o_has_bias = args.o_has_bias if hasattr(args, "o_has_bias") else False

        qkv_proj_linear = o_proj_linear = get_linear_layout_contig_x_contig_y(op_impl)
        if merge_qkv:
            self.qkv_proj = ColumnParallelLinear(
                args.dim,
                (args.n_heads + 2 * self.n_kv_heads) * self.head_dim,
                bias=qkv_has_bias,
                gather_output=False,
                init_method=lambda x: x,
                linear_op=qkv_proj_linear,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                args.dim,
                args.n_heads * self.head_dim,
                bias=qkv_has_bias,
                gather_output=False,
                init_method=lambda x: x,
                linear_op=qkv_proj_linear,
            )
            self.k_proj = ColumnParallelLinear(
                args.dim,
                self.n_kv_heads * self.head_dim,
                bias=qkv_has_bias,
                gather_output=False,
                init_method=lambda x: x,
                linear_op=qkv_proj_linear,
            )
            self.v_proj = ColumnParallelLinear(
                args.dim,
                self.n_kv_heads * self.head_dim,
                bias=qkv_has_bias,
                gather_output=False,
                init_method=lambda x: x,
                linear_op=qkv_proj_linear,
            )
        self.o_proj = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=o_has_bias,
            input_is_parallel=True,
            init_method=lambda x: x,
            linear_op=o_proj_linear,
        )
        self.rotary_emb = RotaryEmbeddingHFLlama(
            self.head_dim // 2 if rotary_type == "glm4" else self.head_dim,
            max_position_embeddings=args.max_position_embeddings,
            base=args.rope_theta,
        )

    def _run_linear(self, x):
        if self.op_impl == "muxi_custom_kernel":
            x_shape = x.shape
            x = x.reshape(-1, x.shape[-1])
            n = x.shape[0]
            if n > 1:
                x_paded = torch.zeros(((n + 15) & ~15, x.shape[1]), device=x.device)
                x_paded[: x.shape[0], :] = x
                x = x_paded
        if self.merge_qkv:
            qkv = self.qkv_proj(x)
            if self.op_impl == "muxi_custom_kernel":
                qkv = qkv[:n, :]
                qkv = qkv.reshape(x_shape[:-1] + (qkv.shape[-1],))
            q, k, v = qkv.split(
                [
                    self.n_local_heads * self.head_dim,
                    self.n_local_kv_heads * self.head_dim,
                    self.n_local_kv_heads * self.head_dim,
                ],
                dim=-1,
            )
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            if self.op_impl == "muxi_custom_kernel":
                q = q[:n, :]
                k = k[:n, :]
                v = v[:n, :]
                q = q.reshape(x_shape[:-1] + (q.shape[-1],))
                k = k.reshape(x_shape[:-1] + (k.shape[-1],))
                v = v.reshape(x_shape[:-1] + (v.shape[-1],))
        return q, k, v

    def _run_output_linear(self, x):
        if self.op_impl == "muxi_custom_kernel":
            x_shape = x.shape
            x = x.reshape(-1, x.shape[-1])
            n = x.shape[0]
            if n > 1:
                x_paded = torch.zeros(((n + 15) & ~15, x.shape[1]), device=x.device)
                x_paded[: x.shape[0], :] = x
                x = x_paded
        y = self.o_proj(x)
        if self.op_impl == "muxi_custom_kernel":
            y = y[:n, :]
            y = y.reshape(x_shape[:-1] + (y.shape[-1],))
        return y

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
        xq, xk = apply_rotary_pos_emb(
            xq,
            xk,
            cos,
            sin,
            position_ids=self.cache.curr_varlens.position_ids,
            rotary_type=self.rotary_type,
        )
        # torch.cuda.synchronize()
        self.cache.finalize_cache_bylayer_prefill(
            xk, xv, self.cache.curr_req_ids, self.cache.curr_varlens, self.layer_id
        )
        output = self.attn_backend.attn_varlen_func(
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
            position_ids=self.cache.get_gpu_seq_lens(),
            rotary_type=self.rotary_type,
        )
        # logger.warning(f"cos shape {cos.shape} {self.cache.curr_seq_lens} {cos.dtype}")
        # torch.cuda.synchronize()

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        cache = self.cache.get_cache_decode(self.layer_id)
        cache_k = cache[0]
        cache_v = cache[1]
        cache_seqlens = self.cache.get_gpu_seq_lens()
        output = self.attn_backend.attn_with_kvcache(
            xq,
            cache_k,
            cache_v,
            xk,
            xv,
            cache_seqlens=cache_seqlens,
        ).view(bsz, seqlen, -1)
        return self._run_output_linear(output)

    def decode_forward_paged(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        bsz, seqlen, _ = x.shape
        assert seqlen == 1, "decode_forward only supports single token decoding"
        xq, xk, xv = self._run_linear(x)

        xq = xq.view(-1, self.n_local_heads, self.head_dim)
        xk = xk.view(-1, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(-1, self.n_local_kv_heads, self.head_dim)

        cos, sin = self.rotary_emb(xv, seq_len=max(self.cache.curr_seq_lens) + 1)
        xq, xk = apply_rotary_pos_emb(
            xq,
            xk,
            cos,
            sin,
            position_ids=self.cache.get_gpu_seq_lens(),
            rotary_type=self.rotary_type,
        )

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        block_table = self.cache.get_gpu_block_table(self.layer_id)
        cache_seqlens = self.cache.get_gpu_seq_lens()
        paged_k_cache, paged_v_cache = self.cache.get_paged_kv_cache()
        output = self.attn_backend.attn_with_kvcache(
            xq,
            paged_k_cache,
            paged_v_cache,
            xk,
            xv,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
        ).view(bsz, seqlen, -1)
        return self._run_output_linear(output)


class FeedForwardHFLlama(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int, op_impl: str, merge_gate_up: bool = True
    ):
        super().__init__()
        self.op_impl = op_impl
        self.merge_gate_up = merge_gate_up

        # Do a parallel + fused linear projection, while ensuring outputs from gate_proj and up_proj are contiguous in memory.
        # Therefore, the projected shape is [model_parallel_size, 2 * hidden_dim]

        gate_up_proj_linear = get_linear_layout_contig_x_native_y(op_impl)
        down_proj_linear = get_linear_layout_native_x_contig_y(op_impl)
        if merge_gate_up:
            self.gate_up_proj = ColumnParallelLinear(
                dim,
                hidden_dim * 2,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
                linear_op=gate_up_proj_linear,
            )
        else:
            self.gate_proj = ColumnParallelLinear(
                dim,
                hidden_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
                linear_op=gate_up_proj_linear,
            )
            self.up_proj = ColumnParallelLinear(
                dim,
                hidden_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
                linear_op=gate_up_proj_linear,
            )
        self.down_proj = RowParallelLinear(
            hidden_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
            linear_op=down_proj_linear,
        )

    def forward(self, x):
        if self.op_impl == "muxi_custom_kernel":
            x_shape = x.shape
            x = x.reshape(-1, x_shape[-1])
            n = x.shape[0]
            if n > 1:
                x_paded = torch.zeros(((n + 15) & ~15, x.shape[1]), device=x.device)
                x_paded[: x.shape[0], :] = x
                x = x_paded
        if self.merge_gate_up:
            gate_up_out = self.gate_up_proj(x)
            if self.op_impl == "muxi_custom_kernel":
                gate_out = gate_up_out[: gate_up_out.shape[0] // 2]
                up_out = gate_up_out[gate_up_out.shape[0] // 2 :]
            else:
                gate_out = gate_up_out[..., : gate_up_out.shape[-1] // 2]
                up_out = gate_up_out[..., gate_up_out.shape[-1] // 2 :]
        else:
            gate_out = self.gate_proj(x)
            up_out = self.up_proj(x)
        y = self.down_proj(F.silu(gate_out) * up_out)
        if self.op_impl == "muxi_custom_kernel":
            y = y[:n, :]
            y = y.reshape(x_shape[:-1] + (y.shape[-1],))
        return y


class TransformerBlockHFLlama(TransformerBlock):
    def __init__(
        self,
        layer_id: int,
        args,
        cache,
        attn_backend,
        op_impl,
        rotary_type="default",
        mlp_type=FeedForwardHFLlama,
        merge_qkv_gate_up=True,
    ):
        super().__init__(layer_id, args, cache, attn_backend, op_impl)
        self.self_attn = AttentionHFLlama(
            args,
            layer_id,
            cache,
            attn_backend,
            rotary_type=rotary_type,
            op_impl=op_impl,
            merge_qkv=merge_qkv_gate_up,
        )
        self.mlp = mlp_type(
            dim=args.dim,
            hidden_dim=args.intermediate_dim,
            op_impl=op_impl,
            merge_gate_up=merge_qkv_gate_up,
        )
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, varlens=None):
        h = self.self_attn(self.input_layernorm(x), freqs_cis, varlens)
        h += x
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class TransformerHFLlama(Transformer):
    def __init__(
        self,
        params,
        cache,
        pipeline_parallel_size,
        model_parallel_size,
        attn_backend,
        op_impl,
        rotary_type="default",
        layer_type=TransformerBlockHFLlama,
        merge_qkv_gate_up=True,
    ):
        self.rotary_type = rotary_type
        self.layer_type = layer_type
        self.merge_qkv_gate_up = merge_qkv_gate_up
        super().__init__(
            params,
            cache,
            pipeline_parallel_size,
            model_parallel_size,
            attn_backend,
            op_impl,
        )

    def _init_pre_layers(self):
        self.embed_tokens = VocabParallelEmbedding(
            self.params.vocab_size, self.params.dim, init_method=lambda x: x
        )

    def _init_layers(self, cache, attn_backend, op_impl):
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(
                self.layer_type(
                    layer_id,
                    self.params,
                    cache,
                    attn_backend=attn_backend,
                    op_impl=op_impl,
                    rotary_type=self.rotary_type,
                    merge_qkv_gate_up=self.merge_qkv_gate_up,
                )
            )

    def _init_post_layers(self):
        self.norm = RMSNorm(self.params.dim, eps=self.params.norm_eps)
        self.lm_head = ColumnParallelLinear(
            self.params.dim, self.params.vocab_size, bias=False, init_method=lambda x: x
        )

    def _pre_layers(self, h):
        return self.embed_tokens(h)

    def _post_layers(self, h):
        """NOTE: _post_layers is assumed to be a token-wise computation"""
        h = self.norm(h)
        h = self.lm_head(h)
        return h


class RotaryEmbeddingHFLlama(nn.Module):
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


def apply_rotary_pos_emb_torch(
    q, k, cos, sin, position_ids, unsqueeze_dim=1, rotary_type="default"
):
    if rotary_type == "default":
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    elif rotary_type == "glm4":
        # TODO: Now we transpose q and k, do the rotary, and transpose back.
        # Maybe we can transpose cos and sin just once instead of transposing q and k.
        q, q_pass = q[..., :64], q[..., 64:]
        k, k_pass = k[..., :64], k[..., 64:]
        q = (
            q.reshape(q.shape[0], q.shape[1], q.shape[2] // 2, 2)
            .permute(0, 1, 3, 2)
            .reshape(q.shape[0], q.shape[1], q.shape[2])
        )
        k = (
            k.reshape(k.shape[0], k.shape[1], k.shape[2] // 2, 2)
            .permute(0, 1, 3, 2)
            .reshape(k.shape[0], k.shape[1], k.shape[2])
        )
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        q_embed = (
            q_embed.reshape(
                q_embed.shape[0], q_embed.shape[1], 2, q_embed.shape[2] // 2
            )
            .permute(0, 1, 3, 2)
            .reshape(q_embed.shape[0], q_embed.shape[1], q_embed.shape[2])
        )
        k_embed = (
            k_embed.reshape(
                k_embed.shape[0], k_embed.shape[1], 2, k_embed.shape[2] // 2
            )
            .permute(0, 1, 3, 2)
            .reshape(k_embed.shape[0], k_embed.shape[1], k_embed.shape[2])
        )
        return torch.cat([q_embed, q_pass], dim=-1), torch.cat(
            [k_embed, k_pass], dim=-1
        )

    else:
        raise ValueError(f"Unknown rotary type: {rotary_type}")


def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids, unsqueeze_dim=1, rotary_type="default"
):
    if rotary_type == "default":
        # NOTE: Performance of triton rotary kernel is untested for large batch sizes.
        # If it's slow on prefill, just switch to torch implementation on the else case.
        q_embed = apply_rotary_pos_emb_triton(q, cos, sin, position_ids)
        k_embed = apply_rotary_pos_emb_triton(k, cos, sin, position_ids)
        return q_embed, k_embed
    else:
        return apply_rotary_pos_emb_torch(
            q,
            k,
            cos,
            sin,
            position_ids,
            unsqueeze_dim=unsqueeze_dim,
            rotary_type=rotary_type,
        )


def get_linear_layout_contig_x_native_y(op_impl: str):
    if op_impl == "torch":
        return torch.nn.functional.linear
    elif op_impl == "muxi_custom_kernel":
        import muxi_layout_kernels

        def linear_layout_contig_x_native_y(x, w, b=None):
            assert x.ndim == 2
            x_is_vector = x.shape[0] == 1
            if not x_is_vector:
                x_transposed = muxi_layout_kernels.layoutB(x)
            # w has already been transposed, but reshaped back for compatibility. We only need to "view" it again.
            w_transposed = w.view(w.shape[0] // 16, w.shape[1] // 8, 16, 8)
            if x_is_vector:
                y = muxi_layout_kernels.gemv_layoutA(w_transposed, x)
                # TODO: support bias
                if b is not None:
                    y += b
                # View as 5D to be compatible with "native layout" but make n's tile to be 1.
                y = y.view(y.shape[1] // 32, 1, 4, 1, 8)
            elif x_transposed.shape[1] * 16 > 256:
                y = muxi_layout_kernels.muxi_hgemm_layout(w_transposed, x_transposed)
                # TODO: support bias
                if b is not None:
                    y += b
                y = muxi_layout_kernels.layoutB(y)
            else:
                y = muxi_layout_kernels.gemm_layoutABC(
                    w_transposed, x_transposed, bias=b
                )
            return y

        return linear_layout_contig_x_native_y
    else:
        raise NotImplementedError()


def get_linear_layout_native_x_contig_y(op_impl: str):
    if op_impl == "torch":
        return torch.nn.functional.linear
    elif op_impl == "muxi_custom_kernel":
        import muxi_layout_kernels

        def linear_layout_native_x_contig_y(x_transposed, w, b=None):
            assert x_transposed.ndim == 5
            x_is_vector = x_transposed.shape[1] == 1 and x_transposed.shape[3] == 1
            # w has already been transposed, but reshaped back for compatibility. We only need to "view" it again.
            w_transposed = w.view(w.shape[0] // 16, w.shape[1] // 8, 16, 8)
            if x_is_vector:
                y = muxi_layout_kernels.gemv_layoutA(
                    w_transposed, x_transposed.view(1, -1)
                )
                # TODO: support bias
                if b is not None:
                    y += b
            elif x_transposed.shape[1] * 16 > 256:
                y = muxi_layout_kernels.muxi_hgemm_layout(w_transposed, x_transposed)
                # TODO: support bias
                if b is not None:
                    y += b
            else:
                y = muxi_layout_kernels.gemm_layoutAB_ContinuousC(
                    w_transposed, x_transposed, bias=b
                )
            return y

        return linear_layout_native_x_contig_y
    else:
        raise NotImplementedError()


def get_linear_layout_contig_x_contig_y(op_impl: str):
    if op_impl == "torch":
        return torch.nn.functional.linear
    elif op_impl == "muxi_custom_kernel":
        import muxi_layout_kernels

        def linear_layout_native_x_contig_y(x, w, b=None):
            assert x.ndim == 2
            x_is_vector = x.shape[0] == 1
            if not x_is_vector:
                x_transposed = muxi_layout_kernels.layoutB(x)
            # w has already been transposed, but reshaped back for compatibility. We only need to "view" it again.
            w_transposed = w.view(w.shape[0] // 16, w.shape[1] // 8, 16, 8)
            if x_is_vector:
                y = muxi_layout_kernels.gemv_layoutA(w_transposed, x)
                # TODO: support bias
                if b is not None:
                    y += b
            elif x_transposed.shape[1] * 16 > 256:
                y = muxi_layout_kernels.muxi_hgemm_layout(w_transposed, x_transposed)
                # TODO: support bias
                if b is not None:
                    y += b
            else:
                y = muxi_layout_kernels.gemm_layoutAB_ContinuousC(
                    w_transposed, x_transposed, bias=b
                )
            return y

        return linear_layout_native_x_contig_y
    else:
        raise NotImplementedError()
