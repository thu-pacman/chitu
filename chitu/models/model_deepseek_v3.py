import math
from logging import getLogger
from typing import Any, List, Mapping, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from typing_extensions import override

import chitu_backend
from chitu.attn_backend import AttnBackend
from chitu.cache_manager import PagedKVCacheManager
from chitu.device_type import get_device_name, is_muxi, is_nvidia
from chitu.global_vars import get_global_args
from chitu.models.model import Attention, RMSNorm, Transformer, TransformerBlock
from chitu.ops import (
    act_quant_deepseek_v3,
    apply_rotary_pos_emb,
    fp8_gemm_deepseek_v3,
    soft_fp8_gemm_deepseek_v3,
    weight_dequant_deepseek_v3,
    weight_dequant_soft_fp8_deepseek_v3,
)
from chitu.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    get_tp_group,
    get_tp_rank,
    get_tp_size,
)
from chitu.utils import try_import_opt_dep

logger = getLogger(__name__)

triton, has_triton = try_import_opt_dep("triton", "triton")
if has_triton:
    from chitu.fused_moe import fused_experts


def parse_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    elif name == "bfloat16":
        return torch.bfloat16
    elif name == "float8_e4m3fn":
        return torch.float8_e4m3fn
    else:
        assert False


def linear_deepseek_v3(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() > 1`), a dequantized version
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm_deepseek_v3` for computation.
    """

    block_size = 128
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif get_global_args().infer.soft_fp8:
        if is_nvidia() or is_muxi():
            y = soft_fp8_gemm_deepseek_v3(x, weight, weight_scale)
            if bias is not None:
                y += bias
            return y
        else:
            logger.warning(
                f"Soft-fp8 fused gemm not implemented for {get_device_name()}, falling back to soft-fp8 conversion"
            )
            weight_dequanted = weight_dequant_soft_fp8_deepseek_v3(
                weight, weight_scale, block_size
            )
            return F.linear(x, weight_dequanted, bias)
    else:
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        x, act_scale = act_quant_deepseek_v3(x, block_size)
        assert weight_scale is not None
        y = fp8_gemm_deepseek_v3(x, act_scale, weight, weight_scale)
        if bias is not None:
            y += bias
        return y.view(x_shape[:-1] + y.shape[-1:])


class LinearDeepSeekV3(nn.Module):
    """
    FP8 linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = False,
        dtype=torch.float8_e4m3fn,
        bias_dtype=None,
    ):
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        block_size = 128
        scale_out_features = (out_features + block_size - 1) // block_size
        scale_in_features = (in_features + block_size - 1) // block_size
        if dtype.itemsize == 1:
            self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
            )
        else:
            self.register_parameter("scale", None)
        if has_bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=bias_dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear_deepseek_v3(x, self.weight, self.scale, self.bias)


class ColumnParallelLinearDeepSeekV3(ColumnParallelLinear):
    """
    FP8 column parallel linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = False,
        dtype=torch.float8_e4m3fn,
        bias_dtype=None,
        gather_output: bool = True,
    ):
        super().__init__(
            in_features,
            out_features,
            has_bias=has_bias,
            dtype=dtype,
            bias_dtype=bias_dtype,
            linear_op=lambda x, w, b: linear_deepseek_v3(x, w, self.scale, b),
            gather_output=gather_output,
        )

        dtype = dtype or torch.get_default_dtype()
        if dtype.itemsize == 1:
            local_out_features, local_in_features = self.weight.shape
            block_size = 128
            scale_out_features = (local_out_features + block_size - 1) // block_size
            scale_in_features = (local_in_features + block_size - 1) // block_size
            self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
            )
        else:
            self.scale = None


class RowParallelLinearDeepSeekV3(RowParallelLinear):
    """
    FP8 row parallel linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = False,
        dtype=torch.float8_e4m3fn,
        bias_dtype=None,
        input_is_parallel: bool = False,
    ):
        super().__init__(
            in_features,
            out_features,
            has_bias=has_bias,
            dtype=dtype,
            bias_dtype=bias_dtype,
            linear_op=lambda x, w, b: linear_deepseek_v3(x, w, self.scale, b),
            input_is_parallel=input_is_parallel,
        )

        dtype = dtype or torch.get_default_dtype()
        if dtype.itemsize == 1:
            local_out_features, local_in_features = self.weight.shape
            block_size = 128
            scale_out_features = (local_out_features + block_size - 1) // block_size
            scale_in_features = (local_in_features + block_size - 1) // block_size
            self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
            )
        else:
            self.scale = None


class GroupColumnParallelLinearDeepSeekV3(torch.nn.Module):
    def __init__(
        self,
        group_size: int,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        gather_output: bool = True,
        dtype=None,
        bias_dtype=None,
    ):
        super().__init__()

        dtype = dtype or torch.get_default_dtype()

        self.tp_group = get_tp_group()
        self.tp_size = get_tp_size()
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        assert (
            out_features % self.tp_size == 0
        ), "out_features must be divisible by tp_size"
        local_out_features = local_out_features = out_features // self.tp_size

        self.weight = torch.nn.Parameter(
            torch.empty(group_size, local_out_features, in_features, dtype=dtype)
        )
        if has_bias:
            self.bias = torch.nn.Parameter(
                torch.empty(group_size, local_out_features, dtype=bias_dtype or dtype)
            )
        else:
            self.bias = None
        if dtype.itemsize == 1:
            block_size = 128
            scale_out_features = (local_out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.scale = nn.Parameter(
                torch.empty(
                    group_size,
                    scale_out_features,
                    scale_in_features,
                    dtype=torch.float32,
                )
            )
        else:
            self.scale = None

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(xs) == self.group_size
        ys = []
        for i in range(self.group_size):
            y = None
            if xs[i] is not None:
                x = xs[i]
                y = linear_deepseek_v3(
                    x,
                    self.weight[i],
                    self.scale[i] if self.scale is not None else None,
                    self.bias[i] if self.bias is not None else None,
                )
                if self.gather_output and self.tp_size > 1:
                    y_transposed = y.permute(-1, *range(y.dim() - 1)).contiguous()
                    shape = list(y_transposed.shape)
                    shape[0] *= self.tp_size
                    y_gathered = y.new_empty(shape)
                    torch.distributed.all_gather_into_tensor(
                        y_gathered, y_transposed, group=self.tp_group
                    )
                    y = y_gathered.permute(*range(1, y.dim()), 0)
            ys.append(y)
        return ys


class GroupRowParallelLinearDeepSeekV3(torch.nn.Module):
    def __init__(
        self,
        group_size: int,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        input_is_parallel: bool = False,
        dtype=None,
        bias_dtype=None,
    ):
        super().__init__()

        dtype = dtype or torch.get_default_dtype()

        self.tp_group = get_tp_group()
        self.tp_size = get_tp_size()
        self.rank = get_tp_rank()
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features

        assert (
            in_features % self.tp_size == 0
        ), "in_features must be divisible by tp_size"
        local_in_features = in_features // self.tp_size

        self.input_is_parallel = input_is_parallel

        self.weight = torch.nn.Parameter(
            torch.empty(group_size, out_features, local_in_features, dtype=dtype)
        )
        if has_bias:
            self.bias = torch.nn.Parameter(
                torch.empty(group_size, out_features, dtype=bias_dtype or dtype)
            )
        else:
            self.bias = None
        if dtype.itemsize == 1:
            block_size = 128
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (local_in_features + block_size - 1) // block_size
            self.scale = nn.Parameter(
                torch.empty(
                    group_size,
                    scale_out_features,
                    scale_in_features,
                    dtype=torch.float32,
                )
            )
        else:
            self.scale = None

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(xs) == self.group_size
        ys = []
        for i in range(self.group_size):
            y = None
            if xs[i] is not None:
                x = xs[i]
                if not self.input_is_parallel and self.tp_size > 1:
                    shape = list(x.shape)
                    this_rank_dim = shape[-1] // self.tp_size
                    shape[-1] = self.tp_size
                    shape.append(this_rank_dim)
                    x = x.view(shape).select(-2, self.rank)
                if self.tp_size > 1:
                    y = linear_deepseek_v3(
                        x,
                        self.weight[i],
                        self.scale[i] if self.scale is not None else None,
                        (
                            self.bias[i]
                            if self.rank == 0 and self.bias is not None
                            else None
                        ),
                    )
                    torch.distributed.all_reduce(y, group=self.tp_group)
                else:
                    y = linear_deepseek_v3(
                        x,
                        self.weight[i],
                        self.scale[i] if self.scale is not None else None,
                        self.bias[i] if self.bias is not None else None,
                    )
            ys.append(y)
        return ys


class AttentionDeepSeekV3(Attention):
    def __init__(
        self,
        args,
        layer_id,
        cache,
        attn_backend,
        mla_absorb,
        merge_qkv,
    ):
        super().__init__(layer_id, cache, attn_backend)
        self.mla_absorb = mla_absorb
        self.merge_qkv = merge_qkv

        model_parallel_size = get_tp_size()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // model_parallel_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        block_size = 128

        if merge_qkv:
            # fp8 gemm can handle weights not divisible by block_size, but it does not hold
            # after merging for the output dimension, except for the last weight.
            assert self.q_lora_rank % block_size == 0
            self.wqkv_a = LinearDeepSeekV3(
                self.dim,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.get_default_dtype(),
            )
        else:
            self.wq_a = LinearDeepSeekV3(
                self.dim,
                self.q_lora_rank,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.get_default_dtype(),
            )
            self.wkv_a = LinearDeepSeekV3(
                self.dim,
                self.kv_lora_rank + self.qk_rope_head_dim,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.get_default_dtype(),
            )
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = ColumnParallelLinearDeepSeekV3(
            self.q_lora_rank,
            self.n_heads * self.qk_head_dim,
            has_bias=False,
            dtype=parse_dtype(args.main_weight_dtype),
            bias_dtype=torch.get_default_dtype(),
            gather_output=False,
        )
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinearDeepSeekV3(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            has_bias=False,
            dtype=parse_dtype(args.main_weight_dtype),
            bias_dtype=torch.get_default_dtype(),
            gather_output=False,
        )
        self.wo = RowParallelLinearDeepSeekV3(
            self.n_heads * self.v_head_dim,
            self.dim,
            has_bias=False,
            dtype=parse_dtype(args.main_weight_dtype),
            bias_dtype=torch.get_default_dtype(),
            input_is_parallel=True,
        )
        self.softmax_scale = compute_softmax_scale_deepseek_v3(args)

    def _run_linear(self, x, freqs_cis_cos, freqs_cis_sin):
        bs_seq, _ = x.size()
        assert self.q_lora_rank > 0
        if self.merge_qkv:
            q_a_kv = self.wqkv_a(x)
            q_a, kv = torch.split(
                q_a_kv,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
        else:
            q_a = self.wq_a(x)
            kv = self.wkv_a(x)
        q = self.wq_b(self.q_norm(q_a, compute_dtype=q_a.dtype))
        q = q.view(bs_seq, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        q_pe, k_pe = apply_rotary_pos_emb(
            q_pe, k_pe, freqs_cis_cos, freqs_cis_sin, rotary_type="llama"
        )
        if self.mla_absorb == "none":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
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
            return q, k, v
        elif self.mla_absorb == "absorb-without-precomp":
            block_size = 128
            weight_dequant_fn = (
                weight_dequant_soft_fp8_deepseek_v3
                if get_global_args().infer.soft_fp8
                else weight_dequant_deepseek_v3
            )
            wkv_b = (
                self.wkv_b.weight
                if self.wkv_b.scale is None
                else weight_dequant_fn(self.wkv_b.weight, self.wkv_b.scale, block_size)
            )
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum(
                "shd,hdc->shc", q_nope, wkv_b[:, : self.qk_nope_head_dim]
            )
            return q_nope, q_pe, kv, k_pe, wkv_b
        else:
            raise NotImplementedError(
                f"MLA absorb mode {self.mla_absorb} not supported"
            )

    def prefill_forward(
        self,
        x: torch.Tensor,
        freqs_cis_cos: torch.Tensor,
        freqs_cis_sin: torch.Tensor,
        varlens,
    ):
        bs_seq, _ = x.size()

        if self.mla_absorb == "none":
            q, k, v = self._run_linear(x, freqs_cis_cos, freqs_cis_sin)
            self.cache.finalize_cache_bylayer_prefill(
                k, v, self.cache.curr_req_ids, self.cache.curr_varlens, self.layer_id
            )
            x = self.attn_backend.attn_varlen_func(
                q,
                k,
                v,
                varlens.prefix_lens,
                varlens.prefix_lens,
                varlens.max_len,
                varlens.max_len,
                causal=True,
                softmax_scale=self.softmax_scale,
            )

        elif self.mla_absorb == "absorb-without-precomp":
            q_nope, q_pe, kv, k_pe, wkv_b = self._run_linear(
                x, freqs_cis_cos, freqs_cis_sin
            )

            kv_cache = self.kv_norm(kv, compute_dtype=kv.dtype)
            pe_cache = k_pe
            kv_pe_cache = torch.cat([kv_cache, pe_cache], dim=-1)
            if isinstance(self.cache, PagedKVCacheManager):
                self.cache.finalize_cache_bylayer_prefill(
                    kv_pe_cache,
                    None,
                    self.cache.curr_req_ids,
                    self.cache.curr_varlens,
                    self.layer_id,
                )
            else:
                self.cache.finalize_cache_bylayer_prefill(
                    kv_cache,
                    pe_cache,
                    self.cache.curr_req_ids,
                    self.cache.curr_varlens,
                    self.layer_id,
                )
            q_nope_pe = torch.cat([q_nope, q_pe], dim=-1)
            x = self.attn_backend.attn_varlen_func(
                q_nope_pe.view(-1, q_nope_pe.shape[-2], q_nope_pe.shape[-1]),
                kv_pe_cache.view(-1, 1, kv_pe_cache.shape[-1]),
                kv_cache.view(-1, 1, kv_cache.shape[-1]),
                varlens.prefix_lens,
                varlens.prefix_lens,
                varlens.max_len,
                varlens.max_len,
                causal=True,
                softmax_scale=self.softmax_scale,
            )

            x = x.view(bs_seq, x.shape[-2], x.shape[-1])
            x = torch.einsum("shc,hdc->shd", x, wkv_b[:, -self.v_head_dim :])

        else:
            raise NotImplementedError(
                f"MLA absorb mode {self.mla_absorb} not supported"
            )

        x = self._run_output_linear(x)
        return x.view(bs_seq, -1)

    def decode_forward(
        self, x: torch.Tensor, freqs_cis_cos: torch.Tensor, freqs_cis_sin: torch.Tensor
    ):
        cache_seqlens_excl_this_decode = self.cache.get_gpu_seq_lens_excl_this_decode()
        bsz, seqlen, _ = x.size()

        if self.mla_absorb == "none":
            q, k, v = self._run_linear(
                x.view(bsz * seqlen, -1), freqs_cis_cos, freqs_cis_sin
            )
            q = q.view(bsz, seqlen, self.n_local_heads, -1)
            k = k.view(bsz, seqlen, self.n_local_heads, -1)
            v = v.view(bsz, seqlen, self.n_local_heads, -1)

            cache = self.cache.get_cache_decode(self.layer_id)
            cache_k = cache[0]
            cache_v = cache[1]
            x = self.attn_backend.attn_with_kvcache(
                q,
                cache_k,
                cache_v,
                k,
                v,
                cache_seqlens=cache_seqlens_excl_this_decode,
                softmax_scale=self.softmax_scale,
            ).view(bsz, seqlen, 1, -1)

        elif self.mla_absorb == "absorb-without-precomp":
            q_nope, q_pe, kv, k_pe, wkv_b = self._run_linear(
                x.view(bsz * seqlen, -1), freqs_cis_cos, freqs_cis_sin
            )

            kv_cache, pe_cache = self.cache.get_cache_decode(self.layer_id)
            this_kv = self.kv_norm(kv, compute_dtype=kv.dtype)
            this_pe = k_pe
            q_nope_pe = torch.cat([q_nope, q_pe], dim=-1)
            kv_pe_cache = torch.cat([kv_cache, pe_cache], dim=-1)
            this_kv_pe = torch.cat([this_kv, this_pe], dim=-1)
            x = self.attn_backend.attn_with_kvcache(
                q_nope_pe.view(bsz, seqlen, q_nope_pe.shape[-2], q_nope_pe.shape[-1]),
                kv_pe_cache.view(kv_pe_cache.shape[0], kv_pe_cache.shape[1], 1, -1),
                kv_cache.view(kv_cache.shape[0], kv_cache.shape[1], 1, -1),
                this_kv_pe.view(bsz, seqlen, 1, -1),
                this_kv.view(bsz, seqlen, 1, -1),
                cache_seqlens=cache_seqlens_excl_this_decode,
                softmax_scale=self.softmax_scale,
            )
            for start_pos in cache_seqlens_excl_this_decode:
                pe_cache[:, start_pos] = kv_pe_cache[:, start_pos, self.kv_lora_rank :]

            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])

        else:
            raise NotImplementedError(
                f"MLA absorb mode {self.mla_absorb} not supported"
            )

        x = self._run_output_linear(x)
        return x

    def decode_forward_paged(
        self, x: torch.Tensor, freqs_cis_cos: torch.Tensor, freqs_cis_sin: torch.Tensor
    ):
        cache_seqlens_excl_this_decode = self.cache.get_gpu_seq_lens_excl_this_decode()
        cache_seqlens_incl_this_decode = self.cache.get_gpu_seq_lens_incl_this_decode()
        bsz, seqlen, _ = x.size()
        q_nope, q_pe, kv, k_pe, wkv_b = self._run_linear(
            x.view(bsz * seqlen, -1), freqs_cis_cos, freqs_cis_sin
        )

        block_table = self.cache.get_gpu_block_table()
        paged_kv_cache = self.cache.get_paged_kv_cache(self.layer_id)
        this_kv = self.kv_norm(kv, compute_dtype=kv.dtype)
        this_pe = k_pe
        this_kv_pe = torch.cat([this_kv, this_pe], dim=-1)
        x = self.attn_backend.mla_attn_with_kvcache(
            q_nope,
            q_pe,
            paged_kv_cache,
            this_kv_pe.view(bsz, seqlen, 1, -1),
            cache_seqlens_excl_this_decode=cache_seqlens_excl_this_decode,
            cache_seqlens_incl_this_decode=cache_seqlens_incl_this_decode,
            block_table=block_table,
            softmax_scale=self.softmax_scale,
        )
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])
        x = self._run_output_linear(x)
        return x

    def _run_output_linear(self, x):
        x = self.wo(x.flatten(-2))
        return x


class MLPDeepSeekV3(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, args, merge_gate_up: bool):
        super().__init__()
        self.merge_gate_up = merge_gate_up

        if merge_gate_up:
            self.w1w3 = ColumnParallelLinearDeepSeekV3(
                args.dim,
                args.inter_dim * 2,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.get_default_dtype(),
                gather_output=False,
            )
        else:
            self.w1 = ColumnParallelLinearDeepSeekV3(
                args.dim,
                args.inter_dim,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.get_default_dtype(),
                gather_output=False,
            )
            self.w3 = ColumnParallelLinearDeepSeekV3(
                args.dim,
                args.inter_dim,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.get_default_dtype(),
                gather_output=False,
            )
        self.w2 = RowParallelLinearDeepSeekV3(
            args.inter_dim,
            args.dim,
            has_bias=False,
            dtype=parse_dtype(args.main_weight_dtype),
            bias_dtype=torch.get_default_dtype(),
            input_is_parallel=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        if self.merge_gate_up:
            w1w3_out = self.w1w3(x)
            w1_out, w3_out = torch.split(w1w3_out, w1w3_out.shape[-1] // 2, dim=-1)
        else:
            w1_out = self.w1(x)
            w3_out = self.w3(x)
        return self.w2(F.silu(w1_out) * w3_out)


class GateDeepSeekV3(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """

    def __init__(self, args):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = (
            nn.Parameter(torch.empty(args.n_routed_experts))
            if self.dim == 7168
            else None
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = F.linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices


class MoEDeepSeekV3(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """

    def __init__(self, args, merge_gate_up: bool):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.merge_gate_up = merge_gate_up
        self.dim = args.dim

        moe_world_size = 1
        moe_rank = 0
        assert (
            args.n_routed_experts % moe_world_size == 0
        ), f"Number of experts must be divisible by world size (world_size={moe_world_size})"
        self.n_shared_experts = args.n_shared_experts
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // moe_world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = moe_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = GateDeepSeekV3(args)
        if merge_gate_up:
            self.w1w3 = GroupColumnParallelLinearDeepSeekV3(
                self.experts_end_idx - self.experts_start_idx + self.n_shared_experts,
                args.dim,
                args.moe_inter_dim * 2,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.get_default_dtype(),
                gather_output=False,
            )
        else:
            self.w1 = GroupColumnParallelLinearDeepSeekV3(
                self.experts_end_idx - self.experts_start_idx + self.n_shared_experts,
                args.dim,
                args.moe_inter_dim,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.get_default_dtype(),
                gather_output=False,
            )
            self.w3 = GroupColumnParallelLinearDeepSeekV3(
                self.experts_end_idx - self.experts_start_idx + self.n_shared_experts,
                args.dim,
                args.moe_inter_dim,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.get_default_dtype(),
                gather_output=False,
            )
        self.w2 = GroupRowParallelLinearDeepSeekV3(
            self.experts_end_idx - self.experts_start_idx + self.n_shared_experts,
            args.moe_inter_dim,
            args.dim,
            has_bias=False,
            dtype=parse_dtype(args.main_weight_dtype),
            bias_dtype=torch.get_default_dtype(),
            input_is_parallel=True,
        )

    def get_expert_weights_for_fp8_w8a8(self, expert_num):
        w1w3_weight = self.w1w3.weight[:expert_num]
        w1w3_scale = self.w1w3.scale[:expert_num]
        w2_weight = self.w2.weight[:expert_num]
        w2_scale = self.w2.scale[:expert_num]
        return w1w3_weight, w1w3_scale, w2_weight, w2_scale

    def get_expert_weights_for_non_fp8(self, expert_num):
        w1w3_weight = self.w1w3.weight[:expert_num]
        w1w3_scale = None
        w2_weight = self.w2.weight[:expert_num]
        w2_scale = None
        return w1w3_weight, w1w3_scale, w2_weight, w2_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """

        shared_experts = 0
        if get_global_args().infer.fuse_shared_experts:
            shared_experts = self.n_shared_experts

        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)

        if has_triton:

            if self.w1w3.scale is None and self.w2.scale is None:
                w1w3_weight, w1w3_scale, w2_weight, w2_scale = (
                    self.get_expert_weights_for_non_fp8(
                        self.n_routed_experts + shared_experts
                    )
                )
                use_fp8_w8a8 = False
                fused_soft_fp8 = False
            else:
                assert self.w1w3.scale is not None
                assert self.w2.scale is not None
                if not get_global_args().infer.soft_fp8:
                    w1w3_weight, w1w3_scale, w2_weight, w2_scale = (
                        self.get_expert_weights_for_fp8_w8a8(
                            self.n_routed_experts + shared_experts
                        )
                    )
                    use_fp8_w8a8 = True
                    fused_soft_fp8 = False
                elif is_nvidia() or is_muxi():
                    w1w3_weight, w1w3_scale, w2_weight, w2_scale = (
                        self.get_expert_weights_for_fp8_w8a8(
                            self.n_routed_experts + shared_experts
                        )
                    )
                    use_fp8_w8a8 = True
                    fused_soft_fp8 = True

                else:
                    logger.warning(
                        f"Soft-fp8 fused gemm not implemented for {get_device_name()}, falling back to soft-fp8 conversion"
                    )
                    block_size = 128
                    w1w3_weight = weight_dequant_soft_fp8_deepseek_v3(
                        self.w1w3.weight[
                            : self.n_routed_experts + self.n_shared_experts
                        ],
                        self.w1w3.scale[
                            : self.n_routed_experts + self.n_shared_experts
                        ],
                        block_size,
                    )
                    w1w3_scale = None
                    w2_weight = weight_dequant_soft_fp8_deepseek_v3(
                        self.w2.weight[: self.n_routed_experts + self.n_shared_experts],
                        self.w2.scale[: self.n_routed_experts + self.n_shared_experts],
                        block_size,
                    )
                    w2_scale = None
                    use_fp8_w8a8 = False
                    fused_soft_fp8 = False

            if not get_global_args().infer.fuse_shared_experts:
                w1w3_out = linear_deepseek_v3(
                    x,
                    self.w1w3.weight[-1],
                    self.w1w3.scale[-1] if self.w1w3.scale is not None else None,
                    self.w1w3.bias[-1] if self.w1w3.bias is not None else None,
                )
                w1_out, w3_out = torch.split(w1w3_out, w1w3_out.shape[-1] // 2, dim=-1)
                act = F.silu(w1_out) * w3_out
                y1 = linear_deepseek_v3(
                    act,
                    self.w2.weight[-1],
                    self.w2.scale[-1] if self.w2.scale is not None else None,
                    self.w2.bias[-1] if self.w2.bias is not None else None,
                )

                y = fused_experts(
                    x,
                    w1w3_weight,
                    w2_weight,
                    topk_weights=weights,
                    topk_ids=indices,
                    use_fp8_w8a8=use_fp8_w8a8,
                    inplace=True,
                    global_num_experts=self.n_routed_experts,
                    expert_map=None,  # use when ep > 1
                    w1_scale=w1w3_scale,
                    w2_scale=w2_scale,
                    block_shape=[128, 128],
                    soft_fp8=fused_soft_fp8,
                )

                y += y1
            else:
                indice_shape = indices.shape
                new_indices = torch.zeros(
                    (indice_shape[0], indice_shape[1] + 1),
                    dtype=indices.dtype,
                    device=indices.device,
                )

                new_weights = torch.zeros(
                    (weights.shape[0], weights.shape[1] + 1),
                    dtype=weights.dtype,
                    device=weights.device,
                )

                chitu_backend.cuda_add_shared_experts(
                    new_weights,
                    new_indices,
                    weights,
                    indices,
                    self.n_routed_experts,
                    self.n_shared_experts,
                )
                del weights, indices
                y = fused_experts(
                    x,
                    w1w3_weight,
                    w2_weight,
                    topk_weights=new_weights,
                    topk_ids=new_indices,
                    use_fp8_w8a8=use_fp8_w8a8,
                    inplace=True,
                    global_num_experts=self.n_routed_experts + self.n_shared_experts,
                    expert_map=None,  # use when ep > 1
                    w1_scale=w1w3_scale,
                    w2_scale=w2_scale,
                    block_shape=[128, 128],
                    soft_fp8=fused_soft_fp8,
                )

            torch.distributed.all_reduce(y, group=get_tp_group())
        else:
            y = torch.zeros_like(x)
            counts = torch.bincount(
                indices.flatten(), minlength=self.n_routed_experts
            ).tolist()

            xs = []
            for i in range(self.experts_start_idx, self.experts_end_idx):
                this_x = None
                if counts[i]:
                    idx, top = torch.where(indices == i)
                    this_x = x[idx]
                xs.append(this_x)
            xs += [x] * self.n_shared_experts

            if self.merge_gate_up:
                w1w3_outs = self.w1w3(xs)
                w1_outs, w3_outs = zip(
                    *[
                        (
                            torch.split(w1w3_out, w1w3_out.shape[-1] // 2, dim=-1)
                            if w1w3_out is not None
                            else (None, None)
                        )
                        for w1w3_out in w1w3_outs
                    ]
                )
            else:
                w1_outs = self.w1(xs)
                w3_outs = self.w3(xs)

            act = [
                F.silu(w1_out) * w3_out if w1_out is not None else None
                for w1_out, w3_out in zip(w1_outs, w3_outs)
            ]

            w2_outs = self.w2(act)

            for i in range(self.experts_start_idx, self.experts_end_idx):
                if counts[i]:
                    idx, top = torch.where(indices == i)
                    y[idx] += (
                        w2_outs[i - self.experts_start_idx] * weights[idx, top, None]
                    )
            for i in range(
                self.experts_end_idx - self.experts_start_idx,
                self.experts_end_idx - self.experts_start_idx + self.n_shared_experts,
            ):
                y += w2_outs[i]
        return y.view(shape)


class TransformerBlockDeepSeekV3(TransformerBlock):
    def __init__(
        self,
        layer_id: int,
        args,
        cache,
        attn_backend,
        op_impl,
        mla_absorb,
        merge_qkv_gate_up,
    ):
        super().__init__(
            layer_id, args, cache, attn_backend=attn_backend, op_impl=op_impl
        )
        self.attn = AttentionDeepSeekV3(
            args,
            layer_id,
            cache,
            attn_backend,
            mla_absorb=mla_absorb,
            merge_qkv=merge_qkv_gate_up,
        )
        self.ffn = (
            MLPDeepSeekV3(
                args,
                merge_gate_up=merge_qkv_gate_up,
            )
            if layer_id < args.n_dense_layers
            else MoEDeepSeekV3(
                args,
                merge_gate_up=merge_qkv_gate_up,
            )
        )
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis_cos: torch.Tensor,
        freqs_cis_sin: torch.Tensor,
        varlens=None,
    ):
        x = x + self.attn(
            self.attn_norm(x, compute_dtype=x.dtype),
            freqs_cis_cos,
            freqs_cis_sin,
            varlens,
        )
        x = x + self.ffn(self.ffn_norm(x, compute_dtype=x.dtype))
        return x


class TransformerDeepSeekV3(Transformer):
    def __init__(
        self,
        params,
        cache,
        *,
        max_position_embeddings: int,
        pipeline_parallel_size: int,
        model_parallel_size: int,
        attn_backend: AttnBackend,
        op_impl: str,
        mla_absorb: str,
        merge_qkv_gate_up=True,
    ):
        self.mla_absorb = mla_absorb
        self.merge_qkv_gate_up = merge_qkv_gate_up
        super().__init__(
            params,
            cache,
            max_position_embeddings=max_position_embeddings,
            pipeline_parallel_size=pipeline_parallel_size,
            model_parallel_size=model_parallel_size,
            attn_backend=attn_backend,
            op_impl=op_impl,
            mla_absorb=mla_absorb,
        )
        if op_impl != "torch":
            raise NotImplementedError("Only op_impl=torch is supported in DeepSeek V3")

    @override
    def _get_tensor_column_parallel_layer_names(self) -> List[str]:
        return ["embed", "wq_b", "wkv_b", "w1", "w3", "head"]

    @override
    def _get_tensor_row_parallel_layer_names(self) -> List[str]:
        return ["wo", "w2"]

    @override
    def _get_pre_layer_prefixes(self) -> List[str]:
        return ["embed."]

    @override
    def _get_post_layer_prefixes(self) -> List[str]:
        return ["head.", "norm."]

    @override
    def _get_layer_i_prefixes(self, i: int) -> List[str]:
        return [f"layers.{i}."]

    def _process_state_dict_for_merging_experts(self, checkpoint: Mapping[str, Any]):
        new_checkpoint = {}
        for k in checkpoint.keys():
            replaced = False
            for w in ["w1", "w2", "w3", "w1w3"]:
                for part in ["weight", "scale", "bias"]:
                    if k.endswith(f".experts.0.{w}.{part}"):
                        prefix = k[: -len(f"experts.0.{w}.{part}")]
                        parts = []
                        for i in range(self.params.n_routed_experts):
                            parts.append(checkpoint[prefix + f"experts.{i}.{w}.{part}"])
                        parts.append(checkpoint[prefix + f"shared_experts.{w}.{part}"])
                        new_checkpoint[prefix + f"{w}.{part}"] = torch.stack(
                            parts, dim=0
                        )
                        replaced = True
                        break
                if replaced:
                    break
            if replaced:
                continue
            if ".experts." in k or ".shared_experts." in k:
                continue
            new_checkpoint[k] = checkpoint[k]
        return new_checkpoint

    def _process_state_dict_for_merging_qkv(self, checkpoint: Mapping[str, Any]):
        new_checkpoint = {}
        for k in checkpoint.keys():
            if k.endswith(".wq_a.weight"):
                prefix = k[: -len("wq_a.weight")]
                assert prefix + "wkv_a.weight" in checkpoint
                q_weight = checkpoint[prefix + "wq_a.weight"]
                kv_weight = checkpoint[prefix + "wkv_a.weight"]
                new_checkpoint[prefix + "wqkv_a.weight"] = torch.cat(
                    [q_weight, kv_weight], dim=0
                )
            elif k.endswith(".wkv_a.weight"):
                continue
            elif k.endswith(".wq_a.scale"):
                prefix = k[: -len("wq_a.scale")]
                assert prefix + "wkv_a.scale" in checkpoint
                q_scale = checkpoint[prefix + "wq_a.scale"]
                kv_scale = checkpoint[prefix + "wkv_a.scale"]
                new_checkpoint[prefix + "wqkv_a.scale"] = torch.cat(
                    [q_scale, kv_scale], dim=0
                )
            elif k.endswith(".wkv_a.scale"):
                continue
            elif k.endswith(".wq_a.bias"):
                prefix = k[: -len("wq_a.bias")]
                assert prefix + "wkv_a.bias" in checkpoint
                q_bias = checkpoint[prefix + "wq_a.bias"]
                kv_bias = checkpoint[prefix + "wkv_a.bias"]
                new_checkpoint[prefix + "wqkv_a.bias"] = torch.cat(
                    [q_bias, kv_bias], dim=0
                )
            elif k.endswith(".wkv_a.bias"):
                continue
            else:
                new_checkpoint[k] = checkpoint[k]
        return new_checkpoint

    def _process_state_dict_for_merging_gate_up(self, checkpoint: Mapping[str, Any]):
        new_checkpoint = {}
        for k in checkpoint.keys():
            if k.endswith(".w1.weight"):
                prefix = k[: -len("w1.weight")]
                assert prefix + "w3.weight" in checkpoint
                assert prefix + "w1w3.weight" not in checkpoint
                gate_weight = checkpoint[prefix + "w1.weight"]
                up_weight = checkpoint[prefix + "w3.weight"]
                new_checkpoint[prefix + "w1w3.weight"] = torch.cat(
                    [gate_weight, up_weight], dim=0
                )
            elif k.endswith(".w3.weight"):
                continue
            elif k.endswith(".w1.scale"):
                prefix = k[: -len("w1.scale")]
                assert prefix + "w3.scale" in checkpoint
                assert prefix + "w1w3.scale" not in checkpoint
                gate_scale = checkpoint[prefix + "w1.scale"]
                up_scale = checkpoint[prefix + "w3.scale"]
                new_checkpoint[prefix + "w1w3.scale"] = torch.cat(
                    [gate_scale, up_scale], dim=0
                )
            elif k.endswith(".w3.scale"):
                continue
            elif k.endswith(".w1.bias"):
                prefix = k[: -len("w1.bias")]
                assert prefix + "w3.bias" in checkpoint
                assert prefix + "w1w3.bias" not in checkpoint
                gate_bias = checkpoint[prefix + "w1.bias"]
                up_bias = checkpoint[prefix + "w3.bias"]
                new_checkpoint[prefix + "w1w3.bias"] = torch.cat(
                    [gate_bias, up_bias], dim=0
                )
            elif k.endswith(".w3.bias"):
                continue
            else:
                new_checkpoint[k] = checkpoint[k]
        return new_checkpoint

    @override
    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        skip_preprocess: bool = False,
        *args,
        **kwargs,
    ):
        if not skip_preprocess:

            if self.merge_qkv_gate_up:
                state_dict = self._process_state_dict_for_merging_qkv(state_dict)
                state_dict = self._process_state_dict_for_merging_gate_up(state_dict)

            state_dict = self._process_state_dict_for_merging_experts(state_dict)

        super().load_state_dict(
            state_dict, skip_preprocess=skip_preprocess, *args, **kwargs
        )

    @override
    def _init_pre_layers(self):
        self.embed = VocabParallelEmbedding(self.params.vocab_size, self.params.dim)

    @override
    def _init_layers(self, cache, attn_backend, op_impl):
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.local_begin_layer_id, self.local_end_layer_id):
            self.layers.append(
                TransformerBlockDeepSeekV3(
                    layer_id,
                    self.params,
                    cache,
                    attn_backend,
                    self.op_impl,
                    mla_absorb=self.mla_absorb,
                    merge_qkv_gate_up=self.merge_qkv_gate_up,
                )
            )

    @override
    def _init_post_layers(self):
        self.norm = RMSNorm(self.params.dim)
        self.head = ColumnParallelLinear(
            self.params.dim,
            self.params.vocab_size,
            has_bias=False,
            dtype=torch.get_default_dtype(),
            gather_output=True,
        )

    @override
    def _pre_layers(self, h):
        return self.embed(h)

    @override
    def _post_layers(self, h):
        """NOTE: _post_layers is assumed to be a token-wise computation"""
        h = self.norm(h, compute_dtype=h.dtype)
        h = self.head(h)
        return h

    @override
    def precompute_freqs_cis(self, max_position_embeddings: int, device):
        self.freqs_cis = precompute_freqs_cis_deepseek_v3(
            self.params, max_position_embeddings
        ).to(device)

    @override
    def prepare_decoding_attn(self):
        cache_seqlens_excl_this_decode = self.cache.get_gpu_seq_lens_excl_this_decode()
        cache_seqlens_incl_this_decode = self.cache.get_gpu_seq_lens_incl_this_decode()
        block_table = self.cache.get_gpu_block_table()
        block_size = self.cache.get_block_size()
        self.attn_backend.prepare_metadata_for_decode(
            cache_seqlens_excl_this_decode,
            cache_seqlens_incl_this_decode,
            block_table,
            block_size,
            softmax_scale=compute_softmax_scale_deepseek_v3(self.params),
        )


def precompute_freqs_cis_deepseek_v3(args, max_position_embeddings) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = max_position_embeddings
    beta_fast: int = 32
    beta_slow: int = 1
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    original_seq_len: int = 4096
    if seqlen > original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def compute_softmax_scale_deepseek_v3(args):
    qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
    mscale: float = 1.0
    mscale = 0.1 * mscale * math.log(args.rope_factor) + 1.0
    return (qk_head_dim**-0.5) * mscale * mscale
