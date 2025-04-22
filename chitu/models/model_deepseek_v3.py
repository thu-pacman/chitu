import math
import functools
from logging import getLogger
from typing import Any, List, Mapping, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed
import torch.nn.functional as F
from torch import nn
from typing_extensions import override

import chitu_backend
from chitu.layers.gate import fused_sigmoid_gate
from chitu.attn_backend import AttnBackend
from chitu.cache_manager import PagedKVCacheManager
from chitu.device_type import (
    get_device_name,
    is_muxi,
    is_nvidia,
    has_native_fp8,
)
from chitu.global_vars import get_global_args
from chitu.models.model import Attention, RMSNorm, Transformer, TransformerBlock
from chitu.ops import (
    apply_rotary_pos_emb,
    weight_dequant_deepseek_v3,
    weight_dequant_soft_fp8_deepseek_v3,
    weight_quant_deepseek_v3,
    silu_and_mul,
    quant_einsum_shc_hdc_shd,
)
from chitu.tensor_parallel import (
    LocalLinear,
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    get_tp_group,
    get_tp_rank,
    get_tp_size,
)
from chitu.utils import try_import_opt_dep
from chitu.quantization import (
    linear_block_fp8,
    linear_block_fp4,
    Blockfp8Linear,
    Blockfp4Linear,
)

import ctypes

logger = getLogger(__name__)

triton, has_triton = try_import_opt_dep("triton", "triton")
if has_triton:
    from chitu.fused_moe import fused_experts


def parse_dtype(
    name: str,
    is_quant_layer: Optional[bool] = False,
) -> torch.dtype:
    if name == "float16":
        return torch.float16
    elif name == "bfloat16":
        return torch.bfloat16
    elif name == "float8_e4m3fn":
        return torch.float8_e4m3fn
    elif name == "float4_e2m1":
        if is_quant_layer:
            return torch.uint8
        else:
            return torch.bfloat16
    else:
        assert False


def linear_deepseek_v3(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    weight_scale_2: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    block_size = 128
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    else:
        if weight_scale_2 is not None:
            return linear_block_fp4(
                x=x,
                weight=weight,
                weight_scale=weight_scale,
                weight_scale_2=weight_scale_2,
                bias=bias,
                block_size=block_size,
            )
        else:
            return linear_block_fp8(
                x=x,
                weight=weight,
                weight_scale=weight_scale,
                bias=bias,
                block_size=block_size,
            )


def getLinearDeepSeekV3(
    is_quant_layer: bool = True,
):
    args = get_global_args()
    quant_method = args.models.quant if hasattr(args.models, "quant") else None
    if quant_method is None:
        return LocalLinear
    elif quant_method == "gguf":
        return Blockfp8Linear
    elif quant_method == "blockfp8":
        return Blockfp8Linear
    elif quant_method == "blockfp4":
        if is_quant_layer:
            return Blockfp4Linear
        else:
            return LocalLinear
    else:
        raise NotImplementedError(f"{quant_method} is not supported for DeepSeek V3.")


class ParallelAbsorbGemm(torch.nn.Module):
    def __init__(
        self,
        global_n_heads: int,
        in_features_per_head: int,
        out_features_per_head: int,
        dtype=None,
        block_size: int = 128,
    ):
        """
        The two group GeMMs in "absorb-without-precomp" mode, embaarrassingly parallel among heads.

        It computes `einsum("shc,hdc->shd", x, weight)` where `weight` may be block-fp8 quantized.
        """

        super().__init__()

        if dtype is None:
            dtype = torch.get_default_dtype()

        tp_size = get_tp_size()
        assert global_n_heads % tp_size == 0
        local_n_heads = global_n_heads // tp_size

        self.weight = torch.nn.Parameter(
            torch.empty(
                local_n_heads, out_features_per_head, in_features_per_head, dtype=dtype
            )
        )

        if dtype.itemsize == 1:
            assert out_features_per_head % block_size == 0
            assert in_features_per_head % block_size == 0
            self.scale = torch.nn.Parameter(
                torch.empty(
                    local_n_heads,
                    out_features_per_head // block_size,
                    in_features_per_head // block_size,
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter("scale", None)

        self.local_n_heads = local_n_heads
        self.in_features_per_head = in_features_per_head
        self.out_features_per_head = out_features_per_head
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            seq, n_head, n_hidden = x.shape
            bs = None
        else:
            bs, seq, n_head, n_hidden = x.shape
            x = x.view(bs * seq, n_head, n_hidden)

        y = quant_einsum_shc_hdc_shd(
            x,
            (
                self.weight
                if self.weight.element_size() > 1 or has_native_fp8()
                else self.weight.view(torch.uint8)
            ),
            self.scale,
            soft_fp8=(get_global_args().infer.raise_lower_bit_float_to == "bfloat16"),
        )

        if bs is not None:
            y = y.view(bs, seq, y.shape[-2], y.shape[-1])
        return y


class GroupColumnParallelLinearDeepSeekV3(torch.nn.Module):
    def __init__(
        self,
        group_size: int,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        gather_output: bool = True,
        dtype=None,
        is_fp4: bool = False,
        bias_dtype=None,
        scale_2_dim: int = 1,
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
            torch.empty(group_size, local_out_features, in_features, dtype=dtype),
            requires_grad=False,
        )
        if has_bias:
            self.bias = torch.nn.Parameter(
                torch.empty(group_size, local_out_features, dtype=bias_dtype or dtype),
                requires_grad=False,
            )
        else:
            self.bias = None
        if dtype.itemsize == 1:
            block_size = 128
            if is_fp4:
                quant_scale_stride = 8
                scale_out_features = local_out_features
                scale_in_features = (
                    in_features + quant_scale_stride - 1
                ) // quant_scale_stride
                self.input_scale = nn.Parameter(
                    torch.empty(
                        group_size,
                        scale_2_dim,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                self.scale_2 = nn.Parameter(
                    torch.empty(
                        group_size,
                        scale_2_dim,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
            else:
                scale_out_features = (local_out_features + block_size - 1) // block_size
                scale_in_features = (in_features + block_size - 1) // block_size
            self.scale = nn.Parameter(
                torch.empty(
                    group_size,
                    scale_out_features,
                    scale_in_features,
                    dtype=torch.float32,
                ),
                requires_grad=False,
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
        is_fp4: bool = False,
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
            torch.empty(group_size, out_features, local_in_features, dtype=dtype),
            requires_grad=False,
        )
        if has_bias:
            self.bias = torch.nn.Parameter(
                torch.empty(group_size, out_features, dtype=bias_dtype or dtype),
                requires_grad=False,
            )
        else:
            self.bias = None
        if dtype.itemsize == 1:
            block_size = 128
            if is_fp4:
                quant_scale_stride = 8
                scale_out_features = out_features
                scale_in_features = (
                    local_in_features + quant_scale_stride - 1
                ) // quant_scale_stride
                self.input_scale = nn.Parameter(
                    torch.empty(
                        group_size,
                        1,  # In FP4 quantized model, one Tensor has one scale_2
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                self.scale_2 = nn.Parameter(
                    torch.empty(
                        group_size,
                        1,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
            else:
                scale_out_features = (out_features + block_size - 1) // block_size
                scale_in_features = (local_in_features + block_size - 1) // block_size
            self.scale = nn.Parameter(
                torch.empty(
                    group_size,
                    scale_out_features,
                    scale_in_features,
                    dtype=torch.float32,
                ),
                requires_grad=False,
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
        is_fp4,
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
            self.wqkv_a = getLinearDeepSeekV3(False)(
                self.dim,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.get_default_dtype(),
            )
        else:
            self.wq_a = getLinearDeepSeekV3(False)(
                self.dim,
                self.q_lora_rank,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.get_default_dtype(),
            )
            self.wkv_a = getLinearDeepSeekV3(False)(
                self.dim,
                self.kv_lora_rank + self.qk_rope_head_dim,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.get_default_dtype(),
            )
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            (
                self.n_heads * self.qk_head_dim
                if self.mla_absorb != "absorb"
                else self.n_heads * (self.kv_lora_rank + self.qk_rope_head_dim)
            ),
            has_bias=False,
            dtype=parse_dtype(args.main_weight_dtype),
            bias_dtype=torch.get_default_dtype(),
            gather_output=False,
            base_linear_class=getLinearDeepSeekV3(False),
            disable_quantization=is_fp4,
        )
        self.kv_norm = RMSNorm(self.kv_lora_rank)

        if self.mla_absorb == "none":
            self.wkv_b = ColumnParallelLinear(
                self.kv_lora_rank,
                self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.get_default_dtype(),
                gather_output=False,
                base_linear_class=getLinearDeepSeekV3(False),
                disable_quantization=is_fp4,
            )
        elif self.mla_absorb == "absorb-without-precomp":
            self.wkv_b_absorb_1 = ParallelAbsorbGemm(
                self.n_heads,
                self.qk_nope_head_dim,
                self.kv_lora_rank,
                dtype=parse_dtype(args.main_weight_dtype),
                block_size=block_size,
            )
            self.wkv_b_absorb_2 = ParallelAbsorbGemm(
                self.n_heads,
                self.kv_lora_rank,
                self.v_head_dim,
                dtype=parse_dtype(args.main_weight_dtype),
                block_size=block_size,
            )

        self.wo = RowParallelLinear(
            (
                self.n_heads * self.v_head_dim
                if self.mla_absorb != "absorb"
                else self.n_heads * self.kv_lora_rank
            ),
            self.dim,
            has_bias=False,
            dtype=parse_dtype(args.main_weight_dtype),
            bias_dtype=torch.get_default_dtype(),
            input_is_parallel=True,
            base_linear_class=getLinearDeepSeekV3(False),
            disable_quantization=is_fp4,
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
        q = q.view(bs_seq, self.n_local_heads, -1)

        q_nope, q_pe = torch.split(
            q,
            [
                q.shape[-1] - self.qk_rope_head_dim,  # Depends on absorption mode
                self.qk_rope_head_dim,
            ],
            dim=-1,
        )
        kv_lora, k_pe = torch.split(
            kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # In-place update to `q_pe` and `k_pe`, which are part of `q` and `kv`, respectively
        apply_rotary_pos_emb(
            q_pe,
            k_pe,
            freqs_cis_cos,
            freqs_cis_sin,
            q_out=q_pe,
            k_out=k_pe,
            rotary_type="llama",
        )

        if self.mla_absorb == "none":
            kv = self.wkv_b(self.kv_norm(kv_lora))
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
            q_nope = self.wkv_b_absorb_1(q_nope)
            return q_nope, q_pe, kv
        elif self.mla_absorb == "absorb":
            return q_nope, q_pe, kv
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

        elif self.mla_absorb == "absorb-without-precomp" or self.mla_absorb == "absorb":
            q_nope, q_pe, kv = self._run_linear(x, freqs_cis_cos, freqs_cis_sin)

            kv_cache = kv[:, : self.kv_lora_rank]
            pe_cache = kv[:, -self.qk_rope_head_dim :]

            # In-place update to `kv_cache`, which is part of `kv`
            self.kv_norm(kv_cache, compute_dtype=kv.dtype, out=kv_cache)

            self.cache.finalize_cache_bylayer_prefill(
                kv,
                None,
                self.cache.curr_req_ids,
                self.cache.curr_varlens,
                self.layer_id,
            )
            q_nope_pe = torch.cat([q_nope, q_pe], dim=-1)
            x = self.attn_backend.attn_varlen_func(
                q_nope_pe.view(-1, q_nope_pe.shape[-2], q_nope_pe.shape[-1]),
                kv.view(-1, 1, kv.shape[-1]),
                kv_cache.view(-1, 1, kv_cache.shape[-1]),
                varlens.prefix_lens,
                varlens.prefix_lens,
                varlens.max_len,
                varlens.max_len,
                causal=True,
                softmax_scale=self.softmax_scale,
            )

            x = x.view(bs_seq, x.shape[-2], x.shape[-1])
            if self.mla_absorb == "absorb-without-precomp":
                x = self.wkv_b_absorb_2(x)

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
        cache_seqlens_incl_this_decode = self.cache.get_gpu_seq_lens_incl_this_decode()
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

        elif self.mla_absorb == "absorb-without-precomp" or self.mla_absorb == "absorb":
            q_nope, q_pe, kv = self._run_linear(
                x.view(bsz * seqlen, -1), freqs_cis_cos, freqs_cis_sin
            )

            kv_cache, _ = self.cache.get_cache_decode(self.layer_id)
            this_kv = kv[..., : self.kv_lora_rank]

            # In-place update to `this_kv`, which is part of `kv`
            self.kv_norm(this_kv, compute_dtype=kv.dtype, out=this_kv)

            x = self.attn_backend.mla_attn_with_kvcache(
                q_nope,
                q_pe,
                kv_cache,
                kv.view(bsz, seqlen, 1, -1),
                cache_seqlens_excl_this_decode=cache_seqlens_excl_this_decode,
                cache_seqlens_incl_this_decode=cache_seqlens_incl_this_decode,
                block_table=None,
                softmax_scale=self.softmax_scale,
            )

            if self.mla_absorb == "absorb-without-precomp":
                x = self.wkv_b_absorb_2(x)

        else:
            raise NotImplementedError(
                f"MLA absorb mode {self.mla_absorb} not supported"
            )

        x = self._run_output_linear(x)
        return x

    def decode_forward_paged(
        self, x: torch.Tensor, freqs_cis_cos: torch.Tensor, freqs_cis_sin: torch.Tensor
    ):
        assert (
            self.mla_absorb == "absorb-without-precomp" or self.mla_absorb == "absorb"
        )

        cache_seqlens_excl_this_decode = self.cache.get_gpu_seq_lens_excl_this_decode()
        cache_seqlens_incl_this_decode = self.cache.get_gpu_seq_lens_incl_this_decode()
        bsz, seqlen, _ = x.size()
        q_nope, q_pe, kv = self._run_linear(
            x.view(bsz * seqlen, -1), freqs_cis_cos, freqs_cis_sin
        )

        block_table = self.cache.get_gpu_block_table()
        paged_kv_cache, _ = self.cache.get_paged_kv_cache(self.layer_id)
        this_kv = kv[..., : self.kv_lora_rank]

        # In-place update to `this_kv`, which is part of `kv`
        self.kv_norm(this_kv, compute_dtype=kv.dtype, out=this_kv)

        x = self.attn_backend.mla_attn_with_kvcache(
            q_nope,
            q_pe,
            paged_kv_cache,
            kv.view(bsz, seqlen, 1, -1),
            cache_seqlens_excl_this_decode=cache_seqlens_excl_this_decode,
            cache_seqlens_incl_this_decode=cache_seqlens_incl_this_decode,
            block_table=block_table,
            softmax_scale=self.softmax_scale,
        )
        if self.mla_absorb == "absorb-without-precomp":
            x = self.wkv_b_absorb_2(x)
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

    def __init__(self, args, merge_gate_up: bool, is_fp4: bool = False):
        super().__init__()
        self.merge_gate_up = merge_gate_up

        if merge_gate_up:
            self.w1w3 = ColumnParallelLinear(
                args.dim // 2 if is_fp4 else args.dim,
                args.inter_dim * 2,
                has_bias=False,
                dtype=parse_dtype(
                    args.main_weight_dtype, is_quant_layer=True
                ),  # In fp4 quantization, dtype of MLP is float4_e2m1
                bias_dtype=torch.get_default_dtype(),
                gather_output=False,
                base_linear_class=getLinearDeepSeekV3(),
            )
            if is_fp4:
                self.w1w3.register_scale_2_param(
                    2
                )  # In merge gate up computation, scale_2 will also be cat into one Tensor
        else:
            self.w1 = ColumnParallelLinear(
                args.dim // 2 if is_fp4 else args.dim,
                args.inter_dim,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype, is_quant_layer=True),
                bias_dtype=torch.get_default_dtype(),
                gather_output=False,
                base_linear_class=getLinearDeepSeekV3(),
            )
            self.w3 = ColumnParallelLinear(
                args.dim // 2 if is_fp4 else args.dim,
                args.inter_dim,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype, is_quant_layer=True),
                bias_dtype=torch.get_default_dtype(),
                gather_output=False,
                base_linear_class=getLinearDeepSeekV3(),
            )
            if is_fp4:
                self.w1.register_scale_2_param()
                self.w3.register_scale_2_param()
        self.w2 = RowParallelLinear(
            args.inter_dim // 2 if is_fp4 else args.inter_dim,
            args.dim,
            has_bias=False,
            dtype=parse_dtype(args.main_weight_dtype, is_quant_layer=True),
            bias_dtype=torch.get_default_dtype(),
            input_is_parallel=True,
            base_linear_class=getLinearDeepSeekV3(),
        )
        if is_fp4:
            self.w2.register_scale_2_param()

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
            return self.w2(silu_and_mul(w1w3_out))
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
        self.is_fp4 = args.main_weight_dtype == "float4_e2m1"
        self.bias = (
            nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))
            if self.dim == 7168
            else None
        )

    def is_fused_sigmoid_gate(self):
        return self.score_func == "sigmoid" and self.n_groups > 1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = F.linear(x, self.weight)
        if self.is_fused_sigmoid_gate() and is_nvidia():
            indices, weights = fused_sigmoid_gate(
                scores, self.topk, self.n_groups, self.topk_groups, self.bias
            )
        else:
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
        return weights.type_as(x), indices.to(torch.int32)


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

    def __init__(self, args, merge_gate_up: bool, is_fp4: bool = False):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.merge_gate_up = merge_gate_up
        self.dim = args.dim
        self.is_fp4 = is_fp4

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
                args.dim // 2 if is_fp4 else args.dim,
                args.moe_inter_dim * 2,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype, is_quant_layer=True),
                is_fp4=is_fp4,
                bias_dtype=torch.get_default_dtype(),
                gather_output=False,
                scale_2_dim=2,
            )
        else:
            self.w1 = GroupColumnParallelLinearDeepSeekV3(
                self.experts_end_idx - self.experts_start_idx + self.n_shared_experts,
                args.dim // 2 if is_fp4 else args.dim,
                args.moe_inter_dim,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype, is_quant_layer=True),
                is_fp4=is_fp4,
                bias_dtype=torch.get_default_dtype(),
                gather_output=False,
            )
            self.w3 = GroupColumnParallelLinearDeepSeekV3(
                self.experts_end_idx - self.experts_start_idx + self.n_shared_experts,
                args.dim // 2 if is_fp4 else args.dim,
                args.moe_inter_dim,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype, is_quant_layer=True),
                is_fp4=is_fp4,
                bias_dtype=torch.get_default_dtype(),
                gather_output=False,
            )
        self.w2 = GroupRowParallelLinearDeepSeekV3(
            self.experts_end_idx - self.experts_start_idx + self.n_shared_experts,
            args.moe_inter_dim // 2 if is_fp4 else args.moe_inter_dim,
            args.dim,
            has_bias=False,
            dtype=parse_dtype(args.main_weight_dtype, is_quant_layer=True),
            is_fp4=is_fp4,
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

    def get_expert_weight_scale_2_for_fp4(self, expert_num):
        w1w3_weight_scale2 = self.w1w3.scale_2[:expert_num]
        w2_scale2 = self.w2.scale_2[:expert_num]
        return w1w3_weight_scale2, w2_scale2

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
                w1w3_scale_2, w2_scale_2 = None, None
                use_fp4_w4a8 = False
                use_fp8_w8a8 = False
                fused_soft_fp8 = False
            else:
                assert self.w1w3.scale is not None
                assert self.w2.scale is not None
                if not get_global_args().infer.raise_lower_bit_float_to == "bfloat16":
                    w1w3_weight, w1w3_scale, w2_weight, w2_scale = (
                        self.get_expert_weights_for_fp8_w8a8(
                            self.n_routed_experts + shared_experts
                        )
                    )
                    if self.is_fp4:
                        w1w3_scale_2, w2_scale_2 = (
                            self.get_expert_weight_scale_2_for_fp4(
                                expert_num=self.n_routed_experts + shared_experts
                            )
                        )
                        use_fp4_w4a8 = True
                        use_fp8_w8a8 = False
                        fused_soft_fp8 = False
                    else:
                        w1w3_scale_2, w2_scale_2 = None, None
                        use_fp4_w4a8 = False
                        use_fp8_w8a8 = True
                        fused_soft_fp8 = False
                elif is_nvidia() or is_muxi():
                    w1w3_weight, w1w3_scale, w2_weight, w2_scale = (
                        self.get_expert_weights_for_fp8_w8a8(
                            self.n_routed_experts + shared_experts
                        )
                    )
                    if self.is_fp4:
                        w1w3_scale_2, w2_scale_2 = (
                            self.get_expert_weight_scale_2_for_fp4(
                                expert_num=self.n_routed_experts + shared_experts
                            )
                        )
                        use_fp4_w4a8 = True
                        use_fp8_w8a8 = False
                        fused_soft_fp8 = True
                    else:
                        w1w3_scale_2, w2_scale_2 = None, None
                        use_fp4_w4a8 = False
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
                    w1w3_scale_2 = None
                    w2_weight = weight_dequant_soft_fp8_deepseek_v3(
                        self.w2.weight[: self.n_routed_experts + self.n_shared_experts],
                        self.w2.scale[: self.n_routed_experts + self.n_shared_experts],
                        block_size,
                    )
                    w2_scale = None
                    w2_scale_2 = None
                    use_fp4_w4a8 = False
                    use_fp8_w8a8 = False
                    fused_soft_fp8 = False

            if not get_global_args().infer.fuse_shared_experts:
                if use_fp4_w4a8:
                    w1w3_out = linear_deepseek_v3(
                        x,
                        self.w1w3.weight[-1],
                        self.w1w3.scale[-1] if self.w1w3.scale is not None else None,
                        self.w1w3.bias[-1] if self.w1w3.bias is not None else None,
                        (
                            self.w1w3.scale_2[-1]
                            if self.w1w3.scale_2 is not None
                            else None
                        ),
                    )
                    act = silu_and_mul(w1w3_out)
                    y1 = linear_deepseek_v3(
                        act,
                        self.w2.weight[-1],
                        self.w2.scale[-1] if self.w2.scale is not None else None,
                        self.w2.bias[-1] if self.w2.bias is not None else None,
                        self.w2.scale_2[-1] if self.w2.scale_2 is not None else None,
                    )
                else:
                    w1w3_out = linear_deepseek_v3(
                        x,
                        self.w1w3.weight[-1],
                        self.w1w3.scale[-1] if self.w1w3.scale is not None else None,
                        self.w1w3.bias[-1] if self.w1w3.bias is not None else None,
                    )
                    act = silu_and_mul(w1w3_out)
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
                    use_fp4_w4a8=use_fp4_w4a8,
                    inplace=True,
                    global_num_experts=self.n_routed_experts + self.n_shared_experts,
                    expert_map=None,  # use when ep > 1
                    w1_scale=w1w3_scale,
                    w2_scale=w2_scale,
                    w1w3_scale_2=w1w3_scale_2,
                    w2_scale_2=w2_scale_2,
                    block_shape=[128, 128],
                    soft_fp8=fused_soft_fp8,
                )

                y += y1
            else:

                indice_shape = indices.shape
                new_indices = torch.empty(
                    (indice_shape[0], indice_shape[1] + 1),
                    dtype=indices.dtype,
                    device=indices.device,
                )

                new_weights = torch.empty(
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
                    use_fp4_w4a8=use_fp4_w4a8,
                    inplace=True,
                    global_num_experts=self.n_routed_experts + self.n_shared_experts,
                    expert_map=None,  # use when ep > 1
                    w1_scale=w1w3_scale,
                    w2_scale=w2_scale,
                    w1w3_scale_2=w1w3_scale_2,
                    w2_scale_2=w2_scale_2,
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
                act = [
                    (silu_and_mul(w1w3_out) if w1w3_out is not None else None)
                    for w1w3_out in w1w3_outs
                ]
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


class MoEDeepSeekV3CPU(nn.Module):
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

    def __init__(self, args, cpu_infer, ggml_type, merge_gate_up: bool):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.merge_gate_up = merge_gate_up
        self.dim = args.dim
        self.tp_group = get_tp_group()
        self.tp_size = get_tp_size()
        self.rank = get_tp_rank()

        moe_world_size = 1
        self.max_batch_size = get_global_args().infer.max_reqs
        assert (
            args.n_routed_experts % moe_world_size == 0
        ), f"Number of experts must be divisible by world size (world_size={moe_world_size})"
        self.n_shared_experts = args.n_shared_experts
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // moe_world_size
        self.n_activated_experts = args.n_activated_experts
        self.gate = GateDeepSeekV3(args)
        if merge_gate_up:
            self.w1w3 = ColumnParallelLinear(
                args.dim,
                args.moe_inter_dim * 2,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.bfloat16,
                gather_output=False,
            )
        else:
            self.w1 = ColumnParallelLinear(
                args.dim,
                args.moe_inter_dim,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.bfloat16,
                gather_output=False,
                base_linear_class=getLinearDeepSeekV3(),
            )
            self.w3 = ColumnParallelLinear(
                args.dim,
                args.moe_inter_dim,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.bfloat16,
                gather_output=False,
                base_linear_class=getLinearDeepSeekV3(),
            )
        self.w2 = RowParallelLinear(
            args.moe_inter_dim,
            args.dim,
            has_bias=False,
            dtype=parse_dtype(args.main_weight_dtype),
            bias_dtype=torch.bfloat16,
            input_is_parallel=True,
            base_linear_class=getLinearDeepSeekV3(),
        )

        if self.rank == 0:

            self.register_buffer(
                "gate_proj",
                torch.empty(
                    int(256 * 2048 * 7168 / 256 * 144),
                    dtype=torch.uint8,
                    device="cpu",
                    requires_grad=False,
                ),
            )
            self.register_buffer(
                "up_proj",
                torch.empty(
                    int(256 * 2048 * 7168 / 256 * 144),
                    dtype=torch.uint8,
                    device="cpu",
                    requires_grad=False,
                ),
            )
            if ggml_type == 12:
                self.register_buffer(
                    "down_proj",
                    torch.empty(
                        int(256 * 2048 * 7168 / 256 * 144),
                        dtype=torch.uint8,
                        device="cpu",
                        requires_grad=False,
                    ),
                )
            elif ggml_type == 14:
                self.register_buffer(
                    "down_proj",
                    torch.empty(
                        int(256 * 2048 * 7168 / 256 * 210),
                        dtype=torch.uint8,
                        device="cpu",
                        requires_grad=False,
                    ),
                )
            else:
                raise ValueError("ggml quantization type unimplemented !")

            self.register_buffer(
                "gate_type",
                torch.empty(
                    1,
                    dtype=torch.int,
                    device="cpu",
                    requires_grad=False,
                ),
            )
            self.register_buffer(
                "up_type",
                torch.empty(
                    1,
                    dtype=torch.int,
                    device="cpu",
                    requires_grad=False,
                ),
            )
            self.register_buffer(
                "down_type",
                torch.empty(
                    1,
                    dtype=torch.int,
                    device="cpu",
                    requires_grad=False,
                ),
            )

        self.stride = 64
        self.cpu_infer = cpu_infer
        self.moe = None

    def to(self, *args, **kwargs):
        self.gate.to(*args, **kwargs)
        if self.merge_gate_up:
            self.w1w3.to(*args, **kwargs)
        else:
            self.w1.to(*args, **kwargs)
            self.w3.to(*args, **kwargs)
        self.w2.to(*args, **kwargs)
        return self

    def init_weights(self):
        if self.rank == 0:
            gate_ptr = ctypes.addressof(
                ctypes.cast(
                    self.gate_proj.data_ptr(), ctypes.POINTER(ctypes.c_uint64)
                ).contents
            )
            up_ptr = ctypes.addressof(
                ctypes.cast(
                    self.up_proj.data_ptr(), ctypes.POINTER(ctypes.c_uint64)
                ).contents
            )
            down_ptr = ctypes.addressof(
                ctypes.cast(
                    self.down_proj.data_ptr(), ctypes.POINTER(ctypes.c_uint64)
                ).contents
            )

            import cpuinfer

            moe_config = cpuinfer.moe.MOEConfig(
                256,
                8,
                7168,
                2048,
                self.stride,
                10,
                1024,
                gate_ptr,
                up_ptr,
                down_ptr,
                self.gate_type.item(),
                self.up_type.item(),
                self.down_type.item(),
                30,
            )

            self.moe = cpuinfer.moe.MOE(moe_config)

            # warm up
            self.cpu_infer.submit(self.moe.warm_up())
            self.cpu_infer.sync()

            self.input_tensor_cpu = torch.empty(
                (self.max_batch_size, 1, 7168),
                device="cpu",
                pin_memory=True,
                dtype=torch.bfloat16,
            )
            self.weights_cpu = torch.empty(
                (self.max_batch_size, 8),
                device="cpu",
                pin_memory=True,
                dtype=torch.float32,
            )
            self.indices_cpu = torch.empty(
                (self.max_batch_size, 8),
                device="cpu",
                pin_memory=True,
                dtype=torch.int64,
            )
            self.output_cpu = torch.empty(
                (self.max_batch_size, 1, 7168),
                device="cpu",
                pin_memory=True,
                dtype=torch.bfloat16,
            )
            self.output_gpu = torch.empty(
                (self.max_batch_size, 1, 7168), device=self.rank, dtype=torch.bfloat16
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()

        if self.rank == 0:
            x_flat = x.view(-1, self.dim)
            weights, indices = self.gate(x_flat)
            indices = indices.contiguous().to(torch.int64)
            weights = weights.contiguous().to(torch.float32)
            if x.shape[1] > 1:
                input_tensor = x.contiguous().cpu()
                indices = indices.cpu()
                weights = weights.cpu()
                output = torch.empty_like(input_tensor).contiguous().pin_memory()
                self.cpu_infer.submit(
                    self.moe.forward(
                        indices.size(0),
                        indices.size(1),
                        indices.data_ptr(),
                        weights.data_ptr(),
                        input_tensor.data_ptr(),
                        output.data_ptr(),
                    )
                )
            else:
                self.input_tensor_cpu.copy_(x, non_blocking=True)
                self.indices_cpu.copy_(indices, non_blocking=True)
                self.weights_cpu.copy_(weights, non_blocking=True)
                self.cpu_infer.submit_with_cuda_stream(
                    torch.cuda.current_stream().cuda_stream,
                    self.moe.forward(
                        self.max_batch_size,
                        8,
                        self.indices_cpu.data_ptr(),
                        self.weights_cpu.data_ptr(),
                        self.input_tensor_cpu.data_ptr(),
                        self.output_cpu.data_ptr(),
                    ),
                )

        if self.merge_gate_up:
            w1w3_out = self.w1w3(x)
            w1_out, w3_out = torch.split(w1w3_out, w1w3_out.shape[-1] // 2, dim=-1)
        else:
            w1_out = self.w1(x)
            w3_out = self.w3(x)
        y = self.w2(F.silu(w1_out) * w3_out)

        if self.rank == 0:
            if x.shape[1] > 1:
                self.cpu_infer.sync()
                output = output.to(x.device, non_blocking=True).view(shape)
                y += output
            else:
                self.cpu_infer.sync_with_cuda_stream(
                    torch.cuda.current_stream().cuda_stream
                )
                self.output_gpu.copy_(self.output_cpu, non_blocking=True)
                y += self.output_gpu
            y_scatter = [y] * self.tp_size
        else:
            y_scatter = None

        torch.distributed.scatter(y, y_scatter, src=0)
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
        cpu_infer=False,
        ggml_type=0,
        is_fp4=False,
    ):
        super().__init__(
            layer_id, args, cache, attn_backend=attn_backend, op_impl=op_impl
        )
        self.layer_id = layer_id
        self.attn = AttentionDeepSeekV3(
            args,
            layer_id,
            cache,
            attn_backend,
            mla_absorb=mla_absorb,
            merge_qkv=merge_qkv_gate_up,
            is_fp4=is_fp4,
        )
        self.ffn = (
            MLPDeepSeekV3(
                args,
                merge_gate_up=merge_qkv_gate_up,
                is_fp4=is_fp4,
            )
            if layer_id < args.n_dense_layers
            else (
                MoEDeepSeekV3(
                    args,
                    merge_gate_up=merge_qkv_gate_up,
                    is_fp4=is_fp4,
                )
                if not cpu_infer
                else MoEDeepSeekV3CPU(
                    args,
                    cpu_infer=cpu_infer,
                    ggml_type=ggml_type,
                    merge_gate_up=merge_qkv_gate_up,
                )
            )
        )
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def to(self, *args, **kwargs):
        self.attn.to(*args, **kwargs)
        self.ffn.to(*args, **kwargs)
        self.attn_norm.to(*args, **kwargs)
        self.ffn_norm.to(*args, **kwargs)
        return self

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
        cpu_infer,
        cpu_layers: List,
        ggml_type: List,
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
        self.cpu_layers = cpu_layers
        self.ggml_type = ggml_type
        self.cpu_infer = cpu_infer
        super().__init__(
            params,
            cache,
            max_position_embeddings=max_position_embeddings,
            pipeline_parallel_size=pipeline_parallel_size,
            model_parallel_size=model_parallel_size,
            attn_backend=attn_backend,
            op_impl=op_impl,
            mla_absorb=mla_absorb,
            is_fp4=(get_global_args().models.main_weight_dtype == "float4_e2m1"),
        )
        if op_impl != "torch":
            raise NotImplementedError("Only op_impl=torch is supported in DeepSeek V3")

    def to(self, *args, **kwargs):
        if hasattr(self, "embed"):
            self.embed.to(*args, **kwargs)
        if hasattr(self, "norm"):
            self.norm.to(*args, **kwargs)
        if hasattr(self, "head"):
            self.head.to(*args, **kwargs)
        for l in self.layers:
            l.to(*args, **kwargs)
        return self

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
                for part in ["weight", "scale", "bias", "input_scale", "scale_2"]:
                    if k.endswith(f".experts.0.{w}.{part}"):
                        prefix = k[: -len(f"experts.0.{w}.{part}")]
                        parts = []
                        for i in range(self.params.n_routed_experts):
                            parts.append(checkpoint[prefix + f"experts.{i}.{w}.{part}"])
                        parts.append(checkpoint[prefix + f"shared_experts.{w}.{part}"])
                        if (
                            len(parts) > 0
                            and parts[0].element_size() == 1
                            and not has_native_fp8()
                        ):
                            new_checkpoint[prefix + f"{w}.{part}"] = torch.stack(
                                [p.view(torch.uint8) for p in parts], dim=0
                            ).view(parts[0].dtype)
                        else:
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

    def _process_state_dict_for_absorption_without_precomputation(
        self, checkpoint: Mapping[str, Any]
    ):
        model_parallel_size = get_tp_size()
        n_local_heads = self.params.n_heads // model_parallel_size
        block_size = 128

        new_checkpoint = {}
        for k in checkpoint.keys():
            if k.endswith(".wkv_b.weight"):
                prefix = k[: -len("wkv_b.weight")]
                wkv_b_weight = checkpoint[prefix + "wkv_b.weight"]
                wkv_b_weight = wkv_b_weight.view(
                    n_local_heads, -1, wkv_b_weight.shape[-1]
                )
                wkv_b_absorb_1_weight = wkv_b_weight[:, : self.params.qk_nope_head_dim]
                wkv_b_absorb_2_weight = wkv_b_weight[:, self.params.qk_nope_head_dim :]
                new_checkpoint[prefix + "wkv_b_absorb_1.weight"] = (
                    wkv_b_absorb_1_weight.permute(0, 2, 1).contiguous()
                )
                new_checkpoint[prefix + "wkv_b_absorb_2.weight"] = wkv_b_absorb_2_weight

                is_fp8 = wkv_b_weight.element_size() == 1
                if is_fp8:
                    wkv_b_scale = checkpoint[prefix + "wkv_b.scale"]
                    wkv_b_scale = wkv_b_scale.view(
                        n_local_heads, -1, wkv_b_scale.shape[-1]
                    )
                    wkv_b_absorb_1_scale = wkv_b_scale[
                        :, : self.params.qk_nope_head_dim // block_size
                    ]
                    wkv_b_absorb_2_scale = wkv_b_scale[
                        :, self.params.qk_nope_head_dim // block_size :
                    ]
                    new_checkpoint[prefix + "wkv_b_absorb_1.scale"] = (
                        wkv_b_absorb_1_scale.permute(0, 2, 1).contiguous()
                    )
                    new_checkpoint[prefix + "wkv_b_absorb_2.scale"] = (
                        wkv_b_absorb_2_scale
                    )
                else:
                    assert prefix + "wkv_b.scale" not in checkpoint

            elif k.endswith(".wkv_b.scale"):
                continue

            elif k.endswith(".wkv_b.bias"):
                raise NotImplementedError(
                    "infer.mla_absorb=absorb-without-precomp is not implemented for wkv_b with a bias"
                )

            else:
                new_checkpoint[k] = checkpoint[k]

        return new_checkpoint

    def _process_state_dict_for_absorption(self, checkpoint: Mapping[str, Any]):
        model_parallel_size = get_tp_size()
        n_local_heads = self.params.n_heads // model_parallel_size
        weight_dequant_fn = (
            weight_dequant_soft_fp8_deepseek_v3
            if get_global_args().infer.raise_lower_bit_float_to == "bfloat16"
            else weight_dequant_deepseek_v3
        )
        block_size = 128

        new_checkpoint = {}
        for k in checkpoint.keys():
            if k.endswith(".wkv_b.weight"):
                prefix = k[: -len("wkv_b.weight")]
                assert prefix + "wkv_b.weight" in checkpoint
                wkv_b_ckpt_weight = checkpoint[prefix + "wkv_b.weight"]
                is_fp8 = wkv_b_ckpt_weight.element_size() == 1
                if is_fp8:
                    assert prefix + "wkv_b.scale" in checkpoint
                    wkv_b_scale = checkpoint[prefix + "wkv_b.scale"]
                    # FIXME: Keep this on GPU
                    wkv_b_weight = weight_dequant_fn(
                        wkv_b_ckpt_weight.cuda(), wkv_b_scale.cuda(), block_size
                    ).cpu()
                else:
                    wkv_b_weight = wkv_b_ckpt_weight
                wkv_b_weight = wkv_b_weight.view(
                    n_local_heads,
                    self.params.qk_nope_head_dim + self.params.v_head_dim,
                    self.params.kv_lora_rank,
                )

                # Absorb into wq_b
                wq_b_ckpt_weight = checkpoint[prefix + "wq_b.weight"]
                if is_fp8:
                    assert prefix + "wq_b.scale" in checkpoint
                    wq_b_scale = checkpoint[prefix + "wq_b.scale"]
                    # FIXME: Keep this on GPU
                    wq_b_weight = weight_dequant_fn(
                        wq_b_ckpt_weight.cuda(), wq_b_scale.cuda(), block_size
                    ).cpu()
                else:
                    wq_b_weight = wq_b_ckpt_weight
                wq_b_weight_per_head = wq_b_weight.view(
                    n_local_heads,
                    self.params.qk_nope_head_dim + self.params.qk_rope_head_dim,
                    self.params.q_lora_rank,
                )
                wq_b_nope = wq_b_weight_per_head[:, : self.params.qk_nope_head_dim]
                wq_b_rope = wq_b_weight_per_head[:, self.params.qk_nope_head_dim :]
                #   x @ wq_b_nope^T @ per_head(wkv_b[:, :qk_nope_head_dim, :])
                # = x @ (per_head(wkv_b[:, :qk_nope_head_dim, :])^T @ wq_b_nope)^T
                wkv_b_for_wq_b = wkv_b_weight[:, : self.params.qk_nope_head_dim]
                assert wkv_b_for_wq_b.shape == (
                    n_local_heads,
                    self.params.qk_nope_head_dim,
                    self.params.kv_lora_rank,
                )
                wkv_b_for_wq_b = torch.block_diag(*wkv_b_for_wq_b)
                new_wq_b_nope = (
                    wkv_b_for_wq_b.t()
                    @ wq_b_nope.contiguous().view(-1, self.params.q_lora_rank)
                ).view(n_local_heads, self.params.kv_lora_rank, self.params.q_lora_rank)
                new_wq_b = torch.cat([new_wq_b_nope, wq_b_rope], dim=1).view(
                    -1, self.params.q_lora_rank
                )
                if is_fp8:
                    new_wq_b, new_wq_b_scale = weight_quant_deepseek_v3(
                        new_wq_b, block_size
                    )
                    new_checkpoint[prefix + "wq_b.weight"] = new_wq_b
                    new_checkpoint[prefix + "wq_b.scale"] = new_wq_b_scale
                else:
                    new_checkpoint[prefix + "wq_b.weight"] = new_wq_b

                # Absorb into wo
                wo_ckpt_weight = checkpoint[prefix + "wo.weight"]
                if is_fp8:
                    assert prefix + "wo.scale" in checkpoint
                    wo_scale = checkpoint[prefix + "wo.scale"]
                    # FIXME: Keep this on GPU
                    wo_weight = weight_dequant_fn(
                        wo_ckpt_weight.cuda(), wo_scale.cuda(), block_size
                    ).cpu()
                else:
                    wo_weight = wo_ckpt_weight
                #   x @ per_head(wkv_b_weight[:, -params.v_head_dim :, :]^T) @ wo_weight^T
                # = x @ (wo_weight @ per_head(wkv_b_weight[:, -params.v_head_dim :, :]))^T
                wkv_b_for_wo = wkv_b_weight[:, -self.params.v_head_dim :]
                assert wkv_b_for_wo.shape == (
                    n_local_heads,
                    self.params.v_head_dim,
                    self.params.kv_lora_rank,
                )
                wkv_b_for_wo = torch.block_diag(*wkv_b_for_wo)
                new_wo = wo_weight @ wkv_b_for_wo
                if is_fp8:
                    new_wo, new_wo_scale = weight_quant_deepseek_v3(new_wo, block_size)
                    new_checkpoint[prefix + "wo.weight"] = new_wo
                    new_checkpoint[prefix + "wo.scale"] = new_wo_scale
                else:
                    new_checkpoint[prefix + "wo.weight"] = new_wo

            elif k.endswith(".wkv_b.scale"):
                continue

            elif k.endswith(".wkv_b.bias"):
                raise NotImplementedError(
                    "infer.mla_absorb=absorb is not implemented for wkv_b with a bias"
                )

            elif k.endswith(".wo.weight") or k.endswith(".wo.scale"):
                continue

            elif k.endswith(".wq_b.weight") or k.endswith(".wq_b.scale"):
                continue

            elif k.endswith(".wq_b.bias"):
                raise NotImplementedError(
                    "infer.mla_absorb=absorb is not implemented for wq_b with a bias"
                )

            else:
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
                if q_weight.element_size() == 1 and not has_native_fp8():
                    new_checkpoint[prefix + "wqkv_a.weight"] = torch.cat(
                        [q_weight.view(torch.uint8), kv_weight.view(torch.uint8)], dim=0
                    ).view(q_weight.dtype)
                else:
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
                if q_scale.element_size() == 1 and not has_native_fp8():
                    new_checkpoint[prefix + "wqkv_a.scale"] = torch.cat(
                        [q_scale.view(torch.uint8), kv_scale.view(torch.uint8)], dim=0
                    ).view(q_scale.dtype)
                else:
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
                if q_bias.element_size() == 1 and not has_native_fp8():
                    new_checkpoint[prefix + "wqkv_a.bias"] = torch.cat(
                        [q_bias.view(torch.uint8), kv_bias.view(torch.uint8)], dim=0
                    ).view(q_bias.dtype)
                else:
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
                if gate_weight.element_size() == 1 and not has_native_fp8():
                    new_checkpoint[prefix + "w1w3.weight"] = torch.cat(
                        [gate_weight.view(torch.uint8), up_weight.view(torch.uint8)],
                        dim=0,
                    ).view(gate_weight.dtype)
                else:
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
                if gate_scale.element_size() == 1 and not has_native_fp8():
                    new_checkpoint[prefix + "w1w3.scale"] = torch.cat(
                        [gate_scale.view(torch.uint8), up_scale.view(torch.uint8)],
                        dim=0,
                    ).view(gate_scale.dtype)
                else:
                    new_checkpoint[prefix + "w1w3.scale"] = torch.cat(
                        [gate_scale, up_scale], dim=0
                    )
            elif k.endswith(".w3.scale"):
                continue
            elif k.endswith(".w1.scale_2"):
                prefix = k[: -len("w1.scale_2")]
                assert prefix + "w3.scale_2" in checkpoint
                assert prefix + "w1w3.scale_2" not in checkpoint
                gate_scale = checkpoint[prefix + "w1.scale_2"]
                up_scale = checkpoint[prefix + "w3.scale_2"]
                if gate_scale.element_size() == 1 and not has_native_fp8():
                    new_checkpoint[prefix + "w1w3.scale_2"] = torch.cat(
                        [gate_scale, up_scale],
                        dim=0,
                    ).view(gate_scale.dtype)
                else:
                    new_checkpoint[prefix + "w1w3.scale_2"] = torch.cat(
                        [gate_scale, up_scale], dim=0
                    )
            elif k.endswith(".w3.scale_2"):
                continue
            elif k.endswith(".w1.input_scale"):
                prefix = k[: -len("w1.input_scale")]
                assert prefix + "w3.input_scale" in checkpoint
                assert prefix + "w1w3.input_scale" not in checkpoint
                gate_scale = checkpoint[prefix + "w1.input_scale"]
                up_scale = checkpoint[prefix + "w3.input_scale"]
                if gate_scale.element_size() == 1 and not has_native_fp8():
                    new_checkpoint[prefix + "w1w3.input_scale"] = torch.cat(
                        [gate_scale, up_scale],
                        dim=0,
                    ).view(gate_scale.dtype)
                else:
                    new_checkpoint[prefix + "w1w3.input_scale"] = torch.cat(
                        [gate_scale, up_scale], dim=0
                    )
            elif k.endswith(".w3.input_scale"):
                continue
            elif k.endswith(".w1.bias"):
                prefix = k[: -len("w1.bias")]
                assert prefix + "w3.bias" in checkpoint
                assert prefix + "w1w3.bias" not in checkpoint
                gate_bias = checkpoint[prefix + "w1.bias"]
                up_bias = checkpoint[prefix + "w3.bias"]
                if gate_bias.element_size() == 1 and not has_native_fp8():
                    new_checkpoint[prefix + "w1w3.bias"] = torch.cat(
                        [gate_bias.view(torch.uint8), up_bias.view(torch.uint8)], dim=0
                    ).view(gate_bias.dtype)
                else:
                    new_checkpoint[prefix + "w1w3.bias"] = torch.cat(
                        [gate_bias, up_bias], dim=0
                    )
            elif k.endswith(".w3.bias"):
                continue
            else:
                new_checkpoint[k] = checkpoint[k]
        return new_checkpoint

    def _process_state_dict_for_fp4_model(self, checkpoint: Mapping[str, Any]):
        new_checkpoint = {}
        for k in checkpoint.keys():
            param = checkpoint[k]
            if param.dtype == torch.uint8:
                param.data = chitu_backend.weight_layout_change(param.data.cuda()).cpu()
            if param.dtype == torch.float8_e4m3fn:
                param.data = param.data.to(torch.bfloat16)
            if "scale_2" in k or "input_scale" in k:
                param.data = param.data.unsqueeze(0)
            new_checkpoint[k] = param
        return new_checkpoint

    @override
    def load_state_dict_parallel(
        self,
        state_dict: Mapping[str, Any],
        skip_preprocess: bool = False,
        replace=True,
        *args,
        **kwargs,
    ):
        if skip_preprocess and replace:
            new_state_dict = {}
            for k in state_dict.keys():
                name = k
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("weight_scale", "scale")
                name = name.replace("weight_scale_2", "scale_2")
                name = name.replace("e_score_correction_bias", "bias")
                key = name.split(".")[-2]
                mapping = {
                    "embed_tokens": ("embed", 0),
                    "input_layernorm": ("attn_norm", None),
                    "post_attention_layernorm": ("ffn_norm", None),
                    "q_proj": ("wq", 0),
                    "q_a_proj": ("wq_a", None),
                    "q_a_layernorm": ("q_norm", None),
                    "q_b_proj": ("wq_b", 0),
                    "kv_a_proj_with_mqa": ("wkv_a", None),
                    "kv_a_layernorm": ("kv_norm", None),
                    "kv_b_proj": ("wkv_b", 0),
                    "o_proj": ("wo", 1),
                    "gate": ("gate", None),
                    "gate_proj": ("w1", 0),
                    "down_proj": ("w2", 1),
                    "up_proj": ("w3", 0),
                    "norm": ("norm", None),
                    "lm_head": ("head", 0),
                    "scale": ("scale", None),
                    "input_scale": ("input_scale", None),
                    "scale_2": ("scale_2", None),
                }
                assert key in mapping, f"Key {key} not found in mapping"
                new_key, dim = mapping[key]
                name = name.replace(key, new_key)
                new_state_dict[name] = state_dict[k]
            state_dict = new_state_dict

        super().load_state_dict_parallel(
            state_dict, skip_preprocess=skip_preprocess, *args, **kwargs
        )

    @override
    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        skip_preprocess: bool = False,
        *args,
        **kwargs,
    ):
        if self.is_fp4:
            state_dict = self._process_state_dict_for_fp4_model(state_dict)
        if not skip_preprocess:

            if self.mla_absorb == "absorb":
                state_dict = self._process_state_dict_for_absorption(state_dict)
            elif self.mla_absorb == "absorb-without-precomp":
                state_dict = (
                    self._process_state_dict_for_absorption_without_precomputation(
                        state_dict
                    )
                )

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
        import logging
        from logging import getLogger

        logger = getLogger(__name__)
        import resource

        memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for layer_id in range(self.local_begin_layer_id, self.local_end_layer_id):
            logger.debug(
                f"initing layer : {layer_id}  cpu memory usage: {memory_usage / 1024**2} GB  gpu memory usage : RANK : {torch.cuda.current_device()} {torch.cuda.memory_allocated()/(1024**3)} GB"
            )
            if layer_id in self.cpu_layers:
                self.layers.append(
                    TransformerBlockDeepSeekV3(
                        layer_id,
                        self.params,
                        cache,
                        attn_backend,
                        self.op_impl,
                        mla_absorb=self.mla_absorb,
                        merge_qkv_gate_up=self.merge_qkv_gate_up,
                        cpu_infer=self.cpu_infer,
                        ggml_type=self.ggml_type[layer_id],
                    )
                )
            else:
                self.layers.append(
                    TransformerBlockDeepSeekV3(
                        layer_id,
                        self.params,
                        cache,
                        attn_backend,
                        self.op_impl,
                        mla_absorb=self.mla_absorb,
                        merge_qkv_gate_up=self.merge_qkv_gate_up,
                        is_fp4=self.is_fp4,
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
            disable_quantization=True,
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
