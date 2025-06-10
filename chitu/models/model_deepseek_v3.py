import math
import functools
from logging import getLogger
from typing import Any, List, Mapping, Optional, Set, Tuple
import re

import torch
import torch.distributed as dist
import torch.distributed
import torch.nn.functional as F
from torch import nn
from typing_extensions import override

from chitu.layers.gate import fused_sigmoid_gate
from chitu.attn_backend import AttnBackend
from chitu.cache_manager import PagedKVCacheManager
from chitu.device_type import (
    get_device_name,
    is_muxi,
    is_nvidia,
)
from chitu.global_vars import get_global_args
from chitu.models.model import (
    Attention,
    RMSNorm,
    Transformer,
    TransformerBlock,
    MoeGate,
    MoeBlock,
    MoeBlockRegistry,
)
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

from chitu.muxi_utils import (
    LinearLayoutContigXContigY,
    Blockfp8LinearLayoutContigXContigY,
    linear_layout_contig_x_contig_y,
    blockfp8_linear_layout_contig_x_contig_y,
    preprocess_weights_for_native_layout,
    get_muxi_padded_input,
    grouped_topk,
    muxi_fused_experts,
)
from chitu.utils import try_import_opt_dep, parse_dtype, ceil_div
from chitu.quantization import (
    linear_block_fp8,
    linear_block_fp4,
    QuantizationRegistry,
)

import ctypes

logger = getLogger(__name__)

triton, has_triton = try_import_opt_dep("triton", "triton")
chitu_backend, has_chitu_backend = try_import_opt_dep("chitu_backend", "chitu_backend")
torch_npu, has_torch_npu = try_import_opt_dep("torch_npu", "torch_npu")

if has_torch_npu:
    from chitu.npu_utils import fused_experts_npu

if has_triton:
    from chitu.fused_moe import fused_experts


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

        # Some platforms do not support float8, but we can run them with `infer.raise_lower_bit_float_to=bfloat16`.
        # However, we need to treat float8 items as uint8 first, to avoid the missing ops on these platforms.
        args = get_global_args()
        if (
            dtype.itemsize == 1
            and parse_dtype(args.infer.raise_lower_bit_float_to).itemsize > 1
        ):
            dtype = torch.uint8

        tp_size = get_tp_size()
        assert global_n_heads % tp_size == 0
        local_n_heads = global_n_heads // tp_size

        self.weight = torch.nn.Parameter(
            torch.empty(
                local_n_heads, out_features_per_head, in_features_per_head, dtype=dtype
            ),
            requires_grad=False,
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
                ),
                requires_grad=False,
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
            self.weight,
            self.scale,
            soft_fp8=(get_global_args().infer.raise_lower_bit_float_to == "bfloat16"),
        )

        if bs is not None:
            y = y.view(bs, seq, y.shape[-2], y.shape[-1])
        return y


class AttentionDeepSeekV3(Attention):
    def __init__(
        self,
        args,
        layer_id,
        cache,
        attn_backend,
        op_impl: str,
        mla_absorb,
        checkpoint_prefix: str,
    ):
        super().__init__(layer_id, cache, attn_backend)
        self.op_impl = op_impl
        self.mla_absorb = mla_absorb
        quant = None
        for rule in args.quant_config.rules:
            pattern = rule.get("regex")
            if pattern and re.search(pattern, checkpoint_prefix):
                quant = rule.type
                break
        self.merge_qkv = (
            quant in QuantizationRegistry._allowed_quant_for_merge_qkv_gate_up
        )

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

        if self.merge_qkv:
            # fp8 gemm can handle weights not divisible by block_size, but it does not hold
            # after merging for the output dimension, except for the last weight.
            assert self.q_lora_rank % block_size == 0
            self.wqkv_a = LocalLinear(
                self.dim,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                has_bias=False,
                dtype=(
                    torch.bfloat16
                    if args.quant_config.type == "blockfp4"
                    else parse_dtype(args.main_weight_dtype)
                ),
                bias_dtype=torch.get_default_dtype(),
                checkpoint_prefix=f"{checkpoint_prefix}.wqkv_a",
            )  # FIXME: Run this layer with muxi_layout_kernels
        else:
            self.q_a_proj = LocalLinear(
                self.dim,
                self.q_lora_rank,
                has_bias=False,
                dtype=(
                    torch.bfloat16
                    if args.quant_config.type == "blockfp4"
                    else parse_dtype(args.main_weight_dtype)
                ),
                bias_dtype=torch.get_default_dtype(),
                checkpoint_prefix=f"{checkpoint_prefix}.q_a_proj",
            )  # FIXME: Run this layer with muxi_layout_kernels
            self.kv_a_proj_with_mqa = LocalLinear(
                self.dim,
                self.kv_lora_rank + self.qk_rope_head_dim,
                has_bias=False,
                dtype=(
                    torch.bfloat16
                    if args.quant_config.type == "blockfp4"
                    else parse_dtype(args.main_weight_dtype)
                ),
                bias_dtype=torch.get_default_dtype(),
                checkpoint_prefix=f"{checkpoint_prefix}.kv_a_proj_with_mqa",
            )  # FIXME: Run this layer with muxi_layout_kernels
        self.q_a_layernorm = RMSNorm(self.q_lora_rank)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            (
                self.n_heads * self.qk_head_dim
                if self.mla_absorb != "absorb"
                else self.n_heads * (self.kv_lora_rank + self.qk_rope_head_dim)
            ),
            has_bias=False,
            dtype=(
                torch.bfloat16
                if args.quant_config.type == "blockfp4"
                else parse_dtype(args.main_weight_dtype)
            ),
            bias_dtype=torch.get_default_dtype(),
            gather_output=False,
            base_linear_class=get_linear_layout_contig_x_contig_y(
                op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.kv_b_proj",
            ),
            checkpoint_prefix=f"{checkpoint_prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)

        if self.mla_absorb == "none":
            self.kv_b_proj = ColumnParallelLinear(
                self.kv_lora_rank,
                self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
                has_bias=False,
                dtype=(
                    torch.bfloat16
                    if args.quant_config.type == "blockfp4"
                    else parse_dtype(args.main_weight_dtype)
                ),
                bias_dtype=torch.get_default_dtype(),
                gather_output=False,
                base_linear_class=get_linear_layout_contig_x_contig_y(
                    op_impl,
                    checkpoint_prefix=f"{checkpoint_prefix}.kv_b_proj",
                ),
                checkpoint_prefix=f"{checkpoint_prefix}.kv_b_proj",
            )
        elif self.mla_absorb == "absorb-without-precomp":
            quant_method = (
                None if not hasattr(args, "quant_config") else args.quant_config.type
            )
            self.kv_b_proj_absorb_1 = ParallelAbsorbGemm(
                self.n_heads,
                self.qk_nope_head_dim,
                self.kv_lora_rank,
                dtype=(
                    torch.bfloat16
                    if quant_method == "blockfp4"
                    else parse_dtype(args.main_weight_dtype)
                ),
                block_size=block_size,
            )
            self.kv_b_proj_absorb_2 = ParallelAbsorbGemm(
                self.n_heads,
                self.kv_lora_rank,
                self.v_head_dim,
                dtype=(
                    torch.bfloat16
                    if quant_method == "blockfp4"
                    else parse_dtype(args.main_weight_dtype)
                ),
                block_size=block_size,
            )

        self.o_proj = RowParallelLinear(
            (
                self.n_heads * self.v_head_dim
                if self.mla_absorb != "absorb"
                else self.n_heads * self.kv_lora_rank
            ),
            self.dim,
            has_bias=False,
            dtype=(
                torch.bfloat16
                if args.quant_config.type == "blockfp4"
                else parse_dtype(args.main_weight_dtype)
            ),
            bias_dtype=torch.get_default_dtype(),
            input_is_parallel=True,
            base_linear_class=get_linear_layout_contig_x_contig_y(
                op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.o_proj",
            ),
            checkpoint_prefix=f"{checkpoint_prefix}.o_proj",
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
            q_a = self.q_a_proj(x)
            kv = self.kv_a_proj_with_mqa(x)
        q = self.q_b_proj(self.q_a_layernorm(q_a, compute_dtype=q_a.dtype))

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
            return q, k, v
        elif self.mla_absorb == "absorb-without-precomp":
            q_nope = self.kv_b_proj_absorb_1(q_nope)
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
            self.kv_a_layernorm(kv_cache, compute_dtype=kv.dtype, out=kv_cache)

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
                x = self.kv_b_proj_absorb_2(x)

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
            self.kv_a_layernorm(this_kv, compute_dtype=kv.dtype, out=this_kv)

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
                x = self.kv_b_proj_absorb_2(x)

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
        self.kv_a_layernorm(this_kv, compute_dtype=kv.dtype, out=this_kv)

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
            x = self.kv_b_proj_absorb_2(x)
        x = self._run_output_linear(x)
        return x

    def _run_output_linear(self, x):
        return self.o_proj(x.flatten(-2))


class MLPDeepSeekV3(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        gate_proj (nn.Module): Linear layer for input-to-hidden transformation.
        down_proj (nn.Module): Linear layer for hidden-to-output transformation.
        up_proj (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(
        self,
        args,
        role: str,  # "standalone" or "shared_experts"
        op_impl: str,
        checkpoint_prefix: str,
        merge_gate_up=None,  # only work when role is "shared_experts"
    ):
        super().__init__()
        if role == "shared_experts":
            assert merge_gate_up is not None
            self.merge_gate_up = merge_gate_up
        else:
            quant = None
            for rule in args.quant_config.rules:
                pattern = rule.get("regex")
                if pattern and re.search(pattern, checkpoint_prefix):
                    quant = rule.type
                    break
            if quant in QuantizationRegistry._allowed_quant_for_merge_qkv_gate_up:
                self.merge_gate_up = True
            else:
                self.merge_gate_up = False
        self.op_impl = op_impl

        if role == "standalone":
            inter_dim = args.inter_dim
        elif role == "shared_experts":
            inter_dim = args.moe_inter_dim
        else:
            raise ValueError(
                f"Invalid role: {role}. Expected 'standalone' or 'shared_experts'."
            )

        if self.merge_gate_up:
            self.gate_up_proj = ColumnParallelLinear(
                args.dim,
                inter_dim * 2,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.get_default_dtype(),
                gather_output=False,
                base_linear_class=get_linear_layout_contig_x_contig_y(
                    op_impl,
                    quant_kwargs={
                        "blockfp4": {
                            "block_shape_2": (args.dim, inter_dim // get_tp_size())
                        }
                    },
                    checkpoint_prefix=f"{checkpoint_prefix}.gate_up_proj",
                ),
                checkpoint_prefix=f"{checkpoint_prefix}.gate_up_proj",
            )
        else:
            self.gate_proj = ColumnParallelLinear(
                args.dim,
                inter_dim,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.get_default_dtype(),
                gather_output=False,
                base_linear_class=get_linear_layout_contig_x_contig_y(
                    op_impl,
                    checkpoint_prefix=f"{checkpoint_prefix}.gate_proj",
                ),
                checkpoint_prefix=f"{checkpoint_prefix}.gate_proj",
            )
            self.up_proj = ColumnParallelLinear(
                args.dim,
                inter_dim,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.get_default_dtype(),
                gather_output=False,
                base_linear_class=get_linear_layout_contig_x_contig_y(
                    op_impl,
                    checkpoint_prefix=f"{checkpoint_prefix}.up_proj",
                ),
                checkpoint_prefix=f"{checkpoint_prefix}.up_proj",
            )
        self.down_proj = RowParallelLinear(
            inter_dim,
            args.dim,
            has_bias=False,
            dtype=parse_dtype(args.main_weight_dtype),
            bias_dtype=torch.get_default_dtype(),
            input_is_parallel=True,
            reduce_output=(role == "standalone"),
            base_linear_class=get_linear_layout_contig_x_contig_y(
                op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.down_proj",
            ),
            checkpoint_prefix=f"{checkpoint_prefix}.down_proj",
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
            gate_up_proj_out = self.gate_up_proj(x)
            return self.down_proj(silu_and_mul(gate_up_proj_out))
        else:
            gate_proj_out = self.gate_proj(x)
            up_proj_out = self.up_proj(x)
            return self.down_proj(F.silu(gate_proj_out) * up_proj_out)


class GateDeepSeekV3(MoeGate):
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

    def __init__(self, args, op_impl: str = "torch"):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__(
            op_impl=op_impl,
            dim=args.dim,
            topk=args.n_activated_experts,
            n_groups=args.n_expert_groups,
            topk_groups=args.n_limited_groups,
            score_func=args.score_func,
            route_scale=args.route_scale,
            n_experts=args.n_routed_experts,
            bias=(
                nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))
                if args.dim == 7168
                else None
            ),
            norm_prob=False,
        )


class MoeDeepSeekV3_params_resolver:
    def __init__(
        self,
        args,
        merge_gate_up: bool,
        op_impl: str,
        checkpoint_prefix: str,
    ):
        # Non-fused shared experts
        if not get_global_args().infer.fuse_shared_experts:
            shared_experts = MLPDeepSeekV3(
                args,
                role="shared_experts",
                merge_gate_up=merge_gate_up,
                op_impl=op_impl,
                checkpoint_prefix=checkpoint_prefix,
            )
        else:
            shared_experts = None
        super().__init__(
            dim=args.dim,
            moe_inter_dim=args.moe_inter_dim,
            n_routed_experts=args.n_routed_experts,
            n_shared_experts=args.n_shared_experts,
            n_activated_experts=args.n_activated_experts,
            moe_world_size=1,
            moe_rank=0,
            do_gather_output=False,
            dtype=args.main_weight_dtype,
            op_impl=op_impl,
            gate=GateDeepSeekV3(args, op_impl),
            fuse_shared_experts=get_global_args().infer.fuse_shared_experts,
            shared_experts=shared_experts,
            checkpoint_prefix=checkpoint_prefix,
            merge_gate_up=merge_gate_up,
        )


def MoEDeepSeekV3(
    args,
    op_impl: str,
    checkpoint_prefix: str,
    base_moe_class: Optional[type] = None,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
):
    checkpoint_prefix = checkpoint_prefix + ".moe"
    if base_moe_class is None:
        base_moe_class = MoeBlockRegistry.get_quantized_MoeBlock_class_from_global_args(
            quant_kwargs=quant_kwargs,
            checkpoint_prefix=checkpoint_prefix,
        )

    class MoeBlockImpl(MoeDeepSeekV3_params_resolver, base_moe_class):
        # NOTE: In Python, super().__init__ calls the next base class in the full inheritance graph
        # of the final class, so we can append a class to the base class, to make it act like a
        # further base class of the original base class.
        # See https://docs.python.org/3/tutorial/classes.html#multiple-inheritance

        pass

    quant = None
    for rule in args.quant_config.rules:
        pattern = rule.get("regex")
        if pattern and re.search(pattern, checkpoint_prefix):
            quant = rule.type
            break
    merge_gate_up = quant in QuantizationRegistry._allowed_quant_for_merge_qkv_gate_up

    return MoeBlockImpl(
        args,
        merge_gate_up=merge_gate_up,
        op_impl=op_impl,
        checkpoint_prefix=checkpoint_prefix,
    )


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

    def __init__(
        self,
        args,
        cpu_infer,
        ggml_type,
        checkpoint_prefix: str,
        merge_qkv_gate_up: bool = False,
    ):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.merge_qkv_gate_up = merge_qkv_gate_up
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
        if self.merge_qkv_gate_up:
            self.gate_up_proj = ColumnParallelLinear(
                args.dim,
                args.moe_inter_dim * 2,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.bfloat16,
                gather_output=False,
                checkpoint_prefix=f"{checkpoint_prefix}.gate_up_proj",
            )
        else:
            self.gate_proj = ColumnParallelLinear(
                args.dim,
                args.moe_inter_dim,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.bfloat16,
                gather_output=False,
                checkpoint_prefix=f"{checkpoint_prefix}.gate_proj",
            )
            self.up_proj = ColumnParallelLinear(
                args.dim,
                args.moe_inter_dim,
                has_bias=False,
                dtype=parse_dtype(args.main_weight_dtype),
                bias_dtype=torch.bfloat16,
                gather_output=False,
                checkpoint_prefix=f"{checkpoint_prefix}.up_proj",
            )
        self.down_proj = RowParallelLinear(
            args.moe_inter_dim,
            args.dim,
            has_bias=False,
            dtype=parse_dtype(args.main_weight_dtype),
            bias_dtype=torch.bfloat16,
            input_is_parallel=True,
            checkpoint_prefix=f"{checkpoint_prefix}.down_proj",
        )

        if self.rank == 0:

            self.register_buffer(
                "gguf_gate_proj",
                torch.empty(
                    int(256 * 2048 * 7168 / 256 * 144),
                    dtype=torch.uint8,
                    device="cpu",
                    requires_grad=False,
                ),
            )
            self.register_buffer(
                "gguf_up_proj",
                torch.empty(
                    int(256 * 2048 * 7168 / 256 * 144),
                    dtype=torch.uint8,
                    device="cpu",
                    requires_grad=False,
                ),
            )
            if ggml_type == 12:
                self.register_buffer(
                    "gguf_down_proj",
                    torch.empty(
                        int(256 * 2048 * 7168 / 256 * 144),
                        dtype=torch.uint8,
                        device="cpu",
                        requires_grad=False,
                    ),
                )
            elif ggml_type == 14:
                self.register_buffer(
                    "gguf_down_proj",
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
        if self.merge_qkv_gate_up:
            self.gate_up_proj.to(*args, **kwargs)
        else:
            self.gate_proj.to(*args, **kwargs)
            self.up_proj.to(*args, **kwargs)
        self.down_proj.to(*args, **kwargs)
        return self

    def init_weights(self):
        if self.rank == 0:
            gate_ptr = ctypes.addressof(
                ctypes.cast(
                    self.gguf_gate_proj.data_ptr(), ctypes.POINTER(ctypes.c_uint64)
                ).contents
            )
            up_ptr = ctypes.addressof(
                ctypes.cast(
                    self.gguf_up_proj.data_ptr(), ctypes.POINTER(ctypes.c_uint64)
                ).contents
            )
            down_ptr = ctypes.addressof(
                ctypes.cast(
                    self.gguf_down_proj.data_ptr(), ctypes.POINTER(ctypes.c_uint64)
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

        if self.merge_qkv_gate_up:
            gate_up_proj_out = self.gate_up_proj(x)
            gate_proj_out, up_proj_out = torch.split(
                gate_up_proj_out, gate_up_proj_out.shape[-1] // 2, dim=-1
            )
        else:
            gate_proj_out = self.gate_proj(x)
            up_proj_out = self.up_proj(x)
        y = self.down_proj(F.silu(gate_proj_out) * up_proj_out)

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
        cpu_infer=False,
        ggml_type=0,
        checkpoint_prefix="",
    ):
        super().__init__(
            layer_id, args, cache, attn_backend=attn_backend, op_impl=op_impl
        )
        self.layer_id = layer_id
        self.self_attn = AttentionDeepSeekV3(
            args,
            layer_id,
            cache,
            attn_backend,
            op_impl=op_impl,
            mla_absorb=mla_absorb,
            checkpoint_prefix=f"{checkpoint_prefix}.self_attn",
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
                MoEDeepSeekV3(
                    args,
                    op_impl=op_impl,
                    checkpoint_prefix=f"{checkpoint_prefix}.mlp",
                )
                if not cpu_infer
                else MoEDeepSeekV3CPU(
                    args,
                    cpu_infer=cpu_infer,
                    ggml_type=ggml_type,
                    checkpoint_prefix=f"{checkpoint_prefix}.mlp",
                )
            )
        )
        self.input_layernorm = RMSNorm(args.dim)
        self.post_attention_layernorm = RMSNorm(args.dim)

    def to(self, *args, **kwargs):
        self.self_attn.to(*args, **kwargs)
        self.mlp.to(*args, **kwargs)
        self.input_layernorm.to(*args, **kwargs)
        self.post_attention_layernorm.to(*args, **kwargs)
        return self

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis_cos: torch.Tensor,
        freqs_cis_sin: torch.Tensor,
        varlens=None,
    ):
        x = x + self.self_attn(
            self.input_layernorm(x, compute_dtype=x.dtype),
            freqs_cis_cos,
            freqs_cis_sin,
            varlens,
        )
        x = x + self.mlp(self.post_attention_layernorm(x, compute_dtype=x.dtype))
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
    ):
        self.mla_absorb = mla_absorb
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
        )

    def to(self, *args, **kwargs):
        if hasattr(self, "embed_tokens"):
            self.embed_tokens.to(*args, **kwargs)
        if hasattr(self, "norm"):
            self.norm.to(*args, **kwargs)
        if hasattr(self, "lm_head"):
            self.lm_head.to(*args, **kwargs)
        for l in self.layers:
            l.to(*args, **kwargs)
        return self

    @override
    def _get_tensor_column_parallel_layer_names(self) -> List[str]:
        return [
            "embed_tokens",
            "q_b_proj",
            "kv_b_proj",
            "gate_proj",
            "up_proj",
            "gate_up_proj",
            "lm_head",
        ]

    @override
    def _get_tensor_row_parallel_layer_names(self) -> List[str]:
        return ["o_proj", "down_proj"]

    @override
    def _get_pre_layer_prefixes(self) -> List[str]:
        return ["embed_tokens."]

    @override
    def _get_post_layer_prefixes(self) -> List[str]:
        return ["lm_head.", "norm."]

    @override
    def _get_layer_i_prefixes(self, i: int) -> List[str]:
        return [f"layers.{i}."]

    def _process_state_dict_for_merging_experts(self, checkpoint: Mapping[str, Any]):
        fuse_shared_experts = get_global_args().infer.fuse_shared_experts

        new_checkpoint = {}
        for k in checkpoint.keys():
            quant = None
            for rule in self.params.quant_config.rules:
                pattern = rule.get("regex")
                if pattern and re.search(pattern, k):
                    quant = rule.type
                    break
            if any(
                k.endswith(f".experts.0.{w}.{part}")
                for w in ["gate_proj", "down_proj", "up_proj", "gate_up_proj"]
                for part in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_2d_in_x_out_tensor_names(quant)
                + self._get_1d_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                w, part = k.split(".")[-2:]
                prefix = k[: -len(f"experts.0.{w}.{part}")]
                parts = []
                for i in range(self.params.n_routed_experts):
                    parts.append(checkpoint[prefix + f"experts.{i}.{w}.{part}"])
                if fuse_shared_experts:
                    parts.append(checkpoint[prefix + f"shared_experts.{w}.{part}"])
                new_checkpoint[prefix + f"{w}.{part}"] = torch.stack(parts, dim=0)
            elif ".experts." in k:
                continue
            elif fuse_shared_experts and ".shared_experts." in k:
                continue
            else:
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
            quant = None
            for rule in self.params.quant_config.rules:
                pattern = rule.get("regex")
                if pattern and re.search(pattern, k):
                    quant = rule.type
                    break
            if any(
                k.endswith(f".kv_b_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names(quant)
            ):
                tensor_name = k.split(".")[-1]
                prefix = k[: -len(f".kv_b_proj.{tensor_name}")]
                kv_b_proj_weight = checkpoint[f"{prefix}.kv_b_proj.{tensor_name}"]
                kv_b_proj_weight = kv_b_proj_weight.view(
                    n_local_heads, -1, kv_b_proj_weight.shape[-1]
                )
                absorbed_dim = self.params.qk_nope_head_dim + self.params.v_head_dim
                assert absorbed_dim % kv_b_proj_weight.shape[1] == 0
                ratio = absorbed_dim // kv_b_proj_weight.shape[1]
                kv_b_proj_absorb_1_weight = kv_b_proj_weight[
                    :, : self.params.qk_nope_head_dim // ratio
                ]
                kv_b_proj_absorb_2_weight = kv_b_proj_weight[
                    :, self.params.qk_nope_head_dim // ratio :
                ]
                new_checkpoint[f"{prefix}.kv_b_proj_absorb_1.{tensor_name}"] = (
                    kv_b_proj_absorb_1_weight.permute(0, 2, 1).contiguous()
                )
                new_checkpoint[f"{prefix}.kv_b_proj_absorb_2.{tensor_name}"] = (
                    kv_b_proj_absorb_2_weight
                )

            elif any(
                k.endswith(f".kv_b_proj.{tensor_name}")
                for tensor_name in self._get_2d_in_x_out_tensor_names(quant)
            ):
                raise NotImplementedError(
                    f"infer.mla_absorb=absorb-without-precomp is not implemented for 2D (in, out) tensor {tensor_name}"
                )

            elif any(
                k.endswith(f".kv_b_proj.{tensor_name}")
                for tensor_name in self._get_1d_in_tensor_names(quant)
            ):
                raise NotImplementedError(
                    f"infer.mla_absorb=absorb-without-precomp is not implemented for 1D (in,) tensor {tensor_name}"
                )

            elif any(
                k.endswith(f".kv_b_proj.{tensor_name}")
                for tensor_name in self._get_1d_out_tensor_names(quant)
            ):
                raise NotImplementedError(
                    f"infer.mla_absorb=absorb-without-precomp is not implemented for 1D (out,) tensor {tensor_name}"
                )

            else:
                new_checkpoint[k] = checkpoint[k]

        return new_checkpoint

    def _process_state_dict_for_absorption(self, checkpoint: Mapping[str, Any]):
        model_parallel_size = get_tp_size()
        n_local_heads = self.params.n_heads // model_parallel_size

        quant = (
            self.params.quant_config.type
            if hasattr(self.params, "quant_config")
            else None
        )

        weight_dequant_fn = (
            weight_dequant_soft_fp8_deepseek_v3
            if get_global_args().infer.raise_lower_bit_float_to == "bfloat16"
            else weight_dequant_deepseek_v3
        )
        block_size = 128

        new_checkpoint = {}
        for k in checkpoint.keys():
            if k.endswith(".kv_b_proj.weight"):
                prefix = k[: -len("kv_b_proj.weight")]
                assert prefix + "kv_b_proj.weight" in checkpoint
                kv_b_proj_ckpt_weight = checkpoint[prefix + "kv_b_proj.weight"]
                if quant in [None, "gguf", "blockfp4"]:  # blockfp4 skips quantizing MLA
                    kv_b_proj_weight = kv_b_proj_ckpt_weight
                elif quant in ["blockfp8", "gguf-blockfp8"]:
                    assert prefix + "kv_b_proj.scale" in checkpoint
                    kv_b_proj_scale = checkpoint[prefix + "kv_b_proj.scale"]
                    # FIXME: Keep this on GPU
                    kv_b_proj_weight = weight_dequant_fn(
                        kv_b_proj_ckpt_weight.cuda(), kv_b_proj_scale.cuda(), block_size
                    ).cpu()
                else:
                    raise NotImplementedError(
                        f"infer.mla_absorb=absorb is not implemented for {quant} quantization"
                    )
                kv_b_proj_weight = kv_b_proj_weight.view(
                    n_local_heads,
                    self.params.qk_nope_head_dim + self.params.v_head_dim,
                    self.params.kv_lora_rank,
                )

                # Absorb into q_b_proj
                q_b_proj_ckpt_weight = checkpoint[prefix + "q_b_proj.weight"]
                if quant in [None, "gguf", "blockfp4"]:  # blockfp4 skips quantizing MLA
                    q_b_proj_weight = q_b_proj_ckpt_weight
                elif quant in ["blockfp8", "gguf-blockfp8"]:
                    assert prefix + "q_b_proj.scale" in checkpoint
                    q_b_proj_scale = checkpoint[prefix + "q_b_proj.scale"]
                    # FIXME: Keep this on GPU
                    q_b_proj_weight = weight_dequant_fn(
                        q_b_proj_ckpt_weight.cuda(), q_b_proj_scale.cuda(), block_size
                    ).cpu()
                else:
                    raise NotImplementedError(
                        f"infer.mla_absorb=absorb is not implemented for {quant} quantization"
                    )
                q_b_proj_weight_per_head = q_b_proj_weight.view(
                    n_local_heads,
                    self.params.qk_nope_head_dim + self.params.qk_rope_head_dim,
                    self.params.q_lora_rank,
                )
                q_b_proj_nope = q_b_proj_weight_per_head[
                    :, : self.params.qk_nope_head_dim
                ]
                q_b_proj_rope = q_b_proj_weight_per_head[
                    :, self.params.qk_nope_head_dim :
                ]
                #   x @ q_b_proj_nope^T @ per_head(kv_b_proj[:, :qk_nope_head_dim, :])
                # = x @ (per_head(kv_b_proj[:, :qk_nope_head_dim, :])^T @ q_b_proj_nope)^T
                kv_b_proj_for_q_b_proj = kv_b_proj_weight[
                    :, : self.params.qk_nope_head_dim
                ]
                assert kv_b_proj_for_q_b_proj.shape == (
                    n_local_heads,
                    self.params.qk_nope_head_dim,
                    self.params.kv_lora_rank,
                )
                kv_b_proj_for_q_b_proj = torch.block_diag(*kv_b_proj_for_q_b_proj)
                new_q_b_proj_nope = (
                    kv_b_proj_for_q_b_proj.t()
                    @ q_b_proj_nope.contiguous().view(-1, self.params.q_lora_rank)
                ).view(n_local_heads, self.params.kv_lora_rank, self.params.q_lora_rank)
                new_q_b_proj = torch.cat(
                    [new_q_b_proj_nope, q_b_proj_rope], dim=1
                ).view(-1, self.params.q_lora_rank)
                if quant in [None, "gguf", "blockfp4"]:  # blockfp4 skips quantizing MLA
                    new_checkpoint[prefix + "q_b_proj.weight"] = new_q_b_proj
                elif quant in ["blockfp8", "gguf-blockfp8"]:
                    # FIXME: Support soft fp8 in weight_quant_deepseek_v3
                    new_q_b_proj, new_q_b_proj_scale = weight_quant_deepseek_v3(
                        new_q_b_proj, block_size
                    )
                    if (
                        parse_dtype(
                            get_global_args().infer.raise_lower_bit_float_to
                        ).itemsize
                        > 1
                    ):
                        new_q_b_proj = new_q_b_proj.view(dtype=torch.uint8)
                    new_checkpoint[prefix + "q_b_proj.weight"] = new_q_b_proj
                    new_checkpoint[prefix + "q_b_proj.scale"] = new_q_b_proj_scale
                else:
                    raise NotImplementedError(
                        f"infer.mla_absorb=absorb is not implemented for {quant} quantization"
                    )

                # Absorb into o_proj
                o_proj_ckpt_weight = checkpoint[prefix + "o_proj.weight"]
                if quant in [None, "gguf", "blockfp4"]:  # blockfp4 skips quantizing MLA
                    o_proj_weight = o_proj_ckpt_weight
                elif quant in ["blockfp8", "gguf-blockfp8"]:
                    assert prefix + "o_proj.scale" in checkpoint
                    o_proj_scale = checkpoint[prefix + "o_proj.scale"]
                    # FIXME: Keep this on GPU
                    o_proj_weight = weight_dequant_fn(
                        o_proj_ckpt_weight.cuda(), o_proj_scale.cuda(), block_size
                    ).cpu()
                else:
                    raise NotImplementedError(
                        f"infer.mla_absorb=absorb is not implemented for {quant} quantization"
                    )
                #   x @ per_head(kv_b_proj_weight[:, -params.v_head_dim :, :]^T) @ o_proj_weight^T
                # = x @ (o_proj_weight @ per_head(kv_b_proj_weight[:, -params.v_head_dim :, :]))^T
                kv_b_proj_for_o_proj = kv_b_proj_weight[:, -self.params.v_head_dim :]
                assert kv_b_proj_for_o_proj.shape == (
                    n_local_heads,
                    self.params.v_head_dim,
                    self.params.kv_lora_rank,
                )
                kv_b_proj_for_o_proj = torch.block_diag(*kv_b_proj_for_o_proj)
                new_o_proj = o_proj_weight @ kv_b_proj_for_o_proj
                if quant in [None, "gguf", "blockfp4"]:  # blockfp4 skips quantizing MLA
                    new_checkpoint[prefix + "o_proj.weight"] = new_o_proj
                elif quant in ["blockfp8", "gguf-blockfp8"]:
                    # FIXME: Support soft fp8 in weight_quant_deepseek_v3
                    new_o_proj, new_o_proj_scale = weight_quant_deepseek_v3(
                        new_o_proj, block_size
                    )
                    if (
                        parse_dtype(
                            get_global_args().infer.raise_lower_bit_float_to
                        ).itemsize
                        > 1
                    ):
                        new_o_proj = new_o_proj.view(dtype=torch.uint8)
                    new_checkpoint[prefix + "o_proj.weight"] = new_o_proj
                    new_checkpoint[prefix + "o_proj.scale"] = new_o_proj_scale
                else:
                    raise NotImplementedError(
                        f"infer.mla_absorb=absorb is not implemented for {quant} quantization"
                    )

            elif k.endswith(".kv_b_proj.scale"):
                continue

            elif k.endswith(".kv_b_proj.bias"):
                raise NotImplementedError(
                    "infer.mla_absorb=absorb is not implemented for kv_b_proj with a bias"
                )

            elif k.endswith(".o_proj.weight") or k.endswith(".o_proj.scale"):
                continue

            elif k.endswith(".q_b_proj.weight") or k.endswith(".q_b_proj.scale"):
                continue

            elif k.endswith(".q_b_proj.bias"):
                raise NotImplementedError(
                    "infer.mla_absorb=absorb is not implemented for q_b_proj with a bias"
                )

            else:
                new_checkpoint[k] = checkpoint[k]

        return new_checkpoint

    def _process_state_dict_for_merging_qkv(self, checkpoint: Mapping[str, Any]):
        new_checkpoint = {}
        for k in checkpoint.keys():
            quant = None
            for rule in self.params.quant_config.rules:
                pattern = rule.get("regex")
                if pattern and re.search(pattern, k):
                    quant = rule.type
                    break
            if quant not in QuantizationRegistry._allowed_quant_for_merge_qkv_gate_up:
                new_checkpoint[k] = checkpoint[k]
            # Cat dim 0
            elif any(
                k.endswith(f".q_a_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                tensor_name = k.split(".")[-1]
                prefix = k[: -len(f".q_a_proj.{tensor_name}")]
                assert f"{prefix}.kv_a_proj_with_mqa.{tensor_name}" in checkpoint
                q_weight = checkpoint[f"{prefix}.q_a_proj.{tensor_name}"]
                kv_weight = checkpoint[f"{prefix}.kv_a_proj_with_mqa.{tensor_name}"]
                new_checkpoint[f"{prefix}.wqkv_a.{tensor_name}"] = torch.cat(
                    [q_weight, kv_weight], dim=0
                )
            elif any(
                k.endswith(f".kv_a_proj_with_mqa.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                continue

            # Cat dim 1
            elif any(
                k.endswith(f".q_a_proj.{tensor_name}")
                for tensor_name in self._get_2d_in_x_out_tensor_names(quant)
            ):
                tensor_name = k.split(".")[-1]
                prefix = k[: -len(f".q_a_proj.{tensor_name}")]
                assert f"{prefix}.kv_a_proj_with_mqa.{tensor_name}" in checkpoint
                q_weight = checkpoint[f"{prefix}.q_a_proj.{tensor_name}"]
                kv_weight = checkpoint[f"{prefix}.kv_a_proj_with_mqa.{tensor_name}"]
                new_checkpoint[f"{prefix}.wqkv_a.{tensor_name}"] = torch.cat(
                    [q_weight, kv_weight], dim=1
                )
            elif any(
                k.endswith(f".kv_a_proj_with_mqa.{tensor_name}")
                for tensor_name in self._get_2d_in_x_out_tensor_names(quant)
            ):
                continue

            # Unchanged tensors
            else:
                new_checkpoint[k] = checkpoint[k]
        return new_checkpoint

    def _process_state_dict_for_merging_gate_up(self, checkpoint: Mapping[str, Any]):
        new_checkpoint = {}
        for k in checkpoint.keys():
            quant = None
            for rule in self.params.quant_config.rules:
                pattern = rule.get("regex")
                if pattern and re.search(pattern, k):
                    quant = rule.type
                    break
            if quant not in QuantizationRegistry._allowed_quant_for_merge_qkv_gate_up:
                new_checkpoint[k] = checkpoint[k]
            # Cat dim 0
            elif any(
                k.endswith(f".gate_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                tensor_name = k.split(".")[-1]
                prefix = k[: -len(f".gate_proj.{tensor_name}")]
                assert f"{prefix}.up_proj.{tensor_name}" in checkpoint
                assert f"{prefix}.gate_up_proj.{tensor_name}" not in checkpoint
                gate_weight = checkpoint[f"{prefix}.gate_proj.{tensor_name}"]
                up_weight = checkpoint[f"{prefix}.up_proj.{tensor_name}"]
                new_checkpoint[f"{prefix}.gate_up_proj.{tensor_name}"] = torch.cat(
                    [gate_weight, up_weight], dim=0
                )
            elif any(
                k.endswith(f".up_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                continue

            # Cat dim 1
            elif any(
                k.endswith(f".gate_proj.{tensor_name}")
                for tensor_name in self._get_2d_in_x_out_tensor_names(quant)
            ):
                tensor_name = k.split(".")[-1]
                prefix = k[: -len(f".gate_proj.{tensor_name}")]
                assert f"{prefix}.up_proj.{tensor_name}" in checkpoint
                assert f"{prefix}.gate_up_proj.{tensor_name}" not in checkpoint
                gate_weight = checkpoint[f"{prefix}.gate_proj.{tensor_name}"]
                up_weight = checkpoint[f"{prefix}.up_proj.{tensor_name}"]
                new_checkpoint[f"{prefix}.gate_up_proj.{tensor_name}"] = torch.cat(
                    [gate_weight, up_weight], dim=1
                )
            elif any(
                k.endswith(f".up_proj.{tensor_name}")
                for tensor_name in self._get_2d_in_x_out_tensor_names(quant)
            ):
                continue

            # Unchanged tensors
            else:
                new_checkpoint[k] = checkpoint[k]
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
        if not skip_preprocess and replace:
            new_state_dict = {}
            for k in state_dict.keys():
                name = k
                name = name.replace(".weight_scale_inv", ".scale")
                name = name.replace(".e_score_correction_bias", ".bias")
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
        if not skip_preprocess:
            if self.mla_absorb == "absorb":
                state_dict = self._process_state_dict_for_absorption(state_dict)
            elif self.mla_absorb == "absorb-without-precomp":
                state_dict = (
                    self._process_state_dict_for_absorption_without_precomputation(
                        state_dict
                    )
                )

            state_dict = self._process_state_dict_for_merging_qkv(state_dict)
            state_dict = self._process_state_dict_for_merging_gate_up(state_dict)

            state_dict = self._process_state_dict_for_merging_experts(state_dict)
            if not self.cpu_infer:
                state_dict = super().process_state_dict_for_renaming_linear_layer(
                    state_dict,
                    get_global_args().models.n_dense_layers,
                )

        if self.op_impl == "muxi_custom_kernel":
            rpl_names = self._get_tensor_row_parallel_layer_names()
            cpl_names = self._get_tensor_column_parallel_layer_names()
            cpl_names = [
                name for name in cpl_names if name not in {"embed_tokens", "lm_head"}
            ]
            if self.mla_absorb == "absorb-without-precomp":
                cpl_names.remove("kv_b_proj")
            state_dict = preprocess_weights_for_native_layout(
                state_dict, rpl_names, cpl_names
            )

        super().load_state_dict(
            state_dict, skip_preprocess=skip_preprocess, *args, **kwargs
        )

    @override
    def _init_pre_layers(self):
        self.embed_tokens = VocabParallelEmbedding(
            self.params.vocab_size, self.params.dim
        )

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
                        cpu_infer=self.cpu_infer,
                        ggml_type=self.ggml_type[layer_id],
                        checkpoint_prefix=f"layers.{layer_id}",
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
                        checkpoint_prefix=f"layers.{layer_id}",
                    )
                )

    @override
    def _init_post_layers(self):
        self.norm = RMSNorm(self.params.dim)
        self.lm_head = ColumnParallelLinear(
            self.params.dim,
            self.params.vocab_size,
            has_bias=False,
            dtype=torch.get_default_dtype(),
            gather_output=True,
            checkpoint_prefix="lm_head",
        )

    @override
    def _pre_layers(self, h):
        return self.embed_tokens(h)

    @override
    def _post_layers(self, h):
        """NOTE: _post_layers is assumed to be a token-wise computation"""
        h = self.norm(h, compute_dtype=h.dtype)
        h = self.lm_head(h)
        return h

    @override
    def precompute_freqs_cis(self, max_position_embeddings: int, device):
        self.freqs_cis = precompute_freqs_cis_deepseek_v3(
            self.params, max_position_embeddings
        )
        self.freqs_cis_real = self.freqs_cis.real.contiguous().to(device)
        self.freqs_cis_imag = self.freqs_cis.imag.contiguous().to(device)

    @override
    def prepare_freqs_cis_prefill(self, varlens):
        index = self.cache.curr_varlens.position_ids
        return self.freqs_cis_real[index], self.freqs_cis_imag[index]

    @override
    def prepare_freqs_cis_decode(self):
        index = self.cache.get_gpu_seq_lens_excl_this_decode()
        return self.freqs_cis_real[index], self.freqs_cis_imag[index]

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


def get_linear_layout_contig_x_contig_y(
    op_impl: str,
    checkpoint_prefix: str,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
):
    if op_impl == "muxi_custom_kernel":
        assert (
            len(quant_kwargs) == 0
        ), "quant_kwargs is not supported for muxi_custom_kernel"
        args = get_global_args()
        quant_method = (
            None
            if not hasattr(args.models, "quant_config")
            else args.models.quant_config.type
        )
        if quant_method is None:
            return LinearLayoutContigXContigY
        elif quant_method == "blockfp8":
            return Blockfp8LinearLayoutContigXContigY
        else:
            raise NotImplementedError(
                f'Quantization method {quant_method} is not implemented for "muxi_custom_kernel"'
            )

    else:
        return QuantizationRegistry.get_quantized_linear_class_from_global_args(
            quant_kwargs=quant_kwargs, checkpoint_prefix=checkpoint_prefix
        )
