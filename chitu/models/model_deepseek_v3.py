import math
import re
from logging import getLogger
from typing import Any, List, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import override

from chitu.attn_backend import AttnBackend
from chitu.global_vars import get_global_args
from chitu.models.model import (
    Attention,
    MoeGate,
    ParallelMoeBlock,
    RMSNorm,
    Transformer,
    TransformerBlock,
)
from chitu.models.registry import ModelType, register_model
from chitu.muxi_utils import (
    Blockfp8LinearMuxiLayoutContigY,
    LinearMuxiLayoutContigY,
    NormalMoeExpertsMuxiLayout,
    Blockfp8MoeExpertsMuxiLayout,
)
from chitu.ops import (
    apply_rotary_pos_emb,
    silu_and_mul,
    weight_dequant_deepseek_v3,
    weight_dequant_soft_fp8_deepseek_v3,
    weight_quant_deepseek_v3,
    unpack_weight_bytes,
    decode_e2m1_from_nibbles,
    fp4_fake_quant,
    pack_weight_nibbles,
    to_e2m1_nibbles,
)
from chitu.quantization import QuantizationRegistry, get_quant_from_checkpoint_prefix
from chitu.tensor_parallel import (
    ColumnParallelLinear,
    LocalLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from chitu.distributed.parallel_state import get_tp_size
from chitu.utils import parse_dtype, try_import_opt_dep

triton, has_triton = try_import_opt_dep("triton", "triton")
chitu_backend, has_chitu_backend = try_import_opt_dep("chitu_backend", "chitu_backend")
torch_npu, has_torch_npu = try_import_opt_dep("torch_npu", "torch_npu")


logger = getLogger(__name__)


def ParallelAbsorbGemm(
    global_n_heads: int,
    in_features_per_head: int,
    out_features_per_head: int,
    *,
    checkpoint_prefix: str,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
):
    """
    Factory function for the two group GeMMs in "absorb-without-precomp" mode, embaarrassingly parallel among heads.

    It computes `einsum("shc,hdc->shd", x, weight)`, maybe quantized.
    """

    base_class = QuantizationRegistry.get_quantized_absorb_gemm_class_from_global_args(
        quant_kwargs=quant_kwargs, checkpoint_prefix=checkpoint_prefix
    )

    tp_size = get_tp_size()
    assert global_n_heads % tp_size == 0
    local_n_heads = global_n_heads // tp_size

    return base_class(local_n_heads, in_features_per_head, out_features_per_head)


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
        quant = get_quant_from_checkpoint_prefix(
            checkpoint_prefix, args.quant_config.rules
        )
        self.merge_qkv = quant in QuantizationRegistry._allowed_quant_for_merge_qkv

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

        block_size = 16 if quant == "blockfp4" else 128

        if self.merge_qkv:
            # fp8 gemm can handle weights not divisible by block_size, but it does not hold
            # after merging for the output dimension, except for the last weight.
            assert self.q_lora_rank % block_size == 0
            self.wqkv_a = LocalLinear(
                self.dim,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                has_bias=False,
                checkpoint_prefix=f"{checkpoint_prefix}.wqkv_a",
            )  # FIXME: Run this layer with muxi_layout_kernels
        else:
            self.q_a_proj = LocalLinear(
                self.dim,
                self.q_lora_rank,
                has_bias=False,
                checkpoint_prefix=f"{checkpoint_prefix}.q_a_proj",
            )  # FIXME: Run this layer with muxi_layout_kernels
            self.kv_a_proj_with_mqa = LocalLinear(
                self.dim,
                self.kv_lora_rank + self.qk_rope_head_dim,
                has_bias=False,
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
            gather_output=False,
            base_linear_class=get_linear_layout_contig_y(
                op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.q_b_proj",
            ),
            checkpoint_prefix=f"{checkpoint_prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)

        if self.mla_absorb == "none":
            self.kv_b_proj = ColumnParallelLinear(
                self.kv_lora_rank,
                self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
                has_bias=False,
                gather_output=False,
                base_linear_class=get_linear_layout_contig_y(
                    op_impl,
                    checkpoint_prefix=f"{checkpoint_prefix}.kv_b_proj",
                ),
                checkpoint_prefix=f"{checkpoint_prefix}.kv_b_proj",
            )
        elif self.mla_absorb == "absorb-without-precomp":
            self.kv_b_proj_absorb_1 = ParallelAbsorbGemm(
                self.n_heads,
                self.qk_nope_head_dim,
                self.kv_lora_rank,
                quant_kwargs={"blockfp8": {"block_size": block_size}},
                checkpoint_prefix=f"{checkpoint_prefix}.kv_b_proj",
            )
            self.kv_b_proj_absorb_2 = ParallelAbsorbGemm(
                self.n_heads,
                self.kv_lora_rank,
                self.v_head_dim,
                quant_kwargs={"blockfp8": {"block_size": block_size}},
                checkpoint_prefix=f"{checkpoint_prefix}.kv_b_proj",
            )

        self.o_proj = RowParallelLinear(
            (
                self.n_heads * self.v_head_dim
                if self.mla_absorb != "absorb"
                else self.n_heads * self.kv_lora_rank
            ),
            self.dim,
            has_bias=False,
            input_is_parallel=True,
            base_linear_class=get_linear_layout_contig_y(
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
            ).view(bsz, seqlen, self.n_local_heads, self.v_head_dim)

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
        cache_seqlens_excl_this_decode = self.cache.get_gpu_seq_lens_excl_this_decode()
        cache_seqlens_incl_this_decode = self.cache.get_gpu_seq_lens_incl_this_decode()
        block_table = self.cache.get_gpu_block_table()

        bsz, seqlen, _ = x.size()

        if self.mla_absorb == "none":
            q, k, v = self._run_linear(
                x.view(bsz * seqlen, -1), freqs_cis_cos, freqs_cis_sin
            )
            q = q.view(bsz, seqlen, self.n_local_heads, -1)
            k = k.view(bsz, seqlen, self.n_local_heads, -1)
            v = v.view(bsz, seqlen, self.n_local_heads, -1)

            paged_k_cache, paged_v_cache = self.cache.get_paged_kv_cache(self.layer_id)
            x = self.attn_backend.attn_with_kvcache(
                q,
                paged_k_cache,
                paged_v_cache,
                k,
                v,
                cache_seqlens=cache_seqlens_excl_this_decode,
                block_table=block_table,
                softmax_scale=self.softmax_scale,
            ).view(bsz, seqlen, self.n_local_heads, self.v_head_dim)

        elif self.mla_absorb == "absorb-without-precomp" or self.mla_absorb == "absorb":
            q_nope, q_pe, kv = self._run_linear(
                x.view(bsz * seqlen, -1), freqs_cis_cos, freqs_cis_sin
            )

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

        else:
            raise NotImplementedError(
                f"MLA absorb mode {self.mla_absorb} not supported"
            )

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
            quant = get_quant_from_checkpoint_prefix(
                checkpoint_prefix, args.quant_config.rules
            )
            if quant in QuantizationRegistry._allowed_quant_for_merge_gate_up:
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
                gather_output=False,
                base_linear_class=get_linear_layout_contig_y(
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
                gather_output=False,
                base_linear_class=get_linear_layout_contig_y(
                    op_impl,
                    checkpoint_prefix=f"{checkpoint_prefix}.gate_proj",
                ),
                checkpoint_prefix=f"{checkpoint_prefix}.gate_proj",
            )
            self.up_proj = ColumnParallelLinear(
                args.dim,
                inter_dim,
                has_bias=False,
                gather_output=False,
                base_linear_class=get_linear_layout_contig_y(
                    op_impl,
                    checkpoint_prefix=f"{checkpoint_prefix}.up_proj",
                ),
                checkpoint_prefix=f"{checkpoint_prefix}.up_proj",
            )
        self.down_proj = RowParallelLinear(
            inter_dim,
            args.dim,
            has_bias=False,
            input_is_parallel=True,
            reduce_output=(role == "standalone"),
            base_linear_class=get_linear_layout_contig_y(
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


def MoeExpertsDeepSeekV3(
    args,
    op_impl: str,
    checkpoint_prefix: str,
    base_moe_experts_class: Optional[type] = None,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
):
    checkpoint_prefix = checkpoint_prefix + ".moe"
    if base_moe_experts_class is None:
        base_moe_experts_class = (
            QuantizationRegistry.get_quantized_moe_experts_class_from_global_args(
                quant_kwargs=quant_kwargs,
                checkpoint_prefix=checkpoint_prefix,
            )
        )

    quant = get_quant_from_checkpoint_prefix(checkpoint_prefix, args.quant_config.rules)
    merge_gate_up = quant in QuantizationRegistry._allowed_quant_for_merge_gate_up

    assert args.moe_inter_dim % get_tp_size() == 0
    return base_moe_experts_class(
        dim=args.dim,
        moe_inter_dim=args.moe_inter_dim // get_tp_size(),
        n_routed_experts=args.n_routed_experts,
        n_shared_experts=args.n_shared_experts,
        n_activated_experts=args.n_activated_experts,
        moe_world_size=1,
        moe_rank=0,
        op_impl=op_impl,
        fuse_shared_experts=get_global_args().infer.fuse_shared_experts,
        checkpoint_prefix=checkpoint_prefix,
        merge_gate_up=merge_gate_up,
    )


class ParallelMoeBlockDeepSeekV3(ParallelMoeBlock):
    def __init__(
        self,
        args,
        op_impl: str,
        checkpoint_prefix: str,
        base_moe_experts_class: Optional[type] = None,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
    ):
        if not get_global_args().infer.fuse_shared_experts:
            quant = get_quant_from_checkpoint_prefix(
                checkpoint_prefix, args.quant_config.rules
            )
            merge_gate_up = (
                quant in QuantizationRegistry._allowed_quant_for_merge_gate_up
            )
            non_fused_shared_experts = MLPDeepSeekV3(
                args,
                role="shared_experts",
                merge_gate_up=merge_gate_up,
                op_impl=op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.shared_experts",
            )
        else:
            non_fused_shared_experts = None

        super().__init__(
            gate=GateDeepSeekV3(args, op_impl=op_impl),
            experts=MoeExpertsDeepSeekV3(
                args,
                op_impl=op_impl,
                checkpoint_prefix=checkpoint_prefix,
                base_moe_experts_class=base_moe_experts_class,
                quant_kwargs=quant_kwargs,
            ),
            non_fused_shared_experts=non_fused_shared_experts,
        )


class TransformerBlockDeepSeekV3(TransformerBlock):
    def __init__(
        self,
        layer_id: int,
        args,
        cache,
        attn_backend,
        op_impl,
        mla_absorb,
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
                )
            )
        )
        self.input_layernorm = RMSNorm(args.dim)
        self.post_attention_layernorm = RMSNorm(args.dim)

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


@register_model(ModelType.DEEPSEEK_V3)
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
    ):
        self.mla_absorb = mla_absorb
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
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
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
                new_checkpoint[prefix + f"experts.{w}.{part}"] = torch.stack(
                    parts, dim=0
                )
            elif re.search(r"\.experts\.\d+", k):
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
        new_checkpoint = {}
        for k in checkpoint.keys():
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            if any(
                k.endswith(f".kv_b_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names(quant)
            ):
                tensor_name = k.split(".")[-1]
                prefix = k[: -len(f".kv_b_proj.{tensor_name}")]
                if k.endswith(f".kv_b_proj.input_scale") or k.endswith(
                    f".kv_b_proj.weight_scale_2"
                ):
                    new_checkpoint[f"{prefix}.kv_b_proj_absorb_1.{tensor_name}"] = (
                        checkpoint[k].view(1, 1)
                    )
                    new_checkpoint[f"{prefix}.kv_b_proj_absorb_2.{tensor_name}"] = (
                        checkpoint[k].view(1, 1)
                    )
                else:
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

        weight_dequant_fn = (
            weight_dequant_soft_fp8_deepseek_v3
            if get_global_args().infer.raise_lower_bit_float_to == "bfloat16"
            else weight_dequant_deepseek_v3
        )

        new_checkpoint = {}
        for k in checkpoint.keys():
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            block_size = 16 if quant in ["blockfp4"] else 128

            if k.endswith(".kv_b_proj.weight"):
                prefix = k[: -len("kv_b_proj.weight")]
                assert prefix + "kv_b_proj.weight" in checkpoint
                kv_b_proj_ckpt_weight = checkpoint[prefix + "kv_b_proj.weight"]
                if quant in [None, "gguf"]:  # blockfp4 skips quantizing MLA
                    kv_b_proj_weight = kv_b_proj_ckpt_weight
                elif quant in ["blockfp8", "q4km"]:
                    assert prefix + "kv_b_proj.scale" in checkpoint
                    kv_b_proj_scale = checkpoint[prefix + "kv_b_proj.scale"]
                    # FIXME: Keep this on GPU
                    kv_b_proj_weight = weight_dequant_fn(
                        kv_b_proj_ckpt_weight.cuda(), kv_b_proj_scale.cuda(), block_size
                    ).cpu()
                elif quant in ["blockfp4"]:
                    assert prefix + "kv_b_proj.weight_scale" in checkpoint
                    assert prefix + "kv_b_proj.weight_scale_2" in checkpoint
                    up_kv_b_proj_weight = decode_e2m1_from_nibbles(
                        unpack_weight_bytes(kv_b_proj_ckpt_weight.cuda())
                    ).reshape(*kv_b_proj_ckpt_weight.shape[:-1], -1, block_size)
                    kv_b_proj_weight = (
                        (
                            up_kv_b_proj_weight
                            * checkpoint[prefix + "kv_b_proj.weight_scale"]
                            .view(torch.float8_e4m3fn)
                            .unsqueeze(-1)
                            .to(
                                dtype=up_kv_b_proj_weight.dtype,
                                device=up_kv_b_proj_weight.device,
                            )
                            * checkpoint[prefix + "kv_b_proj.weight_scale_2"].to(
                                device=up_kv_b_proj_weight.device
                            )
                        )
                        .reshape(
                            kv_b_proj_ckpt_weight.shape[0],
                            kv_b_proj_ckpt_weight.shape[1] * 2,
                        )
                        .to(dtype=torch.bfloat16, device="cpu")
                    )
                else:
                    raise NotImplementedError(
                        f"infer.mla_absorb=absorb is not implemented for {quant} quantization"
                    )
                # kv_lora_rank = self.params.kv_lora_rank // 2 if quant == "blockfp4" else self.params.kv_lora_rank
                kv_lora_rank = self.params.kv_lora_rank

                kv_b_proj_weight = kv_b_proj_weight.view(
                    n_local_heads,
                    self.params.qk_nope_head_dim + self.params.v_head_dim,
                    kv_lora_rank,
                )

                # Absorb into q_b_proj
                q_b_proj_ckpt_weight = checkpoint[prefix + "q_b_proj.weight"]
                if quant in [None, "gguf"]:  # blockfp4 skips quantizing MLA
                    q_b_proj_weight = q_b_proj_ckpt_weight
                elif quant in ["blockfp8", "q4km"]:
                    assert prefix + "q_b_proj.scale" in checkpoint
                    q_b_proj_scale = checkpoint[prefix + "q_b_proj.scale"]
                    # FIXME: Keep this on GPU
                    q_b_proj_weight = weight_dequant_fn(
                        q_b_proj_ckpt_weight.cuda(), q_b_proj_scale.cuda(), block_size
                    ).cpu()
                elif quant in ["blockfp4"]:
                    assert prefix + "q_b_proj.weight_scale" in checkpoint
                    assert prefix + "q_b_proj.weight_scale_2" in checkpoint
                    up_q_b_proj_weight = decode_e2m1_from_nibbles(
                        unpack_weight_bytes(q_b_proj_ckpt_weight.cuda())
                    ).reshape(*q_b_proj_ckpt_weight.shape[:-1], -1, block_size)
                    q_b_proj_weight = (
                        (
                            up_q_b_proj_weight
                            * checkpoint[prefix + "q_b_proj.weight_scale"]
                            .view(torch.float8_e4m3fn)
                            .unsqueeze(-1)
                            .to(
                                dtype=up_q_b_proj_weight.dtype,
                                device=up_q_b_proj_weight.device,
                            )
                            * checkpoint[prefix + "q_b_proj.weight_scale_2"].to(
                                device=up_q_b_proj_weight.device
                            )
                        )
                        .reshape(
                            q_b_proj_ckpt_weight.shape[0],
                            q_b_proj_ckpt_weight.shape[1] * 2,
                        )
                        .to(dtype=torch.bfloat16, device="cpu")
                    )
                else:
                    raise NotImplementedError(
                        f"infer.mla_absorb=absorb is not implemented for {quant} quantization"
                    )
                q_lora_rank = self.params.q_lora_rank
                q_b_proj_weight_per_head = q_b_proj_weight.view(
                    n_local_heads,
                    self.params.qk_nope_head_dim + self.params.qk_rope_head_dim,
                    q_lora_rank,
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
                    kv_lora_rank,
                )
                kv_b_proj_for_q_b_proj = torch.block_diag(*kv_b_proj_for_q_b_proj)
                new_q_b_proj_nope = (
                    kv_b_proj_for_q_b_proj.t()
                    @ q_b_proj_nope.contiguous().view(-1, q_lora_rank)
                ).view(n_local_heads, kv_lora_rank, q_lora_rank)
                new_q_b_proj = torch.cat(
                    [new_q_b_proj_nope, q_b_proj_rope], dim=1
                ).view(-1, q_lora_rank)
                if quant in [None, "gguf"]:  # blockfp4 skips quantizing MLA
                    new_checkpoint[prefix + "q_b_proj.weight"] = new_q_b_proj
                elif quant in ["blockfp4"]:
                    new_q_b_proj, new_q_b_proj_scale, new_q_b_proj_scale_2 = (
                        fp4_fake_quant(
                            new_q_b_proj,
                            block_scale=None,
                            global_scale=None,
                            quant=True,
                        )
                    )
                    new_q_b_proj = pack_weight_nibbles(to_e2m1_nibbles(new_q_b_proj))
                    new_checkpoint[prefix + "q_b_proj.weight"] = new_q_b_proj
                    new_checkpoint[prefix + "q_b_proj.weight_scale"] = (
                        new_q_b_proj_scale.view(torch.uint8)
                    )
                    new_checkpoint[prefix + "q_b_proj.weight_scale_2"] = (
                        new_q_b_proj_scale_2.view(1, 1)
                    )
                elif quant in ["blockfp8", "q4km"]:
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
                if quant in [None, "gguf"]:  # blockfp4 skips quantizing MLA
                    o_proj_weight = o_proj_ckpt_weight
                elif quant in ["blockfp8", "q4km"]:
                    assert prefix + "o_proj.scale" in checkpoint
                    o_proj_scale = checkpoint[prefix + "o_proj.scale"]
                    # FIXME: Keep this on GPU
                    o_proj_weight = weight_dequant_fn(
                        o_proj_ckpt_weight.cuda(), o_proj_scale.cuda(), block_size
                    ).cpu()
                elif quant in ["blockfp4"]:
                    assert prefix + "o_proj.weight_scale" in checkpoint
                    assert prefix + "o_proj.weight_scale_2" in checkpoint
                    up_o_proj_weight = decode_e2m1_from_nibbles(
                        unpack_weight_bytes(o_proj_ckpt_weight.cuda())
                    ).reshape(*o_proj_ckpt_weight.shape[:-1], -1, block_size)
                    o_proj_weight = (
                        (
                            up_o_proj_weight
                            * checkpoint[prefix + "o_proj.weight_scale"]
                            .view(torch.float8_e4m3fn)
                            .unsqueeze(-1)
                            .to(
                                dtype=up_o_proj_weight.dtype,
                                device=up_o_proj_weight.device,
                            )
                            * checkpoint[prefix + "o_proj.weight_scale_2"].to(
                                device=up_o_proj_weight.device
                            )
                        )
                        .reshape(
                            o_proj_ckpt_weight.shape[0], o_proj_ckpt_weight.shape[1] * 2
                        )
                        .to(dtype=torch.bfloat16, device="cpu")
                    )
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
                    kv_lora_rank,
                )
                kv_b_proj_for_o_proj = torch.block_diag(*kv_b_proj_for_o_proj)
                new_o_proj = o_proj_weight @ kv_b_proj_for_o_proj
                if quant in [None, "gguf"]:  # blockfp4 skips quantizing MLA
                    new_checkpoint[prefix + "o_proj.weight"] = new_o_proj
                elif quant in ["blockfp8", "q4km"]:
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
                elif quant in ["blockfp4"]:
                    new_o_proj, new_o_proj_scale, new_o_proj_scale_2 = fp4_fake_quant(
                        new_o_proj, block_scale=None, global_scale=None, quant=True
                    )
                    new_o_proj = pack_weight_nibbles(to_e2m1_nibbles(new_o_proj))
                    new_checkpoint[prefix + "o_proj.weight"] = new_o_proj
                    new_checkpoint[prefix + "o_proj.weight_scale"] = (
                        new_o_proj_scale.view(torch.uint8)
                    )
                    new_checkpoint[prefix + "o_proj.weight_scale_2"] = (
                        new_o_proj_scale_2.view(1, 1)
                    )
                else:
                    raise NotImplementedError(
                        f"infer.mla_absorb=absorb is not implemented for {quant} quantization"
                    )

            elif (
                k.endswith(".kv_b_proj.scale")
                or k.endswith(".kv_b_proj.weight_scale")
                or k.endswith(".kv_b_proj.weight_scale_2")
                or k.endswith(".kv_b_proj.input_scale")
            ):
                continue

            elif k.endswith(".kv_b_proj.bias"):
                raise NotImplementedError(
                    "infer.mla_absorb=absorb is not implemented for kv_b_proj with a bias"
                )

            elif (
                k.endswith(".o_proj.weight")
                or k.endswith(".o_proj.scale")
                or k.endswith(".o_proj.weight_scale")
                or k.endswith(".o_proj.weight_scale_2")
            ):
                continue

            elif (
                k.endswith(".q_b_proj.weight")
                or k.endswith(".q_b_proj.scale")
                or k.endswith(".q_b_proj.weight_scale")
                or k.endswith(".q_b_proj.weight_scale_2")
            ):
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
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            if quant not in QuantizationRegistry._allowed_quant_for_merge_qkv:
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
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            if quant not in QuantizationRegistry._allowed_quant_for_merge_gate_up:
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
            state_dict = super().process_state_dict_for_renaming_linear_layer(
                state_dict,
                get_global_args().models.n_dense_layers,
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
        import resource

        memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for layer_id in range(self.local_begin_layer_id, self.local_end_layer_id):
            logger.debug(
                f"initing layer : {layer_id}  cpu memory usage: {memory_usage / 1024**2} GB  gpu memory usage : RANK : {torch.cuda.current_device()} {torch.cuda.memory_allocated()/(1024**3)} GB"
            )
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


def get_linear_layout_contig_y(
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
            return LinearMuxiLayoutContigY
        elif quant_method == "blockfp8":
            return Blockfp8LinearMuxiLayoutContigY
        else:
            raise NotImplementedError(
                f'Quantization method {quant_method} is not implemented for "muxi_custom_kernel"'
            )

    else:
        return QuantizationRegistry.get_quantized_linear_class_from_global_args(
            quant_kwargs=quant_kwargs, checkpoint_prefix=checkpoint_prefix
        )
