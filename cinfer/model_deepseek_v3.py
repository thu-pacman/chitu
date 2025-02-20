from typing import List, Optional, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from .model import Attention, Transformer, TransformerBlock, RMSNorm
from .tensor_parallel import (
    get_tp_group,
    get_tp_size,
    get_tp_rank,
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from .ops import act_quant_deepseek_v3, weight_dequant_deepseek_v3, fp8_gemm_deepseek_v3


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
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    else:
        block_size = 128
        x, act_scale = act_quant_deepseek_v3(x, block_size)
        assert weight_scale is not None
        y = fp8_gemm_deepseek_v3(x, act_scale, weight, weight_scale)
        if bias is not None:
            y += bias
        return y


class LinearDeepSeekV3(nn.Module):
    """
    FP8 linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = False,
        bias_dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn)
        )
        block_size = 128
        scale_out_features = (out_features + block_size - 1) // block_size
        scale_in_features = (in_features + block_size - 1) // block_size
        self.scale = nn.Parameter(
            torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
        )
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
        bias_dtype=None,
        gather_output: bool = True,
    ):
        super().__init__(
            in_features,
            out_features,
            has_bias=has_bias,
            dtype=torch.float8_e4m3fn,
            bias_dtype=bias_dtype,
            linear_op=lambda x, w, b: linear_deepseek_v3(x, w, self.scale, b),
            gather_output=gather_output,
        )

        local_out_features, local_in_features = self.weight.shape
        block_size = 128
        scale_out_features = (local_out_features + block_size - 1) // block_size
        scale_in_features = (local_in_features + block_size - 1) // block_size
        self.scale = nn.Parameter(
            torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
        )


class RowParallelLinearDeepSeekV3(RowParallelLinear):
    """
    FP8 row parallel linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = False,
        bias_dtype=None,
        input_is_parallel: bool = False,
    ):
        super().__init__(
            in_features,
            out_features,
            has_bias=has_bias,
            dtype=torch.float8_e4m3fn,
            bias_dtype=bias_dtype,
            linear_op=lambda x, w, b: linear_deepseek_v3(x, w, self.scale, b),
            input_is_parallel=input_is_parallel,
        )

        local_out_features, local_in_features = self.weight.shape
        block_size = 128
        scale_out_features = (local_out_features + block_size - 1) // block_size
        scale_in_features = (local_in_features + block_size - 1) // block_size
        self.scale = nn.Parameter(
            torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
        )


class AttentionDeepSeekV3(Attention):
    def __init__(self, args, layer_id, cache, attn_backend, mla_absorb):
        super().__init__(layer_id, cache, attn_backend)
        self.mla_absorb = mla_absorb

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

        self.wq_a = LinearDeepSeekV3(self.dim, self.q_lora_rank, has_bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = ColumnParallelLinearDeepSeekV3(
            self.q_lora_rank,
            self.n_heads * self.qk_head_dim,
            has_bias=False,
            gather_output=False,
        )
        self.wkv_a = LinearDeepSeekV3(
            self.dim, self.kv_lora_rank + self.qk_rope_head_dim, has_bias=False
        )
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinearDeepSeekV3(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            has_bias=False,
            gather_output=False,
        )
        self.wo = RowParallelLinearDeepSeekV3(
            self.n_heads * self.v_head_dim,
            self.dim,
            has_bias=False,
            input_is_parallel=True,
        )
        self.softmax_scale = self.qk_head_dim**-0.5
        original_seq_len: int = 4096
        if args.max_seq_len > original_seq_len:
            mscale: float = 1.0
            mscale = 0.1 * mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

    def _run_linear(self, x, freqs_cis):
        bsz, seqlen, _ = x.size()
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q_a = self.wq_a(x)
            q = self.wq_b(self.q_norm(q_a, compute_dtype=q_a.dtype))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb_deepseek_v3(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb_deepseek_v3(k_pe.unsqueeze(2), freqs_cis)
        if self.mla_absorb == "none":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(
                bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            return q, k, v
        elif self.mla_absorb == "absorb-without-precomp":
            block_size = 128
            wkv_b = (
                self.wkv_b.weight
                if self.wkv_b.scale is None
                else weight_dequant_deepseek_v3(
                    self.wkv_b.weight, self.wkv_b.scale, block_size
                )
            )
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum(
                "bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim]
            )
            return q_nope, q_pe, kv, k_pe, wkv_b
        else:
            raise NotImplementedError(
                f"MLA absorb mode {self.mla_absorb} not supported"
            )

    def prefill_forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        varlens,
    ):
        start_pos = 0
        bsz = len(varlens.cpu_lens)
        x = x.view(bsz, -1, x.shape[-1])
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        if self.mla_absorb == "none":
            q, k, v = self._run_linear(x, freqs_cis)
            self.cache.finalize_cache_bylayer_prefill(
                k, v, self.cache.curr_req_ids, self.cache.curr_varlens, self.layer_id
            )
            x = self.attn_backend.attn_varlen_func(
                q.view(bsz * seqlen, q.shape[-2], q.shape[-1]),
                k.view(bsz * seqlen, k.shape[-2], k.shape[-1]),
                v.view(bsz * seqlen, v.shape[-2], v.shape[-1]),
                varlens.prefix_lens,
                varlens.prefix_lens,
                varlens.max_len,
                varlens.max_len,
                causal=True,
            ).view(bsz, seqlen, -1)

        elif self.mla_absorb == "absorb-without-precomp":
            q_nope, q_pe, kv, k_pe, wkv_b = self._run_linear(x, freqs_cis)

            kv_cache = self.kv_norm(kv, compute_dtype=kv.dtype)
            pe_cache = k_pe.squeeze(2)
            self.cache.finalize_cache_bylayer_prefill(
                kv_cache[:, start_pos:end_pos],
                pe_cache[:, start_pos:end_pos],
                self.cache.curr_req_ids,
                self.cache.curr_varlens,
                self.layer_id,
            )
            q_nope_pe = torch.cat([q_nope, q_pe], dim=-1)
            kv_pe_cache = torch.cat(
                [kv_cache[:, :end_pos], pe_cache[:, :end_pos]], dim=-1
            )
            x = self.attn_backend.attn_varlen_func(
                q_nope_pe.view(-1, q_nope_pe.shape[-2], q_nope_pe.shape[-1]),
                kv_pe_cache.view(-1, 1, kv_pe_cache.shape[-1]),
                kv_cache[:, :end_pos].view(-1, 1, kv_cache.shape[-1]),
                varlens.prefix_lens,
                varlens.prefix_lens,
                varlens.max_len,
                varlens.max_len,
                causal=True,
            )

            x = x.view(bsz, seqlen, x.shape[-2], x.shape[-1])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])

        else:
            raise NotImplementedError(
                f"MLA absorb mode {self.mla_absorb} not supported"
            )

        x = self._run_output_linear(x)
        return x.view(bsz * seqlen, -1)

    def decode_forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        cache_seqlens = self.cache.get_gpu_seq_lens()
        assert torch.all(cache_seqlens[0] == cache_seqlens)
        start_pos = cache_seqlens[0]
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        if self.mla_absorb == "none":
            q, k, v = self._run_linear(x, freqs_cis)
            q = q.view(bsz, seqlen, self.n_local_heads, -1)
            k = k.view(bsz, seqlen, self.n_local_heads, -1)
            v = v.view(bsz, seqlen, self.n_local_heads, -1)

            cache = self.cache.get_cache_decode(self.layer_id)
            cache_k = cache[0]
            cache_v = cache[1]
            cache_seqlens = self.cache.get_gpu_seq_lens()
            x = self.attn_backend.attn_with_kvcache(
                q,
                cache_k,
                cache_v,
                k,
                v,
                cache_seqlens=cache_seqlens,
            ).view(bsz, seqlen, -1)

        elif self.mla_absorb == "absorb-without-precomp":
            q_nope, q_pe, kv, k_pe, wkv_b = self._run_linear(x, freqs_cis)

            kv_cache, pe_cache = self.cache.get_cache_decode(self.layer_id)
            this_kv = self.kv_norm(kv, compute_dtype=kv.dtype)
            this_pe = k_pe.squeeze(2)
            q_nope_pe = torch.cat([q_nope, q_pe], dim=-1)
            kv_pe_cache = torch.cat([kv_cache, pe_cache], dim=-1)
            this_kv_pe = torch.cat([this_kv, this_pe], dim=-1)
            x = self.attn_backend.attn_with_kvcache(
                q_nope_pe,
                kv_pe_cache.view(kv_pe_cache.shape[0], kv_pe_cache.shape[1], 1, -1),
                kv_cache.view(kv_cache.shape[0], kv_cache.shape[1], 1, -1),
                this_kv_pe.view(this_kv_pe.shape[0], this_kv_pe.shape[1], 1, -1),
                this_kv.view(this_kv.shape[0], this_kv.shape[1], 1, -1),
                cache_seqlens=cache_seqlens,
            )
            pe_cache[:, start_pos] = kv_pe_cache[:, start_pos, self.kv_lora_rank :]

            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])

        else:
            raise NotImplementedError(
                f"MLA absorb mode {self.mla_absorb} not supported"
            )

        x = self._run_output_linear(x)
        return x

    def decode_forward_paged(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        assert False  # TODO

    def _run_output_linear(self, x):
        x = self.wo(x.flatten(2))
        return x


class MLPDeepSeekV3(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = ColumnParallelLinearDeepSeekV3(
            dim, inter_dim, has_bias=False, gather_output=False
        )
        self.w2 = RowParallelLinearDeepSeekV3(
            inter_dim, dim, has_bias=False, input_is_parallel=True
        )
        self.w3 = ColumnParallelLinearDeepSeekV3(
            dim, inter_dim, has_bias=False, gather_output=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


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


class ExpertDeepSeekV3(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = ColumnParallelLinearDeepSeekV3(
            dim, inter_dim, has_bias=False, gather_output=False
        )
        self.w2 = RowParallelLinearDeepSeekV3(
            inter_dim, dim, has_bias=False, input_is_parallel=True
        )
        self.w3 = ColumnParallelLinearDeepSeekV3(
            dim, inter_dim, has_bias=False, gather_output=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


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

    def __init__(self, args):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        moe_world_size = 1
        moe_rank = 0
        assert (
            args.n_routed_experts % moe_world_size == 0
        ), f"Number of experts must be divisible by world size (world_size={moe_world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // moe_world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = moe_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = GateDeepSeekV3(args)
        self.experts = nn.ModuleList(
            [
                (
                    ExpertDeepSeekV3(args.dim, args.moe_inter_dim)
                    if self.experts_start_idx <= i < self.experts_end_idx
                    else None
                )
                for i in range(self.n_routed_experts)
            ]
        )
        self.shared_experts = MLPDeepSeekV3(
            args.dim, args.n_shared_experts * args.moe_inter_dim
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
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        return (y + z).view(shape)


class TransformerBlockDeepSeekV3(TransformerBlock):
    def __init__(self, layer_id: int, args, cache, attn_backend, op_impl, mla_absorb):
        super().__init__(layer_id, args, cache, attn_backend, op_impl)
        self.attn = AttentionDeepSeekV3(
            args, layer_id, cache, attn_backend, mla_absorb=mla_absorb
        )
        self.ffn = (
            MLPDeepSeekV3(args.dim, args.inter_dim)
            if layer_id < args.n_dense_layers
            else MoEDeepSeekV3(args)
        )
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, varlens=None):
        x = x + self.attn(self.attn_norm(x, compute_dtype=x.dtype), freqs_cis, varlens)
        x = x + self.ffn(self.ffn_norm(x, compute_dtype=x.dtype))
        return x


class TransformerDeepSeekV3(Transformer):
    def __init__(
        self,
        params,
        cache,
        pipeline_parallel_size,
        model_parallel_size,
        attn_backend,
        op_impl,
        mla_absorb,
        merge_qkv_gate_up=False,
    ):
        torch.set_default_dtype(torch.bfloat16)
        self.mla_absorb = mla_absorb
        super().__init__(
            params,
            cache,
            pipeline_parallel_size,
            model_parallel_size,
            attn_backend,
            op_impl,
            mla_absorb=mla_absorb,
        )
        if op_impl != "torch":
            raise NotImplementedError("Only op_impl=torch is supported in DeepSeek V3")
        if merge_qkv_gate_up:
            raise NotImplementedError(
                "merge_qkv_gate_up is not supported in DeepSeek V3"
            )
        self.freqs_cis = precompute_freqs_cis_deepseek_v3(params)

    def _get_tensor_column_parallel_layer_names(self) -> List[str]:
        return ["embed", "wq_b", "wkv_b", "w1", "w3", "head"]

    def _get_tensor_row_parallel_layer_names(self) -> List[str]:
        return ["wo", "w2"]

    def _get_pre_layer_prefixes(self) -> List[str]:
        return ["embed."]

    def _get_post_layer_prefixes(self) -> List[str]:
        return ["head.", "norm."]

    def _get_layer_i_prefixes(self, i: int) -> List[str]:
        return [f"layers.{i}."]

    def _init_pre_layers(self):
        self.embed = VocabParallelEmbedding(self.params.vocab_size, self.params.dim)

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
                )
            )

    def _init_post_layers(self):
        self.norm = RMSNorm(self.params.dim)
        self.head = ColumnParallelLinear(
            self.params.dim,
            self.params.vocab_size,
            has_bias=False,
            dtype=torch.get_default_dtype(),
            gather_output=True,
        )

    def _pre_layers(self, h):
        return self.embed(h)

    def _post_layers(self, h):
        """NOTE: _post_layers is assumed to be a token-wise computation"""
        h = self.norm(h, compute_dtype=h.dtype)
        h = self.head(h)
        return h


def precompute_freqs_cis_deepseek_v3(args) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
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


def apply_rotary_emb_deepseek_v3(
    x: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)
