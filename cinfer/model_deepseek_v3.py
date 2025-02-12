from typing import List, Optional, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

import triton
import triton.language as tl
from triton import Config

import fairscale.nn.model_parallel.initialize as fs_init

from .model import Attention, Transformer, TransformerBlock, RMSNorm


class ParallelEmbeddingDeepSeekV3(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.model_parallel_size = fs_init.get_model_parallel_world_size()
        self.vocab_size = vocab_size
        self.dim = dim
        assert (
            vocab_size % self.model_parallel_size == 0
        ), f"Vocabulary size must be divisible by world size (world_size={self.model_parallel_size})"
        self.part_vocab_size = vocab_size // self.model_parallel_size
        self.vocab_start_idx = dist.get_rank() * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """
        if self.model_parallel_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if self.model_parallel_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear_deepseek_v3(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
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
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    else:
        block_size = 128
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class LinearDeepSeekV3(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    # dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    dtype = torch.float8_e4m3fn

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(
                out_features, in_features, dtype=dtype or LinearDeepSeekV3.dtype
            )
        )
        if self.weight.element_size() == 1:
            block_size = 128
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
            )
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
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
        return linear_deepseek_v3(x, self.weight, self.bias)


class ColumnParallelLinearDeepSeekV3(LinearDeepSeekV3):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, dtype=None
    ):
        self.model_parallel_size = fs_init.get_model_parallel_world_size()
        assert (
            out_features % self.model_parallel_size == 0
        ), f"Output features must be divisible by world size (world_size={self.model_parallel_size})"
        self.part_out_features = out_features // self.model_parallel_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear_deepseek_v3(x, self.weight, self.bias)
        return y


class RowParallelLinearDeepSeekV3(LinearDeepSeekV3):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, dtype=None
    ):
        self.model_parallel_size = fs_init.get_model_parallel_world_size()
        assert (
            in_features % self.model_parallel_size == 0
        ), f"Input features must be divisible by world size (world_size={self.model_parallel_size})"
        self.part_in_features = in_features // self.model_parallel_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear_deepseek_v3(x, self.weight)
        if self.model_parallel_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


# TODO: Delete this. This should be the same with our own RMSNorm
class RMSNormDeepSeekV3(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


class AttentionDeepSeekV3(Attention):
    def __init__(self, args, layer_id, cache, attn_backend):
        super().__init__(layer_id, cache, attn_backend)
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // model_parallel_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        self.wq_a = LinearDeepSeekV3(self.dim, self.q_lora_rank)
        self.q_norm = RMSNormDeepSeekV3(self.q_lora_rank)
        self.wq_b = ColumnParallelLinearDeepSeekV3(
            self.q_lora_rank, self.n_heads * self.qk_head_dim
        )
        self.wkv_a = LinearDeepSeekV3(
            self.dim, self.kv_lora_rank + self.qk_rope_head_dim
        )
        self.kv_norm = RMSNormDeepSeekV3(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinearDeepSeekV3(
            self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)
        )
        self.wo = RowParallelLinearDeepSeekV3(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim**-0.5
        original_seq_len: int = 4096
        if args.max_seq_len > original_seq_len:
            mscale: float = 1.0
            mscale = 0.1 * mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # TODO: Use our own KV cache
        max_batch_size: int = 1
        self.register_buffer(
            "kv_cache",
            torch.zeros(max_batch_size, args.max_seq_len, self.kv_lora_rank),
            persistent=False,
        )
        self.register_buffer(
            "pe_cache",
            torch.zeros(max_batch_size, args.max_seq_len, self.qk_rope_head_dim),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        varlens=None,
    ):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """

        if varlens is not None:  # Prefill:
            start_pos = 0
            bsz = len(varlens.cpu_lens)
            x = x.view(bsz, -1, x.shape[-1])
        else:  # Decode
            cache_seqlens = self.cache.get_gpu_seq_lens()
            assert torch.all(cache_seqlens[0] == cache_seqlens)
            start_pos = cache_seqlens[0]

        bsz, seqlen, _ = x.size()

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device).triu_(1)

        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb_deepseek_v3(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb_deepseek_v3(k_pe.unsqueeze(2), freqs_cis)
        block_size = 128
        wkv_b = (
            self.wkv_b.weight
            if self.wkv_b.scale is None
            else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
        )
        wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
        q_nope = torch.einsum(
            "bshd,hdc->bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim]
        )
        self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

        if varlens is not None:  # Prefill:
            self.cache.finalize_cache_bylayer_prefill(
                self.kv_cache[:bsz, start_pos:end_pos],
                self.pe_cache[:bsz, start_pos:end_pos],
                self.cache.curr_req_ids,
                self.cache.curr_varlens,
                self.layer_id,
            )

        scores = (
            torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
            + torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
        ) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim :])
        x = self.wo(x.flatten(2))

        if varlens is not None:  # Prefill:
            x = x.view(bsz * seqlen, -1)

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
        self.w1 = ColumnParallelLinearDeepSeekV3(dim, inter_dim)
        self.w2 = RowParallelLinearDeepSeekV3(inter_dim, dim)
        self.w3 = ColumnParallelLinearDeepSeekV3(dim, inter_dim)

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
        scores = linear_deepseek_v3(x, self.weight)
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
        self.w1 = LinearDeepSeekV3(dim, inter_dim)
        self.w2 = LinearDeepSeekV3(inter_dim, dim)
        self.w3 = LinearDeepSeekV3(dim, inter_dim)

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
        # if co_moe.world_size > 1:
        #    dist.all_reduce(y, group=co_moe.group)
        return (y + z).view(shape)


class TransformerBlockDeepSeekV3(TransformerBlock):
    def __init__(self, layer_id: int, args, cache, attn_backend, op_impl):
        super().__init__(layer_id, args, cache, attn_backend, op_impl)
        self.attn = AttentionDeepSeekV3(args, layer_id, cache, attn_backend)
        self.ffn = (
            MLPDeepSeekV3(args.dim, args.inter_dim)
            if layer_id < args.n_dense_layers
            else MoEDeepSeekV3(args)
        )
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, varlens=None):
        x = x + self.attn(self.attn_norm(x), freqs_cis, varlens)
        x = x + self.ffn(self.ffn_norm(x))
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
        merge_qkv_gate_up=False,
    ):
        torch.set_default_dtype(torch.bfloat16)
        super().__init__(
            params,
            cache,
            pipeline_parallel_size,
            model_parallel_size,
            attn_backend,
            op_impl,
        )
        if op_impl != "torch":
            raise NotImplementedError("Only op_impl=torch is supported in DeepSeek V3")
        if merge_qkv_gate_up:
            raise NotImplementedError(
                "merge_qkv_gate_up is not supported in DeepSeek V3"
            )
        self.freqs_cis = precompute_freqs_cis_deepseek_v3(params)

    def _get_tensor_column_parallel_layer_names(self) -> List[str]:
        assert False  # TODO: Implement this

    def _get_tensor_row_parallel_layer_names(self) -> List[str]:
        assert False  # TODO: Implement this

    def _get_pre_layer_prefixes(self) -> List[str]:
        return ["embed."]

    def _get_post_layer_prefixes(self) -> List[str]:
        return ["head.", "norm."]

    def _get_layer_i_prefixes(self, i: int) -> List[str]:
        return [f"layers.{i}."]

    def _init_pre_layers(self):
        self.embed = ParallelEmbeddingDeepSeekV3(
            self.params.vocab_size, self.params.dim
        )

    def _init_layers(self, cache, attn_backend, op_impl):
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.local_begin_layer_id, self.local_end_layer_id):
            self.layers.append(
                TransformerBlockDeepSeekV3(
                    layer_id, self.params, cache, attn_backend, self.op_impl
                )
            )

    def _init_post_layers(self):
        self.norm = RMSNormDeepSeekV3(self.params.dim)
        self.head = ColumnParallelLinearDeepSeekV3(
            self.params.dim, self.params.vocab_size, dtype=torch.get_default_dtype()
        )

    def _pre_layers(self, h):
        return self.embed(h)

    def _post_layers(self, h):
        """NOTE: _post_layers is assumed to be a token-wise computation"""
        h = self.norm(h)
        h = self.head(h)
        return h


def autoretry(fn):
    import time, random

    def wrapped(*args, **kwargs):
        i = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if i >= 30:
                    raise e
                i += 1
                msg = str(e)
                if (
                    "cannot stat shared object: Stale file handle" in msg
                    or "No such file or directory"
                    and "/.triton/cache/" in msg
                ):
                    time.sleep(random.random() * 2 + 1)
                    continue
                raise e

    return wrapped


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


@autoretry
def act_quant(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


@autoretry
def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


fp8_gemm_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=fp8_gemm_configs, key=["N", "K"])
@triton.jit
def fp8_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


@autoretry
def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    """
    Perform a matrix multiplication using FP8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert (
        a_s.is_contiguous() and b_s.is_contiguous()
    ), "Scaling factor tensors must be contiguous"
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c


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
