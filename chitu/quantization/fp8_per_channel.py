# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools

import torch
from typing_extensions import override

from chitu.ops.quant import per_token_quant_fp8
from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.base import (
    QuantizedLinearBase,
    QuantizedMoeExpertsMerged,
)
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
)
from chitu.moe.batched_expert_result import (
    BatchedExpertResult,
    PerTokenBatchedExpertResult,
)
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")


@QuantizationRegistry.register_linear("fp8_per_channel", when=lambda _: has_triton)
class Fp8PerChannelLinear(QuantizedLinearBase):
    """
    FP8 per-channel weight quantized linear layer.
    Weights: float8_e4m3fn, shape (out_features, in_features)
    Weight scale: float32, shape (out_features, 1) — one scale per output channel
    Activations: dynamically quantized per-token at runtime
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
    ):
        super().__init__(in_features, out_features, has_bias)

        self.weight = torch.nn.Parameter(
            torch.zeros(
                self.out_features,
                self.in_features,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        self.weight_scale = torch.nn.Parameter(
            torch.ones(
                self.out_features,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        if has_bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(
                    (self.out_features,),
                    dtype=torch.get_default_dtype(),
                ),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        out = _fp8_per_channel_gemm(x, self.weight, self.weight_scale)
        if self.bias is not None:
            out = out + self.bias
        return out.to(x.dtype).view(*orig_shape[:-1], -1)


def _fp8_per_channel_gemm(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    """
    FP8 per-channel GEMM: x (M, K) bf16 @ weight (N, K) fp8 -> (M, N) bf16
    weight_scale: (N, 1) fp32
    """
    qx, x_scale = per_token_quant_fp8(x)
    out = torch._scaled_mm(
        qx,
        weight.t(),
        out_dtype=torch.bfloat16,
        scale_a=x_scale,
        scale_b=weight_scale.t(),
    )
    return out.to(x.dtype)


@QuantizationRegistry.register_moe_experts(
    "fp8_per_channel", merge_gate_up=True, when=lambda _: has_triton
)
class Fp8PerChannelMoeExpertsMerged(QuantizedMoeExpertsMerged):
    """
    FP8 per-channel quantized MoE experts with merged gate and up projection.
    Uses iterative per-expert forward as initial implementation.
    """

    def __init__(
        self,
        dim: int,
        moe_inter_dim: int,
        global_n_experts: int,
        experts_start_idx: int,
        experts_end_idx: int,
        n_activated_experts: int,
        checkpoint_prefix: str,
    ):
        super().__init__(
            dim,
            moe_inter_dim,
            global_n_experts,
            experts_start_idx,
            experts_end_idx,
            n_activated_experts,
            checkpoint_prefix,
        )

        self.gate_up_proj_weight = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                moe_inter_dim * 2,
                self.dim,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        self.gate_up_proj_weight_scale = torch.nn.Parameter(
            torch.ones(
                self.group_size,
                moe_inter_dim * 2,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.down_proj_weight = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                self.dim,
                moe_inter_dim,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        self.down_proj_weight_scale = torch.nn.Parameter(
            torch.ones(
                self.group_size,
                self.dim,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    @override
    @functools.singledispatchmethod
    def forward_no_sum(
        self, routed_x: BatchedRoutedActivation, impl: str = "auto"
    ) -> BatchedExpertResult:
        return super().forward_no_sum(routed_x, impl=impl)

    @forward_no_sum.register
    def _(
        self, routed_x: IndexedBatchedRoutedActivation, impl: str = "auto"
    ) -> PerTokenBatchedExpertResult:
        from chitu.moe.experts.triton_fused_experts import (
            fused_experts_fp8_per_channel,
        )

        return fused_experts_fp8_per_channel(
            routed_x,
            self.gate_up_proj_weight,
            self.down_proj_weight,
            activation="silu",
            w1_scale=self.gate_up_proj_weight_scale.squeeze(-1),  # (E, N, 1) -> (E, N)
            w2_scale=self.down_proj_weight_scale.squeeze(-1),  # (E, N, 1) -> (E, N)
            experts_start_idx=self.experts_start_idx,
        )

    @override
    def forward_ith_expert_gate_up(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return _fp8_per_channel_gemm(
            x, self.gate_up_proj_weight[i], self.gate_up_proj_weight_scale[i]
        )

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return _fp8_per_channel_gemm(
            x, self.down_proj_weight[i], self.down_proj_weight_scale[i]
        )
