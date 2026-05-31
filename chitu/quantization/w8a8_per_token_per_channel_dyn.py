# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import functools

import torch

from chitu.moe.batched_expert_result import BatchedExpertResult
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    ConcatPermutedBatchedRoutedActivationMinimal,
)
from chitu.native_layout import enable_native_layout_weight, NpuFractalZnTensor
from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.base import QuantizedLinearBase, QuantizedMoeExpertsMerged
from chitu.ops.quant import w8a8_gemm_per_token_per_channel, a8_per_token_act_quant
from chitu.ops.utils import make_op_dispatcher
from chitu.import_utils import try_import_and_setup_torch_npu
from chitu.utils import parse_dtype

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()

if has_torch_npu:
    from chitu.moe.experts import fused_experts_no_sum_npu, fused_experts_npu_for_ep


@QuantizationRegistry.register_linear("w8a8_per_token_per_channel_dyn")
class W8A8PerTokenPerChannelDynLinear(QuantizedLinearBase):
    """
    int8 weight + int8 dynamic activation quantized linear layer.
    """

    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        ############################################
        # No parameters specific to this quantization
        weight_scale_dtype: Optional[torch.dtype | str] = None,
        weight_scale_has_singleton_last_dim: bool = False,
    ):
        super().__init__(in_features, out_features, has_bias)

        self.weight = torch.nn.Parameter(
            torch.zeros(
                self.out_features,
                self.in_features,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )

        if weight_scale_dtype is None:
            weight_scale_dtype = torch.get_default_dtype()
        elif isinstance(weight_scale_dtype, str):
            weight_scale_dtype = parse_dtype(weight_scale_dtype)
        self.weight_scale = torch.nn.Parameter(
            torch.ones(
                (
                    (self.out_features, 1)
                    if weight_scale_has_singleton_last_dim
                    else (self.out_features,)
                ),
                dtype=weight_scale_dtype,
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

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_x, act_scale = a8_per_token_act_quant(x)
        out = w8a8_gemm_per_token_per_channel(
            q_x,
            act_scale,
            self.weight,
            self.weight_scale.view(self.out_features),
        ).view(*x.shape[:-1], -1)
        if self.bias is not None:
            out += self.bias
        return out


@QuantizationRegistry.register_linear(
    "w8a8_per_token_per_channel_dyn", when=lambda _: has_torch_npu, priority=1
)
class AscendW8A8PerTokenPerChannelDynLinear(
    enable_native_layout_weight("weight", NpuFractalZnTensor),
    W8A8PerTokenPerChannelDynLinear,
):
    """
    Ascend implementation of W8A8PerTokenPerChannelDynLinear using NpuFuFractalZnTensor
    layout.
    """

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return torch.empty([0, self.out_features], dtype=x.dtype, device=x.device)
        quantized_x, dynamic_scale = a8_per_token_act_quant(
            x.view(-1, self.in_features)
        )
        y = w8a8_gemm_per_token_per_channel(
            quantized_x,
            dynamic_scale,
            self.get_native_layout_weight(),
            self.weight_scale.view(self.out_features),
        )
        if self.bias is not None:
            y += self.bias
        return y.view(*x.shape[:-1], y.shape[-1])


def _finalize_fused_experts_sum_output(
    output, hidden_states, topk_weights: torch.Tensor, inplace: bool
):
    if hasattr(output, "weighted_sum"):
        out = hidden_states.activation if inplace else None
        return output.weighted_sum(topk_weights, out=out)
    return output


@make_op_dispatcher
def fused_experts_no_sum_w8a8_per_token_per_channel_dyn_indexed(
    hidden_states,
    w1,
    w2,
    *,
    impl: str = "auto",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
    use_int8_w8a8: bool = False,
    swiglu_limit: Optional[float] = None,
): ...


@fused_experts_no_sum_w8a8_per_token_per_channel_dyn_indexed.register_auto
def _auto_fused_experts_no_sum_w8a8_per_token_per_channel_dyn_indexed():
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


fused_experts_no_sum_w8a8_per_token_per_channel_dyn_indexed.register_candidate(
    "torch_npu"
)
if has_torch_npu:
    fused_experts_no_sum_w8a8_per_token_per_channel_dyn_indexed.register("torch_npu")(
        fused_experts_no_sum_npu
    )


@make_op_dispatcher
def fused_experts_no_sum_w8a8_per_token_per_channel_dyn_concat_permuted(
    hidden_states,
    w1,
    w2,
    *,
    impl: str = "auto",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    experts_start_idx: int = 0,
    use_int8_w8a8: bool = False,
    swiglu_limit: Optional[float] = None,
) -> BatchedExpertResult: ...


@fused_experts_no_sum_w8a8_per_token_per_channel_dyn_concat_permuted.register_auto
def _auto_fused_experts_no_sum_w8a8_per_token_per_channel_dyn_concat_permuted():
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


fused_experts_no_sum_w8a8_per_token_per_channel_dyn_concat_permuted.register_candidate(
    "torch_npu"
)
if has_torch_npu:
    fused_experts_no_sum_w8a8_per_token_per_channel_dyn_concat_permuted.register(
        "torch_npu"
    )(fused_experts_npu_for_ep)


@make_op_dispatcher
def fused_experts_sum_w8a8_per_token_per_channel_dyn_indexed(
    hidden_states,
    w1,
    w2,
    topk_weights: torch.Tensor,
    *,
    inplace: bool = False,
    impl: str = "auto",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
    use_int8_w8a8: bool = False,
    swiglu_limit: Optional[float] = None,
) -> torch.Tensor: ...


@fused_experts_sum_w8a8_per_token_per_channel_dyn_indexed.register_auto
def _auto_fused_experts_sum_w8a8_per_token_per_channel_dyn_indexed():
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


@fused_experts_sum_w8a8_per_token_per_channel_dyn_indexed.register("torch_npu")
def _run_sum_indexed_torch_npu(
    hidden_states,
    w1,
    w2,
    topk_weights: torch.Tensor,
    *,
    inplace: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
    use_int8_w8a8: bool = False,
    swiglu_limit: Optional[float] = None,
    impl: str,
) -> torch.Tensor:
    output = fused_experts_no_sum_w8a8_per_token_per_channel_dyn_indexed(
        hidden_states,
        w1,
        w2,
        impl=impl,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        global_num_experts=global_num_experts,
        experts_start_idx=experts_start_idx,
        use_int8_w8a8=use_int8_w8a8,
        swiglu_limit=swiglu_limit,
    )
    return _finalize_fused_experts_sum_output(
        output, hidden_states, topk_weights=topk_weights, inplace=inplace
    )


@make_op_dispatcher
def fused_experts_sum_w8a8_per_token_per_channel_dyn_concat_permuted(
    hidden_states,
    w1,
    w2,
    topk_weights: torch.Tensor,
    *,
    inplace: bool = False,
    impl: str = "auto",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    experts_start_idx: int = 0,
    use_int8_w8a8: bool = False,
    swiglu_limit: Optional[float] = None,
) -> torch.Tensor: ...


@fused_experts_sum_w8a8_per_token_per_channel_dyn_concat_permuted.register_auto
def _auto_fused_experts_sum_w8a8_per_token_per_channel_dyn_concat_permuted():
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


@fused_experts_sum_w8a8_per_token_per_channel_dyn_concat_permuted.register("torch_npu")
def _run_sum_concat_torch_npu(
    hidden_states,
    w1,
    w2,
    topk_weights: torch.Tensor,
    *,
    inplace: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    experts_start_idx: int = 0,
    use_int8_w8a8: bool = False,
    swiglu_limit: Optional[float] = None,
    impl: str,
) -> torch.Tensor:
    output = fused_experts_no_sum_w8a8_per_token_per_channel_dyn_concat_permuted(
        hidden_states,
        w1,
        w2,
        impl=impl,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        experts_start_idx=experts_start_idx,
        use_int8_w8a8=use_int8_w8a8,
        swiglu_limit=swiglu_limit,
    )
    return _finalize_fused_experts_sum_output(
        output, hidden_states, topk_weights=topk_weights, inplace=inplace
    )


@QuantizationRegistry.register_moe_experts(
    "w8a8_per_token_per_channel_dyn", merge_gate_up=True
)
class AscendW8A8PerTokenPerChannelDynMoeExperts(
    enable_native_layout_weight("gate_up_proj_weight", NpuFractalZnTensor),
    enable_native_layout_weight("down_proj_weight", NpuFractalZnTensor),
    QuantizedMoeExpertsMerged,
):
    """
    AscendW8A8Dynamic quantized MoeExperts
    """

    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        dim: int,
        moe_inter_dim: int,
        global_n_experts: int,
        experts_start_idx: int,
        experts_end_idx: int,
        n_activated_experts: int,
        checkpoint_prefix: str,
        ############################################
        # No parameters specific to this quantization
        weight_scale_dtype: Optional[torch.dtype | str] = None,
        weight_scale_has_singleton_last_dim: bool = False,
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

        if weight_scale_dtype is None:
            weight_scale_dtype = torch.get_default_dtype()
        elif isinstance(weight_scale_dtype, str):
            weight_scale_dtype = parse_dtype(weight_scale_dtype)

        self.gate_up_proj_weight = torch.nn.Parameter(
            torch.empty(
                (self.group_size, moe_inter_dim * 2, self.dim),
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        self.gate_up_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                (
                    (self.group_size, moe_inter_dim * 2, 1)
                    if weight_scale_has_singleton_last_dim
                    else (self.group_size, moe_inter_dim * 2)
                ),
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        self.down_proj_weight = torch.nn.Parameter(
            torch.empty(
                (self.group_size, self.dim, moe_inter_dim),
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        self.down_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                (
                    (self.group_size, self.dim, 1)
                    if weight_scale_has_singleton_last_dim
                    else (self.group_size, self.dim)
                ),
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )

    @override
    @functools.singledispatchmethod
    def forward_no_sum(
        self, routed_x: BatchedRoutedActivation, impl: str = "torch_npu"
    ) -> BatchedExpertResult:
        return super().forward_no_sum(routed_x, impl=impl)

    @forward_no_sum.register
    def _(
        self, routed_x: IndexedBatchedRoutedActivation, impl: str = "torch_npu"
    ) -> BatchedExpertResult:
        return fused_experts_no_sum_w8a8_per_token_per_channel_dyn_indexed(
            routed_x,
            w1=self.get_native_layout_gate_up_proj_weight(),
            w1_scale=self.gate_up_proj_weight_scale.view(
                self.group_size, self.moe_inter_dim * 2
            ),
            w2=self.get_native_layout_down_proj_weight(),
            w2_scale=self.down_proj_weight_scale.view(self.group_size, self.dim),
            use_int8_w8a8=True,
            impl=impl,
            global_num_experts=self.global_n_experts,
            experts_start_idx=self.experts_start_idx,
            swiglu_limit=self.swiglu_limit,
        )

    @override
    @functools.singledispatchmethod
    def forward(
        self,
        routed_x: BatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "torch_npu",
    ) -> torch.Tensor:
        return super().forward(routed_x, weights, inplace=inplace, impl=impl)

    @forward.register
    def _(
        self,
        routed_x: IndexedBatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "torch_npu",
    ) -> torch.Tensor:
        return fused_experts_sum_w8a8_per_token_per_channel_dyn_indexed(
            routed_x,
            w1=self.get_native_layout_gate_up_proj_weight(),
            w1_scale=self.gate_up_proj_weight_scale.view(
                self.group_size, self.moe_inter_dim * 2
            ),
            w2=self.get_native_layout_down_proj_weight(),
            w2_scale=self.down_proj_weight_scale.view(self.group_size, self.dim),
            topk_weights=weights,
            use_int8_w8a8=True,
            impl=impl,
            global_num_experts=self.global_n_experts,
            experts_start_idx=self.experts_start_idx,
            swiglu_limit=self.swiglu_limit,
        )

    @forward_no_sum.register
    def _(
        self,
        routed_x: ConcatPermutedBatchedRoutedActivationMinimal,
        impl: str = "torch_npu",
    ) -> BatchedExpertResult:
        return fused_experts_no_sum_w8a8_per_token_per_channel_dyn_concat_permuted(
            routed_x,
            w1=self.get_native_layout_gate_up_proj_weight(),
            w1_scale=self.gate_up_proj_weight_scale.view(
                self.group_size, self.moe_inter_dim * 2
            ),
            w2=self.get_native_layout_down_proj_weight(),
            w2_scale=self.down_proj_weight_scale.view(self.group_size, self.dim),
            use_int8_w8a8=True,
            impl=impl,
            experts_start_idx=self.experts_start_idx,
            swiglu_limit=self.swiglu_limit,
        )

    @forward.register
    def _(
        self,
        routed_x: ConcatPermutedBatchedRoutedActivationMinimal,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "torch_npu",
    ) -> torch.Tensor:
        return fused_experts_sum_w8a8_per_token_per_channel_dyn_concat_permuted(
            routed_x,
            w1=self.get_native_layout_gate_up_proj_weight(),
            w1_scale=self.gate_up_proj_weight_scale.view(
                self.group_size, self.moe_inter_dim * 2
            ),
            w2=self.get_native_layout_down_proj_weight(),
            w2_scale=self.down_proj_weight_scale.view(self.group_size, self.dim),
            topk_weights=weights,
            use_int8_w8a8=True,
            impl=impl,
            experts_start_idx=self.experts_start_idx,
            swiglu_limit=self.swiglu_limit,
        )
