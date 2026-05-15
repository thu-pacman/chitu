# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import functools
import torch

from chitu.utils import try_import_and_setup_torch_npu
from chitu.quantization.base import QuantizedLinearBase, QuantizedMoeExpertsMerged
from chitu.distributed.parallel_state import get_tp_group
from chitu.quantization.registry import QuantizationRegistry
from chitu.native_layout import (
    enable_native_layout_weight,
    NpuFractalZnTensor,
    Repeat1ToLength,
    SqueezeLastSingleton,
)
from chitu.moe.batched_expert_result import BatchedExpertResult
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    ConcatPermutedBatchedRoutedActivationMinimal,
)
from chitu.ops import a8_per_token_act_quant
from chitu.ops.utils import make_op_dispatcher
from chitu.lazy import eval_lazy

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()

if has_torch_npu:
    from chitu.moe.experts import (
        fused_experts_no_sum_npu,
        fused_experts_npu_for_ep,
    )


@QuantizationRegistry.register_linear("ascend_w8a8")
class AscendW8A8Linear(
    enable_native_layout_weight("weight", NpuFractalZnTensor),
    enable_native_layout_weight(
        "input_scale",
        Repeat1ToLength,
        length=(lambda m: m.in_features),
        out_dtype=(lambda m: torch.get_default_dtype()),
    ),
    enable_native_layout_weight(
        "input_offset",
        Repeat1ToLength,
        length=(lambda m: m.in_features),
        out_dtype=(lambda m: torch.get_default_dtype()),
    ),
    QuantizedLinearBase,
):
    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        *,
        ############################################
        # Parameters specific to this quantization
        is_rpl: bool = False,
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

        self.deq_scale = torch.nn.Parameter(
            torch.ones(
                self.out_features,
                dtype=(
                    torch.float32
                    if torch.get_default_dtype() == torch.bfloat16
                    else torch.int64
                ),
            ),
            requires_grad=False,
        )

        self.input_scale = torch.nn.Parameter(
            torch.ones(1, dtype=torch.get_default_dtype()),
            requires_grad=False,
        )
        self.input_offset = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.int8),
            requires_grad=False,
        )

        self.quant_bias = torch.nn.Parameter(
            torch.zeros(
                self.out_features,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        self.is_rpl = is_rpl
        self._input_scale_layout_kwargs = dict(
            length=self.in_features, out_dtype=torch.get_default_dtype()
        )
        self._input_offset_layout_kwargs = dict(
            length=self.in_features, out_dtype=torch.get_default_dtype()
        )
        if has_bias:
            self.register_parameter(
                "bias",
                torch.nn.Parameter(
                    torch.empty(out_features, dtype=torch.get_default_dtype()),
                    requires_grad=False,
                ),
            )
        else:
            self.register_parameter("bias", None)

        self._ready = False

    @torch.no_grad()
    def _maybe_build_quant_params(self):
        if getattr(self, "_ready", False):
            return
        scale_vec = self.input_scale.detach()
        rec = (1.0 / scale_vec).to(scale_vec.dtype)

        if hasattr(self, "aclnn_input_scale_reciprocal"):
            self.aclnn_input_scale_reciprocal.copy_(rec)
        else:
            self.register_buffer("aclnn_input_scale_reciprocal", rec, persistent=True)

        off = self.input_offset.detach().to(dtype=scale_vec.dtype)
        if hasattr(self, "aclnn_input_offset"):
            self.aclnn_input_offset.copy_(off)
        else:
            self.register_buffer("aclnn_input_offset", off, persistent=True)
        self._ready = True

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = eval_lazy(x)

        if x.shape[0] == 0:
            return torch.empty(
                [0, self.out_features], dtype=torch.get_default_dtype(), device=x.device
            )

        self._maybe_build_quant_params()

        if x.dtype != torch.int8:
            x = torch_npu.npu_quantize(
                x,
                self.aclnn_input_scale_reciprocal,
                self.aclnn_input_offset,
                torch.qint8,
                -1,
                False,
            )
        quant_bias = (
            self.quant_bias
            if ((get_tp_group().rank_in_group == 0 and self.is_rpl) or not self.is_rpl)
            else None
        )
        output = torch_npu.npu_quant_matmul(
            x,
            self.weight,
            self.deq_scale,
            bias=quant_bias,
            output_dtype=torch.get_default_dtype(),
        )
        if self.bias is not None:
            output += self.bias
        return output


@QuantizationRegistry.register_linear("ascend_w8a8_dynamic")
class AscendW8A8DynamicLinear(
    enable_native_layout_weight("weight", NpuFractalZnTensor),
    enable_native_layout_weight("weight_scale", SqueezeLastSingleton),
    QuantizedLinearBase,
):
    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        ############################################
        # Parameters specific to this quantization
        weight_scale_dtype: torch.dtype = None,
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

        self.weight_scale = torch.nn.Parameter(
            torch.ones(
                self.out_features,
                1,
                dtype=(
                    torch.get_default_dtype()
                    if weight_scale_dtype is None
                    else eval(weight_scale_dtype)
                ),
            ),
            requires_grad=False,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = eval_lazy(x)

        if x.shape[0] == 0:
            return torch.empty([0, self.out_features], dtype=x.dtype, device=x.device)
        output_dtype = x.dtype
        quantized_x, dynamic_scale = a8_per_token_act_quant(
            x.view(-1, self.in_features)
        )
        output = torch_npu.npu_quant_matmul(
            quantized_x,
            self.weight,
            self.weight_scale,
            pertoken_scale=dynamic_scale,
            bias=None,
            output_dtype=output_dtype,
        )
        return output


def _finalize_fused_experts_sum_output(
    output, hidden_states, topk_weights: torch.Tensor, inplace: bool
):
    if hasattr(output, "weighted_sum"):
        out = hidden_states.activation if inplace else None
        return output.weighted_sum(topk_weights, out=out)
    return output


@make_op_dispatcher
def fused_experts_no_sum_ascend_w8a8_indexed(
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
): ...


@fused_experts_no_sum_ascend_w8a8_indexed.register_auto
def _auto_fused_experts_no_sum_ascend_w8a8_indexed():
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


fused_experts_no_sum_ascend_w8a8_indexed.register_candidate("torch_npu")
if has_torch_npu:
    fused_experts_no_sum_ascend_w8a8_indexed.register("torch_npu")(
        fused_experts_no_sum_npu
    )


@make_op_dispatcher
def fused_experts_no_sum_ascend_w8a8_concat_permuted(
    hidden_states,
    w1,
    w2,
    *,
    impl: str = "auto",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    experts_start_idx: int = 0,
    use_int8_w8a8: bool = False,
): ...


@fused_experts_no_sum_ascend_w8a8_concat_permuted.register_auto
def _auto_fused_experts_no_sum_ascend_w8a8_concat_permuted():
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


fused_experts_no_sum_ascend_w8a8_concat_permuted.register_candidate("torch_npu")
if has_torch_npu:
    fused_experts_no_sum_ascend_w8a8_concat_permuted.register("torch_npu")(
        fused_experts_npu_for_ep
    )


@QuantizationRegistry.register_moe_experts("ascend_w8a8_dynamic", merge_gate_up=True)
class AscendW8A8DynamicMoeExperts(
    enable_native_layout_weight("gate_up_proj_weight", NpuFractalZnTensor),
    enable_native_layout_weight("down_proj_weight", NpuFractalZnTensor),
    enable_native_layout_weight("gate_up_proj_weight_scale", SqueezeLastSingleton),
    enable_native_layout_weight("down_proj_weight_scale", SqueezeLastSingleton),
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
                (self.group_size, moe_inter_dim * 2, self.dim),
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        self.gate_up_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                (self.group_size, moe_inter_dim * 2, 1),
                dtype=torch.get_default_dtype(),
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
                (self.group_size, self.dim, 1),
                dtype=torch.get_default_dtype(),
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
        return fused_experts_no_sum_ascend_w8a8_indexed(
            routed_x,
            w1=self.get_native_layout_gate_up_proj_weight(),
            w1_scale=self.gate_up_proj_weight_scale,
            w2=self.get_native_layout_down_proj_weight(),
            w2_scale=self.down_proj_weight_scale,
            use_int8_w8a8=True,
            impl=impl,
            global_num_experts=self.global_n_experts,
            experts_start_idx=self.experts_start_idx,
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
        return fused_experts_sum_ascend_w8a8_indexed(
            routed_x,
            w1=self.get_native_layout_gate_up_proj_weight(),
            w1_scale=self.gate_up_proj_weight_scale,
            w2=self.get_native_layout_down_proj_weight(),
            w2_scale=self.down_proj_weight_scale,
            topk_weights=weights,
            use_int8_w8a8=True,
            impl=impl,
            global_num_experts=self.global_n_experts,
            experts_start_idx=self.experts_start_idx,
        )

    @forward_no_sum.register
    def _(
        self,
        routed_x: ConcatPermutedBatchedRoutedActivationMinimal,
        impl: str = "torch_npu",
    ) -> BatchedExpertResult:
        return fused_experts_no_sum_ascend_w8a8_concat_permuted(
            routed_x,
            w1=self.get_native_layout_gate_up_proj_weight(),
            w1_scale=self.gate_up_proj_weight_scale,
            w2=self.get_native_layout_down_proj_weight(),
            w2_scale=self.down_proj_weight_scale,
            use_int8_w8a8=True,
            impl=impl,
            global_num_experts=self.global_n_experts,
            experts_start_idx=self.experts_start_idx,
        )

    @forward.register
    def _(
        self,
        routed_x: ConcatPermutedBatchedRoutedActivationMinimal,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "torch_npu",
    ) -> torch.Tensor:
        return fused_experts_sum_ascend_w8a8_concat_permuted(
            routed_x,
            w1=self.get_native_layout_gate_up_proj_weight(),
            w1_scale=self.gate_up_proj_weight_scale,
            w2=self.get_native_layout_down_proj_weight(),
            w2_scale=self.down_proj_weight_scale,
            topk_weights=weights,
            use_int8_w8a8=True,
            impl=impl,
            global_num_experts=self.global_n_experts,
            experts_start_idx=self.experts_start_idx,
        )


@make_op_dispatcher
def fused_experts_sum_ascend_w8a8_indexed(
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
): ...


@fused_experts_sum_ascend_w8a8_indexed.register_auto
def _auto_fused_experts_sum_ascend_w8a8_indexed():
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


@fused_experts_sum_ascend_w8a8_indexed.register("torch_npu")
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
    impl: str,
):
    output = fused_experts_no_sum_ascend_w8a8_indexed(
        hidden_states,
        w1,
        w2,
        impl=impl,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        global_num_experts=global_num_experts,
        experts_start_idx=experts_start_idx,
        use_int8_w8a8=use_int8_w8a8,
    )
    return _finalize_fused_experts_sum_output(
        output, hidden_states, topk_weights=topk_weights, inplace=inplace
    )


@make_op_dispatcher
def fused_experts_sum_ascend_w8a8_concat_permuted(
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
): ...


@fused_experts_sum_ascend_w8a8_concat_permuted.register_auto
def _auto_fused_experts_sum_ascend_w8a8_concat_permuted():
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


@fused_experts_sum_ascend_w8a8_concat_permuted.register("torch_npu")
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
    impl: str,
):
    output = fused_experts_no_sum_ascend_w8a8_concat_permuted(
        hidden_states,
        w1,
        w2,
        impl=impl,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        experts_start_idx=experts_start_idx,
        use_int8_w8a8=use_int8_w8a8,
    )
    return _finalize_fused_experts_sum_output(
        output, hidden_states, topk_weights=topk_weights, inplace=inplace
    )
