# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import torch

from chitu.utils import try_import_and_setup_torch_npu
from chitu.quantization.base import QuantizedLinearBase, QuantizedMoeExpertsBase
from chitu.distributed.parallel_state import get_tp_group, get_ep_size
from chitu.quantization.registry import QuantizationRegistry
from chitu.native_layout import (
    enable_native_layout_weight,
    NpuFractalZnTensor,
    Repeat1ToLength,
    SqueezeLastSingleton,
)
from chitu.moe.batched_routed_activation import BatchedRoutedActivation

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
if has_torch_npu:
    from chitu.npu_utils import fused_experts_npu
    from chitu.moe.experts import fused_experts


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
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
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
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
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
        output_dtype = x.dtype
        quantized_x, dynamic_scale = torch_npu.npu_dynamic_quant(
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


@QuantizationRegistry.register_moe_experts("ascend_w8a8_dynamic")
class AscendW8A8DynamicMoeExperts(
    enable_native_layout_weight("gate_up_proj_weight", NpuFractalZnTensor),
    enable_native_layout_weight("down_proj_weight", NpuFractalZnTensor),
    enable_native_layout_weight("gate_up_proj_weight_scale", SqueezeLastSingleton),
    enable_native_layout_weight("down_proj_weight_scale", SqueezeLastSingleton),
    QuantizedMoeExpertsBase,
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
        n_routed_experts: int,
        n_shared_experts: int,
        n_activated_experts: int,
        fuse_shared_experts: bool,
        checkpoint_prefix: str,
        merge_gate_up: bool,
        layer_id: int,
    ):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__(
            dim,
            moe_inter_dim,
            n_routed_experts,
            n_shared_experts,
            n_activated_experts,
            fuse_shared_experts,
            checkpoint_prefix,
            merge_gate_up,
            layer_id,
        )

        gate_up_proj_in_features = dim

        if self.merge_gate_up:
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
        else:
            raise NotImplementedError("Ascend MoE must use merged gate up")

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
    def forward(
        self,
        routed_x: BatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "npu",
    ) -> torch.Tensor:
        if self.merge_gate_up:
            return fused_experts(
                routed_x,
                w1=self.gate_up_proj_weight,
                w1_scale=self.gate_up_proj_weight_scale,  # fp32
                w2=self.down_proj_weight,
                w2_scale=self.down_proj_weight_scale,  # bf16
                topk_weights=weights,
                use_int8_w8a8=True,
                impl=impl,
            )

        else:
            return super().forward(routed_x, weights, inplace=inplace, impl=impl)
