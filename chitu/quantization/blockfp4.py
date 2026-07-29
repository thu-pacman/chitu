# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
from logging import getLogger

from chitu.lazy import eval_lazy
import torch

from chitu.quantization.base import (
    QuantizedLinearBase,
    QuantizedMoeExpertsUnmerged,
    QuantizedMoeExpertsMerged,
)
from chitu.quantization.registry import QuantizationRegistry
from chitu.ops.quant import (
    soft_fp4_raise_to_fp8_blockfp4_gemm,
    soft_fp4_raise_to_bf16_blockfp4_gemm,
    soft_fp4_raise_to_bf16_blockfp4_single_scale_gemm,
    blockfp8_act_quant,
    blockfp4_gemm,
)
from chitu.device_type import get_device_name, is_muxi, is_nvidia, is_blackwell
from chitu.utils import (
    ceil_div,
    try_import_opt_dep,
    try_import_platform_dep,
    try_import_and_setup_torch_npu,
    parse_dtype,
)
from chitu.global_vars import get_global_args
from chitu.native_layout import (
    NativeLayoutMixin,
    BlackwellMXFP4MOEPadWeight,
    BlackwellMXFP4MOEScalePadToSwizzled,
    Blockfp4LinearPackedWeightPadToShape,
    Blockfp4LinearScalePadToSwizzled,
    Packed4BitWeightAlongK,
    Packed4BitWeightAlongKContig,
    Packed4BitWeightNPUNative,
    LinearScaleToSwizzled,
    nvfp4_moe_down_proj_n_padded,
)
from chitu.moe.batched_expert_result import BatchedExpertResult
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
)
from chitu.ops.utils import make_op_dispatcher

hard_fp4_kernels, has_hard_fp4_kernels = try_import_opt_dep(
    "hard_fp4_kernels", "hard_fp4_kernels"
)
triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
if has_torch_npu:
    from chitu.npu_utils import fused_experts_no_sum_npu
if has_triton:
    from chitu.moe.experts import fused_experts_soft_fp4


logger = getLogger(__name__)


@make_op_dispatcher
def linear_block_fp4(
    x: torch.Tensor,
    weight: Packed4BitWeightAlongK,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    act_block_size: int,
    bias: Optional[torch.Tensor] = None,
    *,
    x_scale: Optional[torch.Tensor] = None,
    impl: str = "auto",
) -> torch.Tensor:
    """
    Quantized linear with blockfp4 quantization.

    Args:
        x (torch.Tensor): The input tensor.
        weight (Packed4BitWeightAlongK): The weight tensor.
        weight_scale (torch.Tensor): The first-level scale tensor.
        weight_scale_2 (torch.Tensor): The second-level scale tensor.
        act_block_size (int): The block size for activation quantization.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.
        x_scale: The scale of the input tensor, if the input tensor is quantized.
        impl: The implementation of linear transformation. "blackwell" means the blackwell nvfp4 implementation. "fp8" and "bf16" means raise the input tensor to fp8 and bf16. Default is auto.

    Returns:
        torch.Tensor: The result of the linear transformation.
    """
    raise NotImplementedError


@linear_block_fp4.register_auto
def _auto_linear_block_fp4(
    x: torch.Tensor,
    weight: Packed4BitWeightAlongK,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    act_block_size: int,
    bias: Optional[torch.Tensor] = None,
    *,
    x_scale: Optional[torch.Tensor] = None,
):
    if is_blackwell():
        if weight.k_stride == 1:
            return "blackwell"
        return "fp8"
    if get_global_args().infer.raise_lower_bit_float_to == "bfloat16":
        if is_nvidia() or is_muxi():
            return "bf16"
        return "fp8"
    return "fp8"


@linear_block_fp4.register("blackwell", available=is_blackwell())
def _linear_block_fp4_blackwell(
    x: torch.Tensor,
    weight: Packed4BitWeightAlongK,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    act_block_size: int,
    bias: Optional[torch.Tensor] = None,
    *,
    x_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Note: blackwell impl need swizzled weights, while others weights are linear

    # FIXME: Add fp4 option to infer.raise_lower_bit_float_to and use it here
    if x.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(f"Unsupported input type: {x.dtype}")
    if x_scale is not None:
        raise ValueError(f"No x_scale is supported for {x.dtype=}")
    k_w = weight.layout_tensor.shape[-1] * 2
    k_x = x.shape[-1]
    if k_x < k_w:
        x = eval_lazy(x)
        x = torch.nn.functional.pad(x, (0, k_w - k_x), value=0).contiguous()
    elif k_x > k_w:
        raise AssertionError(
            f"activation K {k_x} exceeds weight K {k_w} (packed last dim "
            f"{weight.layout_tensor.shape[-1]})"
        )
    assert (
        x.shape[-1] == weight.layout_tensor.shape[-1] * 2
    ), f"{x.shape=}, {weight.layout_tensor.shape=}"
    y = blockfp4_gemm(
        x,
        weight.layout_tensor,
        weight_scale,
        weight_scale_2,
        alpha=None,
        out_dtype=parse_dtype(get_global_args().infer.raise_lower_bit_float_to),
    )
    if bias is not None:
        y += bias
    return y


@linear_block_fp4.register("fp8")
def _linear_block_fp4_fp8(
    x: torch.Tensor,
    weight: Packed4BitWeightAlongK,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    act_block_size: int,
    bias: Optional[torch.Tensor] = None,
    *,
    x_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x_shape = x.shape
    if x.dtype in {torch.float16, torch.bfloat16}:
        x = x.view(-1, x_shape[-1])
        x, x_scale = blockfp8_act_quant(
            x, scale_block_shape=[act_block_size, act_block_size]
        )
    elif x.dtype == torch.float8_e4m3fn:
        assert x_scale is not None
        x = x.view(-1, x_shape[-1])
        x_scale = x_scale.view(-1, x_scale.shape[-1])
    else:
        raise ValueError(f"Unsupported input type: {x.dtype}")
    assert weight_scale is not None
    y = soft_fp4_raise_to_fp8_blockfp4_gemm(
        x,
        x_scale,
        weight,
        weight_scale,
        weight_scale_2,
        act_block_size=act_block_size,
    )
    if bias is not None:
        y = (y + bias).to(y.dtype)
    return y.view(x_shape[:-1] + y.shape[-1:])


class Blockfp4LinearBase(NativeLayoutMixin, QuantizedLinearBase):
    """
    block 4-bit weight and activation quantized linear layer.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        has_bias: If set to True, the layer will have a bias.
        bias_dtype: The desired data type of the bias.
        block_shape: The block shape (in, out) of first-level scaling. Defaults to
            (16, 1).
        block_shape_2: The block shape (in, out) of second-level scaling. Defaults
            to the same shape as the full weight tensor.
        act_block_size: The block size for activation quantization.
    """

    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        in_features: int,
        out_features: int,
        has_bias: bool = False,
        *,
        ############################################
        # Parameters specific to this quantization
        bias_dtype=None,
        block_shape: tuple[int, int] = (16, 1),
        block_shape_2: Optional[tuple[int, int]] = None,
        act_block_size: int = 128,
        no_input_scale: bool = False,
    ):
        super().__init__(in_features, out_features, has_bias)

        if block_shape_2 is None:
            block_shape_2 = (in_features, out_features)

        self.act_block_size = act_block_size

        self.weight = torch.nn.Parameter(
            torch.empty(
                (
                    out_features,
                    in_features,
                ),
            ),
            requires_grad=False,
        )

        block_in, block_out = block_shape
        self.block_shape = block_shape

        from chitu.models.registry import ModelType

        if (
            get_global_args().models.type == ModelType.HF_LLAMA
            and get_global_args().infer.npu_fusion_fp4
        ):
            dtype = torch.bfloat16
        else:
            dtype = torch.uint8

        self.register_parameter(
            "weight_scale",
            torch.nn.Parameter(
                torch.empty(
                    ceil_div(out_features, block_out),
                    ceil_div(in_features, block_in),
                    dtype=dtype,
                ),
                requires_grad=False,
            ),
        )

        block_2_in, block_2_out = block_shape_2
        assert out_features % block_2_out == 0, f"{out_features=}, {block_2_out=}"
        assert in_features % block_2_in == 0, f"{in_features=}, {block_2_in=}"
        if not no_input_scale:
            self.register_parameter(
                "input_scale",
                torch.nn.Parameter(
                    torch.empty(
                        out_features // block_2_out,
                        in_features // block_2_in,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                ),
            )
        self.register_parameter(
            "weight_scale_2",
            torch.nn.Parameter(
                torch.empty(
                    out_features // block_2_out,
                    in_features // block_2_in,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            ),
        )

        if has_bias:
            self.register_parameter(
                "bias",
                torch.nn.Parameter(
                    torch.empty(out_features, dtype=bias_dtype),
                    requires_grad=False,
                ),
            )
        else:
            self.register_parameter("bias", None)

    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(
            self.weight, Packed4BitWeightAlongKContig, state_dict_convert=False
        )


@QuantizationRegistry.register_linear("blockfp4", when=lambda _: is_nvidia())
@QuantizationRegistry.register_linear("blockfp4_merged", when=lambda _: is_nvidia())
class Blockfp4LinearPackKStride64(Blockfp4LinearBase):
    """
    Blockfp4Linear with weight in Packed4BitWeightAlongK (k_stride=64) layout.
    """

    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(self.weight, Packed4BitWeightAlongK, k_stride=64)

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        return linear_block_fp4(
            x,
            self.get_native_layout_weight(),
            self.weight_scale,
            self.weight_scale_2,
            act_block_size=self.act_block_size,
            bias=self.bias,
        )


@QuantizationRegistry.register_linear(
    "blockfp4", when=lambda _: is_blackwell(), priority=1
)
@QuantizationRegistry.register_linear(
    "blockfp4_merged", when=lambda _: is_blackwell(), priority=1
)
class Blockfp4LinearPackKStride1(Blockfp4LinearBase):
    """
    Blockfp4Linear with weight in Packed4BitWeightAlongK (k_stride=1) layout.
    """

    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(
            self.weight,
            Blockfp4LinearPackedWeightPadToShape,
            padded_shape=[
                self.weight.shape[0],
                ((self.weight.shape[1] * 2 + 255) // 256 * 256) // 2,
            ],
        )
        self.apply_native_layout(
            self.weight_scale,
            Blockfp4LinearScalePadToSwizzled,
            padded_shape=[
                self.weight_scale.shape[0],
                max(
                    (self.weight_scale.shape[1] + 7) // 8 * 8,
                    ceil_div(
                        ((self.weight.shape[1] * 2 + 255) // 256 * 256),
                        self.block_shape[0],
                    ),
                ),
            ],
        )

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        return linear_block_fp4(
            x,
            self.get_native_layout_weight(),
            self.weight_scale,
            self.weight_scale_2,
            act_block_size=self.act_block_size,
            bias=self.bias,
        )


@QuantizationRegistry.register_linear(
    "blockfp4", when=lambda _: has_torch_npu, priority=2
)
class Blockfp4LinearPackNPUNative(Blockfp4LinearBase):
    """
    Blockfp4Linear with weight in Packed4BitWeightNPUNative layout.
    """

    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(self.weight, Packed4BitWeightNPUNative)

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        y = soft_fp4_raise_to_bf16_blockfp4_single_scale_gemm(
            x, self.get_native_layout_weight(), self.weight_scale
        )
        if self.bias is not None:
            y += self.bias
        return y


def _finalize_fused_experts_sum_output(
    output, hidden_states, topk_weights: Optional[torch.Tensor], inplace: bool
):
    if hasattr(output, "weighted_sum"):
        if (
            inplace
            and isinstance(hidden_states, IndexedBatchedRoutedActivation)
            and hidden_states.activation.dtype == torch.get_default_dtype()
        ):
            out = hidden_states.activation
        else:
            out = None
        return output.weighted_sum(topk_weights, out=out)
    return output


@make_op_dispatcher
def fused_experts_no_sum_blockfp4_indexed(
    hidden_states,
    w1,
    w2,
    *,
    impl: str = "auto",
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_scale_2: Optional[torch.Tensor] = None,
    w2_scale_2: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    round_scale_to_pow2: bool = False,
    swiglu_limit: Optional[float] = None,
    experts_start_idx: int = 0,
): ...


@fused_experts_no_sum_blockfp4_indexed.register_auto
def _auto_fused_experts_no_sum_blockfp4_indexed():
    if has_triton:
        return "triton"
    raise NotImplementedError


@fused_experts_no_sum_blockfp4_indexed.register("triton", available=has_triton)
def _fused_experts_no_sum_blockfp4_indexed_triton(
    hidden_states,
    w1,
    w2,
    *,
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_scale_2: Optional[torch.Tensor] = None,
    w2_scale_2: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    round_scale_to_pow2: bool = False,
    swiglu_limit: Optional[float] = None,
    experts_start_idx: int = 0,
):
    return fused_experts_soft_fp4(
        hidden_states,
        w1=w1,
        w2=w2,
        activation=activation,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_scale2=w1_scale_2,
        w2_scale2=w2_scale_2,
        block_shape=block_shape,
        soft_fp8=soft_fp8,
        round_scale_to_pow2=round_scale_to_pow2,
        swiglu_limit=swiglu_limit,
        experts_start_idx=experts_start_idx,
    )


@make_op_dispatcher
def fused_experts_sum_blockfp4_indexed(
    hidden_states,
    w1,
    w2,
    topk_weights: Optional[torch.Tensor],
    *,
    inplace: bool = False,
    impl: str = "auto",
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_scale_2: Optional[torch.Tensor] = None,
    w2_scale_2: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    round_scale_to_pow2: bool = False,
    swiglu_limit: Optional[float] = None,
    experts_start_idx: int = 0,
): ...


@fused_experts_sum_blockfp4_indexed.register_auto
def _auto_fused_experts_sum_blockfp4_indexed():
    if has_triton:
        return "triton"
    raise NotImplementedError


@fused_experts_sum_blockfp4_indexed.register("triton", available=has_triton)
def _fused_experts_sum_blockfp4_indexed_triton(
    hidden_states,
    w1,
    w2,
    topk_weights: Optional[torch.Tensor],
    *,
    inplace: bool = False,
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_scale_2: Optional[torch.Tensor] = None,
    w2_scale_2: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    round_scale_to_pow2: bool = False,
    swiglu_limit: Optional[float] = None,
    experts_start_idx: int = 0,
    impl: str,
):
    output = fused_experts_no_sum_blockfp4_indexed(
        hidden_states,
        w1,
        w2,
        impl=impl,
        activation=activation,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_scale_2=w1_scale_2,
        w2_scale_2=w2_scale_2,
        block_shape=block_shape,
        soft_fp8=soft_fp8,
        round_scale_to_pow2=round_scale_to_pow2,
        swiglu_limit=swiglu_limit,
        experts_start_idx=experts_start_idx,
    )
    return _finalize_fused_experts_sum_output(
        output, hidden_states, topk_weights=topk_weights, inplace=inplace
    )


class Blockfp4MoeExpertsUnmergedBase(NativeLayoutMixin, QuantizedMoeExpertsUnmerged):
    """
    blockfp4 quantized MoeExperts with weights in Packed4BitWeightAlongK (k_stride=1) layout,
    and unmerged gate and up projection.
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
        no_input_scale: bool = False,
        merged_global_scale: bool = False,
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

        quant_scale_stride = 16

        scale_in_features = ceil_div(dim, quant_scale_stride)
        self.gate_proj_weight = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                moe_inter_dim,
                dim,
            ),
            requires_grad=False,
        )
        self.gate_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                moe_inter_dim,
                scale_in_features,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.gate_proj_weight_scale_2 = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                1,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        if not no_input_scale:
            self.gate_proj_input_scale = torch.nn.Parameter(
                torch.empty(
                    self.group_size,
                    1,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        self.up_proj_weight = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                moe_inter_dim,
                dim,
            ),
            requires_grad=False,
        )
        self.up_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                moe_inter_dim,
                scale_in_features,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.up_proj_weight_scale_2 = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                1,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        if not no_input_scale:
            self.up_proj_input_scale = torch.nn.Parameter(
                torch.empty(
                    self.group_size,
                    1,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        down_proj_scale_in_features = ceil_div(moe_inter_dim, quant_scale_stride)
        down_proj_scale_out_features = dim
        self.down_proj_weight = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                dim,
                moe_inter_dim,
            ),
            requires_grad=False,
        )
        self.down_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                down_proj_scale_out_features,
                down_proj_scale_in_features,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.down_proj_weight_scale_2 = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                1,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        if not no_input_scale:
            self.down_proj_input_scale = torch.nn.Parameter(
                torch.empty(
                    self.group_size,
                    1,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )

    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(
            self.gate_proj_weight,
            Packed4BitWeightAlongKContig,
            state_dict_convert=False,
        )
        self.apply_native_layout(
            self.up_proj_weight, Packed4BitWeightAlongKContig, state_dict_convert=False
        )
        self.apply_native_layout(
            self.down_proj_weight,
            Packed4BitWeightAlongKContig,
            state_dict_convert=False,
        )

    @override
    def forward_ith_expert_gate(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return linear_block_fp4(
            x,
            self.get_native_layout_gate_proj_weight()[i],
            self.gate_proj_weight_scale[i],
            self.gate_proj_weight_scale_2[i],
            128,
            None,
            x_scale=x_scale,
        )

    @override
    def forward_ith_expert_up(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return linear_block_fp4(
            x,
            self.get_native_layout_up_proj_weight()[i],
            self.up_proj_weight_scale[i],
            self.up_proj_weight_scale_2[i],
            128,
            None,
            x_scale=x_scale,
        )

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear_block_fp4(
            x,
            self.get_native_layout_down_proj_weight()[i],
            self.down_proj_weight_scale[i],
            self.down_proj_weight_scale_2[i],
            128,
            None,
        )


class Blockfp4MoeExpertsMergedBase(NativeLayoutMixin, QuantizedMoeExpertsMerged):
    """
    blockfp4 quantized MoeExperts with weights in Packed4BitWeightAlongK (k_stride=1) layout,
    and merged gate and up projection.
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
        no_input_scale: bool = False,
        merged_global_scale: bool = False,
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

        quant_scale_stride = 16

        scale_in_features = ceil_div(dim, quant_scale_stride)
        self.gate_up_proj_weight = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                moe_inter_dim * 2,
                dim,
            ),
            requires_grad=False,
        )
        self.gate_up_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                moe_inter_dim * 2,
                scale_in_features,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.gate_up_proj_weight_scale_2 = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                1 if merged_global_scale else 2,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.gate_up_proj_input_scale = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                1 if merged_global_scale else 2,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        down_proj_scale_in_features = ceil_div(moe_inter_dim, quant_scale_stride)
        down_proj_scale_out_features = dim
        self.down_proj_weight = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                dim,
                moe_inter_dim,
            ),
            requires_grad=False,
        )
        self.down_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                down_proj_scale_out_features,
                down_proj_scale_in_features,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.down_proj_weight_scale_2 = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                1,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        if not no_input_scale:
            self.down_proj_input_scale = torch.nn.Parameter(
                torch.empty(
                    self.group_size,
                    1,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )

    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(
            self.gate_up_proj_weight,
            Packed4BitWeightAlongKContig,
            state_dict_convert=False,
        )
        self.apply_native_layout(
            self.down_proj_weight,
            Packed4BitWeightAlongKContig,
            state_dict_convert=False,
        )

    @override
    def forward_ith_expert_gate_up(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return linear_block_fp4(
            x,
            self.get_native_layout_gate_up_proj_weight()[i],
            self.gate_up_proj_weight_scale[i],
            self.gate_up_proj_weight_scale_2[i],
            128,
            None,
            x_scale=x_scale,
        )

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear_block_fp4(
            x,
            self.get_native_layout_down_proj_weight()[i],
            self.down_proj_weight_scale[i],
            self.down_proj_weight_scale_2[i],
            128,
            None,
        )


@QuantizationRegistry.register_moe_experts(
    "blockfp4",
    merge_gate_up=True,
    when=lambda _: is_blackwell() and has_hard_fp4_kernels,
    priority=1,
)
@QuantizationRegistry.register_moe_experts(
    "blockfp4_merged",
    merge_gate_up=True,
    when=lambda _: is_blackwell() and has_hard_fp4_kernels,
    priority=1,
)
class Blockfp4MoeExpertsBlackwell(Blockfp4MoeExpertsMergedBase):

    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(
            self.gate_up_proj_weight,
            BlackwellMXFP4MOEPadWeight,
            padded_shape=[
                self.gate_up_proj_weight.shape[0],
                (self.gate_up_proj_weight.shape[1] // 2 + 255) // 256 * 256 * 2,
                ((self.gate_up_proj_weight.shape[2] * 2 + 255) // 256 * 256) // 2,
            ],
        )
        self.apply_native_layout(
            self.down_proj_weight,
            BlackwellMXFP4MOEPadWeight,
            padded_shape=[
                self.down_proj_weight.shape[0],
                nvfp4_moe_down_proj_n_padded(
                    self.down_proj_weight.shape[1], self.down_proj_weight.shape[2]
                ),
                ((self.down_proj_weight.shape[2] * 2 + 255) // 256 * 256) // 2,
            ],
        )
        self.apply_native_layout(
            self.gate_up_proj_weight_scale,
            BlackwellMXFP4MOEScalePadToSwizzled,
            padded_shape=[
                self.gate_up_proj_weight_scale.shape[0],
                (self.gate_up_proj_weight_scale.shape[1] // 2 + 255) // 256 * 256 * 2,
                (self.gate_up_proj_weight_scale.shape[2] + 7) // 8 * 8,
            ],
        )
        self.apply_native_layout(
            self.down_proj_weight_scale,
            BlackwellMXFP4MOEScalePadToSwizzled,
            padded_shape=[
                self.down_proj_weight_scale.shape[0],
                nvfp4_moe_down_proj_n_padded(
                    self.down_proj_weight.shape[1], self.down_proj_weight.shape[2]
                ),
                ((self.down_proj_weight.shape[2] * 2 + 255) // 256 * 256) // 16,
            ],
        )

    @override
    def forward(
        self,
        routed_x: BatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        if self.swiglu_limit is not None:
            raise NotImplementedError(
                "swiglu_limit is not implemented for Chitu's Blackwell blockfp4 fused MoE wrapper"
            )
        x, indices = routed_x.activation, routed_x.token_to_expert_indices
        shape = x.size()
        x = eval_lazy(x)
        x = x.view(-1, self.dim)
        w1 = self.get_native_layout_gate_up_proj_weight().layout_tensor
        k1 = w1.shape[2] * 2
        if x.shape[1] < k1:
            x = torch.nn.functional.pad(x, (0, k1 - x.shape[1]), value=0).contiguous()
        elif x.shape[1] > k1:
            raise AssertionError(
                f"MoE activation K {x.shape[1]} exceeds padded gate_up K {k1}"
            )

        bs = x.size(0)
        use_decode_path = bs <= 128
        backend = (
            hard_fp4_kernels.fused_moe_decode.scaled_fp4_fused_moe_decode
            if use_decode_path
            else hard_fp4_kernels.cuda_nvfp4_fused_moe
        )
        w2 = self.get_native_layout_down_proj_weight().layout_tensor
        n2_pad = w2.size(1)
        # Fused decode and Cutlass paths both fill ``output[:, :N2]``. Inplace uses ``X`` as
        # ``output``; that buffer is only ``K1`` wide after ``x`` pad. If ``N2 > K1``, inplace
        # overflows (often surfaces as illegal access at the next NCCL sync).
        fused_out_rows = x.size(0)
        fused_safe_inplace = n2_pad <= k1
        y = x
        if not (inplace and fused_safe_inplace):
            y = torch.empty(
                (fused_out_rows, n2_pad),
                dtype=x.dtype,
                device=x.device,
            )
        backend(
            y,
            x,
            w1,
            self.gate_up_proj_weight_scale,
            self.gate_up_proj_weight_scale_2,
            w2,
            self.down_proj_weight_scale,
            self.down_proj_weight_scale_2,
            weights,
            indices,
        )

        if n2_pad != self.dim:
            y = y[..., : self.dim]
        return y.reshape(shape)


@QuantizationRegistry.register_moe_experts(
    "blockfp4", merge_gate_up=True, when=lambda _: is_nvidia() and has_triton
)
@QuantizationRegistry.register_moe_experts(
    "blockfp4_merged", merge_gate_up=True, when=lambda _: is_nvidia() and has_triton
)
class Blockfp4MoeExpertsPackKStride64(Blockfp4MoeExpertsMergedBase):
    """
    blockfp4 quantized MoeExperts with weights in Packed4BitWeightAlongK (k_stride=64) layout.
    """

    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(
            self.gate_up_proj_weight, Packed4BitWeightAlongK, k_stride=64
        )
        self.apply_native_layout(
            self.down_proj_weight, Packed4BitWeightAlongK, k_stride=64
        )

    @override
    def forward_no_sum(
        self, routed_x: BatchedRoutedActivation, impl: str = "auto"
    ) -> BatchedExpertResult:
        assert isinstance(routed_x, IndexedBatchedRoutedActivation)
        raise_to_16 = (
            parse_dtype(get_global_args().infer.raise_lower_bit_float_to).itemsize != 1
        )
        return fused_experts_no_sum_blockfp4_indexed(
            routed_x,
            w1=self.get_native_layout_gate_up_proj_weight().layout_tensor,
            w2=self.get_native_layout_down_proj_weight().layout_tensor,
            w1_scale=self.gate_up_proj_weight_scale,
            w2_scale=self.down_proj_weight_scale,
            w1_scale_2=self.gate_up_proj_weight_scale_2,
            w2_scale_2=self.down_proj_weight_scale_2,
            block_shape=[128, 128],
            soft_fp8=raise_to_16,
            swiglu_limit=self.swiglu_limit,
            experts_start_idx=self.experts_start_idx,
            impl=impl,
        )

    @override
    def forward(
        self,
        routed_x: BatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        assert isinstance(routed_x, IndexedBatchedRoutedActivation)
        raise_to_16 = (
            parse_dtype(get_global_args().infer.raise_lower_bit_float_to).itemsize != 1
        )
        return fused_experts_sum_blockfp4_indexed(
            routed_x,
            w1=self.get_native_layout_gate_up_proj_weight().layout_tensor,
            w2=self.get_native_layout_down_proj_weight().layout_tensor,
            topk_weights=weights,
            inplace=inplace,
            w1_scale=self.gate_up_proj_weight_scale,
            w2_scale=self.down_proj_weight_scale,
            w1_scale_2=self.gate_up_proj_weight_scale_2,
            w2_scale_2=self.down_proj_weight_scale_2,
            block_shape=[128, 128],
            soft_fp8=raise_to_16,
            swiglu_limit=self.swiglu_limit,
            experts_start_idx=self.experts_start_idx,
            impl=impl,
        )


@QuantizationRegistry.register_moe_experts(
    "blockfp4",
    merge_gate_up=False,
    when=lambda _: has_torch_npu,
    priority=2,
)
class Blockfp4MoeExpertsUnmergedPackNPUNative(Blockfp4MoeExpertsMergedBase):
    """
    blockfp4 quantized MoeExperts with weights in Packed4BitWeightNPUNative layout,
    with unmerged gate and up projection.
    """

    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(self.gate_proj_weight, Packed4BitWeightNPUNative)
        self.apply_native_layout(self.up_proj_weight, Packed4BitWeightNPUNative)
        self.apply_native_layout(self.down_proj_weight, Packed4BitWeightNPUNative)

    @override
    def forward_ith_expert_gate(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert x_scale is None
        return soft_fp4_raise_to_bf16_blockfp4_single_scale_gemm(
            x,
            self.get_native_layout_gate_proj_weight()[i],
            self.gate_proj_weight_scale[i],
        )

    @override
    def forward_ith_expert_up(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert x_scale is None
        return soft_fp4_raise_to_bf16_blockfp4_single_scale_gemm(
            x,
            self.get_native_layout_up_proj_weight()[i],
            self.up_proj_weight_scale[i],
        )

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return soft_fp4_raise_to_bf16_blockfp4_single_scale_gemm(
            x,
            self.get_native_layout_down_proj_weight()[i],
            self.down_proj_weight_scale[i],
        )


@QuantizationRegistry.register_moe_experts(
    "blockfp4",
    merge_gate_up=True,
    when=lambda _: has_torch_npu,
    priority=2,
)
class Blockfp4MoeExpertsMergedPackNPUNative(Blockfp4MoeExpertsMergedBase):
    """
    blockfp4 quantized MoeExperts with weights in Packed4BitWeightNPUNative layout,
    with merged gate and up projection.
    """

    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(self.gate_up_proj_weight, Packed4BitWeightNPUNative)
        self.apply_native_layout(self.down_proj_weight, Packed4BitWeightNPUNative)

    @override
    def forward_no_sum(
        self, routed_x: BatchedRoutedActivation, impl="auto"
    ) -> BatchedExpertResult:
        return fused_experts_no_sum_npu(
            routed_x,
            w1=self.gate_up_proj_weight,
            w2=self.down_proj_weight,
            w1_scale=self.gate_up_proj_weight_scale,
            w2_scale=self.down_proj_weight_scale,
            global_num_experts=self.global_n_experts,
            swiglu_limit=self.swiglu_limit,
        )

    @override
    def forward_ith_expert_gate_up(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert x_scale is None
        return soft_fp4_raise_to_bf16_blockfp4_single_scale_gemm(
            x,
            self.get_native_layout_gate_up_proj_weight()[i],
            self.gate_up_proj_weight_scale[i],
        )

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return soft_fp4_raise_to_bf16_blockfp4_single_scale_gemm(
            x,
            self.get_native_layout_down_proj_weight()[i],
            self.down_proj_weight_scale[i],
        )
