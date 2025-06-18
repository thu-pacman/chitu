from typing import Optional, Tuple

import torch

from chitu.quantization.registry import QuantizedLinearBase, QuantizationRegistry
from chitu.ops import (
    soft_fp4_raise_to_fp8_gemm_deepseek_v3,
    soft_fp4_raise_to_bf16_gemm_deepseek_v3,
    act_quant_deepseek_v3,
)
from chitu.device_type import get_device_name, is_muxi, is_nvidia
from chitu.utils import ceil_div
from chitu.global_vars import get_global_args


def linear_block_fp4(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    act_block_size: int,
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
        weight_scale (torch.Tensor): The first-level scale tensor.
        weight_scale_2 (torch.Tensor): The second-level scale tensor.
        act_block_size (int): The block size for activation quantization.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve
        quantization-aware computations depending on the input parameters.
    """

    assert weight.element_size() == 1

    if get_global_args().infer.raise_lower_bit_float_to == "bfloat16":
        if is_nvidia() or is_muxi():
            y = soft_fp4_raise_to_bf16_gemm_deepseek_v3(
                x, weight, weight_scale, weight_scale_2
            )
            if bias is not None:
                y += bias
            return y
        else:
            raise NotImplementedError(
                f"Soft-fp8 fused gemm not implemented for {get_device_name()}"
            )
            # FIXME: Use a dequant-then-compute approach
    else:
        x_dtype = x.dtype
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        x, act_scale = act_quant_deepseek_v3(x, act_block_size)
        assert weight_scale is not None
        y = soft_fp4_raise_to_fp8_gemm_deepseek_v3(
            x,
            act_scale,
            weight,
            weight_scale,
            weight_scale_2,
            act_block_size=act_block_size,
        )
        if bias is not None:
            y += bias
        return y.view(x_shape[:-1] + y.shape[-1:]).to(x_dtype)


@QuantizationRegistry.register_method("blockfp4")
class Blockfp4Linear(QuantizedLinearBase):
    """
    block 4-bit weight and activation quantized linear layer.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        has_bias: If set to True, the layer will have a bias.
        dtype: The desired data type of the parameters.
        bias_dtype: The desired data type of the bias.
        block_shape: The block shape (in, out) of first-level scaling. Defaults to
            (16, 1).
        block_shape_2: The block shape (in, out) of second-level scaling. Defaults
            to the same shape as the full weight tensor.
        act_block_size: The block size for activation quantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = False,
        dtype=torch.uint8,
        bias_dtype=None,
        block_shape: Tuple[int, int] = (16, 1),
        block_shape_2: Optional[Tuple[int, int]] = None,
        act_block_size: int = 128,
    ):
        super().__init__()

        dtype = dtype or torch.uint8
        if block_shape_2 is None:
            block_shape_2 = (in_features, out_features)

        self.in_features = in_features
        self.out_features = out_features
        self.act_block_size = act_block_size

        self.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty(
                    (
                        out_features,
                        in_features // 2,  # Every 2 float4 is packed into 1 uint8
                    ),
                    dtype=dtype,
                ),
                requires_grad=False,
            ),
        )

        block_in, block_out = block_shape
        self.register_parameter(
            "weight_scale",
            torch.nn.Parameter(
                torch.empty(
                    ceil_div(out_features, block_out),
                    ceil_div(in_features, block_in),
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            ),
        )

        block_2_in, block_2_out = block_shape_2
        assert out_features % block_2_out == 0, f"{out_features=}, {block_2_out=}"
        assert in_features % block_2_in == 0, f"{in_features=}, {block_2_in=}"
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

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        return linear_block_fp4(
            x,
            self.weight,
            self.weight_scale,
            self.weight_scale_2,
            act_block_size=self.act_block_size,
            bias=self.bias,
        )
