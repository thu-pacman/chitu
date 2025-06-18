from typing import Optional
from logging import getLogger

import torch

from chitu.quantization.registry import QuantizedLinearBase, QuantizationRegistry
from chitu.ops import (
    fp8_gemm_deepseek_v3,
    soft_fp8_gemm_deepseek_v3,
    weight_dequant_soft_fp8_deepseek_v3,
    act_quant_deepseek_v3,
)
from chitu.device_type import get_device_name, is_muxi, is_nvidia
from chitu.utils import parse_dtype
from chitu.global_vars import get_global_args


logger = getLogger(__name__)


def linear_block_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    block_size: Optional[int] = 128,
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
    """

    assert weight.element_size() == 1

    if get_global_args().infer.raise_lower_bit_float_to == "bfloat16":
        if is_nvidia() or is_muxi():
            y = soft_fp8_gemm_deepseek_v3(x, weight, weight_scale)
            if bias is not None:
                y += bias
            return y
        else:
            logger.warning(
                f"Soft-fp8 fused gemm not implemented for {get_device_name()}, falling back to soft-fp8 conversion"
            )
            weight_dequanted = weight_dequant_soft_fp8_deepseek_v3(
                weight, weight_scale, block_size
            )
            return torch.nn.functional.linear(x, weight_dequanted, bias)
    else:
        x_dtype = x.dtype
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        x, act_scale = act_quant_deepseek_v3(x, block_size)
        assert weight_scale is not None
        y = fp8_gemm_deepseek_v3(x, act_scale, weight, weight_scale)
        if bias is not None:
            y += bias
        return y.view(x_shape[:-1] + y.shape[-1:]).to(x_dtype)


@QuantizationRegistry.register_method("gguf-blockfp8")
@QuantizationRegistry.register_method("blockfp8")
class Blockfp8Linear(QuantizedLinearBase):
    """
    block 8-bit weight and activation quantized linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = False,
        dtype=torch.float8_e4m3fn,
        bias_dtype=None,
        block_size=128,
        **kwarg,
    ):
        super().__init__()

        dtype = dtype or torch.float8_e4m3fn

        # Some platforms do not support float8, but we can run them with `infer.raise_lower_bit_float_to=bfloat16`.
        # However, we need to treat float8 items as uint8 first, to avoid the missing ops on these platforms.
        args = get_global_args()
        if parse_dtype(args.infer.raise_lower_bit_float_to).itemsize > 1:
            dtype = torch.uint8

        assert dtype.itemsize == 1

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        self.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty((out_features, in_features), dtype=dtype),
                requires_grad=False,
            ),
        )

        scale_out_features = (out_features + block_size - 1) // block_size
        scale_in_features = (in_features + block_size - 1) // block_size
        self.register_parameter(
            "scale",
            torch.nn.Parameter(
                torch.empty(
                    scale_out_features,
                    scale_in_features,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            ),
        )

        if has_bias:
            self.register_parameter(
                "bias",
                torch.nn.Parameter(
                    torch.empty(out_features, dtype=bias_dtype), requires_grad=False
                ),
            )
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        return linear_block_fp8(
            x, self.weight, self.scale, self.bias, block_size=self.block_size
        )
