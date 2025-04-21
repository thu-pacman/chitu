import torch
import logging
from typing import Dict, Tuple, Optional, Type

from chitu.utils import try_import_opt_dep
from chitu.ops import (
    fp8_gemm_deepseek_v3,
    soft_fp8_gemm_deepseek_v3,
    soft_fp4_raise_to_fp8_gemm_deepseek_v3,
    soft_fp4_raise_to_bf16_gemm_deepseek_v3,
    weight_dequant_soft_fp8_deepseek_v3,
    act_quant_deepseek_v3,
)
from chitu.global_vars import get_global_args
from chitu.device_type import get_device_name, is_muxi, is_nvidia, has_native_fp8


logger = logging.getLogger(__name__)


class QuantizedLinearBase(torch.nn.Module):
    """
    Base class for all quantized linear layers.

    Defines the interface that all quantized linear implementations must follow.
    """

    pass


class LLMInt8Linear(QuantizedLinearBase):
    """
    8-bit linear layer implementation using bitsandbytes.
    """

    def __init__(
        self, in_features: int, out_features: int, has_bias: bool = True, **kwargs
    ) -> torch.nn.Module:

        super().__init__()

        import bitsandbytes as bnb

        bnb_module = bnb.nn.Linear8bitLt(
            in_features,
            out_features,
            bias=has_bias,
            has_fp16_weights=kwargs.get("has_fp16_weights", False),
            threshold=kwargs.get("threshold", 6.0),
        )
        for name, buffer in bnb_module.named_buffers():
            self.register_buffer(name, buffer)
        for name, param in bnb_module.named_parameters():
            self.register_parameter(name, param)

        self.state = bnb_module.state
        self.init_8bit_state = bnb_module.init_8bit_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import bitsandbytes as bnb

        self.state.is_training = False
        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

        if not self.state.has_fp16_weights and self.state.CB is not None:
            self.weight.data = self.state.CB

        return out


class AutoAWQLinear(QuantizedLinearBase):
    """
    Auto awq 8-bit linear layer.
    """

    def __init__(
        self, in_features: int, out_features: int, has_bias: bool = True, **kwargs
    ):
        super().__init__()

        from awq.modules.linear import WQLinear_GEMM

        wqlinear = WQLinear_GEMM(
            w_bit=4,
            group_size=128,
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            dev=None,
        )

        for name, buffer in wqlinear.named_buffers():
            self.register_buffer(name, buffer)
        for name, param in wqlinear.named_parameters():
            self.register_parameter(name, param)

        self.w_bit = wqlinear.w_bit
        self.group_size = wqlinear.group_size
        self.bias = wqlinear.bias
        self.out_features = wqlinear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from awq.modules.linear.gemm import WQLinearMMFunction

        out_shape = x.shape[:-1] + (self.out_features,)

        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()

        with torch.no_grad():
            out = WQLinearMMFunction.apply(
                x,
                self.qweight,
                self.qzeros,
                self.scales,
                self.w_bit,
                self.group_size,
                self.bias,
                self.out_features,
            )

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        return out.reshape(out_shape)


class W8A8Linear(QuantizedLinearBase):
    """
    8-bit weight and activation quantized linear layer.
    """

    @staticmethod
    @torch.no_grad()
    def quant_act(act):
        act_shape = act.shape
        act.view(-1, act_shape[-1])
        scales = act.abs().max(dim=-1, keepdim=True)[0]
        scales = scales.to(torch.float)
        scales.clamp_(min=1e-5).div_(127.0)
        aa = act.div(scales).round_()
        return aa.to(torch.int8).view(-1, act_shape[-1]), scales.view(-1)

    def __init__(
        self, in_features: int, out_features: int, has_bias: bool = True, **kwargs
    ) -> torch.nn.Module:

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight",
            torch.zeros(
                self.out_features,
                self.in_features,
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "scale_channel",
            torch.ones(
                [self.out_features],
                dtype=torch.float,
                requires_grad=False,
            ),
        )
        if has_bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (self.out_features,), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w8a8gemm, _ = try_import_opt_dep("w8a8gemm", "quant")
        w8a8gemv, _ = try_import_opt_dep("w8a8gemv", "quant")

        if x.dim() == 2:
            q_x, act_scale = W8A8Linear.quant_act(x)

            out = torch.zeros(
                [x.shape[0], self.out_features], dtype=torch.float16, device="cuda"
            )
            w8a8gemm.mm(out, q_x, self.weight, act_scale, self.scale_channel, None)
        else:
            bs, seq, _ = x.shape
            q_x, act_scale = W8A8Linear.quant_act(x)
            if bs <= 4:
                q_x = q_x.view(bs, seq, -1)
                out = w8a8gemv.mv(q_x, self.weight, act_scale, self.scale_channel)
            else:
                out = torch.zeros(
                    [x.shape[0], self.out_features], dtype=torch.float16, device="cuda"
                )
                w8a8gemm.mm(out, q_x, self.weight, act_scale, self.scale_channel, None)
                out = out.reshape(bs, seq, -1)

        if self.bias is not None:
            out += self.bias

        return out


class W8A8MuxiLinear(QuantizedLinearBase):
    """
    Muxi 8-bit weight and activation quantized linear layer.
    """

    def __init__(
        self, in_features: int, out_features: int, has_bias: bool = True, **kwargs
    ) -> torch.nn.Module:

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight",
            torch.zeros(
                self.out_features,
                self.in_features,
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "scale_channel",
            torch.ones(
                [self.out_features],
                dtype=torch.float,
                requires_grad=False,
            ),
        )
        if has_bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (self.out_features,), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from chitu.muxi_utils import tbsgemm

        if isinstance(x, Tuple):
            q_x = x[0]
            act_scale = x[1]
            if q_x.dim() == 2:
                out = tbsgemm.mm(
                    self.weight, q_x, self.scale_channel, act_scale.to(torch.float32)
                )
            else:
                bs, seq, _ = q_x.shape
                q_x = q_x.view(bs * seq, q_x.shape[-1])
                out = tbsgemm.mm(
                    self.weight, q_x, self.scale_channel, act_scale.to(torch.float32)
                )
                out = out.reshape(bs, seq, -1)
            if self.bias is not None:
                out += self.bias
            return out
        else:
            if x.dim() == 2:
                m, _ = x.shape
                q_x, act_scale = tbsgemm.quant(x)
                out = tbsgemm.mm(
                    self.weight, q_x, self.scale_channel, act_scale.to(torch.float32)
                )
            else:
                bs, seq, _ = x.shape
                x = x.view(bs * seq, x.shape[-1])
                q_x, act_scale = tbsgemm.quant(x)
                out = tbsgemm.mm(
                    self.weight, q_x, self.scale_channel, act_scale.to(torch.float32)
                )
                out = out.reshape(bs, seq, -1)

            if self.bias is not None:
                out += self.bias

            return out


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


def linear_block_fp4(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: Optional[torch.Tensor] = None,
    weight_scale_2: Optional[torch.Tensor] = None,
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
            y = soft_fp4_raise_to_bf16_gemm_deepseek_v3(
                x, weight, weight_scale, weight_scale_2
            )
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
        y = soft_fp4_raise_to_fp8_gemm_deepseek_v3(
            x, act_scale, weight, weight_scale, weight_scale_2
        )
        if bias is not None:
            y += bias
        return y.view(x_shape[:-1] + y.shape[-1:]).to(x_dtype)


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
    ):
        super().__init__()

        dtype = dtype or torch.get_default_dtype()
        assert dtype.itemsize == 1

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        self.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty(
                    (out_features, in_features), dtype=dtype, requires_grad=False
                )
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
                    requires_grad=False,
                )
            ),
        )

        if has_bias:
            self.register_parameter(
                "bias",
                torch.nn.Parameter(
                    torch.empty(out_features, dtype=bias_dtype, requires_grad=False)
                ),
            )
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        return linear_block_fp8(
            x, self.weight, self.scale, self.bias, block_size=self.block_size
        )


class Blockfp4Linear(QuantizedLinearBase):
    """
    block 8-bit weight and activation quantized linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = False,
        dtype=torch.uint8,
        bias_dtype=None,
        block_size=8,
        scale_2_dim=1,
        **kwargs,
    ):
        super().__init__()

        dtype = dtype or torch.get_default_dtype()

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

        scale_out_features = out_features
        scale_in_features = (in_features + block_size - 1) // block_size
        self.register_parameter(
            "scale",
            torch.nn.Parameter(
                torch.empty(
                    scale_out_features,
                    scale_in_features,
                    dtype=torch.float8_e4m3fn,
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

    def register_scale_2_param(
        self,
        scale_2_size: int = 1,
    ):
        self.register_parameter(
            "input_scale",
            torch.nn.Parameter(
                torch.empty(
                    scale_2_size,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "scale_2",
            torch.nn.Parameter(
                torch.empty(
                    scale_2_size,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            ),
        )

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        return linear_block_fp4(
            x, self.weight, self.scale, self.scale_2, self.bias, block_size=128
        )


class QuantizationRegistry:
    """
    Registry of available quantization methods and their implementations.
    """

    _registry: Dict[str, Type[QuantizedLinearBase]] = {
        "llmint8": LLMInt8Linear,
        "autoawq": AutoAWQLinear,
        "simple_w8a8": W8A8Linear,
        "simple_w8a8_muxi": W8A8MuxiLinear,
        "blockfp8": Blockfp8Linear,
        "blockfp4": Blockfp4Linear,
    }

    @classmethod
    def get_quantized_linear_class(
        cls, method: Optional[str]
    ) -> Optional[Type[QuantizedLinearBase]]:
        """
        Get the quantized linear implementation for the specified method.

        Arguments:
            method: Quantization method name, or None for no quantization

        Returns:
            The quantized linear class, or None if method is None or not found
        """
        if method is None:
            return None

        impl = cls._registry.get(method)
        if impl is None:
            logger.warning(f"Unknown quantization method: {method}")

        return impl

    @classmethod
    def register_method(
        cls, name: str, implementation: Type[QuantizedLinearBase]
    ) -> None:
        """
        Register a new quantization method.

        Arguments:
            name: Name of the quantization method
            implementation: Implementation class
        """
        cls._registry[name] = implementation
