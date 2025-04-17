import torch
import logging
from typing import Dict, Tuple, Optional, Type

from chitu.utils import try_import_opt_dep


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


class Blockfp8Linear(QuantizedLinearBase):
    """
    block 8-bit weight and activation quantized linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        block_size=128,
        **kwargs,
    ) -> torch.nn.Module:

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        self.register_buffer(
            "weight",
            torch.empty(
                (out_features, in_features),
                dtype=torch.float8_e4m3fn,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "scale",
            torch.empty(
                (out_features // block_size, in_features // block_size),
                dtype=torch.float32,
                requires_grad=False,
            ),
        )

        if has_bias:
            self.register_buffer(
                "bias",
                torch.empty((out_features), dtype=torch.float16, requires_grad=False),
            )
        else:
            self.bias = None

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        from chitu.models.model_deepseek_v3 import linear_deepseek_v3

        out = linear_deepseek_v3(x, self.weight, self.scale, self.bias).to(x.dtype)
        return out


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
