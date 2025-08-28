# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.lazy import eval_lazy
from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.base import QuantizedLinearBase


@QuantizationRegistry.register_linear("autoawq")
class AutoAWQLinear(QuantizedLinearBase):
    """
    Auto awq 4-bit linear layer.
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
            self.register_parameter(
                name, torch.nn.Parameter(buffer, requires_grad=False)
            )
        for name, param in wqlinear.named_parameters():
            self.register_parameter(name, param)

        if not hasattr(self, "bias"):
            self.register_parameter("bias", None)

        self.w_bit = wqlinear.w_bit
        self.group_size = wqlinear.group_size
        self.out_features = wqlinear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from awq.modules.linear.gemm import WQLinearMMFunction

        x = eval_lazy(x)

        if x.dtype != torch.float16:
            x = x.to(torch.float16)

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
