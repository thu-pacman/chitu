# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import torch

from chitu.utils import try_import_and_setup_torch_npu
from chitu.quantization.base import QuantizedLinearBase
from chitu.distributed.parallel_state import get_tp_group
from chitu.quantization.registry import QuantizationRegistry
from chitu.native_layout import (
    NativeLayoutMixin,
    NpuFractalZnTensor,
    Repeat1ToLength,
)
from chitu.lazy import eval_lazy

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


@QuantizationRegistry.register_linear("ascend_w8a8")
class AscendW8A8Linear(NativeLayoutMixin, QuantizedLinearBase):
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

    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(self.weight, NpuFractalZnTensor)
        self.apply_native_layout(
            self.input_scale,
            Repeat1ToLength,
            length=self.in_features,
            out_dtype=torch.get_default_dtype(),
        )
        self.apply_native_layout(
            self.input_offset,
            Repeat1ToLength,
            length=self.in_features,
            out_dtype=torch.get_default_dtype(),
        )

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
