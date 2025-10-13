# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
from chitu.quantization import (
    NormalLinear,
)
from chitu.native_layout import (
    enable_native_layout_weight,
    InXOutWeight,
)
from chitu.lazy import single_dispatch_lazy_tensor
from chitu.quantization.registry import QuantizationRegistry
from chitu.device_type import is_hygon


@single_dispatch_lazy_tensor
def hygon_native_linear(
    x: torch.Tensor,
    weight: InXOutWeight,
):
    w = weight.layout_tensor
    return torch.matmul(x, w)


@QuantizationRegistry.register_linear(
    None,
    when=lambda: is_hygon(),
    priority=1,
)
class InXOutLinear(
    enable_native_layout_weight("weight", InXOutWeight),
    NormalLinear,
):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hygon_native_linear(x, self.get_native_layout_weight())
