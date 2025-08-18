# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch

from chitu.lazy import single_dispatch_lazy_tensor


@single_dispatch_lazy_tensor
def linear(
    act: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    return torch.nn.functional.linear(act, weight, bias)
