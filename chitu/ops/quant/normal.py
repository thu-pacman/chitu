# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch

from chitu.lazy import single_dispatch_lazy_tensor


@single_dispatch_lazy_tensor
def linear(
    act: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if out_dtype is None:
        return torch.nn.functional.linear(act, weight, bias)
    if (
        out_dtype == torch.float32
        and act.is_cuda
        and act.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype in (torch.float16, torch.bfloat16)
    ):
        out = torch.mm(
            act.reshape(-1, act.size(-1)), weight.t(), out_dtype=torch.float32
        )
        if bias is not None:
            out = out + bias.float()
        return out.reshape(*act.shape[:-1], weight.size(0))
    return torch.nn.functional.linear(act, weight, bias).to(out_dtype)
