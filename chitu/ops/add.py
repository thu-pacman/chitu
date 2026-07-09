# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch

from chitu.lazy import LazyTensor, single_dispatch_lazy_tensor
from chitu.ops.moe_sum import moe_sum_per_token, moe_sum_per_token_with_shared


@single_dispatch_lazy_tensor
def add(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
):
    """
    Add two tensors, optionally writing the result to `out`.

    Lazy MoE sums can be fused with this addition before materialization.
    """
    if isinstance(y, LazyTensor):
        return add(y, x, out=out)
    return torch.add(x, y, out=out)


@add.register
def _add_moe_sum_per_token(
    x: moe_sum_per_token.lazy_tensor_type(),
    y: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
):
    return moe_sum_per_token_with_shared(
        x.kwargs["x"],
        x.kwargs["topk_weights"],
        y,
        out=out,
    )


_moe_sum_per_token_lazy_tensor_type = moe_sum_per_token.lazy_tensor_type()
_moe_sum_per_token_lazy_tensor_type.__add__ = lambda self, other: add(self, other)
_moe_sum_per_token_lazy_tensor_type.__radd__ = lambda self, other: add(self, other)
_moe_sum_per_token_lazy_tensor_type.__iadd__ = lambda self, other: add(
    self, other, out=self.kwargs["out"]
)
