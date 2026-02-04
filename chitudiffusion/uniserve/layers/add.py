from torch import nn
from torch.autograd import Function
import torch
from torch import Tensor
from typing import Sequence
import torch._custom_ops

import uniserve_cuda


# Registers the custom op
@torch._custom_ops.custom_op("uniserve::addB_jr_rr")
def addB_jr_rr(
    A: Tensor,
    idx_cuda: Tensor,
    B: Tensor,
) -> Tensor:
    raise NotImplementedError()


# for compile
@torch._custom_ops.impl_abstract("uniserve::addB_jr_rr")
def addB_jr_rr_abstract(
    A: Tensor,
    idx_cuda: Tensor,
    B: Tensor,
):
    assert A.shape[-1] == B.shape[-1]
    return A.new_empty(A.shape)


# Next, let’s add an implementation for the operator:
# Adds an implementation for the custom op
@torch._custom_ops.impl("uniserve::addB_jr_rr")
def addB_jr_rr_impl(
    A: Tensor,
    idx_cuda: Tensor,
    B: Tensor,
):
    assert idx_cuda.is_cuda
    assert A.shape[-1] == B.shape[-1]
    return uniserve_cuda.addB_jr_rr(A, idx_cuda[2], B)
