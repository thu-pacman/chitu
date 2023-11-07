from torch import nn
from torch.autograd import Function
import torch
from torch import Tensor
from typing import Sequence
import torch._custom_ops

import uniserve_cuda


# Registers the custom op
@torch._custom_ops.custom_op("uniserve::ragged_nhwc_to_nchw")
def ragged_nhwc_to_nchw(
    input: Tensor,
    n: int,
    c: int,
    HxWs: Sequence[int],
) -> Tensor:
    raise NotImplementedError()


# for compile
@torch._custom_ops.impl_abstract("uniserve::ragged_nhwc_to_nchw")
def ragged_nhwc_to_nchw_abstract(
    input: Tensor,
    n: int,
    c: int,
    HxWs: Sequence[int],
):
    return input.new_empty(input.shape)


# Next, let’s add an implementation for the operator:
# Adds an implementation for the custom op
@torch._custom_ops.impl("uniserve::ragged_nhwc_to_nchw")
def ragged_nhwc_to_nchw_impl(
    input: Tensor,
    n: int,
    c: int,
    HxWs: Sequence[int],
):
    return uniserve_cuda.ragged_nhwc_to_nchw(input, n, c, HxWs)


# Registers the custom op
@torch._custom_ops.custom_op("uniserve::ragged_nchw_to_nhwc")
def ragged_nchw_to_nhwc(
    input: Tensor,
    n: int,
    c: int,
    HxWs: Sequence[int],
) -> Tensor:
    raise NotImplementedError()


# for compile
@torch._custom_ops.impl_abstract("uniserve::ragged_nchw_to_nhwc")
def ragged_nchw_to_nhwc_abstract(
    input: Tensor,
    n: int,
    c: int,
    HxWs: Sequence[int],
):
    return input.new_empty(input.shape).reshape(-1, c)


# Next, let’s add an implementation for the operator:
# Adds an implementation for the custom op
@torch._custom_ops.impl("uniserve::ragged_nchw_to_nhwc")
def ragged_nchw_to_nhwc_impl(
    input: Tensor,
    n: int,
    c: int,
    HxWs: Sequence[int],
):
    return uniserve_cuda.ragged_nchw_to_nhwc(input, n, c, HxWs)
