from torch import nn
from torch.autograd import Function
import torch
from torch import Tensor
from typing import Sequence
import torch._custom_ops

import uniserve_cuda


try:
    # Registers the custom op
    @torch._custom_ops.custom_op("uniserve::ragged_nhwc_to_nchw")
    def ragged_nhwc_to_nchw(
        x: Tensor,
        c: int,
        idx_cpu: Tensor,
    ) -> Tensor:
        raise NotImplementedError()

    # for compile
    @torch._custom_ops.impl_abstract("uniserve::ragged_nhwc_to_nchw")
    def ragged_nhwc_to_nchw_abstract(
        x: Tensor,
        c: int,
        idx_cpu: Tensor,
    ):
        assert x.dim() == 2
        return x.new_empty([x.shape[0] * x.shape[1]]).flatten()

    # Next, let’s add an implementation for the operator:
    # Adds an implementation for the custom op
    @torch._custom_ops.impl("uniserve::ragged_nhwc_to_nchw")
    def ragged_nhwc_to_nchw_impl(
        x: Tensor,
        c: int,
        idx_cpu: Tensor,
    ):
        assert idx_cpu.device.type == "cpu"
        return uniserve_cuda.ragged_nhwc_to_nchw(x, c, idx_cpu)

except RuntimeError as e:
    print(e)


# Registers the custom op
@torch._custom_ops.custom_op("uniserve::ragged_nchw_to_nhwc")
def ragged_nchw_to_nhwc(
    x: Tensor,
    c: int,
    idx_cpu: Tensor,
) -> Tensor:
    raise NotImplementedError()


# for compile
@torch._custom_ops.impl_abstract("uniserve::ragged_nchw_to_nhwc")
def ragged_nchw_to_nhwc_abstract(
    x: Tensor,
    c: int,
    idx_cpu: Tensor,
):
    return x.new_empty(x.shape).reshape(-1, c)


# Next, let’s add an implementation for the operator:
# Adds an implementation for the custom op
@torch._custom_ops.impl("uniserve::ragged_nchw_to_nhwc")
def ragged_nchw_to_nhwc_impl(
    x: Tensor,
    c: int,
    idx_cpu: Tensor,
):
    assert idx_cpu.device.type == "cpu"
    return uniserve_cuda.ragged_nchw_to_nhwc(x, c, idx_cpu)
