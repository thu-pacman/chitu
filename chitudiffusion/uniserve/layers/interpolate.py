from torch import nn
from torch.autograd import Function
import torch
from torch import Tensor
from typing import Sequence
import torch._custom_ops

import uniserve_cuda


# Registers the custom op
# // {'scale_factor': 2.0, 'mode': 'nearest'}
@torch._custom_ops.custom_op("uniserve::ragged_nchw_interpolate")
def ragged_nchw_interpolate(
    x: Tensor,
    c: int,
    idx_cpu: Tensor,
    scale_factor: float,
    mode: str,
) -> Tensor:
    raise NotImplementedError()


# for compile
@torch._custom_ops.impl_abstract("uniserve::ragged_nchw_interpolate")
def ragged_nchw_interpolate_abstract(
    x: Tensor,
    c: int,
    idx_cpu: Tensor,
    scale_factor: float,
    mode: str,
):
    assert int(scale_factor) == scale_factor
    scale_factor = int(scale_factor)
    return x.new_empty([x.numel() * (scale_factor**2)])


# Next, let’s add an implementation for the operator:
# Adds an implementation for the custom op
@torch._custom_ops.impl("uniserve::ragged_nchw_interpolate")
def ragged_nchw_interpolate_impl(
    x: Tensor,
    c: int,
    idx_cpu: Tensor,
    scale_factor: float,
    mode: str,
):
    assert idx_cpu.device.type == "cpu"
    assert mode == "nearest"
    return uniserve_cuda.ragged_nchw_interpolate(x, c, idx_cpu, scale_factor, mode)
