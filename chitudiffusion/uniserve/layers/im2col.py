from torch import nn
import torch
from torch import Tensor
from typing import Sequence
import torch._custom_ops

import uniserve_cuda


# Registers the custom op
@torch._custom_ops.custom_op("uniserve::ragged_nhwc_im2col")
def ragged_nhwc_im2col(
    input: Tensor,
    idx_cuda: Tensor,
    idx_cpu: Tensor,
    idx_out_cuda: Tensor,
    idx_out_cpu: Tensor,
    kernel_size: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    stride: Sequence[int],
) -> Tensor:
    raise NotImplementedError()


# for compile
@torch._custom_ops.impl_abstract("uniserve::ragged_nhwc_im2col")
def ragged_nhwc_im2col_abstract(
    input: Tensor,
    idx_cuda: Tensor,
    idx_cpu: Tensor,
    idx_out_cuda: Tensor,
    idx_out_cpu: Tensor,
    kernel_size: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    stride: Sequence[int],
):
    # TODO: deal with downscale and upscale
    assert kernel_size[0] == kernel_size[1]
    assert padding[0] == padding[1] == kernel_size[0] // 2
    assert dilation[0] == 1 and dilation[1] == 1
    assert stride[0] == stride[1]
    assert input.dim() == 2
    c = input.shape[1]
    output_size = [
        input.shape[0],
        kernel_size[0] * kernel_size[1] * c,
    ]  # [nhw, rsc] c last dim for performance
    if stride != (1, 1):
        # Unsupported data-dependent control flow
        # for v in idx_cpu[0]:
        #     assert v % 2 == 0
        # for v in idx_cpu[1]:
        #     assert v % 2 == 0
        divisor = stride[0] * stride[1]
        assert output_size[0] % divisor == 0
        output_size[0] //= divisor
    return input.new_empty(output_size)


# Next, let’s add an implementation for the operator:
# Adds an implementation for the custom op
@torch._custom_ops.impl("uniserve::ragged_nhwc_im2col")
def ragged_nhwc_im2col_impl(
    input: Tensor,
    idx_cuda: Tensor,
    idx_cpu: Tensor,
    idx_out_cuda: Tensor,
    idx_out_cpu: Tensor,
    kernel_size: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    stride: Sequence[int],
):
    assert idx_cpu.device.type == "cpu"
    assert idx_out_cpu.device.type == "cpu"
    assert idx_cuda.device.type == "cuda"
    assert idx_out_cuda.device.type == "cuda"
    return uniserve_cuda.ragged_nhwc_im2col(
        input,
        idx_cuda,
        idx_cpu,
        idx_out_cuda,
        idx_out_cpu,
        kernel_size,
        padding,
        dilation,
        stride,
    )
