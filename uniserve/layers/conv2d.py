from torch import nn
import torch
from torch import Tensor
from typing import Sequence
import torch._custom_ops

import uniserve_cuda
from diffusers.models.lora import LoRACompatibleConv


# Registers the custom op
@torch._custom_ops.custom_op("uniserve::ragged_nchw2nhwc_unfold_matmul")
def ragged_nchw2nhwc_unfold_matmul(
    input: Tensor,
    c: int,
    idx_cpu: Tensor,
    weight: Tensor,
    kh: int,
    kw: int,
    ph: int,
    pw: int,
    dh: int,
    dw: int,
    sh: int,
    sw: int,
) -> Tensor:
    raise NotImplementedError()


# for compile
@torch._custom_ops.impl_abstract("uniserve::ragged_nchw2nhwc_unfold_matmul")
def ragged_nchw2nhwc_unfold_matmul_abstract(
    input: Tensor,
    c: int,
    idx_cpu: Tensor,
    weight: Tensor,
    kh: int,
    kw: int,
    ph: int,
    pw: int,
    dh: int,
    dw: int,
    sh: int,
    sw: int,
):
    # TODO: deal with downscale and upscale
    assert kh == kw
    assert ph == pw == kh // 2
    assert ph % 2 == 1
    assert dh == dw == 1
    assert sh == sw == 1
    return input.new_empty(input.shape).reshape(-1, c)


# Next, let’s add an implementation for the operator:
# Adds an implementation for the custom op
@torch._custom_ops.impl("uniserve::ragged_nchw2nhwc_unfold_matmul")
def ragged_nchw2nhwc_unfold_matmul_impl(
    input: Tensor,
    c: int,
    idx_cpu: Tensor,
    weight: Tensor,
    kh: int,
    kw: int,
    ph: int,
    pw: int,
    dh: int,
    dw: int,
    sh: int,
    sw: int,
):
    assert idx_cpu.device.type == "cpu"
    return uniserve_cuda.ragged_nchw2nhwc_unfold_matmul(
        input, c, idx_cpu, weight, kh, kw, ph, pw, dh, dw, sh, sw
    )


class RaggedNhwcConv2d(nn.Module):
    def __init__(self, shadow_model: LoRACompatibleConv):
        assert isinstance(shadow_model, LoRACompatibleConv)
        super().__init__()
        self.shadow = shadow_model
        weight, bias = self.weight_conv2gemm(shadow_model)
        self.weight = weight
        self.bias = bias
        # If we want to sperate matmul from the C kernel, shape inference
        # for im2col is requried.
        # self.mm = nn.Linear(*weight.shape)
        # assert self.mm.weight.shape == weight.T.shape
        # assert self.mm.bias.shape == bias.shape
        # self.mm.weight = nn.Parameter(weight.T)
        # self.mm.bias = bias

    def weight_conv2gemm(self, conv: LoRACompatibleConv):
        f = conv.weight.shape[0]
        kernel = conv.weight.reshape(f, -1).T.contiguous()  # [crs, f]
        bias = conv.bias.flatten()  # [f]
        return nn.Parameter(kernel), nn.Parameter(bias)

    def forward(self, x, c, idx_cpu):
        """
        x: [nhw, c]
        output: [nhw, c]"""
        x = torch.ops.uniserve.ragged_nhwc_to_nchw(x, c, idx_cpu)
        x = torch.ops.uniserve.ragged_nchw2nhwc_unfold_matmul(
            x,
            c,
            idx_cpu,
            self.weight,
            *self.shadow.kernel_size,
            *self.shadow.padding,
            *self.shadow.dilation,
            *self.shadow.stride
        )
        x = x + self.bias
        # x = self.mm(x)  # [nhw, c]
        return x
