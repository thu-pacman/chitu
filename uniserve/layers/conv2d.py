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
    assert dh == dw == 1
    assert sh == sw
    assert input.dim() == 1
    assert weight.dim() == 2
    output_size = [input.reshape(-1, c).shape[0], weight.shape[1]]
    if sh != 1:
        # Unsupported data-dependent control flow
        # for v in idx_cpu[0]:
        #     assert v % 2 == 0
        # for v in idx_cpu[1]:
        #     assert v % 2 == 0
        divisor = sh * sw
        assert output_size[0] % divisor == 0
        output_size[0] //= divisor
    return input.new_empty(output_size)


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
    def __init__(self, shadow_model: nn.Conv2d):
        assert isinstance(shadow_model, nn.Conv2d)
        super().__init__()
        self.shadow = shadow_model
        weight, bias = self.weight_conv2gemm(shadow_model)
        self.weight = weight
        self.bias = bias
        self.in_chalnels = shadow_model.in_channels
        self.out_channels = shadow_model.out_channels
        # If we want to sperate matmul from the C kernel, shape inference
        # for im2col is requried.
        # self.mm = nn.Linear(*weight.shape)
        # assert self.mm.weight.shape == weight.T.shape
        # assert self.mm.bias.shape == bias.shape
        # self.mm.weight = nn.Parameter(weight.T)
        # self.mm.bias = bias

    def weight_conv2gemm(self, conv: nn.Conv2d):
        f = conv.weight.shape[0]
        kernel = conv.weight.reshape(f, -1).T.contiguous()  # [crs, f]
        bias = conv.bias.flatten()  # [f]
        return nn.Parameter(kernel), nn.Parameter(bias)

    # TODO: remove c and infer it from x.shape
    def forward(self, x, c, idx_cpu):
        """
        x: [nhw, c]
        output: [nhw, c]"""
        assert c == self.shadow.in_channels
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
