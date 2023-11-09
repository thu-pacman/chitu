from torch import nn
from torch.autograd import Function
import torch
from torch import Tensor
from typing import Sequence
import torch._custom_ops

import uniserve_cuda


# Registers the custom op
@torch._custom_ops.custom_op("uniserve::ragged_nchw_groupnorm")
def ragged_nchw_groupnorm(
    input: Tensor,
    c: int,
    idx_cpu: Tensor,
    num_groups: int,
    weight: Tensor,
    bias: Tensor,
    eps: float,
) -> Tensor:
    raise NotImplementedError()


# for compile
@torch._custom_ops.impl_abstract("uniserve::ragged_nchw_groupnorm")
def ragged_nchw_groupnorm_abstract(
    input: Tensor,
    c: int,
    idx_cpu: Tensor,
    num_groups: int,
    weight: Tensor,
    bias: Tensor,
    eps: float,
):
    return input.new_empty(input.shape)


# Next, let’s add an implementation for the operator:
# Adds an implementation for the custom op
@torch._custom_ops.impl("uniserve::ragged_nchw_groupnorm")
def ragged_nchw_groupnorm_impl(
    input: Tensor,
    c: int,
    idx_cpu: Tensor,
    num_groups: int,
    weight: Tensor,
    bias: Tensor,
    eps: float,
):
    assert idx_cpu.device.type == 'cpu'
    return uniserve_cuda.ragged_nchw_groupnorm_forward(
        input, c, idx_cpu, num_groups, weight, bias, eps
    )


class RaggedNchwGroupNorm(nn.Module):
    def __init__(self, shadow_norm: nn.GroupNorm):
        super(RaggedNchwGroupNorm, self).__init__()
        self.weight = shadow_norm.weight
        self.bias = shadow_norm.bias
        self.num_groups = shadow_norm.num_groups
        self.eps = shadow_norm.eps

    def forward(self, input, c, idx_cpu):
        return torch.ops.uniserve.ragged_nchw_groupnorm(
            input, c, idx_cpu, self.num_groups, self.weight, self.bias, self.eps
        )
