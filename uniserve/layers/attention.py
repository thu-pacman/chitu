from torch import nn
from torch.autograd import Function
import torch

from torch import Tensor
from typing import Sequence
import torch._custom_ops

import uniserve_cuda


# Registers the custom op
@torch._custom_ops.custom_op("uniserve::ragged_nseqf_attention_forward")
def ragged_nseqf_attention_forward(
    input1: Tensor,
    input2: Tensor,
    input3: Tensor,
    heads: int,
    features: int,
    idx_cpu: Tensor,
    enco: bool,
) -> Tensor:
    raise NotImplementedError()


# for compile
@torch._custom_ops.impl_abstract("uniserve::ragged_nseqf_attention_forward")
def ragged_nseqf_attention_forward_abstract(
    input1: Tensor,
    input2: Tensor,
    input3: Tensor,
    heads: int,
    features: int,
    idx_cpu: Tensor,
    enco: bool,
):
    return input1.new_empty(input1.shape)


# Next, let's add an implementation for the operator:
# Adds an implementation for the custom op


@torch._custom_ops.impl("uniserve::ragged_nseqf_attention_forward")
def ragged_nseqf_attention_forward_impl(
    input1: Tensor,
    input2: Tensor,
    input3: Tensor,
    heads: int,
    features: int,
    idx_cpu: Tensor,
    enco: bool,
):
    assert idx_cpu.device.type == "cpu"
    return uniserve_cuda.ragged_nseqf_attention_forward(
        input1, input2, input3, heads, features, idx_cpu, enco
    )


class RaggedNseqfAttentionForward(nn.Module):
    def __init__(self):
        super(RaggedNseqfAttentionForward, self).__init__()

    def forward(self, input1, input2, input3, heads, features, idx_cpu, enco=False):
        return torch.ops.uniserve.ragged_nseqf_attention_forward(
            input1, input2, input3, heads, features, idx_cpu, enco
        )
