from torch import nn
from torch.autograd import Function
import torch

from torch import Tensor
from typing import Sequence
import torch._custom_ops

import uniserve_cuda

import flash_attn_2_cuda


# Registers the custom op
@torch._custom_ops.custom_op("uniserve::ragged_nseqf_attention_forward")
def ragged_nseqf_attention_forward(
    input1: Tensor,
    input2: Tensor,
    input3: Tensor,
    heads_num: int,
    heads_dim: int,
    idx_cpu: Tensor,
    enco: bool,
) -> Tensor:
    raise NotImplementedError()


@torch._custom_ops.custom_op("uniserve::flashattn_varlen_fwd")
def flashattn_varlen_fwd(
    input1: Tensor,
    input2: Tensor,
    input3: Tensor,
    cu_seqlen_q: Tensor,
    cu_seqlen_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> Tensor:
    raise NotImplementedError()


# for compile
@torch._custom_ops.impl_abstract("uniserve::ragged_nseqf_attention_forward")
def ragged_nseqf_attention_forward_abstract(
    input1: Tensor,
    input2: Tensor,
    input3: Tensor,
    heads_num: int,
    heads_dim: int,
    idx_cpu: Tensor,
    enco: bool,
):
    return input1.new_empty(input1.shape)


@torch._custom_ops.impl_abstract("uniserve::flashattn_varlen_fwd")
def flashattn_varlen_fwd_abstract(
    input1: Tensor,  # [total_q, num_heads, head_size]
    input2: Tensor,
    input3: Tensor,
    cu_seqlen_q: Tensor,
    cu_seqlen_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
):
    assert input1.dim() == 3
    assert input2.dim() == 3
    assert input3.dim() == 3
    # FA2 only suports QKV with the same hidden size
    return input1.new_empty(list(input1.shape[:-1]) + [input3.shape[-1]])


# Next, let's add an implementation for the operator:
# Adds an implementation for the custom op


@torch._custom_ops.impl("uniserve::ragged_nseqf_attention_forward")
def ragged_nseqf_attention_forward_impl(
    input1: Tensor,
    input2: Tensor,
    input3: Tensor,
    heads_num: int,
    heads_dim: int,
    idx_cpu: Tensor,
    enco: bool,
):
    assert idx_cpu.device.type == "cpu"
    # print(input1.shape, input2.shape, input3.shape, heads, features, idx_cpu, enco)
    return uniserve_cuda.ragged_nseqf_attention_forward(
        input1, input2, input3, heads_num, heads_dim, idx_cpu, enco
    )


@torch._custom_ops.impl("uniserve::flashattn_varlen_fwd")
def flashattn_varlen_fwd_impl(
    input1: Tensor,
    input2: Tensor,
    input3: Tensor,
    cu_seqlen_q: Tensor,
    cu_seqlen_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
):
    softmax_scale = input1.shape[-1] ** (-0.5)
    # A fixed max length seems have the same performance
    # max_seqlen_q = max_seqlen_k = 32 * 32
    (
        out,
        q,
        k,
        v,
        out_padded,
        softmax_lse,
        S_dmask,
        rng_state,
    ) = flash_attn_2_cuda.varlen_fwd(
        input1,
        input2,
        input3,
        None,
        cu_seqlen_q,
        cu_seqlen_k,
        None,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        softmax_scale,
        False,
        False,
        -1,
        -1,
        False,
        None,
    )
    return out


class RaggedNseqfAttentionForward(nn.Module):
    def __init__(self):
        super(RaggedNseqfAttentionForward, self).__init__()

    def forward(
        self, input1, input2, input3, heads_num, heads_dim, idx_cpu, enco=False
    ):
        return torch.ops.uniserve.ragged_nseqf_attention_forward(
            input1, input2, input3, heads_num, heads_dim, idx_cpu, enco
        )


class FlashAttentionVarlenForward(nn.Module):
    def __init__(self):
        super(FlashAttentionVarlenForward, self).__init__()

    def forward(self, input1, input2, input3, cu_seqlens_q, cu_seqlens_k):
        return torch.ops.uniserve.flashattn_varlen_fwd(
            input1, input2, input3, cu_seqlens_q, cu_seqlens_k
        )
