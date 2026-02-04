from torch import nn
from torch.autograd import Function
import torch

from torch import Tensor
from typing import Sequence
import torch._custom_ops

import uniserve_cuda

import flash_attn_2_cuda
import flash_attn


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
    # Flash-attn 2.8.3 的 varlen_fwd API 已经改变
    # 新的参数顺序：q, k, v, out, cu_seqlens_q, cu_seqlens_k, seqused_k, leftpad_k,
    #                block_table, alibi_slopes, max_seqlen_q, max_seqlen_k, dropout_p,
    #                softmax_scale, zero_tensors, causal, window_size_left,
    #                window_size_right, softcap, return_softmax, generator

    # Flash-attn varlen_fwd 期望输入格式: [total_seq_len, num_heads, head_dim]
    # 其中 head_dim <= 256

    # 打印原始形状用于调试
    # print(f"Original shapes - Q: {input1.shape}, K: {input2.shape}, V: {input3.shape}")

    def reshape_for_flash_attn(tensor, name=""):
        """将 tensor reshape 为 flash-attn 期望的格式 [total_seq_len, num_heads, head_dim]"""
        original_shape = tensor.shape

        if tensor.dim() == 4:
            batch, dim1, dim2, dim3 = tensor.shape
            # print(f"{name} 4D input: batch={batch}, dims=({dim1}, {dim2}, {dim3})")

            # 从 scaled_dot_product_attention 传入的通常是 [batch, num_heads, seq_len, head_dim]
            # 最后一个维度应该是 head_dim（通常 <= 256）
            head_dim = dim3

            if head_dim > 256:
                raise ValueError(
                    f"{name} head_dim ({head_dim}) exceeds 256. Shape: {original_shape}"
                )

            # 判断格式：dim1 和 dim2 哪个是 num_heads，哪个是 seq_len
            # num_heads 通常较小（如 10），seq_len 通常较大
            if dim1 < dim2 and dim1 <= 128:
                # 格式是 [batch, num_heads, seq_len, head_dim]
                # transpose 为 [batch, seq_len, num_heads, head_dim]
                tensor = tensor.transpose(1, 2).contiguous()
            elif dim2 < dim1 and dim2 <= 128:
                # 格式已经是 [batch, seq_len, num_heads, head_dim]，不需要 transpose
                pass
            else:
                # 无法确定，默认假设 [batch, num_heads, seq_len, head_dim] 并 transpose
                tensor = tensor.transpose(1, 2).contiguous()

            # 现在 tensor 是 [batch, seq_len, num_heads, head_dim]
            batch, seq_len, num_heads, head_dim = tensor.shape
            # print(f"{name} reshaped: total_seq={batch * seq_len}, num_heads={num_heads}, head_dim={head_dim}")
            # Reshape 为 [batch*seq_len, num_heads, head_dim]
            return tensor.contiguous().view(-1, num_heads, head_dim)

        elif tensor.dim() == 3:
            total_seq_len, dim1, dim2 = tensor.shape
            # print(f"{name} 3D input: ({total_seq_len}, {dim1}, {dim2})")
            # 假设格式是 [total_seq_len, num_heads, head_dim]
            if dim2 > 256:
                raise ValueError(
                    f"{name} 3D tensor head_dim ({dim2}) > 256. Shape: {original_shape}"
                )
            return tensor.contiguous()
        else:
            raise ValueError(
                f"{name} unexpected tensor dimension: {tensor.dim()}, shape: {original_shape}"
            )

    q = reshape_for_flash_attn(input1, "Q")
    k = reshape_for_flash_attn(input2, "K")
    v = reshape_for_flash_attn(input3, "V")

    # print(f"After reshape - Q: {q.shape}, K: {k.shape}, V: {v.shape}")

    # 验证形状和 head_dim
    assert (
        q.shape[1] == k.shape[1] == v.shape[1]
    ), f"num_heads mismatch: q={q.shape[1]}, k={k.shape[1]}, v={v.shape[1]}"
    head_dim = q.shape[-1]
    if head_dim > 256:
        raise ValueError(
            f"head_dim ({head_dim}) exceeds 256. Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}"
        )

    # 计算 softmax_scale: head_dim 是最后一个维度
    softmax_scale = head_dim ** (-0.5)

    out, softmax_lse, S_dmask, rng_state = flash_attn_2_cuda.varlen_fwd(
        q,  # [total_seq_len, num_heads, head_dim]
        k,  # [total_seq_len, num_heads, head_dim]
        v,  # [total_seq_len, num_heads, head_dim]
        None,  # out (output tensor, optional)
        cu_seqlen_q,  # cu_seqlens_q
        cu_seqlen_k,  # cu_seqlens_k
        None,  # seqused_k (optional)
        None,  # leftpad_k (optional)
        None,  # block_table (optional)
        None,  # alibi_slopes (optional)
        max_seqlen_q,  # max_seqlen_q
        max_seqlen_k,  # max_seqlen_k
        0.0,  # dropout_p
        softmax_scale,  # softmax_scale
        False,  # zero_tensors
        False,  # causal
        -1,  # window_size_left
        -1,  # window_size_right
        0.0,  # softcap
        False,  # return_softmax
        None,  # generator (optional)
    )

    # 如果原始输入是 4D，需要将输出 reshape 回原始形状
    if input1.dim() == 4:
        batch, seq_len, num_heads, head_dim = input1.shape
        out = out.view(batch, seq_len, num_heads, head_dim)

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
