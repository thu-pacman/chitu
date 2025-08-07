# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import torch
import triton
import triton.language as tl

from chitu.ops.triton_ops.utils import auto_retry_triton_compilation


@auto_retry_triton_compilation
def apply_rotary_pos_emb_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_type: str = "separated",
    block_size=128,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if rotary_type == "separated-half":
        # Currently we split and fallback to "interleaved". TODO: Implement a fully
        # fused kernel including the split and cat.
        q_rot, q_pass = q[..., : q.shape[-1] // 2], q[..., q.shape[-1] // 2 :]
        k_rot, k_pass = k[..., : k.shape[-1] // 2], k[..., k.shape[-1] // 2 :]
        q_rot_out, k_rot_out = apply_rotary_pos_emb_triton(
            q_rot,
            k_rot,
            cos,
            sin,
            rotary_type="separated",
            block_size=block_size,
        )
        q_out = torch.cat([q_rot_out, q_pass], dim=-1)
        k_out = torch.cat([k_rot_out, k_pass], dim=-1)
        return q_out, k_out

    if rotary_type == "interleaved-half":
        # Currently we split and fallback to "interleaved". TODO: Implement a fully
        # fused kernel including the split and cat.
        q_rot, q_pass = q[..., : q.shape[-1] // 2], q[..., q.shape[-1] // 2 :]
        k_rot, k_pass = k[..., : k.shape[-1] // 2], k[..., k.shape[-1] // 2 :]
        q_rot_out, k_rot_out = apply_rotary_pos_emb_triton(
            q_rot, k_rot, cos, sin, rotary_type="interleaved", block_size=block_size
        )
        q_out = torch.cat([q_rot_out, q_pass], dim=-1)
        k_out = torch.cat([k_rot_out, k_pass], dim=-1)
        return q_out, k_out

    # Prepare output tensor
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    q_shape = q.shape
    k_shape = k.shape

    if q.dim() == 4:
        q = q.view(-1, q_shape[-2], q_shape[-1])
        q_out = q_out.view(-1, q_shape[-2], q_shape[-1])
    elif q.dim() == 3:
        pass
    elif q.dim() == 2:
        q = q.view(-1, 1, q_shape[-1])
        q_out = q_out.view(-1, 1, q_shape[-1])
    else:
        assert False
    if k.dim() == 4:
        k = k.view(-1, k_shape[-2], k_shape[-1])
        k_out = k_out.view(-1, k_shape[-2], k_shape[-1])
    elif k.dim() == 3:
        pass
    elif k.dim() == 2:
        k = k.view(-1, 1, k_shape[-1])
        k_out = k_out.view(-1, 1, k_shape[-1])
    else:
        assert False

    assert q.shape[-1] == k.shape[-1]
    assert q.shape[0] == k.shape[0]
    assert q.shape[-1] // 2 == cos.shape[-1]
    assert q.shape[-1] // 2 == sin.shape[-1]

    if rotary_type == "separated":
        # "separated" has an [real, real, ..., real, imag, imag, ..., imag] layout.

        # Get tensor shapes
        q_batch_size, q_n_local_heads, q_head_dim = q.shape
        k_batch_size, k_n_local_heads, k_head_dim = k.shape

        # Define grid size
        q_grid = (q_batch_size * q_n_local_heads, triton.cdiv(q_head_dim, block_size))
        k_grid = (k_batch_size * k_n_local_heads, triton.cdiv(k_head_dim, block_size))

        # Launch kernel (TODO: use only 1 kernel)
        assert cos.is_contiguous()
        assert sin.is_contiguous()
        rotary_embedding_kernel_separated[q_grid](
            q,
            cos,
            sin,
            q_out,
            q_n_local_heads,
            q_head_dim,
            q.stride(0),
            q.stride(1),
            cos.stride(0),
            sin.stride(0),
            q_out.stride(0),
            q_out.stride(1),
            BLOCK_SIZE=block_size,
        )
        rotary_embedding_kernel_separated[k_grid](
            k,
            cos,
            sin,
            k_out,
            k_n_local_heads,
            k_head_dim,
            k.stride(0),
            k.stride(1),
            cos.stride(0),
            sin.stride(0),
            k_out.stride(0),
            k_out.stride(1),
            BLOCK_SIZE=block_size,
        )

    elif rotary_type == "interleaved":
        # "interleaved" has an [real, imag, real, imag, ..., real, imag] layout.

        bs, head_num_q, rotary_dim = q.shape
        bs, head_num_k, rotary_dim = k.shape

        assert cos.is_contiguous()
        assert sin.is_contiguous()

        # Launch kernel
        BLOCK_H = min(
            triton.cdiv(triton.next_power_of_2(bs), 128), max(head_num_q, head_num_k)
        )
        grid = lambda meta: (bs, triton.cdiv(max(head_num_q, head_num_k), BLOCK_H), 1)
        rotary_embedding_kernel_interleaved[grid](
            q,
            k,
            q_out,
            k_out,
            cos,
            sin,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            q_out.stride(0),
            q_out.stride(1),
            k_out.stride(0),
            k_out.stride(1),
            head_num_q,
            head_num_k,
            rotary_dim,
            BLOCK_H,
        )

    else:
        raise NotImplementedError(
            f"Unsupported rotary type: {rotary_type} for Triton implementation"
        )

    return q_out.view(q_shape), k_out.view(k_shape)


@triton.jit
def rotary_embedding_kernel_separated(
    Q,
    COS,
    SIN,
    OUTPUT,
    num_head,
    head_dim,
    stride_q1,
    stride_q2,
    stride_cos1,
    stride_sin1,
    stride_out1,
    stride_out2,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(axis=0)

    # Compute batch index and head index
    batch_idx = pid // num_head
    head_idx = pid % num_head

    # Pointers to the beginning of the Q, COS, SIN, and OUTPUT
    Q_ptr = Q + batch_idx * stride_q1 + head_idx * stride_q2
    COS_ptr = COS + batch_idx * stride_cos1
    SIN_ptr = SIN + batch_idx * stride_sin1
    OUTPUT_ptr = OUTPUT + batch_idx * stride_out1 + head_idx * stride_out2

    # Create block IDs
    block_id = tl.program_id(axis=1)

    # Create offsets for reading and writing
    offsets_0 = block_id * (BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2)
    offsets_1 = offsets_0 + head_dim // 2

    # Load data
    cos0 = tl.load(COS_ptr + offsets_0, mask=offsets_0 < head_dim // 2)
    sin0 = tl.load(SIN_ptr + offsets_0, mask=offsets_0 < head_dim // 2)
    q0 = tl.load(Q_ptr + offsets_0, mask=offsets_0 < head_dim // 2)
    q1 = tl.load(Q_ptr + offsets_1, mask=offsets_1 < head_dim)

    # Apply rotary embedding
    q_embed0 = q0 * cos0 - q1 * sin0
    q_embed1 = q1 * cos0 + q0 * sin0

    # Store result
    tl.store(OUTPUT_ptr + offsets_0, q_embed0, mask=offsets_0 < head_dim // 2)
    tl.store(OUTPUT_ptr + offsets_1, q_embed1, mask=offsets_1 < head_dim)


@triton.jit
def rotary_embedding_kernel_interleaved(
    Q,
    K,
    Out_q,
    Out_k,
    COS,
    SIN,
    stride_q_b: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_k_b: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_oq_b: tl.constexpr,
    stride_oq_h: tl.constexpr,
    stride_ok_b: tl.constexpr,
    stride_ok_h: tl.constexpr,
    HEAD_DIM_Q: tl.constexpr,
    HEAD_DIM_K: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Performs rotary embedding on the input tensor Q and K, and stores the results in Out_q and Out_k specified for deepseek.

    Args:
        Q (tl.tensor): The input tensor Q. Shape: [batch_seq, num_head, rotary_dim]
        K (tl.tensor): The input tensor K. Shape: [batch_seq, num_head, rotary_dim]
        Out_q (tl.tensor): The output tensor for Q.
        Out_k (tl.tensor): The output tensor for K.
    """
    cur_batch = tl.program_id(0)
    cur_block_head_id = tl.program_id(1)

    cos_ptr = COS + cur_batch * ROTARY_DIM // 2 + tl.arange(0, ROTARY_DIM // 2)
    sin_ptr = SIN + cur_batch * ROTARY_DIM // 2 + tl.arange(0, ROTARY_DIM // 2)
    cos = tl.load(cos_ptr)
    sin = tl.load(sin_ptr)

    for block_head_start in range(BLOCK_H):
        cur_head_id = cur_block_head_id * BLOCK_H + block_head_start

        if cur_head_id < HEAD_DIM_Q:
            offs_oq = (
                cur_batch * stride_oq_b
                + cur_head_id * stride_oq_h
                + tl.arange(0, ROTARY_DIM)
            )
            offs_q_0 = (
                cur_batch * stride_q_b
                + cur_head_id * stride_q_h
                + tl.arange(0, ROTARY_DIM // 2) * 2
            )
            offs_q_1 = (
                cur_batch * stride_q_b
                + cur_head_id * stride_q_h
                + tl.arange(0, ROTARY_DIM // 2) * 2
                + 1
            )
            q_0 = tl.load(Q + offs_q_0)
            q_1 = tl.load(Q + offs_q_1)
            o_q_0 = q_0 * cos - q_1 * sin
            o_q_1 = q_1 * cos + q_0 * sin
            o_q = tl.interleave(o_q_0, o_q_1)

            tl.store(Out_q + offs_oq, o_q)

        if cur_head_id < HEAD_DIM_K:
            offs_ok = (
                cur_batch * stride_ok_b
                + cur_head_id * stride_ok_h
                + tl.arange(0, ROTARY_DIM)
            )
            offs_k_0 = (
                cur_batch * stride_k_b
                + cur_head_id * stride_k_h
                + tl.arange(0, ROTARY_DIM // 2) * 2
            )
            offs_k_1 = (
                cur_batch * stride_k_b
                + cur_head_id * stride_k_h
                + tl.arange(0, ROTARY_DIM // 2) * 2
                + 1
            )
            k_0 = tl.load(K + offs_k_0)
            k_1 = tl.load(K + offs_k_1)
            o_k_0 = k_0 * cos - k_1 * sin
            o_k_1 = k_1 * cos + k_0 * sin
            o_k = tl.interleave(o_k_0, o_k_1)

            tl.store(Out_k + offs_ok, o_k)
