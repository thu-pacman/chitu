# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import triton
import triton.language as tl

from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.ops.triton_ops.utils import auto_retry_triton_compilation


def apply_rotary_pos_emb_single_triton(
    x: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    out: Optional[torch.Tensor] = None,
    rotary_type: str = "separated",
) -> torch.Tensor:
    if rotary_type == "interleaved" and not hasattr(tl, "interleave"):
        raise RuntimeError(
            "triton.language.interleave is not supported, please check triton version"
        )

    x_embed = apply_rotary_pos_emb_single_triton_out_of_place(x, freqs_cis, rotary_type)
    if out is not None:
        out.copy_(x_embed)
    else:
        out = x_embed
    return out


@auto_retry_triton_compilation
def apply_rotary_pos_emb_single_triton_out_of_place(
    x: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    rotary_type: str = "separated",
    block_size=128,
) -> torch.Tensor:
    x_out = torch.empty_like(x)

    if x.numel() == 0:
        return x_out

    x_shape = x.shape
    if x.dim() == 4:
        x = x.view(-1, x_shape[-2], x_shape[-1])
        x_out = x_out.view(-1, x_shape[-2], x_shape[-1])
    elif x.dim() == 3:
        pass
    elif x.dim() == 2:
        x = x.view(-1, 1, x_shape[-1])
        x_out = x_out.view(-1, 1, x_shape[-1])
    else:
        assert False

    assert x.shape[-1] // 2 == freqs_cis.cos.shape[-1]
    assert x.shape[-1] // 2 == freqs_cis.sin.shape[-1]

    if rotary_type == "separated":
        bs, head_num, rotary_dim = x.shape

        assert freqs_cis.cos.is_contiguous()
        assert freqs_cis.sin.is_contiguous()

        grid = (bs, head_num, triton.cdiv(rotary_dim, block_size))
        rotary_embedding_single_kernel_separated[grid](
            x,
            freqs_cis.cos,
            freqs_cis.sin,
            x_out,
            head_num,
            rotary_dim,
            x.stride(0),
            x.stride(1),
            freqs_cis.cos.stride(0),
            freqs_cis.sin.stride(0),
            x_out.stride(0),
            x_out.stride(1),
            BLOCK_SIZE=block_size,
        )

    elif rotary_type == "interleaved":
        bs, head_num, rotary_dim = x.shape

        assert freqs_cis.cos.is_contiguous()
        assert freqs_cis.sin.is_contiguous()

        BLOCK_H = min(triton.cdiv(triton.next_power_of_2(bs), 128), head_num)
        grid = (bs, triton.cdiv(head_num, BLOCK_H), 1)
        rotary_embedding_single_kernel_interleaved[grid](
            x,
            x_out,
            freqs_cis.cos,
            freqs_cis.sin,
            x.stride(0),
            x.stride(1),
            x_out.stride(0),
            x_out.stride(1),
            head_num,
            rotary_dim,
            BLOCK_H,
        )

    else:
        raise NotImplementedError(
            f"Unsupported rotary type: {rotary_type} for Triton implementation"
        )

    return x_out.view(x_shape)


@triton.jit
def rotary_embedding_single_kernel_separated(
    X,
    COS,
    SIN,
    X_OUTPUT,
    head_num,
    head_dim,
    stride_x1,
    stride_x2,
    stride_cos1,
    stride_sin1,
    stride_x_out1,
    stride_x_out2,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    block_id = tl.program_id(axis=2)

    offsets_0 = block_id * (BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2)
    offsets_1 = offsets_0 + head_dim // 2

    COS_ptr = COS + batch_idx * stride_cos1
    SIN_ptr = SIN + batch_idx * stride_sin1
    cos0 = tl.load(COS_ptr + offsets_0, mask=offsets_0 < head_dim // 2)
    sin0 = tl.load(SIN_ptr + offsets_0, mask=offsets_0 < head_dim // 2)

    if head_idx < head_num:
        X_ptr = X + batch_idx * stride_x1 + head_idx * stride_x2
        X_OUTPUT_ptr = X_OUTPUT + batch_idx * stride_x_out1 + head_idx * stride_x_out2

        x0 = tl.load(X_ptr + offsets_0, mask=offsets_0 < head_dim // 2)
        x1 = tl.load(X_ptr + offsets_1, mask=offsets_1 < head_dim)

        x_embed0 = x0 * cos0 - x1 * sin0
        x_embed1 = x1 * cos0 + x0 * sin0

        tl.store(X_OUTPUT_ptr + offsets_0, x_embed0, mask=offsets_0 < head_dim // 2)
        tl.store(X_OUTPUT_ptr + offsets_1, x_embed1, mask=offsets_1 < head_dim)


@triton.jit
def rotary_embedding_single_kernel_interleaved(
    X,
    Out_x,
    COS,
    SIN,
    stride_x_b: tl.constexpr,
    stride_x_h: tl.constexpr,
    stride_ox_b: tl.constexpr,
    stride_ox_h: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_block_head_id = tl.program_id(1)

    cos_ptr = COS + cur_batch * ROTARY_DIM // 2 + tl.arange(0, ROTARY_DIM // 2)
    sin_ptr = SIN + cur_batch * ROTARY_DIM // 2 + tl.arange(0, ROTARY_DIM // 2)
    cos = tl.load(cos_ptr)
    sin = tl.load(sin_ptr)

    for block_head_start in range(BLOCK_H):
        cur_head_id = cur_block_head_id * BLOCK_H + block_head_start

        if cur_head_id < HEAD_NUM:
            offs_ox = (
                cur_batch * stride_ox_b
                + cur_head_id * stride_ox_h
                + tl.arange(0, ROTARY_DIM)
            )
            offs_x_0 = (
                cur_batch * stride_x_b
                + cur_head_id * stride_x_h
                + tl.arange(0, ROTARY_DIM // 2) * 2
            )
            offs_x_1 = (
                cur_batch * stride_x_b
                + cur_head_id * stride_x_h
                + tl.arange(0, ROTARY_DIM // 2) * 2
                + 1
            )
            x_0 = tl.load(X + offs_x_0)
            x_1 = tl.load(X + offs_x_1)
            o_x_0 = x_0 * cos - x_1 * sin
            o_x_1 = x_1 * cos + x_0 * sin
            o_x = tl.interleave(o_x_0, o_x_1)

            tl.store(Out_x + offs_ox, o_x)


def apply_rotary_pos_emb_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    q_out: Optional[torch.Tensor] = None,
    k_out: Optional[torch.Tensor] = None,
    rotary_type: str = "separated",
) -> tuple[torch.Tensor, torch.Tensor]:
    if rotary_type == "interleaved" and not hasattr(tl, "interleave"):
        raise RuntimeError(
            "triton.language.interleave is not supported, please check triton version"
        )

    # Triton does not support in-place operation. This function is only a compatitive
    # adaptor for in-place interface, but not for performance.
    q_embed, k_embed = apply_rotary_pos_emb_triton_out_of_place(
        q, k, freqs_cis, rotary_type
    )
    if q_out is not None:
        q_out.copy_(q_embed)
    else:
        q_out = q_embed
    if k_out is not None:
        k_out.copy_(k_embed)
    else:
        k_out = k_embed
    return q_out, k_out


@auto_retry_triton_compilation
def apply_rotary_pos_emb_triton_out_of_place(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    rotary_type: str = "separated",
    block_size=128,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Prepare output tensor
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    if q.numel() == 0 or k.numel() == 0:
        assert q.numel() == 0
        assert k.numel() == 0
        return q_out, k_out

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
    assert q.shape[-1] // 2 == freqs_cis.cos.shape[-1]
    assert q.shape[-1] // 2 == freqs_cis.sin.shape[-1]

    if rotary_type == "separated":
        # "separated" has an [real, real, ..., real, imag, imag, ..., imag] layout.

        # Get tensor shapes
        bs, head_num_q, rotary_dim = q.shape
        bs, head_num_k, rotary_dim = k.shape

        assert freqs_cis.cos.is_contiguous()
        assert freqs_cis.sin.is_contiguous()

        # Launch kernel
        grid = (bs, max(head_num_q, head_num_k), triton.cdiv(rotary_dim, block_size))
        rotary_embedding_kernel_separated[grid](
            q,
            k,
            freqs_cis.cos,
            freqs_cis.sin,
            q_out,
            k_out,
            head_num_q,
            head_num_k,
            rotary_dim,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            freqs_cis.cos.stride(0),
            freqs_cis.sin.stride(0),
            q_out.stride(0),
            q_out.stride(1),
            k_out.stride(0),
            k_out.stride(1),
            BLOCK_SIZE=block_size,
        )

    elif rotary_type == "interleaved":
        # "interleaved" has an [real, imag, real, imag, ..., real, imag] layout.

        bs, head_num_q, rotary_dim = q.shape
        bs, head_num_k, rotary_dim = k.shape

        assert freqs_cis.cos.is_contiguous()
        assert freqs_cis.sin.is_contiguous()

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
            freqs_cis.cos,
            freqs_cis.sin,
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
    K,
    COS,
    SIN,
    Q_OUTPUT,
    K_OUTPUT,
    q_num_head,
    k_num_head,
    head_dim,
    stride_q1,
    stride_q2,
    stride_k1,
    stride_k2,
    stride_cos1,
    stride_sin1,
    stride_q_out1,
    stride_q_out2,
    stride_k_out1,
    stride_k_out2,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Create block IDs
    block_id = tl.program_id(axis=2)

    # Create offsets for reading and writing
    offsets_0 = block_id * (BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2)
    offsets_1 = offsets_0 + head_dim // 2

    # Load cos and sin (shared for both Q and K)
    COS_ptr = COS + batch_idx * stride_cos1
    SIN_ptr = SIN + batch_idx * stride_sin1
    cos0 = tl.load(COS_ptr + offsets_0, mask=offsets_0 < head_dim // 2)
    sin0 = tl.load(SIN_ptr + offsets_0, mask=offsets_0 < head_dim // 2)

    # NOTE: q_num_head and k_num_head can be different.
    if head_idx < q_num_head:
        Q_ptr = Q + batch_idx * stride_q1 + head_idx * stride_q2
        Q_OUTPUT_ptr = Q_OUTPUT + batch_idx * stride_q_out1 + head_idx * stride_q_out2

        q0 = tl.load(Q_ptr + offsets_0, mask=offsets_0 < head_dim // 2)
        q1 = tl.load(Q_ptr + offsets_1, mask=offsets_1 < head_dim)

        # Apply rotary embedding
        q_embed0 = q0 * cos0 - q1 * sin0
        q_embed1 = q1 * cos0 + q0 * sin0

        # Store result
        tl.store(Q_OUTPUT_ptr + offsets_0, q_embed0, mask=offsets_0 < head_dim // 2)
        tl.store(Q_OUTPUT_ptr + offsets_1, q_embed1, mask=offsets_1 < head_dim)

    if head_idx < k_num_head:
        K_ptr = K + batch_idx * stride_k1 + head_idx * stride_k2
        K_OUTPUT_ptr = K_OUTPUT + batch_idx * stride_k_out1 + head_idx * stride_k_out2

        k0 = tl.load(K_ptr + offsets_0, mask=offsets_0 < head_dim // 2)
        k1 = tl.load(K_ptr + offsets_1, mask=offsets_1 < head_dim)

        # Apply rotary embedding
        k_embed0 = k0 * cos0 - k1 * sin0
        k_embed1 = k1 * cos0 + k0 * sin0

        # Store result
        tl.store(K_OUTPUT_ptr + offsets_0, k_embed0, mask=offsets_0 < head_dim // 2)
        tl.store(K_OUTPUT_ptr + offsets_1, k_embed1, mask=offsets_1 < head_dim)


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
