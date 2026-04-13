# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from chitu.ops.triton_ops.utils import auto_retry_triton_compilation


@auto_retry_triton_compilation
def apply_frequency_penalty_triton(
    logits: torch.Tensor,
    indices: torch.Tensor,
    blocks: torch.Tensor,
    sizes: torch.Tensor,
    penalties: torch.Tensor,
):
    """
    使用Triton实现的频率惩罚函数
    参数:
        logits: (bs, VOCAB_SIZE) 的logits张量
        indices: (n,) 需要更新的行索引
        blocks: (n, BLOCK_SIZE) 已生成的token
        sizes: (n,) blocks中每行的有效长度
        penalties: (n,) 频率惩罚系数
    """
    assert logits.is_contiguous()
    assert indices.is_contiguous()
    assert penalties.is_contiguous()
    assert sizes.is_contiguous()
    vocab_size = logits.size(-1)
    grid = lambda meta: (meta["batch_size"], meta["num_threads"])
    batch_size = indices.shape[0]
    apply_frequency_penalty_kernel[grid](
        logits_ptr=logits,
        indices_ptr=indices,
        blocks_ptr=blocks,
        sizes_ptr=sizes,
        penalties_ptr=penalties,
        logits_row_stride=logits.stride(0),
        logits_col_stride=logits.stride(1),
        blocks_row_stride=blocks.stride(0),
        vocab_size=vocab_size,
        batch_size=batch_size,
        num_threads=256,
    )


@triton.jit
def apply_frequency_penalty_kernel(
    logits_ptr,
    indices_ptr,
    blocks_ptr,
    sizes_ptr,
    penalties_ptr,
    logits_row_stride: tl.constexpr,
    logits_col_stride: tl.constexpr,
    blocks_row_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    batch_size: tl.constexpr,  # Number of elements in indices/blocks/sizes/penalties
    num_threads: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    thread_id = tl.program_id(axis=1)

    logits_row = tl.load(indices_ptr + pid)
    row_start = logits_row * logits_row_stride
    size = tl.load(sizes_ptr + pid)
    penalty = tl.load(penalties_ptr + pid)

    for token_pos in range(thread_id, size, num_threads):
        token_id = tl.load(blocks_ptr + pid * blocks_row_stride + token_pos)
        logits_pos = row_start + token_id * logits_col_stride
        tl.atomic_add(logits_ptr + logits_pos, -penalty, token_id < vocab_size)
