# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from chitu.ops.triton_ops.utils import auto_retry_triton_compilation
from chitu.device_list import DeviceList


@auto_retry_triton_compilation
def apply_frequency_penalty_triton(
    logits: torch.Tensor,
    logits_index: torch.Tensor,
    response_list: list[DeviceList],
    response_len_list: torch.Tensor,
    frequency_penalty: torch.Tensor,
):
    """
    使用Triton实现的频率惩罚函数
    参数:
        logits: 形状为 [batch, vocab] 的logits张量
        logits_index: 需要更新的行索引
        response: 已生成的token序列
        frequency_penalty: 频率惩罚系数
    """
    assert logits.is_contiguous()
    assert logits_index.is_contiguous()
    assert frequency_penalty.is_contiguous()
    assert response_len_list.is_contiguous()
    vocab_size = logits.size(-1)

    grid = lambda meta: (meta["batch_size"], meta["num_threads"])
    batch_size = logits_index.shape[0]

    max_len = response_len_list[0]
    for i in range(1, batch_size):
        max_len = max(max_len, response_len_list[i])
    response = torch.empty(
        (batch_size, max_len),
        dtype=response_list[0].to_tensor().dtype,
        device=response_list[0].to_tensor().device,
    )
    for i in range(batch_size):
        response[i, : response_len_list[i]] = response_list[i].to_tensor()
    apply_frequency_penalty_kernel[grid](
        logits_ptr=logits,
        logits_index_ptr=logits_index,
        response_ptr=response,
        response_len_list=response_len_list,
        frequency_penalty_list=frequency_penalty,
        logits_row_stride=logits.stride(0),
        logits_col_stride=logits.stride(1),
        response_row_stride=response.stride(0),
        vocab_size=vocab_size,
        batch_size=batch_size,
        num_threads=256,
    )


@triton.jit
def apply_frequency_penalty_kernel(
    logits_ptr,
    logits_index_ptr,
    response_ptr,
    logits_row_stride: tl.constexpr,
    logits_col_stride: tl.constexpr,
    response_row_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    response_len_list,
    frequency_penalty_list,
    batch_size: tl.constexpr,  # Number of elements in logits_index
    num_threads: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    thread_id = tl.program_id(axis=1)

    logits_row = tl.load(logits_index_ptr + pid)
    row_start = logits_row * logits_row_stride
    response_len = tl.load(response_len_list + pid)
    frequency_penalty = tl.load(frequency_penalty_list + pid)

    for token_pos in range(thread_id, response_len, num_threads):
        token_id = tl.load(response_ptr + pid * response_row_stride + token_pos)
        logits_pos = row_start + token_id * logits_col_stride
        tl.atomic_add(
            logits_ptr + logits_pos, -frequency_penalty, token_id < vocab_size
        )
