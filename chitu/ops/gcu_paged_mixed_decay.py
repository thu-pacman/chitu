# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Exact paged recurrent-state read plus FP32 decay for Enflame GCU.

This module is intentionally imported only by the opt-in Qwen3-Next classic
decode path.  The narrow kernel replaces a singleton ``index_select`` followed
by a mixed BF16*FP32 multiply with one launch while preserving the established
FP32 recurrent-state boundary.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_mixed_decay_kernel(
    cache,
    pages,
    decay,
    output,
    cache_page_stride: tl.constexpr,
    cache_head_stride: tl.constexpr,
    page_batch_stride: tl.constexpr,
    decay_batch_stride: tl.constexpr,
    output_batch_stride: tl.constexpr,
    output_head_stride: tl.constexpr,
    HEADS: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    ELEMENTS_PER_HEAD: tl.constexpr,
):
    program = tl.program_id(0)
    groups_per_batch = HEADS // HEADS_PER_PROGRAM
    batch_idx = program // groups_per_batch
    group = program - batch_idx * groups_per_batch
    heads = group * HEADS_PER_PROGRAM + tl.arange(0, HEADS_PER_PROGRAM)
    offsets = tl.arange(0, ELEMENTS_PER_HEAD)
    page = tl.load(pages + batch_idx * page_batch_stride)
    value = tl.load(
        cache
        + page * cache_page_stride
        + heads[:, None] * cache_head_stride
        + offsets[None, :]
    ).to(tl.float32)
    scale = tl.load(decay + batch_idx * decay_batch_stride + heads)
    tl.store(
        output
        + batch_idx * output_batch_stride
        + heads[:, None] * output_head_stride
        + offsets[None, :],
        value * scale[:, None],
    )


def paged_mixed_decay(
    cache: torch.Tensor,
    pages: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    """Read B singleton pages and apply one FP32 decay per value head."""
    batch = pages.shape[0]
    if (
        batch not in (2, 4, 8)
        or cache.ndim != 5
        or cache.shape[0] < batch
        or tuple(cache.shape[1:]) != (1, 48, 128, 128)
        or tuple(cache.stride()[1:]) != (786432, 16384, 128, 1)
        or tuple(pages.shape) != (batch, 1)
        or tuple(pages.stride()) != (1, 1)
        or tuple(decay.shape) != (batch, 48)
        or tuple(decay.stride()) != (48, 1)
        or cache.dtype != torch.bfloat16
        or pages.dtype != torch.int32
        or decay.dtype != torch.float32
        or cache.device.type != "gcu"
        or len({cache.device, pages.device, decay.device}) != 1
        or not cache.is_contiguous()
        or not pages.is_contiguous()
        or not decay.is_contiguous()
    ):
        raise ValueError("unsupported GCU paged mixed-decay contract")

    output = torch.empty(
        (batch, 48, 128, 128),
        dtype=torch.float32,
        device=cache.device,
    )
    heads_per_program = 4
    _paged_mixed_decay_kernel[(batch * 48 // heads_per_program,)](
        cache,
        pages,
        decay,
        output,
        cache.stride(0),
        cache.stride(2),
        pages.stride(0),
        decay.stride(0),
        output.stride(0),
        output.stride(1),
        HEADS=48,
        HEADS_PER_PROGRAM=heads_per_program,
        ELEMENTS_PER_HEAD=128 * 128,
        num_warps=4,
    )
    return output
