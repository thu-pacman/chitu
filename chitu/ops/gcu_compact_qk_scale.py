# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Exact compact Q/K post-norm scaling for Qwen3-Next on Enflame GCU.

The fixed decode path has two BF16 RMSNorm outputs.  This kernel preserves the
existing BF16 result of their first scalar scale, widens that rounded value to
FP32, and applies the query-only second FP32 scale.  Combining the four small
framework nodes removes launch overhead without changing either downstream
vendor reduction.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _compact_qk_scale_kernel(
    query,
    key,
    query_scale,
    query_out,
    key_out,
    element_count: tl.constexpr,
    inv_scale: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < element_count
    query_value = tl.load(query + offsets, mask=mask, other=0.0)
    key_value = tl.load(key + offsets, mask=mask, other=0.0)

    # A Python scalar multiply keeps the source tensors in BF16, but performs
    # the arithmetic with the FP32 scalar before the BF16 output rounding.
    # Keep that boundary explicit: it is required for bitwise equivalence.
    scale = tl.full((block_size,), inv_scale, tl.float32)
    query_first = (query_value.to(tl.float32) * scale).to(tl.bfloat16)
    key_first = (key_value.to(tl.float32) * scale).to(tl.bfloat16)
    fp32_query_scale = tl.load(query_scale)
    tl.store(
        query_out + offsets,
        query_first.to(tl.float32) * fp32_query_scale,
        mask=mask,
    )
    tl.store(key_out + offsets, key_first.to(tl.float32), mask=mask)


def compact_qk_scale(
    query: torch.Tensor,
    key: torch.Tensor,
    query_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale fixed [B,1,16,128] compact Q/K tensors with exact boundaries."""
    batch = query.shape[0]
    if (
        batch not in (2, 4, 8)
        or tuple(query.shape) != (batch, 1, 16, 128)
        or key.shape != query.shape
        or tuple(query_scale.shape) != (1,)
        or query.dtype != torch.bfloat16
        or key.dtype != torch.bfloat16
        or query_scale.dtype != torch.float32
        or query.device.type != "gcu"
        or len({query.device, key.device, query_scale.device}) != 1
        or not query.is_contiguous()
        or not key.is_contiguous()
        or not query_scale.is_contiguous()
    ):
        raise ValueError("unsupported GCU compact Q/K scale contract")

    output_shape = (batch, 16, 128)
    query_out = torch.empty(output_shape, dtype=torch.float32, device=query.device)
    key_out = torch.empty(output_shape, dtype=torch.float32, device=key.device)
    element_count = query.numel()
    block_size = 1024
    _compact_qk_scale_kernel[(triton.cdiv(element_count, block_size),)](
        query,
        key,
        query_scale,
        query_out,
        key_out,
        element_count=element_count,
        inv_scale=128**-0.5,
        block_size=block_size,
        num_warps=4,
    )
    return query_out, key_out
