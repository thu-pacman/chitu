# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import triton
import triton.language as tl

from chitu.ops.triton_ops.utils import (
    auto_retry_triton_compilation,
    to_triton_dtype,
    autotune_compat,
)
from chitu.device_type import is_muxi
from chitu.ops.utils import compatible_with_inplace


@compatible_with_inplace
@auto_retry_triton_compilation
def rms_norm_triton(
    x: torch.Tensor, weight: torch.Tensor, *, eps, compute_dtype: torch.dtype
):
    out = torch.empty_like(x)
    if x.numel() == 0:
        return out

    x_shape = x.shape
    num_cols = x.shape[-1]
    if x.ndim >= 3:
        num_heads = x.shape[-2]
        num_seqs = x.numel() // (num_cols * num_heads)
    else:
        num_heads = 1
        num_seqs = x.numel() // num_cols

    # Assume the batch dimensions are contiguous, but it can be non-contiguous
    # in the last two dimensions.
    x = x.view(num_seqs, num_heads, num_cols)
    out = out.view(num_seqs, num_heads, num_cols)

    assert weight.is_contiguous()

    # SPDX-SnippetBegin
    # SPDX-License-Identifier: Apache-2.0
    # SPDX-SnippetCopyrightText: 2025 unslothai
    # SDPX—SnippetName: calculate_settings from unsloth
    def calculate_settings(n):
        # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

        MAX_FUSED_SIZE = 65536
        BLOCK_SIZE = triton.next_power_of_2(n)
        if BLOCK_SIZE > MAX_FUSED_SIZE:
            raise RuntimeError(
                f"Cannot launch Triton kernel since n = {n} exceeds "
                f"the recommended Triton blocksize = {MAX_FUSED_SIZE}."
            )

        num_warps = 4
        if BLOCK_SIZE >= 32768:
            num_warps = 32
        elif BLOCK_SIZE >= 8192:
            num_warps = 16
        elif BLOCK_SIZE >= 2048:
            num_warps = 8
        return BLOCK_SIZE, num_warps

    # SPDX-SnippetEnd

    BLOCK_SIZE, num_warps = calculate_settings(num_cols)
    rms_norm_kernel[num_seqs, num_heads](
        out,
        out.stride(-3),
        out.stride(-2),
        x,
        x.stride(-3),
        x.stride(-2),
        weight,
        num_cols,
        eps,
        compute_dtype=to_triton_dtype(compute_dtype),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.view(x_shape)


rms_norm_configs = [
    triton.Config(
        {},
        num_warps=num_warps,
    )
    for num_warps in ([4, 8] if is_muxi() else [4, 8, 16])
]

if os.environ.get("CI_TESTS", "false") == "true":
    rms_norm_configs = [
        triton.Config(
            {},
            num_warps=8,
        )
    ]


@autotune_compat(
    configs=rms_norm_configs,
    key=[
        "Y_seq_stride",
        "Y_head_stride",
        "X_seq_stride",
        "X_head_stride",
        "compute_dtype",
    ],
)
@triton.jit
def rms_norm_kernel(
    Y,
    Y_seq_stride: tl.constexpr,
    Y_head_stride: tl.constexpr,
    X,
    X_seq_stride: tl.constexpr,
    X_head_stride: tl.constexpr,
    W,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    compute_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fast RMS Layernorm kernel
    Inspiration from a Triton tutorial:
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """

    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += seq_idx * Y_seq_stride + head_idx * Y_head_stride
    X += seq_idx * X_seq_stride + head_idx * X_head_stride

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(compute_dtype)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)

    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype)  # Be consistent with impl="ref"
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask=mask)
