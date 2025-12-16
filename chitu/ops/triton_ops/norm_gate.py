# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Optional
import triton
import triton.language as tl
from chitu.ops.triton_ops.utils import to_triton_dtype
import os
from chitu.device_type import is_muxi


def rms_norm_gate_triton(
    X: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: Optional[torch.Tensor],
    compute_dtype: torch.dtype,
):
    num_cols = X.shape[-1]
    num_rows = X.numel() // num_cols

    if out is None:
        out = torch.empty_like(X)
    X = X.view(num_rows, num_cols)
    out = out.view(num_rows, num_cols)
    gate = gate.view(num_rows, num_cols)
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

    rms_norm_gate_kernel[(num_rows,)](
        X,
        X.stride(-2),
        gate,
        gate.stride(-2),
        weight,
        eps,
        out,
        out.stride(-2),
        num_cols,
        compute_dtype=to_triton_dtype(compute_dtype),
        input_dtype=to_triton_dtype(X.dtype),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.to(X.dtype)


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


@triton.autotune(
    configs=rms_norm_configs,
    key=[
        "Y_row_stride",
        "X_row_stride",
        "G_row_stride",
        "compute_dtype",
        "input_dtype",
    ],
)
@triton.jit
def rms_norm_gate_kernel(
    X_ptr,
    X_row_stride: tl.constexpr,
    G_ptr,
    G_row_stride: tl.constexpr,
    W_ptr,
    eps: tl.constexpr,
    Y_ptr,
    Y_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    compute_dtype: tl.constexpr,
    input_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # load
    row_X = tl.load(
        X_ptr + row_idx * X_row_stride + col_offsets, mask=mask, other=0.0
    ).to(compute_dtype)
    row_W = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    row_G = tl.load(
        G_ptr + row_idx * G_row_stride + col_offsets, mask=mask, other=0.0
    ).to(compute_dtype)

    # rmsnorm
    sq = row_X * row_X
    var = tl.sum(sq) / n_cols
    inv_var = tl.math.rsqrt(var + eps)

    normd_X = row_X * inv_var

    # apply weight
    weighted_X = row_W * normd_X.to(input_dtype)

    # silu and gate
    sig_G = tl.sigmoid(row_G)
    silu_G = row_G * sig_G

    out = weighted_X * silu_G
    out = out.to(input_dtype)

    tl.store(Y_ptr + row_idx * Y_row_stride + col_offsets, out, mask=mask)
