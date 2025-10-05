# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from chitu.native_layout import Vector
from chitu.ops.triton_ops.utils import auto_retry_triton_compilation
from chitu.device_type import is_muxi


@auto_retry_triton_compilation
def silu_and_mul_triton(x):
    if isinstance(x, Vector):
        return Vector(
            list(x.plain_shape[:-1]) + [x.plain_shape[-1] // 2],
            silu_and_mul_triton(x.layout_tensor),
        )

    assert isinstance(x, torch.Tensor)

    n_rows = x.nelement() // x.shape[-1]
    n_cols = x.shape[-1]
    assert n_cols % 2 == 0

    output_shape = x.shape[:-1] + (n_cols // 2,)
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)

    if output.device == torch.device("meta"):
        return output

    if output.numel() == 0:
        return output

    assert x.is_contiguous()
    assert output.is_contiguous()

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

    BLOCK_SIZE, _ = calculate_settings(n_cols // 2)
    silu_and_mul_kernel[(n_rows,)](
        output,
        x,
        n_cols // 2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


silu_and_mul_configs = [
    triton.Config(
        {},
        num_warps=num_warps,
    )
    for num_warps in ([4, 8] if is_muxi() else [4, 8, 16])
]


@triton.autotune(configs=silu_and_mul_configs, key=["output_n_cols"])
@triton.jit
def silu_and_mul_kernel(output_ptr, x_ptr, output_n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * output_n_cols * 2
    offsets = tl.arange(0, BLOCK_SIZE)
    part1 = tl.load(row_start_ptr + offsets, mask=(offsets < output_n_cols), other=0)
    part2 = tl.load(
        row_start_ptr + output_n_cols + offsets, mask=(offsets < output_n_cols), other=0
    )
    part1_fp32 = part1.to(tl.float32)
    silu_part1_fp32 = part1_fp32 / (1 + tl.exp(-1 * part1_fp32))
    silu_part1 = silu_part1_fp32.to(part1.dtype)
    result = silu_part1 * part2
    output = output_ptr + row_idx * output_n_cols + offsets
    tl.store(output, result, mask=(offsets < output_n_cols))
