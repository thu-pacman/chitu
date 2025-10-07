# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import struct
import functools

import torch
import triton
import triton.language as tl
from triton import Config

from chitu.device_type import is_muxi
from chitu.ops.triton_ops.utils import (
    auto_retry_triton_compilation,
    auto_tuning_logger,
    SIGNED_INT32_0x87F00000,
)


@auto_retry_triton_compilation
def blockfp8_einsum_shc_hdc_shd_triton(
    group_A: torch.Tensor,
    group_B: torch.Tensor,
    group_b_s: torch.Tensor,
    *,
    group_n: int = 128,
    group_k: int = 128,
    soft_fp8: bool = False,
):
    assert group_B.shape[1] == group_b_s.shape[1] * group_k
    assert group_B.shape[2] == group_b_s.shape[2] * group_n
    s, h, c, d = (
        group_A.shape[0],
        group_A.shape[1],
        group_A.shape[2],
        group_B.shape[1],
    )
    group_size = h
    M = s
    K = c
    N = d
    stride_A_group, stride_A_m = group_A.stride()[1], group_A.stride()[0]
    stride_B_group, stride_B_1 = group_B.stride()[0], group_B.stride()[1]
    stride_C_group, stride_C_m = d, h * d
    assert group_b_s.is_contiguous()
    group_C = torch.empty((s, h, d), dtype=group_A.dtype, device=group_A.device)

    if soft_fp8:
        fp8_to_fp32_scale = struct.unpack(">f", bytes.fromhex("7b800000"))[0]
    else:
        fp8_to_fp32_scale = None

    grid = lambda META: (
        group_size,
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    blockfp8_einsum_shc_hdc_shd_kernel[grid](
        group_A,
        group_B,
        group_b_s,
        group_C,
        M,
        K,
        N,
        stride_A_group,
        stride_A_m,
        stride_B_group,
        stride_B_1,
        stride_C_group,
        stride_C_m,
        group_n,
        group_k,
        fp8_to_fp32_scale=fp8_to_fp32_scale,
    )

    return group_C


def blockfp8_einsum_shc_hdc_shd_config_filter(*, block_m, block_n, num_stages):
    # Work around some bugs that not only make a config invalid, and even crashes the program
    if is_muxi():
        if num_stages > 1:
            # Reproduce the bug on image mxc500-torch2.1-py310:mc2.29.0.7-ubuntu22.04-amd64 on host mx-oam-181
            return False
    return True


blockfp8_einsum_shc_hdc_shd_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
        pre_hook=functools.partial(
            auto_tuning_logger,
            name="blockfp8_einsum_shc_hdc_shd_kernel",
            block_m=block_m,
            block_n=block_n,
            num_stages=num_stages,
        ),
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [1, 3, 5]
    if blockfp8_einsum_shc_hdc_shd_config_filter(
        block_m=block_m, block_n=block_n, num_stages=num_stages
    )
]


@triton.autotune(
    configs=blockfp8_einsum_shc_hdc_shd_configs, key=["N", "K", "fp8_to_fp32_scale"]
)
@triton.jit
def blockfp8_einsum_shc_hdc_shd_kernel(
    # Pointers:
    group_a_ptrs,
    group_b_ptrs,
    group_b_s_ptrs,
    group_c_ptrs,
    # Shapes:
    M,  # Sequence length
    K: tl.constexpr,
    N: tl.constexpr,
    stride_a_group: tl.constexpr,
    stride_a_m: tl.constexpr,
    stride_b_group: tl.constexpr,
    stride_b_1: tl.constexpr,
    stride_c_group: tl.constexpr,
    stride_c_m: tl.constexpr,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Soft fp8:
    fp8_to_fp32_scale: tl.constexpr,
    # Tunable parameters:
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Performs group matrix multiplication operation on matrices.

    Args:
        group_a_ptrs (tl.pointer): Pointer to the lhs.
        group_b_ptrs (tl.pointer): Pointer to the rhs.
        group_b_s_ptrs (tl.pointer): Pointer to the scaling factors of rhs.
        group_c_ptrs (tl.pointer): output buffer for dequantized group matrix multiplication.
    """

    pid_g = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    scale_k = tl.cdiv(K, group_k)
    scale_n = tl.cdiv(N, group_n)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (
        group_a_ptrs
        + pid_g * stride_a_group
        + offs_m[:, None] * stride_a_m
        + offs_k[None, :]
    )
    b_ptrs = (
        group_b_ptrs
        + pid_g * stride_b_group
        + offs_n[None, :] * stride_b_1
        + offs_k[:, None]
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        b_s = tl.load(
            group_b_s_ptrs
            + pid_g * scale_k * scale_n
            + pid_n * BLOCK_SIZE_N // group_n * scale_k
            + i * BLOCK_SIZE_K // group_k
        )

        if fp8_to_fp32_scale is None:
            accumulator += tl.dot(a, b.to(a.dtype)) * b_s
        else:
            t = b.to(tl.int8, bitcast=True).to(
                tl.int32
            )  # Do signed cast to copy the sign bit
            t = (t << 20) & SIGNED_INT32_0x87F00000
            b_unscaled_fp32 = t.to(tl.float32, bitcast=True)
            b_new_scale = b_s * fp8_to_fp32_scale
            b_scaled_fp32 = b_unscaled_fp32 * b_new_scale
            b_scaled_fp32 = b_scaled_fp32.to(dtype=a.dtype)
            accumulator += tl.dot(a, b_scaled_fp32)

        b_ptrs += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K
    c = accumulator.to(group_c_ptrs.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (
        group_c_ptrs
        + pid_g * stride_c_group
        + offs_m[:, None] * stride_c_m
        + offs_n[None, :]
    )
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)
