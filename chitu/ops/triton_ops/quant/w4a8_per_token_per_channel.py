# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools

import torch
import triton
import triton.language as tl
from triton import Config

from chitu.ops.triton_ops.utils import auto_retry_triton_compilation, auto_tuning_logger
from chitu.native_layout import Packed4BitWeightAlongK


@auto_retry_triton_compilation
def w4a8_gemm_per_token_per_channel_asymm_triton(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: Packed4BitWeightAlongK,
    b_s: torch.Tensor,
    b_z: torch.Tensor,
):
    assert isinstance(b, Packed4BitWeightAlongK)
    assert b.k_stride == 64

    assert a.is_contiguous(), "Input tensors must be contiguous"
    assert b.layout_tensor.is_contiguous(), "Input tensors must be contiguous"
    assert (
        a_s.is_contiguous() and b_s.is_contiguous()
    ), "Scaling factor tensors must be contiguous"
    assert b_z.is_contiguous(), "Zero-point tensor must be contiguous"
    K = a.size(-1)
    M = a.numel() // K
    N = b.plain_shape[0]
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    w4a8_gemm_per_token_per_channel_asymm_kernel[grid](
        a, b.layout_tensor, c, a_s, b_s, b_z, M, N, K
    )
    return c


w4a8_gemm_per_token_per_channel_asymm_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
        pre_hook=functools.partial(
            auto_tuning_logger,
            name="w4a8_gemm_per_token_per_channel_asymm",
            block_m=block_m,
            block_n=block_n,
            num_stages=num_stages,
        ),
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=w4a8_gemm_per_token_per_channel_asymm_configs, key=["N", "K"])
@triton.jit
def w4a8_gemm_per_token_per_channel_asymm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    b_z_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K // 2)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * (K // 2) + offs_k[:, None]

    a_s = tl.load(a_s_ptr + offs_m)
    b_s = tl.load(b_s_ptr + offs_n)
    b_z = tl.load(b_z_ptr + offs_n)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        b_packed = tl.load(b_ptrs)

        a = tl.load(a_ptrs)
        b = (b_packed.to(tl.uint8, bitcast=True) & 0x0F).to(tl.int8)
        accumulator += (
            tl.dot(a, b).to(tl.float32) * b_s[None, :]
            - tl.sum(a.to(tl.int32), axis=1, keep_dims=True).to(tl.float32)
            * b_z[None, :]
        ) * a_s[:, None]
        a_ptrs += BLOCK_SIZE_K // 2

        a = tl.load(a_ptrs)
        b = (b_packed.to(tl.uint8, bitcast=True) >> 4).to(tl.int8)
        accumulator += (
            tl.dot(a, b).to(tl.float32) * b_s[None, :]
            - tl.sum(a.to(tl.int32), axis=1, keep_dims=True).to(tl.float32)
            * b_z[None, :]
        ) * a_s[:, None]
        a_ptrs += BLOCK_SIZE_K // 2

        b_ptrs += BLOCK_SIZE_K // 2
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)
