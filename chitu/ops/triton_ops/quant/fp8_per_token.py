# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Triton kernels for FP8 per-token dynamic quantization.

- _per_token_quant_fp8_kernel: fused absmax / scale / quantize per row
- _silu_mul_quant_fp8_kernel: fused silu_and_mul + per-token FP8 quantize
"""

import torch
import triton
import triton.language as tl

from chitu.lazy import single_dispatch_lazy_tensor


_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
_FP8_MIN = -_FP8_MAX
_FP8_EPS = 1e-12


@triton.jit
def _per_token_quant_fp8_kernel(
    x_ptr,
    xq_ptr,
    scale_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_xqm,
    stride_xqk,
    eps,
    fp8_min,
    fp8_max,
    BLOCK_K: tl.constexpr,
):
    """One program per row. Requires K <= BLOCK_K (single tile per row).

    Computes absmax / scale / quantized output in a single kernel, replacing
    the chain of 4-6 PyTorch element-wise ops in the original implementation.
    """
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    cols = tl.arange(0, BLOCK_K)
    mask = cols < K

    x_row = x_ptr + pid_m * stride_xm
    x = tl.load(x_row + cols * stride_xk, mask=mask, other=0.0).to(tl.float32)

    absmax = tl.maximum(tl.max(tl.abs(x)), eps)
    scale = absmax / fp8_max
    inv_scale = 1.0 / scale

    xq = tl.clamp(x * inv_scale, fp8_min, fp8_max).to(xq_ptr.dtype.element_ty)

    tl.store(xq_ptr + pid_m * stride_xqm + cols * stride_xqk, xq, mask=mask)
    tl.store(scale_ptr + pid_m, scale)


@triton.jit
def _silu_mul_quant_fp8_kernel(
    x_ptr,  # input (M, 2*N) bf16  (gate || up concatenated on last dim)
    xq_ptr,  # output (M, N) fp8_e4m3
    scale_ptr,  # output (M, 1) fp32
    M,
    N,
    stride_xm,
    stride_xk,
    stride_xqm,
    stride_xqk,
    eps,
    fp8_min,
    fp8_max,
    BLOCK_N: tl.constexpr,
):
    """Fused silu_and_mul + per-token FP8 quantization.

    Replaces the (silu_and_mul -> per_token_quant) two-kernel chain with one
    kernel that loads gate/up once, computes silu(gate)*up in-register,
    derives absmax/scale, and writes fp8 output. Saves one HBM round trip
    on the (M, N) intermediate plus one kernel launch.
    """
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # gate occupies the first N cols, up the next N.
    x_row = x_ptr + pid_m * stride_xm
    gate = tl.load(x_row + cols * stride_xk, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_row + (cols + N) * stride_xk, mask=mask, other=0.0).to(tl.float32)

    silu = gate / (1.0 + tl.exp(-gate))
    h = silu * up

    absmax = tl.maximum(tl.max(tl.abs(h)), eps)
    scale = absmax / fp8_max
    inv_scale = 1.0 / scale

    hq = tl.clamp(h * inv_scale, fp8_min, fp8_max).to(xq_ptr.dtype.element_ty)

    tl.store(xq_ptr + pid_m * stride_xqm + cols * stride_xqk, hq, mask=mask)
    tl.store(scale_ptr + pid_m, scale)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@single_dispatch_lazy_tensor
def per_token_quant_fp8(x: torch.Tensor):
    """Per-token dynamic FP8 quantization.

    x: (M, K) bf16/fp16 -> xq: (M, K) float8_e4m3fn, scale: (M, 1) float32

    Uses a single triton kernel that fuses absmax / scale / divide / clamp /
    cast — replaces a chain of 4-6 PyTorch element-wise ops.
    """
    x_2d = x.contiguous().view(-1, x.shape[-1])
    M, K = x_2d.shape

    xq = torch.empty_like(x_2d, dtype=torch.float8_e4m3fn)
    scale = torch.empty(M, 1, dtype=torch.float32, device=x.device)

    BLOCK_K = max(_next_pow2(K), 16)
    num_warps = 8 if BLOCK_K >= 2048 else 4

    _per_token_quant_fp8_kernel[(M,)](
        x_2d,
        xq,
        scale,
        M,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        xq.stride(0),
        xq.stride(1),
        _FP8_EPS,
        _FP8_MIN,
        _FP8_MAX,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )

    return xq.view(x.shape), scale


@single_dispatch_lazy_tensor
def silu_mul_quant_fp8(x: torch.Tensor):
    """Fused silu_and_mul + per-token FP8 quantization.

    x: (M, 2N) bf16/fp16 -> xq: (M, N) float8_e4m3fn, scale: (M, 1) float32
    Computes:  h = silu(x[..., :N]) * x[..., N:]; then per-token quant of h.
    """
    orig_shape = x.shape
    assert orig_shape[-1] % 2 == 0, "last dim must be 2N for silu_and_mul"
    N = orig_shape[-1] // 2

    x_2d = x.contiguous().view(-1, orig_shape[-1])
    M, _ = x_2d.shape

    xq = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty(M, 1, dtype=torch.float32, device=x.device)

    BLOCK_N = max(_next_pow2(N), 16)
    num_warps = 8 if BLOCK_N >= 2048 else 4

    _silu_mul_quant_fp8_kernel[(M,)](
        x_2d,
        xq,
        scale,
        M,
        N,
        x_2d.stride(0),
        x_2d.stride(1),
        xq.stride(0),
        xq.stride(1),
        _FP8_EPS,
        _FP8_MIN,
        _FP8_MAX,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    return xq.view(*orig_shape[:-1], N), scale
