# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.jit
def _unpack_low_high_u4_to_i8(x_byte):
    x_u8 = x_byte.to(tl.uint8)
    low_u4 = x_u8 & 0x0F
    high_u4 = (x_u8 >> 4) & 0x0F
    low_i8 = tl.cast((tl.cast(low_u4, tl.int16) ^ 0x8) - 0x8, tl.int8)
    high_i8 = tl.cast((tl.cast(high_u4, tl.int16) ^ 0x8) - 0x8, tl.int8)
    return low_i8, high_i8


w4a4_gemm_per_token_per_channel_gemm_configs = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
        num_warps=nw,
        num_stages=ns,
    )
    for (bm, bn, bk, nw, ns) in [
        (128, 128, 128, 8, 3),
        (64, 64, 256, 4, 3),
        (128, 64, 128, 8, 2),
        (64, 128, 128, 8, 2),
        (32, 128, 128, 4, 2),
    ]
]


@triton.autotune(
    configs=w4a4_gemm_per_token_per_channel_gemm_configs, key=["M", "N", "K"]
)
@triton.jit
def w4a4_gemm_per_token_per_channel_gemm_kernel(
    AP_ptr,
    BP_ptr,
    AS_ptr,
    BS_ptr,
    C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    Kp: tl.constexpr,
    stride_apm,
    stride_apk,
    stride_bpk,
    stride_bpn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Performs a matrix multiplication between packed INT4 activations and packed
    INT4 weights, and writes FP16 output.

    Layout & math:
        - A is packed along the original K dimension into AP of shape [M, Kp],
          where Kp = ceil(K / 2). Each byte holds two signed 4-bit values:
              even column → low nibble, odd column → high nibble.
        - B is packed along K into BP of shape [Kp, N] using the same nibble
          convention (even→low4, odd→high4).
        - The kernel unpacks both AP and BP on the fly into INT8 values in
          the range [-8, 7], producing two streams per packed column:
              even_i8 and odd_i8.
        - If K is odd, the final odd nibble is invalid and is masked to zero.
        - Accumulation is done in INT32 as:
              acc = dot(a_even_i8, b_even_i8) + dot(a_odd_i8, b_odd_i8)
          reduced over the packed-K tiles.
        - Per-token (row) scales AS (shape [M]) and per-channel (col) scales BS
          (shape [N]) are loaded and combined as an outer product:
              S = AS[:, None] * BS[None, :].
          The final output is:
              C = acc.to(fp32) * S  → stored as fp16.

    Args:
        AP_ptr (tl.tensor): Pointer to packed INT4 activations, shape [M, Kp].
            Strides (stride_apm, stride_apk). Container dtype may be uint8 or int8;
            each byte packs (even→low4, odd→high4) along original K.
        BP_ptr (tl.tensor): Pointer to packed INT4 weights, shape [Kp, N].
            Strides (stride_bpk, stride_bpn). Same packing rule as AP_ptr.
        AS_ptr (tl.tensor): Pointer to per-row scales (float32), shape [M].
            One scale per token / row of A.
        BS_ptr (tl.tensor): Pointer to per-column scales (float32), shape [N].
            One scale per output channel / column of B.
        C_ptr (tl.tensor): Pointer to FP16 output, shape [M, N].
            Strides (stride_cm, stride_cn).

        M (tl.constexpr): Number of rows in A and C.
        N (tl.constexpr): Number of columns in B and C.
        K (tl.constexpr): Original (unpacked) reduction dimension.
        Kp (tl.constexpr): Packed-K = ceil(K / 2).

        stride_apm (int): Row stride of AP (elements).
        stride_apk (int): Packed-K stride of AP (elements).
        stride_bpk (int): Packed-K stride of BP (elements).
        stride_bpn (int): Column stride of BP (elements).
        stride_cm (int): Row stride of C (elements).
        stride_cn (int): Column stride of C (elements).

        BLOCK_M (tl.constexpr): Tile size along M.
        BLOCK_N (tl.constexpr): Tile size along N.
        BLOCK_K (tl.constexpr): Tile size along the packed-K dimension (in bytes),
            i.e., the number of packed columns processed per iteration.

    Notes:
        - Unpacking performs two's-complement sign extension from 4-bit to int8.
        - The odd-nibble of the last packed column is masked out when K is odd.
        - Scales are applied after accumulation to match qdq semantics:
              dequant(A) @ dequant(B) = (A_i4 * AS) @ (B_i4 * BS).
    Returns:
        None
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for kp0 in range(0, Kp, BLOCK_K):
        offs_kp = kp0 + tl.arange(0, BLOCK_K)
        mask_kp = offs_kp < Kp

        # Load packed tiles
        a_pack_ptrs = (
            AP_ptr + offs_m[:, None] * stride_apm + offs_kp[None, :] * stride_apk
        )
        b_pack_ptrs = (
            BP_ptr + offs_kp[:, None] * stride_bpk + offs_n[None, :] * stride_bpn
        )
        a_pack = tl.load(a_pack_ptrs, mask=mask_m[:, None] & mask_kp[None, :], other=0)
        b_pack = tl.load(b_pack_ptrs, mask=mask_kp[:, None] & mask_n[None, :], other=0)

        # Unpack → int8 in [-8,7]
        a_even_i8, a_odd_i8 = _unpack_low_high_u4_to_i8(a_pack)
        b_even_i8, b_odd_i8 = _unpack_low_high_u4_to_i8(b_pack)

        odd_valid = (2 * offs_kp + 1) < K
        a_odd_i8 = tl.where(odd_valid[None, :], a_odd_i8, tl.zeros_like(a_odd_i8))
        b_odd_i8 = tl.where(odd_valid[:, None], b_odd_i8, tl.zeros_like(b_odd_i8))

        # int8 dot → int32 acc
        acc += tl.dot(a_even_i8, b_even_i8, out_dtype=tl.int32)
        acc += tl.dot(a_odd_i8, b_odd_i8, out_dtype=tl.int32)

    # load scales: a_s per-row (per-token), b_s per-col (per-channel)
    a_scale = tl.load(AS_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    b_scale = tl.load(BS_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    out = acc.to(tl.float32) * (a_scale[:, None] * b_scale[None, :])
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, out.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


def w4a4_gemm_per_token_per_channel_triton(
    a_p4: torch.Tensor,
    a_s: torch.Tensor,
    b_p4: torch.Tensor,
    b_s: torch.Tensor,
    K: int,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_kp: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:

    assert a_p4.dtype in (torch.uint8, torch.int8)
    assert b_p4.dtype in (torch.uint8, torch.int8)
    assert a_p4.is_cuda and b_p4.is_cuda
    assert a_p4.dim() == 2 and b_p4.dim() == 2

    M, Kp_a = a_p4.shape
    Kp_b, N = b_p4.shape
    Kp = (K + 1) // 2
    assert (
        Kp_a == Kp and Kp_b == Kp
    ), f"packed dim mismatch: a={Kp_a}, b={Kp_b}, expect {Kp}"

    a_s = a_s.reshape(M).to(torch.float32).contiguous()
    b_s = b_s.reshape(N).to(torch.float32).contiguous()

    a_ctg = a_p4.contiguous()
    b_ctg = b_p4.contiguous()
    c = torch.empty((M, N), device=a_p4.device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    w4a4_gemm_per_token_per_channel_gemm_kernel[grid](
        a_ctg,
        b_ctg,
        a_s,
        b_s,
        c,
        M,
        N,
        K,
        Kp,
        a_ctg.stride(0),
        a_ctg.stride(1),
        b_ctg.stride(0),
        b_ctg.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c
