# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from chitu.ops.triton_ops.quant.w8a8_per_token_per_channel import (
    w8a8_gemm_per_token_per_channel_triton,
)
from chitu.ops.triton_ops.quant.w4a4_per_token_per_channel import (
    w4a4_gemm_per_token_per_channel_triton,
)


@triton.jit
def _round_half_to_even(x):
    f = tl.floor(x)
    frac = x - f
    up = frac > 0.5
    y = tl.where(up, f + 1.0, f)
    tie = frac == 0.5
    f_i = tl.cast(f, tl.int32)
    f_is_even = (f_i & 1) == 0
    tie_and_odd = tie & (~f_is_even)
    y = tl.where(tie_and_odd, f + 1.0, y)
    return y


int8_per_token_quant_configs = [
    triton.Config(
        {"BLOCK_M": block_m, "BLOCK_K": block_k},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for (block_m, block_k, num_warps, num_stages) in [
        (128, 512, 8, 3),
        (128, 256, 8, 2),
        (64, 512, 4, 2),
        (64, 256, 4, 2),
        (32, 256, 4, 2),
    ]
]


@triton.autotune(configs=int8_per_token_quant_configs, key=["M", "K"])
@triton.jit
def int8_per_token_quant_with_outliers_kernel(
    A_ptr,
    FP_MASK_ptr,
    FP_IDX_ptr,
    AQI8_ptr,
    SCALE_ptr,
    AFP_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N_FP: tl.constexpr,
    stride_am,
    stride_ak,
    stride_aqi8m,
    stride_aqi8k,
    stride_sm,
    stride_afpm,
    stride_afpk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Performs per-token symmetric INT8 quantization on rows of A while handling
    designated outlier columns. The kernel:
      1) Computes a per-row scale using the max absolute value over non-outlier
         columns only (outliers are masked out of the reduction).
      2) Quantizes non-outlier elements with round-to-nearest-even into INT8
         in the range [-127, 127]; outlier columns are written as zeros in the
         quantized output.
      3) Gathers the original (float) values of the outlier columns and stores
         them as FP16 for a parallel high-precision path.

    Args:
        A_ptr (tl.tensor): Pointer to input activations A of shape [M, K].
            Element type is floating-point (fp16/fp32); strides given by
            (stride_am, stride_ak).
        FP_MASK_ptr (tl.tensor): Pointer to a length-K uint8 mask over columns,
            where 1 marks an outlier column and 0 marks a quantized column.
        FP_IDX_ptr (tl.tensor): Pointer to a length-N_FP int32 index list of
            outlier column indices (order is preserved when extracting AFP).
        AQI8_ptr (tl.tensor): Pointer to the quantized INT8 output of shape
            [M, K] with strides (stride_aqi8m, stride_aqi8k). Outlier columns
            are written as zeros.
        SCALE_ptr (tl.tensor): Pointer to per-row scales of shape [M]
            (stored as float32) with stride stride_sm. Each scale is
            max(abs(A_row over non-outliers)) / 127, clamped to >= 1e-8.
        AFP_ptr (tl.tensor): Pointer to gathered outlier values of shape
            [M, N_FP] stored as float16 with strides (stride_afpm, stride_afpk).
            Column j in AFP corresponds to column FP_IDX_ptr[j] in A.
        M (tl.constexpr): Number of rows in A (tokens).
        K (tl.constexpr): Number of columns in A (features).
        N_FP (tl.constexpr): Number of outlier columns (len(FP_IDX_ptr)).
        stride_am (int): Row stride for A (in elements).
        stride_ak (int): Column stride for A (in elements).
        stride_aqi8m (int): Row stride for AQI8 (in elements).
        stride_aqi8k (int): Column stride for AQI8 (in elements).
        stride_sm (int): Stride for SCALE (in elements).
        stride_afpm (int): Row stride for AFP (in elements).
        stride_afpk (int): Column stride for AFP (in elements).
        BLOCK_M (tl.constexpr): Tile size along M (rows).
        BLOCK_K (tl.constexpr): Tile size along K (columns) for reductions and
            quantization.

    Notes:
        - Outlier columns do not participate in the max-abs reduction and are
          not quantized; they are set to zero in AQI8 and copied (as fp16) to AFP.
    Returns:
        None
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    qmax = 127.0
    eps = 1e-8
    neg_inf = -float("inf")

    max_abs = tl.full((BLOCK_M,), neg_inf, tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        x = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(
            tl.float32
        )

        fp_mask_tile_u8 = tl.load(FP_MASK_ptr + offs_k, mask=mask_k, other=1)
        is_outlier = fp_mask_tile_u8[None, :].to(tl.int1)

        x_abs = tl.abs(x)
        x_abs = tl.where(is_outlier, neg_inf, x_abs)

        tile_max = tl.max(x_abs, axis=1)
        max_abs = tl.maximum(max_abs, tile_max)

    scale = tl.maximum(max_abs / qmax, eps)
    tl.store(SCALE_ptr + offs_m * stride_sm, scale, mask=mask_m)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        x = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(
            tl.float32
        )

        f = tl.floor(x / scale[:, None])
        frac = x / scale[:, None] - f
        up = frac > 0.5
        q = tl.where(up, f + 1.0, f)
        tie = frac == 0.5
        f_i = tl.cast(f, tl.int32)
        f_is_even = (f_i & 1) == 0
        q = tl.where(tie & (~f_is_even), f + 1.0, q)

        q = tl.minimum(q, qmax)
        q = tl.maximum(q, -qmax)
        q_i8 = tl.cast(q, tl.int8)

        fp_mask_tile_u8 = tl.load(FP_MASK_ptr + offs_k, mask=mask_k, other=1)
        is_outlier = fp_mask_tile_u8[None, :].to(tl.int1)
        q_i8 = tl.where(is_outlier, tl.zeros_like(q_i8), q_i8)

        dst = AQI8_ptr + offs_m[:, None] * stride_aqi8m + offs_k[None, :] * stride_aqi8k
        tl.store(dst, q_i8, mask=mask_m[:, None] & mask_k[None, :])

    if N_FP > 0:
        j = 0
        while j < N_FP:
            col = tl.load(FP_IDX_ptr + j, mask=True, other=0).to(tl.int32)
            a_col_ptrs = A_ptr + offs_m[:, None] * stride_am + col * stride_ak
            v = tl.load(a_col_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
            v16 = tl.cast(v, tl.float16)
            out_ptrs = AFP_ptr + offs_m[:, None] * stride_afpm + j * stride_afpk
            tl.store(out_ptrs, v16, mask=mask_m[:, None])
            j += 1


def quant_int8_and_process_outliers(a: torch.Tensor, fp_idx: torch.Tensor):
    assert a.is_floating_point()
    device = a.device
    *prefix, K = a.shape
    M = int(torch.tensor(prefix).prod()) if prefix else 1
    a2d = a.reshape(M, K).contiguous()

    if fp_idx.numel() > 0:
        fp_idx = fp_idx.to(device=device, dtype=torch.int32).contiguous()
        n_fp = int(fp_idx.numel())
    else:
        fp_idx = torch.empty(0, device=device, dtype=torch.int32)
        n_fp = 0

    fp_mask = torch.zeros(K, dtype=torch.uint8, device=device)
    if n_fp > 0:
        fp_mask.index_fill_(0, fp_idx.to(torch.long), 1)

    a_q_i8 = torch.empty_like(a2d, dtype=torch.int8)
    a_scale = torch.empty((M,), dtype=torch.float32, device=device)
    a_fp = (
        torch.empty((M, n_fp), dtype=torch.float16, device=device)
        if n_fp > 0
        else torch.empty((M, 0), dtype=torch.float16, device=device)
    )

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    int8_per_token_quant_with_outliers_kernel[grid](
        a2d,
        fp_mask,
        fp_idx,
        a_q_i8,
        a_scale,
        a_fp,
        M,
        K,
        n_fp,
        a2d.stride(0),
        a2d.stride(1),
        a_q_i8.stride(0),
        a_q_i8.stride(1),
        a_scale.stride(0),
        a_fp.stride(0) if n_fp > 0 else 0,
        a_fp.stride(1) if n_fp > 0 else 0,
    )

    return (
        a_q_i8.reshape(*prefix, K),
        a_scale.reshape(*prefix, 1),
        a_fp.reshape(*prefix, n_fp),
    )


int4_per_token_quant_configs = [
    triton.Config(
        {"BLOCK_M": block_m, "BLOCK_K": block_k},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for (block_m, block_k, num_warps, num_stages) in [
        (128, 512, 8, 3),
        (128, 256, 8, 3),
        (64, 512, 4, 3),
        (64, 256, 4, 2),
    ]
]


@triton.autotune(configs=int4_per_token_quant_configs, key=["M", "K"])
@triton.jit
def int4_per_token_quant_pack_with_outliers_kernel(
    A_ptr,
    FP_MASK_ptr,
    FP_IDX_ptr,
    AQP4_ptr,
    SCALE_ptr,
    AFP_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N_FP: tl.constexpr,
    stride_am,
    stride_ak,
    stride_apm,
    stride_apk,
    stride_sm,
    stride_afpm,
    stride_afpk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Performs per-token symmetric INT4 quantization on rows of A, packs two 4-bit
    values into one byte, and extracts designated outlier columns for a
    high-precision bypass.

    Pipeline:
      1) Scale: For each row, compute a per-row scale using the max absolute
         value over non-outlier columns only (outliers are excluded from the
         reduction). scale = max_abs / 7, clamped to >= 1e-8.
      2) Quantize: Quantize non-outlier elements to INT4 in [-8, 7] using
         round-to-nearest-even. Outlier columns are not
         quantized and contribute zeros in the packed output.
      3) Pack: Pack two 4-bit values per byte along K: even column → low nibble,
         odd column → high nibble. If K is odd, the final odd nibble is treated
         as invalid and written as zero.
      4) Extract outliers: Gather original (float) values at outlier column
         indices and store them as FP16 in AFP for a separate FP path.

    Args:
        A_ptr (tl.tensor): Pointer to input activations A of shape [M, K]
            (fp16/fp32). Strides: (stride_am, stride_ak).
        FP_MASK_ptr (tl.tensor): Pointer to a length-K uint8 mask over columns;
            1 marks an outlier column, 0 marks a quantized column.
        FP_IDX_ptr (tl.tensor): Pointer to a length-N_FP int32 list of outlier
            column indices. The order is preserved when writing AFP.
        AQP4_ptr (tl.tensor): Pointer to packed INT4 output of shape
            [M, ceil(K/2)] stored as bytes (uint8/int8 container) with strides
            (stride_apm, stride_apk). Packing rule: (even→low4, odd→high4).
        SCALE_ptr (tl.tensor): Pointer to per-row scales of shape [M] (fp32),
            stride stride_sm. Each scale is max_abs(non-outliers)/7, clamped.
        AFP_ptr (tl.tensor): Pointer to gathered outlier values of shape
            [M, N_FP] stored as fp16, strides (stride_afpm, stride_afpk).
            Column j in AFP corresponds to column FP_IDX_ptr[j] in A.
        M (tl.constexpr): Number of rows in A.
        K (tl.constexpr): Number of columns in A.
        N_FP (tl.constexpr): Number of outlier columns (len(FP_IDX_ptr)).
        stride_am (int): Row stride for A (elements).
        stride_ak (int): Column stride for A (elements).
        stride_apm (int): Row stride for AQP4 (elements).
        stride_apk (int): Column stride for AQP4 (elements).
        stride_sm (int): Stride for SCALE (elements).
        stride_afpm (int): Row stride for AFP (elements).
        stride_afpk (int): Column stride for AFP (elements).
        BLOCK_M (tl.constexpr): Tile size along M (rows).
        BLOCK_K (tl.constexpr): Tile size along K (columns). Must be even to
            process (even, odd) column pairs during packing.

    Notes:
        - Outlier columns do not participate in the max-abs reduction and are
          not quantized; their packed nibble(s) are written as zero.
        - Packed output uses the convention even→low nibble, odd→high nibble.
        - This kernel only packs along K; consumers must interpret bytes using
          the same nibble ordering.
    Returns:
        None
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    qmax = 7.0
    qmin = -8.0
    eps = 1e-8
    neg_inf = -float("inf")

    max_abs = tl.full((BLOCK_M,), neg_inf, tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        x = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(
            tl.float32
        )

        fp_mask_tile_u8 = tl.load(FP_MASK_ptr + offs_k, mask=mask_k, other=1)
        is_outlier = fp_mask_tile_u8[None, :].to(tl.int1)

        x_abs = tl.abs(x)
        x_abs = tl.where(is_outlier, neg_inf, x_abs)
        tile_max = tl.max(x_abs, axis=1)
        max_abs = tl.maximum(max_abs, tile_max)

    scale = tl.maximum(max_abs / 7.0, eps)
    tl.store(SCALE_ptr + offs_m * stride_sm, scale, mask=mask_m)

    Kp = (K + 1) // 2
    for k0 in range(0, K, BLOCK_K):
        base_pair = k0 // 2
        pair_idx_in = tl.arange(0, BLOCK_K // 2)
        pair_out = base_pair + pair_idx_in
        mask_pair = pair_out < Kp

        offs_even = 2 * pair_out
        offs_odd = offs_even + 1
        mask_even = offs_even < K
        mask_odd = offs_odd < K

        a_even_ptrs = (
            A_ptr + offs_m[:, None] * stride_am + offs_even[None, :] * stride_ak
        )
        a_odd_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_odd[None, :] * stride_ak
        x_even = tl.load(
            a_even_ptrs, mask=mask_m[:, None] & mask_even[None, :], other=0.0
        ).to(tl.float32)
        x_odd = tl.load(
            a_odd_ptrs, mask=mask_m[:, None] & mask_odd[None, :], other=0.0
        ).to(tl.float32)

        y_even = x_even / scale[:, None]
        y_odd = x_odd / scale[:, None]
        q_even = _round_half_to_even(y_even)
        q_odd = _round_half_to_even(y_odd)
        q_even = tl.minimum(q_even, qmax)
        q_even = tl.maximum(q_even, qmin)
        q_odd = tl.minimum(q_odd, qmax)
        q_odd = tl.maximum(q_odd, qmin)
        q_even_i8 = tl.cast(q_even, tl.int8)
        q_odd_i8 = tl.cast(q_odd, tl.int8)

        fp_mask_even = tl.load(FP_MASK_ptr + offs_even, mask=mask_even, other=1)
        fp_mask_odd = tl.load(FP_MASK_ptr + offs_odd, mask=mask_odd, other=1)
        is_out_even = fp_mask_even[None, :].to(tl.int1)
        is_out_odd = fp_mask_odd[None, :].to(tl.int1)
        q_even_i8 = tl.where(is_out_even, tl.zeros_like(q_even_i8), q_even_i8)
        q_odd_i8 = tl.where(is_out_odd, tl.zeros_like(q_odd_i8), q_odd_i8)

        even_u8 = tl.cast(q_even_i8, tl.uint8) & 0x0F
        odd_u8 = tl.cast(q_odd_i8, tl.uint8) & 0x0F
        packed = (odd_u8 << 4) | even_u8

        dst = AQP4_ptr + offs_m[:, None] * stride_apm + pair_out[None, :] * stride_apk
        tl.store(dst, packed, mask=mask_m[:, None] & mask_pair[None, :])

    if N_FP > 0:
        j = 0
        while j < N_FP:
            col = tl.load(FP_IDX_ptr + j, mask=True, other=0).to(tl.int32)
            a_col_ptrs = A_ptr + offs_m[:, None] * stride_am + col * stride_ak
            v = tl.load(a_col_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
            v16 = tl.cast(v, tl.float16)
            out_ptrs = AFP_ptr + offs_m[:, None] * stride_afpm + j * stride_afpk
            tl.store(out_ptrs, v16, mask=mask_m[:, None])
            j += 1


def quant_int4_and_process_outliers(a: torch.Tensor, fp_idx: torch.Tensor):
    assert a.is_floating_point()
    device = a.device
    *prefix, K = a.shape
    M = int(torch.tensor(prefix).prod()) if prefix else 1
    a2d = a.reshape(M, K).contiguous()

    if fp_idx.numel() > 0:
        fp_idx = fp_idx.to(device=device, dtype=torch.int32).contiguous()
        n_fp = int(fp_idx.numel())
    else:
        fp_idx = torch.empty(0, device=device, dtype=torch.int32)
        n_fp = 0

    fp_mask = torch.zeros(K, dtype=torch.uint8, device=device)
    if n_fp > 0:
        fp_mask.index_fill_(0, fp_idx.to(torch.long), 1)

    K_packed = (K + 1) // 2
    a_q_p4 = torch.empty((M, K_packed), dtype=torch.uint8, device=device)
    a_scale = torch.empty((M,), dtype=torch.float32, device=device)
    a_fp = (
        torch.empty((M, n_fp), dtype=torch.float16, device=device)
        if n_fp > 0
        else torch.empty((M, 0), dtype=torch.float16, device=device)
    )

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    int4_per_token_quant_pack_with_outliers_kernel[grid](
        a2d,
        fp_mask,
        fp_idx,
        a_q_p4,
        a_scale,
        a_fp,
        M,
        K,
        n_fp,
        a2d.stride(0),
        a2d.stride(1),
        a_q_p4.stride(0),
        a_q_p4.stride(1),
        a_scale.stride(0),
        a_fp.stride(0) if n_fp > 0 else 0,
        a_fp.stride(1) if n_fp > 0 else 0,
    )

    return (
        a_q_p4.reshape(*prefix, K_packed),
        a_scale.reshape(*prefix, 1).contiguous(),
        a_fp.reshape(*prefix, n_fp),
    )


def mixq_w8a8_gemm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    b_fp: torch.Tensor,
    outliers_idx_grouped: torch.Tensor,
) -> torch.Tensor:
    assert a.is_floating_point() and b.dtype == torch.int8 and b.dim() == 2
    K, N = b.shape
    assert a.shape[-1] == K, "K mismatch between a and b"
    assert b_s.reshape(-1).numel() == N

    orig_shape = a.shape
    *prefix, _K = orig_shape
    M = int(torch.tensor(prefix).prod()) if prefix else 1
    A2D = a.reshape(M, K).contiguous()

    a_q_i8, a_s, a_fp = quant_int8_and_process_outliers(A2D, outliers_idx_grouped)

    a_s_vec = a_s.reshape(M).contiguous()
    b_s_vec = b_s.reshape(-1).contiguous()
    q_out = w8a8_gemm_per_token_per_channel_triton(
        a=a_q_i8, a_s=a_s_vec, b=b.T.contiguous(), b_s=b_s_vec
    )

    if a_fp.numel() > 0:
        assert (
            b_fp.shape[0] == a_fp.shape[1]
        ), "b_fp rows must match #outliers after reorder"
        fp_out = torch.matmul(a_fp.to(torch.float16), b_fp.to(torch.float16))
        out = q_out + fp_out
    else:
        out = q_out

    return out.reshape(*orig_shape[:-1], N)


def mixq_w4a4_gemm_triton(
    a: torch.Tensor,
    b_p4: torch.Tensor,
    b_s: torch.Tensor,
    b_fp: torch.Tensor,
    outliers_idx_grouped: torch.Tensor,
) -> torch.Tensor:

    assert a.is_floating_point(), "a must be float"
    device = a.device
    *prefix, K = a.shape
    Kp_expect = (K + 1) // 2

    assert b_p4.dtype in (torch.uint8, torch.int8) and b_p4.is_cuda
    if b_p4.dim() != 2:
        raise AssertionError("b_p4 must be 2D")
    if b_p4.shape[0] == Kp_expect:
        Bp = b_p4.contiguous()
        N = b_p4.shape[1]
    elif b_p4.shape[1] == Kp_expect:
        N = b_p4.shape[0]
        Bp = b_p4.t().contiguous()
    else:
        raise AssertionError(
            f"b_p4 shape must be [Kp,N] or [N,Kp] with Kp={Kp_expect}, got {tuple(b_p4.shape)}"
        )

    if b_fp is not None and b_fp.numel() > 0:
        if b_fp.dim() != 2:
            raise AssertionError("b_fp must be 2D when provided")
        if b_fp.shape[1] == N:
            Bfp = b_fp.contiguous()
        elif b_fp.shape[0] == N:
            Bfp = b_fp.t().contiguous()
        else:
            raise AssertionError(
                f"b_fp shape must be [N_fp,N] or [N,N_fp] with N={N}, got {tuple(b_fp.shape)}"
            )
    else:
        Bfp = torch.empty((0, N), dtype=torch.float16, device=device)

    M = int(torch.tensor(prefix).prod()) if prefix else 1
    A2D = a.reshape(M, K).contiguous()

    fp_idx = outliers_idx_grouped.to(device=device, dtype=torch.int32).contiguous()
    A_p4, A_s, A_fp = quant_int4_and_process_outliers(A2D, fp_idx)

    a_s_vec = A_s.reshape(M).contiguous()
    b_s_vec = b_s.reshape(-1).contiguous()
    assert b_s_vec.numel() == N, f"b_s length {b_s_vec.numel()} != N {N}"
    C_q = w4a4_gemm_per_token_per_channel_triton(
        a_p4=A_p4, a_s=a_s_vec, b_p4=Bp, b_s=b_s_vec, K=K
    )

    if A_fp.numel() > 0:
        assert (
            Bfp.shape[0] == A_fp.shape[1]
        ), "b_fp rows must match #outliers (same order as fp_idx)"
        C_fp = torch.matmul(A_fp.to(torch.float16), Bfp.to(torch.float16))
        C = C_q + C_fp
    else:
        C = C_q

    return C.reshape(*prefix, N)
