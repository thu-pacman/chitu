__all__ = [
    "append_to_paged_kv_cache_kernel",
    "rotary_embedding_kernel_hf_llama",
    "rotary_embedding_kernel_llama",
    "act_quant_deepseek_v3_kernel",
    "weight_dequant_deepseek_v3_kernel",
    "weight_dequant_soft_fp8_deepseek_v3_kernel_step_1",
    "weight_dequant_soft_fp8_deepseek_v3_kernel_step_2",
    "fp8_gemm_deepseek_v3_kernel",
    "soft_fp8_gemm_deepseek_v3_kernel",
]

import triton
import triton.language as tl
from triton import Config


@triton.jit
def append_to_paged_kv_cache_kernel(
    kv_cache_ptr,  # (num_pages, page_size, other dims...)
    page_table_ptr,  # (batch_size, num_pages_per_sample)
    this_kv_ptr,  # (batch_size, other dims...)
    old_seq_lens_ptr,  # (batch_size,)
    PAGE_SIZE: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    NUM_PAGES_PER_SAMPLE: tl.constexpr,
    TOT_LEN_OF_OTHER_DIMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # GPU block size, not page size
):
    batch_id = tl.program_id(axis=0)
    dim_id_0 = tl.program_id(axis=1)
    dim_id_1 = tl.arange(0, BLOCK_SIZE)
    dim_id = dim_id_0 * BLOCK_SIZE + dim_id_1
    dim_mask = dim_id < TOT_LEN_OF_OTHER_DIMS

    seqlen = tl.load(old_seq_lens_ptr + batch_id)

    page_table_offset = batch_id * NUM_PAGES_PER_SAMPLE + seqlen // 64
    page_id = tl.load(page_table_ptr + page_table_offset)

    kv_cache_offset = (
        page_id * PAGE_SIZE + seqlen % 64
    ) * TOT_LEN_OF_OTHER_DIMS + dim_id

    this_kv_offset = batch_id * TOT_LEN_OF_OTHER_DIMS + dim_id

    this_kv_data = tl.load(this_kv_ptr + this_kv_offset, mask=dim_mask)
    tl.store(kv_cache_ptr + kv_cache_offset, this_kv_data, mask=dim_mask)


@triton.jit
def rotary_embedding_kernel_hf_llama(
    Q,
    COS,
    SIN,
    OUTPUT,
    num_head,
    stride_q1,
    stride_q2,
    stride_cos1,
    stride_sin1,
    stride_out1,
    stride_out2,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(axis=0)

    # Compute batch index and head index
    batch_idx = pid // num_head
    head_idx = pid % num_head

    # Pointers to the beginning of the Q, COS, SIN, and OUTPUT
    Q_ptr = Q + batch_idx * stride_q1 + head_idx * stride_q2
    COS_ptr = COS + batch_idx * stride_cos1
    SIN_ptr = SIN + batch_idx * stride_sin1
    OUTPUT_ptr = OUTPUT + batch_idx * stride_out1 + head_idx * stride_out2

    # Create block IDs
    block_id = tl.program_id(axis=1)

    # Create offsets for reading and writing
    offsets_0 = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE // 2)
    offsets_1 = block_id * BLOCK_SIZE + BLOCK_SIZE // 2 + tl.arange(0, BLOCK_SIZE // 2)

    # Load data
    cos0 = tl.load(COS_ptr + offsets_0)
    sin0 = tl.load(SIN_ptr + offsets_0)
    q0 = tl.load(Q_ptr + offsets_0)
    q1 = tl.load(Q_ptr + offsets_1)

    # Apply rotary embedding
    q_embed0 = q0 * cos0 - q1 * sin0
    q_embed1 = q1 * cos0 + q0 * sin0

    # Store result
    tl.store(OUTPUT_ptr + offsets_0, q_embed0)
    tl.store(OUTPUT_ptr + offsets_1, q_embed1)


@triton.jit
def rotary_embedding_kernel_llama(
    Q,
    K,
    Out_q,
    Out_k,
    COS,
    SIN,
    stride_q_b: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_k_b: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_oq_b: tl.constexpr,
    stride_oq_h: tl.constexpr,
    stride_ok_b: tl.constexpr,
    stride_ok_h: tl.constexpr,
    HEAD_DIM_Q: tl.constexpr,
    HEAD_DIM_K: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Performs rotary embedding on the input tensor Q and K, and stores the results in Out_q and Out_k specified for deepseek.

    Args:
        Q (tl.tensor): The input tensor Q. Shape: [batch_seq, num_head, rotary_dim]
        K (tl.tensor): The input tensor K. Shape: [batch_seq, num_head, rotary_dim]
        Out_q (tl.tensor): The output tensor for Q.
        Out_k (tl.tensor): The output tensor for K.
    """
    cur_batch = tl.program_id(0)
    cur_block_head_id = tl.program_id(1)

    cos_ptr = COS + cur_batch * ROTARY_DIM // 2 + tl.arange(0, ROTARY_DIM // 2)
    sin_ptr = SIN + cur_batch * ROTARY_DIM // 2 + tl.arange(0, ROTARY_DIM // 2)
    cos = tl.load(cos_ptr)
    sin = tl.load(sin_ptr)

    for block_head_start in range(BLOCK_H):
        cur_head_id = cur_block_head_id * BLOCK_H + block_head_start

        if cur_head_id < HEAD_DIM_Q:
            offs_oq = (
                cur_batch * stride_oq_b
                + cur_head_id * stride_oq_h
                + tl.arange(0, ROTARY_DIM)
            )
            offs_q_0 = (
                cur_batch * stride_q_b
                + cur_head_id * stride_q_h
                + tl.arange(0, ROTARY_DIM // 2) * 2
            )
            offs_q_1 = (
                cur_batch * stride_q_b
                + cur_head_id * stride_q_h
                + tl.arange(0, ROTARY_DIM // 2) * 2
                + 1
            )
            q_0 = tl.load(Q + offs_q_0)
            q_1 = tl.load(Q + offs_q_1)
            o_q_0 = q_0 * cos - q_1 * sin
            o_q_1 = q_1 * cos + q_0 * sin
            o_q = tl.interleave(o_q_0, o_q_1)

            tl.store(Out_q + offs_oq, o_q)

        if cur_head_id < HEAD_DIM_K:
            offs_ok = (
                cur_batch * stride_ok_b
                + cur_head_id * stride_ok_h
                + tl.arange(0, ROTARY_DIM)
            )
            offs_k_0 = (
                cur_batch * stride_k_b
                + cur_head_id * stride_k_h
                + tl.arange(0, ROTARY_DIM // 2) * 2
            )
            offs_k_1 = (
                cur_batch * stride_k_b
                + cur_head_id * stride_k_h
                + tl.arange(0, ROTARY_DIM // 2) * 2
                + 1
            )
            k_0 = tl.load(K + offs_k_0)
            k_1 = tl.load(K + offs_k_1)
            o_k_0 = k_0 * cos - k_1 * sin
            o_k_1 = k_1 * cos + k_0 * sin
            o_k = tl.interleave(o_k_0, o_k_1)

            tl.store(Out_k + offs_ok, o_k)


@triton.jit
def act_quant_deepseek_v3_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


@triton.jit
def weight_dequant_deepseek_v3_kernel(
    x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr
):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    n = tl.cdiv(N, BLOCK_SIZE)
    m = tl.cdiv(M, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = pid_b * M * N + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_b * m * n + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def weight_dequant_soft_fp8_deepseek_v3_kernel_step_1(
    x_ptr,  # fp8 as uint8
    y_ptr,  # fp32 as uint32
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask).to(tl.uint32)
    y = ((x & 0x80) << 24) | ((x & 0x7F) << 20)
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def weight_dequant_soft_fp8_deepseek_v3_kernel_step_2(
    x_ptr,  # fp32
    s_ptr,  # fp32
    y_ptr,  # bf16
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    fp8_to_fp32_scale: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    n = tl.cdiv(N, BLOCK_SIZE)
    m = tl.cdiv(M, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = pid_b * M * N + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask)
    s = tl.load(s_ptr + pid_b * m * n + pid_m * n + pid_n)
    y = x * (s * fp8_to_fp32_scale)
    tl.store(y_ptr + offs, y, mask=mask)


fp8_gemm_deepseek_v3_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=fp8_gemm_deepseek_v3_configs, key=["N", "K"])
@triton.jit
def fp8_gemm_deepseek_v3_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        group_n (tl.constexpr): Quantization group size for the N dimension.
        group_k (tl.constexpr): Quantization group size for the K dimension.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // group_n) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs + i * BLOCK_SIZE_K // group_k)
        b_s = tl.load(b_s_ptrs + i * BLOCK_SIZE_K // group_k)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


soft_fp8_gemm_deepseek_v3_configs = [
    Config(
        {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_m,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for block_k in [128]
    for group_m in [1, 32]
    for num_stages in [3, 4, 5, 6]
    for num_warps in [4, 8]
]


@triton.autotune(configs=soft_fp8_gemm_deepseek_v3_configs, key=["N", "K"])
@triton.jit
def soft_fp8_gemm_deepseek_v3_kernel(
    A,
    B,
    C,
    Bs,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    fp8_to_fp32_scale: tl.constexpr,
    compute_dtype: tl.constexpr,
):
    """
    Perform a matrix multiplication with FP8 dynamically casted to BF16.

    Args:
        A (tl.tensor): Pointer to the first input matrix A.
        B (tl.tensor): Pointer to the second input matrix B.
        C (tl.tensor): Pointer to the output matrix C.
        Bs (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.
        GROUP_SIZE_M (tl.constexpr): Block-swizzle group size for the M dimension.

    Returns:
        None
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = B + (offs_k[:, None] + offs_bn[None, :] * K)

    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * tl.cdiv(K, BLOCK_SIZE_K)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        b_s = tl.load(Bs_ptrs + offs_ks)

        b_uint32 = b.to(tl.uint8, bitcast=True).to(tl.uint32)
        b_unscaled_fp32 = (((b_uint32 & 0x80) << 24) | ((b_uint32 & 0x7F) << 20)).to(
            tl.float32, bitcast=True
        )
        b_new_scale = b_s * fp8_to_fp32_scale
        b_scaled_fp32 = b_unscaled_fp32 * b_new_scale
        b_scaled_fp32 = b_scaled_fp32.to(dtype=compute_dtype)
        accumulator += tl.dot(a, b_scaled_fp32)

        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    c = accumulator.to(compute_dtype)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
