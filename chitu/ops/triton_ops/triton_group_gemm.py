# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl

from chitu.device_type import has_accelerator

if has_accelerator():
    from chitu.ops.triton_ops.utils import (
        SIGNED_INT32_0x87F00000,
        SIGNED_INT16_0x81C0,
        SIGNED_INT16_0x87F0,
        SIGNED_INT8_0x9C,
    )


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 SGLang Team
# SPDX-SnippetCopyrightText: 2025 vLLM Team
# SPDX-SnippetCopyrightText: 2025 Qingcheng.AI
# SDPX—SnippetName: The Triton implementation of fused MoE kernel
#
# The Triton implementation of fused MoE kernel is originally from vLLM
# (https://github.com/vllm-project/vllm/blob/ed6e9075d31e32c8548b480a47d1ffb77da1f54c/vllm/model_executor/layers/fused_moe/fused_moe.py),
# licensed under Apache 2.0, which is further based on SGLang
# (https://github.com/sgl-project/sglang/commit/ba5112ff691d791a9e38c6c71f59324a5fcb49d0),
# licensed under Apache 2.0. Qingcheng.AI added soft fp4/fp8 support to this implementation.
@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


fused_moe_kernel_configs = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_m,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_m in [32]
    for block_n in [32, 64, 128]
    for block_k in [64, 128]
    for group_m in [8, 16]
    for num_stages in [3, 4]
    for num_warps in [4, 8]
]


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_blocks_post_padded_ptr,
    # Matrix dimensions
    E,
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    bs_if_in_graph: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_blocks_post_padded = tl.load(num_blocks_post_padded_ptr)
    if pid_m >= num_blocks_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts < 0 or off_experts >= E:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_block_fp8(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_blocks_post_padded_ptr,
    E,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    soft_fp8: tl.constexpr,
    per_channel_quant: tl.constexpr,
    bs_if_in_graph: tl.constexpr,
):
    fp8_to_fp32_scale = tl.cast(0x7B800000, dtype=tl.float32, bitcast=True)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_blocks_post_padded = tl.load(num_blocks_post_padded_ptr)
    if pid_m >= num_blocks_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts < 0 or off_experts >= E:
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    if group_k > 0 and group_n > 0:
        if not soft_fp8:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
        offs_bsn = offs_bn // group_n
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
    elif per_channel_quant:
        # per-channel: b_scale shape [E, N], load per output channel
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn * stride_bsn
        b_scale = tl.load(b_scale_ptrs)  # [BLOCK_SIZE_N]
        # per-token: a_scale shape [M, 1], load per token
        a_scale = tl.load(
            a_scale_ptr + (offs_token // top_k) * stride_asm,
            mask=token_mask,
            other=0.0,
        )  # [BLOCK_SIZE_M]
    else:
        a_scale = tl.load(a_scale_ptr)
        b_scale = tl.load(b_scale_ptr + off_experts)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        if group_k > 0 and group_n > 0:
            k_start = k * BLOCK_SIZE_K
            offs_ks = k_start // group_k
            b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

            if soft_fp8:
                t = b.to(tl.int8, bitcast=True).to(
                    tl.int32
                )  # Do signed cast to copy the sign bit
                t = (t << 20) & SIGNED_INT32_0x87F00000
                b_unscaled_fp32 = t.to(tl.float32, bitcast=True)
                b_new_scale = b_scale * fp8_to_fp32_scale
                b_scaled_fp32 = b_unscaled_fp32 * b_new_scale
                b_scaled_fp32 = b_scaled_fp32.to(dtype=compute_type)
                accumulator += tl.dot(a, b_scaled_fp32)
            else:
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
        else:
            accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if group_k > 0 and group_n > 0:
        accumulator = accumulator.to(compute_type)
    elif per_channel_quant:
        accumulator = (accumulator * a_scale[:, None] * b_scale[None, :]).to(
            compute_type
        )
    else:
        accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_soft_fp4(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    b_scale2_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_blocks_post_padded_ptr,
    E,
    N,
    K,
    EM,
    num_valid_tokens_x_topk,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_b_s: tl.constexpr,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    soft_fp8: tl.constexpr,
    is_w1w3: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_blocks_post_padded = tl.load(num_blocks_post_padded_ptr)
    if pid_m >= num_blocks_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens_x_topk

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts < 0 or off_experts >= E:
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K // 2)
    num_b_s_in_block = BLOCK_SIZE_K // stride_b_s
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )
    if group_k > 0 and group_n > 0:
        if not soft_fp8:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
        b_scale_ptrs = (
            b_scale_ptr
            + off_experts * stride_bse
            + offs_bn[None, :] * stride_bsn
            + offs_k[:, None] // stride_b_s
        )
        if is_w1w3:
            if pid_n * BLOCK_SIZE_N >= N // 2:
                b_scale2 = tl.load(b_scale2_ptr + off_experts * 2 + 1)
            else:
                b_scale2 = tl.load(b_scale2_ptr + off_experts * 2 + 0)

        else:
            b_scale2 = tl.load(b_scale2_ptr + off_experts)

    fp8_to_bf16_scale = 0x7B800000
    fp8_to_bf16_scale = fp8_to_bf16_scale.to(tl.float32, bitcast=True).to(tl.bfloat16)
    if soft_fp8:
        fp4_to_bf16_scale = 0x7E800000
        fp4_to_bf16_scale = fp4_to_bf16_scale.to(tl.float32, bitcast=True).to(
            tl.bfloat16
        )
    else:
        fp4_to_fp8_scale = 64.0
        fp4_max = 6.0

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_1 = tl.load(
            a_ptrs,
            mask=token_mask[:, None]
            & (offs_k[None, :] < K - k * BLOCK_SIZE_K - BLOCK_SIZE_K // 2),
            other=0.0,
        )
        a_2 = tl.load(
            a_ptrs + BLOCK_SIZE_K // 2,
            mask=token_mask[:, None]
            & (offs_k[None, :] < K - k * BLOCK_SIZE_K - BLOCK_SIZE_K // 2),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        if group_k > 0 and group_n > 0:
            k_start = k * BLOCK_SIZE_K
            offs_ks = k_start // group_k
            if soft_fp8:
                b_scale_1 = tl.load(b_scale_ptrs)
                b_scale_2 = tl.load(b_scale_ptrs + num_b_s_in_block // 2)
                b_scale_1 = b_scale_1.to(tl.int8, bitcast=True).to(tl.int16)
                b_scale_2 = b_scale_2.to(tl.int8, bitcast=True).to(tl.int16)
                bf16_s_1 = (b_scale_1 << 4) & SIGNED_INT16_0x87F0
                bf16_s_2 = (b_scale_2 << 4) & SIGNED_INT16_0x87F0
                b_scale_1 = bf16_s_1.to(tl.bfloat16, bitcast=True) * fp8_to_bf16_scale
                b_scale_2 = bf16_s_2.to(tl.bfloat16, bitcast=True) * fp8_to_bf16_scale
                b = b.to(tl.int8, bitcast=True).to(tl.int16)
                bf16_weight_1 = (b << 12 >> 6) & SIGNED_INT16_0x81C0
                bf16_weight_1 = (
                    bf16_weight_1.to(tl.bfloat16, bitcast=True)
                    * b_scale_1
                    * fp4_to_bf16_scale
                )
                accumulator += tl.dot(a_1, bf16_weight_1)
                bf16_weight_2 = (b << 2) & SIGNED_INT16_0x81C0
                bf16_weight_2 = (
                    bf16_weight_2.to(tl.bfloat16, bitcast=True)
                    * b_scale_2
                    * fp4_to_bf16_scale
                )
                accumulator += tl.dot(a_2, bf16_weight_2)
            else:
                b_scale_1 = tl.load(b_scale_ptrs).to(tl.float8e4nv, bitcast=True)
                b_scale_2 = tl.load(b_scale_ptrs + num_b_s_in_block // 2).to(
                    tl.float8e4nv, bitcast=True
                )
                b_scale_1 = b_scale_1.to(tl.bfloat16)
                b_scale_2 = b_scale_2.to(tl.bfloat16)
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                tmp_accumulator = tl.zeros(
                    (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                )
                fp8_weight_1 = (
                    b.to(tl.int8, bitcast=True) << 4 >> 2
                ) & SIGNED_INT8_0x9C
                fp8_weight_1 = (
                    fp8_weight_1.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
                    * (fp4_to_fp8_scale / fp4_max)
                    * b_scale_1
                ).to(tl.float8e4nv)
                tmp_accumulator += tl.dot(a_1, fp8_weight_1)
                fp8_weight_2 = (b.to(tl.int8, bitcast=True) >> 2) & SIGNED_INT8_0x9C
                fp8_weight_2 = (
                    fp8_weight_2.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
                    * (fp4_to_fp8_scale / fp4_max)
                    * b_scale_2
                ).to(tl.float8e4nv)
                tmp_accumulator += tl.dot(a_2, fp8_weight_2)
                accumulator += tmp_accumulator * a_scale[:, None]
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk // 2
        b_scale_ptrs += num_b_s_in_block

    if not soft_fp8:
        accumulator *= fp4_max
    accumulator *= b_scale2
    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_int8(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_blocks_post_padded_ptr,
    E,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    bs_if_in_graph: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_blocks_post_padded = tl.load(num_blocks_post_padded_ptr)
    if pid_m >= num_blocks_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts < 0 or off_experts >= E:
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_int8_w8a8:
        a_scale_ptrs = a_scale_ptr + offs_token // top_k
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        a_scale = tl.load(a_scale_ptrs)
        b_scale = tl.load(b_scale_ptrs)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        else:
            accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_int8_w8a8:
        accumulator = (accumulator * a_scale[:, None] * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: Copyright contributors to the vLLM project
# SPDX-SnippetName: fused_moe_kernel_gptq_awq from vLLM
#
# MoE kernel for INT4/INT8 weight-only quantization (W4A16/W8A16)
# Originally from vLLM: https://github.com/vllm-project/vllm


@triton.jit
def fused_moe_kernel_gptq_awq(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    block_k_diviable: tl.constexpr,
    group_size: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
):
    """
    MoE kernel for INT4/INT8 weight-only quantization (W4A16/W8A16).

    Args:
        A: Input tensor [M, K]
        B: Weight tensor [E, N, K//8] for int4 (packed as 8 int4 per int32 along K)
        C: Output tensor [M*topk, N]
        B_scale: Scale tensor [E, num_groups, N]
        B_zp: Zero point tensor (optional)
        sorted_token_ids: Sorted token indices
        expert_ids: Expert IDs for each block
    """
    # Map program ids to block of C
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    if use_int4_w4a16:
        # AWQ packs 8 int4 values per int32 (K//8 format)
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] // 8) * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] % 8) * 4
    elif use_int8_w8a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if not block_k_diviable:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs)
        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF

        b_scale_ptrs = (
            b_scale_ptr
            + off_experts * stride_bse
            + offs_bn[None, :] * stride_bsn
            + ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsk
        )
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
        b_scale = b_scale.to(tl.float32)

        if has_zp and use_int4_w4a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + (offs_bn[None, :] // 2) * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = (b_zp >> b_zp_shifter) & 0xF
            b_zp = b_zp.to(tl.float32)
        elif has_zp and use_int8_w8a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + offs_bn[None, :] * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = b_zp.to(tl.float32)

        if has_zp:
            b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
        else:
            b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
        accumulator = tl.dot(a, b, acc=accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            # With 8 int4 per int32, advance by BLOCK_SIZE_K // 8
            b_ptrs += (BLOCK_SIZE_K // 8) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# SPDX-SnippetEnd
