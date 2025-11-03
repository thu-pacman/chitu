# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import struct
from typing import Any, Optional
from logging import getLogger

import torch

import triton
import triton.language as tl

from chitu.device_type import is_muxi, is_nvidia
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    ExpertBlockIndexedBatchedRoutedActivation,
)
from chitu.moe.batched_expert_result import PerTokenBatchedExpertResult
from chitu.ops.activation import silu_and_mul
from chitu.ops.quant import blockfp8_act_quant

if torch.cuda.is_available():
    from chitu.ops.triton_ops.utils import (
        SIGNED_INT32_0x87F00000,
        SIGNED_INT16_0x81C0,
        SIGNED_INT16_0x87F0,
        SIGNED_INT8_0x9C,
    )
    from chitu.ops.triton_ops.utils import to_triton_dtype
from chitu.lazy import single_dispatch_lazy_tensor
from chitu.distributed.parallel_state import get_ep_size

logger = getLogger(__name__)


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
    # Matrix dimensions
    E,
    N,
    K,
    EM,
    num_valid_tokens_x_topk,
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
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_b_s: tl.constexpr,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    soft_fp8: tl.constexpr,
    is_w1w3: tl.constexpr,
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
    token_mask = offs_token < num_valid_tokens_x_topk

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

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
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
        # We accumulate along the K dimension.
        if group_k > 0 and group_n > 0:
            k_start = k * BLOCK_SIZE_K
            offs_ks = k_start // group_k
            if soft_fp8:
                b_scale_1 = tl.load(b_scale_ptrs)
                b_scale_2 = tl.load(b_scale_ptrs + num_b_s_in_block // 2)
                b_scale_1 = b_scale_1.to(tl.int8, bitcast=True).to(
                    tl.int16
                )  # Do signed cast to copy the sign bit
                b_scale_2 = b_scale_2.to(tl.int8, bitcast=True).to(
                    tl.int16
                )  # Do signed cast to copy the sign bit
                bf16_s_1 = (b_scale_1 << 4) & SIGNED_INT16_0x87F0
                bf16_s_2 = (b_scale_2 << 4) & SIGNED_INT16_0x87F0
                b_scale_1 = bf16_s_1.to(tl.bfloat16, bitcast=True) * fp8_to_bf16_scale
                b_scale_2 = bf16_s_2.to(tl.bfloat16, bitcast=True) * fp8_to_bf16_scale
                b = b.to(tl.int8, bitcast=True).to(
                    tl.int16
                )  # Do signed cast to copy the sign bit
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
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
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
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    soft_fp8: tl.constexpr,
    fp8_to_fp32_scale: tl.constexpr,
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
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            if not soft_fp8:
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

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
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8:
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
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@single_dispatch_lazy_tensor
def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    B_scale2: Optional[torch.Tensor],
    B_zp: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_blocks_post_padded: torch.Tensor,
    num_valid_tokens_x_topk: int,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,  # NOTE only support fp8_w8a8
    use_fp4_w4a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    is_w1w3: bool = False,
) -> None:
    assert sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8:
        assert B_scale is not None
        assert block_shape is not None
        if not soft_fp8:
            block_n, block_k = block_shape
            assert A_scale is not None
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
            assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
        else:
            B = B.view(dtype=torch.uint8)
            assert A_scale is None
            assert B_scale is not None
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    elif use_fp4_w4a8:
        assert B_scale is not None
        assert B_scale2 is not None
        assert len(block_shape) == 2
        if not soft_fp8:
            block_n, block_k = block_shape
            assert A_scale is not None
        else:
            A_scale = None
    else:
        assert A_scale is None
        assert B_scale is None

    # Some of our platforms only has Triton with low versions, where these is no `tl.cast`
    # which is used for initializing a constant with a given type. Therefore, we need to
    # pass `fp8_to_fp32_scale` as a constant from outside.
    fp8_to_fp32_scale = struct.unpack(">f", bytes.fromhex("7b800000"))[0]

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config["BLOCK_SIZE_M"])
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    # TODO: only for merged models
    # if gate & up shares the weight scale 2, then w1w3 should be false
    if B_scale2 is not None:
        is_w1w3 = B_scale2.flatten().shape[0] >= 2 * B.shape[0]
    if use_fp4_w4a8:
        fused_moe_kernel_soft_fp4[grid](
            A,
            B,
            C,
            A_scale,
            B_scale,
            B_scale2,
            sorted_token_ids,
            expert_ids,
            num_blocks_post_padded,
            B.shape[0],
            B.shape[1],
            A.shape[1],
            EM,
            num_valid_tokens_x_topk,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(1),
            C.stride(2),
            A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
            A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
            B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
            B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
            B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
            16,
            0 if block_shape is None else block_shape[0],
            0 if block_shape is None else block_shape[1],
            top_k=top_k,
            compute_type=compute_type,
            soft_fp8=soft_fp8,
            is_w1w3=is_w1w3,
            **config,
        )
    else:
        fused_moe_kernel[grid](
            A,
            B,
            C,
            A_scale,
            B_scale,
            sorted_token_ids,
            expert_ids,
            num_blocks_post_padded,
            B.shape[0],
            B.shape[1],
            A.shape[1],
            EM,
            num_valid_tokens_x_topk,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(1),
            C.stride(2),
            A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
            A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
            B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
            B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
            B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
            0 if block_shape is None else block_shape[0],
            0 if block_shape is None else block_shape[1],
            top_k=top_k,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            soft_fp8=soft_fp8,
            fp8_to_fp32_scale=fp8_to_fp32_scale,
            **config,
        )


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
    is_marlin: bool,
    block_shape: Optional[list[int]] = None,
) -> dict[str, int]:
    if (dtype == "fp8_w8a8" or dtype == "fp4_w4a8") and block_shape is not None:
        # Block-wise quant: BLOCK_SIZE_N must be divisible by block_shape[0]
        # BLOCK_SIZE_K must be divisible by block_shape[1]
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_shape[0],
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3,
        }
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
        # A heuristic: fused marlin works faster with this config for small M
        if M <= E or (is_marlin and M <= 32):
            config = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
            }
    return config


def try_get_optimal_moe_config(
    w1_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    is_marlin: bool = False,
    block_shape: Optional[list[int]] = None,
):
    # First try to load optimal config from the file
    E, _, N = w2_shape

    config = get_default_config(
        M, E, N, w1_shape[2], top_k, dtype, is_marlin, block_shape
    )
    return config


def get_config_dtype_str(
    dtype: torch.dtype,
    use_int4_w4a16: Optional[bool] = False,
    use_int8_w8a16: Optional[bool] = False,
    use_fp8_w8a8: Optional[bool] = False,
    use_fp4_w4a8: Optional[bool] = False,
):
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif use_int4_w4a16:
        return "int4_w8a16"
    elif use_fp4_w4a8:
        return "fp4_w4a8"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None


def fused_experts(
    hidden_states: BatchedRoutedActivation,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    use_fp8_w8a8: bool = False,
    use_fp4_w4a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_scale_2: Optional[torch.Tensor] = None,
    w2_scale_2: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    experts_start_idx: int = 0,
) -> torch.Tensor:
    if get_ep_size() > 1:
        assert isinstance(hidden_states, IndexedBatchedRoutedActivation)
        n_local_experts = w1.shape[0]
        new_token_to_expert_indices = (
            hidden_states.token_to_expert_indices - experts_start_idx
        )
        mask = (new_token_to_expert_indices < 0) | (
            new_token_to_expert_indices >= n_local_experts
        )
        new_token_to_expert_indices[mask] = n_local_experts
        hidden_states = IndexedBatchedRoutedActivation(
            hidden_states.activation, new_token_to_expert_indices
        )

    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        inplace,
        activation,
        use_fp8_w8a8,
        use_fp4_w4a8,
        use_int8_w8a16,
        use_int4_w4a16,
        global_num_experts,
        w1_scale,
        w2_scale,
        w1_scale_2,
        w2_scale_2,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        soft_fp8,
    )


@functools.singledispatch
def fused_experts_impl(
    hidden_states: BatchedRoutedActivation,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    use_fp8_w8a8: bool = False,
    use_fp4_w4a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_scale2: Optional[torch.Tensor] = None,
    w2_scale2: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
):
    raise ValueError(f"Unsupported hidden_states type: {type(hidden_states)}")


@fused_experts_impl.register
def _(
    hidden_states: IndexedBatchedRoutedActivation,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    use_fp8_w8a8: bool = False,
    use_fp4_w4a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_scale2: Optional[torch.Tensor] = None,
    w2_scale2: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
):
    assert (
        topk_weights.shape == hidden_states.token_to_expert_indices.shape
    ), "topk shape mismatch"

    M, _ = hidden_states.activation.shape
    E, N, _ = w1.shape
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_weights.shape[1]
    config_dtype = get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        use_fp4_w4a8=use_fp4_w4a8,
        dtype=hidden_states.activation.dtype,
    )
    config = try_get_optimal_moe_config(
        w1.shape,
        w2.shape,
        top_k_num,
        config_dtype,
        M,
        block_shape=block_shape,
    )

    return fused_experts_impl(
        ExpertBlockIndexedBatchedRoutedActivation.convert_from(
            hidden_states,
            n_experts=global_num_experts,
            block_size=config["BLOCK_SIZE_M"],
        ),
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        inplace=inplace,
        activation=activation,
        use_fp8_w8a8=use_fp8_w8a8,
        use_fp4_w4a8=use_fp4_w4a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        global_num_experts=global_num_experts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_scale2=w1_scale2,
        w2_scale2=w2_scale2,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        soft_fp8=soft_fp8,
    )


@fused_experts_impl.register
def _(
    hidden_states: ExpertBlockIndexedBatchedRoutedActivation,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    use_fp8_w8a8: bool = False,
    use_fp4_w4a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_scale2: Optional[torch.Tensor] = None,
    w2_scale2: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
):
    # Check constraints.
    if use_int4_w4a16:
        assert (
            hidden_states.activation.shape[1] // 2 == w1.shape[2]
        ), "Hidden size mismatch"
    elif use_fp4_w4a8:
        assert (
            hidden_states.activation.shape[1] // 2 == w1.shape[2]
        ), "Hidden size mismatch"
    else:
        assert hidden_states.activation.shape[1] == w1.shape[2], "Hidden size mismatch"

    assert hidden_states.activation.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.activation.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ]

    M, _ = hidden_states.activation.shape
    E, N, _ = w1.shape
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_weights.shape[1]

    if M > 32768:
        logger.warning(
            f"fused_experts_impl is not intended for a batch containing more than 32768 "
            f"tokens (batch_size * seq_len), but got {M} tokens. Please set "
            f"infer.prefill_chunk_size to reduce the token number during prefilling."
        )

    config_dtype = get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        use_fp4_w4a8=use_fp4_w4a8,
        dtype=hidden_states.activation.dtype,
    )
    config = try_get_optimal_moe_config(
        w1.shape,
        w2.shape,
        top_k_num,
        config_dtype,
        M,
        block_shape=block_shape,
    )
    assert (
        hidden_states.block_to_token_x_topk_indices.shape[-1] == config["BLOCK_SIZE_M"]
    )

    intermediate_cache1 = torch.zeros(
        (M, top_k_num, N),
        device=hidden_states.activation.device,
        dtype=hidden_states.activation.dtype,
    )
    intermediate_cache3 = torch.zeros(
        (M, top_k_num, w2.shape[1]),
        device=hidden_states.activation.device,
        dtype=hidden_states.activation.dtype,
    )

    compute_type = to_triton_dtype(hidden_states.activation.dtype)

    if inplace:
        out_hidden_states = hidden_states.activation
    else:
        out_hidden_states = torch.empty_like(hidden_states.activation)

    if (use_fp8_w8a8 or use_fp4_w4a8) and not soft_fp8:
        block_n, block_k = block_shape
        hidden_states_activation, a1_scale = blockfp8_act_quant(
            hidden_states.activation, block_k
        )
    else:
        hidden_states_activation = hidden_states.activation
    invoke_fused_moe_kernel(
        hidden_states_activation,
        w1,
        intermediate_cache1,
        a1_scale,
        w1_scale,
        w1_scale2,
        w1_zp,
        hidden_states.block_to_token_x_topk_indices.flatten(),
        hidden_states.block_to_expert_indices,
        hidden_states.n_blocks_scalar_tensor,
        topk_weights.numel(),
        top_k_num,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_fp4_w4a8=use_fp4_w4a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        block_shape=block_shape,
        soft_fp8=soft_fp8,
        is_w1w3=True,
    )

    if activation == "silu":
        intermediate_cache2 = silu_and_mul(intermediate_cache1.view(-1, N))
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    if (use_fp8_w8a8 or use_fp4_w4a8) and not soft_fp8:
        block_n, block_k = block_shape
        intermediate_cache2, a2_scale = blockfp8_act_quant(intermediate_cache2, block_k)
    invoke_fused_moe_kernel(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        a2_scale,
        w2_scale,
        w2_scale2,
        w2_zp,
        hidden_states.block_to_token_x_topk_indices.flatten(),
        hidden_states.block_to_expert_indices,
        hidden_states.n_blocks_scalar_tensor,
        topk_weights.numel(),
        1,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_fp4_w4a8=use_fp4_w4a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        block_shape=block_shape,
        soft_fp8=soft_fp8,
    )

    return PerTokenBatchedExpertResult(
        intermediate_cache3.view(*intermediate_cache3.shape)
    ).weighted_sum(topk_weights, out=out_hidden_states)


# SPDX-SnippetEnd
