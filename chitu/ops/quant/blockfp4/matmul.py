# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.device_type import is_blackwell
from chitu.lazy import single_dispatch_lazy_tensor
from chitu.native_layout import Packed4BitWeightAlongK, Packed4BitWeightNPUNative
from chitu.global_vars import get_global_args
from chitu.ops.quant.blockfp4.convert import blockfp4_act_quant
from chitu.utils import try_import_platform_dep, try_import_opt_dep

triton, has_triton = try_import_platform_dep("triton")
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
hard_fp4_kernels, has_hard_fp4_kernels = try_import_opt_dep(
    "hard_fp4_kernels", "hard_fp4_kernels"
)
cinfer_ascendc, _ = try_import_opt_dep("cinfer_ascendc", "ascend_kernels")

if has_triton:
    from chitu.ops.triton_ops import (
        soft_fp4_raise_to_fp8_blockfp4_gemm_triton,
        soft_fp4_raise_to_bf16_blockfp4_gemm_triton,
    )


def soft_fp4_raise_to_fp8_blockfp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: Packed4BitWeightAlongK,
    b_s: torch.Tensor,
    b_s_2: torch.Tensor,
    act_block_size: int,
    impl: str = "auto",
):
    """
    Perform a matrix multiplication with FP8 dynamically casted to BF16.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor of first input matrix, must be contiguous.
        b (Packed4BitWeightAlongK): The second input matrix, must be in Packed4BitWeightAlongK layout.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.
        b_s_2 (torch.Tensor): The scaling factor for b_s, must be contiguous.
        act_block_size (int): The block size for activation quantization.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """

    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return soft_fp4_raise_to_fp8_blockfp4_gemm_triton(
            a, a_s, b, b_s, b_s_2, act_block_size
        )
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def soft_fp4_raise_to_bf16_blockfp4_gemm(
    a: torch.Tensor,
    b: Packed4BitWeightAlongK,
    b_s: torch.Tensor,
    b_s_2: torch.Tensor,
    impl: str = "auto",
):
    """
    Perform a matrix multiplication with FP4 in blockfp4 dynamically casted to BF16.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        b (Packed4BitWeightAlongK): The second input matrix, must be in Packed4BitWeightAlongK layout.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.
        b_s_2 (torch.Tensor): The scaling factor for b_s, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """

    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return soft_fp4_raise_to_bf16_blockfp4_gemm_triton(a, b, b_s, b_s_2)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def soft_fp4_raise_to_bf16_blockfp4_single_scale_gemm(
    a: torch.Tensor,
    b: Packed4BitWeightNPUNative,
    b_s: torch.Tensor,
    impl: str = "auto",
):
    """
    Perform a matrix multiplication with FP4 in blockfp4 (single scale variant) dynamically
    casted to BF16.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        b (Packed4BitWeightAlongK): The second input matrix, must be in Packed4BitWeightAlongK layout.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """

    if impl == "auto":
        impl = "npu"

    if impl == "npu":
        return soft_fp4_raise_to_bf16_blockfp4_single_scale_gemm_npu(a, b, b_s)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


@single_dispatch_lazy_tensor
def soft_fp4_raise_to_bf16_blockfp4_single_scale_gemm_npu(
    x: torch.Tensor,
    weight: Packed4BitWeightNPUNative,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    assert isinstance(weight, Packed4BitWeightNPUNative)
    weight = weight.layout_tensor

    if get_global_args().infer.raise_lower_bit_float_to != "bfloat16":
        raise NotImplementedError(
            "infer.raise_lower_bit_float_to must be 'bfloat16' for NPU linear_block_fp4_npu"
        )

    assert (
        weight.shape[-2] % 2 == 0
    ), f"Weight shape[-2] must be even, but got {weight.shape[-2]}"
    assert (
        weight.shape[-1] % 2 == 0
    ), f"Weight shape[-1] must be even, but got {weight.shape[-1]}"
    # Shape adaptation for dequantization matmul operator
    weight = weight.reshape(weight.shape[-1] * 2, weight.shape[-2] // 2)
    weight = weight.unsqueeze(0)
    weight_scale = weight_scale.unsqueeze(0)
    scale = weight_scale.transpose(-2, -1)
    if not scale.is_contiguous():
        scale = scale.contiguous()
    scale_off = torch.empty_like(scale)
    output = torch.empty(
        [x.shape[0], weight.shape[-1] * 2], dtype=x.dtype, device=x.device
    )
    # NOTE: Generate a tensor with a single element (value N) and ensure export tokens is 1-D tensor
    expert_tokens = torch.full([1], x.shape[0], device=x.device, dtype=torch.int64)

    if x.dim() == 3:
        # 3D x needs to be squeezed to 2D; in NpuAttnBackend mla_decode_paged_kv, x will be unsqueezed to 3D
        x = x.squeeze(1)
        if x.shape[0] <= 2:
            cinfer_ascendc.grouped_soft_gemv(
                x,
                weight,
                scale=scale,
                groupList=expert_tokens,
                output=output,
                computeType="fp4",
            )
        else:
            cinfer_ascendc.grouped_gemm(
                x,
                weight,
                antiquantOffsetOptional=scale_off,
                antiquantScaleOptional=scale,
                groupListOptional=expert_tokens,
                output=output,
                computeType="fp4",
            )
        output = output.unsqueeze(1)
    else:
        if x.shape[0] <= 2:
            cinfer_ascendc.grouped_soft_gemv(
                x,
                weight,
                scale=scale,
                groupList=expert_tokens,
                output=output,
                computeType="fp4",
            )
        else:
            cinfer_ascendc.grouped_gemm(
                x,
                weight,
                antiquantOffsetOptional=scale_off,
                antiquantScaleOptional=scale,
                groupListOptional=expert_tokens,
                output=output,
                computeType="fp4",
            )

    return output


def pad_tensor_to_size(tensor, target_size):
    """
    Pad the first dimension of tensor to target_size
    """
    a, n = tensor.shape

    if target_size <= a:
        return tensor[:target_size]

    new_tensor = torch.zeros((target_size, n), dtype=tensor.dtype, device=tensor.device)
    new_tensor[:a] = tensor

    return new_tensor


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 vllm Team
# SPDX-SnippetCopyrightText: 2025 Qingcheng.AI
# SDPX—SnippetName: cutlass_scaled_fp4_mm from vllm
def cutlass_scaled_fp4_mm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha,
    out_dtype: torch.dtype,
    batch_size_threshold=128,
) -> torch.Tensor:
    # reference: https://github.com/vllm-project/vllm/blob/a7b8788d2c2fae6bf52c128916de19e85f2b0a25/vllm/_custom_ops.py#L663

    assert x.ndim == 2 and weight.ndim == 2
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=torch.float32, device=x.device)
    m, n, k = x.shape[0], weight.shape[0], x.shape[1] * 2
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    cuda_nvfp4_scaled_mm_kernel = (
        (
            hard_fp4_kernels.cuda_nvfp4_scaled_mm_decode
            if m <= batch_size_threshold and k % 256 == 0
            else hard_fp4_kernels.cuda_nvfp4_scaled_mm
        )
        if has_hard_fp4_kernels
        else chitu_backend.cuda_nvfp4_scaled_mm
    )
    cuda_nvfp4_scaled_mm_kernel(out, x, weight, x_scale, weight_scale, alpha)
    return out


# SPDX-SnippetEnd


@single_dispatch_lazy_tensor
def blockfp4_gemm_chitu_backend(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    original_shape = x.shape
    original_lines = x.numel() // x.shape[-1]

    def round_up(x, y):
        return (x + y - 1) // y * y

    rounded_m = round_up(x.shape[0], 128)

    if x.ndim > 2:
        x = x.view(-1, x.shape[-1])
    # fp4_scaled_mm kernel requires the first dimension of the input matrix to be a multiple of 128
    x = pad_tensor_to_size(x, rounded_m)
    x_global_scale = ((448 * 6) / torch.amax(x.flatten(), dim=-1)).to(torch.float32)

    x, x_scale = blockfp4_act_quant(x, x_global_scale)
    if alpha is None:
        alpha = ((1.0 / x_global_scale) * weight_scale_2).to(torch.float32).to(x.device)

    y = cutlass_scaled_fp4_mm(
        x, x_scale, weight, weight_scale.view(torch.float8_e4m3fn), alpha, out_dtype
    )[:original_lines]
    y = y.view(*original_shape[:-1], -1)
    return y


@single_dispatch_lazy_tensor
def blockfp4_act_quant_gemm(
    input: torch.Tensor, weight_scale_2: torch.Tensor, swizzled: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."
    input_shape = input.size()
    input = input.view((-1, input_shape[-1]))
    m, n = input.shape
    # output size is n / 2(fp4 2 8bit)
    # output scale size is (n / 16(block)) / 4(fp8 2 32bit)
    rounded_m = (m + 127) // 128 * 128 if swizzled else m
    rounded_n = (n + 63) // 64 * 4
    output = torch.empty((m, n // 2), device=input.device, dtype=torch.uint8)
    output_scale = torch.empty(
        (rounded_m, rounded_n // 4), device=input.device, dtype=torch.int32
    )
    alpha = torch.empty_like(weight_scale_2)
    quant_kernel = hard_fp4_kernels.cuda_scaled_fp4_quant_with_alpha
    quant_kernel(output, input, output_scale, alpha, weight_scale_2, swizzled=swizzled)
    return output, output_scale.view(torch.float8_e4m3fn), alpha


@single_dispatch_lazy_tensor
def blockfp4_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    batch_size_threshold: int = 128,
) -> torch.Tensor:
    assert is_blackwell()
    if not has_hard_fp4_kernels:
        return blockfp4_gemm_chitu_backend(
            x, weight, weight_scale, weight_scale_2, alpha, out_dtype
        )
    original_shape = x.shape
    original_lines = x.numel() // x.shape[-1]

    if x.ndim > 2:
        x = x.view(-1, x.shape[-1])
    # fp4_scaled_mm kernel requires the first dimension of the input matrix to be a multiple of 128
    # x = pad_tensor_to_size(x, rounded_m)
    if x.shape[0] <= batch_size_threshold:
        x, x_scale, _alpha = blockfp4_act_quant_gemm(x, weight_scale_2, swizzled=False)
    else:
        x, x_scale, _alpha = blockfp4_act_quant_gemm(x, weight_scale_2, swizzled=True)
    if alpha is None:
        alpha = _alpha

    y = cutlass_scaled_fp4_mm(
        x,
        x_scale,
        weight,
        weight_scale.view(torch.float8_e4m3fn),
        alpha,
        out_dtype,
        batch_size_threshold,
    )[:original_lines]
    y = y.view(*original_shape[:-1], -1)
    return y
