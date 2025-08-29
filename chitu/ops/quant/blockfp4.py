# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.device_type import is_blackwell
from chitu.utils import try_import_platform_dep, try_import_opt_dep
from chitu.native_layout import Packed4BitWeightAlongK, Packed4BitWeightNPUNative
from chitu.lazy import single_dispatch_lazy_tensor
from chitu.global_vars import get_global_args

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_platform_dep("torch_npu")
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
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
) -> torch.Tensor:
    # reference: https://github.com/vllm-project/vllm/blob/a7b8788d2c2fae6bf52c128916de19e85f2b0a25/vllm/_custom_ops.py#L663

    assert x.ndim == 2 and weight.ndim == 2
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=torch.float32, device=x.device)
    m, n = x.shape[0], weight.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    chitu_backend.cuda_nvfp4_scaled_mm(out, x, weight, x_scale, weight_scale, alpha)
    return out


# SPDX-SnippetEnd


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 vllm Team
# SPDX-SnippetCopyrightText: 2025 Qingcheng.AI
# SDPX—SnippetName: scaled_fp4_quant from vllm
@single_dispatch_lazy_tensor
def blockfp4_act_quant(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 and return quantized tensor and scale.

    This function quantizes the last dimension of the given tensor `input`. For
    every 16 consecutive elements, a single dynamically computed scaling factor
    is shared. This scaling factor is quantized using the `input_global_scale`
    and is stored in a swizzled layout (see
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x).

    Args:
        input: The input tensor to be quantized to FP4
        input_global_scale: A scalar scaling factor for the entire tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP4 but every
            two values are packed into a uint8 and float8_e4m3 scaling factors
            in the sizzled layout.
    """
    # reference: https://github.com/vllm-project/vllm/blob/a7b8788d2c2fae6bf52c128916de19e85f2b0a25/vllm/_custom_ops.py#L1117

    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    block_size = 16
    device = input.device

    assert n % block_size == 0, f"last dim has to be multiple of 16, but got {n}."
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."

    # Two fp4 values will be packed into an uint8.
    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)

    # We use the rounded values to store the swizzled values. Due to the
    # requirement of the Tensor Core, the minimum tile is 128x4 for the scales.
    # So, we first pad the scales to multiples of 128 and 4. Then, the scales
    # (in float8_e4m3fn) are packed into an int32 for every 4 values. More:
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x
    round_up = lambda x, y: (x + y - 1) // y * y
    rounded_m = round_up(m, 128)
    scale_n = n // block_size
    rounded_n = round_up(scale_n, 4)
    output_scale = torch.empty(
        (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
    )

    chitu_backend.cuda_scaled_fp4_quant(output, input, output_scale, input_global_scale)
    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale


# SPDX-SnippetEnd


@single_dispatch_lazy_tensor
def blockfp4_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    assert is_blackwell()
    original_shape = x.shape
    original_lines = x.numel() // x.shape[-1]

    def round_up(x, y):
        return (x + y - 1) // y * y

    rounded_m = round_up(x.shape[0], 128)
    k = x.shape[-1]

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


FP4_E2M1_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def unpack_every_uint8_to_two_fp4_e2m1_in_uint8(packed: torch.Tensor) -> torch.Tensor:
    assert packed.dtype == torch.uint8
    out, half_in = packed.shape
    high_nibble = packed & 0x0F  # [out, in // 2]
    low_nibble = packed >> 4  # [out, in // 2]
    return torch.stack([high_nibble, low_nibble], dim=2).view(out, half_in * 2)


def from_fp4_e2m1_in_uint8(nibbles: torch.Tensor) -> torch.Tensor:
    n = nibbles.to(torch.uint8)
    sign = torch.where((n >> 3).bool(), -1.0, 1.0)
    idx = (n & 0x7).to(torch.long)
    levels = FP4_E2M1_LEVELS.to(n.device)
    val = sign * levels[idx]
    return val  # float32


def pack_every_two_fp4_e2m1_in_uint8_to_one_uint8(w_nib: torch.Tensor) -> torch.Tensor:
    out, inp = w_nib.shape
    assert inp % 2 == 0
    high = w_nib[:, 0::2]  # [out, in // 2]
    low = w_nib[:, 1::2]  # [out, in // 2]
    packed = (low << 4) | high
    return packed  # uint8, [out, in // 2]


def to_fp4_e2m1_in_uint8(x: torch.Tensor) -> torch.Tensor:
    abs_x = x.abs()
    levels = FP4_E2M1_LEVELS.to(abs_x.device).view(*([1] * abs_x.dim()), -1)
    idx = (abs_x.unsqueeze(-1) == levels).to(torch.uint8).argmax(dim=-1).to(torch.uint8)
    sign = (x < 0).to(torch.uint8) << 3
    nibble = sign | idx
    return nibble


def fp4_fake_quant(x, block_size=16, block_scale=None, global_scale=None, quant=False):
    if x.numel() == 0:
        return x, x, x
    shape, dtype = x.size(), x.dtype
    x = x.reshape(*x.shape[:-1], -1, block_size)
    if global_scale is None:
        global_scale = x.abs().max().float() / (448 * 6)
    if block_scale is None:
        block_max = torch.max(torch.abs(x), dim=-1, keepdim=True).values
        block_scale = torch.clamp((block_max / (6 * global_scale)), -448, 448).to(
            torch.float8_e4m3fn
        )
    dq_block_scale = block_scale.to(torch.float32) * global_scale
    scaled_x = x / dq_block_scale
    # Quantize to FP4 values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}, following round to even
    abs_scaled_x = torch.abs(scaled_x)
    qx = fp4_rtn(abs_scaled_x)
    sign = torch.where(scaled_x >= 0, 1.0, -1.0)
    if quant:
        return (
            (qx * sign).reshape(shape).to(dtype),
            block_scale.squeeze(-1),
            global_scale,
        )
    else:
        qdq_x = qx * dq_block_scale * sign
        return qdq_x.reshape(shape).to(dtype), block_scale.squeeze(-1), global_scale


def fp4_rtn(abs_scaled_x):
    qx = torch.where(
        abs_scaled_x <= 0.25,
        0.0,
        torch.where(
            abs_scaled_x < 0.75,
            0.5,
            torch.where(
                abs_scaled_x <= 1.25,
                1.0,
                torch.where(
                    abs_scaled_x < 1.75,
                    1.5,
                    torch.where(
                        abs_scaled_x <= 2.5,
                        2,
                        torch.where(
                            abs_scaled_x < 3.5,
                            3.0,
                            torch.where(abs_scaled_x <= 5.0, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return qx


def convert_linear_to_swizzled(a_sf_linear: torch.Tensor, m, k, block_size):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_linear, (1, m_tiles, 4, 32, k_tiles, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k]
