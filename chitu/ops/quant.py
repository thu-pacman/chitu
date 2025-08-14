# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple, Optional, Dict, Any, List

from chitu.device_type import is_blackwell
import torch
import torch.nn.functional as F

from chitu.utils import try_import_platform_dep
from chitu.native_layout import Packed4BitWeightAlongK

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
triton, has_triton = try_import_platform_dep("triton")
hygon_mixq_kernels, has_hygon = try_import_platform_dep("sugon_mixQ4_kernels")
if has_triton:
    from chitu.ops.triton_ops import (
        quant_einsum_shc_hdc_shd_triton,
        w8a8_gemm_per_token_per_channel_triton,
        w4a8_gemm_per_token_per_channel_asymm_triton,
        fp8_gemm_deepseek_v3_triton_default,
        soft_fp8_gemm_deepseek_v3_triton,
        soft_fp4_raise_to_fp8_gemm_deepseek_v3_triton,
        soft_fp4_raise_to_bf16_gemm_deepseek_v3_triton,
        weight_dequant_deepseek_v3_triton,
        weight_dequant_soft_fp8_deepseek_v3_triton,
        act_quant_deepseek_v3_triton,
        mixq_w8a8_gemm_triton,
        mixq_w4a4_gemm_triton,
    )


def unpack_weight_bytes(packed):
    assert packed.dtype == torch.uint8
    out, half_in = packed.shape

    high_nibble = packed & 0x0F  # [out, half_in]
    low_nibble = packed >> 4  # [out, half_in]

    return torch.stack([high_nibble, low_nibble], dim=2).view(out, half_in * 2)


def decode_e2m1_from_nibbles(nibbles: torch.Tensor):
    _LEVELS = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    n = nibbles.to(torch.uint8)
    sign = torch.where((n >> 3).bool(), -1.0, 1.0)
    idx = (n & 0x7).to(torch.long)
    levels = _LEVELS.to(n.device)
    val = sign * levels[idx]
    return val


def pack_weight_nibbles(w_nib):
    out, inp = w_nib.shape
    assert inp % 2 == 0
    high = w_nib[:, 0::2]  # [out, in//2]
    low = w_nib[:, 1::2]  # [out, in//2]
    packed = (low << 4) | high
    return packed  # dtype uint8, shape [out, in//2]


def to_e2m1_nibbles(x):
    _LEVELS = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    abs_x = x.abs()
    levels = _LEVELS.to(abs_x.device).view(*([1] * abs_x.dim()), -1)
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


def weight_quant_deepseek_v3(
    w: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    row, col = w.shape
    assert row % block_size == 0
    assert col % block_size == 0
    w_block_at_last = (
        w.view(row // block_size, block_size, col // block_size, block_size)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(-1, block_size * block_size)
    ).to(torch.float32)
    s = torch.amax(torch.abs(w_block_at_last), dim=-1, keepdim=True)
    w_block_at_last = (w_block_at_last / s).to(torch.float8_e4m3fn)
    w = (
        w_block_at_last.view(
            row // block_size, col // block_size, block_size, block_size
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(row, col)
    )
    s = s.view(row // block_size, col // block_size)
    return w, s


def quant_einsum_shc_hdc_shd(
    group_A: torch.Tensor,
    group_B: torch.Tensor,
    group_b_s: torch.Tensor,
    *,
    block_size: int = 128,
    group_n: int = 128,
    group_k: int = 128,
    soft_fp8: bool = False,
    impl: str = "auto",
):
    assert group_A.dim() == 3
    assert group_B.dim() == 3
    assert group_A.shape[1] == group_B.shape[0]
    assert group_A.shape[2] == group_B.shape[2]

    if impl == "auto":
        if has_triton:
            impl = "triton"
        else:
            impl = "torch"

    if impl == "torch":
        weight_dequant_fn = (
            weight_dequant_soft_fp8_deepseek_v3
            if soft_fp8
            else weight_dequant_deepseek_v3
        )
        group_B = weight_dequant_fn(group_B, group_b_s, block_size=block_size)
        return torch.einsum("shc,hdc->shd", group_A, group_B)
    elif impl == "triton":
        assert block_size == 128
        return quant_einsum_shc_hdc_shd_triton(
            group_A,
            group_B,
            group_b_s,
            group_n=group_n,
            group_k=group_k,
            soft_fp8=soft_fp8,
        )
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def w8a8_gemm_per_token_per_channel(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    impl: str = "auto",
):
    if impl == "auto":
        impl = "triton"

    if impl == "triton":
        assert has_triton
        return w8a8_gemm_per_token_per_channel_triton(a, a_s, b, b_s)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def w4a8_gemm_per_token_per_channel_asymm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: Packed4BitWeightAlongK,
    b_s: torch.Tensor,
    b_z: torch.Tensor,
    impl: str = "auto",
):
    if impl == "auto":
        impl = "triton"

    if impl == "triton":
        assert has_triton
        return w4a8_gemm_per_token_per_channel_asymm_triton(a, a_s, b, b_s, b_z)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def w4a8_gemm_per_token_per_group_asymm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_z: torch.Tensor,
    b_s: torch.Tensor,
    b_s2: torch.Tensor,
    out_feats: torch.Tensor,
    impl: str = "cuda",
    group_size: int = 128,
):
    if impl == "auto":
        if has_chitu_backend:
            impl = "cuda"
        else:
            raise NotImplementedError("No GEMM implementation available")

    if impl == "cuda":
        chitu_backend.w4a8_per_group_gemm_forward_cuda(
            a, b, b_z, b_s, b_s2, a_s, out_feats
        )
        return out_feats
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def fp8_gemm_deepseek_v3(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    impl: str = "auto",
):
    if impl == "auto":
        impl = "triton"

    if impl == "triton":
        assert has_triton
        return fp8_gemm_deepseek_v3_triton_default(a, a_s, b, b_s)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def soft_fp8_gemm_deepseek_v3(
    a: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    impl: str = "auto",
):
    """
    Perform a matrix multiplication with FP8 dynamically casted to BF16.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """

    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return soft_fp8_gemm_deepseek_v3_triton(a, b, b_s)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def soft_fp4_raise_to_fp8_gemm_deepseek_v3(
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
        return soft_fp4_raise_to_fp8_gemm_deepseek_v3_triton(
            a, a_s, b, b_s, b_s_2, act_block_size
        )
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def soft_fp4_raise_to_bf16_gemm_deepseek_v3(
    a: torch.Tensor,
    b: Packed4BitWeightAlongK,
    b_s: torch.Tensor,
    b_s_2: torch.Tensor,
    impl: str = "auto",
):
    """
    Perform a matrix multiplication with FP8 dynamically casted to BF16.

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
        return soft_fp4_raise_to_bf16_gemm_deepseek_v3_triton(a, b, b_s, b_s_2)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def weight_dequant_soft_fp8_deepseek_v3(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128, impl: str = "auto"
) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M / block_size, N / block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """

    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return weight_dequant_soft_fp8_deepseek_v3_triton(x, s, block_size)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def act_quant_deepseek_v3(
    x: torch.Tensor, block_size: int = 128, impl: str = "auto"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """

    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return act_quant_deepseek_v3_triton(x, block_size)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def weight_dequant_deepseek_v3(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128, impl: str = "auto"
) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M / block_size, N / block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """

    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return weight_dequant_deepseek_v3_triton(x, s, block_size)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


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


def convert_linear_to_swizzled(a_sf_linear: torch.Tensor, m, k, block_size):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_linear, (1, m_tiles, 4, 32, k_tiles, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k]


def hard_fp4_scaled_mm(
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

    x, x_scale = scaled_fp4_quant(x, x_global_scale)
    if alpha is None:
        alpha = ((1.0 / x_global_scale) * weight_scale_2).to(torch.float32).to(x.device)

    y = cutlass_scaled_fp4_mm(
        x, x_scale, weight, weight_scale.view(torch.float8_e4m3fn), alpha, out_dtype
    )[:original_lines]
    y = y.view(*original_shape[:-1], -1)
    return y


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 vllm Team
# SPDX-SnippetCopyrightText: 2025 Qingcheng.AI
# SDPX—SnippetName: scaled_fp4_quant from vllm
def scaled_fp4_quant(
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


def mixq_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    b_fp: torch.Tensor,
    num_outliers: int,
    outliers_idx_grouped: torch.Tensor,
    outliers_idx_start: torch.Tensor = None,
    w_bits: int = 4,
    a_bits: int = 4,
    impl: str = "auto",
):
    if impl == "auto":
        if has_triton:
            impl = "triton"
        elif has_hygon:
            impl = "hygon"
        else:
            NotImplementedError(f"Unsupported implementation: {impl}")

    if impl == "hygon" and has_hygon:
        assert outliers_idx_grouped.is_cuda and outliers_idx_start.is_cuda
        if (w_bits, a_bits) == (4, 4):
            return hygon_mixq_kernels.mixq_w4a4_gemm(
                a, b, b_s, b_fp, num_outliers, outliers_idx_grouped, outliers_idx_start
            )
        elif (w_bits, a_bits) == (8, 8):
            return hygon_mixq_kernels.mixq_w8a8_gemm(
                a, b, b_s, b_fp, num_outliers, outliers_idx_grouped, outliers_idx_start
            )
        else:
            NotImplementedError(f"Unsupported bits num: w{w_bits}a{a_bits}")
    elif impl == "triton" and has_triton:
        if (w_bits, a_bits) == (4, 4):
            return mixq_w4a4_gemm_triton(a, b.T, b_s, b_fp.T, outliers_idx_grouped)
        elif (w_bits, a_bits) == (8, 8):
            return mixq_w8a8_gemm_triton(a, b.T, b_s, b_fp.T, outliers_idx_grouped)
        else:
            NotImplementedError(f"Unsupported bits num: w{w_bits}a{a_bits}")
    else:
        NotImplementedError(f"Unsupported implementation: {impl}")
