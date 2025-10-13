# SPDX-FileCopyrightText: 2025 vllm Team
# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# This file has adaption of open-source code from the following sources:
# https://github.com/vllm-project/vllm/blob/a7b8788d2c2fae6bf52c128916de19e85f2b0a25/tests/kernels/quantization/nvfp4_utils.py
# https://github.com/vllm-project/vllm/blob/a7b8788d2c2fae6bf52c128916de19e85f2b0a25/tests/kernels/quantization/test_nvfp4_scaled_mm.py
# licensed under Apache 2.0.

from chitu.device_type import is_blackwell
import pytest
import torch

from chitu import ops
from chitu.ops.quant.blockfp4.matmul import cutlass_scaled_fp4_mm

kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def convert_swizzled_to_linear(a_sf_swizzled: torch.Tensor, m, k, block_size):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k]


def dequantize_nvfp4_to_dtype(
    tensor_fp4, tensor_sf, global_scale, dtype, device, block_size=16
):
    """Dequantize the fp4 tensor back to high precision."""
    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, dtype)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale

    # scale the tensor
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out.to(dtype=dtype)


def break_fp4_bytes(a, dtype):
    assert a.dtype == torch.uint8
    m, n = a.shape

    # Vectorized nibble processing
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4  # Upper nibbles
    low = a_flat & 0x0F  # Lower nibbles

    # Combine nibbles for batch processing
    combined = torch.stack((low, high), dim=1).flatten()

    # Vectorized sign and magnitude extraction
    signs = (combined & 0x08).to(torch.bool)  # Sign bits
    abs_vals = (combined & 0x07).to(torch.long)  # Magnitude indices

    # Device-aware lookup and sign application
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)

    # Reshape to final form
    return values.reshape(m, n * 2).to(dtype=dtype)


DTYPES = [torch.float16, torch.bfloat16]
# m, n, k
SHAPES = [(128, 128, 64), (128, 128, 128), (256, 128, 64), (128, 256, 128)]
PAD_SHAPES = [(150, 128, 64), (128, 128, 96)]
SHAPES.extend(PAD_SHAPES)

CUDA_DEVICES = ["cuda:0"]


def get_ref_results(
    a_fp4,
    b_fp4,
    a_sf,
    b_sf,
    a_global_scale,
    b_global_scale,
    m,
    n,
    dtype,
    block_size,
    device,
):
    _, m_k = a_fp4.shape
    _, n_k = b_fp4.shape
    assert m_k == n_k
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4, a_sf, a_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    b_in_dtype = dequantize_nvfp4_to_dtype(
        b_fp4, b_sf, b_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    return torch.matmul(a_in_dtype, b_in_dtype.t())


def pad_tensor_to_size(tensor, target_size):
    a, n = tensor.shape

    if target_size <= a:
        return tensor[:target_size]

    new_tensor = torch.zeros((target_size, n), dtype=tensor.dtype, device=tensor.device)
    new_tensor[:a] = tensor

    return new_tensor


@pytest.mark.skipif(
    not is_blackwell(),
    reason="This test requires the GPU to have hard FP4 support",
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_nvfp4_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    device: str,
) -> None:
    m, n, packed_k = shape
    k = packed_k * 2
    block_size = 16
    a_dtype = torch.randn((m, k), dtype=dtype, device=device)
    b_dtype = torch.randn((n, k), dtype=dtype, device=device)

    def round_up(x, y):
        return (x + y - 1) // y * y

    rounded_m = round_up(m, 128)
    rounded_n = round_up(n, 128)
    rounded_k = round_up(k, 256)
    a_dtype = pad_tensor_to_size(a_dtype, rounded_m)
    b_dtype = pad_tensor_to_size(b_dtype, rounded_n)

    a_global_scale = ((448 * 6) / torch.amax(a_dtype.flatten(), dim=-1)).to(
        torch.float32
    )
    b_global_scale = ((448 * 6) / torch.amax(b_dtype.flatten(), dim=-1)).to(
        torch.float32
    )
    alpha = 1.0 / (a_global_scale * b_global_scale)
    a_fp4, a_scale_interleaved = ops.blockfp4_act_quant(a_dtype, a_global_scale)
    b_fp4, b_scale_interleaved = ops.blockfp4_act_quant(b_dtype, b_global_scale)

    expected_out = get_ref_results(
        a_fp4,
        b_fp4,
        a_scale_interleaved,
        b_scale_interleaved,
        a_global_scale,
        b_global_scale,
        m,
        n,
        dtype,
        block_size,
        device,
    )
    out_1 = cutlass_scaled_fp4_mm(
        a_fp4,
        a_scale_interleaved,
        b_fp4,
        b_scale_interleaved,
        alpha,
        dtype,
        batch_size_threshold=0,
    )
    a_padded = a_dtype.new_zeros((rounded_m, rounded_k))
    a_padded[:m, :k] = a_dtype[:m, :k]
    b_padded = b_dtype.new_zeros((rounded_n, rounded_k))
    b_padded[:n, :k] = b_dtype[:n, :k]
    a_padded_fp4, a_padded_scale = ops.blockfp4_act_quant(a_padded, a_global_scale)
    b_padded_fp4, b_padded_scale = ops.blockfp4_act_quant(b_padded, b_global_scale)
    out_2 = cutlass_scaled_fp4_mm(
        a_padded_fp4,
        convert_swizzled_to_linear(a_padded_scale, rounded_m, rounded_k, 16),
        b_padded_fp4,
        b_padded_scale,
        alpha,
        dtype,
        batch_size_threshold=1e20,
    )

    torch.testing.assert_close(
        out_1, expected_out.to(dtype=dtype), atol=2e-1, rtol=1e-1
    )
    torch.testing.assert_close(
        out_2, expected_out.to(dtype=dtype), atol=2e-1, rtol=1e-1
    )
