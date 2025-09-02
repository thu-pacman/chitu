import packaging.version
import torch
import pytest

from chitu.native_layout import Packed4BitWeightAlongK
from chitu.ops import (
    soft_fp4_raise_to_fp8_blockfp4_gemm,
    soft_fp4_raise_to_bf16_blockfp4_gemm,
    blockfp8_act_quant,
    pack_every_two_fp4_e2m1_in_uint8_to_one_uint8,
    unpack_every_uint8_to_two_fp4_e2m1_in_uint8,
    to_fp4_e2m1_in_uint8,
    from_fp4_e2m1_in_uint8,
)
from chitu.device_type import has_native_fp8, is_hopper
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")


def init_weight_and_scales(dim, block_size):
    assert dim % block_size == 0
    b = torch.randn(
        dim,
        dim // block_size,
        block_size,
        dtype=torch.float32,
        device="cuda",
    )

    # Following nvfp4 quantization.
    # See https://github.com/NVIDIA/TensorRT-LLM/blob/b331d62f9812874d9aaf55aecd1946143fddf440/cpp/tensorrt_llm/thop/fp4Quantize.cpp#L29-L38
    b_s_2 = b.abs().max() / (448 * 6)
    b_s = torch.clamp((b.amax(dim=2, keepdim=True) / (6 * b_s_2)), -448, 448)
    b = b / (b_s * b_s_2)

    b = pack_every_two_fp4_e2m1_in_uint8_to_one_uint8(
        to_fp4_e2m1_in_uint8(b.view(dim, dim))
    )
    b_s = b_s.view(dim, dim // block_size).to(torch.float8_e4m3fn)
    b_s_2 = b_s_2.view(1, 1).to(torch.float32)

    return b, b_s, b_s_2


def do_dequant_b(b, b_s, b_s_2, dim, block_size):
    return (
        from_fp4_e2m1_in_uint8(unpack_every_uint8_to_two_fp4_e2m1_in_uint8(b))
        .view(dim, dim // block_size, block_size)
        .to(torch.float32)
        * b_s.view(dim, dim // block_size, 1).to(torch.float32)
        * b_s_2.view(1, 1, 1)
    ).view(dim, dim)


def do_dequant_a(a_fp8, a_s, dim, act_block_size):
    return (
        a_fp8.to(a_s.dtype).view(dim, dim // act_block_size, act_block_size)
        * a_s.view(dim, dim // act_block_size, 1)
    ).view(dim, dim)


@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
@pytest.mark.skipif(
    not has_triton
    or packaging.version.parse(triton.__version__) < packaging.version.parse("3.2.0"),
    reason="This test requires Triton version >= 3.2.0",
)
def test_fp4_raise_to_bf16_gemm_is_close_to_dequanted_gemm():
    default_dtype = torch.bfloat16
    torch.set_default_dtype(default_dtype)
    dim = 256
    block_size = 16
    a = torch.randn(dim, dim, dtype=default_dtype, device="cuda")
    b, b_s, b_s_2 = init_weight_and_scales(dim, block_size)

    # Dequant from `b`, `b_s`, `b_s_2` instead of directly using the tensor generated from
    # `torch.randn`, so the numerical difference is controlled inside the kernels
    dequant_b = do_dequant_b(b, b_s, b_s_2, dim, block_size).to(default_dtype)

    std_y = torch.nn.functional.linear(a, dequant_b)
    preprocessed_b = Packed4BitWeightAlongK.convert_from(
        Packed4BitWeightAlongK((dim, dim), b),
        k_stride=64,
    )
    y = soft_fp4_raise_to_bf16_blockfp4_gemm(a, preprocessed_b, b_s, b_s_2)

    assert torch.allclose(std_y, y, atol=0.1, rtol=0.1)


@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
@pytest.mark.skipif(
    not is_hopper(),
    reason="This test requires the GPU to be Hopper or newer",
)
@pytest.mark.skipif(
    not has_triton
    or packaging.version.parse(triton.__version__) < packaging.version.parse("3.2.0"),
    reason="This test requires Triton version >= 3.2.0",
)
def test_fp4_raise_to_fp8_gemm_is_close_to_dequanted_gemm():
    default_dtype = torch.bfloat16
    torch.set_default_dtype(default_dtype)
    dim = 256
    block_size = 16
    act_block_size = 128
    a = torch.randn(dim, dim, dtype=default_dtype, device="cuda")
    b, b_s, b_s_2 = init_weight_and_scales(dim, block_size)

    a_fp8, a_s = blockfp8_act_quant(a, act_block_size)

    # Dequant from `a_fp8` and `a_s` instead of directly using `a` in dequanted implementation,
    # so the numerical difference is controlled inside the kernels
    dequant_a = do_dequant_a(a_fp8, a_s, dim, act_block_size).to(default_dtype)

    # Dequant from `b`, `b_s`, `b_s_2` instead of directly using the tensor generated from
    # `torch.randn`, so the numerical difference is controlled inside the kernels
    dequant_b = do_dequant_b(b, b_s, b_s_2, dim, block_size).to(default_dtype)

    std_y = torch.nn.functional.linear(dequant_a, dequant_b)
    preprocessed_b = Packed4BitWeightAlongK.convert_from(
        Packed4BitWeightAlongK((dim, dim), b), k_stride=64
    )
    y = soft_fp4_raise_to_fp8_blockfp4_gemm(
        a_fp8, a_s, preprocessed_b, b_s, b_s_2, act_block_size=act_block_size
    )

    assert torch.allclose(std_y, y, atol=0.1, rtol=0.1)
