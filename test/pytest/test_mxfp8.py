import pytest
import torch

from chitu.device_type import has_native_fp8
from chitu.ops import (
    blockfp8_act_quant,
    blockfp8_gemm,
    soft_fp8_blockfp8_gemm,
    soft_fp8_blockfp8_weight_dequant,
)
from chitu.testing import assert_close
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")

BLOCK_SIZE = 32
SCALE_BLOCK_SHAPE = [1, 32]


def init_mxfp8_weight(out_features, in_features):
    assert in_features % BLOCK_SIZE == 0
    weight = torch.randn(
        out_features, in_features, dtype=torch.float32, device="cuda"
    ).to(torch.float8_e4m3fn)
    scale = torch.randint(
        120,
        135,
        (out_features, in_features // BLOCK_SIZE),
        dtype=torch.uint8,
        device="cuda",
    )
    return weight, scale


def dequant_mxfp8_weight(weight, scale):
    out_features, in_features = weight.shape
    return (
        weight.float()
        .view(out_features, in_features // BLOCK_SIZE, BLOCK_SIZE)
        .mul_(torch.exp2(scale.float() - 127).unsqueeze(-1))
        .view_as(weight)
        .to(torch.get_default_dtype())
    )


@pytest.mark.parametrize("out_features,in_features", [(128, 256), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not has_triton, reason="Triton is not available")
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_mxfp8_soft_dequant_is_close_to_ref(
    out_features, in_features, dtype, record_benchmark
):
    torch.set_default_dtype(dtype)
    weight, scale = init_mxfp8_weight(out_features, in_features)

    output = record_benchmark.run(
        lambda: soft_fp8_blockfp8_weight_dequant(
            weight,
            scale,
            scale_block_shape=SCALE_BLOCK_SHAPE,
        ),
        out_features=out_features,
        in_features=in_features,
        impl="soft_fp8_dequant",
    )
    output_ref = dequant_mxfp8_weight(weight, scale)

    assert_close(output, output_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("bs", [0, 1, 64])
@pytest.mark.parametrize("out_features,in_features", [(128, 256), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not has_triton, reason="Triton is not available")
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_mxfp8_soft_gemm_is_close_to_dequantized_gemm(
    bs, out_features, in_features, dtype, record_benchmark
):
    torch.set_default_dtype(dtype)
    x = torch.randn(bs, in_features, dtype=dtype, device="cuda")
    weight, scale = init_mxfp8_weight(out_features, in_features)

    output_ref = torch.nn.functional.linear(x, dequant_mxfp8_weight(weight, scale))
    output = record_benchmark.run(
        lambda: soft_fp8_blockfp8_gemm(
            x,
            weight,
            scale,
            scale_block_shape=SCALE_BLOCK_SHAPE,
        ),
        bs=bs,
        out_features=out_features,
        in_features=in_features,
        impl="soft_fp8_gemm",
    )

    assert_close(output, output_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("bs", [0, 1, 64])
@pytest.mark.parametrize("out_features,in_features", [(128, 256), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not has_triton, reason="Triton is not available")
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_mxfp8_gemm_is_close_to_dequantized_gemm(
    bs, out_features, in_features, dtype, record_benchmark
):
    torch.set_default_dtype(dtype)
    x = torch.randn(bs, in_features, dtype=dtype, device="cuda")
    weight, scale = init_mxfp8_weight(out_features, in_features)
    x_fp8, x_scale = blockfp8_act_quant(
        x, scale_block_shape=SCALE_BLOCK_SHAPE, round_scale_to_pow2=False
    )

    output = record_benchmark.run(
        lambda: blockfp8_gemm(
            x_fp8,
            x_scale,
            weight,
            scale,
            round_scale_to_pow2=True,
            scale_block_shape=SCALE_BLOCK_SHAPE,
        ),
        bs=bs,
        out_features=out_features,
        in_features=in_features,
        impl="fp8_gemm",
    )
    dequant_x = (
        x_fp8.float()
        .view(bs, in_features // BLOCK_SIZE, BLOCK_SIZE)
        .mul_(x_scale.view(bs, in_features // BLOCK_SIZE, 1))
        .view_as(x)
        .to(dtype)
    )
    output_ref = torch.nn.functional.linear(
        dequant_x, dequant_mxfp8_weight(weight, scale)
    )

    assert_close(output, output_ref, atol=0.15, rtol=0.15, cos_sim_tol=0.001)
