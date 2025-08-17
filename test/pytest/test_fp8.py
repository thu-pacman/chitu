import torch
import pytest
import triton

from chitu.ops import (
    blockfp8_act_quant,
    blockfp8_gemm,
    blockfp8_weight_dequant,
    soft_fp8_blockfp8_weight_dequant,
    soft_fp8_blockfp8_gemm,
)
from chitu.device_type import has_native_fp8


def init_b_and_b_s(dim, block_size):
    assert dim % block_size == 0
    b = torch.randn(
        dim // block_size,
        block_size,
        dim // block_size,
        block_size,
        dtype=torch.float32,
        device="cuda",
    )
    b_s = b.amax(dim=1, keepdim=True).amax(dim=3, keepdim=True)
    b /= b_s
    return b.view(dim, dim).to(torch.float8_e4m3fn), b_s.view(
        dim // block_size, dim // block_size
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_dequanted_gemm_is_close_to_fp8_gemm(dtype: torch.dtype):
    torch.set_default_dtype(dtype)
    dim = 256
    block_size = 128
    assert dim % block_size == 0, "dim must be divisible by block_size"
    a = torch.randn(dim, dim, dtype=dtype, device="cuda")
    b, b_s = init_b_and_b_s(dim, block_size)

    a_fp8, a_s = blockfp8_act_quant(a, block_size)
    std_y = blockfp8_gemm(a_fp8, a_s, b, b_s)

    # Dequant from `a_fp8` and `a_s` instead of directly using `a` in dequanted implementation,
    # so the numerical difference is controlled inside the kernels
    dequant_a = (
        (
            a_fp8.to(a_s.dtype).view(dim, dim // block_size, block_size)
            * a_s.view(dim, dim // block_size, 1)
        )
        .to(dtype)
        .view(dim, dim)
    )

    dequant_b = blockfp8_weight_dequant(b, b_s)
    y = torch.nn.functional.linear(dequant_a, dequant_b)

    assert torch.allclose(std_y, y, atol=0.15, rtol=0.15)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "dim"],
        x_vals=[
            (1, 1024),
            (4, 1024),
            (16, 1024),
            (64, 1024),
            (256, 1024),
            (1, 4096),
            (4, 4096),
            (16, 4096),
            (64, 4096),
            (256, 4096),
        ],
        line_arg="provider",
        line_vals=["torch_bf16", "triton_fp8"],
        line_names=["Torch_BF16", "Triton_FP8"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="fp8_gemm-performance",
        args={
            "dtype": torch.bfloat16,
            "block_size": 128,
        },
    )
)
def benchmark_fp8_gemm(bs, dim, dtype, block_size, provider):
    torch.manual_seed(42)
    torch.set_default_dtype(dtype)
    device = torch.device("cuda")
    a = torch.randn(bs, dim, dtype=dtype, device=device)
    b, b_s = init_b_and_b_s(dim, block_size)

    if provider == "torch_bf16":
        dequant_b = blockfp8_weight_dequant(b, b_s)
        ms = triton.testing.do_bench(lambda: torch.nn.functional.linear(a, dequant_b))
    elif provider == "triton_fp8":
        a_fp8, a_s = blockfp8_act_quant(a, block_size)
        ms = triton.testing.do_bench(lambda: blockfp8_gemm(a_fp8, a_s, b, b_s))
    else:
        assert False, f"Unknown provider: {provider}"
    return ms * 1000


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_soft_fp8_dequant_is_close_to_dequant(dtype: torch.dtype):
    torch.set_default_dtype(dtype)
    dim = 256
    block_size = 128
    b, b_s = init_b_and_b_s(dim, block_size)

    dequant_b = soft_fp8_blockfp8_weight_dequant(b, b_s)
    soft_dequant_b = soft_fp8_blockfp8_weight_dequant(b, b_s)

    assert torch.allclose(dequant_b, soft_dequant_b, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_soft_fp8_gemm_is_close_to_dequanted_gemm(dtype: torch.dtype):
    torch.set_default_dtype(dtype)
    dim = 256
    block_size = 128
    a = torch.randn(dim, dim, dtype=dtype, device="cuda")
    b, b_s = init_b_and_b_s(dim, block_size)

    dequant_b = soft_fp8_blockfp8_weight_dequant(b, b_s)
    std_y = torch.nn.functional.linear(a, dequant_b)
    y = soft_fp8_blockfp8_gemm(a, b, b_s)

    assert torch.allclose(std_y, y, atol=1e-2, rtol=1e-2)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "dim"],
        x_vals=[
            (1, 1024),
            (4, 1024),
            (16, 1024),
            (64, 1024),
            (256, 1024),
            (1, 4096),
            (4, 4096),
            (16, 4096),
            (64, 4096),
            (256, 4096),
        ],
        line_arg="provider",
        line_vals=["torch_bf16", "triton_soft_fp8"],
        line_names=["Torch_BF16", "Triton_Soft_FP8"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="soft_fp8_gemm-performance",
        args={
            "dtype": torch.bfloat16,
            "block_size": 128,
        },
    )
)
def benchmark_soft_fp8_gemm(bs, dim, dtype, block_size, provider):
    torch.manual_seed(42)
    torch.set_default_dtype(dtype)
    device = torch.device("cuda")
    a = torch.randn(bs, dim, dtype=dtype, device=device)
    b, b_s = init_b_and_b_s(dim, block_size)

    if provider == "torch_bf16":
        dequant_b = soft_fp8_blockfp8_weight_dequant(b, b_s)
        ms = triton.testing.do_bench(lambda: torch.nn.functional.linear(a, dequant_b))
    elif provider == "triton_soft_fp8":
        ms = triton.testing.do_bench(lambda: soft_fp8_blockfp8_gemm(a, b, b_s))
    else:
        assert False, f"Unknown provider: {provider}"
    return ms * 1000


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["dim"],
        x_vals=[128, 256, 512, 1024],
        line_arg="provider",
        line_vals=["triton_soft_fp8_dequant"],
        line_names=["Triton_Soft_FP8_Dequant"],
        styles=[("green", "-")],
        ylabel="us",
        plot_name="soft_fp8_dequant-performance",
        args={
            "dtype": torch.bfloat16,
            "block_size": 128,
        },
    )
)
def benchmark_soft_fp8_dequant(dim, dtype, block_size, provider):
    torch.manual_seed(42)
    torch.set_default_dtype(dtype)
    b, b_s = init_b_and_b_s(dim, block_size)

    ms = triton.testing.do_bench(lambda: soft_fp8_blockfp8_weight_dequant(b, b_s))
    return ms * 1000


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["dim"],
        x_vals=[128, 256, 512, 1024],
        line_arg="provider",
        line_vals=["triton_fp8_dequant"],
        line_names=["Triton_FP8_Dequant"],
        styles=[("blue", "-")],
        ylabel="us",
        plot_name="fp8_dequant-performance",
        args={
            "dtype": torch.bfloat16,
            "block_size": 128,
        },
    )
)
def benchmark_fp8_dequant(dim, dtype, block_size, provider):
    torch.manual_seed(42)
    torch.set_default_dtype(dtype)
    b, b_s = init_b_and_b_s(dim, block_size)

    ms = triton.testing.do_bench(lambda: blockfp8_weight_dequant(b, b_s))
    return ms * 1000


if __name__ == "__main__":
    benchmark_fp8_gemm.run(show_plots=True, print_data=True)
    benchmark_soft_fp8_gemm.run(show_plots=True, print_data=True)
    benchmark_soft_fp8_dequant.run(show_plots=True, print_data=True)
    benchmark_fp8_dequant.run(show_plots=True, print_data=True)
