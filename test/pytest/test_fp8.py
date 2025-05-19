import torch
import pytest
import triton
import triton.language as tl
from chitu.ops import (
    act_quant_deepseek_v3,
    fp8_gemm_deepseek_v3,
    weight_dequant_deepseek_v3,
    weight_dequant_soft_fp8_deepseek_v3,
    soft_fp8_gemm_deepseek_v3,
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
    a = torch.randn(dim, dim, dtype=dtype, device="cuda")
    b, b_s = init_b_and_b_s(dim, block_size)

    a_fp8, a_s = act_quant_deepseek_v3(a, block_size)
    std_y = fp8_gemm_deepseek_v3(a_fp8, a_s, b, b_s)

    dequant_b = weight_dequant_deepseek_v3(b, b_s)
    y = torch.nn.functional.linear(a, dequant_b)

    # Assert no more than 10% of the elements are different more than 15%
    assert (
        len(torch.nonzero(~torch.isclose(std_y, y, atol=0.15, rtol=0.15))) / (dim * dim)
        < 0.1
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["dim"],
        x_vals=[128, 256, 512, 1024],
        line_arg="provider",
        line_vals=["torch", "triton_fp8"],
        line_names=["Torch", "Triton_FP8"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="fp8_gemm-performance",
        args={
            "dtype": torch.bfloat16,
            "block_size": 128,
        },
    )
)
def benchmark_fp8_gemm(dim, dtype, block_size, provider):
    torch.manual_seed(42)
    torch.set_default_dtype(dtype)
    device = torch.device("cuda")
    a = torch.randn(dim, dim, dtype=dtype, device=device)
    b, b_s = init_b_and_b_s(dim, block_size)

    if provider == "torch":
        dequant_b = weight_dequant_deepseek_v3(b, b_s)
        ms = triton.testing.do_bench(lambda: torch.nn.functional.linear(a, dequant_b))
    if provider == "triton_fp8":
        a_fp8, a_s = act_quant_deepseek_v3(a, block_size)
        ms = triton.testing.do_bench(lambda: fp8_gemm_deepseek_v3(a_fp8, a_s, b, b_s))
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

    dequant_b = weight_dequant_soft_fp8_deepseek_v3(b, b_s)
    soft_dequant_b = weight_dequant_soft_fp8_deepseek_v3(b, b_s)

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

    dequant_b = weight_dequant_soft_fp8_deepseek_v3(b, b_s)
    std_y = torch.nn.functional.linear(a, dequant_b)
    y = soft_fp8_gemm_deepseek_v3(a, b, b_s)

    assert torch.allclose(std_y, y, atol=1e-2, rtol=1e-2)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["dim"],
        x_vals=[128, 256, 512, 1024],
        line_arg="provider",
        line_vals=["torch", "triton_soft_fp8"],
        line_names=["Torch", "Triton_Soft_FP8"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="soft_fp8_gemm-performance",
        args={
            "dtype": torch.bfloat16,
            "block_size": 128,
        },
    )
)
def benchmark_soft_fp8_gemm(dim, dtype, block_size, provider):
    torch.manual_seed(42)
    torch.set_default_dtype(dtype)
    device = torch.device("cuda")
    a = torch.randn(dim, dim, dtype=dtype, device=device)
    b, b_s = init_b_and_b_s(dim, block_size)

    if provider == "torch":
        dequant_b = weight_dequant_soft_fp8_deepseek_v3(b, b_s)
        ms = triton.testing.do_bench(lambda: torch.nn.functional.linear(a, dequant_b))
    if provider == "triton_soft_fp8":
        ms = triton.testing.do_bench(lambda: soft_fp8_gemm_deepseek_v3(a, b, b_s))
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

    ms = triton.testing.do_bench(lambda: weight_dequant_soft_fp8_deepseek_v3(b, b_s))
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

    ms = triton.testing.do_bench(lambda: weight_dequant_deepseek_v3(b, b_s))
    return ms * 1000


if __name__ == "__main__":
    benchmark_fp8_gemm.run(show_plots=True, print_data=True)
    benchmark_soft_fp8_gemm.run(show_plots=True, print_data=True)
    benchmark_soft_fp8_dequant.run(show_plots=True, print_data=True)
    benchmark_fp8_dequant.run(show_plots=True, print_data=True)
