import torch
import pytest
import triton

from chitu.utils import try_import_opt_dep

w8a8gemv, has_w8a8gemv = try_import_opt_dep("w8a8gemv", "quant")
w8a8gemm, has_w8a8gemm = try_import_opt_dep("w8a8gemm", "quant")


@pytest.mark.skipif(
    not has_w8a8gemm, reason="Optional dependency [quant] is not installed."
)
def test_w8a8gemm():
    torch.manual_seed(0)

    m = 1024
    n = 2048
    k = 4096
    a = (torch.randn([m, k], device="cuda") * 4).to(torch.int8)
    b = (torch.randn([n, k], device="cuda") * 4).to(torch.int8)

    c = torch.zeros([m, n], dtype=torch.float16, device="cuda")
    a_scales = torch.ones([m], device="cuda").to(torch.float)
    b_scales = torch.ones([n], device="cuda").to(torch.float)

    w8a8gemm.mm(c, a, b, a_scales, b_scales, None)

    c1 = torch.mm(a.to(torch.float16), b.transpose(0, 1).to(torch.float16))
    assert torch.allclose(c, c1, rtol=5e-3, atol=5e-3)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[
            (128, 512, 4096),
            (256, 1024, 4096),
            (512, 2048, 4096),
            (1024, 2048, 4096),
        ],
        line_arg="provider",
        line_vals=["torch", "w8a8gemm"],
        line_names=["Torch", "W8A8GEMM"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="w8a8gemm-performance",
        args={
            "dtype": torch.float16,
        },
    )
)
def benchmark_w8a8gemm(M, N, K, dtype, provider):
    torch.manual_seed(42)
    torch.set_default_dtype(dtype)
    device = torch.device("cuda")
    a = (torch.randn(M, K, device=device) * 4).to(torch.int8)
    b = (torch.randn(N, K, device=device) * 4).to(torch.int8)
    c = torch.zeros([M, N], dtype=dtype, device=device)
    a_scales = torch.ones([M], device=device).to(torch.float)
    b_scales = torch.ones([N], device=device).to(torch.float)

    if provider == "torch":
        ms = triton.testing.do_bench(
            lambda: torch.mm(a.to(dtype), b.transpose(0, 1).to(dtype))
        )
    if provider == "w8a8gemm":
        ms = triton.testing.do_bench(
            lambda: w8a8gemm.mm(c, a, b, a_scales, b_scales, None)
        )
    return ms * 1000


@pytest.mark.skipif(
    not has_w8a8gemv, reason="Optional dependency [quant] is not installed."
)
def test_w8a8gemv():
    torch.manual_seed(0)

    a = (torch.randn([2, 1, 11008], device="cuda") * 4).to(torch.int8)
    b = (torch.randn([4096, 11008], device="cuda") * 4).to(torch.int8)
    sclt = torch.ones([2], dtype=torch.float32, device="cuda")
    sclcl = torch.ones([4096], dtype=torch.float32, device="cuda")

    c = w8a8gemv.mv(a, b, sclt, sclcl)
    c0 = torch.mm(
        a.reshape(2, 11008).to(torch.float16), b.transpose(0, 1).to(torch.float16)
    ).reshape(2, 1, 4096)
    assert torch.allclose(c, c0, rtol=5e-3, atol=5e-3)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["dim"],
        x_vals=[11008],
        line_arg="provider",
        line_vals=["torch", "w8a8gemv"],
        line_names=["Torch", "W8A8GEMV"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="w8a8gemv-performance",
        args={
            "dtype": torch.float16,
        },
    )
)
def benchmark_w8a8gemv(dim, dtype, provider):
    torch.manual_seed(42)
    torch.set_default_dtype(dtype)
    device = torch.device("cuda")
    a = (torch.randn([2, 1, dim], device=device) * 4).to(torch.int8)
    b = (torch.randn([4096, dim], device=device) * 4).to(torch.int8)
    a_scales = torch.ones([2], device=device).to(torch.float)
    b_scales = torch.ones([4096], device=device).to(torch.float)

    if provider == "torch":
        ms = triton.testing.do_bench(
            lambda: torch.mm(
                a.reshape(2, dim).to(dtype), b.transpose(0, 1).to(dtype)
            ).reshape(2, 1, 4096)
        )
    if provider == "w8a8gemv":
        ms = triton.testing.do_bench(lambda: w8a8gemv.mv(a, b, a_scales, b_scales))
    return ms * 1000


if __name__ == "__main__":
    benchmark_w8a8gemm.run(show_plots=True, print_data=True)
    benchmark_w8a8gemv.run(show_plots=True, print_data=True)
