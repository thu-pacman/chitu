from chitu.ops import silu_and_mul
import triton
import triton.language as tl
import torch


def test_silu_and_mul():
    torch.manual_seed(42)
    M_values = [32, 64, 128]
    N_values = [256, 512, 1024]

    for M in M_values:
        for N in N_values:
            input_tensor = torch.rand(M, N, device="cuda", dtype=torch.bfloat16)
            baseline_result = silu_and_mul(input_tensor, impl="torch")
            result = silu_and_mul(input_tensor, impl="triton")
            assert torch.allclose(
                baseline_result, result, rtol=1e-3, atol=1e-3
            ), f"Results don't match for shape M={M}, N={N}"

    print("all case pass")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(1, 9)],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=["triton", "torch"],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines
        styles=[("blue", "-"), ("green", "-")],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="silu_and_mul-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={"M": 4096},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, dtype=torch.bfloat16).cuda()
    DEVICE = x.device
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: silu_and_mul(x, impl="torch"))
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: silu_and_mul(x, impl="triton"))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


if __name__ == "__main__":
    test_silu_and_mul()
    benchmark.run(show_plots=True, print_data=True)
