from chitu.models.model import RMSNorm
import torch
import triton
import triton.language as tl
import pytest


@pytest.mark.parametrize("compute_dtype", [torch.float32])
@pytest.mark.parametrize("dim", [64, 1024])
@pytest.mark.parametrize("head_dim", [256, 1024])
@pytest.mark.parametrize("impl", ["cuda", "triton", "torch"])
@torch.inference_mode()
def test_rms_norm(compute_dtype, dim, head_dim, impl):
    if impl == "torch" and not hasattr(torch.nn.functional, "rms_norm"):
        pytest.skip("The torch version does not support RMSNorm")

    torch.set_default_dtype(torch.float16)
    x = torch.rand(head_dim, dim).cuda()
    weight = torch.randn(dim)
    R = RMSNorm(dim, eps=1e-5).cuda()
    R.weight.copy_(weight)
    y = R(x, compute_dtype=compute_dtype, impl=impl)
    y_ref = R(x, compute_dtype=compute_dtype, impl="ref")
    assert torch.allclose(y, y_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("compute_dtype", [torch.float32])
@pytest.mark.parametrize("dim", [64, 1536, 512, 7168])
@pytest.mark.parametrize("head_dim", [256])
@pytest.mark.parametrize(
    "impl", ["cuda", "torch", "ref"]
)  # Also test "ref"'s in-place with itself's out-of-place
@torch.inference_mode()
def test_rms_norm_in_place(compute_dtype, dim, head_dim, impl):
    if impl == "torch" and not hasattr(torch.nn.functional, "rms_norm"):
        pytest.skip("The torch version does not support RMSNorm")

    torch.set_default_dtype(torch.float16)
    x = torch.rand(head_dim, dim).cuda()
    weight = torch.randn(dim)
    R = RMSNorm(dim, eps=1e-5).cuda()
    R.weight.copy_(weight)
    y = x.clone()
    R(y, compute_dtype=compute_dtype, out=y, impl=impl)
    y_ref = R(x, compute_dtype=compute_dtype, impl="ref")
    assert torch.allclose(y, y_ref, rtol=1e-3, atol=1e-3)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # argument names to use as an x-axis for the plot
        x_vals=[64, 512, 1536, 7168],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=["triton", "torch"],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines
        styles=[("blue", "-"), ("green", "-")],  # line styles
        ylabel="us",  # label name for the y-axis
        plot_name="rms_norm-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={
            "compute_dtype": torch.bfloat16
        },  # values for function arguments not in `x_names` and `y_name`
    )
)
@torch.inference_mode()
def benchmark(M, N, compute_dtype, provider):
    x = torch.rand(M, N).cuda()
    DEVICE = x.device
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    weight = torch.randn(N)
    R = RMSNorm(N, eps=1e-5).cuda()
    R.weight.copy_(weight)
    if provider == "torch":
        ms = triton.testing.do_bench(
            lambda: R(x, compute_dtype=compute_dtype, impl="torch")
        )
    if provider == "triton":
        ms = triton.testing.do_bench(
            lambda: R(x, compute_dtype=compute_dtype, impl="triton")
        )
    # gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return ms * 1000


if __name__ == "__main__":
    benchmark.run(M=1, show_plots=True, print_data=True)
    benchmark.run(M=64, show_plots=True, print_data=True)
    benchmark.run(M=512, show_plots=True, print_data=True)
