import torch
import pytest
import triton

from chitu.moe.experts.triton_fused_experts import moe_sum


def torch_moe_sum(input_tensor, output_tensor):
    output_tensor.copy_(input_tensor.sum(dim=1))


@pytest.mark.parametrize("M", [32, 64, 128])
@pytest.mark.parametrize("N", [256, 512, 1024])
@pytest.mark.parametrize("compute_dtype", [torch.float16])
def test_moe_sum(M, N, compute_dtype):
    topk = 8
    input_tensor = torch.rand(M, topk, N, device="cuda", dtype=compute_dtype)
    output_tensor = torch.zeros(M, N, device="cuda", dtype=compute_dtype)
    torch_moe_sum(input_tensor, output_tensor)
    triton_output = torch.zeros(M, N, device="cuda", dtype=compute_dtype)
    moe_sum(input_tensor, triton_output)
    assert torch.allclose(output_tensor, triton_output, rtol=1e-3, atol=1e-3)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[256, 512, 1024],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["Torch", "Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="moe_sum-performance",
        args={"compute_dtype": torch.bfloat16},
    )
)
def benchmark(M, N, compute_dtype, provider):
    topk = 8
    input_tensor = torch.rand(M, topk, N, device="cuda", dtype=compute_dtype)
    DEVICE = input_tensor.device
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    output_tensor = torch.zeros(M, N, device="cuda", dtype=compute_dtype)
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch_moe_sum(input_tensor, output_tensor))
    elif provider == "triton":
        ms = triton.testing.do_bench(lambda: moe_sum(input_tensor, output_tensor))
    else:
        raise ValueError(f"Unknown provider: {provider}")
    # gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return ms * 1000


if __name__ == "__main__":
    benchmark.run(M=1, show_plots=True, print_data=True)
    # benchmark.run(M=1,N=1024,compute_dtype=torch.float16,provider="triton",show_plots=True, print_data=True)
