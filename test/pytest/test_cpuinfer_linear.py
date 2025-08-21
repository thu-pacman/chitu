import torch
import pytest
import triton
import importlib

cpuinfer_available = importlib.util.find_spec("cpuinfer") is not None
if cpuinfer_available:
    try:
        import cpuinfer
    except (ImportError, ModuleNotFoundError):
        cpuinfer_available = False

cpuinfer_skip_reason = "cpuinfer module not available"


def torch_linear(input_tensor, weight):
    return torch.mm(input_tensor, weight.t())


def cpuinfer_linear(input_tensor, weight, output_tensor, CPUInfer, linear):
    CPUInfer.submit(
        linear.forward(
            input_tensor.size(0), input_tensor.data_ptr(), output_tensor.data_ptr()
        )
    )
    CPUInfer.sync()
    return output_tensor


@pytest.mark.skipif(not cpuinfer_available, reason=cpuinfer_skip_reason)
@pytest.mark.parametrize("input_size", [1024, 5120])
@pytest.mark.parametrize("output_size", [4096, 25600])
@pytest.mark.parametrize("qlen", [1, 8])
@pytest.mark.parametrize("compute_dtype", [torch.bfloat16])
def test_cpu_linear(input_size, output_size, qlen, compute_dtype):
    stride = 64
    group_max_len = 1024
    proj_type = 30
    hidden_type = 30

    proj = (
        torch.randn((output_size, input_size), dtype=compute_dtype)
        .to("cpu")
        .contiguous()
    )

    config = cpuinfer.linear.LinearConfig(
        input_size,
        output_size,
        stride,
        group_max_len,
        proj.data_ptr(),
        proj_type,
        hidden_type,
    )
    linear = cpuinfer.linear.Linear(config)
    CPUInfer = cpuinfer.CPUInfer("physical_core")

    input_tensor = (
        torch.randn((qlen, input_size), dtype=compute_dtype).contiguous() / 100
    )
    cpuinfer_output = torch.empty((qlen, output_size), dtype=compute_dtype).contiguous()

    cpuinfer_linear(input_tensor, proj, cpuinfer_output, CPUInfer, linear)

    torch_output = torch_linear(input_tensor, proj)

    diff = torch.mean(torch.abs(cpuinfer_output - torch_output)) / torch.mean(
        torch.abs(torch_output)
    )
    assert torch.allclose(cpuinfer_output, torch_output, rtol=1e-2, atol=1e-2)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["output_size"],
        x_vals=[4096, 8192, 25600],
        line_arg="provider",
        line_vals=["torch", "cpuinfer"],
        line_names=["Torch", "CPUInfer"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="linear-performance",
        args={"compute_dtype": torch.bfloat16, "input_size": 5120, "qlen": 1},
    )
)
def benchmark(input_size, output_size, qlen, compute_dtype, provider):
    stride = 64
    group_max_len = 1024
    proj_type = 30
    hidden_type = 30

    proj = (
        torch.randn((output_size, input_size), dtype=compute_dtype)
        .to("cpu")
        .contiguous()
    )
    input_tensor = (
        torch.randn((qlen, input_size), dtype=compute_dtype).contiguous() / 100
    )
    output_tensor = torch.empty((qlen, output_size), dtype=compute_dtype).contiguous()

    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch_linear(input_tensor, proj))
    elif provider == "cpuinfer":
        if not cpuinfer_available:
            return float("nan")

        config = cpuinfer.linear.LinearConfig(
            input_size,
            output_size,
            stride,
            group_max_len,
            proj.data_ptr(),
            proj_type,
            hidden_type,
        )
        linear = cpuinfer.linear.Linear(config)
        CPUInfer = cpuinfer.CPUInfer("physical_core")

        ms = triton.testing.do_bench(
            lambda: cpuinfer_linear(input_tensor, proj, output_tensor, CPUInfer, linear)
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return ms * 1000


if __name__ == "__main__":
    if cpuinfer_available:
        benchmark.run(show_plots=True, print_data=True)
    else:
        print(f"Skipping benchmark: {cpuinfer_skip_reason}")
