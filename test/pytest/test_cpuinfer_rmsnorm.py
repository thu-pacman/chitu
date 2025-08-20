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


def rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    normalized = x / rms
    return normalized * weight


def cpuinfer_rms_norm(input_tensor, output_tensor, CPUInfer, rmsnorm):
    CPUInfer.submit(
        rmsnorm.forward(
            input_tensor.size(0), input_tensor.data_ptr(), output_tensor.data_ptr()
        )
    )
    CPUInfer.sync()
    return output_tensor


@pytest.mark.skipif(not cpuinfer_available, reason=cpuinfer_skip_reason)
@pytest.mark.parametrize("input_size", [512, 1024, 4096])
@pytest.mark.parametrize("qlen", [1, 10, 32])
@pytest.mark.parametrize("compute_dtype", [torch.bfloat16])
def test_rmsnorm(input_size, qlen, compute_dtype):
    group_max_len = 1024
    weight_type = 30
    hidden_type = 30
    eps = 1e-6

    weight = torch.randn((input_size,), dtype=compute_dtype).contiguous()

    config = cpuinfer.rmsnorm.RMSNormConfig(
        input_size,
        group_max_len,
        eps,
        weight.data_ptr(),
        hidden_type,
        weight_type,
        hidden_type,
    )
    rmsnorm = cpuinfer.rmsnorm.RMSNorm(config)
    CPUInfer = cpuinfer.CPUInfer("physical_core")

    input_tensor = torch.randn((qlen, input_size), dtype=compute_dtype).contiguous()
    cpuinfer_output = torch.empty((qlen, input_size), dtype=compute_dtype).contiguous()

    cpuinfer_rms_norm(input_tensor, cpuinfer_output, CPUInfer, rmsnorm)

    torch_output = rms_norm(input_tensor, weight, eps)

    diff = torch.mean(torch.abs(cpuinfer_output - torch_output)) / torch.mean(
        torch.abs(torch_output)
    )
    assert diff < 0.01


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["input_size"],
        x_vals=[512, 1024, 2048, 4096],
        line_arg="provider",
        line_vals=["torch", "cpuinfer"],
        line_names=["Torch", "CPUInfer"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="rmsnorm-performance",
        args={"compute_dtype": torch.bfloat16, "qlen": 32},
    )
)
def benchmark(input_size, qlen, compute_dtype, provider):
    if provider == "cpuinfer" and not cpuinfer_available:
        return float("nan")

    group_max_len = 1024
    weight_type = 30
    hidden_type = 30
    eps = 1e-6

    weight = torch.randn((input_size,), dtype=compute_dtype).contiguous()
    input_tensor = torch.randn((qlen, input_size), dtype=compute_dtype).contiguous()
    output_tensor = torch.empty((qlen, input_size), dtype=compute_dtype).contiguous()

    if provider == "torch":
        ms = triton.testing.do_bench(lambda: rms_norm(input_tensor, weight, eps))
    elif provider == "cpuinfer":
        config = cpuinfer.rmsnorm.RMSNormConfig(
            input_size,
            group_max_len,
            eps,
            weight.data_ptr(),
            hidden_type,
            weight_type,
            hidden_type,
        )
        rmsnorm = cpuinfer.rmsnorm.RMSNorm(config)
        CPUInfer = cpuinfer.CPUInfer("physical_core")

        ms = triton.testing.do_bench(
            lambda: cpuinfer_rms_norm(input_tensor, output_tensor, CPUInfer, rmsnorm)
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return ms * 1000


if __name__ == "__main__":
    if cpuinfer_available:
        benchmark.run(show_plots=True, print_data=True)
    else:
        print(f"Skipping benchmark: {cpuinfer_skip_reason}")
