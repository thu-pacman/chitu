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


def silu_and_mul_torch(x):
    d = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :d]) * x[..., d:]


def cpuinfer_silu_and_mul(input_tensor, CPUInfer, silu_and_mul):
    half_size = input_tensor.shape[-1] // 2
    output_tensor = torch.empty(
        (input_tensor.shape[0], half_size), dtype=input_tensor.dtype
    ).contiguous()

    CPUInfer.submit(
        silu_and_mul.forward(
            input_tensor.shape[0], input_tensor.data_ptr(), output_tensor.data_ptr()
        )
    )
    CPUInfer.sync()

    return output_tensor


@pytest.mark.skipif(not cpuinfer_available, reason=cpuinfer_skip_reason)
@pytest.mark.parametrize("input_size", [512, 1024, 8192])
@pytest.mark.parametrize("qlen", [1, 16, 30])
@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
def test_silu_and_mul(input_size, qlen, compute_dtype):
    group_max_len = 1024
    hidden_type = 30
    if compute_dtype == torch.float32:
        hidden_type = 0

    CPUInfer = cpuinfer.CPUInfer("physical_core")

    config = cpuinfer.silu_and_mul.SiluAndMulConfig(
        input_size,
        group_max_len,
        hidden_type,
    )
    silu_and_mul = cpuinfer.silu_and_mul.SiluAndMul(config)

    input_tensor = (
        torch.randn((qlen, input_size), dtype=compute_dtype).contiguous() / 100
    )

    cpuinfer_output = cpuinfer_silu_and_mul(input_tensor, CPUInfer, silu_and_mul)

    torch_output = silu_and_mul_torch(input_tensor)

    diff = torch.mean(torch.abs(cpuinfer_output - torch_output)) / torch.mean(
        torch.abs(torch_output)
    )
    assert diff < 0.01, f"Difference too large: {diff}"


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["input_size"],
        x_vals=[512, 1024, 2048, 4096, 8192],
        line_arg="provider",
        line_vals=["torch", "cpuinfer"],
        line_names=["Torch", "CPUInfer"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="silu-and-mul-performance",
        args={"compute_dtype": torch.float32, "qlen": 32},
    )
)
def benchmark(input_size, qlen, compute_dtype, provider):
    if provider == "cpuinfer" and not cpuinfer_available:
        return float("nan")

    group_max_len = 1024
    hidden_type = 0

    input_tensor = (
        torch.randn((qlen, input_size), dtype=compute_dtype).contiguous() / 100
    )

    if provider == "torch":
        ms = triton.testing.do_bench(lambda: silu_and_mul_torch(input_tensor))
    elif provider == "cpuinfer":
        CPUInfer = cpuinfer.CPUInfer("physical_core")
        config = cpuinfer.silu_and_mul.SiluAndMulConfig(
            input_size,
            group_max_len,
            hidden_type,
        )
        silu_and_mul = cpuinfer.silu_and_mul.SiluAndMul(config)

        ms = triton.testing.do_bench(
            lambda: cpuinfer_silu_and_mul(input_tensor, CPUInfer, silu_and_mul)
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return ms * 1000


if __name__ == "__main__":
    if cpuinfer_available:
        benchmark.run(show_plots=True, print_data=True)
    else:
        print(f"Skipping benchmark: {cpuinfer_skip_reason}")
