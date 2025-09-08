import torch
import pytest

from chitu.utils import try_import_platform_dep, try_import_opt_dep

triton, has_triton = try_import_platform_dep("triton")
cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")


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


@pytest.mark.skipif(not has_cpuinfer, reason="cpuinfer module not available")
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

    assert torch.allclose(cpuinfer_output, torch_output, rtol=1e-2, atol=1e-2)
