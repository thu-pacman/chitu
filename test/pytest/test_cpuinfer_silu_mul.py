import torch
import pytest

from chitu.utils import try_import_platform_dep, try_import_opt_dep
from chitu.ops.activation import silu_and_mul_torch

triton, has_triton = try_import_platform_dep("triton")
cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")


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


@pytest.mark.skipif(not has_cpuinfer, reason="cpuinfer module not available")
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
