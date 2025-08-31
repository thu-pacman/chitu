import torch
import pytest

from chitu.utils import try_import_platform_dep, try_import_opt_dep
from chitu.ops.norm import rms_norm_torch

triton, has_triton = try_import_platform_dep("triton")
cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")


def cpuinfer_rms_norm(input_tensor, output_tensor, CPUInfer, rmsnorm):
    CPUInfer.submit(
        rmsnorm.forward(
            input_tensor.size(0), input_tensor.data_ptr(), output_tensor.data_ptr()
        )
    )
    CPUInfer.sync()
    return output_tensor


@pytest.mark.skipif(not has_cpuinfer, reason="cpuinfer module not available")
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

    torch_output = rms_norm_torch(
        input_tensor, weight, compute_dtype=torch.float32, eps=eps
    )

    diff = torch.mean(torch.abs(cpuinfer_output - torch_output)) / torch.mean(
        torch.abs(torch_output)
    )
    assert diff < 0.01
