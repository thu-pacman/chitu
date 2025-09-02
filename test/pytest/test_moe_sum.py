import torch
import pytest

from chitu.ops import moe_sum
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")


@pytest.mark.parametrize("M", [32, 64, 128])
@pytest.mark.parametrize("N", [256, 512, 1024])
@pytest.mark.parametrize("compute_dtype", [torch.float16])
@pytest.mark.skipif(not has_triton, reason="triton is not available")
def test_moe_sum(M, N, compute_dtype):
    topk = 8
    input_tensor = torch.rand(M, topk, N, device="cuda", dtype=compute_dtype)
    output_tensor = torch.zeros(M, N, device="cuda", dtype=compute_dtype)
    moe_sum(input_tensor, output_tensor, impl="torch")
    triton_output = torch.zeros(M, N, device="cuda", dtype=compute_dtype)
    moe_sum(input_tensor, triton_output, impl="triton")
    assert torch.allclose(output_tensor, triton_output, rtol=1e-3, atol=1e-3)
