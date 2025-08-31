import pytest
import triton
import torch

from chitu.ops import silu_and_mul
from chitu.lazy import eval_lazy
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")


@pytest.mark.parametrize("M", [32, 64, 128])
@pytest.mark.parametrize("N", [256, 512, 1024, 18944])
def test_silu_and_mul(M, N):
    torch.manual_seed(42)
    input_tensor = torch.rand(M, N, device="cuda", dtype=torch.bfloat16)
    baseline_result = eval_lazy(silu_and_mul(input_tensor, impl="torch"))
    result = eval_lazy(silu_and_mul(input_tensor, impl="triton"))
    assert torch.allclose(
        baseline_result, result, rtol=1e-3, atol=1e-3
    ), f"Results don't match for shape M={M}, N={N}"
