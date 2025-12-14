import pytest
import torch

from chitu.ops import silu_and_mul
from chitu.lazy import eval_lazy
from chitu.utils import try_import_platform_dep, try_import_and_setup_torch_npu

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


@pytest.mark.parametrize("M", [32, 64, 128])
@pytest.mark.parametrize("N", [256, 512, 1024, 18944])
@pytest.mark.parametrize("impl", ["triton", "torch_npu"])
def test_silu_and_mul(M, N, impl):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    if impl == "torch_npu" and not has_torch_npu:
        pytest.skip("torch_npu is missing")

    torch.manual_seed(42)
    input_tensor = torch.rand(M, N, device="cuda", dtype=torch.bfloat16)
    baseline_result = eval_lazy(silu_and_mul(input_tensor, impl="torch"))
    result = eval_lazy(silu_and_mul(input_tensor, impl=impl))
    torch.testing.assert_close(result, baseline_result, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("E", [32, 128])
@pytest.mark.parametrize("M", [1, 128])
@pytest.mark.parametrize("N", [256, 512])
@pytest.mark.parametrize("impl", ["triton"])
def test_silu_and_mul_with_expert_mask(E, M, N, impl):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")

    torch.manual_seed(42)
    input_tensor = torch.rand(E, M, N, device="cuda", dtype=torch.bfloat16)
    expert_n_tokens = torch.randint(
        low=0, high=M, size=(E,), device="cuda", dtype=torch.int32
    )
    baseline_result = eval_lazy(
        silu_and_mul(input_tensor, expert_n_tokens=expert_n_tokens, impl="torch")
    )
    result = eval_lazy(
        silu_and_mul(input_tensor, expert_n_tokens=expert_n_tokens, impl=impl)
    )

    # Zero out non-data elements
    mask = torch.arange(M, device="cuda", dtype=torch.int32).repeat(
        E, 1
    ) < expert_n_tokens.view(E, 1)
    baseline_result[~mask] = 0
    result[~mask] = 0

    torch.testing.assert_close(result, baseline_result, rtol=1e-2, atol=1e-2)
