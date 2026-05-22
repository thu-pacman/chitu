import pytest
import torch

from chitu.ops import silu_and_mul
from chitu.lazy import eval_lazy
from chitu.utils import try_import_platform_dep, try_import_and_setup_torch_npu
from chitu.testing import assert_close

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


@pytest.mark.parametrize("M", [0, 2])
@pytest.mark.parametrize("N", [6, 256])
def test_silu_and_mul_swiglu_limit_torch(M, N):
    swiglu_limit = 10.0
    input_tensor = torch.linspace(
        -2 * swiglu_limit,
        2 * swiglu_limit,
        steps=M * N,
        dtype=torch.float32,
    ).view(M, N)
    gate, up = input_tensor.chunk(2, dim=-1)
    baseline_result = torch.nn.functional.silu(
        torch.clamp(gate, max=swiglu_limit)
    ) * torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)

    result = eval_lazy(
        silu_and_mul(input_tensor, swiglu_limit=swiglu_limit, impl="torch")
    )
    assert_close(result, baseline_result, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("M", [0, 32, 64, 128])
@pytest.mark.parametrize("N", [256, 512, 1024, 18944])
@pytest.mark.parametrize("impl", ["triton", "torch_npu"])
def test_silu_and_mul(M, N, impl, record_benchmark):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    if impl == "torch_npu" and not has_torch_npu:
        pytest.skip("torch_npu is missing")

    torch.manual_seed(42)
    input_tensor = torch.rand(M, N, device="cuda", dtype=torch.bfloat16)
    baseline_result = eval_lazy(silu_and_mul(input_tensor, impl="torch"))

    result = record_benchmark.run(
        lambda: eval_lazy(silu_and_mul(input_tensor, impl=impl)),
        N=N,
        impl=impl,
    )
    assert_close(result, baseline_result, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("E", [32, 128])
@pytest.mark.parametrize("M", [1, 128])
@pytest.mark.parametrize("N", [256, 512])
@pytest.mark.parametrize("impl", ["triton"])
def test_silu_and_mul_with_expert_mask(E, M, N, impl, record_benchmark):
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

    result = record_benchmark.run(
        lambda: eval_lazy(
            silu_and_mul(input_tensor, expert_n_tokens=expert_n_tokens, impl=impl)
        ),
        N=N,
        impl=impl,
    )

    # Zero out non-data elements
    mask = torch.arange(M, device="cuda", dtype=torch.int32).repeat(
        E, 1
    ) < expert_n_tokens.view(E, 1)
    baseline_result[~mask] = 0
    result[~mask] = 0

    assert_close(result, baseline_result, rtol=1e-2, atol=1e-2)
