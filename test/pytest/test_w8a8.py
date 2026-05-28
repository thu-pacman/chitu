import pytest
import torch
from omegaconf import OmegaConf

from chitu.global_vars import set_global_args
from chitu.lazy import eval_lazy
from chitu.ops import a8_per_token_act_quant, silu_and_mul
from chitu.import_utils import try_import_platform_dep, try_import_and_setup_torch_npu
from chitu.testing import assert_close, AssertOpCalled

lmslim, has_lmslim = try_import_platform_dep("lmslim")
lightop, has_lightop = try_import_platform_dep("lightop")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


@pytest.mark.parametrize("M", [0, 32])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("impl", ["hygon", "torch_npu"])
def test_a8_per_token_act_quant(M, N, impl, record_benchmark):
    if impl == "hygon" and not has_lmslim:
        pytest.skip("lmslim is missing")
    if impl == "torch_npu" and not has_torch_npu:
        pytest.skip("torch_npu is missing")

    x = torch.rand(M, N, device="cuda", dtype=torch.bfloat16)
    ref_y, ref_y_scale = a8_per_token_act_quant(x, impl="torch")

    y, y_scale = record_benchmark.run(
        lambda: a8_per_token_act_quant(x, impl=impl),
        M=M,
        N=N,
        impl=impl,
    )
    assert_close(y, ref_y, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M", [0, 32])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize(
    "silu_and_mul_impl, a8_per_token_act_quant_impl",
    [("torch", "torch"), ("triton", "hygon")],
)
def test_silu_and_mul_and_a8_per_token_act_quant(
    M, N, silu_and_mul_impl, a8_per_token_act_quant_impl, record_benchmark
):
    if a8_per_token_act_quant_impl == "hygon" and not has_lmslim:
        pytest.skip("lmslim is missing")

    set_global_args(
        OmegaConf.create({"infer": {"op_impl": "torch"}}), need_ensure=False
    )

    x = torch.rand(M, N * 2, device="cuda", dtype=torch.bfloat16)

    # Tested: maybe fused op via lazy
    def testee():
        if (
            a8_per_token_act_quant_impl == "hygon"
            and has_lightop
            and hasattr(lightop, "fuse_silu_mul_quant")
        ):
            expected_call_cnt = 1
        else:
            expected_call_cnt = 0
        with AssertOpCalled(
            "silu_and_mul_and_a8_per_token_act_quant",
            expected_call_cnt=expected_call_cnt,
        ):
            return a8_per_token_act_quant(
                silu_and_mul(x, impl=silu_and_mul_impl),
                impl=a8_per_token_act_quant_impl,
            )

    y, y_scale = record_benchmark.run(
        testee,
        M=M,
        N=N,
        silu_and_mul_impl=silu_and_mul_impl,
        a8_per_token_act_quant_impl=a8_per_token_act_quant_impl,
    )

    # Reference: explicitly non-fused op
    ref_y, ref_y_scale = a8_per_token_act_quant(
        eval_lazy(silu_and_mul(x, impl="torch")), impl="torch"
    )

    assert_close(y, ref_y, rtol=2e-2, atol=1e-2)
