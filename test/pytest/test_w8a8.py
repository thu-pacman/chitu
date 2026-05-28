import pytest
import torch
from omegaconf import OmegaConf

from chitu.global_vars import set_global_args
from chitu.lazy import eval_lazy
from chitu.ops import (
    a8_per_token_act_quant,
    silu_and_mul,
    w8a8_gemm_per_token_per_channel,
)
from chitu.import_utils import try_import_platform_dep, try_import_and_setup_torch_npu
from chitu.testing import assert_close, AssertOpCalled

triton, has_triton = try_import_platform_dep("triton")
lmslim, has_lmslim = try_import_platform_dep("lmslim")
lightop, has_lightop = try_import_platform_dep("lightop")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()

# Although imported lmslim and lightop, some version of them does not contain the following
# functions. So treat the following functions with `try_import_platform_dep`, too.
lmslim_quant_ops, has_lmslim_quant_ops = try_import_platform_dep(
    "lmslim.quantize.quant_ops"
)


def init_b_and_b_s(dim):
    b = torch.randn(dim, dim, dtype=torch.float32, device="cuda")
    b_s = b.amax(dim=-1, keepdim=True)
    b /= b_s
    return b.to(torch.int8), b_s


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


@pytest.mark.parametrize("bs", [0, 1, 256])
@pytest.mark.parametrize("dim", [256])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "impl", ["triton", "torch_npu", "hipblaslt_w8a8_gemm", "lightop_gemm_w8a8_smooth"]
)
def test_dequanted_gemm_is_close_to_w8a8_gemm(
    bs, dim, dtype: torch.dtype, impl, record_benchmark
):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    if impl == "torch_npu" and not has_torch_npu:
        pytest.skip("torch_npu is missing")
    if impl == "hipblaslt_w8a8_gemm" and not (
        has_lmslim_quant_ops and hasattr(lmslim_quant_ops, "hipblaslt_w8a8_gemm")
    ):
        pytest.skip(
            "lmslim is missing or lsmlim does not have lsmlim.quantize.quant_ops.hipblaslt_w8a8_gemm"
        )
    if impl == "lightop_gemm_w8a8_smooth" and not (
        has_lightop and hasattr(lightop, "gemm_w8a8_smooth")
    ):
        pytest.skip(
            "lightop is missing or lightop does not have lightop.gemm_w8a8_smooth"
        )

    if impl == "lightop_gemm_w8a8_smooth" and bs == 1:
        pytest.skip("lightop_gemm_w8a8_smooth is known to fail with bs=1")

    torch.set_default_dtype(dtype)
    a = torch.randn(bs, dim, dtype=dtype, device="cuda")
    b, b_s = init_b_and_b_s(dim)

    a_int8, a_s = a8_per_token_act_quant(a)

    y = record_benchmark.run(
        lambda: w8a8_gemm_per_token_per_channel(a_int8, a_s, b, b_s, impl=impl),
        bs=bs,
        dim=dim,
        impl=impl,
    )

    # Dequant from `a_int8` and `a_s` instead of directly using `a` in dequanted implementation,
    # so the numerical difference is controlled inside the kernels
    dequant_y = (
        torch.nn.functional.linear(a_int8.to(torch.float32), b.to(torch.float32))
        * a_s.view(bs, 1)
        * b_s.view(dim, 1).T
    ).to(dtype)

    assert_close(y, dequant_y, atol=0.1, rtol=0.1)
