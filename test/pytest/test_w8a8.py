import pytest
import torch

from chitu.ops import a8_per_token_act_quant
from chitu.import_utils import try_import_platform_dep, try_import_and_setup_torch_npu
from chitu.testing import assert_close

lmslim, has_lmslim = try_import_platform_dep("lmslim")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


@pytest.mark.parametrize("M", [0, 32])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("impl", ["hygon", "torch_npu"])
def test_w8a8_gemm_per_token_per_channel(M, N, impl, record_benchmark):
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
