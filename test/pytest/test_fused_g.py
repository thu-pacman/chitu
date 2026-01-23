import torch
import pytest

from chitu.ops.fused_g import fused_g
from chitu.utils import try_import_platform_dep
from chitu.testing import assert_close

triton, has_triton = try_import_platform_dep("triton")


@pytest.mark.parametrize(
    "default_dtype", [torch.float32, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize("dim", [4, 8, 16, 32, 128, 1024])
@pytest.mark.parametrize("bs", [0, 64, 256, 1024])
@pytest.mark.parametrize("impl", ["triton", "torch"])
@torch.inference_mode()
def test_fused_g(default_dtype, dim, bs, impl, record_benchmark):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")

    a = torch.randn(bs, dim, dtype=default_dtype).cuda()
    A_log = torch.randn(dim, dtype=default_dtype).cuda()
    dt_bias = torch.randn(dim, dtype=default_dtype).cuda()

    y = record_benchmark.run(
        lambda: fused_g(a, A_log, dt_bias, impl=impl),
        dim=dim,
        impl=impl,
    )

    y_ref = fused_g(a, A_log, dt_bias, impl="torch")

    assert_close(y, y_ref, rtol=1e-3, atol=1e-3)
