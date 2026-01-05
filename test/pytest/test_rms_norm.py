import torch
import pytest

from chitu.models.model import RMSNorm
from chitu.utils import try_import_platform_dep, try_import_and_setup_torch_npu
from chitu.testing import assert_close

triton, has_triton = try_import_platform_dep("triton")
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


@pytest.mark.parametrize(
    "default_dtype,weight_dtype",
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
    ],
)
@pytest.mark.parametrize("compute_dtype", [torch.float32])
@pytest.mark.parametrize("dim", [64, 1024])
@pytest.mark.parametrize("bs", [0, 256, 1024])
@pytest.mark.parametrize("impl", ["cuda", "triton", "torch", "torch_npu"])
@torch.inference_mode()
def test_rms_norm(
    default_dtype, weight_dtype, compute_dtype, dim, bs, impl, record_benchmark
):
    if impl == "torch" and not hasattr(torch.nn.functional, "rms_norm"):
        pytest.skip("The torch version does not support RMSNorm")
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    if impl == "cuda" and not has_chitu_backend:
        pytest.skip("chitu_backend is not available, skipping CUDA tests")
    if impl == "torch_npu" and not has_torch_npu:
        pytest.skip("torch_npu is missing")

    torch.set_default_dtype(default_dtype)
    x = torch.rand(bs, dim).cuda()
    weight = torch.randn(dim, dtype=weight_dtype).cuda()
    R = RMSNorm(dim, eps=1e-5, dtype=weight_dtype).cuda()
    R.weight.copy_(weight)

    y = record_benchmark.run(
        lambda: R(x, compute_dtype=compute_dtype, impl=impl),
        dim=dim,
        impl=impl,
    )
    y_ref = R(x, compute_dtype=compute_dtype, impl="ref")
    if default_dtype == torch.bfloat16:
        assert_close(y, y_ref, rtol=1e-2, atol=1e-2)
    else:
        assert_close(y, y_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("compute_dtype", [torch.float32])
@pytest.mark.parametrize("dim", [64, 1536, 512, 7168])
@pytest.mark.parametrize("bs", [0, 256])
@pytest.mark.parametrize("impl", ["cuda", "torch", "torch_npu", "ref"])
@torch.inference_mode()
def test_rms_norm_in_place(compute_dtype, dim, bs, impl, record_benchmark):
    if impl == "torch" and not hasattr(torch.nn.functional, "rms_norm"):
        pytest.skip("The torch version does not support RMSNorm")
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    if impl == "cuda" and not has_chitu_backend:
        pytest.skip("chitu_backend is not available, skipping CUDA tests")
    if impl == "torch_npu" and not has_torch_npu:
        pytest.skip("torch_npu is missing")

    torch.set_default_dtype(torch.float16)
    x = torch.rand(bs, dim).cuda()
    weight = torch.randn(dim)
    R = RMSNorm(dim, eps=1e-5).cuda()
    R.weight.copy_(weight)

    def bench_impl(impl_name):
        out = x.clone()
        return R(out, compute_dtype=compute_dtype, out=out, impl=impl_name)

    y = record_benchmark.run(
        lambda: bench_impl(impl),
        dim=dim,
        impl=impl,
    )
    y_ref = R(x, compute_dtype=compute_dtype, impl="ref")
    assert_close(y, y_ref, rtol=1e-3, atol=1e-3)
