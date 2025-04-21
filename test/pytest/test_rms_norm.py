from chitu.models.model import RMSNorm
import torch
import pytest


@pytest.mark.parametrize("compute_dtype", [torch.float32])
@pytest.mark.parametrize("dim", [64, 1024])
@pytest.mark.parametrize("head_dim", [256, 1024])
@pytest.mark.parametrize("impl", ["cuda", "triton", "torch"])
@torch.inference_mode()
def test_rms_norm(compute_dtype, dim, head_dim, impl):
    if impl == "torch" and not hasattr(torch.nn.functional, "rms_norm"):
        pytest.skip("The torch version does not support RMSNorm")

    torch.set_default_dtype(torch.float16)
    x = torch.rand(head_dim, dim).cuda()
    weight = torch.randn(dim)
    R = RMSNorm(dim, eps=1e-5).cuda()
    R.weight.copy_(weight)
    y = R(x, compute_dtype=compute_dtype, impl=impl)
    y_ref = R(x, compute_dtype=compute_dtype, impl="ref")
    assert torch.allclose(y, y_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("compute_dtype", [torch.float32])
@pytest.mark.parametrize("dim", [64])
@pytest.mark.parametrize("head_dim", [256])
@pytest.mark.parametrize(
    "impl", ["cuda", "torch", "ref"]
)  # Also test "ref"'s in-place with itself's out-of-place
@torch.inference_mode()
def test_rms_norm_in_place(compute_dtype, dim, head_dim, impl):
    if impl == "torch" and not hasattr(torch.nn.functional, "rms_norm"):
        pytest.skip("The torch version does not support RMSNorm")

    torch.set_default_dtype(torch.float16)
    x = torch.rand(head_dim, dim).cuda()
    weight = torch.randn(dim)
    R = RMSNorm(dim, eps=1e-5).cuda()
    R.weight.copy_(weight)
    y = x.clone()
    R(y, compute_dtype=compute_dtype, out=y, impl=impl)
    y_ref = R(x, compute_dtype=compute_dtype, impl="ref")
    assert torch.allclose(y, y_ref, rtol=1e-3, atol=1e-3)
