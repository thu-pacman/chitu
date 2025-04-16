from chitu.models.model import RMSNorm
import torch
import pytest


@pytest.mark.parametrize("compute_dtype", [torch.float32])
@pytest.mark.parametrize("dim", [64, 1024])
@pytest.mark.parametrize("head_dim", [256, 1024])
@pytest.mark.parametrize("impl", ["triton", "torch"])
@torch.inference_mode()
def test_rms_norm(compute_dtype, dim, head_dim, impl):
    if impl == "torch" and not hasattr(torch.nn.functional, "rms_norm"):
        pytest.skip("The torch version does not support RMSNorm")

    torch.set_default_dtype(torch.float16)
    x = torch.rand(head_dim, dim).cuda()
    weight = torch.randn(dim)
    R = RMSNorm(dim, eps=1e-5, impl=impl).cuda()
    R_ref = RMSNorm(dim, eps=1e-5, impl="ref").cuda()
    R.weight.copy_(weight)
    R_ref.weight.copy_(weight)
    y = R(x, compute_dtype=compute_dtype)
    y_ref = R_ref(x, compute_dtype=compute_dtype)
    assert torch.allclose(y, y_ref, rtol=1e-3, atol=1e-3)
