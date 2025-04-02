from chitu.ops import rms_norm
import torch
import pytest


def naive_norm(x, weight, eps):
    dtype = x.dtype
    # x = x.to(torch.float32)
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y.to(dtype) * weight


@pytest.mark.parametrize("compute_dtype", [torch.float32])
@pytest.mark.parametrize("dim", [64, 1024])
@pytest.mark.parametrize("head_dim", [256, 1024])
def test_rms_norm(compute_dtype, dim, head_dim):
    x = torch.rand(head_dim, dim).cuda().to(compute_dtype)
    weight = torch.rand(dim).cuda().to(compute_dtype)
    y = rms_norm(x, weight, dim, 1e-5).to(torch.float32)
    y_ref = naive_norm(x, weight, 1e-5).to(torch.float32)
    assert torch.allclose(y, y_ref, rtol=1e-5, atol=1e-5)
