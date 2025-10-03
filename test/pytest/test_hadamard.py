import torch
import pytest

from chitu.ops import hadamard_transform
from chitu.utils import try_import_opt_dep

scipy, has_scipy = try_import_opt_dep("scipy", "scipy")
fast_hadamard_transform, has_fast_hadamard_transform = try_import_opt_dep(
    "fast_hadamard_transform", "fast_hadamard_transform"
)


@pytest.mark.parametrize("bs", [8])
@pytest.mark.parametrize("dim", [192])
@pytest.mark.skipif(
    not has_scipy or not has_fast_hadamard_transform,
    reason="scipy or fast_hadamard_transform is not available",
)
def test_hadamard_transform(bs, dim):
    x = torch.randn(bs, dim, dtype=torch.bfloat16, device="cuda")
    scale = dim**-0.5
    y = hadamard_transform(x, scale, impl="fast_hadamard_transform")
    y_ref = hadamard_transform(x, scale, impl="scipy")
    assert torch.allclose(y, y_ref, rtol=1e-2, atol=1e-2)
