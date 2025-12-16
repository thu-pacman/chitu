import torch
import pytest

from chitu.models.model_hf_qwen3_next import Qwen3NextRMSNormGated
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")


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
@pytest.mark.parametrize("bsz", [256, 1024])
@pytest.mark.parametrize("impl", ["triton", "torch"])
@torch.inference_mode()
def test_rms_norm_gate(bsz, dim, impl, default_dtype, compute_dtype, weight_dtype):
    import torch.nn.functional as F

    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")

    torch.set_default_dtype(default_dtype)
    x = torch.randn(bsz, dim).cuda()
    gate = torch.randn(bsz, dim).cuda()
    weight = torch.randn(dim)
    NormGate = Qwen3NextRMSNormGated(dim, eps=1e-5, dtype=weight_dtype).cuda()
    NormGate.weight.copy_(weight)

    y_ref = NormGate(x.clone(), gate.clone(), compute_dtype=compute_dtype, impl="torch")
    y = NormGate(x.clone(), gate.clone(), compute_dtype=compute_dtype, impl=impl)

    if default_dtype == torch.bfloat16 or default_dtype == torch.float16:
        assert torch.allclose(y_ref, y, rtol=1e-2, atol=1e-2)
    else:
        assert torch.allclose(y_ref, y, rtol=1e-3, atol=1e-3)


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
@pytest.mark.parametrize("bsz", [256, 1024])
@pytest.mark.parametrize("impl", ["triton", "torch"])
@torch.inference_mode()
def test_rms_norm_gate_inplace(
    bsz, dim, impl, default_dtype, compute_dtype, weight_dtype
):
    import torch.nn.functional as F

    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")

    torch.set_default_dtype(default_dtype)
    x = torch.randn(bsz, dim).cuda()
    gate = torch.randn(bsz, dim).cuda()
    weight = torch.randn(dim)
    NormGate = Qwen3NextRMSNormGated(dim, eps=1e-5, dtype=weight_dtype).cuda()
    NormGate.weight.copy_(weight)

    y_ref = NormGate(x.clone(), gate.clone(), compute_dtype=compute_dtype, impl="torch")
    y = x.clone()
    NormGate(x.clone(), gate.clone(), out=y, compute_dtype=compute_dtype, impl=impl)

    if default_dtype == torch.bfloat16 or default_dtype == torch.float16:
        assert torch.allclose(y_ref, y, rtol=1e-2, atol=1e-2)
    else:
        assert torch.allclose(y_ref, y, rtol=1e-3, atol=1e-3)
