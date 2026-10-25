import torch
import pytest

from chitu.models.model_hf_qwen3_next import Qwen3NextRMSNormGated
from chitu.utils import try_import_platform_dep
from chitu.testing import assert_close
from chitu.device_type import is_enflame

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
@pytest.mark.parametrize("bsz", [0, 256, 1024])
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
        assert_close(y_ref, y, rtol=1e-2, atol=1e-2)
    else:
        assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


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
@pytest.mark.parametrize("bsz", [0, 256, 1024])
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
        assert_close(y_ref, y, rtol=1e-2, atol=1e-2)
    else:
        assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not is_enflame(), reason="Enflame GCU is required")
@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize(
    "dual_widen,fused_square,fused_add_rsqrt",
    [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)],
)
@torch.inference_mode()
def test_qwen3_next_gated_rms_tail_is_exact(
    batch_size, dual_widen, fused_square, fused_add_rsqrt, monkeypatch
):
    torch.manual_seed(20260830 + batch_size)
    rows = batch_size * 48
    x = (torch.randn(rows, 128, device="cuda") * 0.25).bfloat16()
    gate = (torch.randn(rows, 128, device="cuda") * 1.5).bfloat16()
    norm = Qwen3NextRMSNormGated(128, eps=1.0e-6, dtype=torch.bfloat16).cuda()
    norm.weight.copy_((1.0 + torch.randn(128, device="cuda") * 0.05).bfloat16())

    monkeypatch.setenv("CHITU_GCU_QWEN3_NEXT_GATED_RMS_TAIL", "0")
    monkeypatch.setenv("CHITU_GCU_QWEN3_NEXT_GATED_RMS_DUAL_WIDEN", "0")
    monkeypatch.setenv("CHITU_GCU_QWEN3_NEXT_GATED_RMS_WIDEN_SQUARE", "0")
    monkeypatch.setenv("CHITU_GCU_QWEN3_NEXT_GATED_RMS_ADD_RSQRT", "0")
    expected = norm(x, gate, compute_dtype=torch.float32, impl="torch")
    monkeypatch.setenv("CHITU_GCU_QWEN3_NEXT_GATED_RMS_TAIL", "1")
    monkeypatch.setenv("CHITU_GCU_QWEN3_NEXT_GATED_RMS_DUAL_WIDEN", str(dual_widen))
    monkeypatch.setenv("CHITU_GCU_QWEN3_NEXT_GATED_RMS_WIDEN_SQUARE", str(fused_square))
    monkeypatch.setenv("CHITU_GCU_QWEN3_NEXT_GATED_RMS_ADD_RSQRT", str(fused_add_rsqrt))
    actual = norm(x, gate, compute_dtype=torch.float32, impl="torch")

    assert torch.equal(actual, expected)
