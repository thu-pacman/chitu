# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")).split(
        ":", 1
    )[0]
    != "gfx936",
    reason="Requires Hygon gfx936",
)


@pytest.fixture
def ops():
    pytest.importorskip("triton")
    pytest.importorskip("lmslim.quantize.quant_ops")
    old_dtype = torch.get_default_dtype()
    old_threads = torch.get_num_threads()
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(1)
    torch.manual_seed(173)
    yield importlib.import_module("chitu.ops.quant.w8a8_per_token_per_channel")
    torch.set_default_dtype(old_dtype)
    torch.set_num_threads(old_threads)


@pytest.fixture
def external():
    return pytest.importorskip("hygon_w8a8_kernel.dense")


@pytest.mark.parametrize(
    "m,n,k", [(1, 16384, 2048), (12, 6144, 2048), (48, 2752, 6144)]
)
def test_auto_dispatches_to_external_gemm(ops, external, monkeypatch, m, n, k):
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    a, a_scale = ops.a8_per_token_act_quant(x, impl="hygon")
    b = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    b_scale = torch.full((n,), 1 / 64, device="cuda", dtype=torch.float32)
    calls = []
    original = ops.hygon_w8a8.small_m_gemm

    def counted(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(ops.hygon_w8a8, "small_m_gemm", counted)
    out = ops.w8a8_gemm_per_token_per_channel(a, a_scale, b, b_scale, impl="auto")
    assert len(calls) == 1
    reference = ops.w8a8_gemm_per_token_per_channel(
        a, a_scale, b, b_scale, impl="hipblaslt_w8a8_gemm"
    )
    assert len(calls) == 1
    torch.testing.assert_close(out, reference, rtol=1 / 128, atol=1e-4)


def test_missing_external_module_keeps_native_backend(ops, monkeypatch):
    class MissingModule:
        def __getattr__(self, name):
            raise AssertionError("The unavailable optional kernel module was accessed")

    monkeypatch.setattr(ops, "has_hygon_w8a8", False)
    monkeypatch.setattr(ops, "hygon_w8a8", MissingModule())
    assert ops.w8a8_gemm_per_token_per_channel.resolve_impl() == "hipblaslt_w8a8_gemm"
    a = torch.ones((1, 2048), device="cuda", dtype=torch.int8)
    b = torch.ones((16384, 2048), device="cuda", dtype=torch.int8)
    a_scale = torch.ones(1, device="cuda", dtype=torch.float32)
    b_scale = torch.ones(16384, device="cuda", dtype=torch.float32)
    out = ops.w8a8_gemm_per_token_per_channel(a, a_scale, b, b_scale, impl="auto")
    reference = ops.w8a8_gemm_per_token_per_channel(
        a, a_scale, b, b_scale, impl="hipblaslt_w8a8_gemm"
    )
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
    # An explicit Hygon selection also keeps the native fallback contract.
    explicit = ops.w8a8_gemm_per_token_per_channel(
        a, a_scale, b, b_scale, impl="hygon_w8a8_gemm"
    )
    torch.testing.assert_close(explicit, reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (0, 16384, 2048),
        (49, 16384, 2048),
        (6, 6144, 16384),
    ],
)
def test_other_projections_keep_original_backend(ops, monkeypatch, m, n, k):
    a = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    a_scale = torch.rand(m, device="cuda", dtype=torch.float32) * 0.01
    b_scale = torch.rand(n, device="cuda", dtype=torch.float32) * 0.01
    original = ops.w8a8_gemm_per_token_per_channel_hipblaslt_w8a8_gemm
    calls = []

    def counted(*args):
        calls.append(True)
        return original(*args)

    def unexpected(*args):
        raise AssertionError("Unvalidated projection reached the specialized kernel")

    monkeypatch.setattr(
        ops, "w8a8_gemm_per_token_per_channel_hipblaslt_w8a8_gemm", counted
    )
    if ops.has_hygon_w8a8:
        monkeypatch.setattr(ops.hygon_w8a8, "small_m_gemm", unexpected)
    out = ops.w8a8_gemm_per_token_per_channel(a, a_scale, b, b_scale, impl="auto")
    assert len(calls) == 1
    reference = original(a, a_scale, b, b_scale)
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
