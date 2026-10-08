# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import math
import torch
import chitu.ops.rotary as rotary_ops

from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.ops import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_partial,
    apply_rotary_pos_emb_single,
    apply_rotary_pos_emb_single_partial,
)
from chitu.native_layout import NativeLayoutTensor, ColumnOddEvenSeparatedTensor
from chitu.utils import try_import_platform_dep, try_import_and_setup_torch_npu
from chitu.testing import assert_close
from chitu.device_type import is_enflame

triton, has_triton = try_import_platform_dep("triton")
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


@pytest.mark.skipif(not is_enflame(), reason="Enflame GCU is required")
@pytest.mark.parametrize("rows", [1, 16, 127, 1024])
def test_apply_rotary_pos_emb_gcu_is_bit_exact_and_out_of_place(rows):
    q = torch.randn((rows, 40, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((rows, 8, 128), device="cuda", dtype=torch.bfloat16)
    q_before = q.clone()
    k_before = k.clone()
    angles = torch.randn((rows, 64), device="cuda", dtype=torch.bfloat16)
    freqs_cis = BatchedFreqsCis(torch.cos(angles), torch.sin(angles))

    q_ref, k_ref = apply_rotary_pos_emb(q, k, freqs_cis, impl="torch")
    q_actual, k_actual = apply_rotary_pos_emb(q, k, freqs_cis, impl="gcu")
    # Reuse the same BatchedFreqsCis to exercise its instance-owned vendor
    # metadata cache and ensure repeated calls do not retain mutable q/k state.
    q_replay, k_replay = apply_rotary_pos_emb(q, k, freqs_cis, impl="gcu")

    assert torch.equal(q, q_before)
    assert torch.equal(k, k_before)
    assert torch.equal(q_actual, q_ref)
    assert torch.equal(k_actual, k_ref)
    assert torch.equal(q_replay, q_ref)
    assert torch.equal(k_replay, k_ref)


@pytest.mark.skipif(not is_enflame(), reason="Enflame GCU is required")
@pytest.mark.parametrize("rows", [1, 16, 127, 1024])
def test_apply_rotary_pos_emb_gcu_auto_uses_permitted_inplace_output(rows):
    q = torch.randn((rows, 40, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((rows, 8, 128), device="cuda", dtype=torch.bfloat16)
    angles = torch.randn((rows, 64), device="cuda", dtype=torch.bfloat16)
    freqs_cis = BatchedFreqsCis(torch.cos(angles), torch.sin(angles))
    q_ref, k_ref = apply_rotary_pos_emb(
        q.clone(), k.clone(), freqs_cis, inplace=False, impl="torch"
    )
    q_ptr, k_ptr = q.data_ptr(), k.data_ptr()
    q_actual, k_actual = apply_rotary_pos_emb(q, k, freqs_cis, impl="auto")

    assert q_actual.data_ptr() == q_ptr
    assert k_actual.data_ptr() == k_ptr
    assert torch.equal(q_actual, q_ref)
    assert torch.equal(k_actual, k_ref)


@pytest.mark.skipif(not is_enflame(), reason="Enflame GCU is required")
@pytest.mark.parametrize("rows", [1, 16, 127, 1024])
def test_apply_rotary_pos_emb_partial_gcu_stages_strided_prefix_inplace(
    rows, monkeypatch
):
    monkeypatch.setattr(rotary_ops, "_GCU_ROTARY_ALLOW_STRIDED", True)
    q = torch.randn((rows, 40, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((rows, 8, 128), device="cuda", dtype=torch.bfloat16)
    angles = torch.randn((rows, 16), device="cuda", dtype=torch.bfloat16)
    freqs_cis = BatchedFreqsCis(torch.cos(angles), torch.sin(angles))
    q_ref, k_ref, *_ = apply_rotary_pos_emb_partial(
        q.clone(),
        k.clone(),
        freqs_cis,
        q_rotary_end=32,
        k_rotary_end=32,
        inplace=True,
        impl="torch",
    )
    q_ptr, k_ptr = q.data_ptr(), k.data_ptr()

    q_actual, k_actual, *_ = apply_rotary_pos_emb_partial(
        q,
        k,
        freqs_cis,
        q_rotary_end=32,
        k_rotary_end=32,
        inplace=True,
        impl="auto",
    )

    assert q_actual.data_ptr() == q_ptr
    assert k_actual.data_ptr() == k_ptr
    assert torch.equal(q_actual, q_ref)
    assert torch.equal(k_actual, k_ref)


@pytest.mark.skipif(not is_enflame(), reason="Enflame GCU is required")
def test_apply_rotary_pos_emb_gcu_auto_does_not_force_unsupported_inplace():
    rows = 16
    q = torch.randn((rows, 40, 128), device="cuda", dtype=torch.float16)
    k = torch.randn((rows, 8, 128), device="cuda", dtype=torch.float16)
    angles = torch.randn((rows, 64), device="cuda", dtype=torch.float16)
    freqs_cis = BatchedFreqsCis(torch.cos(angles), torch.sin(angles))
    q_before, k_before = q.clone(), k.clone()
    q_ref, k_ref = apply_rotary_pos_emb(
        q.clone(), k.clone(), freqs_cis, inplace=False, impl="torch"
    )
    if has_triton:
        assert (
            rotary_ops._dispatch_apply_rotary_pos_emb.resolve_impl(
                q, k, freqs_cis, impl="auto"
            )
            == "triton"
        )
    q_actual, k_actual = apply_rotary_pos_emb(q, k, freqs_cis, impl="auto")

    # FP16 is outside the vendor RoPE contract. ``inplace=True`` must not
    # manufacture output buffers before dispatch and thereby block Triton.
    assert q_actual.data_ptr() != q.data_ptr()
    assert k_actual.data_ptr() != k.data_ptr()
    assert torch.equal(q, q_before)
    assert torch.equal(k, k_before)
    assert_close(q_actual, q_ref)
    assert_close(k_actual, k_ref)


@pytest.mark.skipif(not is_enflame(), reason="Enflame GCU is required")
def test_apply_rotary_pos_emb_gcu_rejects_shared_and_crossed_storage():
    rows, heads, dim = 16, 8, 128
    elements = rows * heads * dim
    shared = torch.randn((elements * 2,), device="cuda", dtype=torch.bfloat16)
    q = shared[:elements].view(rows, heads, dim)
    k = shared[elements:].view(rows, heads, dim)
    angles = torch.randn((rows, dim // 2), device="cuda", dtype=torch.bfloat16)
    freqs_cis = BatchedFreqsCis(torch.cos(angles), torch.sin(angles))
    q_ref, k_ref = apply_rotary_pos_emb(
        q.clone(), k.clone(), freqs_cis, inplace=False, impl="torch"
    )

    # Different contiguous views can still share an allocation. The native
    # in-place kernel must fall back instead of assuming pointer independence.
    q_actual, k_actual = apply_rotary_pos_emb(q, k, freqs_cis, impl="gcu")
    assert q_actual.data_ptr() != q.data_ptr()
    assert k_actual.data_ptr() != k.data_ptr()
    assert torch.equal(q_actual, q_ref)
    assert torch.equal(k_actual, k_ref)

    q2, k2 = q.clone(), k.clone()
    q_actual, k_actual = apply_rotary_pos_emb(
        q2, k2, freqs_cis, q_out=k2, k_out=q2, impl="gcu"
    )
    assert q_actual is k2
    assert k_actual is q2
    assert torch.equal(q_actual, q_ref)
    assert torch.equal(k_actual, k_ref)


@pytest.mark.parametrize(
    "rotary_type,n_local_heads,head_dim",
    [
        ("separated", 64, 256),
        ("interleaved", 128, 64),
        ("separated-half", 32, 128),
        ("interleaved-half", 32, 128),
    ],
)
@pytest.mark.parametrize("batch_size", [0, 16, 64])
@pytest.mark.parametrize("is_mqa", [False, True])
@pytest.mark.parametrize(
    "impl", ["cuda", "triton", "torch_npu", "torch_npu_with_output_layout"]
)
@pytest.mark.parametrize(
    "qk_dtype,freqs_dtype",
    [(torch.float16, torch.float16), (torch.float16, torch.float32)],
)
def test_apply_rotary_pos_emb(
    rotary_type,
    batch_size,
    n_local_heads,
    head_dim,
    is_mqa,
    qk_dtype,
    freqs_dtype,
    impl,
    record_benchmark,
):
    if impl == "triton":
        if not has_triton:
            pytest.skip("triton is missing")
        if rotary_type in ["interleaved", "interleaved-half"] and not hasattr(
            triton.language, "interleave"
        ):
            pytest.skip("This op require Triton to support tl.interleave")
    if impl == "cuda" and not has_chitu_backend:
        pytest.skip("chitu_backend is not available, skipping CUDA tests")
    if impl in ["torch_npu", "torch_npu_with_output_layout"] and not has_torch_npu:
        pytest.skip("torch_npu is missing")
    if impl == "torch_npu_with_output_layout":
        if rotary_type != "interleaved":
            pytest.skip("torch_npu_with_output_layout only supports interleaved")
        if head_dim != 64:
            pytest.skip("torch_npu_with_output_layout only supports head_dim=64")
        if freqs_dtype == torch.float32:
            pytest.skip(
                "torch_npu_with_output_layout does not support freqs_dtype=float32"
            )

    torch.set_default_dtype(qk_dtype)
    q = torch.randn(batch_size, n_local_heads, head_dim, device="cuda")
    if is_mqa:
        k = torch.randn(batch_size, head_dim, device="cuda")
    else:
        k = torch.randn(batch_size, n_local_heads, head_dim, device="cuda")

    # Generate cos and sin from a unit circle
    if rotary_type in ["separated-half", "interleaved-half"]:
        precomp_head_dim = head_dim // 2
    else:
        precomp_head_dim = head_dim
    complex_freqs = torch.polar(
        torch.ones(
            batch_size, precomp_head_dim // 2, device="cuda", dtype=torch.float32
        ),
        torch.rand(
            batch_size, precomp_head_dim // 2, device="cuda", dtype=torch.float32
        )
        * 2
        * math.pi,
    )
    freqs_cis = BatchedFreqsCis(
        complex_freqs.real.contiguous().to(freqs_dtype),
        complex_freqs.imag.contiguous().to(freqs_dtype),
    )

    out_q, out_k = record_benchmark.run(
        lambda: apply_rotary_pos_emb(
            q, k, freqs_cis, rotary_type=rotary_type, impl=impl
        ),
        batch_size=batch_size,
        impl=impl,
    )
    out_q_torch, out_k_torch = apply_rotary_pos_emb(
        q, k, freqs_cis, rotary_type=rotary_type, impl="torch"
    )

    if isinstance(out_q, NativeLayoutTensor):
        out_q = out_q.convert_to_plain()
    if isinstance(out_k, NativeLayoutTensor):
        out_k = out_k.convert_to_plain()

    # Check if out_q and out_q_torch are the same
    # Use rtol and atol for more precise comparison
    rtol = 5e-3
    atol = 5e-3
    assert_close(out_q, out_q_torch, rtol=rtol, atol=atol)
    assert_close(out_k, out_k_torch, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "rotary_type,n_local_heads,head_dim",
    [
        ("separated", 64, 256),
        ("interleaved", 128, 64),
        ("separated-half", 32, 128),
        ("interleaved-half", 32, 128),
    ],
)
@pytest.mark.parametrize("batch_size", [0, 16, 64])
@pytest.mark.parametrize("is_mqa", [False, True])
@pytest.mark.parametrize(
    "qk_dtype,freqs_dtype",
    [(torch.float16, torch.float16), (torch.float16, torch.float32)],
)
@pytest.mark.parametrize(
    "impl", ["cuda", "triton", "torch_npu", "torch_npu_with_output_layout", "torch"]
)  # Also test "torch"'s in-place with itself's out-of-place
def test_apply_rotary_pos_emb_in_place(
    rotary_type,
    batch_size,
    n_local_heads,
    head_dim,
    is_mqa,
    qk_dtype,
    freqs_dtype,
    impl,
    record_benchmark,
):
    if impl == "triton":
        if not has_triton:
            pytest.skip("triton is missing")
        if rotary_type in ["interleaved", "interleaved-half"] and not hasattr(
            triton.language, "interleave"
        ):
            pytest.skip("This op require Triton to support tl.interleave")
    if impl == "cuda" and not has_chitu_backend:
        pytest.skip("chitu_backend is not available, skipping CUDA tests")
    if impl in ["torch_npu", "torch_npu_with_output_layout"] and not has_torch_npu:
        pytest.skip("torch_npu is missing")
    if impl == "torch_npu_with_output_layout":
        if rotary_type != "interleaved":
            pytest.skip("torch_npu_with_output_layout only supports interleaved")
        if head_dim != 64:
            pytest.skip("torch_npu_with_output_layout only supports head_dim=64")
        if freqs_dtype == torch.float32:
            pytest.skip(
                "torch_npu_with_output_layout does not support freqs_dtype=float32"
            )

    torch.set_default_dtype(qk_dtype)
    q = torch.randn(batch_size, n_local_heads, head_dim, device="cuda")
    if is_mqa:
        k = torch.randn(batch_size, head_dim, device="cuda")
    else:
        k = torch.randn(batch_size, n_local_heads, head_dim, device="cuda")

    # Generate cos and sin from a unit circle
    if rotary_type in ["separated-half", "interleaved-half"]:
        precomp_head_dim = head_dim // 2
    else:
        precomp_head_dim = head_dim
    complex_freqs = torch.polar(
        torch.ones(
            batch_size, precomp_head_dim // 2, device="cuda", dtype=torch.float32
        ),
        torch.rand(
            batch_size, precomp_head_dim // 2, device="cuda", dtype=torch.float32
        )
        * 2
        * math.pi,
    )
    freqs_cis = BatchedFreqsCis(
        complex_freqs.real.contiguous().to(freqs_dtype),
        complex_freqs.imag.contiguous().to(freqs_dtype),
    )

    q_clone = q.clone()
    k_clone = k.clone()
    if impl == "torch_npu_with_output_layout":
        out_q = ColumnOddEvenSeparatedTensor(
            plain_shape=q_clone.shape, layout_tensor=q_clone
        )
        out_k = ColumnOddEvenSeparatedTensor(
            plain_shape=k_clone.shape, layout_tensor=k_clone
        )
    else:
        out_q = q_clone
        out_k = k_clone

    def do_rotary_inplace():
        apply_rotary_pos_emb(
            q_clone,
            k_clone,
            freqs_cis,
            rotary_type=rotary_type,
            q_out=out_q,
            k_out=out_k,
            impl=impl,
        )
        return out_q, out_k

    out_q, out_k = record_benchmark.run(
        do_rotary_inplace,
        batch_size=batch_size,
        impl=impl,
    )
    out_q_torch, out_k_torch = apply_rotary_pos_emb(
        q, k, freqs_cis, rotary_type=rotary_type, impl="torch"
    )

    if isinstance(out_q, NativeLayoutTensor):
        out_q = out_q.convert_to_plain()
    if isinstance(out_k, NativeLayoutTensor):
        out_k = out_k.convert_to_plain()

    # Check if out_q and out_q_torch are the same
    # Use rtol and atol for more precise comparison
    rtol = 5e-3
    atol = 5e-3
    assert_close(out_q, out_q_torch, rtol=rtol, atol=atol)
    assert_close(out_k, out_k_torch, rtol=rtol, atol=atol)


@pytest.mark.parametrize("rotary_type", ["separated", "interleaved"])
@pytest.mark.parametrize("batch_size", [0, 16])
@pytest.mark.parametrize("impl", ["triton", "torch"])
@pytest.mark.parametrize("qk_dtype", [torch.float16, torch.bfloat16])
def test_apply_rotary_pos_emb_single(
    rotary_type,
    batch_size,
    impl,
    qk_dtype,
    record_benchmark,
):
    if impl == "triton":
        if not has_triton:
            pytest.skip("triton is missing")
        if rotary_type == "interleaved" and not hasattr(triton.language, "interleave"):
            pytest.skip("This op require Triton to support tl.interleave")

    n_local_heads, head_dim = 32, 128
    x = torch.randn(batch_size, n_local_heads, head_dim, device="cuda", dtype=qk_dtype)
    complex_freqs = torch.polar(
        torch.ones(batch_size, head_dim // 2, device="cuda", dtype=torch.float32),
        torch.rand(batch_size, head_dim // 2, device="cuda", dtype=torch.float32)
        * 2
        * math.pi,
    )
    freqs_cis = BatchedFreqsCis(
        complex_freqs.real.contiguous(),
        complex_freqs.imag.contiguous(),
    )

    out = record_benchmark.run(
        lambda: apply_rotary_pos_emb_single(
            x, freqs_cis, rotary_type=rotary_type, impl=impl
        ),
        batch_size=batch_size,
        impl=impl,
    )
    out_q_torch, _ = apply_rotary_pos_emb(
        x, x, freqs_cis, rotary_type=rotary_type, impl="torch"
    )

    assert_close(out, out_q_torch, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("rotary_type", ["separated", "interleaved"])
@pytest.mark.parametrize("impl", ["triton", "torch"])
def test_apply_rotary_pos_emb_single_partial_in_place(
    rotary_type,
    impl,
    record_benchmark,
):
    if impl == "triton":
        if not has_triton:
            pytest.skip("triton is missing")
        if rotary_type == "interleaved" and not hasattr(triton.language, "interleave"):
            pytest.skip("This op require Triton to support tl.interleave")

    batch_size, n_local_heads, head_dim = 16, 32, 128
    rotary_dim = head_dim // 2
    x = torch.randn(
        batch_size, n_local_heads, head_dim, device="cuda", dtype=torch.float16
    )
    complex_freqs = torch.polar(
        torch.ones(batch_size, rotary_dim // 2, device="cuda", dtype=torch.float32),
        torch.rand(batch_size, rotary_dim // 2, device="cuda", dtype=torch.float32)
        * 2
        * math.pi,
    )
    freqs_cis = BatchedFreqsCis(
        complex_freqs.real.contiguous(),
        complex_freqs.imag.contiguous(),
    )

    x_in_place = x.clone()

    def do_rotary_inplace():
        out, _, _, _ = apply_rotary_pos_emb_single_partial(
            x_in_place,
            freqs_cis,
            rotary_end=rotary_dim,
            rotary_type=rotary_type,
            inplace=True,
            impl=impl,
        )
        return out

    out = record_benchmark.run(
        do_rotary_inplace,
        batch_size=batch_size,
        impl=impl,
    )

    ref = x.clone()
    ref[..., :rotary_dim] = apply_rotary_pos_emb_single(
        x[..., :rotary_dim], freqs_cis, rotary_type=rotary_type, impl="torch"
    )
    assert out is x_in_place
    assert_close(out, ref, rtol=5e-3, atol=5e-3)
