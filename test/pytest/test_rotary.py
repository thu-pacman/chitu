import pytest
import math
import torch

from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.ops import apply_rotary_pos_emb
from chitu.native_layout import NativeLayoutTensor, ColumnOddEvenSeparatedTensor
from chitu.utils import try_import_platform_dep, try_import_and_setup_torch_npu

triton, has_triton = try_import_platform_dep("triton")
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


@pytest.mark.parametrize(
    "rotary_type,batch_size,n_local_heads,head_dim",
    [
        ("separated", 16, 64, 256),
        ("interleaved", 64, 128, 64),
        ("separated-half", 16, 32, 128),
        ("interleaved-half", 16, 32, 128),
    ],
)
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
):
    if impl == "triton":
        if not has_triton:
            pytest.skip("triton is missing")
        if rotary_type in ["interleaved", "interleaved-half"] and not hasattr(
            triton.language, "interleave"
        ):
            pytest.skip("This op require Triton to support tl.interleave")
    if impl == "cuda":
        if not has_chitu_backend:
            pytest.skip("chitu_backend is not available, skipping CUDA tests")
        if rotary_type in ["separated", "separated-half", "interleaved-half"]:
            pytest.skip("This op is not implemented in CUDA yet")
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

    out_q, out_k = apply_rotary_pos_emb(
        q, k, freqs_cis, rotary_type=rotary_type, impl=impl
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
    assert torch.all(torch.isclose(out_q, out_q_torch, rtol=rtol, atol=atol))
    assert torch.all(torch.isclose(out_k, out_k_torch, rtol=rtol, atol=atol))


@pytest.mark.parametrize(
    "rotary_type,batch_size,n_local_heads,head_dim",
    [
        ("separated", 16, 64, 256),
        ("interleaved", 64, 128, 64),
        ("separated-half", 16, 32, 128),
        ("interleaved-half", 16, 32, 128),
    ],
)
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
):
    if impl == "triton":
        if not has_triton:
            pytest.skip("triton is missing")
        if rotary_type in ["interleaved", "interleaved-half"] and not hasattr(
            triton.language, "interleave"
        ):
            pytest.skip("This op require Triton to support tl.interleave")
    if impl == "cuda":
        if not has_chitu_backend:
            pytest.skip("chitu_backend is not available, skipping CUDA tests")
        if rotary_type in ["separated", "separated-half", "interleaved-half"]:
            pytest.skip("This op is not implemented in CUDA yet")
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

    apply_rotary_pos_emb(
        q_clone,
        k_clone,
        freqs_cis,
        rotary_type=rotary_type,
        q_out=out_q,
        k_out=out_k,
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
    assert torch.all(torch.isclose(out_q, out_q_torch, rtol=rtol, atol=atol))
    assert torch.all(torch.isclose(out_k, out_k_torch, rtol=rtol, atol=atol))
