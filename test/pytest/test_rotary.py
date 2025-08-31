import pytest
import math
import torch

from chitu.ops import apply_rotary_pos_emb
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")


@pytest.mark.parametrize(
    "rotary_type,batch_size,n_local_heads,head_dim",
    [
        ("separated", 16, 64, 256),
        ("interleaved", 16, 64, 256),
        ("separated-half", 16, 32, 128),
        ("interleaved-half", 16, 32, 128),
    ],
)
@pytest.mark.parametrize("is_mqa", [False, True])
@pytest.mark.parametrize("impl", ["cuda", "triton"])
def test_apply_rotary_pos_emb(
    rotary_type, batch_size, n_local_heads, head_dim, is_mqa, impl
):
    if (
        impl == "triton"
        and rotary_type in ["interleaved", "interleaved-half"]
        and not hasattr(triton.language, "interleaved")
    ):
        pytest.skip("This op require Triton to support tl.interleave")
    if impl == "cuda" and rotary_type in [
        "separated",
        "separated-half",
        "interleaved-half",
    ]:
        pytest.skip("This op is not implemented in CUDA yet")

    torch.set_default_dtype(torch.float16)
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
    cos = complex_freqs.real.contiguous()
    sin = complex_freqs.imag.contiguous()

    out_q, out_k = apply_rotary_pos_emb(
        q, k, cos, sin, rotary_type=rotary_type, impl=impl
    )
    out_q_torch, out_k_torch = apply_rotary_pos_emb(
        q, k, cos, sin, rotary_type=rotary_type, impl="torch"
    )
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
        ("interleaved", 16, 64, 256),
        ("separated-half", 16, 32, 128),
        ("interleaved-half", 16, 32, 128),
    ],
)
@pytest.mark.parametrize("is_mqa", [False, True])
@pytest.mark.parametrize(
    "impl", ["cuda", "torch"]
)  # Also test "torch"'s in-place with itself's out-of-place
def test_apply_rotary_pos_emb_in_place(
    rotary_type, batch_size, n_local_heads, head_dim, is_mqa, impl
):
    if impl == "cuda" and rotary_type in [
        "separated",
        "separated-half",
        "interleaved-half",
    ]:
        pytest.skip("This op is not implemented in CUDA yet")

    torch.set_default_dtype(torch.float16)
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
    cos = complex_freqs.real.contiguous()
    sin = complex_freqs.imag.contiguous()

    out_q = q.clone()
    out_k = k.clone()
    apply_rotary_pos_emb(
        out_q,
        out_k,
        cos,
        sin,
        rotary_type=rotary_type,
        q_out=out_q,
        k_out=out_k,
        impl=impl,
    )
    out_q_torch, out_k_torch = apply_rotary_pos_emb(
        q, k, cos, sin, rotary_type=rotary_type, impl="torch"
    )
    # Check if out_q and out_q_torch are the same
    # Use rtol and atol for more precise comparison
    rtol = 5e-3
    atol = 5e-3
    assert torch.all(torch.isclose(out_q, out_q_torch, rtol=rtol, atol=atol))
    assert torch.all(torch.isclose(out_k, out_k_torch, rtol=rtol, atol=atol))
