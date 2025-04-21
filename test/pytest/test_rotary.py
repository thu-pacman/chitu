import pytest
import math
import torch
import triton
import triton.language as tl

from chitu.ops import apply_rotary_pos_emb


@pytest.mark.parametrize("impl", ["cuda", "triton"])
def test_apply_rotary_pos_emb_interleave_deepseek(
    impl, batch_size=16, n_local_heads=64, head_dim=256
):
    if impl == "triton" and not hasattr(triton.language, "interleave"):
        pytest.skip("This op require Triton to support tl.interleave")

    torch.set_default_dtype(torch.float16)
    q = torch.randn(batch_size, n_local_heads, head_dim, device="cuda")
    k = torch.randn(batch_size, head_dim, device="cuda")

    # Generate cos and sin from a unit circle
    complex_freqs = torch.polar(
        torch.ones(batch_size, head_dim // 2, device="cuda", dtype=torch.float32),
        torch.rand(batch_size, head_dim // 2, device="cuda", dtype=torch.float32)
        * 2
        * math.pi,
    )
    cos = complex_freqs.real.contiguous()
    sin = complex_freqs.imag.contiguous()

    out_q, out_k = apply_rotary_pos_emb(q, k, cos, sin, rotary_type="llama", impl=impl)
    out_q_torch, out_k_torch = apply_rotary_pos_emb(
        q, k, cos, sin, rotary_type="llama", impl="torch"
    )
    # Check if out_q and out_q_torch are the same
    # Use rtol and atol for more precise comparison
    rtol = 5e-3
    atol = 5e-3
    assert torch.all(torch.isclose(out_q, out_q_torch, rtol=rtol, atol=atol))
    assert torch.all(torch.isclose(out_k, out_k_torch, rtol=rtol, atol=atol))


@pytest.mark.parametrize(
    "impl", ["cuda", "torch"]
)  # Also test "torch"'s in-place with itself's out-of-place
def test_apply_rotary_pos_emb_interleave_deepseek_in_place(
    impl, batch_size=16, n_local_heads=64, head_dim=256
):
    torch.set_default_dtype(torch.float16)
    q = torch.randn(batch_size, n_local_heads, head_dim, device="cuda")
    k = torch.randn(batch_size, head_dim, device="cuda")

    # Generate cos and sin from a unit circle
    complex_freqs = torch.polar(
        torch.ones(batch_size, head_dim // 2, device="cuda", dtype=torch.float32),
        torch.rand(batch_size, head_dim // 2, device="cuda", dtype=torch.float32)
        * 2
        * math.pi,
    )
    cos = complex_freqs.real.contiguous()
    sin = complex_freqs.imag.contiguous()

    out_q = q.clone()
    out_k = k.clone()
    apply_rotary_pos_emb(
        out_q, out_k, cos, sin, rotary_type="llama", q_out=out_q, k_out=out_k, impl=impl
    )
    out_q_torch, out_k_torch = apply_rotary_pos_emb(
        q, k, cos, sin, rotary_type="llama", impl="torch"
    )
    # Check if out_q and out_q_torch are the same
    # Use rtol and atol for more precise comparison
    rtol = 5e-3
    atol = 5e-3
    assert torch.all(torch.isclose(out_q, out_q_torch, rtol=rtol, atol=atol))
    assert torch.all(torch.isclose(out_k, out_k_torch, rtol=rtol, atol=atol))


if __name__ == "__main__":
    test_apply_rotary_pos_emb_triton_interleave_deepseek()
