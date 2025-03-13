import pytest
import torch
import triton
import triton.language as tl

from chitu.ops import (
    apply_rotary_pos_emb_triton,
    apply_rotary_pos_emb_torch,
)


@pytest.mark.skipif(
    not hasattr(triton.language, "interleave"),
    reason="This op require Triton to support tl.interleave",
)
def test_apply_rotary_pos_emb_triton_interleave_deepseek(
    batch_size=16, n_local_heads=64, head_dim=256
):
    q = torch.randn(batch_size, n_local_heads, head_dim, device="cuda")
    k = torch.randn(batch_size, head_dim, device="cuda")
    cos = torch.randn(batch_size, head_dim // 2, device="cuda") * 2
    sin = torch.randn(batch_size, head_dim // 2, device="cuda")
    out_q, out_k = apply_rotary_pos_emb_triton(q, k, cos, sin, rotary_type="llama")
    out_q_torch, out_k_torch = apply_rotary_pos_emb_torch(
        q, k, cos, sin, rotary_type="llama"
    )
    # Check if out_q and out_q_torch are the same
    # Use rtol and atol for more precise comparison
    rtol = 1e-5
    atol = 1e-5
    assert torch.all(torch.isclose(out_q, out_q_torch, rtol=rtol, atol=atol))
    assert torch.all(torch.isclose(out_k, out_k_torch, rtol=rtol, atol=atol))


if __name__ == "__main__":
    test_apply_rotary_pos_emb_triton_interleave_deepseek()
