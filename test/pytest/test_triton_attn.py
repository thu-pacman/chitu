import pytest
import torch
from chitu.triton_flash_attention import context_attention_fwd
from chitu.attn_backend import RefAttnBackend


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_attention_matches_reference():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Test parameters
    seq_lens = [9]
    max_seq_len = max(seq_lens)

    # Create test tensors
    q = torch.randn(sum(seq_lens), 16, 576, dtype=torch.bfloat16).to("cuda")
    k = torch.randn(sum(seq_lens), 1, 576, dtype=torch.bfloat16).to("cuda")
    v = torch.randn(sum(seq_lens), 1, 512, dtype=torch.bfloat16).to("cuda")

    # Create metadata tensors
    b_start_loc = torch.tensor([0, seq_lens[0]], device="cuda")
    b_seq_len = torch.tensor(seq_lens, device="cuda")

    # Set attention parameters
    max_seqlen_q = max_seq_len
    max_seqlen_k = max_seq_len
    softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
    is_causal = True

    # Run triton implementation
    o = torch.empty(sum(seq_lens), 16, 512, dtype=torch.float32).to("cuda")
    context_attention_fwd(
        q, k, v, o, b_start_loc, b_seq_len, max_seqlen_q, softmax_scale, is_causal
    )

    # Run reference implementation
    ref_attn = RefAttnBackend()
    new_o = ref_attn.attn_varlen_func(
        q,
        k,
        v,
        b_start_loc,
        b_start_loc,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0,
        causal=is_causal,
        window_size=(-1, -1),
        softcap=0,
        softmax_scale=softmax_scale,
    )

    # Check if tensors are close
    assert torch.allclose(
        o.to(torch.float32), new_o.to(torch.float32), atol=1e-2, rtol=1e-1
    )
