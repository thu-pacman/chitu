import pytest
import torch

from chitu.ops import causal_conv1d_update


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("hidden_size", [2048, 4096])
@pytest.mark.parametrize("state_len", [4])
@pytest.mark.parametrize("impl", ["torch"])
def test_causal_conv1d_update(batch_size, hidden_size, state_len, impl):
    torch.set_default_dtype(torch.bfloat16)
    old_hidden_state = torch.randn(
        batch_size, hidden_size, state_len, dtype=torch.bfloat16, device="cuda"
    )
    this_hidden_state = torch.randn(
        batch_size, hidden_size, 1, dtype=torch.bfloat16, device="cuda"
    )
    weight = torch.randn(hidden_size, 1, state_len, dtype=torch.bfloat16, device="cuda")
    output, new_hidden_state = causal_conv1d_update(
        this_hidden_state, old_hidden_state, weight, impl=impl
    )
    output_ref, new_hidden_state_ref = causal_conv1d_update(
        this_hidden_state, old_hidden_state, weight, impl="ref"
    )
    torch.testing.assert_close(output, output_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        new_hidden_state, new_hidden_state_ref, atol=1e-2, rtol=1e-2
    )
