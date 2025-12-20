import pytest
import torch
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")

from chitu.ops import causal_conv1d_update, causal_conv1d_prefill


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("hidden_size", [2048, 4096])
@pytest.mark.parametrize("state_len", [4])
@pytest.mark.parametrize("impl", ["torch", "triton"])
def test_causal_conv1d_update(batch_size, hidden_size, state_len, impl):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    torch.set_default_dtype(torch.bfloat16)
    old_hidden_state = torch.randn(
        batch_size, hidden_size, state_len, dtype=torch.bfloat16, device="cuda"
    )
    this_hidden_state = torch.randn(
        batch_size, hidden_size, dtype=torch.bfloat16, device="cuda"
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


@pytest.mark.parametrize(
    "prefix_lens",
    [
        torch.tensor([0, 4, 1024, 2048, 4096], device="cuda"),
    ],
)
@pytest.mark.parametrize("impl", ["triton"])
@pytest.mark.parametrize("hidden_size", [2048, 4096])
@pytest.mark.parametrize(
    "state_len",
    [
        4,
    ],
)
@pytest.mark.parametrize(
    "padding",
    [
        3,
    ],
)
def test_causal_conv1d_prefill(prefix_lens, hidden_size, state_len, padding, impl):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    torch.set_default_dtype(torch.bfloat16)
    batch_size = prefix_lens.shape[0] - 1
    total_len = prefix_lens[-1].item()

    inputs = torch.randn([total_len, hidden_size], device="cuda")
    weight = torch.randn([hidden_size, 1, state_len], device="cuda")
    output, new_hidden_state = causal_conv1d_prefill(
        inputs, weight, prefix_lens, padding=padding, impl=impl
    )
    output_ref, new_hidden_state_ref = causal_conv1d_prefill(
        inputs, weight, prefix_lens, padding=padding, impl="ref"
    )
    torch.testing.assert_close(output, output_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        new_hidden_state, new_hidden_state_ref, atol=1e-2, rtol=1e-2
    )
