import importlib

import pytest
import torch

from chitu.ops import (
    causal_conv1d_update,
    causal_conv1d_update_qwen3_next,
    causal_conv1d_prefill,
)
from chitu.utils import try_import_platform_dep
from chitu.testing import assert_close
from chitu.device_type import is_enflame

triton, has_triton = try_import_platform_dep("triton")
if has_triton:
    from chitu.ops.triton_ops.causal_conv import (
        causal_conv1d_update_gcu_batch_inplace,
    )

causal_conv_module = importlib.import_module("chitu.ops.causal_conv")


@pytest.mark.parametrize("batch_size", [0, 1, 4])
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
    assert_close(output, output_ref, atol=1e-2, rtol=1e-2)
    assert_close(new_hidden_state, new_hidden_state_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not is_enflame(), reason="Enflame GCU is required")
def test_causal_conv1d_update_qwen3_next_b1_is_exact_and_inplace():
    torch.manual_seed(1234)
    this_hidden_state = torch.randn(
        1, 10240, dtype=torch.bfloat16, device="cuda"
    )
    old_hidden_state = torch.randn(
        1, 10240, 4, dtype=torch.bfloat16, device="cuda"
    )
    weight = torch.randn(
        10240, 1, 4, dtype=torch.bfloat16, device="cuda"
    )
    state_ref_input = old_hidden_state.clone()
    # The production control for this shape is Chitu's generic Triton kernel.
    # Compare byte-for-byte at that exact numerical boundary (the framework
    # reference uses a different reduction/materialization sequence).
    output_ref, state_ref = causal_conv1d_update(
        this_hidden_state, state_ref_input, weight, impl="triton"
    )
    state_ptr = old_hidden_state.data_ptr()
    output, state = causal_conv1d_update_qwen3_next(
        this_hidden_state, old_hidden_state, weight
    )

    assert state.data_ptr() == state_ptr
    assert torch.equal(output, output_ref)
    assert torch.equal(state, state_ref)


@pytest.mark.skipif(not is_enflame(), reason="Enflame GCU is required")
@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize("strided_input", [False, True])
def test_causal_conv1d_update_qwen3_next_batch_is_exact_and_inplace(
    batch_size, strided_input
):
    torch.manual_seed(1234 + batch_size)
    input_width = 16384 if strided_input else 10240
    this_hidden_state_storage = torch.randn(
        batch_size, input_width, dtype=torch.bfloat16, device="cuda"
    )
    this_hidden_state = this_hidden_state_storage[:, :10240]
    old_hidden_state = torch.randn(
        batch_size, 10240, 4, dtype=torch.bfloat16, device="cuda"
    )
    weight = torch.randn(
        10240, 1, 4, dtype=torch.bfloat16, device="cuda"
    )
    state_ref_input = old_hidden_state.clone()
    output_ref, state_ref = causal_conv1d_update(
        this_hidden_state, state_ref_input, weight, impl="triton"
    )
    state_ptr = old_hidden_state.data_ptr()
    output, state = causal_conv1d_update_gcu_batch_inplace(
        this_hidden_state,
        old_hidden_state,
        weight,
    )

    assert state.data_ptr() == state_ptr
    assert torch.equal(output, output_ref)
    assert torch.equal(state, state_ref)


@pytest.mark.skipif(not is_enflame(), reason="Enflame GCU is required")
@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_qwen3_next_batch_dispatcher_accepts_strided_input(monkeypatch, batch_size):
    storage = torch.empty(
        batch_size, 16384, dtype=torch.bfloat16, device="cuda"
    )
    this_hidden_state = storage[:, :10240]
    old_hidden_state = torch.empty(
        batch_size, 10240, 4, dtype=torch.bfloat16, device="cuda"
    )
    weight = torch.empty(10240, 1, 4, dtype=torch.bfloat16, device="cuda")
    calls = []

    def fake_batch(this, state, conv_weight):
        calls.append(this.stride())
        return this, state

    def unexpected_generic(*args, **kwargs):
        raise AssertionError("supported batch shape unexpectedly fell back")

    monkeypatch.setattr(causal_conv_module, "_GCU_QWEN3_NEXT_CONV_BATCH", True)
    monkeypatch.setattr(
        causal_conv_module, "causal_conv1d_update_gcu_batch_inplace", fake_batch
    )
    monkeypatch.setattr(
        causal_conv_module, "causal_conv1d_update", unexpected_generic
    )

    output, state = causal_conv_module.causal_conv1d_update_qwen3_next(
        this_hidden_state, old_hidden_state, weight
    )

    assert output is this_hidden_state
    assert state is old_hidden_state
    assert calls == [(16384, 1)]


@pytest.mark.skipif(not is_enflame(), reason="Enflame GCU is required")
@pytest.mark.parametrize("batch_enabled", [False, True])
def test_qwen3_next_batch_dispatcher_fails_closed_for_cpu_or_disabled(
    monkeypatch, batch_enabled
):
    if batch_enabled:
        this_hidden_state = torch.empty(2, 10240, dtype=torch.bfloat16)
        old_hidden_state = torch.empty(2, 10240, 4, dtype=torch.bfloat16)
        weight = torch.empty(10240, 1, 4, dtype=torch.bfloat16)
    else:
        this_hidden_state = torch.empty(
            2, 10240, dtype=torch.bfloat16, device="cuda"
        )
        old_hidden_state = torch.empty(
            2, 10240, 4, dtype=torch.bfloat16, device="cuda"
        )
        weight = torch.empty(10240, 1, 4, dtype=torch.bfloat16, device="cuda")
    calls = []

    def unexpected_batch(*args, **kwargs):
        raise AssertionError("unsupported input unexpectedly entered fast path")

    def fake_generic(this, state, conv_weight):
        calls.append((this.device.type, state.device.type, conv_weight.device.type))
        return this, state

    monkeypatch.setattr(
        causal_conv_module, "_GCU_QWEN3_NEXT_CONV_BATCH", batch_enabled
    )
    monkeypatch.setattr(
        causal_conv_module,
        "causal_conv1d_update_gcu_batch_inplace",
        unexpected_batch,
    )
    monkeypatch.setattr(causal_conv_module, "causal_conv1d_update", fake_generic)

    output, state = causal_conv_module.causal_conv1d_update_qwen3_next(
        this_hidden_state, old_hidden_state, weight
    )

    assert output is this_hidden_state
    assert state is old_hidden_state
    assert len(calls) == 1


@pytest.mark.skipif(not is_enflame(), reason="Enflame GCU is required")
@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize("sequence_seed", [20260830, 20260831, 20260832])
def test_qwen3_next_batch_dispatcher_is_exact_across_state_sequence(
    monkeypatch, batch_size, sequence_seed
):
    torch.manual_seed(sequence_seed + batch_size)
    state = torch.randn(
        batch_size, 10240, 4, dtype=torch.bfloat16, device="cuda"
    )
    state_ref = state.clone()
    weight = torch.randn(10240, 1, 4, dtype=torch.bfloat16, device="cuda")
    state_ptr = state.data_ptr()
    monkeypatch.setattr(causal_conv_module, "_GCU_QWEN3_NEXT_CONV_BATCH", True)

    for step in range(100):
        storage = torch.randn(
            batch_size, 16384, dtype=torch.bfloat16, device="cuda"
        )
        this_hidden_state = storage[:, :10240]
        output_ref, state_ref = causal_conv1d_update(
            this_hidden_state, state_ref, weight, impl="triton"
        )
        output, state = causal_conv_module.causal_conv1d_update_qwen3_next(
            this_hidden_state, state, weight
        )

        assert state.data_ptr() == state_ptr
        output_bad = torch.count_nonzero(output != output_ref).item()
        state_bad = torch.count_nonzero(state != state_ref).item()
        output_max_abs = (output.float() - output_ref.float()).abs().max().item()
        state_max_abs = (state.float() - state_ref.float()).abs().max().item()
        assert output_bad == 0, (
            f"step={step} output_bad={output_bad} "
            f"output_max_abs={output_max_abs}"
        )
        assert state_bad == 0, (
            f"step={step} state_bad={state_bad} state_max_abs={state_max_abs}"
        )


@pytest.mark.parametrize(
    "prefix_lens",
    [
        torch.tensor([0], device="cuda"),
        torch.tensor([0, 2, 5], device="cuda"),
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
def test_causal_conv1d_prefill(prefix_lens, hidden_size, state_len, impl):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    torch.set_default_dtype(torch.bfloat16)
    batch_size = prefix_lens.shape[0] - 1
    total_len = prefix_lens[-1].item()

    inputs = torch.randn([total_len, hidden_size], device="cuda")
    conv_state = torch.randn([batch_size, hidden_size, state_len], device="cuda")
    weight = torch.randn([hidden_size, 1, state_len], device="cuda")
    output, new_hidden_state = causal_conv1d_prefill(
        inputs, conv_state, weight, prefix_lens, impl=impl
    )
    output_ref, new_hidden_state_ref = causal_conv1d_prefill(
        inputs, conv_state, weight, prefix_lens, impl="ref"
    )
    assert_close(output, output_ref, atol=1e-2, rtol=1e-2)
    assert_close(new_hidden_state, new_hidden_state_ref, atol=1e-2, rtol=1e-2)
