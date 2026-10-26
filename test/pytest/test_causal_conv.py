import pytest
import torch

from chitu.ops import (
    causal_conv1d_update,
    causal_conv1d_update_qwen3_next,
    causal_conv1d_prefill,
)
from chitu.utils import try_import_platform_dep
from chitu.testing import AssertOpCalled, assert_close
from chitu.device_type import is_enflame
from chitu.ops.causal_conv import has_boltops_causal_conv

triton, has_triton = try_import_platform_dep("triton")


@pytest.mark.parametrize("batch_size", [0, 1, 4])
@pytest.mark.parametrize("hidden_size", [2048, 4096])
@pytest.mark.parametrize("state_len", [4])
@pytest.mark.parametrize("impl", ["torch", "triton", "boltops"])
def test_causal_conv1d_update(batch_size, hidden_size, state_len, impl):
    if impl == "boltops" and not has_boltops_causal_conv:
        pytest.skip("BoltOps conv is unavailable")
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


@pytest.mark.skipif(not has_boltops_causal_conv, reason="requires BoltOps on Hygon")
def test_boltops_causal_conv1d_update_bf16_cancellation():
    # Rounding 3.1875**2 to BF16 before subtracting 10 loses significant bits.
    old_state = (
        torch.tensor([0, 3.1875, -10, 1], device="cuda", dtype=torch.bfloat16)
        .view(1, 1, 4)
        .expand(4, 128, 4)
        .contiguous()
    )
    inputs = torch.zeros(4, 128, device="cuda", dtype=torch.bfloat16)
    weight = (
        torch.tensor([3.1875, 1, 1, 0], device="cuda", dtype=torch.bfloat16)
        .view(1, 1, 4)
        .expand(128, 1, 4)
        .contiguous()
    )
    output, new_state = causal_conv1d_update(inputs, old_state, weight, impl="boltops")
    output_ref, new_state_ref = causal_conv1d_update(
        inputs, old_state, weight, impl="ref"
    )
    assert_close(output, output_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(new_state, new_state_ref, atol=0, rtol=0)


@pytest.mark.parametrize("impl", ["torch", "triton", "boltops"])
def test_causal_conv1d_update_fp32_weight(impl):
    if impl == "boltops" and not has_boltops_causal_conv:
        pytest.skip("BoltOps conv is unavailable")
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    torch.set_default_dtype(torch.bfloat16)
    batch_size, hidden_size, state_len = 4, 2048, 4
    old_hidden_state = torch.randn(
        batch_size, hidden_size, state_len, dtype=torch.bfloat16, device="cuda"
    )
    this_hidden_state = torch.randn(
        batch_size, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    weight = torch.randn(hidden_size, 1, state_len, dtype=torch.float32, device="cuda")
    output, new_hidden_state = causal_conv1d_update(
        this_hidden_state, old_hidden_state, weight, impl=impl
    )
    output_ref, new_hidden_state_ref = causal_conv1d_update(
        this_hidden_state, old_hidden_state, weight, impl="ref"
    )
    # Outputs keep the activation dtype; the new state keeps the cache dtype
    # (NOT the fp32 weight dtype).
    assert output.dtype == torch.bfloat16
    assert new_hidden_state.dtype == torch.bfloat16
    assert_close(output, output_ref, atol=1e-2, rtol=1e-2)
    assert_close(new_hidden_state, new_hidden_state_ref, atol=1e-2, rtol=1e-2)
    # The fp32-weight result must be close to the bf16-weight result: the only
    # difference is the rounding of the weight values (and, for the torch impl,
    # the einsum accumulating in the weight dtype), which on N(0,1) 4-tap dots
    # can exceed the usual 1e-2 on rare elements — hence the looser bound. The
    # new state does not depend on the weight at all.
    output_bf16, new_hidden_state_bf16 = causal_conv1d_update(
        this_hidden_state, old_hidden_state, weight.to(torch.bfloat16), impl=impl
    )
    assert_close(output, output_bf16, atol=5e-2, rtol=5e-2)
    assert_close(new_hidden_state, new_hidden_state_bf16, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not is_enflame(), reason="Enflame GCU is required")
def test_causal_conv1d_update_qwen3_next_b1_is_exact_and_inplace():
    torch.manual_seed(1234)
    this_hidden_state = torch.randn(1, 10240, dtype=torch.bfloat16, device="cuda")
    old_hidden_state = torch.randn(1, 10240, 4, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(10240, 1, 4, dtype=torch.bfloat16, device="cuda")
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


@pytest.mark.parametrize(
    "prefix_lens",
    [
        torch.tensor([0], device="cuda"),
        torch.tensor([0, 2, 5], device="cuda"),
        torch.tensor([0, 4, 1024, 2048, 4096], device="cuda"),
    ],
)
@pytest.mark.parametrize("impl", ["triton", "boltops"])
@pytest.mark.parametrize("hidden_size", [2048, 4096])
@pytest.mark.parametrize(
    "state_len",
    [
        4,
    ],
)
def test_causal_conv1d_prefill(prefix_lens, hidden_size, state_len, impl):
    if impl == "boltops" and not has_boltops_causal_conv:
        pytest.skip("BoltOps conv is unavailable")
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


@pytest.mark.parametrize("impl", ["triton", "boltops"])
def test_causal_conv1d_prefill_fp32_weight(impl):
    if impl == "boltops" and not has_boltops_causal_conv:
        pytest.skip("BoltOps conv is unavailable")
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    torch.set_default_dtype(torch.bfloat16)
    prefix_lens = torch.tensor([0, 2, 5], device="cuda")
    hidden_size, state_len = 2048, 4
    batch_size = prefix_lens.shape[0] - 1
    total_len = prefix_lens[-1].item()

    inputs = torch.randn([total_len, hidden_size], device="cuda")
    conv_state = torch.randn([batch_size, hidden_size, state_len], device="cuda")
    weight = torch.randn(
        [hidden_size, 1, state_len], device="cuda", dtype=torch.float32
    )
    output, new_hidden_state = causal_conv1d_prefill(
        inputs, conv_state, weight, prefix_lens, impl=impl
    )
    output_ref, new_hidden_state_ref = causal_conv1d_prefill(
        inputs, conv_state, weight, prefix_lens, impl="ref"
    )
    assert output.dtype == torch.bfloat16
    assert new_hidden_state.dtype == torch.bfloat16
    assert_close(output, output_ref, atol=1e-2, rtol=1e-2)
    assert_close(new_hidden_state, new_hidden_state_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "prefix_lens",
    [
        torch.tensor([0, 4], device="cuda"),
        torch.tensor([0, 8, 12], device="cuda"),
    ],
)
@pytest.mark.parametrize("checkpoint_every_n_tokens", [4])
@pytest.mark.parametrize("hidden_size", [2048, 4096])
@pytest.mark.parametrize("state_len", [4])
@pytest.mark.parametrize("impl", ["ref", "boltops"])
def test_causal_conv1d_prefill_checkpoints(
    prefix_lens, checkpoint_every_n_tokens, hidden_size, state_len, impl
):
    if impl == "boltops" and not has_boltops_causal_conv:
        pytest.skip("BoltOps conv is unavailable")
    """checkpoint_every_n_tokens > 0 时 state_checkpoints 是调用方预分配的输出 buffer。"""
    torch.set_default_dtype(torch.bfloat16)
    C = checkpoint_every_n_tokens
    batch_size = prefix_lens.shape[0] - 1
    total_len = prefix_lens[-1].item()
    actual_lens = (prefix_lens[1:] - prefix_lens[:-1]).tolist()
    # 这里只测 chunk 起点对齐到 C 的情况（cache 侧也是这么校验的，见
    # SingletonPagedKVCache.ckpt_cu_starts），此时「从 chunk 起点数」和 cache「按 seq 绝对
    # 位置判定」是同一批位置
    assert all(lens % C == 0 for lens in actual_lens)

    inputs = torch.randn([total_len, hidden_size], device="cuda")
    conv_state = torch.randn([batch_size, hidden_size, state_len], device="cuda")
    weight = torch.randn([hidden_size, 1, state_len], device="cuda")

    cu_starts = [0]
    for lens in actual_lens:
        cu_starts.append(cu_starts[-1] + lens // C)
    state_checkpoints = torch.empty(
        [cu_starts[-1], hidden_size, state_len], device="cuda"
    )
    output, new_hidden_state = causal_conv1d_prefill(
        inputs,
        conv_state,
        weight,
        prefix_lens,
        impl=impl,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=torch.tensor(cu_starts, device="cuda", dtype=torch.int64),
        checkpoint_every_n_tokens=C,
    )

    # 存 checkpoint 不应该影响算子本身的输出
    output_ref, new_hidden_state_ref = causal_conv1d_prefill(
        inputs, conv_state, weight, prefix_lens, impl="ref"
    )
    assert_close(output, output_ref, atol=1e-2, rtol=1e-2)
    assert_close(new_hidden_state, new_hidden_state_ref, atol=1e-2, rtol=1e-2)

    # 第 t 个 token 之后的 state 等价于用同一个初始 state 只跑前 t 个 token 得到的末状态
    for i, lens in enumerate(actual_lens):
        seq_start = prefix_lens[i].item()
        for j, t in enumerate(range(C, lens + 1, C)):
            _, state_after_t = causal_conv1d_prefill(
                inputs[seq_start : seq_start + t],
                conv_state[i : i + 1],
                weight,
                torch.tensor([0, t], device="cuda"),
                impl="ref",
            )
            assert_close(
                state_checkpoints[cu_starts[i] + j],
                state_after_t[0],
                atol=1e-2,
                rtol=1e-2,
            )


@pytest.mark.skipif(not has_boltops_causal_conv, reason="requires BoltOps on Hygon")
@pytest.mark.parametrize(
    "lengths,dtype,weight_dtype",
    [
        ([0, 1, 2, 3, 7], torch.bfloat16, torch.bfloat16),
        ([3, 9, 17], torch.float32, torch.float32),
        ([31, 65], torch.float16, torch.float16),
        ([1, 513], torch.bfloat16, torch.bfloat16),
        ([0, 0], torch.bfloat16, torch.bfloat16),
        ([31, 65], torch.bfloat16, torch.float32),
    ],
)
def test_boltops_conv_prefill_full_width_state(lengths, dtype, weight_dtype):
    """Merged-QKV strides, short/empty requests and full-width checkpoints."""
    import torch.nn.functional as F

    torch.manual_seed(20261021)
    D, C, width = 67, 2, 4
    prefix = [0]
    starts = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)
        starts.append(starts[-1] + length // C)
    packed = torch.randn(prefix[-1], D + 13, device="cuda", dtype=dtype)
    inputs = packed[:, :D]
    old = torch.randn(len(lengths), width, D, device="cuda", dtype=dtype).transpose(
        1, 2
    )
    weight = torch.randn(D, 1, width, device="cuda", dtype=weight_dtype)
    packed_before, old_before = packed.clone(), old.clone()
    guard = torch.full((starts[-1] + 2, D, width), 123, device="cuda", dtype=dtype)
    snapshots = guard[1:-1]
    with AssertOpCalled("causal_conv1d_prefill", "boltops"):
        out, final = causal_conv1d_prefill(
            inputs,
            old,
            weight,
            torch.tensor(prefix, device="cuda"),
            impl="boltops",
            state_checkpoints=snapshots,
            checkpoint_cu_starts=torch.tensor(starts, device="cuda", dtype=torch.int64),
            checkpoint_every_n_tokens=C,
            seq_len_list=lengths,
        )
    outputs, finals, checkpoints = [], [], []
    for i, length in enumerate(lengths):
        full = torch.cat((old_before[i], inputs[prefix[i] : prefix[i + 1]].T), dim=-1)
        y = F.silu(F.conv1d(full[None].float(), weight.float(), groups=D)[:, :, 1:])
        outputs.append(y[0].T.to(dtype))
        finals.append(full[:, -width:])
        checkpoints.extend(full[:, t : t + width] for t in range(C, length + 1, C))
    tol = 2e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(out, torch.cat(outputs), atol=tol, rtol=tol)
    torch.testing.assert_close(final, torch.stack(finals), atol=0, rtol=0)
    if checkpoints:
        torch.testing.assert_close(snapshots, torch.stack(checkpoints), atol=0, rtol=0)
    torch.testing.assert_close(packed, packed_before, atol=0, rtol=0)
    torch.testing.assert_close(old, old_before, atol=0, rtol=0)
    assert torch.all(guard[[0, -1]] == 123)


@pytest.mark.skipif(not has_boltops_causal_conv, reason="requires BoltOps on Hygon")
def test_boltops_conv_prefill_graph_and_single_channel():
    # Single-channel history slices can already be contiguous: they must still
    # be copied before BoltOps' in-place cache update.
    for D in (1, 128):
        lengths, C = [2, 9], 2
        x = torch.randn(11, D, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(2, D, 4, device="cuda", dtype=x.dtype)
        weight = torch.randn(D, 1, 4, device="cuda", dtype=torch.float32)
        prefix = torch.tensor([0, 2, 11], device="cuda")
        starts = torch.tensor([0, 1, 5], device="cuda", dtype=torch.int64)
        checkpoints = torch.empty(5, D, 4, device="cuda", dtype=x.dtype)
        before = state.clone()
        kwargs = dict(
            impl="boltops",
            seq_len_list=lengths,
            state_checkpoints=checkpoints,
            checkpoint_cu_starts=starts,
            checkpoint_every_n_tokens=C,
        )
        causal_conv1d_prefill(x, state, weight, prefix, **kwargs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, final = causal_conv1d_prefill(x, state, weight, prefix, **kwargs)
        x.mul_(0.5)
        checkpoints.fill_(float("nan"))
        graph.replay()
        expected_checkpoints = torch.empty_like(checkpoints)
        expected_out, expected_state = causal_conv1d_prefill(
            x,
            state,
            weight,
            prefix,
            impl="ref",
            state_checkpoints=expected_checkpoints,
            checkpoint_cu_starts=starts,
            checkpoint_every_n_tokens=C,
        )
        assert_close(out, expected_out, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(final, expected_state, atol=0, rtol=0)
        torch.testing.assert_close(checkpoints, expected_checkpoints, atol=0, rtol=0)
        torch.testing.assert_close(state, before, atol=0, rtol=0)


@pytest.mark.skipif(not has_boltops_causal_conv, reason="requires BoltOps on Hygon")
def test_boltops_conv_update_graph_readonly_state():
    """Replay changed inputs/weights/states, including merged-QKV strides."""
    batch_size, hidden_size = 3, 67
    packed = torch.randn(
        batch_size, hidden_size + 13, device="cuda", dtype=torch.bfloat16
    )
    x = packed[:, :hidden_size]
    state = torch.randn(
        batch_size, 4, hidden_size, device="cuda", dtype=x.dtype
    ).transpose(1, 2)
    weight = torch.randn(hidden_size, 1, 4, device="cuda", dtype=torch.float32)
    causal_conv1d_update(x, state, weight, impl="boltops")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, final = causal_conv1d_update(x, state, weight, impl="boltops")
    for _ in range(2):
        packed.mul_(0.5)
        state.add_(0.125)
        weight.add_(0.01)
        packed_before, state_before = packed.clone(), state.clone()
        graph.replay()
        expected_out, expected_state = causal_conv1d_update(
            x, state, weight, impl="ref"
        )
        assert_close(out, expected_out, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(final, expected_state, atol=0, rtol=0)
        torch.testing.assert_close(state, state_before, atol=0, rtol=0)
        torch.testing.assert_close(packed, packed_before, atol=0, rtol=0)


def test_causal_conv1d_prefill_checkpoints_auto_impl(monkeypatch):
    """不显式指定 impl 时，带 checkpoint 的调用必须自动落到支持 checkpoint 的实现。

    这里不传 impl（= "auto"），由 register_auto 决定。它以前只看有没有 triton：装了 triton
    的机器上带 checkpoint 的调用会被派到 triton 实现，而 triton 实现不支持 checkpoint，
    直接 assert 失败。
    """
    torch.set_default_dtype(torch.bfloat16)
    C, hidden_size, state_len = 4, 64, 4
    prefix_lens = torch.tensor([0, 8], device="cuda")
    inputs = torch.randn([8, hidden_size], device="cuda")
    conv_state = torch.randn([1, hidden_size, state_len], device="cuda")
    weight = torch.randn([hidden_size, 1, state_len], device="cuda")
    state_checkpoints = torch.empty([2, hidden_size, state_len], device="cuda")
    checkpoint_cu_starts = torch.tensor([0, 2], device="cuda", dtype=torch.int64)

    with AssertOpCalled("causal_conv1d_prefill", "ref"):
        causal_conv1d_prefill(
            inputs,
            conv_state,
            weight,
            prefix_lens,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=C,
        )

    # GLM supplies CPU lengths; auto then selects BoltOps on supported Hygon.
    kwargs = dict(
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=C,
        seq_len_list=[8],
    )
    selected = "boltops" if has_boltops_causal_conv else "ref"
    with AssertOpCalled("causal_conv1d_prefill", selected):
        causal_conv1d_prefill(inputs, conv_state, weight, prefix_lens, **kwargs)

    # Missing BoltOps and unsupported widths keep the existing fallback.
    assert (
        causal_conv1d_prefill.resolve_impl(
            inputs,
            conv_state,
            torch.empty(hidden_size, 1, 5, device="cuda"),
            prefix_lens,
            **kwargs,
        )
        == "ref"
    )
    monkeypatch.setattr("chitu.ops.causal_conv.has_boltops_causal_conv", False)
    with AssertOpCalled("causal_conv1d_prefill", "ref"):
        causal_conv1d_prefill(inputs, conv_state, weight, prefix_lens, **kwargs)


def test_causal_conv1d_prefill_checkpoints_invalid_args():
    torch.set_default_dtype(torch.bfloat16)
    C = 4
    hidden_size, state_len = 64, 4
    prefix_lens = torch.tensor([0, 8], device="cuda")
    inputs = torch.randn([8, hidden_size], device="cuda")
    conv_state = torch.randn([1, hidden_size, state_len], device="cuda")
    weight = torch.randn([hidden_size, 1, state_len], device="cuda")

    # checkpoint_every_n_tokens > 0 时必须同时给 state_checkpoints 和 checkpoint_cu_starts
    with pytest.raises(AssertionError):
        causal_conv1d_prefill(
            inputs,
            conv_state,
            weight,
            prefix_lens,
            impl="ref",
            checkpoint_every_n_tokens=C,
        )
    # 不存 checkpoint 时不能给 buffer
    with pytest.raises(AssertionError):
        causal_conv1d_prefill(
            inputs,
            conv_state,
            weight,
            prefix_lens,
            impl="ref",
            state_checkpoints=torch.empty([2, hidden_size, state_len], device="cuda"),
            checkpoint_cu_starts=torch.tensor([0, 2], device="cuda", dtype=torch.int64),
        )
    # 调用方给的个数和算子按 checkpoint_every_n_tokens 数出来的对不上（会写到错位的页上）
    with pytest.raises(AssertionError):
        causal_conv1d_prefill(
            inputs,
            conv_state,
            weight,
            prefix_lens,
            impl="ref",
            state_checkpoints=torch.empty([3, hidden_size, state_len], device="cuda"),
            checkpoint_cu_starts=torch.tensor([0, 3], device="cuda", dtype=torch.int64),
            checkpoint_every_n_tokens=C,
        )
