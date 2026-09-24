import pytest
import torch

from chitu.ops import causal_conv1d_update, causal_conv1d_prefill
from chitu.utils import try_import_platform_dep
from chitu.testing import AssertOpCalled, assert_close

triton, has_triton = try_import_platform_dep("triton")


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
def test_causal_conv1d_prefill_checkpoints(
    prefix_lens, checkpoint_every_n_tokens, hidden_size, state_len
):
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
        impl="ref",
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


def test_causal_conv1d_prefill_checkpoints_auto_impl():
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
