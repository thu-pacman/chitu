import pytest
import torch
import torch.nn.functional as F

from chitu.device_type import is_muxi
from chitu.ops.linear_attn import (
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_torch_dense,
    chunk_kimi_delta_attention,
    chunk_kimi_delta_attention_torch_dense,
    recurrent_gated_delta_rule_torch,
)
from chitu.utils import try_import_opt_dep
from chitu.testing import AssertOpCalled, assert_close

if is_muxi():
    has_fla = False
else:
    fla, has_fla = try_import_opt_dep("fla", "fla")

if has_fla:
    from fla.ops import chunk_gated_delta_rule as chunk_gated_delta_rule_fla
    from fla.ops import (
        fused_recurrent_gated_delta_rule as fused_recurrent_gated_delta_rule_fla,
    )


@pytest.mark.parametrize("bs", [0, 1, 8])
@pytest.mark.parametrize("seq_len", [64, 1024, 4096])
@pytest.mark.parametrize("linear_head_dim", [128])
@pytest.mark.parametrize("linear_n_v_heads", [32])
def test_chunk_gated_delta_rule(
    bs,
    seq_len,
    linear_head_dim,
    linear_n_v_heads,
    record_benchmark,
):
    if not has_fla:
        pytest.skip("fla is missing")

    torch.set_default_dtype(torch.float32)

    q = torch.randn(bs, seq_len, linear_n_v_heads, linear_head_dim, device="cuda")
    k = torch.randn(bs, seq_len, linear_n_v_heads, linear_head_dim, device="cuda")
    v = torch.randn(bs, seq_len, linear_n_v_heads, linear_head_dim, device="cuda")
    g = F.logsigmoid(torch.randn(bs, seq_len, linear_n_v_heads, device="cuda"))
    g = g * (torch.rand_like(g))
    beta = torch.randn(bs, seq_len, linear_n_v_heads, device="cuda").sigmoid()

    fla_out, _ = record_benchmark.run(
        lambda: chunk_gated_delta_rule_fla(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        ),
        seq_len=seq_len,
        impl="fla",
    )
    torch_out, _ = chunk_gated_delta_rule_torch_dense(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    assert_close(torch_out, fla_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("bs", [0, 1, 8])
@pytest.mark.parametrize("linear_head_dim", [128])
@pytest.mark.parametrize("linear_n_v_heads", [32])
def test_recurrent_gated_delta_rule(
    bs,
    linear_head_dim,
    linear_n_v_heads,
    record_benchmark,
):
    if not has_fla:
        pytest.skip("fla is missing")

    torch.set_default_dtype(torch.float32)

    q = torch.randn(bs, 1, linear_n_v_heads, linear_head_dim, device="cuda")
    k = torch.randn(bs, 1, linear_n_v_heads, linear_head_dim, device="cuda")
    v = torch.randn(bs, 1, linear_n_v_heads, linear_head_dim, device="cuda")
    g = F.logsigmoid(torch.randn(bs, 1, linear_n_v_heads, device="cuda"))
    g = g * (torch.rand_like(g))
    beta = torch.randn(bs, 1, linear_n_v_heads, device="cuda").sigmoid()
    initial_state = torch.randn(
        bs, linear_n_v_heads, linear_head_dim, linear_head_dim, device="cuda"
    )

    fla_out, _ = record_benchmark.run(
        lambda: fused_recurrent_gated_delta_rule_fla(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        ),
        bs=bs,
        impl="fla",
    )
    torch_out, _ = recurrent_gated_delta_rule_torch(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    assert_close(torch_out, fla_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("seq_len_list", [[4], [10], [8, 5]])
@pytest.mark.parametrize("checkpoint_every_n_tokens", [4])
@pytest.mark.parametrize("linear_head_dim", [128])
@pytest.mark.parametrize("linear_n_v_heads", [32])
def test_chunk_gated_delta_rule_checkpoints(
    seq_len_list,
    checkpoint_every_n_tokens,
    linear_head_dim,
    linear_n_v_heads,
):
    """checkpoint_every_n_tokens > 0 时 state_checkpoints 是调用方预分配的输出 buffer。"""
    torch.set_default_dtype(torch.float32)
    C = checkpoint_every_n_tokens
    bs = len(seq_len_list)
    total_len = sum(seq_len_list)
    # chunk 起点要是 C 的倍数（cache 侧也是这么校验的，见
    # SingletonPagedKVCache.ckpt_cu_starts），这里两个 seq 分别从位置 0 和 8 开始
    assert all(start % C == 0 for start in [0] + seq_len_list[:-1])

    q = torch.randn(1, total_len, linear_n_v_heads, linear_head_dim, device="cuda")
    k = torch.randn(1, total_len, linear_n_v_heads, linear_head_dim, device="cuda")
    v = torch.randn(1, total_len, linear_n_v_heads, linear_head_dim, device="cuda")
    g = F.logsigmoid(torch.randn(1, total_len, linear_n_v_heads, device="cuda"))
    g = g * torch.rand_like(g)
    beta = torch.randn(1, total_len, linear_n_v_heads, device="cuda").sigmoid()
    initial_state = torch.randn(
        bs, linear_n_v_heads, linear_head_dim, linear_head_dim, device="cuda"
    )

    cu_starts = [0]
    for seq_len in seq_len_list:
        cu_starts.append(cu_starts[-1] + seq_len // C)
    state_checkpoints = torch.empty(
        (cu_starts[-1], linear_n_v_heads, linear_head_dim, linear_head_dim),
        device="cuda",
    )
    out, final_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        seq_len_list=seq_len_list,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=torch.tensor(cu_starts, device="cuda", dtype=torch.int64),
        checkpoint_every_n_tokens=C,
        impl="torch",
    )

    seq_start = 0
    for i, seq_len in enumerate(seq_len_list):
        # 第 t 个 token 之后的 state 等价于用同一个 initial_state 只跑前 t 个 token 得到的
        # 末状态（qk l2norm 是逐 token 的，不影响这个等价关系）
        for j, t in enumerate(range(C, seq_len + 1, C)):
            _, state_after_t = chunk_gated_delta_rule_torch_dense(
                q[:, seq_start : seq_start + t],
                k[:, seq_start : seq_start + t],
                v[:, seq_start : seq_start + t],
                g=g[:, seq_start : seq_start + t],
                beta=beta[:, seq_start : seq_start + t],
                initial_state=initial_state[i : i + 1],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            assert_close(
                state_checkpoints[cu_starts[i] + j],
                state_after_t[0],
                atol=1e-2,
                rtol=1e-2,
            )
        _, seq_final_state = chunk_gated_delta_rule_torch_dense(
            q[:, seq_start : seq_start + seq_len],
            k[:, seq_start : seq_start + seq_len],
            v[:, seq_start : seq_start + seq_len],
            g=g[:, seq_start : seq_start + seq_len],
            beta=beta[:, seq_start : seq_start + seq_len],
            initial_state=initial_state[i : i + 1],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        assert_close(final_state[i], seq_final_state[0], atol=1e-2, rtol=1e-2)
        seq_start += seq_len


def test_chunk_gated_delta_rule_checkpoints_auto_impl():
    """不显式指定 impl 时，带 checkpoint 的调用必须自动落到支持 checkpoint 的实现。

    这里不传 impl（= "auto"），由 register_auto 决定。它以前只看有没有 fla：装了 fla 的机器
    上带 checkpoint 的调用会被派到 fla 实现，而 fla 实现不支持 checkpoint，直接 assert 失败。
    """
    torch.set_default_dtype(torch.float32)
    C = 4
    total_len = 8
    linear_head_dim, linear_n_v_heads = 32, 8

    q = torch.randn(1, total_len, linear_n_v_heads, linear_head_dim, device="cuda")
    k = torch.randn(1, total_len, linear_n_v_heads, linear_head_dim, device="cuda")
    v = torch.randn(1, total_len, linear_n_v_heads, linear_head_dim, device="cuda")
    g = F.logsigmoid(
        torch.randn(1, total_len, linear_n_v_heads, device="cuda")
    ) * torch.rand(1, total_len, linear_n_v_heads, device="cuda")
    beta = torch.randn(1, total_len, linear_n_v_heads, device="cuda").sigmoid()
    state_checkpoints = torch.empty(
        (2, linear_n_v_heads, linear_head_dim, linear_head_dim), device="cuda"
    )

    with AssertOpCalled("chunk_gated_delta_rule", "torch"):
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            seq_len_list=[total_len],
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=torch.tensor([0, 2], device="cuda", dtype=torch.int64),
            checkpoint_every_n_tokens=C,
        )


def test_chunk_gated_delta_rule_checkpoints_invalid_args():
    torch.set_default_dtype(torch.float32)
    C = 4
    seq_len_list = [8]
    num_heads, head_dim = 4, 64
    total_len = sum(seq_len_list)

    q = torch.randn(1, total_len, num_heads, head_dim, device="cuda")
    k = torch.randn(1, total_len, num_heads, head_dim, device="cuda")
    v = torch.randn(1, total_len, num_heads, head_dim, device="cuda")
    g = F.logsigmoid(torch.randn(1, total_len, num_heads, device="cuda"))
    beta = torch.randn(1, total_len, num_heads, device="cuda").sigmoid()
    kwargs = dict(
        g=g,
        beta=beta,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        seq_len_list=seq_len_list,
        impl="torch",
    )

    # checkpoint_every_n_tokens > 0 时必须同时给 state_checkpoints 和 checkpoint_cu_starts
    with pytest.raises(AssertionError):
        chunk_gated_delta_rule(q, k, v, checkpoint_every_n_tokens=C, **kwargs)
    # 不存 checkpoint 时不能给 buffer
    with pytest.raises(AssertionError):
        chunk_gated_delta_rule(
            q,
            k,
            v,
            state_checkpoints=torch.empty(
                (2, num_heads, head_dim, head_dim), device="cuda"
            ),
            checkpoint_cu_starts=torch.tensor([0, 2], device="cuda", dtype=torch.int64),
            **kwargs,
        )
    # 调用方给的个数和算子按 checkpoint_every_n_tokens 数出来的对不上（会写到错位的页上）
    with pytest.raises(AssertionError):
        chunk_gated_delta_rule(
            q,
            k,
            v,
            state_checkpoints=torch.empty(
                (3, num_heads, head_dim, head_dim), device="cuda"
            ),
            checkpoint_cu_starts=torch.tensor([0, 3], device="cuda", dtype=torch.int64),
            checkpoint_every_n_tokens=C,
            **kwargs,
        )


@pytest.mark.parametrize("seq_len_list", [[4], [10], [8, 5]])
@pytest.mark.parametrize("checkpoint_every_n_tokens", [4])
@pytest.mark.parametrize("linear_head_dim", [128])
@pytest.mark.parametrize("linear_n_v_heads", [32])
def test_chunk_kimi_delta_attention_checkpoints(
    seq_len_list,
    checkpoint_every_n_tokens,
    linear_head_dim,
    linear_n_v_heads,
):
    """kimi 版 chunk 算子（GLM-5.3 的 GDN）也要按同一套约定写出 checkpoint。"""
    torch.set_default_dtype(torch.float32)
    C = checkpoint_every_n_tokens
    bs = len(seq_len_list)
    total_len = sum(seq_len_list)
    # chunk 起点要是 C 的倍数（cache 侧也是这么校验的，见
    # SingletonPagedKVCache.ckpt_cu_starts），这里两个 seq 分别从位置 0 和 8 开始
    assert all(start % C == 0 for start in [0] + seq_len_list[:-1])

    q = torch.randn(1, total_len, linear_n_v_heads, linear_head_dim, device="cuda")
    k = torch.randn(1, total_len, linear_n_v_heads, linear_head_dim, device="cuda")
    v = torch.randn(1, total_len, linear_n_v_heads, linear_head_dim, device="cuda")
    # kimi 的 g 是逐 head_dim 的（不是逐 head 一个标量）
    g = F.logsigmoid(
        torch.randn(1, total_len, linear_n_v_heads, linear_head_dim, device="cuda")
    )
    g = g * torch.rand_like(g)
    beta = torch.randn(1, total_len, linear_n_v_heads, device="cuda").sigmoid()
    initial_state = torch.randn(
        bs, linear_n_v_heads, linear_head_dim, linear_head_dim, device="cuda"
    )

    cu_starts = [0]
    for seq_len in seq_len_list:
        cu_starts.append(cu_starts[-1] + seq_len // C)
    state_checkpoints = torch.empty(
        (cu_starts[-1], linear_n_v_heads, linear_head_dim, linear_head_dim),
        device="cuda",
    )
    out, final_state = chunk_kimi_delta_attention(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        seq_len_list=seq_len_list,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=torch.tensor(cu_starts, device="cuda", dtype=torch.int64),
        checkpoint_every_n_tokens=C,
        impl="torch",
    )

    seq_start = 0
    for i, seq_len in enumerate(seq_len_list):
        # 第 t 个 token 之后的 state 等价于用同一个 initial_state 只跑前 t 个 token 得到的
        # 末状态（qk l2norm 是逐 token 的，不影响这个等价关系）
        for j, t in enumerate(range(C, seq_len + 1, C)):
            _, state_after_t = chunk_kimi_delta_attention_torch_dense(
                q[:, seq_start : seq_start + t],
                k[:, seq_start : seq_start + t],
                v[:, seq_start : seq_start + t],
                g=g[:, seq_start : seq_start + t],
                beta=beta[:, seq_start : seq_start + t],
                initial_state=initial_state[i : i + 1],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            assert_close(
                state_checkpoints[cu_starts[i] + j],
                state_after_t[0],
                atol=1e-2,
                rtol=1e-2,
            )
        _, seq_final_state = chunk_kimi_delta_attention_torch_dense(
            q[:, seq_start : seq_start + seq_len],
            k[:, seq_start : seq_start + seq_len],
            v[:, seq_start : seq_start + seq_len],
            g=g[:, seq_start : seq_start + seq_len],
            beta=beta[:, seq_start : seq_start + seq_len],
            initial_state=initial_state[i : i + 1],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        assert_close(final_state[i], seq_final_state[0], atol=1e-2, rtol=1e-2)
        seq_start += seq_len


def test_chunk_kimi_delta_attention_checkpoints_auto_impl():
    """不显式指定 impl 时，带 checkpoint 的调用必须自动落到支持 checkpoint 的 torch 实现。

    fla 的 kda 实现不支持 checkpoint，auto 只有看 checkpoint 参数才能避开它。
    """
    torch.set_default_dtype(torch.float32)
    C = 4
    total_len = 8
    linear_head_dim, linear_n_v_heads = 32, 8

    q = torch.randn(1, total_len, linear_n_v_heads, linear_head_dim, device="cuda")
    k = torch.randn(1, total_len, linear_n_v_heads, linear_head_dim, device="cuda")
    v = torch.randn(1, total_len, linear_n_v_heads, linear_head_dim, device="cuda")
    g = F.logsigmoid(
        torch.randn(1, total_len, linear_n_v_heads, linear_head_dim, device="cuda")
    ) * torch.rand(1, total_len, linear_n_v_heads, linear_head_dim, device="cuda")
    beta = torch.randn(1, total_len, linear_n_v_heads, device="cuda").sigmoid()
    state_checkpoints = torch.empty(
        (2, linear_n_v_heads, linear_head_dim, linear_head_dim), device="cuda"
    )

    with AssertOpCalled("chunk_kimi_delta_attention", "torch"):
        chunk_kimi_delta_attention(
            q,
            k,
            v,
            g=g,
            beta=beta,
            seq_len_list=[total_len],
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=torch.tensor([0, 2], device="cuda", dtype=torch.int64),
            checkpoint_every_n_tokens=C,
        )


def test_chunk_kimi_delta_attention_checkpoints_invalid_args():
    torch.set_default_dtype(torch.float32)
    C = 4
    seq_len_list = [8]
    num_heads, head_dim = 4, 64
    total_len = sum(seq_len_list)

    q = torch.randn(1, total_len, num_heads, head_dim, device="cuda")
    k = torch.randn(1, total_len, num_heads, head_dim, device="cuda")
    v = torch.randn(1, total_len, num_heads, head_dim, device="cuda")
    g = F.logsigmoid(torch.randn(1, total_len, num_heads, head_dim, device="cuda"))
    beta = torch.randn(1, total_len, num_heads, device="cuda").sigmoid()
    kwargs = dict(
        g=g,
        beta=beta,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        seq_len_list=seq_len_list,
        impl="torch",
    )

    # checkpoint_every_n_tokens > 0 时必须同时给 state_checkpoints 和 checkpoint_cu_starts
    with pytest.raises(AssertionError):
        chunk_kimi_delta_attention(q, k, v, checkpoint_every_n_tokens=C, **kwargs)
    # 不存 checkpoint 时不能给 buffer
    with pytest.raises(AssertionError):
        chunk_kimi_delta_attention(
            q,
            k,
            v,
            state_checkpoints=torch.empty(
                (2, num_heads, head_dim, head_dim), device="cuda"
            ),
            checkpoint_cu_starts=torch.tensor([0, 2], device="cuda", dtype=torch.int64),
            **kwargs,
        )
    # 调用方给的个数和算子按 checkpoint_every_n_tokens 数出来的对不上（会写到错位的页上）
    with pytest.raises(AssertionError):
        chunk_kimi_delta_attention(
            q,
            k,
            v,
            state_checkpoints=torch.empty(
                (3, num_heads, head_dim, head_dim), device="cuda"
            ),
            checkpoint_cu_starts=torch.tensor([0, 3], device="cuda", dtype=torch.int64),
            checkpoint_every_n_tokens=C,
            **kwargs,
        )
