import pytest
import torch
import torch.nn.functional as F

from chitu.device_type import is_enflame, is_muxi
from chitu.lazy import eval_lazy
from chitu.ops.linear_attn import (
    _prepare_kimi_delta_beta,
    _prepare_kimi_delta_gate,
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_torch_dense,
    chunk_kimi_delta_attention,
    chunk_kimi_delta_attention_torch_dense,
    recurrent_gated_delta_rule_compact_qk_gcu,
    has_boltops_kda,
    has_fla_kda,
    recurrent_gated_delta_rule_torch,
    recurrent_kimi_delta_attention,
    recurrent_kimi_delta_attention_torch,
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


@pytest.mark.parametrize(
    "handler_name",
    ["_recurrent_kimi_delta_attention_fla", "recurrent_kimi_delta_attention_torch"],
)
def test_kda_decode_k_first_backends_reject_v_first(handler_name):
    import chitu.ops.linear_attn as linear_attn

    with pytest.raises(ValueError, match="only supports K-first"):
        getattr(linear_attn, handler_name)(
            None,
            None,
            None,
            None,
            None,
            None,
            True,
            state_v_first=True,
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
@pytest.mark.skipif(not is_enflame(), reason="Enflame GCU is required")
@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize(
    (
        "single_step",
        "raw_beta_delta",
        "mixed_decay",
        "grouped_qk",
        "mixed_query_scale",
        "fused_qk_scale",
        "paged_mixed_decay",
    ),
    [
        (False, False, False, False, False, False, False),
        (True, False, False, False, False, False, False),
        (True, True, False, False, False, False, False),
        (True, False, True, False, False, False, False),
        (True, True, True, False, False, False, False),
        (True, False, False, True, False, False, False),
        (True, True, True, True, False, False, False),
        (True, True, True, True, True, False, False),
        (True, True, True, True, True, True, False),
        (True, True, True, True, True, False, True),
        (True, True, True, True, True, True, True),
    ],
)
def test_recurrent_gated_delta_rule_compact_qk_gcu_is_exact(
    batch_size,
    single_step,
    raw_beta_delta,
    mixed_decay,
    grouped_qk,
    mixed_query_scale,
    fused_qk_scale,
    paged_mixed_decay,
    monkeypatch,
):
    monkeypatch.setenv(
        "CHITU_GCU_QWEN3_NEXT_RECURRENCE_SINGLE_STEP", str(int(single_step))
    )
    monkeypatch.setenv(
        "CHITU_GCU_QWEN3_NEXT_RAW_BETA_DELTA_BATCH", str(int(raw_beta_delta))
    )
    torch.manual_seed(20260829 + batch_size)
    qkv_storage = (
        torch.randn(batch_size, 1, 128, 128, device="cuda") * 0.25
    ).bfloat16()
    qkv = qkv_storage[:, :, :80]
    query = qkv[:, :, :16]
    key = qkv[:, :, 16:32]
    value = qkv[:, :, 32:]
    g = -(torch.rand(batch_size, 1, 48, device="cuda") * 0.8 + 0.0001)
    beta_storage = (
        torch.randn(batch_size, 1, 96, device="cuda") * 0.5
    ).bfloat16()
    beta_logits = beta_storage[:, :, :48]
    assert query.stride(0) == 128 * 128
    assert value.stride(0) == 128 * 128
    assert beta_logits.stride(0) == 96
    beta = beta_logits.sigmoid()
    state = (
        torch.randn(batch_size, 48, 128, 128, device="cuda") * 0.02
    ).bfloat16()
    state_cache = None
    page_table = None
    if paged_mixed_decay:
        state_cache = torch.zeros(
            8, 1, 48, 128, 128, device="cuda", dtype=torch.bfloat16
        )
        state_cache[:batch_size, 0].copy_(state.flip(0))
        page_table = torch.arange(
            batch_size - 1, -1, -1, device="cuda", dtype=torch.int32
        )[:, None].contiguous()
    query_full = query.repeat_interleave(3, dim=-2)
    key_full = key.repeat_interleave(3, dim=-2)
    expected_output, expected_state = recurrent_gated_delta_rule_torch(
        query_full,
        key_full,
        value,
        g,
        beta,
        state,
        True,
        True,
    )
    output, state_out = recurrent_gated_delta_rule_compact_qk_gcu(
        query,
        key,
        value,
        g,
        beta_logits if raw_beta_delta else beta,
        None if paged_mixed_decay else state,
        True,
        torch.ones(128, dtype=torch.float32, device="cuda"),
        single_step=single_step,
        raw_beta_delta=raw_beta_delta,
        mixed_decay=mixed_decay,
        grouped_qk=grouped_qk,
        mixed_query_scale=mixed_query_scale,
        fused_qk_scale=fused_qk_scale,
        query_scale=torch.full(
            (1,), 128**-0.5, dtype=torch.float32, device="cuda"
        ),
        initial_state_cache=state_cache,
        page_table=page_table,
        paged_mixed_decay=paged_mixed_decay,
    )

    assert torch.equal(output, expected_output)
    assert torch.equal(state_out, expected_state)


@pytest.mark.parametrize("lower_bound", [-5.0, None])
def test_kimi_delta_gate_beta_lazy(lower_bound):
    torch.manual_seed(20261009)
    raw_g = torch.randn(1, 7, 3, 8, device="cuda", dtype=torch.bfloat16)
    raw_beta = torch.randn(1, 7, 3, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(3, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(3, 8, device="cuda", dtype=torch.float32)

    g = _prepare_kimi_delta_gate(
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )
    beta = _prepare_kimi_delta_beta(raw_beta=raw_beta)
    assert not g.is_evaluated()
    assert not beta.is_evaluated()

    decay_rate = A_log.float().exp().unsqueeze(-1)
    gate_source = raw_g.float() + dt_bias.float()
    expected_g = (
        lower_bound * torch.sigmoid(decay_rate * gate_source)
        if lower_bound is not None
        else -decay_rate * F.softplus(gate_source)
    )
    assert_close(eval_lazy(g), expected_g)
    assert_close(eval_lazy(beta), raw_beta.sigmoid())


@pytest.mark.parametrize(
    "head_dim,lower_bound,has_raw_gates,requires_checkpoints",
    [
        (128, -5.0, True, False),
        (64, -5.0, True, False),
        (128, None, True, False),
        (128, -5.0, False, False),
        (128, -5.0, True, True),
    ],
)
def test_chunk_kimi_delta_attention_dispatch(
    head_dim, lower_bound, has_raw_gates, requires_checkpoints
):
    """Initialization and runtime use the same registered selection policy."""
    query = torch.empty(1, 8, 2, head_dim, device="meta")
    raw_beta = torch.empty(1, 8, 2, device="meta")
    if has_raw_gates:
        g = _prepare_kimi_delta_gate(
            raw_g=query,
            raw_beta=raw_beta,
            A_log=torch.empty(2, device="meta"),
            dt_bias=torch.empty(2, head_dim, device="meta"),
            lower_bound=lower_bound,
        )
        beta = _prepare_kimi_delta_beta(raw_beta=raw_beta)
    else:
        g = torch.empty_like(query)
        beta = raw_beta

    metadata = dict(
        head_dim=head_dim,
        lower_bound=lower_bound,
        has_raw_gates=has_raw_gates,
        requires_checkpoints=requires_checkpoints,
    )
    metadata_impl = chunk_kimi_delta_attention.resolve_impl(**metadata)
    runtime_impl = chunk_kimi_delta_attention.resolve_impl(
        query,
        query,
        query,
        g,
        beta,
        checkpoint_every_n_tokens=4 if requires_checkpoints else 0,
    )
    assert metadata_impl == runtime_impl
    boltops_supported = (
        head_dim == 128
        and lower_bound is not None
        and has_raw_gates
        and not requires_checkpoints
    )
    if requires_checkpoints:
        assert metadata_impl == "torch"
    elif has_boltops_kda and boltops_supported:
        assert metadata_impl == "boltops_kda"
    else:
        assert metadata_impl == ("fla" if has_fla_kda else "torch")
    if not boltops_supported:
        with pytest.raises(NotImplementedError):
            chunk_kimi_delta_attention.resolve_impl(
                **metadata, requested_impl="boltops_kda"
            )


@pytest.mark.skipif(not has_boltops_kda, reason="requires BoltOps on a Hygon device")
@pytest.mark.parametrize("checkpoint_interval", [None, 32])
def test_fixed_prefill_state_feeds_boltops_decode(checkpoint_interval):
    """Both fixed prefill backends feed BoltOps decode in their native layouts."""
    torch.manual_seed(20261010)
    batch, prefill_tokens, heads, k_dim, v_dim = 1, 64, 2, 128, 32
    query = torch.randn(
        batch,
        prefill_tokens,
        heads,
        k_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn_like(query)
    value = torch.randn(
        batch,
        prefill_tokens,
        heads,
        v_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    raw_g = torch.randn_like(query) * 0.1
    raw_beta = torch.randn(
        batch,
        prefill_tokens,
        heads,
        device="cuda",
        dtype=torch.bfloat16,
    )
    A_log = torch.randn(heads, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(heads, k_dim, device="cuda", dtype=torch.float32) * 0.1
    state_v_first = checkpoint_interval is None
    state_dims = (v_dim, k_dim) if state_v_first else (k_dim, v_dim)
    initial_state = (
        torch.randn(batch, heads, *state_dims, device="cuda", dtype=torch.float32)
        * 0.01
    )
    cu_seqlens = torch.tensor([0, prefill_tokens], device="cuda", dtype=torch.int32)

    g = _prepare_kimi_delta_gate(
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
    )
    beta = _prepare_kimi_delta_beta(raw_beta=raw_beta)
    prefill_impl = "boltops_kda" if state_v_first else "torch"
    checkpoint_kwargs = {}
    if checkpoint_interval is not None:
        n_checkpoints = prefill_tokens // checkpoint_interval
        checkpoint_kwargs = dict(
            state_checkpoints=torch.empty(
                n_checkpoints,
                heads,
                *state_dims,
                device="cuda",
                dtype=torch.float32,
            ),
            checkpoint_cu_starts=torch.tensor(
                [0, n_checkpoints], device="cuda", dtype=torch.int64
            ),
            checkpoint_every_n_tokens=checkpoint_interval,
        )
    with AssertOpCalled("chunk_kimi_delta_attention", prefill_impl):
        prefill_out, prefill_state = chunk_kimi_delta_attention(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            seq_len_list=[prefill_tokens],
            impl=prefill_impl,
            **checkpoint_kwargs,
        )
    assert g.is_evaluated() == (prefill_impl == "torch")
    assert beta.is_evaluated() == (prefill_impl == "torch")
    assert prefill_out.shape == value.shape
    assert prefill_state.shape == initial_state.shape

    # Reference-only views: the actual prefill -> decode path below passes
    # the native state through unchanged, including non-square K/V dimensions.
    expected_prefill_out, expected_prefill_state = recurrent_kimi_delta_attention_torch(
        query,
        key,
        value,
        eval_lazy(g),
        eval_lazy(beta),
        initial_state.transpose(-1, -2) if state_v_first else initial_state,
        True,
        use_qk_l2norm_in_kernel=True,
    )
    assert_close(prefill_out, expected_prefill_out, atol=1e-2, rtol=1e-2)
    assert_close(
        prefill_state.transpose(-1, -2) if state_v_first else prefill_state,
        expected_prefill_state,
        atol=1e-2,
        rtol=1e-2,
    )

    decode_query = torch.randn(
        batch, 1, heads, k_dim, device="cuda", dtype=torch.bfloat16
    )
    decode_key = torch.randn_like(decode_query)
    decode_value = torch.randn(
        batch, 1, heads, v_dim, device="cuda", dtype=torch.bfloat16
    )
    decode_g = -torch.rand(batch, 1, heads, k_dim, device="cuda", dtype=torch.float32)
    decode_beta = torch.rand(batch, 1, heads, device="cuda", dtype=torch.bfloat16)
    expected_out, expected_state = recurrent_kimi_delta_attention_torch(
        decode_query,
        decode_key,
        decode_value,
        decode_g,
        decode_beta,
        prefill_state.transpose(-1, -2) if state_v_first else prefill_state,
        True,
        use_qk_l2norm_in_kernel=True,
    )
    if state_v_first:
        expected_state = expected_state.transpose(-1, -2)
    with AssertOpCalled("recurrent_kimi_delta_attention", "boltops_kda"):
        actual_out, actual_state = recurrent_kimi_delta_attention(
            decode_query,
            decode_key,
            decode_value,
            decode_g,
            decode_beta,
            prefill_state,
            True,
            use_qk_l2norm_in_kernel=True,
            state_v_first=state_v_first,
            impl="boltops_kda",
        )

    assert_close(actual_out, expected_out, atol=3e-3, rtol=3e-3)
    assert_close(actual_state, expected_state, atol=4e-3, rtol=4e-3)
