# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""``chitu.cp_linear_attention`` 纯张量部分的单测。

只覆盖不依赖分布式进程组的逻辑：连续切分计划、交错<->连续的 all_to_all 路由、
仿射映射的前缀链式合并。跨 rank 的 all_to_all 与算子接入需要多卡环境验证。
"""

import torch

import pytest

from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.cp_linear_attention import (
    build_contiguous_cp_plan,
    build_layout_route,
    build_reshuffle_route,
    contiguous_interval,
    contiguous_rank_indices,
    contiguous_rank_of_position,
    interleaved_rank_indices,
    merge_prefix_states,
    segment_affine_transition,
)
from chitu.cp_utils import (
    CPContextContiguous,
    CPContextInterleaved,
    from_layout,
    to_layout,
)
from chitu.ops import chunk_kimi_delta_attention


def _plan(cu, rank, world_size):
    return build_contiguous_cp_plan(cu, rank, world_size)


def _contiguous_context(rank, world_size):
    from types import SimpleNamespace

    return CPContextContiguous(
        world_size,
        SimpleNamespace(rank_in_group=rank, is_first_rank=rank == 0),
    )


def test_layout_transform_is_noop_for_contiguous_context():
    ctx = _contiguous_context(1, 4)
    x = torch.arange(12, dtype=torch.float32).view(3, 4)

    x_target, restore_route = to_layout(x, ctx, 12, "contiguous")

    assert x_target is x
    assert restore_route is None
    assert from_layout(x_target, restore_route, ctx) is x


def test_layout_transform_round_trips_interleaved_context(monkeypatch):
    from types import SimpleNamespace

    ctx = CPContextInterleaved(
        4,
        SimpleNamespace(
            rank_in_group=1,
            is_first_rank=False,
            gpu_group=None,
        ),
    )
    x = torch.arange(12, dtype=torch.float32).view(3, 4)
    calls = []

    def _fake_route(world_size, total_tokens, *, to_contiguous):
        tag = "fwd" if to_contiguous else "bwd"
        return (None, None, tag, tag, [[3]], [[3]])

    def _fake_all_to_all(x, rank, send, recv, send_counts, recv_counts, group):
        calls.append((rank, send, recv, group))
        return x + (1.0 if send == "fwd" else -1.0)

    monkeypatch.setattr("chitu.cp_linear_attention.build_layout_route", _fake_route)
    monkeypatch.setattr(
        "chitu.cp_linear_attention.all_to_all_reshape", _fake_all_to_all
    )

    x_target, restore_route = to_layout(x, ctx, 12, "contiguous")
    x_restored = from_layout(x_target, restore_route, ctx)

    assert calls == [(1, "fwd", "fwd", None), (1, "bwd", "bwd", None)]
    torch.testing.assert_close(x_target, x + 1.0)
    torch.testing.assert_close(x_restored, x)


@pytest.mark.parametrize("rank", range(4))
def test_contiguous_context_rows_match_contiguous_plan(rank):
    """CPContextContiguous 的 local rows 与 build_contiguous_cp_plan 一致。"""
    total = 10
    cu = [0, total]
    plan = _plan(cu, rank, 4)
    ctx = _contiguous_context(rank, 4)
    freqs = BatchedFreqsCis(
        torch.arange(total, dtype=torch.float32).view(-1, 1),
        torch.zeros(total, 1, dtype=torch.float32),
    )

    tokens, _, local_freqs = ctx.split_prefill(torch.arange(total), None, freqs, total)

    assert ctx.n_local == plan.part_len
    assert tokens.tolist() == list(range(plan.interval_start, plan.interval_end))
    assert local_freqs.cos[:, 0].tolist() == [
        float(i) for i in range(plan.interval_start, plan.interval_end)
    ]


def test_plan_single_long_sequence_spans_all_ranks():
    cu = [0, 30]
    plans = [_plan(cu, r, 3) for r in range(3)]
    assert [p.n_local for p in plans] == [10, 10, 10]
    assert [p.local_cu_seqlens for p in plans] == [[0, 10], [0, 10], [0, 10]]
    assert [p.chain_len for p in plans] == [[0], [1], [2]]
    assert [p.continues_from_prev_rank for p in plans] == [False, True, True]
    assert [p.pre_conv_tokens for p in plans] == [0, 10, 20]


def test_plan_aligned_sequence_boundaries_need_no_chain():
    cu = [0, 5, 10, 15, 20, 25, 30]
    plans = [_plan(cu, r, 3) for r in range(3)]
    assert [p.local_lens for p in plans] == [[5, 5], [5, 5], [5, 5]]
    assert [p.global_seq_indices for p in plans] == [[0, 1], [2, 3], [4, 5]]
    assert all(not p.continues_from_prev_rank for p in plans)
    assert all(p.pre_conv_tokens == 0 for p in plans)


def test_plan_two_sequences_with_one_crossing():
    # seq0=[0,20), seq1=[20,32); W=4, part_len=8
    cu = [0, 20, 32]
    plans = [_plan(cu, r, 4) for r in range(4)]
    assert [p.local_lens for p in plans] == [[8], [8], [4, 4], [8]]
    assert [p.global_seq_indices for p in plans] == [[0], [0], [0, 1], [1]]
    # rank2 的第一段仍续 seq0，chain=2
    # 第二段是 seq1，在本区间内新起，chain=0
    assert plans[2].chain_len == [2, 0]
    # rank3 的第一段是 seq1，它从 rank2 续过来
    assert plans[3].chain_len == [1]
    assert plans[2].pre_conv_tokens == 16
    assert plans[3].pre_conv_tokens == 4


def test_contiguous_interval_is_a_partition():
    """区间必须无缝覆盖 [0, T)，且各段长最多相差 1（T 不整除 W 也一样）。"""
    for total, world in [(31, 3), (65, 4), (10, 3), (7, 7), (33, 4), (32, 4)]:
        intervals = [contiguous_interval(r, world, total) for r in range(world)]
        assert intervals[0][0] == 0
        assert intervals[-1][1] == total
        for prev, cur in zip(intervals, intervals[1:]):
            assert prev[1] == cur[0], (total, world, prev, cur)
        lens = [end - start for start, end in intervals]
        assert max(lens) - min(lens) <= 1
        # 每个位置都能且只能被一个 rank 认领
        owners = [contiguous_rank_of_position(p, world, total) for p in range(total)]
        assert owners == sorted(owners)
        for r, (start, end) in enumerate(intervals):
            assert owners[start:end] == [r] * (end - start)


def test_plan_uneven_split_uses_floor_boundaries():
    """total 不整除 world 时按 floor(r*T/W) 切开，段长为 [10, 10, 11]。"""
    cu = [0, 31]
    plans = [_plan(cu, r, 3) for r in range(3)]
    assert [p.n_local for p in plans] == [10, 10, 11]
    assert [p.local_cu_seqlens for p in plans] == [[0, 10], [0, 10], [0, 11]]
    assert [p.chain_len for p in plans] == [[0], [1], [2]]
    assert [p.pre_conv_tokens for p in plans] == [0, 10, 20]


def test_plan_uneven_split_multi_seq():
    """两条 seq + 不整除：seq1 正好从 rank2 的区间起点开始，不需要跨 rank。"""
    cu = [0, 20, 31]
    plans = [_plan(cu, r, 3) for r in range(3)]
    assert [p.global_seq_indices for p in plans] == [[0], [0], [1]]
    assert [p.local_lens for p in plans] == [[10], [10], [11]]
    assert [p.chain_len for p in plans] == [[0], [1], [0]]


def test_contiguous_interval_rejects_out_of_range():
    with pytest.raises(ValueError):
        contiguous_interval(3, 3, 10)
    with pytest.raises(ValueError):
        contiguous_rank_of_position(10, 3, 10)


def _simulate_all_to_all(
    world_size,
    total,
    local_global_ids,
    send_index,
    recv_index,
    send_counts,
    recv_counts,
):
    """在单进程里模拟 all_to_all：把每个 rank 的 send buffer 按目的地铺开，
    再按来源拼给每个 rank，最后用 recv_index 还原顺序。"""
    # send_buf[r] 是 rank r 的发送 buffer（行内容用全局下标标识）
    send_buf = [local_global_ids[r][send_index[r]] for r in range(world_size)]
    # 每个来源 rank 在接收 buffer 里的起始偏移
    offsets = [[0] * world_size for _ in range(world_size)]
    for dst in range(world_size):
        acc = 0
        for src in range(world_size):
            offsets[dst][src] = acc
            acc += send_counts[src][dst]
        assert acc == sum(recv_counts[dst])
    rows = []
    for dst in range(world_size):
        src_blocks = []
        for src in range(world_size):
            start = 0
            for d in range(dst):
                start += send_counts[src][d]
            n = send_counts[src][dst]
            src_blocks.append(send_buf[src][start : start + n])
        recv_buf = (
            torch.cat(src_blocks) if src_blocks else torch.empty(0, dtype=torch.long)
        )
        rows.append(recv_buf[recv_index[dst]])
    return rows


@pytest.mark.parametrize("to_contiguous", [True, False])
def test_route_matches_target_layout(to_contiguous):
    world_size, total = 4, 32
    inter = [interleaved_rank_indices(r, world_size, total) for r in range(world_size)]
    cont = [contiguous_rank_indices(r, world_size, total) for r in range(world_size)]
    src, dst = (inter, cont) if to_contiguous else (cont, inter)

    send_index, recv_index, send_counts, recv_counts = build_reshuffle_route(
        src, dst, total
    )
    rows = _simulate_all_to_all(
        world_size, total, src, send_index, recv_index, send_counts, recv_counts
    )
    for r in range(world_size):
        assert torch.equal(rows[r], dst[r]), (to_contiguous, r)


def test_route_genuinely_uneven_rows():
    # 收发两个方向的行数都不相等：源 rank 行数 4/3/3，目标 rank 行数 3/4/3；
    # 单个 rank 的「发给不同的 rank 的行数」以及「从不同 rank 收到的行数」也都不相等
    world_size, total = 3, 10
    src = [
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        torch.tensor([4, 5, 6], dtype=torch.long),
        torch.tensor([7, 8, 9], dtype=torch.long),
    ]
    dst = [
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([3, 4, 5, 6], dtype=torch.long),
        torch.tensor([7, 8, 9], dtype=torch.long),
    ]
    send_index, recv_index, send_counts, recv_counts = build_reshuffle_route(
        src, dst, total
    )
    for r in range(world_size):
        assert sum(send_counts[r]) == int(src[r].numel())
        assert sum(recv_counts[r]) == int(dst[r].numel())
    assert send_counts[0] != send_counts[1]
    assert recv_counts[0] != recv_counts[1]
    rows = _simulate_all_to_all(
        world_size, total, src, send_index, recv_index, send_counts, recv_counts
    )
    for r in range(world_size):
        assert torch.equal(rows[r], dst[r]), (r, rows[r].tolist(), dst[r].tolist())


def test_build_layout_route_shapes():
    src, dst, send_index, recv_index, send_counts, recv_counts = build_layout_route(
        4, 32, to_contiguous=True
    )
    assert [int(p.numel()) for p in src] == [8, 8, 8, 8]
    assert [int(p.numel()) for p in dst] == [8, 8, 8, 8]
    for r in range(4):
        assert sum(send_counts[r]) == 8
        assert sum(recv_counts[r]) == 8


def test_layout_route_roundtrip_uneven_tokens():
    """T 不整除 world 时，模型实际走的重排仍要是合法置换（两个方向都测）。"""
    world_size, total = 4, 65
    for to_contiguous in (True, False):
        src, dst, send_index, recv_index, send_counts, recv_counts = build_layout_route(
            world_size, total, to_contiguous=to_contiguous
        )
        assert sum(int(p.numel()) for p in src) == total
        assert sum(int(p.numel()) for p in dst) == total
        rows = _simulate_all_to_all(
            world_size, total, src, send_index, recv_index, send_counts, recv_counts
        )
        for r in range(world_size):
            assert torch.equal(rows[r], dst[r]), (to_contiguous, r, rows[r], dst[r])


def test_merge_prefix_states_empty_slots_is_identity():
    h, k, v = 2, 4, 3
    s0 = torch.randn(h, k, v, dtype=torch.float32)
    got = merge_prefix_states(torch.empty(0, h, k, k), torch.empty(0, h, k, v), s0)
    assert torch.allclose(got, s0)


def _random_kda_inputs(T, H, K, V, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(1, T, H, K)
    k = torch.randn(1, T, H, K)
    v = torch.randn(1, T, H, V)
    beta = torch.rand(1, T, H)
    # KDA 的 g 是 log 域的衰减，取负值
    g = -torch.nn.functional.softplus(torch.randn(1, T, H, K))
    return q, k, v, beta, g


def _run_kda(q, k, v, beta, g, cu, lengths, initial_state, impl="torch"):
    return chunk_kimi_delta_attention(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        seq_len_list=list(lengths),
        impl=impl,
    )


def test_affine_merge_reproduces_full_sequence():
    """段仿射映射 + 前缀链式合并，必须等价于在整条序列上直接跑一遍。

    这条单进程测试用的就是连续切分 CP 的数学核心（每个 rank 只算本段并把段映射
    ``S_out = M @ S_in + S_ext`` 交给后面的 rank），所以不需要多卡就能验证 CP 的正确性。
    """
    T, H, K, V = 48, 2, 8, 6
    world_size = 3
    part = T // world_size
    q, k, v, beta, g = _random_kda_inputs(T, H, K, V)

    full_out, full_state = _run_kda(
        q,
        k,
        v,
        beta,
        g,
        torch.tensor([0, T], dtype=torch.int32),
        [T],
        None,
    )

    state = torch.zeros(H, K, V)
    seg_out = seg_state = None
    for i in range(world_size):
        sl = slice(i * part, (i + 1) * part)
        m, ext = segment_affine_transition(
            chunk_kimi_delta_attention,
            q[:, sl],
            k[:, sl],
            v[:, sl],
            g[:, sl],
            beta[:, sl],
            torch.tensor([0, part], dtype=torch.int32),
            [part],
            use_qk_l2norm_in_kernel=True,
            impl="torch",
        )
        seg_out, seg_state = _run_kda(
            q[:, sl],
            k[:, sl],
            v[:, sl],
            beta[:, sl],
            g[:, sl],
            torch.tensor([0, part], dtype=torch.int32),
            [part],
            state.unsqueeze(0),
        )
        torch.testing.assert_close(seg_out[0], full_out[0][sl], atol=1e-4, rtol=1e-4)
        state = m[0] @ state + ext[0]

    torch.testing.assert_close(seg_state[0], full_state[0], atol=1e-4, rtol=1e-4)


def test_affine_merge_is_associative_over_slots():
    """merge_prefix_states 与逐段 apply 一致（fp32 链式相乘的语义）。"""
    torch.manual_seed(1)
    n_slots, h, k, v = 4, 2, 5, 3
    m = torch.randn(n_slots, h, k, k) * 0.2
    ext = torch.randn(n_slots, h, k, v)
    s0 = torch.randn(h, k, v)

    got = merge_prefix_states(m, ext, s0)
    want = s0
    for j in range(n_slots):
        want = m[j] @ want + ext[j]
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)

    # 分两半合并：先作用前一半，再把后一半作用到结果上
    half = merge_prefix_states(m[:2], ext[:2], s0)
    two_step = merge_prefix_states(m[2:], ext[2:], half)
    torch.testing.assert_close(got, two_step, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
