# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""在 CPU 上逐 rank 模拟连续切分 CP 的**整层**计算，和 all-gather 复制路径对比。

``test_cp_linear_attention.py`` 只验证了段仿射映射/重排路由这些构件；这里把
``Glm5NextLinearAttention._forward_prefill_cp`` 的调用顺序完整复刻一遍
（交错->连续重排、短卷积左边界补 tail、本地 KDA、跨 rank 段映射合并、state 归约），
和「整段重算」的参考实现对比输出与最终 state。这样不需要多卡也不需要 GPU 就能
验证整层逻辑；多 rank 的集合通信在这里退化成串行取值（等价于 all_to_all/all_gather
收到的内容）。
"""

import pytest
import torch

from chitu.cp_linear_attention import (
    build_contiguous_cp_plan,
    contiguous_rank_indices,
    merge_prefix_states,
    segment_affine_transition,
)
from chitu.ops.causal_conv import causal_conv1d_prefill
from chitu.ops.linear_attn import chunk_kimi_delta_attention


def _setup(T=64, H=2, K=8, V=6, kernel=4, seed=0):
    torch.manual_seed(seed)
    C = 2 * H * K + H * V
    qkv = torch.randn(T, C, dtype=torch.float32) * 0.5
    weight = torch.randn(C, 1, kernel, dtype=torch.float32) * 0.3
    beta = torch.rand(1, T, H, dtype=torch.float32)
    g = -torch.nn.functional.softplus(torch.randn(1, T, H, K, dtype=torch.float32))
    return qkv, weight, beta, g


def _split_qkv(conv_out, H, K, V):
    q, k, v = torch.split(conv_out, [H * K, H * K, H * V], dim=-1)
    return (
        q.reshape(-1, H, K),
        k.reshape(-1, H, K),
        v.reshape(-1, H, V),
    )


def _int32(xs):
    return torch.tensor(list(xs), dtype=torch.int32)


def _reference(qkv, weight, beta, g, cu, lens, H, K, V, init_conv=None, init_rec=None):
    """all-gather 复制路径：整条全局序列上一把算完。

    ``init_conv`` / ``init_rec`` 是上一段（chunked prefill 的前一个 step）留下的
    conv/recurrent state；不传就是全新请求（全零）。
    """
    C = qkv.shape[-1]
    kernel = weight.shape[2]
    conv_state = (
        torch.zeros(len(lens), C, kernel) if init_conv is None else init_conv.clone()
    )
    conv_out, conv_state_new = causal_conv1d_prefill(
        qkv, conv_state, weight, _int32(cu), impl="ref"
    )
    q, k, v = _split_qkv(conv_out, H, K, V)
    out, last_state = chunk_kimi_delta_attention(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        g=g,
        beta=beta,
        initial_state=(
            torch.zeros(len(lens), H, K, V) if init_rec is None else init_rec.clone()
        ),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=_int32(cu),
        seq_len_list=list(lens),
        impl="torch",
    )
    return out[0], last_state, conv_state_new


def _contiguous(
    qkv, weight, beta, g, cu, lens, H, K, V, world, init_conv=None, init_rec=None
):
    """逐 rank 串行模拟 ``_forward_prefill_cp`` 的每一步。"""
    T = int(cu[-1])
    C = qkv.shape[-1]
    kernel = weight.shape[2]
    out_global = torch.zeros(T, H, V, dtype=torch.float32)
    conv_full = torch.zeros(len(lens), C, kernel)
    rec_full = torch.zeros(len(lens), H, K, V)
    init_conv = torch.zeros(len(lens), C, kernel) if init_conv is None else init_conv
    init_rec = torch.zeros(len(lens), H, K, V) if init_rec is None else init_rec
    m_by_rank = {}
    ext_by_rank = {}

    for r in range(world):
        plan = build_contiguous_cp_plan(cu, r, world)
        rows = contiguous_rank_indices(r, world, T)
        qkv_local = qkv[rows]
        beta_local = beta[0][rows].unsqueeze(0)
        g_local = g[0][rows].unsqueeze(0)

        # 1. 短卷积左边界：第一条本地 seq 从上一个 rank 续过来时，用它的最后几个
        #    token 填窗口（和 model 里 kernel > 1 的 all_gather 分支一致）
        conv_state_local = init_conv[plan.global_seq_indices].clone()
        if kernel > 1 and plan.continues_from_prev_rank:
            valid = min(kernel - 1, plan.pre_conv_tokens)
            if valid > 0:
                prev_rows = contiguous_rank_indices(r - 1, world, T)
                window = conv_state_local[0]
                window[:, -valid:] = qkv[prev_rows[-valid:]].transpose(0, 1)

        local_cu = _int32(plan.local_cu_seqlens)
        conv_out, new_conv_state = causal_conv1d_prefill(
            qkv_local, conv_state_local, weight, local_cu, impl="ref"
        )
        q, k, v = _split_qkv(conv_out, H, K, V)

        # 2. 导出本 rank 最后一条本地 seq 的段仿射映射（跨到下一个 rank 的就是它）
        m_all, ext_all = segment_affine_transition(
            chunk_kimi_delta_attention,
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            g_local,
            beta_local,
            local_cu,
            plan.local_lens,
            use_qk_l2norm_in_kernel=True,
            impl="torch",
        )
        m_by_rank[r] = m_all[-1].detach()
        ext_by_rank[r] = ext_all[-1].detach()

        # 3. 跨 rank 前缀合并
        initial_state = init_rec[plan.global_seq_indices].clone()
        if plan.continues_from_prev_rank and plan.chain_len[0] > 0:
            chain = plan.chain_len[0]
            first_slot = r - chain
            initial_state[0] = merge_prefix_states(
                torch.stack([m_by_rank[j] for j in range(first_slot, r)]),
                torch.stack([ext_by_rank[j] for j in range(first_slot, r)]),
                initial_state[0],
            )

        out_local, last_local = chunk_kimi_delta_attention(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            g=g_local,
            beta=beta_local,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=local_cu,
            seq_len_list=plan.local_lens,
            impl="torch",
        )
        out_global[rows] = out_local[0]

        # 4. 每条 seq 的 final state 由「持有它最后一个 token」的 rank 贡献
        local_pos = {gidx: i for i, gidx in enumerate(plan.global_seq_indices)}
        for seq_idx in range(len(lens)):
            if lens[seq_idx] <= 0:
                continue
            # 只有持有这条 seq 最后一个 token 的 rank 才贡献它的全局 final state
            last_pos = cu[seq_idx + 1] - 1
            if not (plan.interval_start <= last_pos < plan.interval_end):
                continue
            li = local_pos[seq_idx]
            conv_full[seq_idx] = new_conv_state[li]
            rec_full[seq_idx] = last_local[li]

    return out_global, rec_full, conv_full


@pytest.mark.parametrize(
    "T,lens,seed",
    [
        (64, [20, 30, 14], 0),
        (64, [16, 16, 16, 16], 3),
        (65, [20, 30, 15], 7),
    ],
    ids=["crossing", "one-seq-per-rank", "uneven-split"],
)
def test_contiguous_layer_matches_replicate(T, lens, seed):
    H, K, V, kernel, world = 2, 8, 6, 4, 4
    cu = [0] + list(torch.tensor(lens).cumsum(0).tolist())
    assert cu[-1] == T
    qkv, weight, beta, g = _setup(T=T, H=H, K=K, V=V, kernel=kernel, seed=seed)

    ref_out, ref_rec, ref_conv = _reference(qkv, weight, beta, g, cu, lens, H, K, V)
    out, rec, conv = _contiguous(qkv, weight, beta, g, cu, lens, H, K, V, world)

    torch.testing.assert_close(out, ref_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(rec, ref_rec, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(conv, ref_conv, atol=1e-4, rtol=1e-4)


def test_contiguous_layer_chunked_prefill_state_carry():
    """chunked prefill：把 128 token 切成 2 个 64-token 的 step，第 2 个 step 从第 1
    个 step 留下的 conv/recurrent state 续算。

    这正是 FLA 的 CP 做不到的（它断言 ``initial_state is None``），而连续切分路径
    每个 step 都从 cache 读 state、算完再写回，所以跨 chunk 续算与「整段重算」等价。
    """
    T, H, K, V, kernel, world = 128, 2, 8, 6, 4, 4
    chunk = 64
    qkv, weight, beta, g = _setup(T=T, H=H, K=K, V=V, kernel=kernel, seed=11)

    out_ref, out_con = [], []
    conv_r = rec_r = conv_c = rec_c = None
    for lo in range(0, T, chunk):
        sl = slice(lo, lo + chunk)
        cq, cb, cg = qkv[sl], beta[:, sl], g[:, sl]
        lens, cu = [chunk], [0, chunk]

        r_out, r_rec, r_conv = _reference(
            cq,
            weight,
            cb,
            cg,
            cu,
            lens,
            H,
            K,
            V,
            init_conv=conv_r,
            init_rec=rec_r,
        )
        c_out, c_rec, c_conv = _contiguous(
            cq,
            weight,
            cb,
            cg,
            cu,
            lens,
            H,
            K,
            V,
            world,
            init_conv=conv_c,
            init_rec=rec_c,
        )
        out_ref.append(r_out)
        out_con.append(c_out)
        conv_r, rec_r = r_conv, r_rec
        conv_c, rec_c = c_conv, c_rec

    torch.testing.assert_close(
        torch.cat(out_con), torch.cat(out_ref), atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(rec_c, rec_r, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(conv_c, conv_r, atol=1e-4, rtol=1e-4)
