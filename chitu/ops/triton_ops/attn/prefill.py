# SPDX-FileCopyrightText: 2023-2024 SGLang Team
# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    QO_Start_Loc,
    KV_Start_Loc,
    QO_Seqlen,
    KV_Seqlen,
    Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_qo_seq_len = tl.load(QO_Seqlen + cur_batch)
    cur_batch_kv_seq_len = tl.load(KV_Seqlen + cur_batch)
    cur_batch_qo_offset = cur_batch_kv_seq_len - cur_batch_qo_seq_len
    cur_batch_in_all_qo_start_index = tl.load(QO_Start_Loc + cur_batch)
    cur_batch_in_all_kv_start_index = tl.load(KV_Start_Loc + cur_batch)
    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_newv = tl.arange(0, BLOCK_DMODEL_V)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_qo_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_newv[None, :]

    mask_d = offs_d < Lk
    mask_for_load_v = offs_newv < Lv

    q = tl.load(
        Q + off_q,
        mask=(offs_m[:, None] < cur_batch_qo_seq_len) & (mask_d[None, :]),
        other=0.0,
    )

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, Lv], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_qo_seq_len, 1, 0)

    end_n = (
        cur_batch_kv_seq_len
        if not IS_CAUSAL
        else tl.minimum(
            (start_m + 1) * BLOCK_M + cur_batch_qo_offset,
            cur_batch_kv_seq_len,
        )
    )
    for start_n in range(0, block_mask * end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch_in_all_kv_start_index + start_n) * stride_kbs,
            mask=((start_n + offs_n[None, :]) < cur_batch_kv_seq_len)
            & (mask_d[:, None]),
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] + cur_batch_qo_offset >= (
                start_n + offs_n[None, :]
            )
            blk_causal_mask = offs_m + cur_batch_qo_offset >= start_n
            qk += tl.where(
                (start_n + offs_n[None, :] < cur_batch_kv_seq_len) & causal_mask,
                0,
                float("-inf"),
            )
        else:
            qk += tl.where(
                (start_n + offs_n[None, :]) < cur_batch_kv_seq_len, 0, float("-inf")
            )

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        if IS_CAUSAL:
            acc_scale = tl.where(blk_causal_mask, acc_scale, 1)
        acc = acc * acc_scale[:, None]

        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_kv_start_index + start_n) * stride_vbs,
            mask=((start_n + offs_n[:, None]) < cur_batch_kv_seq_len)
            & (mask_for_load_v[None, :]),
            other=0.0,
        )
        if IS_CAUSAL:
            p = tl.where(blk_causal_mask[:, None], p, 0)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)

        # update m_i and l_i
        if IS_CAUSAL:
            m_i = tl.where(blk_causal_mask, m_i_new, m_i)
            l_i = tl.where(blk_causal_mask, l_i_new, l_i)
        else:
            l_i = l_i_new
            m_i = m_i_new

    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_qo_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_newv[None, :]
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs,
        acc,
        mask=(offs_m[:, None] < cur_batch_qo_seq_len) & (mask_for_load_v[None, :]),
    )


def prefill_ragged_qkvo_triton(
    q,
    k,
    v,
    o,
    qo_start_loc,
    kv_start_loc,
    qo_seq_len,
    kv_seq_len,
    max_qo_seq_len,
    softmax_scale,
    is_causal=True,
):
    """
    q: [b * s_q, head, head_dim_qk]
    k, v: [b * s_kv, head, head_dim_qk]
    qo_start_loc, kv_start_loc: [b]
    qo_seq_len, kv_start_loc: [b]
    out: [b * s_q, head, head_dim_v]
    """
    BLOCK = 16

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]

    if softmax_scale is None:
        sm_scale = 1.0 / (Lq**0.5)
    else:
        sm_scale = softmax_scale

    batch = qo_seq_len.shape[0]
    head = q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = (batch, head, triton.cdiv(max_qo_seq_len, BLOCK))
    num_warps = 4  # if Lk <= 64 else

    assert q.stride(-1) == 1
    assert k.stride(-1) == 1
    assert v.stride(-1) == 1
    assert o.stride(-1) == 1

    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        qo_start_loc,
        kv_start_loc,
        qo_seq_len,
        kv_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_DMODEL_V=triton.next_power_of_2(Lv),
        BLOCK_N=BLOCK,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=1,
        Lk=Lk,
        Lv=Lv,
    )
