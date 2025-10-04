# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 vLLM Team
# SPDX-SnippetCopyrightText: monellz
# SDPX—SnippetName: The triton implementation of MLA
#
# The triton implementation of MLA is credited to monellz, which is a fork of vLLM
# (https://github.com/monellz/vllm/commit/feebaa7c063be6bfb590a876741aeef1c5f58cf8)

import torch
import triton
import triton.language as tl


@triton.jit
def triton_or(x, y):
    return x | y


_mla_attn_kernel_configs = [
    triton.Config(
        {"BLOCK_N": block_n},
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_n in [16, 32, 64, 128]
    for num_stages in [1, 2, 3, 4]
    for num_warps in [2, 4, 8, 16]
]


@triton.autotune(configs=_mla_attn_kernel_configs, key=[])
@triton.jit
def _mla_attn_kernel(
    Q_nope,
    Q_pe,
    Kv_c_cache,
    K_pe_cache,
    Req_to_tokens,
    B_seq_len,
    O,
    Topk_indices,
    sm_scale: tl.constexpr,
    stride_q_nope_bs: tl.constexpr,
    stride_q_nope_h: tl.constexpr,
    stride_q_pe_bs: tl.constexpr,
    stride_q_pe_h: tl.constexpr,
    stride_kv_c_bs: tl.constexpr,
    stride_k_pe_bs: tl.constexpr,
    stride_req_to_tokens_bs: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_o_s: tl.constexpr,
    stride_topk_indices_bs: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
    INDEX_TOPK: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_batch_seq_len = tl.load(B_seq_len + cur_batch)
    if INDEX_TOPK is not None:
        cur_batch_seq_len_maybe_topk = INDEX_TOPK
    else:
        cur_batch_seq_len_maybe_topk = cur_batch_seq_len

    offs_d_ckv = tl.arange(0, HEAD_DIM_CKV)
    cur_head = cur_head_id * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_q_nope = (
        cur_batch * stride_q_nope_bs
        + cur_head[:, None] * stride_q_nope_h
        + offs_d_ckv[None, :]
    )
    q_nope = tl.load(Q_nope + offs_q_nope)

    offs_d_kpe = tl.arange(0, HEAD_DIM_KPE)
    offs_q_pe = (
        cur_batch * stride_q_pe_bs
        + cur_head[:, None] * stride_q_pe_h
        + offs_d_kpe[None, :]
    )
    q_pe = tl.load(Q_pe + offs_q_pe)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM_CKV], dtype=tl.float32)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len_maybe_topk, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(
        split_kv_start + kv_len_per_split, cur_batch_seq_len_maybe_topk
    )

    any_block_valid = False

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            if INDEX_TOPK is not None:
                offs_n_maybe_topk = tl.load(
                    Topk_indices + cur_batch * stride_topk_indices_bs + offs_n,
                    mask=offs_n < split_kv_end,
                )
                offs_n_mask_maybe_topk = (offs_n < split_kv_end) & (
                    offs_n_maybe_topk < cur_batch_seq_len
                )
                block_valid = tl.reduce(
                    offs_n_mask_maybe_topk, axis=None, combine_fn=triton_or
                )
            else:
                offs_n_maybe_topk = offs_n
                offs_n_mask_maybe_topk = offs_n < split_kv_end
                block_valid = True

            if block_valid:
                any_block_valid = True

                kv_page_number = tl.load(
                    Req_to_tokens
                    + stride_req_to_tokens_bs * cur_batch
                    + offs_n_maybe_topk // PAGE_SIZE,
                    mask=offs_n_mask_maybe_topk,
                    other=0,
                )
                kv_loc = kv_page_number * PAGE_SIZE + offs_n_maybe_topk % PAGE_SIZE
                offs_k_c = kv_loc[None, :] * stride_kv_c_bs + offs_d_ckv[:, None]
                k_c = tl.load(
                    Kv_c_cache + offs_k_c,
                    mask=offs_n_mask_maybe_topk[None, :],
                    other=0.0,
                )

                qk = tl.dot(q_nope, k_c.to(q_nope.dtype))

                offs_k_pe = kv_loc[None, :] * stride_k_pe_bs + offs_d_kpe[:, None]
                k_pe = tl.load(
                    K_pe_cache + offs_k_pe,
                    mask=offs_n_mask_maybe_topk[None, :],
                    other=0.0,
                )

                qk += tl.dot(q_pe, k_pe.to(q_pe.dtype))
                qk *= sm_scale

                qk = tl.where(offs_n_mask_maybe_topk[None, :], qk, float("-inf"))

                v_c = tl.trans(k_c)

                n_e_max = tl.maximum(tl.max(qk, 1), e_max)
                re_scale = tl.exp(e_max - n_e_max)
                p = tl.exp(qk - n_e_max[:, None])
                acc *= re_scale[:, None]
                acc += tl.dot(p.to(v_c.dtype), v_c)

                e_sum = e_sum * re_scale + tl.sum(p, 1)
                e_max = n_e_max

    if any_block_valid:
        offs_o = (
            cur_batch * stride_o_b
            + cur_head[:, None] * stride_o_h
            + split_kv_id * stride_o_s
            + offs_d_ckv[None, :]
        )
        tl.store(O + offs_o, acc / e_sum[:, None])
        offs_o_1 = (
            cur_batch * stride_o_b
            + cur_head * stride_o_h
            + split_kv_id * stride_o_s
            + HEAD_DIM_CKV
        )
        tl.store(O + offs_o_1, e_max + tl.log(e_sum))


def _mla_attn(
    q_nope,
    q_pe,
    kv_c_cache,
    k_pe_cache,
    attn_logits,
    req_to_tokens,
    b_seq_len,
    num_kv_splits,
    sm_scale,
    page_size,
    topk_indices,
):
    batch_size, head_num = q_nope.shape[0], q_nope.shape[1]
    head_dim_ckv = q_nope.shape[-1]
    head_dim_kpe = q_pe.shape[-1]

    BLOCK_H = 16
    grid = (
        batch_size,
        triton.cdiv(head_num, BLOCK_H),
        num_kv_splits,
    )

    _mla_attn_kernel[grid](
        q_nope,
        q_pe,
        kv_c_cache,
        k_pe_cache,
        req_to_tokens,
        b_seq_len,
        attn_logits,
        topk_indices,
        sm_scale,
        q_nope.stride(0),
        q_nope.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        kv_c_cache.stride(-2),
        k_pe_cache.stride(-2),
        req_to_tokens.stride(0),
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        topk_indices.stride(0) if topk_indices is not None else 0,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=num_kv_splits,
        PAGE_SIZE=page_size,
        HEAD_DIM_CKV=head_dim_ckv,
        HEAD_DIM_KPE=head_dim_kpe,
        INDEX_TOPK=topk_indices.shape[-1] if topk_indices is not None else None,
    )


@triton.autotune(configs=_mla_attn_kernel_configs, key=[])
@triton.jit
def _mla_attn_non_paged_kernel(
    Q_nope,
    Q_pe,
    Kv_c_cache,
    K_pe_cache,
    B_seq_len,
    O,
    Topk_indices,
    sm_scale: tl.constexpr,
    stride_q_nope_bs: tl.constexpr,
    stride_q_nope_h: tl.constexpr,
    stride_q_pe_bs: tl.constexpr,
    stride_q_pe_h: tl.constexpr,
    stride_kv_c_bs: tl.constexpr,
    stride_k_pe_bs: tl.constexpr,
    stride_kv_c_s: tl.constexpr,
    stride_k_pe_s: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_o_s: tl.constexpr,
    stride_topk_indices_bs: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
    INDEX_TOPK: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_batch_seq_len = tl.load(B_seq_len + cur_batch)
    if INDEX_TOPK is not None:
        cur_batch_seq_len_maybe_topk = INDEX_TOPK
    else:
        cur_batch_seq_len_maybe_topk = cur_batch_seq_len

    offs_d_ckv = tl.arange(0, HEAD_DIM_CKV)
    cur_head = cur_head_id * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_q_nope = (
        cur_batch * stride_q_nope_bs
        + cur_head[:, None] * stride_q_nope_h
        + offs_d_ckv[None, :]
    )
    q_nope = tl.load(Q_nope + offs_q_nope)

    offs_d_kpe = tl.arange(0, HEAD_DIM_KPE)
    offs_q_pe = (
        cur_batch * stride_q_pe_bs
        + cur_head[:, None] * stride_q_pe_h
        + offs_d_kpe[None, :]
    )
    q_pe = tl.load(Q_pe + offs_q_pe)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM_CKV], dtype=tl.float32)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len_maybe_topk, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(
        split_kv_start + kv_len_per_split, cur_batch_seq_len_maybe_topk
    )

    any_block_valid = False

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            if INDEX_TOPK is not None:
                offs_n_maybe_topk = tl.load(
                    Topk_indices + cur_batch * stride_topk_indices_bs + offs_n,
                    mask=offs_n < split_kv_end,
                )
                offs_n_mask_maybe_topk = (offs_n < split_kv_end) & (
                    offs_n_maybe_topk < cur_batch_seq_len
                )
                block_valid = tl.reduce(
                    offs_n_mask_maybe_topk, axis=None, combine_fn=triton_or
                )
            else:
                offs_n_maybe_topk = offs_n
                offs_n_mask_maybe_topk = offs_n < split_kv_end
                block_valid = True

            if block_valid:
                any_block_valid = True

                offs_k_c = (
                    cur_batch * stride_kv_c_bs
                    + offs_n_maybe_topk[None, :] * stride_kv_c_s
                    + offs_d_ckv[:, None]
                )
                k_c = tl.load(
                    Kv_c_cache + offs_k_c,
                    mask=offs_n_mask_maybe_topk[None, :],
                    other=0.0,
                )

                qk = tl.dot(q_nope, k_c.to(q_nope.dtype))

                offs_k_pe = (
                    cur_batch * stride_k_pe_bs
                    + offs_n_maybe_topk[None, :] * stride_k_pe_s
                    + offs_d_kpe[:, None]
                )
                k_pe = tl.load(
                    K_pe_cache + offs_k_pe,
                    mask=offs_n_mask_maybe_topk[None, :],
                    other=0.0,
                )

                qk += tl.dot(q_pe, k_pe.to(q_pe.dtype))
                qk *= sm_scale

                qk = tl.where(offs_n_mask_maybe_topk[None, :], qk, float("-inf"))

                v_c = tl.trans(k_c)

                n_e_max = tl.maximum(tl.max(qk, 1), e_max)
                re_scale = tl.exp(e_max - n_e_max)
                p = tl.exp(qk - n_e_max[:, None])
                acc *= re_scale[:, None]
                acc += tl.dot(p.to(v_c.dtype), v_c)

                e_sum = e_sum * re_scale + tl.sum(p, 1)
                e_max = n_e_max

    if any_block_valid:
        offs_o = (
            cur_batch * stride_o_b
            + cur_head[:, None] * stride_o_h
            + split_kv_id * stride_o_s
            + offs_d_ckv[None, :]
        )
        tl.store(O + offs_o, acc / e_sum[:, None])
        offs_o_1 = (
            cur_batch * stride_o_b
            + cur_head * stride_o_h
            + split_kv_id * stride_o_s
            + HEAD_DIM_CKV
        )
        tl.store(O + offs_o_1, e_max + tl.log(e_sum))


def _mla_attn_non_paged(
    q_nope,
    q_pe,
    kv_c_cache,
    k_pe_cache,
    attn_logits,
    b_seq_len,
    num_kv_splits,
    sm_scale,
    topk_indices,
):
    batch_size, head_num = q_nope.shape[0], q_nope.shape[1]
    head_dim_ckv = q_nope.shape[-1]
    head_dim_kpe = q_pe.shape[-1]

    BLOCK_H = 16
    grid = (
        batch_size,
        triton.cdiv(head_num, BLOCK_H),
        num_kv_splits,
    )

    _mla_attn_non_paged_kernel[grid](
        q_nope,
        q_pe,
        kv_c_cache,
        k_pe_cache,
        b_seq_len,
        attn_logits,
        topk_indices,
        sm_scale,
        q_nope.stride(0),
        q_nope.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        kv_c_cache.stride(0),
        k_pe_cache.stride(0),
        kv_c_cache.stride(1),
        k_pe_cache.stride(1),
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        topk_indices.stride(0) if topk_indices is not None else 0,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=num_kv_splits,
        HEAD_DIM_CKV=head_dim_ckv,
        HEAD_DIM_KPE=head_dim_kpe,
        INDEX_TOPK=topk_indices.shape[-1] if topk_indices is not None else None,
    )


@triton.jit
def _mla_softmax_reducev_kernel(
    Logits,
    B_seq_len,
    B_last_pos_id,
    O,
    Topk_indices,
    stride_l_b,
    stride_l_h,
    stride_l_s,
    stride_o_b,
    stride_o_h,
    stride_topk_indices_bs: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    INDEX_TOPK: tl.constexpr,
    HAS_B_SEQ_LEN: tl.constexpr,
    HAS_B_LAST_POS_ID: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    if HAS_B_SEQ_LEN:
        cur_batch_seq_len = tl.load(B_seq_len + cur_batch)
    elif HAS_B_LAST_POS_ID:
        cur_batch_seq_len = tl.load(B_last_pos_id + cur_batch) + 1
    else:
        tl.static_assert(False)
    if INDEX_TOPK is not None:
        cur_batch_seq_len_maybe_topk = INDEX_TOPK
    else:
        cur_batch_seq_len_maybe_topk = cur_batch_seq_len

    offs_d_ckv = tl.arange(0, HEAD_DIM_CKV)

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([HEAD_DIM_CKV], dtype=tl.float32)

    offs_l = cur_batch * stride_l_b + cur_head * stride_l_h + offs_d_ckv
    offs_l_1 = cur_batch * stride_l_b + cur_head * stride_l_h + HEAD_DIM_CKV

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len_maybe_topk, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(
            split_kv_start + kv_len_per_split, cur_batch_seq_len_maybe_topk
        )

        any_block_valid = split_kv_end > split_kv_start
        if INDEX_TOPK is not None:
            if any_block_valid:
                offs_n = tl.arange(0, triton.next_power_of_2(INDEX_TOPK))
                offs_n_mask = (offs_n >= split_kv_start) & (offs_n < split_kv_end)
                offs_n_maybe_topk = tl.load(
                    Topk_indices + cur_batch * stride_topk_indices_bs + offs_n,
                    mask=offs_n_mask,
                )
                any_block_valid = tl.reduce(
                    offs_n_mask & (offs_n_maybe_topk < cur_batch_seq_len),
                    axis=None,
                    combine_fn=triton_or,
                )

        if any_block_valid:
            logits = tl.load(Logits + offs_l + split_kv_id * stride_l_s)
            logits_1 = tl.load(Logits + offs_l_1 + split_kv_id * stride_l_s)

            n_e_max = tl.maximum(logits_1, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(logits_1 - n_e_max)
            acc += exp_logic * logits

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        O + cur_batch * stride_o_b + cur_head * stride_o_h + offs_d_ckv,
        acc / e_sum,
    )


def _mla_softmax_reducev(
    logits,
    o,
    b_seq_len,
    b_last_pos_id,
    num_kv_splits,
    topk_indices,
):
    batch_size, head_num, head_dim_ckv = o.shape[0], o.shape[1], o.shape[2]
    grid = (batch_size, head_num)
    _mla_softmax_reducev_kernel[grid](
        logits,
        b_seq_len,
        b_last_pos_id,
        o,
        topk_indices,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        topk_indices.stride(0) if topk_indices is not None else 0,
        NUM_KV_SPLITS=num_kv_splits,
        HEAD_DIM_CKV=head_dim_ckv,
        INDEX_TOPK=topk_indices.shape[-1] if topk_indices is not None else None,
        HAS_B_SEQ_LEN=b_seq_len is not None,
        HAS_B_LAST_POS_ID=b_last_pos_id is not None,
        num_warps=4,
        num_stages=2,
    )


def mla_decode_paged_kv_triton(
    q_nope,
    q_pe,
    kv_c_cache,
    k_pe_cache,
    o,
    req_to_tokens,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
    topk_indices,
):
    assert num_kv_splits == attn_logits.shape[2]
    _mla_attn(
        q_nope,
        q_pe,
        kv_c_cache,
        k_pe_cache,
        attn_logits,
        req_to_tokens,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        page_size,
        topk_indices,
    )
    _mla_softmax_reducev(
        attn_logits,
        o,
        b_seq_len,
        None,
        num_kv_splits,
        topk_indices,
    )


def mla_decode_dense_kv_triton(
    q_nope,
    q_pe,
    kv_c_cache,
    k_pe_cache,
    o,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    topk_indices,
):
    assert num_kv_splits == attn_logits.shape[2]
    _mla_attn_non_paged(
        q_nope,
        q_pe,
        kv_c_cache,
        k_pe_cache,
        attn_logits,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        topk_indices,
    )
    _mla_softmax_reducev(
        attn_logits,
        o,
        b_seq_len,
        None,
        num_kv_splits,
        topk_indices,
    )


# SPDX-SnippetEnd


@triton.autotune(configs=_mla_attn_kernel_configs, key=[])
@triton.jit
def _mla_decode_topk_ragged_qkvo_kernel(
    Q_nope,
    Q_pe,
    Kv_c_cache,
    K_pe_cache,
    Delta_position_ids,
    Delta_seq_ids,
    New_prefix_lens,
    O,
    Topk_indices,
    sm_scale: tl.constexpr,
    stride_q_nope_bs: tl.constexpr,
    stride_q_nope_h: tl.constexpr,
    stride_q_pe_bs: tl.constexpr,
    stride_q_pe_h: tl.constexpr,
    stride_kv_c_bs: tl.constexpr,
    stride_k_pe_bs: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_o_s: tl.constexpr,
    stride_topk_indices_bs: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
    INDEX_TOPK: tl.constexpr,
):
    cur_token = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_token_position_id = tl.load(Delta_position_ids + cur_token)
    cur_token_seq_id = tl.load(Delta_seq_ids + cur_token)
    cur_seq_prefix_len = tl.load(New_prefix_lens + cur_token_seq_id)

    offs_d_ckv = tl.arange(0, HEAD_DIM_CKV)
    cur_head = cur_head_id * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_q_nope = (
        cur_token * stride_q_nope_bs
        + cur_head[:, None] * stride_q_nope_h
        + offs_d_ckv[None, :]
    )
    q_nope = tl.load(Q_nope + offs_q_nope)

    offs_d_kpe = tl.arange(0, HEAD_DIM_KPE)
    offs_q_pe = (
        cur_token * stride_q_pe_bs
        + cur_head[:, None] * stride_q_pe_h
        + offs_d_kpe[None, :]
    )
    q_pe = tl.load(Q_pe + offs_q_pe)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM_CKV], dtype=tl.float32)

    kv_len_per_split = tl.cdiv(INDEX_TOPK, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, INDEX_TOPK)

    any_block_valid = False

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            offs_n_topk = tl.load(
                Topk_indices + cur_token * stride_topk_indices_bs + offs_n,
                mask=offs_n < split_kv_end,
            )
            offs_n_mask_topk = (offs_n < split_kv_end) & (
                offs_n_topk <= cur_token_position_id  # causal
            )
            block_valid = tl.reduce(offs_n_mask_topk, axis=None, combine_fn=triton_or)

            if block_valid:
                any_block_valid = True

                offs_k_c = (
                    cur_seq_prefix_len + offs_n_topk[None, :]
                ) * stride_kv_c_bs + offs_d_ckv[:, None]
                k_c = tl.load(
                    Kv_c_cache + offs_k_c, mask=offs_n_mask_topk[None, :], other=0.0
                )

                qk = tl.dot(q_nope, k_c.to(q_nope.dtype))

                offs_k_pe = (
                    cur_seq_prefix_len + offs_n_topk[None, :]
                ) * stride_k_pe_bs + offs_d_kpe[:, None]
                k_pe = tl.load(
                    K_pe_cache + offs_k_pe, mask=offs_n_mask_topk[None, :], other=0.0
                )

                qk += tl.dot(q_pe, k_pe.to(q_pe.dtype))
                qk *= sm_scale

                qk = tl.where(offs_n_mask_topk[None, :], qk, float("-inf"))

                v_c = tl.trans(k_c)

                n_e_max = tl.maximum(tl.max(qk, 1), e_max)
                re_scale = tl.exp(e_max - n_e_max)
                p = tl.exp(qk - n_e_max[:, None])
                acc *= re_scale[:, None]
                acc += tl.dot(p.to(v_c.dtype), v_c)

                e_sum = e_sum * re_scale + tl.sum(p, 1)
                e_max = n_e_max

    if any_block_valid:
        offs_o = (
            cur_token * stride_o_b
            + cur_head[:, None] * stride_o_h
            + split_kv_id * stride_o_s
            + offs_d_ckv[None, :]
        )
        tl.store(O + offs_o, acc / e_sum[:, None])
        offs_o_1 = (
            cur_token * stride_o_b
            + cur_head * stride_o_h
            + split_kv_id * stride_o_s
            + HEAD_DIM_CKV
        )
        tl.store(O + offs_o_1, e_max + tl.log(e_sum))


def _mla_decode_topk_ragged_qkvo(
    q_nope,
    q_pe,
    kv_c_cache,
    k_pe_cache,
    attn_logits,
    delta_position_ids,
    delta_seq_ids,
    new_prefix_lens,
    num_kv_splits,
    sm_scale,
    topk_indices,
):
    n_tokens, head_num = q_nope.shape[0], q_nope.shape[1]
    head_dim_ckv = q_nope.shape[-1]
    head_dim_kpe = q_pe.shape[-1]

    BLOCK_H = 16
    grid = (
        n_tokens,
        triton.cdiv(head_num, BLOCK_H),
        num_kv_splits,
    )

    _mla_decode_topk_ragged_qkvo_kernel[grid](
        q_nope,
        q_pe,
        kv_c_cache,
        k_pe_cache,
        delta_position_ids,
        delta_seq_ids,
        new_prefix_lens,
        attn_logits,
        topk_indices,
        sm_scale,
        q_nope.stride(0),
        q_nope.stride(1),
        q_pe.stride(0),
        q_pe.stride(1),
        kv_c_cache.stride(0),
        k_pe_cache.stride(0),
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        topk_indices.stride(0),
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=num_kv_splits,
        HEAD_DIM_CKV=head_dim_ckv,
        HEAD_DIM_KPE=head_dim_kpe,
        INDEX_TOPK=topk_indices.shape[-1],
    )


# NOTE: When topk_indices is enabled during prefill, we can't reuse Q along its sequence
# dimensions. This makes the prefill kernel behave more like a decode kernel. Therefore,
# this `mla_decode_ragged_qkvo_triton` is actually for prefilling, although it's using
# the algorithm alike `mla_decode_paged_kv_triton` or `mla_decode_dense_kv_triton`,
# and it can also be used for decoding if the data structure is ragged QKVO.
def mla_decode_topk_ragged_qkvo_triton(
    q_nope,
    q_pe,
    kv_c_cache,
    k_pe_cache,
    o,
    delta_position_ids,
    delta_seq_ids,
    new_prefix_lens,
    attn_logits,
    num_kv_splits,
    sm_scale,
    topk_indices,
):
    assert num_kv_splits == attn_logits.shape[2]
    _mla_decode_topk_ragged_qkvo(
        q_nope,
        q_pe,
        kv_c_cache,
        k_pe_cache,
        attn_logits,
        delta_position_ids,
        delta_seq_ids,
        new_prefix_lens,
        num_kv_splits,
        sm_scale,
        topk_indices,
    )
    _mla_softmax_reducev(
        attn_logits,
        o,
        None,
        delta_position_ids,
        num_kv_splits,
        topk_indices,
    )
