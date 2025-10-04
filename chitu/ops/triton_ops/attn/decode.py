# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import math
import os

import torch
import triton
import triton.language as tl

from chitu.device_type import is_muxi

# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2023-2024 SGLang Team
# SDPX—SnippetName: The triton implementation of _fwd_kernel
#
# The triton implementation of _fwd_kernel is originally from SGLang
# (https://github.com/sgl-project/sglang/commit/df191254abc002b3284560d9c4b94214a4656265),


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = cur_batch

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + off_q, mask=mask_d, other=0.0)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(
                Req_to_tokens
                + stride_req_to_tokens_b * cur_batch_req_idx
                + offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
            )
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
            offs_buf_k = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


def _decode_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    Req_to_tokens,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap,
):
    BLOCK = 64
    NUM_KV_SPLITS = num_kv_splits
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, NUM_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    num_warps = 4 if kv_group_num == 1 else 2

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        Req_to_tokens,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )


_muxi_triton_supports_scenario = None


def _check_muxi_triton_scenario_support():
    global _muxi_triton_supports_scenario
    if _muxi_triton_supports_scenario is None:
        config_signature = inspect.signature(triton.Config.__init__)
        config_params = list(config_signature.parameters.keys())[1:]
        _muxi_triton_supports_scenario = "scenario" in config_params
    return _muxi_triton_supports_scenario


def _create_triton_config(block_n, num_stages, num_warps):
    if is_muxi() and _check_muxi_triton_scenario_support():
        return triton.Config(
            {"BLOCK_N": block_n},
            num_stages=num_stages,
            num_warps=num_warps,
            scenario="flashattn-fwd",
        )
    else:
        return triton.Config(
            {"BLOCK_N": block_n},
            num_stages=num_stages,
            num_warps=num_warps,
        )


_fwd_grouped_kernel_stage1_configs = [
    _create_triton_config(block_n, num_stages, num_warps)
    for block_n in [16, 32, 64, 128]
    for num_stages in [1, 2, 3, 4]
    for num_warps in [2, 4, 8, 16]
]

if os.environ.get("CI_TESTS", "false") == "true":
    _fwd_grouped_kernel_stage1_configs = [
        _create_triton_config(block_n, num_stages, num_warps)
        for block_n in [32]
        for num_stages in [1, 2, 3, 4]
        for num_warps in [2, 4, 8, 16]
    ]


@triton.autotune(configs=_fwd_grouped_kernel_stage1_configs, key=[])
@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = cur_batch

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )
        qpe = tl.load(
            Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
        )

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(
                Req_to_tokens
                + stride_req_to_tokens_b * cur_batch_req_idx
                + offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
            )
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
            offs_buf_k = (
                kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = (
                    kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    Req_to_tokens,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap,
):
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    BLOCK_H = 16
    NUM_KV_SPLITS = num_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_KV_SPLITS,
    )

    extra_kargs = {}
    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        Req_to_tokens,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        Lk=Lk,
        Lv=Lv,
        **extra_kargs,
    )


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    o,
    B_Seqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def _decode_softmax_reducev_fwd(
    logits,
    q,
    o,
    v_buffer,
    b_seq_len,
    num_kv_splits,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    NUM_KV_SPLITS = num_kv_splits

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        o,
        b_seq_len,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
    )


def decode_attention_fwd_normal(
    q,
    k_buffer,
    v_buffer,
    o,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap=0.0,
):
    _decode_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        req_to_token,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        page_size,
        logit_cap,
    )
    _decode_softmax_reducev_fwd(attn_logits, q, o, v_buffer, b_seq_len, num_kv_splits)


def decode_attention_fwd_grouped(
    q,
    k_buffer,
    v_buffer,
    o,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap=0.0,
):
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        req_to_token,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        page_size,
        logit_cap,
    )
    _decode_softmax_reducev_fwd(attn_logits, q, o, v_buffer, b_seq_len, num_kv_splits)


def decode_paged_kv_triton(
    q,
    k_buffer,
    v_buffer,
    o,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size=1,
    logit_cap=0.0,
):
    assert num_kv_splits == attn_logits.shape[2]
    kv_group_num = q.shape[1] // v_buffer.shape[-2]

    if kv_group_num == 1:
        # MHA
        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            req_to_token,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
            page_size,
            logit_cap,
        )
    else:
        # GQA/MQA/MLA
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            req_to_token,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
            page_size,
            logit_cap,
        )


# SPDX-SnippetEnd


block_m_sizes = [32] if is_muxi() else [32, 64]
triton_skew_decode_configs = [
    triton.Config({"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN})
    for BM in block_m_sizes
    for BN in [32]
]


@triton.autotune(configs=triton_skew_decode_configs, key=["num_heads_q", "head_dim"])
@triton.jit
def triton_skew_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    softmax_scale,
    softcap: tl.constexpr,
    attn_bias,
    dropout_p,
    local_mask,
    query_padding_mask,
    key_padding_mask,
    dropout_mask,
    stride_q_b,
    stride_q_t,
    stride_q_h,
    stride_q_d,  # q.shape=bthd
    stride_k_b,
    stride_k_s,
    stride_k_h,
    stride_k_d,  # k.shape=bshd
    stride_v_b,
    stride_v_s,
    stride_v_h,
    stride_v_d,  # v.shape=bshd
    stride_o_b,
    stride_o_t,
    stride_o_h,
    stride_o_d,  # o.shape=bthd
    batch_size: tl.constexpr,
    seq_len_q: tl.constexpr,
    seq_len_kv: tl.constexpr,
    num_heads_q: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_bh = tl.program_id(axis=0)
    pid_t = tl.program_id(axis=1)

    b = pid_bh // num_heads_q
    h = pid_bh % num_heads_q

    t_range = pid_t * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    mask_t = t_range < seq_len_q

    q_offsets = (
        b * stride_q_b
        + h * stride_q_h
        + t_range[:, None] * stride_q_t
        + tl.arange(0, head_dim)[None, :] * stride_q_d
    )
    q = tl.load(q_ptr + q_offsets, mask=mask_t[:, None], other=0.0)
    acc = tl.zeros([BLOCK_SIZE_M, head_dim], dtype=tl.float32)
    last_max = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")
    last_sum = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)

    if query_padding_mask is not None:
        query_mask_offset = b * seq_len_q + t_range[:, None]
        query_mask = tl.load(
            query_padding_mask + query_mask_offset, mask=mask_t[:, None], other=0.0
        )

    for i in range(0, seq_len_kv, BLOCK_SIZE_N):
        s_range = i + tl.arange(0, BLOCK_SIZE_N)
        mask_s = s_range < seq_len_kv

        k_offsets = (
            b * stride_k_b
            + h // group_size * stride_k_h
            + s_range[:, None] * stride_k_s
            + tl.arange(0, head_dim)[None, :] * stride_k_d
        )
        k = tl.load(k_ptr + k_offsets, mask=mask_s[:, None], other=0.0)
        k = tl.trans(k)
        block_scores = tl.dot(q, k)
        block_scores = block_scores * softmax_scale

        if softcap > 0:
            block_scores = block_scores / softcap
            block_scores_clamped = tl.minimum(tl.maximum(block_scores, -4.97), 4.97)
            exp_block_scores = tl.exp(2 * block_scores_clamped)
            tanh_val = (exp_block_scores - 1) / (exp_block_scores + 1)
            block_scores = tanh_val * softcap

        if key_padding_mask is not None:
            key_mask = tl.load(
                key_padding_mask + b * seq_len_kv + s_range, mask=mask_s, other=0
            )
            block_scores = tl.where(~key_mask[None, :], float("-inf"), block_scores)

        if local_mask is not None:
            mask = tl.load(local_mask + b * seq_len_kv + s_range, mask=mask_s, other=0)
            block_scores = tl.where(mask[None, :], float("-inf"), block_scores)

        if attn_bias is not None:
            attn_bias_offsets = (
                b * num_heads_q * seq_len_q * seq_len_kv
                + h * seq_len_q * seq_len_kv
                + t_range[:, None] * seq_len_kv
                + s_range
            )
            attn_bias_mask = mask_t[:, None] & mask_s[None, :]
            attn_bias_val = tl.load(
                attn_bias + attn_bias_offsets, mask=attn_bias_mask, other=0.0
            )
            block_scores += attn_bias_val

        new_max = tl.maximum(last_max, tl.max(block_scores, 1))
        block_scores -= new_max[:, None]
        p = tl.math.exp(block_scores)

        if local_mask is not None:
            full_local_mask_row_offset = b * seq_len_kv + tl.arange(0, seq_len_kv)
            mask = tl.load(local_mask + full_local_mask_row_offset)
            all_true = tl.min(mask) == 1
            p = tl.where(all_true[None, :], 0.0, p)

        new_sum = tl.sum(p, 1)
        alpha = tl.math.exp(last_max - new_max)
        last_sum = last_sum * alpha + new_sum
        acc = acc * alpha[:, None]
        if query_padding_mask is not None:
            p = tl.where(~query_mask[None, :], 0.0, p)
        dropout_scaling = 1.0 / (1 - dropout_p)
        if dropout_mask is not None:  # assume shape is bhts
            dropout_mask_offset = (
                b * num_heads_q * seq_len_q * seq_len_kv
                + h * seq_len_q * seq_len_kv
                + t_range[:, None] * seq_len_kv
                + s_range[None, :]
            )
            mask = tl.load(
                dropout_mask + dropout_mask_offset,
                mask=mask_t[None, :] & mask_s[:, None],
                other=0,
            )
            p = tl.where(~mask[None, :], 0.0, p)
        v_offsets = (
            b * stride_v_b
            + h // group_size * stride_v_h
            + s_range[:, None] * stride_v_s
            + tl.arange(0, head_dim)[None, :] * stride_v_d
        )
        v = tl.load(v_ptr + v_offsets, mask=mask_s[:, None], other=0.0)
        v = v * dropout_scaling
        p = p.to(q_ptr.type.element_ty)
        v = v.to(q_ptr.type.element_ty)
        acc_i = tl.dot(p, v)
        if query_padding_mask is not None:
            acc_i = tl.where(~query_mask[None, :], 0.0, acc_i)
        acc = acc + acc_i
        last_max = new_max
    acc = acc / last_sum[:, None]

    o_offsets = (
        b * stride_o_b
        + h * stride_o_h
        + t_range[:, None] * stride_o_t
        + tl.arange(0, head_dim)[None, :] * stride_o_d
    )
    tl.store(o_ptr + o_offsets, acc.to(o_ptr.type.element_ty), mask=mask_t[:, None])


def decode_dense_kv_triton(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=False,
    reorder_ops=False,
    key_leftpad=None,
    softmax_scale=None,
    local_mask=None,
):
    if causal:
        window_size = (window_size[0], 0)
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    group_size = q.shape[2] // k.shape[2]
    d = q.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    o = torch.empty_like(q)
    if attn_bias is not None:
        assert attn_bias.shape[-1] == k.shape[1] and attn_bias.shape[-2] == q.shape[1]
        attn_bias = torch.broadcast_to(
            attn_bias, (q.shape[0], q.shape[2], q.shape[1], k.shape[1])
        )  # bhts
        if attn_bias.is_contiguous() == False:
            attn_bias = attn_bias.contiguous()
    grid = lambda META: (
        q.shape[0] * q.shape[2],
        triton.cdiv(q.shape[1], META["BLOCK_SIZE_M"]),
        1,
    )
    triton_skew_decode_kernel[grid](
        q,
        k,
        v,
        o,
        softmax_scale,
        softcap,
        attn_bias,
        dropout_p,
        local_mask,
        query_padding_mask,
        key_padding_mask,
        dropout_mask,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        q.shape[0],
        q.shape[1],
        k.shape[1],
        q.shape[2],
        q.shape[3],
        group_size,
    )

    return o
