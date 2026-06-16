# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Fused triton kernels for HCA and CSA compressors.

Three kernels shared by both modes:
  1. gather_pending_and_new   - concat pending_state + new tokens into flat buffer
  2. compress_hca / compress_csa - softmax weighted sum (HCA: ratio tokens; CSA: 2*ratio with overlap inlined)
  3. writeback_pending        - write remainder tokens back to kv_state/score_state

The prefill attention path also uses pack_prefill_kv to materialize each request's
ragged [sliding_history | current_kv | compressed_kv] layout.
"""

import torch
import triton
import triton.language as tl

from chitu.ops.deepseek_compressor import build_compress_metadata
from chitu.ops.triton_ops.utils import auto_retry_triton_compilation

# FIXME: The ragged metadata tensors and most pointer-offset arithmetic in this
# file currently assume int32 indices. Other KV-cache Triton ops switch to
# int64 offsets for large tensors; do the same here before relying on these
# kernels when the flat buffers or KV state/cache can exceed the int32-safe
# address range, e.g. multi-GB KV-cache tensors.

# ---------------------------------------------------------------------------
# Kernel 1: gather pending state + new kv/score into a flat contiguous buffer
# ---------------------------------------------------------------------------


@triton.jit
def _gather_kernel(
    # inputs
    kv_state_ptr,  # [max_batch, max_pending_rows, D]  float32
    score_state_ptr,  # [max_batch, max_pending_rows, D]  float32
    kv_cat_ptr,  # [total_new_tokens, D]             float32
    score_cat_ptr,  # [total_new_tokens, D]             float32
    ape_ptr,  # [ratio, D]                        float32
    # outputs
    flat_kv_ptr,  # [total_eff, D]  float32
    flat_score_ptr,  # [total_eff, D]  float32
    # ragged metadata
    cu_eff_ptr,  # [n+1]  int32  cumsum of effective_lens
    cu_new_ptr,  # [n+1]  int32  cumsum of seqlens (new token counts)
    pending_lens_ptr,  # [n]   int32
    cache_slots_ptr,  # [n]   int32
    # strides
    state_s0: tl.constexpr,  # kv_state.stride(0) = max_pending_rows * D
    state_s1: tl.constexpr,  # kv_state.stride(1) = D
    # dims
    D: tl.constexpr,
    RATIO: tl.constexpr,
    PENDING_OFFSET: tl.constexpr,  # = ratio if overlap (CSA) else 0 (HCA)
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    req = tl.program_id(0)
    t_blk = tl.program_id(1)

    eff_start = tl.load(cu_eff_ptr + req)
    eff_end = tl.load(cu_eff_ptr + req + 1)
    eff_len = eff_end - eff_start

    pending = tl.load(pending_lens_ptr + req)
    new_start = tl.load(cu_new_ptr + req)
    slot = tl.load(cache_slots_ptr + req)

    t_offs = t_blk * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < eff_len

    is_pending = t_offs < pending

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    mask2d = t_mask[:, None] & d_mask[None, :]

    # ---- load kv ----
    # pending part: from kv_state[slot, PENDING_OFFSET + t_offs, :]
    # HCA: PENDING_OFFSET=0; CSA: PENDING_OFFSET=ratio (prev-group ref occupies [0:ratio])
    pend_row = PENDING_OFFSET + t_offs
    pend_kv_ptr = (
        kv_state_ptr + slot * state_s0 + pend_row[:, None] * state_s1 + d_offs[None, :]
    )
    pend_kv = tl.load(
        pend_kv_ptr,
        mask=t_mask[:, None] & d_mask[None, :] & is_pending[:, None],
        other=0.0,
    )

    # new part: from kv_cat[new_start + (t - pending), :]
    new_row = new_start + (t_offs - pending)
    new_kv_ptr = kv_cat_ptr + new_row[:, None] * D + d_offs[None, :]
    new_kv = tl.load(new_kv_ptr, mask=mask2d & ~is_pending[:, None], other=0.0)

    kv_val = tl.where(is_pending[:, None], pend_kv, new_kv)

    # ---- load score ----
    # pending part: kv_state stores score WITH ape already added; subtract ape back.
    # ape index within group = t_offs % RATIO
    ape_row = t_offs % RATIO
    ape_ptr2 = ape_ptr + ape_row[:, None] * D + d_offs[None, :]
    ape_val = tl.load(ape_ptr2, mask=mask2d, other=0.0)

    pend_score_ptr = (
        score_state_ptr
        + slot * state_s0
        + pend_row[:, None] * state_s1
        + d_offs[None, :]
    )
    pend_score = (
        tl.load(
            pend_score_ptr,
            mask=t_mask[:, None] & d_mask[None, :] & is_pending[:, None],
            other=0.0,
        )
        - ape_val
    )

    new_score_ptr = score_cat_ptr + new_row[:, None] * D + d_offs[None, :]
    new_score = tl.load(new_score_ptr, mask=mask2d & ~is_pending[:, None], other=0.0)

    score_val = tl.where(is_pending[:, None], pend_score, new_score)

    # ---- store into flat ----
    dst_row = eff_start + t_offs
    tl.store(flat_kv_ptr + dst_row[:, None] * D + d_offs[None, :], kv_val, mask=mask2d)
    tl.store(
        flat_score_ptr + dst_row[:, None] * D + d_offs[None, :], score_val, mask=mask2d
    )


@auto_retry_triton_compilation
def gather_pending_and_new(
    kv_state: torch.Tensor,  # [max_batch, max_pending_rows, D]
    score_state: torch.Tensor,
    kv_cat: torch.Tensor,  # [total_new_tokens, D]
    score_cat: torch.Tensor,
    ape: torch.Tensor,  # [ratio, D]
    flat_kv: torch.Tensor,  # [total_eff, D]  pre-allocated output
    flat_score: torch.Tensor,
    cu_eff: torch.Tensor,  # [n+1] int32 on GPU
    cu_new: torch.Tensor,  # [n+1] int32 on GPU
    pending_lens: torch.Tensor,  # [n]   int32 on GPU
    cache_slots: torch.Tensor,  # [n]   int32 on GPU
    ratio: int,
    pending_offset: int = 0,  # = ratio for CSA (overlap), 0 for HCA
):
    n = pending_lens.shape[0]
    D = kv_cat.shape[1]
    BLOCK_T = 32
    BLOCK_D = triton.next_power_of_2(D)
    max_eff = int((cu_eff[-1] - cu_eff[0]).item()) if n > 0 else 0
    grid = (n, triton.cdiv(max_eff, BLOCK_T))
    if n == 0 or max_eff == 0:
        return
    _gather_kernel[grid](
        kv_state,
        score_state,
        kv_cat,
        score_cat,
        ape,
        flat_kv,
        flat_score,
        cu_eff,
        cu_new,
        pending_lens,
        cache_slots,
        state_s0=kv_state.stride(0),
        state_s1=kv_state.stride(1),
        D=D,
        RATIO=ratio,
        PENDING_OFFSET=pending_offset,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
    )


# ---------------------------------------------------------------------------
# Kernel 2: compress kernels — HCA and CSA, one group per program
# ---------------------------------------------------------------------------


@triton.jit
def _compress_hca_kernel(
    flat_kv_ptr,  # [total_eff, D]
    flat_score_ptr,  # [total_eff, D]
    ape_ptr,  # [RATIO, D]
    out_kv_ptr,  # [total_groups, D]
    cu_eff_ptr,  # [n+1]  int32
    cu_groups_ptr,  # [n+1]  int32
    group_to_req_ptr,  # [total_groups]  int32  precomputed lookup
    RATIO: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    g = tl.program_id(0)

    req = tl.load(group_to_req_ptr + g)
    eff_start = tl.load(cu_eff_ptr + req)
    grp_in_req = g - tl.load(cu_groups_ptr + req)
    token_start = eff_start + grp_in_req * RATIO

    r_offs = tl.arange(0, RATIO)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # load kv [RATIO, D]
    kv = tl.load(
        flat_kv_ptr + (token_start + r_offs[:, None]) * D + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    )

    # load score + ape [RATIO, D]
    score = tl.load(
        flat_score_ptr + (token_start + r_offs[:, None]) * D + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    )
    ape = tl.load(
        ape_ptr + r_offs[:, None] * D + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    )
    score = score + ape

    # softmax along RATIO dim (numerically stable)
    score_max = tl.max(score, axis=0)  # [D]
    score_exp = tl.exp(score - score_max[None, :])  # [RATIO, D]
    score_sum = tl.sum(score_exp, axis=0)  # [D]
    weights = score_exp / score_sum[None, :]  # [RATIO, D]

    # weighted sum
    compressed = tl.sum(kv * weights, axis=0)  # [D]

    tl.store(out_kv_ptr + g * D + d_offs, compressed, mask=d_mask)


@auto_retry_triton_compilation
def compress_hca(
    flat_kv: torch.Tensor,  # [total_eff, D]
    flat_score: torch.Tensor,  # [total_eff, D]
    ape: torch.Tensor,  # [ratio, D]
    out_kv: torch.Tensor,  # [total_groups, D]  pre-allocated
    cu_eff: torch.Tensor,  # [n+1] int32
    cu_groups: torch.Tensor,  # [n+1] int32
    group_to_req: torch.Tensor,  # [total_groups] int32
    ratio: int,
):
    total_groups = out_kv.shape[0]
    if total_groups == 0:
        return
    D = flat_kv.shape[1]
    BLOCK_D = triton.next_power_of_2(D)
    _compress_hca_kernel[(total_groups,)](
        flat_kv,
        flat_score,
        ape,
        out_kv,
        cu_eff,
        cu_groups,
        group_to_req,
        RATIO=ratio,
        D=D,
        BLOCK_D=BLOCK_D,
    )


# ---------------------------------------------------------------------------
# Kernel 2b: CSA compress — overlap_transform inlined, no serial dependency
#
# CSA layout: flat_kv[t] has shape [coff*head_dim] = [2*head_dim]
#   first head:  flat_kv[t, :head_dim]
#   second head: flat_kv[t, head_dim:]
#
# For group g, the effective input is [2*ratio, head_dim]:
#   rows  0..ratio-1  : group (g-1)'s tokens' first head   (overlap prev)
#   rows ratio..2*ratio-1 : group g's tokens' second head   (current)
#
# Group 0's prev comes from kv_state[slot, :ratio, :head_dim] when pending_len > 0,
# or is zero-padded otherwise.
#
# The compressed output is head_dim (not 2*head_dim).
# ---------------------------------------------------------------------------


@triton.jit
def _compress_csa_kernel(
    flat_kv_ptr,  # [total_eff, 2*HEAD_DIM]
    flat_score_ptr,  # [total_eff, 2*HEAD_DIM]
    ape_ptr,  # [RATIO, HEAD_DIM]  contiguous first-half ape
    prev_ref_kv_ptr,  # [n, RATIO, HEAD_DIM]  per-request group-0 prev, precomputed in Python
    prev_ref_score_ptr,  # [n, RATIO, HEAD_DIM]  raw score (ape already subtracted)
    out_kv_ptr,  # [total_groups, HEAD_DIM]
    cu_eff_ptr,  # [n+1] int32
    cu_groups_ptr,  # [n+1] int32
    group_to_req_ptr,  # [total_groups] int32
    RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    FULL_D: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    g = tl.program_id(0)

    req = tl.load(group_to_req_ptr + g)
    eff_start = tl.load(cu_eff_ptr + req)
    grp_in_req = g - tl.load(cu_groups_ptr + req)
    token_start = eff_start + grp_in_req * RATIO

    r_offs = tl.arange(0, RATIO)
    d_offs = tl.arange(0, BLOCK_HD)
    d_mask = d_offs < HEAD_DIM

    # ---- current group's second head ----
    cur_kv = tl.load(
        flat_kv_ptr
        + (token_start + r_offs[:, None]) * FULL_D
        + (HEAD_DIM + d_offs[None, :]),
        mask=d_mask[None, :],
        other=0.0,
    )
    cur_score = tl.load(
        flat_score_ptr
        + (token_start + r_offs[:, None]) * FULL_D
        + (HEAD_DIM + d_offs[None, :]),
        mask=d_mask[None, :],
        other=0.0,
    )

    # ---- prev group's first head ----
    # group > 0: read from flat_kv[token_start - RATIO, :HEAD_DIM]
    # group == 0: read from prev_ref[req] (precomputed in Python, no conditional needed)
    is_group_zero = grp_in_req == 0
    safe_flat_prev = tl.where(is_group_zero, token_start, token_start - RATIO)

    flat_prev_kv = tl.load(
        flat_kv_ptr + (safe_flat_prev + r_offs[:, None]) * FULL_D + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    )
    flat_prev_score = tl.load(
        flat_score_ptr + (safe_flat_prev + r_offs[:, None]) * FULL_D + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    )

    # prev_ref is valid for all requests (Python fills zeros/-inf when no prev)
    ref_prev_kv = tl.load(
        prev_ref_kv_ptr
        + req * RATIO * HEAD_DIM
        + r_offs[:, None] * HEAD_DIM
        + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    )
    ref_prev_score = tl.load(
        prev_ref_score_ptr
        + req * RATIO * HEAD_DIM
        + r_offs[:, None] * HEAD_DIM
        + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    )

    prev_kv = tl.where(is_group_zero, ref_prev_kv, flat_prev_kv)
    prev_score = tl.where(is_group_zero, ref_prev_score, flat_prev_score)

    # ---- apply ape and softmax over 2*RATIO ----
    ape_r = tl.load(
        ape_ptr + r_offs[:, None] * HEAD_DIM + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    )
    score_prev_final = prev_score + ape_r
    score_cur_final = cur_score + ape_r

    s_max = tl.maximum(
        tl.max(score_prev_final, axis=0), tl.max(score_cur_final, axis=0)
    )
    exp_p = tl.exp(score_prev_final - s_max[None, :])
    exp_c = tl.exp(score_cur_final - s_max[None, :])
    s_sum = tl.sum(exp_p, axis=0) + tl.sum(exp_c, axis=0)

    compressed = (
        tl.sum(prev_kv * exp_p, axis=0) + tl.sum(cur_kv * exp_c, axis=0)
    ) / s_sum

    tl.store(out_kv_ptr + g * HEAD_DIM + d_offs, compressed, mask=d_mask)


# ---------------------------------------------------------------------------
# Kernel 2c: writeback prev-group reference to kv_state[:ratio] for CSA
#   After compress, store the last complete group's first head into kv_state[slot, :ratio].
#   This is needed for the *next chunk*'s group 0.
# ---------------------------------------------------------------------------


@triton.jit
def _writeback_csa_prev_kernel(
    flat_kv_ptr,  # [total_eff, FULL_D]
    flat_score_ptr,
    kv_state_ptr,  # [max_batch, coff*RATIO, coff*HEAD_DIM]
    score_state_ptr,
    ape_ptr,  # [RATIO, HEAD_DIM]  ape for first half
    cu_eff_ptr,  # [n+1] int32
    cutoffs_ptr,  # [n]   int32  (= n_groups * ratio for each req)
    n_groups_ptr,  # [n]   int32
    cache_slots_ptr,  # [n]   int32
    state_s0,
    state_s1,
    RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    FULL_D: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    req = tl.program_id(0)

    n_groups = tl.load(n_groups_ptr + req)
    if n_groups == 0:
        return

    slot = tl.load(cache_slots_ptr + req)
    eff_start = tl.load(cu_eff_ptr + req)
    cutoff = tl.load(cutoffs_ptr + req)
    # last complete group starts at cutoff - RATIO in flat space
    last_grp_start = eff_start + cutoff - RATIO

    r_offs = tl.arange(0, RATIO)
    d_offs = tl.arange(0, BLOCK_HD)
    d_mask = d_offs < HEAD_DIM

    # load last group's first head from flat_kv
    kv_val = tl.load(
        flat_kv_ptr + (last_grp_start + r_offs[:, None]) * FULL_D + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    )
    score_val = tl.load(
        flat_score_ptr + (last_grp_start + r_offs[:, None]) * FULL_D + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    )
    # add ape so kv_state stores score+ape (consistent with existing convention)
    ape_val = tl.load(
        ape_ptr + r_offs[:, None] * HEAD_DIM + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    )
    score_val = score_val + ape_val

    # write to kv_state[slot, :RATIO, :HEAD_DIM]
    tl.store(
        kv_state_ptr + slot * state_s0 + r_offs[:, None] * state_s1 + d_offs[None, :],
        kv_val,
        mask=d_mask[None, :],
    )
    tl.store(
        score_state_ptr
        + slot * state_s0
        + r_offs[:, None] * state_s1
        + d_offs[None, :],
        score_val,
        mask=d_mask[None, :],
    )


@auto_retry_triton_compilation
def compress_csa(
    flat_kv: torch.Tensor,  # [total_eff, 2*head_dim]
    flat_score: torch.Tensor,
    ape: torch.Tensor,  # [ratio, head_dim]  contiguous first-half ape
    kv_state: torch.Tensor,  # [max_batch, coff*ratio, coff*head_dim]
    score_state: torch.Tensor,
    out_kv: torch.Tensor,  # [total_groups, head_dim]
    cu_eff: torch.Tensor,
    cu_groups: torch.Tensor,
    group_to_req: torch.Tensor,
    pending_lens: torch.Tensor,
    cache_slots: torch.Tensor,
    cutoffs: torch.Tensor,  # [n] int32
    n_groups_list_t: torch.Tensor,  # [n] int32
    ratio: int,
    head_dim: int,
):
    """Compress CSA groups and update the per-slot prev-group state in-place.

    Besides writing compressed vectors to ``out_kv``, CSA must carry the last
    complete group's first head into ``kv_state[slot, :ratio, :head_dim]`` and
    ``score_state[slot, :ratio, :head_dim]``. The next chunk's group-0 overlap
    reads that state. Keep this state update aligned with the torch fallback.
    """
    total_groups = out_kv.shape[0]
    n = pending_lens.shape[0]
    FULL_D = flat_kv.shape[1]  # = 2 * head_dim
    BLOCK_HD = triton.next_power_of_2(head_dim)
    ape = ape.contiguous()

    # Build prev_ref[n, ratio, head_dim] on Python side to avoid scalar bool in kernel.
    # For each request:
    #   pending > 0: use kv_state[slot, :ratio, :head_dim] as prev (raw score = state - ape)
    #   pending == 0: zeros kv, -inf score (softmax → weight 0)
    pending_lens_cpu = pending_lens.cpu().tolist()
    cache_slots_cpu = cache_slots.cpu().tolist()
    device = flat_kv.device
    prev_ref_kv = torch.zeros(n, ratio, head_dim, device=device, dtype=torch.float32)
    prev_ref_score = torch.full(
        (n, ratio, head_dim), float("-inf"), device=device, dtype=torch.float32
    )
    for i in range(n):
        if pending_lens_cpu[i] > 0:
            slot = cache_slots_cpu[i]
            prev_ref_kv[i] = kv_state[slot, :ratio, :head_dim]
            prev_ref_score[i] = score_state[slot, :ratio, :head_dim] - ape
    prev_ref_kv = prev_ref_kv.contiguous()
    prev_ref_score = prev_ref_score.contiguous()

    if total_groups > 0:
        _compress_csa_kernel[(total_groups,)](
            flat_kv,
            flat_score,
            ape,
            prev_ref_kv,
            prev_ref_score,
            out_kv,
            cu_eff,
            cu_groups,
            group_to_req,
            RATIO=ratio,
            HEAD_DIM=head_dim,
            FULL_D=FULL_D,
            BLOCK_HD=BLOCK_HD,
        )

    if n > 0:
        _writeback_csa_prev_kernel[(n,)](
            flat_kv,
            flat_score,
            kv_state,
            score_state,
            ape,
            cu_eff,
            cutoffs,
            n_groups_list_t,
            cache_slots,
            state_s0=kv_state.stride(0),
            state_s1=kv_state.stride(1),
            RATIO=ratio,
            HEAD_DIM=head_dim,
            FULL_D=FULL_D,
            BLOCK_HD=BLOCK_HD,
        )


# ---------------------------------------------------------------------------


@triton.jit
def _writeback_pending_kernel(
    flat_kv_ptr,  # [total_eff, D]
    flat_score_ptr,  # [total_eff, D]
    kv_state_ptr,  # [max_batch, max_pending_rows, D]
    score_state_ptr,  # [max_batch, max_pending_rows, D]
    ape_ptr,  # [ratio, D]
    cu_eff_ptr,  # [n+1] int32
    cutoffs_ptr,  # [n]   int32
    remainders_ptr,  # [n]   int32
    cache_slots_ptr,  # [n]   int32
    pending_offset,  # int: = ratio if overlap else 0
    state_s0: tl.constexpr,
    state_s1: tl.constexpr,
    D: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK_R: tl.constexpr,  # constexpr >= ratio (max_remainder = ratio-1)
    BLOCK_D: tl.constexpr,
):
    req = tl.program_id(0)

    remainder = tl.load(remainders_ptr + req)
    if remainder == 0:
        return

    eff_start = tl.load(cu_eff_ptr + req)
    cutoff = tl.load(cutoffs_ptr + req)
    slot = tl.load(cache_slots_ptr + req)

    r_offs = tl.arange(0, BLOCK_R)
    d_offs = tl.arange(0, BLOCK_D)
    r_mask = r_offs < remainder
    d_mask = d_offs < D
    mask2d = r_mask[:, None] & d_mask[None, :]

    # source: flat_kv[eff_start + cutoff + r, :]
    src_row = eff_start + cutoff + r_offs
    kv_val = tl.load(
        flat_kv_ptr + src_row[:, None] * D + d_offs[None, :], mask=mask2d, other=0.0
    )
    score_val = tl.load(
        flat_score_ptr + src_row[:, None] * D + d_offs[None, :], mask=mask2d, other=0.0
    )

    # score: re-add ape (kv_state stores score WITH ape)
    # r_offs is the index within the group = r_offs % RATIO (r_offs already < ratio)
    ape_val = tl.load(
        ape_ptr + r_offs[:, None] * D + d_offs[None, :], mask=mask2d, other=0.0
    )
    score_val = score_val + ape_val

    # dest: kv_state[slot, pending_offset + r, :]
    dst_row = pending_offset + r_offs
    tl.store(
        kv_state_ptr + slot * state_s0 + dst_row[:, None] * state_s1 + d_offs[None, :],
        kv_val,
        mask=mask2d,
    )
    tl.store(
        score_state_ptr
        + slot * state_s0
        + dst_row[:, None] * state_s1
        + d_offs[None, :],
        score_val,
        mask=mask2d,
    )


@auto_retry_triton_compilation
def writeback_pending(
    flat_kv: torch.Tensor,
    flat_score: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    ape: torch.Tensor,
    cu_eff: torch.Tensor,  # [n+1] int32
    cutoffs: torch.Tensor,  # [n] int32
    remainders: torch.Tensor,  # [n] int32
    cache_slots: torch.Tensor,  # [n] int32
    pending_offset: int,
    ratio: int,
):
    n = remainders.shape[0]
    if n == 0:
        return
    D = flat_kv.shape[1]
    BLOCK_R = triton.next_power_of_2(ratio)  # ratio - 1 max remainder, power-of-2 fits
    BLOCK_D = triton.next_power_of_2(D)
    _writeback_pending_kernel[(n,)](
        flat_kv,
        flat_score,
        kv_state,
        score_state,
        ape,
        cu_eff,
        cutoffs,
        remainders,
        cache_slots,
        pending_offset=pending_offset,
        state_s0=kv_state.stride(0),
        state_s1=kv_state.stride(1),
        D=D,
        RATIO=ratio,
        BLOCK_R=BLOCK_R,
        BLOCK_D=BLOCK_D,
    )


# ---------------------------------------------------------------------------
# Kernel 4: pack prefill attention KV into ragged [sliding_history | kv_new | compressed_kv]
# ---------------------------------------------------------------------------


@triton.jit
def _pack_prefill_kv_kernel(
    history_kv_ptr,  # [total_history, D]
    current_kv_ptr,  # [total_current, D]
    compressed_kv_ptr,  # [total_compressed, D]
    out_kv_ptr,  # [total_out, D]
    history_offsets_ptr,  # [n+1]
    current_offsets_ptr,  # [n+1]
    compressed_offsets_ptr,  # [n+1]
    out_offsets_ptr,  # [n+1]
    history_lens_ptr,  # [n]
    current_lens_ptr,  # [n]
    compressed_lens_ptr,  # [n]
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    req = tl.program_id(0)
    t_blk = tl.program_id(1)

    h_len = tl.load(history_lens_ptr + req)
    cur_len = tl.load(current_lens_ptr + req)
    comp_len = tl.load(compressed_lens_ptr + req)
    out_len = h_len + cur_len + comp_len

    h_start = tl.load(history_offsets_ptr + req)
    cur_start = tl.load(current_offsets_ptr + req)
    comp_start = tl.load(compressed_offsets_ptr + req)
    out_start = tl.load(out_offsets_ptr + req)

    t_offs = t_blk * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offs = tl.arange(0, BLOCK_D)
    t_mask = t_offs < out_len
    d_mask = d_offs < D
    mask = t_mask[:, None] & d_mask[None, :]

    is_history = t_offs < h_len
    is_current = (t_offs >= h_len) & (t_offs < h_len + cur_len)
    is_compressed = t_offs >= h_len + cur_len

    history_rows = h_start + t_offs
    current_rows = cur_start + (t_offs - h_len)
    compressed_rows = comp_start + (t_offs - h_len - cur_len)

    history_val = tl.load(
        history_kv_ptr + history_rows[:, None] * D + d_offs[None, :],
        mask=mask & is_history[:, None],
        other=0.0,
    )
    current_val = tl.load(
        current_kv_ptr + current_rows[:, None] * D + d_offs[None, :],
        mask=mask & is_current[:, None],
        other=0.0,
    )
    compressed_val = tl.load(
        compressed_kv_ptr + compressed_rows[:, None] * D + d_offs[None, :],
        mask=mask & is_compressed[:, None],
        other=0.0,
    )
    out_val = history_val + current_val + compressed_val

    out_rows = out_start + t_offs
    tl.store(
        out_kv_ptr + out_rows[:, None] * D + d_offs[None, :],
        out_val,
        mask=mask,
    )


@auto_retry_triton_compilation
def pack_prefill_kv(
    history_kv: torch.Tensor,
    current_kv: torch.Tensor,
    compressed_kv: torch.Tensor,
    out_kv: torch.Tensor,
    history_offsets: torch.Tensor,
    current_offsets: torch.Tensor,
    compressed_offsets: torch.Tensor,
    out_offsets: torch.Tensor,
    history_lens: torch.Tensor,
    current_lens: torch.Tensor,
    compressed_lens: torch.Tensor,
):
    n = history_lens.shape[0]
    if n == 0 or out_kv.numel() == 0:
        return

    D = out_kv.shape[1]
    max_out = int((history_lens + current_lens + compressed_lens).max().item())
    if max_out == 0:
        return

    BLOCK_T = 16
    BLOCK_D = triton.next_power_of_2(D)
    grid = (n, triton.cdiv(max_out, BLOCK_T))
    _pack_prefill_kv_kernel[grid](
        history_kv,
        current_kv,
        compressed_kv,
        out_kv,
        history_offsets,
        current_offsets,
        compressed_offsets,
        out_offsets,
        history_lens,
        current_lens,
        compressed_lens,
        D=D,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
    )
