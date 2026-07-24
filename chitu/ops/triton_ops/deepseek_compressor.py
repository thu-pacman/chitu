# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Fused triton kernels for HCA and CSA compressors.

Three kernels shared by both modes:
  1. gather_pending_and_new   - concat pending_state + new tokens into flat buffer
  2. compress_hca / compress_csa - softmax weighted sum (HCA: ratio tokens; CSA: previous group + current group)
  3. writeback_pending        - write the live logical suffix back to the pending-state ring

The prefill attention path also uses pack_prefill_kv to materialize each request's
ragged [sliding_history | current_kv | compressed_kv] layout.
"""

import torch
import triton
import triton.language as tl

from chitu.ops.triton_ops.utils import auto_retry_triton_compilation

# FIXME: The ragged metadata tensors and most pointer-offset arithmetic in this
# file currently assume int32 indices. Other KV-cache Triton ops switch to
# int64 offsets for large tensors; do the same here before relying on these
# kernels when the flat buffers or KV state/cache can exceed the int32-safe
# address range, e.g. multi-GB KV-cache tensors.


@triton.jit
def _masked_write_kv_cache_kernel(
    values_ptr,
    kv_cache_ptr,
    block_table_ptr,
    cache_slots_ptr,
    positions_ptr,
    cache_seq_ids_ptr,
    valid_mask_ptr,
    kv_stride0: tl.constexpr,
    kv_stride1: tl.constexpr,
    kv_stride2: tl.constexpr,
    block_table_stride0: tl.constexpr,
    page_size: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGED: tl.constexpr,
):
    token_id = tl.program_id(0)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    valid = tl.load(valid_mask_ptr + token_id)

    position = tl.load(positions_ptr + token_id).to(tl.int64)
    if PAGED:
        seq_id = tl.load(cache_seq_ids_ptr + token_id).to(tl.int64)
        page_id = tl.load(
            block_table_ptr + seq_id * block_table_stride0 + position // page_size,
            mask=valid,
            other=0,
        ).to(tl.int64)
        row0 = page_id
        row1 = position % page_size
    else:
        row0 = tl.load(cache_slots_ptr + token_id).to(tl.int64)
        row1 = position

    vals = tl.load(values_ptr + token_id * D + d_offs, mask=d_mask, other=0.0)
    tl.store(
        kv_cache_ptr + row0 * kv_stride0 + row1 * kv_stride1 + d_offs * kv_stride2,
        vals,
        mask=valid & d_mask,
    )


@auto_retry_triton_compilation
def masked_write_kv_cache(
    values: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor | None,
    cache_slots: torch.Tensor,
    positions: torch.Tensor,
    cache_seq_ids: torch.Tensor,
    valid_mask: torch.Tensor,
):
    values = values.contiguous()
    cache_slots = cache_slots.contiguous()
    positions = positions.contiguous()
    cache_seq_ids = cache_seq_ids.contiguous()
    valid_mask = valid_mask.to(device=values.device, dtype=torch.bool).contiguous()
    n = values.shape[0]
    if n == 0:
        return
    D = values.shape[-1]
    BLOCK_D = triton.next_power_of_2(D)
    paged = block_table is not None
    dummy_block_table = block_table if paged else kv_cache
    _masked_write_kv_cache_kernel[(n,)](
        values,
        kv_cache,
        dummy_block_table,
        cache_slots,
        positions,
        cache_seq_ids,
        valid_mask,
        kv_stride0=kv_cache.stride(0),
        kv_stride1=kv_cache.stride(1),
        kv_stride2=kv_cache.stride(2),
        block_table_stride0=block_table.stride(0) if paged else 0,
        page_size=kv_cache.shape[1] if paged else 1,
        D=D,
        BLOCK_D=BLOCK_D,
        PAGED=paged,
    )


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
    start_positions_ptr,  # [n] int64
    cache_slots_ptr,  # [n]   int32
    # strides
    state_s0: tl.constexpr,  # kv_state.stride(0) = max_pending_rows * D
    state_s1: tl.constexpr,  # kv_state.stride(1) = D
    # dims
    D: tl.constexpr,
    RATIO: tl.constexpr,
    PENDING_OFFSET: tl.constexpr,  # = ratio for CSA, 0 for HCA
    STATE_ROWS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    req = tl.program_id(0)
    t_blk = tl.program_id(1)

    eff_start = tl.load(cu_eff_ptr + req)
    eff_end = tl.load(cu_eff_ptr + req + 1)
    eff_len = eff_end - eff_start

    pending = tl.load(pending_lens_ptr + req)
    start_position = tl.load(start_positions_ptr + req)
    pending_abs_start = start_position - pending
    new_start = tl.load(cu_new_ptr + req)
    slot = tl.load(cache_slots_ptr + req)

    t_offs = t_blk * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < eff_len

    is_pending = t_offs < pending

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    mask2d = t_mask[:, None] & d_mask[None, :]

    # ---- load kv ----
    # pending part: from the logical-position ring. Rejected speculative rows
    # may remain physically present, but the rolled-back logical position makes
    # them invisible to this gather.
    pend_pos = pending_abs_start + t_offs
    pend_row = pend_pos % STATE_ROWS
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
    ape_row = pend_pos % RATIO
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
    start_positions: torch.Tensor,  # [n] int64 on GPU
    cache_slots: torch.Tensor,  # [n]   int32 on GPU
    ratio: int,
    pending_offset: int = 0,  # = ratio for CSA, 0 for HCA
    max_eff: int | None = None,
):
    n = pending_lens.shape[0]
    D = kv_cat.shape[1]
    BLOCK_T = 32
    BLOCK_D = triton.next_power_of_2(D)
    if max_eff is None:
        # Prefer passing this from CPU-built metadata. Falling back keeps the
        # helper usable for generic callers, but it synchronizes the stream.
        max_eff = int((cu_eff[-1] - cu_eff[0]).item()) if n > 0 else 0
    else:
        max_eff = int(max_eff)
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
        start_positions,
        cache_slots,
        state_s0=kv_state.stride(0),
        state_s1=kv_state.stride(1),
        D=D,
        RATIO=ratio,
        PENDING_OFFSET=pending_offset,
        STATE_ROWS=kv_state.shape[1],
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
#   rows  0..ratio-1  : group (g-1)'s tokens' first head   (CSA prev group)
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
    ape_ptr,  # [RATIO, FULL_D]
    kv_state_ptr,  # [max_batch, coff*RATIO, coff*HEAD_DIM]
    score_state_ptr,
    out_kv_ptr,  # [total_groups, HEAD_DIM]
    cu_eff_ptr,  # [n+1] int32
    cu_groups_ptr,  # [n+1] int32
    group_to_req_ptr,  # [total_groups] int32
    pending_lens_ptr,  # [n] int32
    start_positions_ptr,  # [n] int64
    cache_slots_ptr,  # [n] int32
    state_s0,
    state_s1,
    RATIO: tl.constexpr,
    STATE_ROWS: tl.constexpr,
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

    ape_prev = tl.load(
        ape_ptr + r_offs[:, None] * FULL_D + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    )
    ape_cur = tl.load(
        ape_ptr + r_offs[:, None] * FULL_D + HEAD_DIM + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    )

    # ---- prev group's first head ----
    # group > 0: read from flat_kv[token_start - RATIO, :HEAD_DIM]
    # group == 0: read from kv_state[slot, :RATIO, :HEAD_DIM] if there is pending
    # state, otherwise use zero kv / -inf score.
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

    pending = tl.load(pending_lens_ptr + req)
    start_position = tl.load(start_positions_ptr + req)
    group_abs_start = start_position - pending
    prev_pos = group_abs_start - RATIO + r_offs
    slot = tl.load(cache_slots_ptr + req)
    has_prev_ref = is_group_zero & (prev_pos >= 0)
    prev_rows = prev_pos % STATE_ROWS
    ref_prev_kv = tl.load(
        kv_state_ptr
        + slot * state_s0
        + prev_rows[:, None] * state_s1
        + d_offs[None, :],
        mask=d_mask[None, :] & has_prev_ref[:, None],
        other=0.0,
    )
    ref_prev_score = tl.load(
        score_state_ptr
        + slot * state_s0
        + prev_rows[:, None] * state_s1
        + d_offs[None, :],
        mask=d_mask[None, :] & has_prev_ref[:, None],
        other=-float("inf"),
    )

    prev_kv = tl.where(is_group_zero, ref_prev_kv, flat_prev_kv)
    prev_score = tl.where(is_group_zero, ref_prev_score - ape_prev, flat_prev_score)

    # ---- apply ape and softmax over 2*RATIO ----
    score_prev_final = prev_score + ape_prev
    score_cur_final = cur_score + ape_cur

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


@auto_retry_triton_compilation
def compress_csa(
    flat_kv: torch.Tensor,  # [total_eff, 2*head_dim]
    flat_score: torch.Tensor,
    ape: torch.Tensor,  # [ratio, full_d]
    kv_state: torch.Tensor,  # [max_batch, coff*ratio, coff*head_dim]
    score_state: torch.Tensor,
    out_kv: torch.Tensor,  # [total_groups, head_dim]
    cu_eff: torch.Tensor,
    cu_groups: torch.Tensor,
    group_to_req: torch.Tensor,
    pending_lens: torch.Tensor,
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    n_groups_list_t: torch.Tensor,  # [n] int32
    ratio: int,
    head_dim: int,
):
    """Compress CSA groups using logical-position pending-state ring rows."""
    total_groups = out_kv.shape[0]
    n = pending_lens.shape[0]
    FULL_D = flat_kv.shape[1]  # = 2 * head_dim
    BLOCK_HD = triton.next_power_of_2(head_dim)
    ape = ape.contiguous()

    if total_groups > 0:
        _compress_csa_kernel[(total_groups,)](
            flat_kv,
            flat_score,
            ape,
            kv_state,
            score_state,
            out_kv,
            cu_eff,
            cu_groups,
            group_to_req,
            pending_lens,
            start_positions,
            cache_slots,
            kv_state.stride(0),
            kv_state.stride(1),
            RATIO=ratio,
            STATE_ROWS=kv_state.shape[1],
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
    pending_lens_ptr,  # [n] int32
    start_positions_ptr,  # [n] int64
    cache_slots_ptr,  # [n]   int32
    pending_offset,  # int: = ratio for CSA, 0 for HCA
    state_s0: tl.constexpr,
    state_s1: tl.constexpr,
    D: tl.constexpr,
    RATIO: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    req = tl.program_id(0)

    eff_start = tl.load(cu_eff_ptr + req)
    eff_end = tl.load(cu_eff_ptr + req + 1)
    eff_len = eff_end - eff_start
    if eff_len == 0:
        return

    pending = tl.load(pending_lens_ptr + req)
    start_position = tl.load(start_positions_ptr + req)
    slot = tl.load(cache_slots_ptr + req)
    suffix_len = tl.minimum(eff_len, STATE_ROWS)
    suffix_abs_start = start_position - pending + eff_len - suffix_len
    suffix_src_start = eff_start + eff_len - suffix_len

    r_offs = tl.arange(0, BLOCK_R)
    d_offs = tl.arange(0, BLOCK_D)
    r_mask = r_offs < suffix_len
    d_mask = d_offs < D
    mask2d = r_mask[:, None] & d_mask[None, :]

    src_row = suffix_src_start + r_offs
    kv_val = tl.load(
        flat_kv_ptr + src_row[:, None] * D + d_offs[None, :], mask=mask2d, other=0.0
    )
    score_val = tl.load(
        flat_score_ptr + src_row[:, None] * D + d_offs[None, :], mask=mask2d, other=0.0
    )

    pos = suffix_abs_start + r_offs
    dst_row = pos % STATE_ROWS
    ape_row = pos % RATIO
    ape_val = tl.load(
        ape_ptr + ape_row[:, None] * D + d_offs[None, :], mask=mask2d, other=0.0
    )
    score_val = score_val + ape_val

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
    pending_lens: torch.Tensor,  # [n] int32
    start_positions: torch.Tensor,  # [n] int64
    cache_slots: torch.Tensor,  # [n] int32
    pending_offset: int,
    ratio: int,
):
    n = pending_lens.shape[0]
    if n == 0:
        return
    D = flat_kv.shape[1]
    state_rows = kv_state.shape[1]
    BLOCK_R = triton.next_power_of_2(state_rows)
    BLOCK_D = triton.next_power_of_2(D)
    _writeback_pending_kernel[(n,)](
        flat_kv,
        flat_score,
        kv_state,
        score_state,
        ape,
        cu_eff,
        pending_lens,
        start_positions,
        cache_slots,
        pending_offset=pending_offset,
        state_s0=kv_state.stride(0),
        state_s1=kv_state.stride(1),
        D=D,
        RATIO=ratio,
        STATE_ROWS=state_rows,
        BLOCK_R=BLOCK_R,
        BLOCK_D=BLOCK_D,
    )


# ---------------------------------------------------------------------------
# MTP decode compressor fast path
#
# MTP decode appends a uniform short chunk per request. For DeepSeek-V4 the
# chunk length is <= compress ratio in the supported fast path, so each request
# can produce at most one compressed row. These kernels avoid the generic
# prefill flow's flat gather buffers for the hot decode path.
# ---------------------------------------------------------------------------


@triton.jit
def _decode_should_compress(sp, RATIO: tl.constexpr, Q_LEN: tl.constexpr):
    return sp % RATIO + Q_LEN >= RATIO


@triton.jit
def _decode_mtp_hca_compress_kernel(
    kv_cat_ptr,  # [bsz * Q_LEN, D] raw wkv output
    score_cat_ptr,  # [bsz * Q_LEN, D] raw gate output
    kv_state_ptr,  # [max_batch, RATIO, D]
    score_state_ptr,
    ape_ptr,  # [RATIO, D]
    out_kv_ptr,  # [n_compressed, D]
    compressed_reqs_ptr,  # [n_compressed] int64 request ids
    start_positions_ptr,  # [bsz]
    cache_slots_ptr,  # [bsz]
    state_s0: tl.constexpr,
    state_s1: tl.constexpr,
    RATIO: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    Q_LEN: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    out_req = tl.program_id(0)
    req = tl.load(compressed_reqs_ptr + out_req)
    sp = tl.load(start_positions_ptr + req)
    slot = tl.load(cache_slots_ptr + req)
    pending = sp % RATIO
    has_group = _decode_should_compress(sp, RATIO, Q_LEN)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    if not has_group:
        tl.store(
            out_kv_ptr + out_req * D + d_offs,
            tl.zeros((BLOCK_D,), tl.float32),
            mask=d_mask,
        )
        return
    group_start = sp - pending

    r_offs = tl.arange(0, RATIO)
    is_pending = r_offs < pending
    new_idx = r_offs - pending
    state_rows = (group_start + r_offs) % STATE_ROWS

    state_kv = tl.load(
        kv_state_ptr
        + slot * state_s0
        + state_rows[:, None] * state_s1
        + d_offs[None, :],
        mask=d_mask[None, :] & is_pending[:, None],
        other=0.0,
    )
    state_score = tl.load(
        score_state_ptr
        + slot * state_s0
        + state_rows[:, None] * state_s1
        + d_offs[None, :],
        mask=d_mask[None, :] & is_pending[:, None],
        other=0.0,
    )

    new_rows = req * Q_LEN + new_idx
    new_mask = (~is_pending)[:, None] & (new_idx[:, None] < Q_LEN) & d_mask[None, :]
    new_kv = tl.load(
        kv_cat_ptr + new_rows[:, None] * D + d_offs[None, :],
        mask=new_mask,
        other=0.0,
    )
    new_score = tl.load(
        score_cat_ptr + new_rows[:, None] * D + d_offs[None, :],
        mask=new_mask,
        other=0.0,
    )
    ape = tl.load(
        ape_ptr + r_offs[:, None] * D + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    )

    kv = tl.where(is_pending[:, None], state_kv, new_kv)
    score = tl.where(is_pending[:, None], state_score, new_score + ape)

    score_max = tl.max(score, axis=0)
    score_exp = tl.exp(score - score_max[None, :])
    score_sum = tl.sum(score_exp, axis=0)
    compressed = tl.sum(kv * score_exp, axis=0) / score_sum

    tl.store(out_kv_ptr + out_req * D + d_offs, compressed, mask=d_mask)


@triton.jit
def _decode_mtp_hca_update_kernel(
    kv_cat_ptr,  # [bsz * Q_LEN, D]
    score_cat_ptr,  # [bsz * Q_LEN, D]
    kv_state_ptr,  # [max_batch, RATIO, D]
    score_state_ptr,
    ape_ptr,  # [RATIO, D]
    start_positions_ptr,  # [bsz]
    cache_slots_ptr,  # [bsz]
    state_s0: tl.constexpr,
    state_s1: tl.constexpr,
    RATIO: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    Q_LEN: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    req = tl.program_id(0)
    sp = tl.load(start_positions_ptr + req)
    slot = tl.load(cache_slots_ptr + req)

    t_offs = tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, BLOCK_D)
    t_mask = t_offs < Q_LEN
    d_mask = d_offs < D
    positions = sp + t_offs
    dst_rows = positions % STATE_ROWS
    ape_rows = positions % RATIO
    src_rows = req * Q_LEN + t_offs
    kv = tl.load(
        kv_cat_ptr + src_rows[:, None] * D + d_offs[None, :],
        mask=t_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    score = tl.load(
        score_cat_ptr + src_rows[:, None] * D + d_offs[None, :],
        mask=t_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    ape = tl.load(
        ape_ptr + ape_rows[:, None] * D + d_offs[None, :],
        mask=t_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    tl.store(
        kv_state_ptr + slot * state_s0 + dst_rows[:, None] * state_s1 + d_offs[None, :],
        kv,
        mask=t_mask[:, None] & d_mask[None, :],
    )
    tl.store(
        score_state_ptr
        + slot * state_s0
        + dst_rows[:, None] * state_s1
        + d_offs[None, :],
        score + ape,
        mask=t_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _decode_mtp_csa_kernel(
    kv_cat_ptr,  # [bsz * Q_LEN, 2 * HEAD_DIM]
    score_cat_ptr,
    kv_state_ptr,  # [max_batch, 2 * RATIO, 2 * HEAD_DIM]
    score_state_ptr,
    ape_ptr,  # [RATIO, 2 * HEAD_DIM]
    out_kv_ptr,  # [bsz, HEAD_DIM]
    start_positions_ptr,
    cache_slots_ptr,
    state_s0: tl.constexpr,
    state_s1: tl.constexpr,
    RATIO: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    Q_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    FULL_D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    BLOCK_FULL_D: tl.constexpr,
):
    req = tl.program_id(0)
    sp = tl.load(start_positions_ptr + req)
    slot = tl.load(cache_slots_ptr + req)
    pending = sp % RATIO
    has_group = _decode_should_compress(sp, RATIO, Q_LEN)
    group_start = sp - pending
    hd_offs = tl.arange(0, BLOCK_HD)
    hd_mask = hd_offs < HEAD_DIM
    r_offs = tl.arange(0, RATIO)
    is_pending = r_offs < pending
    new_idx = r_offs - pending
    cur_pos = group_start + r_offs
    cur_rows = cur_pos % STATE_ROWS

    # Save every draft token to the logical ring. If a suffix is rejected, the
    # scheduler rolls back the logical length; these physical rows are then
    # ignored until the same positions are produced again and overwritten.
    t_offs = tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, BLOCK_FULL_D)
    t_mask = t_offs < Q_LEN
    d_mask = d_offs < FULL_D
    positions = sp + t_offs
    dst_rows = positions % STATE_ROWS
    ape_rows = positions % RATIO
    src_rows = req * Q_LEN + t_offs
    new_kv = tl.load(
        kv_cat_ptr + src_rows[:, None] * FULL_D + d_offs[None, :],
        mask=t_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    new_score = tl.load(
        score_cat_ptr + src_rows[:, None] * FULL_D + d_offs[None, :],
        mask=t_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    new_ape = tl.load(
        ape_ptr + ape_rows[:, None] * FULL_D + d_offs[None, :],
        mask=t_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    tl.store(
        kv_state_ptr + slot * state_s0 + dst_rows[:, None] * state_s1 + d_offs[None, :],
        new_kv,
        mask=t_mask[:, None] & d_mask[None, :],
    )
    tl.store(
        score_state_ptr
        + slot * state_s0
        + dst_rows[:, None] * state_s1
        + d_offs[None, :],
        new_score + new_ape,
        mask=t_mask[:, None] & d_mask[None, :],
    )

    if not has_group:
        tl.store(
            out_kv_ptr + req * HEAD_DIM + hd_offs,
            tl.zeros((BLOCK_HD,), tl.float32),
            mask=hd_mask,
        )
        return

    # Current group's second head. Pending rows are read by logical position;
    # draft rows that later get rejected are ignored after cached length rolls back.
    pending_cur_kv = tl.load(
        kv_state_ptr
        + slot * state_s0
        + cur_rows[:, None] * state_s1
        + (HEAD_DIM + hd_offs[None, :]),
        mask=is_pending[:, None] & hd_mask[None, :] & has_group,
        other=0.0,
    )
    pending_cur_score = tl.load(
        score_state_ptr
        + slot * state_s0
        + cur_rows[:, None] * state_s1
        + (HEAD_DIM + hd_offs[None, :]),
        mask=is_pending[:, None] & hd_mask[None, :] & has_group,
        other=0.0,
    )

    new_rows = req * Q_LEN + new_idx
    new_mask = (~is_pending)[:, None] & hd_mask[None, :] & has_group
    new_cur_kv = tl.load(
        kv_cat_ptr + new_rows[:, None] * FULL_D + (HEAD_DIM + hd_offs[None, :]),
        mask=new_mask,
        other=0.0,
    )
    new_cur_score = tl.load(
        score_cat_ptr + new_rows[:, None] * FULL_D + (HEAD_DIM + hd_offs[None, :]),
        mask=new_mask,
        other=0.0,
    )

    ape_cur = tl.load(
        ape_ptr + r_offs[:, None] * FULL_D + HEAD_DIM + hd_offs[None, :],
        mask=hd_mask[None, :] & has_group,
        other=0.0,
    )
    cur_kv = tl.where(is_pending[:, None], pending_cur_kv, new_cur_kv)
    cur_score = tl.where(
        is_pending[:, None], pending_cur_score, new_cur_score + ape_cur
    )

    prev_pos = group_start - RATIO + r_offs
    prev_rows = prev_pos % STATE_ROWS
    has_prev_ref = has_group & (prev_pos >= 0)
    prev_kv = tl.load(
        kv_state_ptr
        + slot * state_s0
        + prev_rows[:, None] * state_s1
        + hd_offs[None, :],
        mask=hd_mask[None, :] & has_prev_ref[:, None],
        other=0.0,
    )
    prev_score = tl.load(
        score_state_ptr
        + slot * state_s0
        + prev_rows[:, None] * state_s1
        + hd_offs[None, :],
        mask=hd_mask[None, :] & has_prev_ref[:, None],
        other=-float("inf"),
    )

    cur_score = tl.where(has_group, cur_score, -float("inf"))
    prev_score = tl.where(has_group, prev_score, -float("inf"))
    s_max = tl.maximum(tl.max(prev_score, axis=0), tl.max(cur_score, axis=0))
    exp_prev = tl.exp(prev_score - s_max[None, :])
    exp_cur = tl.exp(cur_score - s_max[None, :])
    s_sum = tl.sum(exp_prev, axis=0) + tl.sum(exp_cur, axis=0)
    compressed = (
        tl.sum(prev_kv * exp_prev, axis=0) + tl.sum(cur_kv * exp_cur, axis=0)
    ) / s_sum
    tl.store(
        out_kv_ptr + req * HEAD_DIM + hd_offs,
        compressed,
        mask=hd_mask,
    )


@auto_retry_triton_compilation
def decode_mtp_hca(
    kv_cat: torch.Tensor,
    score_cat: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    ape: torch.Tensor,
    out_kv: torch.Tensor | None,
    compressed_reqs: torch.Tensor | None,
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    ratio: int,
    q_len: int,
):
    bsz = start_positions.shape[0]
    if bsz == 0:
        return
    D = kv_cat.shape[1]
    state_rows = kv_state.shape[1]
    BLOCK_Q = triton.next_power_of_2(q_len)
    BLOCK_D = triton.next_power_of_2(D)
    n_compressed = 0 if compressed_reqs is None else compressed_reqs.shape[0]
    if n_compressed > 0:
        assert out_kv is not None
        _decode_mtp_hca_compress_kernel[(n_compressed,)](
            kv_cat,
            score_cat,
            kv_state,
            score_state,
            ape,
            out_kv,
            compressed_reqs,
            start_positions,
            cache_slots,
            state_s0=kv_state.stride(0),
            state_s1=kv_state.stride(1),
            RATIO=ratio,
            STATE_ROWS=state_rows,
            Q_LEN=q_len,
            D=D,
            BLOCK_D=BLOCK_D,
        )
    _decode_mtp_hca_update_kernel[(bsz,)](
        kv_cat,
        score_cat,
        kv_state,
        score_state,
        ape,
        start_positions,
        cache_slots,
        state_s0=kv_state.stride(0),
        state_s1=kv_state.stride(1),
        RATIO=ratio,
        STATE_ROWS=state_rows,
        Q_LEN=q_len,
        D=D,
        BLOCK_Q=BLOCK_Q,
        BLOCK_D=BLOCK_D,
    )


@auto_retry_triton_compilation
def decode_mtp_csa(
    kv_cat: torch.Tensor,
    score_cat: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    ape: torch.Tensor,
    out_kv: torch.Tensor,
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    ratio: int,
    q_len: int,
    head_dim: int,
):
    bsz = start_positions.shape[0]
    if bsz == 0:
        return
    full_d = kv_cat.shape[1]
    state_rows = kv_state.shape[1]
    BLOCK_Q = triton.next_power_of_2(q_len)
    BLOCK_HD = triton.next_power_of_2(head_dim)
    BLOCK_FULL_D = triton.next_power_of_2(full_d)
    _decode_mtp_csa_kernel[(bsz,)](
        kv_cat,
        score_cat,
        kv_state,
        score_state,
        ape,
        out_kv,
        start_positions,
        cache_slots,
        state_s0=kv_state.stride(0),
        state_s1=kv_state.stride(1),
        RATIO=ratio,
        STATE_ROWS=state_rows,
        Q_LEN=q_len,
        HEAD_DIM=head_dim,
        FULL_D=full_d,
        BLOCK_Q=BLOCK_Q,
        BLOCK_HD=BLOCK_HD,
        BLOCK_FULL_D=BLOCK_FULL_D,
    )


# ---------------------------------------------------------------------------
# Classic single-token HCA decode fast path
#
# This is the hot DeepSeek-V4 HCA decode path when mtp_size == 1. It keeps the
# batch shape fixed for CUDA graph capture, but follows the SGLang-style split:
# every request writes pending state; only compression-boundary requests continue
# into the expensive 128-row reduction.
# ---------------------------------------------------------------------------


@triton.jit
def _decode_hca_kernel(
    kv_ptr,  # [bsz, 1, D]
    score_ptr,  # [bsz, 1, D]
    kv_state_ptr,  # [max_batch, RATIO, D]
    score_state_ptr,
    ape_ptr,  # [RATIO, D]
    out_kv_ptr,  # [bsz, D]
    start_positions_ptr,  # [bsz]
    cache_slots_ptr,  # [bsz]
    state_s0: tl.constexpr,
    state_s1: tl.constexpr,
    RATIO: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    req = tl.program_id(0)
    d_blk = tl.program_id(1)
    sp = tl.load(start_positions_ptr + req)
    slot = tl.load(cache_slots_ptr + req)

    d_offs = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    ape_row = sp % RATIO
    dst_row = sp % STATE_ROWS

    kv = tl.load(kv_ptr + req * D + d_offs, mask=d_mask, other=0.0)
    score = tl.load(score_ptr + req * D + d_offs, mask=d_mask, other=0.0)
    ape = tl.load(ape_ptr + ape_row * D + d_offs, mask=d_mask, other=0.0)
    score_with_ape = score + ape

    tl.store(
        kv_state_ptr + slot * state_s0 + dst_row * state_s1 + d_offs,
        kv,
        mask=d_mask,
    )
    tl.store(
        score_state_ptr + slot * state_s0 + dst_row * state_s1 + d_offs,
        score_with_ape,
        mask=d_mask,
    )

    if not _decode_should_compress(sp, RATIO, 1):
        tl.store(
            out_kv_ptr + req * D + d_offs,
            tl.zeros((BLOCK_D,), tl.float32),
            mask=d_mask,
        )
        return

    group_start = sp - (RATIO - 1)
    r_offs = tl.arange(0, RATIO)
    rows = (group_start + r_offs) % STATE_ROWS

    state_kv = tl.load(
        kv_state_ptr + slot * state_s0 + rows[:, None] * state_s1 + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    )
    state_score = tl.load(
        score_state_ptr + slot * state_s0 + rows[:, None] * state_s1 + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    )

    score_max = tl.max(state_score, axis=0)
    score_exp = tl.exp(state_score - score_max[None, :])
    score_sum = tl.sum(score_exp, axis=0)
    compressed = tl.sum(state_kv * score_exp, axis=0) / score_sum

    tl.store(out_kv_ptr + req * D + d_offs, compressed, mask=d_mask)


@auto_retry_triton_compilation
def decode_hca(
    kv: torch.Tensor,
    score: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    ape: torch.Tensor,
    out_kv: torch.Tensor,
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    ratio: int,
):
    bsz = start_positions.shape[0]
    if bsz == 0:
        return
    D = out_kv.shape[1]
    BLOCK_D = min(64, triton.next_power_of_2(D))
    grid = (bsz, triton.cdiv(D, BLOCK_D))
    _decode_hca_kernel[grid](
        kv,
        score,
        kv_state,
        score_state,
        ape,
        out_kv,
        start_positions,
        cache_slots,
        state_s0=kv_state.stride(0),
        state_s1=kv_state.stride(1),
        RATIO=ratio,
        STATE_ROWS=kv_state.shape[1],
        D=D,
        BLOCK_D=BLOCK_D,
    )


@triton.jit
def _postprocess_write_kv_cache_kernel(
    kv_ptr,  # [bsz, D] fp32 compressed KV
    norm_weight_ptr,  # [D]
    freqs_real_ptr,
    freqs_imag_ptr,
    kv_cache_ptr,  # dense [slots, len, D] or paged [blocks, page, D]
    block_table_ptr,
    start_positions_ptr,
    cache_slots_ptr,
    cache_seq_ids_ptr,
    norm_eps: tl.constexpr,
    freqs_stride0: tl.constexpr,
    freqs_stride1: tl.constexpr,
    kv_stride0: tl.constexpr,
    kv_stride1: tl.constexpr,
    kv_stride2: tl.constexpr,
    block_table_stride0: tl.constexpr,
    page_size: tl.constexpr,
    RATIO: tl.constexpr,
    Q_LEN: tl.constexpr,
    D: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGED: tl.constexpr,
    USE_HADAMARD: tl.constexpr,
):
    req = tl.program_id(0)
    sp = tl.load(start_positions_ptr + req)
    if not _decode_should_compress(sp, RATIO, Q_LEN):
        return

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    kv = (
        tl.load(kv_ptr + req * D + offs_d, mask=d_mask, other=0.0)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    rms = tl.sqrt(tl.sum(kv * kv, axis=0) / D + norm_eps)
    inv_rms = 1.0 / rms
    weight = tl.load(norm_weight_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    kv_norm = (kv * inv_rms * weight).to(tl.bfloat16).to(tl.float32)

    rotary_begin: tl.constexpr = D - ROPE_DIM
    rope_local = offs_d - rotary_begin
    pair_idx = rope_local // 2
    even_idx = rotary_begin + pair_idx * 2
    odd_idx = even_idx + 1

    rope_pos = sp - sp % RATIO
    freqs_off = rope_pos * freqs_stride0 + pair_idx * freqs_stride1
    cos = tl.load(freqs_real_ptr + freqs_off, mask=offs_d >= rotary_begin, other=1.0)
    sin = tl.load(freqs_imag_ptr + freqs_off, mask=offs_d >= rotary_begin, other=0.0)

    kv_even = tl.load(
        kv_ptr + req * D + even_idx, mask=offs_d >= rotary_begin, other=0.0
    )
    kv_odd = tl.load(kv_ptr + req * D + odd_idx, mask=offs_d >= rotary_begin, other=0.0)
    weight_even = tl.load(
        norm_weight_ptr + even_idx, mask=offs_d >= rotary_begin, other=0.0
    ).to(tl.float32)
    weight_odd = tl.load(
        norm_weight_ptr + odd_idx, mask=offs_d >= rotary_begin, other=0.0
    ).to(tl.float32)
    q0 = kv_even.to(tl.bfloat16).to(tl.float32) * inv_rms * weight_even
    q1 = kv_odd.to(tl.bfloat16).to(tl.float32) * inv_rms * weight_odd
    q0 = q0.to(tl.bfloat16).to(tl.float32)
    q1 = q1.to(tl.bfloat16).to(tl.float32)
    rope_even = q0 * cos - q1 * sin
    rope_odd = q1 * cos + q0 * sin
    rope_val = tl.where(rope_local % 2 == 0, rope_even, rope_odd)
    values = tl.where(
        offs_d >= rotary_begin,
        rope_val.to(tl.bfloat16).to(tl.float32),
        kv_norm,
    )

    if USE_HADAMARD:
        values = _hadamard_128(values).to(tl.bfloat16)

    position = sp // RATIO
    if PAGED:
        seq_id = tl.load(cache_seq_ids_ptr + req).to(tl.int64)
        row0 = tl.load(
            block_table_ptr + seq_id * block_table_stride0 + position // page_size
        ).to(tl.int64)
        row1 = position % page_size
    else:
        row0 = tl.load(cache_slots_ptr + req).to(tl.int64)
        row1 = position

    tl.store(
        kv_cache_ptr + row0 * kv_stride0 + row1 * kv_stride1 + offs_d * kv_stride2,
        values,
        mask=d_mask,
    )


@triton.jit
def _hadamard_128_stage(x, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    half = tl.arange(0, 2)
    select_a = tl.reshape(tl.where(half == 0, 1.0, 0.0), (1, 2, 1))
    select_b = tl.reshape(tl.where(half == 1, 1.0, 0.0), (1, 2, 1))
    half = tl.reshape(half, (1, 2, 1))
    x = tl.reshape(x, (GROUPS, 2, STRIDE))
    a = tl.sum(x * select_a, axis=1)
    b = tl.sum(x * select_b, axis=1)
    x = tl.where(
        half == 0,
        a[:, None, :] + b[:, None, :],
        a[:, None, :] - b[:, None, :],
    )
    return tl.reshape(x, (128,))


@triton.jit
def _hadamard_128(x):
    x = _hadamard_128_stage(x, 64, 1)
    x = _hadamard_128_stage(x, 32, 2)
    x = _hadamard_128_stage(x, 16, 4)
    x = _hadamard_128_stage(x, 8, 8)
    x = _hadamard_128_stage(x, 4, 16)
    x = _hadamard_128_stage(x, 2, 32)
    x = _hadamard_128_stage(x, 1, 64)
    return x * (128.0**-0.5)


@auto_retry_triton_compilation
def postprocess_write_kv_cache_hadamard(
    kv_compress: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor | None,
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    cache_seq_ids: torch.Tensor,
    ratio: int,
    norm_eps: float,
    q_len: int = 1,
    rope_dim: int = 64,
):
    assert kv_cache.shape[-1] == kv_compress.shape[-1]
    assert kv_compress.shape[-1] == 128
    bsz = start_positions.shape[0]
    if bsz == 0:
        return
    paged = block_table is not None
    freqs_real = freqs_cis.real
    freqs_imag = freqs_cis.imag
    D = kv_compress.shape[-1]
    BLOCK_D = triton.next_power_of_2(D)
    _postprocess_write_kv_cache_kernel[(bsz,)](
        kv_compress,
        norm_weight,
        freqs_real,
        freqs_imag,
        kv_cache,
        block_table,
        start_positions,
        cache_slots,
        cache_seq_ids,
        norm_eps=norm_eps,
        freqs_stride0=freqs_real.stride(0),
        freqs_stride1=freqs_real.stride(1),
        kv_stride0=kv_cache.stride(0),
        kv_stride1=kv_cache.stride(1),
        kv_stride2=kv_cache.stride(2),
        block_table_stride0=block_table.stride(0) if paged else 0,
        page_size=kv_cache.shape[1],
        RATIO=ratio,
        Q_LEN=q_len,
        D=D,
        ROPE_DIM=rope_dim,
        BLOCK_D=BLOCK_D,
        PAGED=paged,
        USE_HADAMARD=True,
        num_warps=4,
    )


@auto_retry_triton_compilation
def postprocess_write_kv_cache(
    kv_compress: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor | None,
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    cache_seq_ids: torch.Tensor,
    ratio: int,
    norm_eps: float,
    q_len: int = 1,
    rope_dim: int = 64,
):
    assert kv_cache.shape[-1] == kv_compress.shape[-1]
    bsz = start_positions.shape[0]
    if bsz == 0:
        return
    paged = block_table is not None
    freqs_real = freqs_cis.real
    freqs_imag = freqs_cis.imag
    D = kv_compress.shape[-1]
    BLOCK_D = triton.next_power_of_2(D)
    _postprocess_write_kv_cache_kernel[(bsz,)](
        kv_compress,
        norm_weight,
        freqs_real,
        freqs_imag,
        kv_cache,
        block_table,
        start_positions,
        cache_slots,
        cache_seq_ids,
        norm_eps=norm_eps,
        freqs_stride0=freqs_real.stride(0),
        freqs_stride1=freqs_real.stride(1),
        kv_stride0=kv_cache.stride(0),
        kv_stride1=kv_cache.stride(1),
        kv_stride2=kv_cache.stride(2),
        block_table_stride0=block_table.stride(0) if paged else 0,
        page_size=kv_cache.shape[1],
        RATIO=ratio,
        Q_LEN=q_len,
        D=D,
        ROPE_DIM=rope_dim,
        BLOCK_D=BLOCK_D,
        PAGED=paged,
        USE_HADAMARD=False,
        num_warps=8,
    )


@triton.jit
def _postprocess_write_flashmla_kernel(
    kv_ptr,  # [bsz, D] fp32 compressed KV
    norm_weight_ptr,  # [D]
    freqs_real_ptr,  # [max_seq, ROPE_DIM // 2], view of complex freqs
    freqs_imag_ptr,
    kv_cache_ptr,  # [num_blocks, page_size, 584] uint8
    block_table_ptr,
    start_positions_ptr,  # [bsz]
    cache_seq_ids_ptr,  # [bsz]
    norm_eps: tl.constexpr,
    freqs_stride0: tl.constexpr,
    freqs_stride1: tl.constexpr,
    page_size: tl.constexpr,
    block_table_stride0: tl.constexpr,
    kv_block_stride: tl.constexpr,
    RATIO: tl.constexpr,
    Q_LEN: tl.constexpr,
    D: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    TOKEN_DATA_BYTES: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    req = tl.program_id(0)
    sp = tl.load(start_positions_ptr + req)
    if not _decode_should_compress(sp, RATIO, Q_LEN):
        return

    offs_d = tl.arange(0, D)
    kv = tl.load(kv_ptr + req * D + offs_d).to(tl.bfloat16).to(tl.float32)
    rms = tl.sqrt(tl.sum(kv * kv, axis=0) / D + norm_eps)
    inv_rms = 1.0 / rms
    weight = tl.load(norm_weight_ptr + offs_d).to(tl.float32)
    kv_norm = (kv * inv_rms * weight).to(tl.bfloat16).to(tl.float32)

    rotary_begin: tl.constexpr = D - ROPE_DIM
    rope_local = offs_d - rotary_begin
    pair_idx = rope_local // 2
    even_idx = rotary_begin + pair_idx * 2
    odd_idx = even_idx + 1

    rope_pos = sp - sp % RATIO
    freqs_off = rope_pos * freqs_stride0 + pair_idx * freqs_stride1
    cos = tl.load(freqs_real_ptr + freqs_off, mask=offs_d >= rotary_begin, other=1.0)
    sin = tl.load(freqs_imag_ptr + freqs_off, mask=offs_d >= rotary_begin, other=0.0)

    kv_even = tl.load(
        kv_ptr + req * D + even_idx, mask=offs_d >= rotary_begin, other=0.0
    )
    kv_odd = tl.load(kv_ptr + req * D + odd_idx, mask=offs_d >= rotary_begin, other=0.0)
    weight_even = tl.load(
        norm_weight_ptr + even_idx, mask=offs_d >= rotary_begin, other=0.0
    ).to(tl.float32)
    weight_odd = tl.load(
        norm_weight_ptr + odd_idx, mask=offs_d >= rotary_begin, other=0.0
    ).to(tl.float32)
    q0 = kv_even.to(tl.bfloat16).to(tl.float32) * inv_rms * weight_even
    q1 = kv_odd.to(tl.bfloat16).to(tl.float32) * inv_rms * weight_odd
    q0 = q0.to(tl.bfloat16).to(tl.float32)
    q1 = q1.to(tl.bfloat16).to(tl.float32)
    rope_even = q0 * cos - q1 * sin
    rope_odd = q1 * cos + q0 * sin
    rope_val = tl.where(rope_local % 2 == 0, rope_even, rope_odd)
    kv_pack = tl.where(
        offs_d >= rotary_begin,
        rope_val.to(tl.bfloat16).to(tl.float32),
        kv_norm,
    )

    position = sp // RATIO
    seq_id = tl.load(cache_seq_ids_ptr + req).to(tl.int64)
    block_idx = tl.load(
        block_table_ptr + seq_id * block_table_stride0 + position // page_size
    ).to(tl.int64)
    pos_in_block = position % page_size
    block_base = kv_cache_ptr + block_idx * kv_block_stride
    token_data_base = block_base + pos_in_block * TOKEN_DATA_BYTES
    scale_base = block_base + page_size * TOKEN_DATA_BYTES + pos_in_block * SCALE_DIM

    n_quant_blocks: tl.constexpr = D // QUANT_BLOCK
    n_nope_blocks: tl.constexpr = NOPE_DIM // QUANT_BLOCK
    quant_2d = tl.reshape(
        kv_pack.to(tl.bfloat16).to(tl.float32), (n_quant_blocks, QUANT_BLOCK)
    )
    block_absmax = tl.max(tl.abs(quant_2d), axis=1)
    block_absmax = tl.maximum(block_absmax, 1e-4)
    raw_scales = block_absmax / FP8_MAX
    exponents = tl.ceil(tl.log2(raw_scales))
    inv_scales = tl.exp2(-exponents)
    x_scaled = quant_2d * tl.reshape(inv_scales, (n_quant_blocks, 1))
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    x_uint8 = x_clamped.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    x_uint8_flat = tl.reshape(x_uint8, (D,))

    tl.store(token_data_base + offs_d, x_uint8_flat, mask=offs_d < NOPE_DIM)

    bf16_ptr = (token_data_base + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
    tl.store(
        bf16_ptr + tl.maximum(rope_local, 0),
        kv_pack.to(tl.bfloat16),
        mask=(offs_d >= NOPE_DIM) & (offs_d < NOPE_DIM + ROPE_DIM),
    )

    scale_idx = tl.arange(0, SCALE_DIM)
    encoded = exponents + 127.0
    encoded = tl.maximum(tl.minimum(encoded, 255.0), 0.0)
    tl.store(
        scale_base + scale_idx,
        encoded.to(tl.uint8),
        mask=scale_idx < n_nope_blocks,
    )
    tl.store(scale_base + n_nope_blocks, tl.zeros((), dtype=tl.uint8))


@auto_retry_triton_compilation
def postprocess_write_flashmla(
    kv_compress: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    start_positions: torch.Tensor,
    cache_seq_ids: torch.Tensor,
    ratio: int,
    norm_eps: float,
    q_len: int = 1,
):
    if kv_cache.dim() == 4:
        assert kv_cache.shape[-2] == 1
        kv_cache = kv_cache.squeeze(-2)
    assert kv_cache.dtype == torch.uint8 and kv_cache.shape[-1] == 584
    assert kv_compress.shape[-1] == 512
    bsz = start_positions.shape[0]
    if bsz == 0:
        return
    freqs_real = freqs_cis.real
    freqs_imag = freqs_cis.imag
    _postprocess_write_flashmla_kernel[(bsz,)](
        kv_compress,
        norm_weight,
        freqs_real,
        freqs_imag,
        kv_cache,
        block_table,
        start_positions,
        cache_seq_ids,
        norm_eps=norm_eps,
        freqs_stride0=freqs_real.stride(0),
        freqs_stride1=freqs_real.stride(1),
        page_size=kv_cache.shape[1],
        block_table_stride0=block_table.stride(0),
        kv_block_stride=kv_cache.stride(0),
        RATIO=ratio,
        Q_LEN=q_len,
        D=kv_compress.shape[-1],
        NOPE_DIM=448,
        ROPE_DIM=64,
        TOKEN_DATA_BYTES=576,
        QUANT_BLOCK=64,
        SCALE_DIM=8,
        FP8_MAX=448.0,
        num_warps=8,
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
