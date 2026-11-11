# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

# T: query rows, C: DCP ranks, H: heads per TP rank, K: selected tokens.
# W: local score width, B: requests, P: page-table width, S: tokens per page.


@triton.jit
def unpack_dcp_q_kernel(
    q_ptr,  # [C,T,H,DN+DP]
    nope_ptr,  # [T,C*H,DN]
    pe_ptr,  # [T,C*H,DP]
    C: tl.constexpr,
    num_query_rows,
    H: tl.constexpr,
    DN: tl.constexpr,
    DP: tl.constexpr,
    stride_c,  # T*H*(DN+DP) for contiguous input
    stride_t: tl.constexpr,
    stride_h: tl.constexpr,
    stride_d: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = i % (DN + DP)
    head_row = i // (DN + DP)  # flattened [T,C,Hl], the output order
    h = head_row % H
    c = head_row // H % C
    t = head_row // (C * H)
    value = tl.load(
        q_ptr + c * stride_c + t * stride_t + h * stride_h + d * stride_d,
        t < num_query_rows,
        other=0,
    )
    # Write both Q components directly, without a concatenated intermediate.
    tl.store(nope_ptr + head_row * DN + d, value, (t < num_query_rows) & (d < DN))
    tl.store(pe_ptr + head_row * DP + d - DN, value, (t < num_query_rows) & (d >= DN))


def unpack_dcp_q_triton(
    gathered,  # [C,T,H,DN+DP]
    nope_width,
):
    c, t, h, d = gathered.shape
    pe_width = d - nope_width
    nope = gathered.new_empty((t, c * h, nope_width))
    pe = gathered.new_empty((t, c * h, pe_width))
    if gathered.numel():
        unpack_dcp_q_kernel[(triton.cdiv(gathered.numel(), 1024),)](
            gathered,
            nope,
            pe,
            c,
            t,
            h,
            nope_width,
            pe_width,
            *gathered.stride(),
            BLOCK=1024,
            num_warps=4,
        )
    return nope, pe


@triton.jit
def pack_dcp_partials_kernel(
    o_ptr,  # BF16 [T,C*H,D]
    lse_ptr,  # FP32 [T,C*H]
    wire_ptr,  # BF16 [C,T,H,D+2], two words hold one FP32 LSE
    C: tl.constexpr,
    num_query_rows,
    H: tl.constexpr,
    D: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_o_d: tl.constexpr,
    stride_lse_t: tl.constexpr,
    stride_lse_h: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = i % D
    head_row = i // D  # flattened destination-major [C,T,Hl]
    h = head_row % H
    t = head_row // H % num_query_rows
    c = head_row // (num_query_rows * H)
    value = tl.load(
        o_ptr + t * stride_o_t + (c * H + h) * stride_o_h + d * stride_o_d,
        c < C,
        other=0,
    )
    wire_row = head_row * (D + 2)
    tl.store(wire_ptr + wire_row + d, value, c < C)
    # Exactly one lane per head stores the FP32 LSE as two raw 16-bit words.
    lse_mask = (c < C) & (d == 0)
    lse = tl.load(
        lse_ptr + t * stride_lse_t + (c * H + h) * stride_lse_h, lse_mask, other=0
    )
    lse_bits = lse.to(tl.uint32, bitcast=True)
    low = (lse_bits & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    high = (lse_bits >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    tl.store(wire_ptr + wire_row + D, low, lse_mask)
    tl.store(wire_ptr + wire_row + D + 1, high, lse_mask)


def pack_dcp_partials_triton(
    partial_o,  # BF16 [T,C*H,D]
    partial_lse,  # FP32 [T,C*H]
    group_size,
):
    t, heads, d = partial_o.shape
    h = heads // group_size
    wire = partial_o.new_empty((group_size, t, h, d + 2))
    if partial_o.numel():
        pack_dcp_partials_kernel[(triton.cdiv(partial_o.numel(), 1024),)](
            partial_o,
            partial_lse,
            wire,
            group_size,
            t,
            h,
            d,
            *partial_o.stride(),
            *partial_lse.stride(),
            BLOCK=1024,
            num_warps=4,
        )
    return wire


@triton.jit
def build_dcp_local_metadata_kernel(
    scores,  # FP32 [T,W]
    selected,  # [T,K], local token indices
    lengths,  # [T], visible local token counts
    wire,  # INT32 [T,K,2], score bits and global token ID
    stride_scores_t: tl.constexpr,
    stride_scores_w: tl.constexpr,
    stride_selected_t: tl.constexpr,
    stride_selected_k: tl.constexpr,
    stride_lengths: tl.constexpr,
    K: tl.constexpr,
    num_elements,
    PAGE_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // K, i % K
    slot = tl.load(
        selected + row * stride_selected_t + col * stride_selected_k,
        i < num_elements,
        other=-1,
    ).to(tl.int64)
    length = tl.load(lengths + row * stride_lengths, i < num_elements, other=0)
    valid = (i < num_elements) & (slot >= 0) & (slot < length)
    score = tl.load(
        scores + row * stride_scores_t + slot * stride_scores_w,
        valid,
        other=-float("inf"),
    )
    valid &= score == score
    gid = (slot // PAGE_SIZE * SIZE + RANK) * PAGE_SIZE + slot % PAGE_SIZE
    bits = tl.where(valid, score.to(tl.int32, bitcast=True), -8388608)  # FP32 -inf.
    tl.store(wire + 2 * i, bits, i < num_elements)
    tl.store(wire + 2 * i + 1, tl.where(valid, gid, -1).to(tl.int32), i < num_elements)


def build_dcp_local_metadata_triton(
    scores,  # FP32 [T,W]
    selected,  # [T,K]
    lengths,  # [T]
    page_size,
    rank,
    group_size,
):
    wire = torch.empty((*selected.shape, 2), dtype=torch.int32, device=scores.device)
    if selected.numel():
        build_dcp_local_metadata_kernel[(triton.cdiv(selected.numel(), 256),)](
            scores,
            selected,
            lengths,
            wire,
            scores.stride(0),
            scores.stride(1),
            selected.stride(0),
            selected.stride(1),
            lengths.stride(0),
            selected.shape[1],
            selected.numel(),
            page_size,
            rank,
            group_size,
            256,
            num_warps=4,
        )
    return wire


@triton.jit
def dcp_global_topk_kernel(
    metadata_ptr,  # INT32 [C,T,LOCAL_K,2], score bits and global token ID
    out_ptr,  # INT32 [T,K], selected global token IDs
    C: tl.constexpr,
    num_query_rows,
    LOCAL_K: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    c, slot = col // LOCAL_K, col % LOCAL_K
    source = ((c * num_query_rows + row) * LOCAL_K + slot) * 2
    bits = tl.load(metadata_ptr + source, col < C * LOCAL_K, other=0)
    gid = tl.load(metadata_ptr + source + 1, col < C * LOCAL_K, other=-1)
    # Signed high words preserve score order; low words break ties by token ID.
    magnitude = bits & 0x7FFFFFFF
    bits = tl.where(magnitude == 0, 0, bits)
    score_key = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)
    key = (score_key << 32) | ((~gid).to(tl.uint32).to(tl.int64))
    valid = (gid >= 0) & (magnitude <= 0x7F800000)
    keys = key.to(tl.uint64) ^ 0x8000000000000000
    selected = valid
    if tl.sum(valid.to(tl.int32)) > K:
        # Refine the K-th key one byte at a time, using a 256-bin histogram.
        prefix = tl.full((), 0, tl.uint64)
        threshold = tl.full((), 0, tl.uint64)
        remaining = tl.full((), K, tl.int32)
        bins = tl.arange(0, 256)
        for shift in tl.static_range(56, -1, -8):
            if remaining > 0:
                if shift == 56:
                    in_prefix = valid
                else:
                    in_prefix = valid & ((keys >> (shift + 8)) == prefix)
                digits = ((keys >> shift) & 0xFF).to(tl.int32)
                counts = tl.histogram(tl.where(in_prefix, digits, 0), 256)
                # Exclude inactive lanes from bin zero after counting.
                inactive_count = tl.sum((~in_prefix).to(tl.int32))
                counts -= tl.where(bins == 0, inactive_count, 0)
                above = tl.sum(counts) - tl.cumsum(counts)
                boundary = (above < remaining) & (above + counts >= remaining)
                digit = tl.sum(tl.where(boundary, bins, 0))
                remaining -= tl.sum(tl.where(bins > digit, counts, 0))
                prefix = (prefix << 8) | digit.to(tl.uint64)
                # Stop when the entire boundary bin fits in the remaining quota.
                if tl.sum(tl.where(bins == digit, counts, 0)) == remaining:
                    threshold = prefix << shift
                    remaining = 0
        selected = valid & (keys >= threshold)
    offsets = tl.cumsum(selected.to(tl.int32)) - 1
    tl.store(out_ptr + row * K + col, -1, col < K)
    tl.debug_barrier()
    tl.store(out_ptr + row * K + offsets, gid, selected)


def dcp_global_topk_triton(
    gathered_metadata,  # INT32 [C,T,LOCAL_K,2]
    k,
):
    if gathered_metadata.dtype != torch.int32 or gathered_metadata.shape[-1] != 2:
        raise ValueError("Decode-CP metadata must be int32 [C,T,K,2]")
    gathered_metadata = gathered_metadata.contiguous()
    c, rows, local_k, _ = gathered_metadata.shape
    ids = torch.empty((rows, k), dtype=torch.int32, device=gathered_metadata.device)
    if ids.numel():
        dcp_global_topk_kernel[(rows,)](
            gathered_metadata,
            ids,
            C=c,
            num_query_rows=rows,
            LOCAL_K=local_k,
            K=k,
            BLOCK=triton.next_power_of_2(c * local_k),
            num_warps=8,
        )
    return ids


@triton.jit
def compact_dcp_indices_kernel(
    ids_ptr,  # INT32 [T,K], global token IDs
    page_ptr,  # INT32 [B,P], local physical page IDs
    seq_ptr,  # [T] | None, query row to request index
    out_ptr,  # INT32 [T,K], compact local physical token indices
    counts_ptr,  # INT32 [T], compact prefix lengths
    K: tl.constexpr,
    table_width: tl.constexpr,
    S: tl.constexpr,
    RANK: tl.constexpr,
    C: tl.constexpr,
    num_local_pages: tl.constexpr,
    stride_ids_t: tl.constexpr,
    stride_ids_k: tl.constexpr,
    stride_page_b: tl.constexpr,
    stride_page_p: tl.constexpr,
    stride_seq: tl.constexpr,
    HAS_SEQ: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK_K)
    request = tl.load(seq_ptr + row * stride_seq) if HAS_SEQ else row
    gid = tl.load(ids_ptr + row * stride_ids_t + j * stride_ids_k, j < K, other=-1)
    sequence_page_index = gid // S
    group_index = sequence_page_index // C
    owner = sequence_page_index % C
    mine = (j < K) & (gid >= 0) & (owner == RANK) & (group_index < table_width)
    physical = tl.load(
        page_ptr
        + request.to(tl.int64) * stride_page_b
        + group_index.to(tl.int64) * stride_page_p,
        mask=mine,
        other=-1,
    )
    keep = mine & (physical >= 0) & (physical < num_local_pages)
    slot = tl.cumsum(keep.to(tl.int32), axis=0) - 1
    count = tl.sum(keep.to(tl.int32), axis=0)
    # These stores target disjoint ranges: tail [count,K), compact [0,count).
    # Valid slots are unique, so no sink column, atomics or barrier is needed.
    tl.store(out_ptr + row * K + j, -1, mask=(j < K) & (j >= count))
    tl.store(out_ptr + row * K + slot, physical * S + gid % S, mask=keep)
    tl.store(counts_ptr + row, count)


def compact_dcp_indices_triton(
    global_ids,  # INT32 [T,K]
    page_table,  # INT32 [B,P]
    request_seq_ids,  # [T] | None
    block_size,
    rank,
    group_size,
    num_local_pages,
):
    """Map and compact [T,K] from [B,P], without copying the page table per query."""
    rows, k = global_ids.shape
    out = torch.empty((rows, k), dtype=torch.int32, device=global_ids.device)
    counts = (
        torch.zeros((rows,), dtype=torch.int32, device=global_ids.device)
        if k == 0
        else torch.empty((rows,), dtype=torch.int32, device=global_ids.device)
    )
    if rows and k:
        compact_dcp_indices_kernel[(rows,)](
            global_ids,
            page_table,
            request_seq_ids,
            out,
            counts,
            K=k,
            table_width=page_table.shape[1],
            S=block_size,
            RANK=rank,
            C=group_size,
            num_local_pages=num_local_pages,
            stride_ids_t=global_ids.stride(0),
            stride_ids_k=global_ids.stride(1),
            stride_page_b=page_table.stride(0),
            stride_page_p=page_table.stride(1),
            stride_seq=request_seq_ids.stride(0) if request_seq_ids is not None else 0,
            HAS_SEQ=request_seq_ids is not None,
            BLOCK_K=triton.next_power_of_2(k),
            num_warps=4,
        )
    return out, counts


@triton.jit
def merge_dcp_attention_partials_kernel(
    owner_wire_ptr,  # BF16 [C,R,D+2], R=T*Hl
    output_ptr,  # BF16 [R,D]
    num_rows,
    VALUE_WIDTH: tl.constexpr,
    OWNER_COUNT: tl.constexpr,
    BLOCK_OWNERS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    dims = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    owners = tl.arange(0, BLOCK_OWNERS)
    owner_mask = owners < OWNER_COUNT

    wire_row = (owners * num_rows + row) * (VALUE_WIDTH + 2)
    low = (
        tl.load(
            owner_wire_ptr + wire_row + VALUE_WIDTH,
            mask=owner_mask,
            other=0,
        )
        .to(tl.uint16, bitcast=True)
        .to(tl.uint32)
    )
    high = (
        tl.load(
            owner_wire_ptr + wire_row + VALUE_WIDTH + 1,
            mask=owner_mask,
            other=0,
        )
        .to(tl.uint16, bitcast=True)
        .to(tl.uint32)
    )
    lse = (low | (high << 16)).to(tl.float32, bitcast=True)
    finite = owner_mask & (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
    masked_lse = tl.where(finite, lse, -float("inf"))
    max_lse = tl.max(masked_lse, axis=0)
    has_any = tl.max(finite.to(tl.int32), axis=0) != 0
    safe_max = tl.where(has_any, max_lse, 0.0)
    weights = tl.where(finite, tl.exp(masked_lse - safe_max), 0.0)

    # One program merges one [T,Hl] row and a Dv tile. The LSE reduction,
    # output weighting and owner reduction stay in registers; no FP32
    # [C,T,Hl,Dv] temporary is materialized.
    values = tl.load(
        owner_wire_ptr + wire_row[:, None] + dims[None, :],
        mask=owner_mask[:, None] & (dims[None, :] < VALUE_WIDTH),
        other=0.0,
    ).to(tl.float32)
    values = tl.where(finite[:, None], values, 0.0)
    numerator = tl.sum(values * weights[:, None], axis=0)
    denominator = tl.sum(weights, axis=0)
    merged = tl.where(has_any, numerator / denominator, 0.0)
    tl.store(
        output_ptr + row * VALUE_WIDTH + dims,
        merged,
        mask=dims < VALUE_WIDTH,
    )


def merge_dcp_attention_partials_triton(
    owner_wire: torch.Tensor,  # BF16 [C,T,Hl,Dv+2]
) -> torch.Tensor:  # BF16 [T,Hl,Dv]
    """Decode the one-wire payload and fuse distributed-softmax merge."""

    if owner_wire.ndim != 4 or owner_wire.shape[-1] < 3:
        raise ValueError("owner_wire must have shape [C,T,H,D+2]")
    if owner_wire.dtype != torch.bfloat16 or not owner_wire.is_contiguous():
        raise ValueError("Decode-CP owner wire must be contiguous BF16")

    owner_count, query_rows, local_heads, wire_width = owner_wire.shape
    value_width = wire_width - 2
    output = owner_wire.new_empty((query_rows, local_heads, value_width))
    rows = query_rows * local_heads
    if rows == 0:
        return output

    block_owners = triton.next_power_of_2(owner_count)
    block_d = min(128, triton.next_power_of_2(value_width))
    grid = (rows, triton.cdiv(value_width, block_d))
    merge_dcp_attention_partials_kernel[grid](
        owner_wire,
        output,
        num_rows=rows,
        VALUE_WIDTH=value_width,
        OWNER_COUNT=owner_count,
        BLOCK_OWNERS=block_owners,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return output
