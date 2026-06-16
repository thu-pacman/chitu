# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")

from chitu.ops.deepseek_compressor import (
    build_compress_metadata,
    gather_pending_and_new as gather_pending_and_new_torch,
    compress_hca as compress_hca_torch,
    compress_csa as compress_csa_torch,
    writeback_pending as writeback_pending_torch,
)

if has_triton:
    from chitu.ops.triton_ops.deepseek_compressor import (
        gather_pending_and_new as gather_pending_and_new_triton,
        compress_hca as compress_hca_triton,
        compress_csa as compress_csa_triton,
        writeback_pending as writeback_pending_triton,
        pack_prefill_kv,
    )

_COMPRESS_BACKENDS = [
    pytest.param(
        (
            "torch",
            gather_pending_and_new_torch,
            compress_hca_torch,
            compress_csa_torch,
            writeback_pending_torch,
        ),
        id="torch",
    )
]
if has_triton:
    _COMPRESS_BACKENDS.append(
        pytest.param(
            (
                "triton",
                gather_pending_and_new_triton,
                compress_hca_triton,
                compress_csa_triton,
                writeback_pending_triton,
            ),
            id="triton",
        )
    )

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton,
    reason="CUDA and Triton are required for these kernel tests",
)


@pytest.fixture(params=_COMPRESS_BACKENDS)
def compress_backend(request):
    return request.param


def _reference_gather_and_compress(
    kv_state,
    score_state,
    kv_cat,
    score_cat,
    ape,
    start_poses,
    seqlens,
    cache_slots,
    ratio,
):
    pending_lens = [sp % ratio for sp in start_poses]
    kv_list = kv_cat.split(seqlens, dim=0)
    score_list = score_cat.split(seqlens, dim=0)

    flat_kv_parts = []
    flat_score_parts = []
    out_parts = []
    remainders = []
    cutoffs = []

    for i in range(len(start_poses)):
        p = pending_lens[i]
        slot = cache_slots[i]
        kv = kv_list[i]
        score = score_list[i]
        if p > 0:
            pending_kv = kv_state[slot, :p]
            pending_score = score_state[slot, :p] - ape[:p]
            kv = torch.cat([pending_kv, kv], dim=0)
            score = torch.cat([pending_score, score], dim=0)
        flat_kv_parts.append(kv)
        flat_score_parts.append(score)

        remainder = kv.size(0) % ratio
        cutoff = kv.size(0) - remainder
        remainders.append(remainder)
        cutoffs.append(cutoff)
        if cutoff > 0:
            kv_cut = kv[:cutoff].unflatten(0, (-1, ratio))
            score_cut = score[:cutoff].unflatten(0, (-1, ratio)) + ape
            out = (kv_cut * score_cut.softmax(dim=1)).sum(dim=1)
            out_parts.append(out)
        else:
            out_parts.append(
                torch.empty(0, kv.shape[1], device=kv.device, dtype=kv.dtype)
            )

    flat_kv = torch.cat(flat_kv_parts, dim=0) if flat_kv_parts else kv_cat[:0]
    flat_score = (
        torch.cat(flat_score_parts, dim=0) if flat_score_parts else score_cat[:0]
    )
    return flat_kv, flat_score, out_parts, remainders, cutoffs


def test_hca_gather_compress_and_writeback_matches_reference(compress_backend):
    (
        _backend_name,
        gather_pending_and_new,
        compress_hca,
        _compress_csa,
        writeback_pending,
    ) = compress_backend
    torch.manual_seed(0)
    device = "cuda"
    ratio = 8
    head_dim = 16
    max_batch = 4
    max_pending_rows = ratio

    start_poses = [0, 3, 8, 15]
    seqlens = [5, 7, 4, 6]
    cache_slots = [1, 3, 0, 2]

    total_new = sum(seqlens)
    kv_state = torch.randn(
        max_batch, max_pending_rows, head_dim, device=device, dtype=torch.float32
    )
    score_state = torch.randn(
        max_batch, max_pending_rows, head_dim, device=device, dtype=torch.float32
    )
    kv_cat = torch.randn(total_new, head_dim, device=device, dtype=torch.float32)
    score_cat = torch.randn(total_new, head_dim, device=device, dtype=torch.float32)
    ape = torch.randn(ratio, head_dim, device=device, dtype=torch.float32)

    meta = build_compress_metadata(start_poses, seqlens, ratio, torch.device(device))
    cache_slots_t = torch.tensor(cache_slots, device=device, dtype=torch.int32)

    flat_kv = torch.empty(
        meta["total_eff"], head_dim, device=device, dtype=torch.float32
    )
    flat_score = torch.empty_like(flat_kv)
    out_kv = torch.empty(
        meta["total_groups"], head_dim, device=device, dtype=torch.float32
    )

    gather_pending_and_new(
        kv_state,
        score_state,
        kv_cat,
        score_cat,
        ape,
        flat_kv,
        flat_score,
        meta["cu_eff"],
        meta["cu_new"],
        meta["pending_lens"],
        cache_slots_t,
        ratio,
    )
    compress_hca(
        flat_kv,
        flat_score,
        ape,
        out_kv,
        meta["cu_eff"],
        meta["cu_groups"],
        meta["group_to_req"],
        ratio,
    )

    flat_kv_ref, flat_score_ref, out_parts_ref, remainders_ref, cutoffs_ref = (
        _reference_gather_and_compress(
            kv_state,
            score_state,
            kv_cat,
            score_cat,
            ape,
            start_poses,
            seqlens,
            cache_slots,
            ratio,
        )
    )
    out_ref = torch.cat(out_parts_ref, dim=0) if out_parts_ref else out_kv[:0]

    torch.testing.assert_close(flat_kv, flat_kv_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(flat_score, flat_score_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_kv, out_ref, atol=1e-5, rtol=1e-5)

    kv_state_after = kv_state.clone()
    score_state_after = score_state.clone()
    writeback_pending(
        flat_kv,
        flat_score,
        kv_state_after,
        score_state_after,
        ape,
        meta["cu_eff"],
        meta["cutoffs"],
        meta["remainders"],
        cache_slots_t,
        pending_offset=0,
        ratio=ratio,
    )

    for i, slot in enumerate(cache_slots):
        r = remainders_ref[i]
        c = cutoffs_ref[i]
        if r == 0:
            continue
        req_eff_start = int(meta["cu_eff"][i].item())
        src_kv = flat_kv_ref[req_eff_start + c : req_eff_start + c + r]
        src_score = flat_score_ref[req_eff_start + c : req_eff_start + c + r] + ape[:r]
        torch.testing.assert_close(
            kv_state_after[slot, :r], src_kv, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            score_state_after[slot, :r], src_score, atol=1e-5, rtol=1e-5
        )


def _reference_csa_compress(
    flat_kv,  # [total_eff, 2*head_dim]
    flat_score,  # [total_eff, 2*head_dim]
    ape,  # [ratio, head_dim]  first-half ape
    kv_state,  # [max_batch, coff*ratio, coff*head_dim]
    score_state,
    start_poses,
    seqlens,
    cache_slots,
    ratio,
    head_dim,
):
    """Reference implementation matching the triton CSA kernel logic."""
    pending_lens = [sp % ratio for sp in start_poses]
    effective_lens = [p + s for p, s in zip(pending_lens, seqlens)]
    cutoffs = [e - e % ratio for e in effective_lens]
    n_groups_list = [c // ratio for c in cutoffs]
    cu_eff_cpu = [0]
    for e in effective_lens:
        cu_eff_cpu.append(cu_eff_cpu[-1] + e)

    out_parts = []
    for i in range(len(start_poses)):
        slot = cache_slots[i]
        pending = pending_lens[i]
        n_groups = n_groups_list[i]
        eff_start = cu_eff_cpu[i]
        grps = []
        for grp in range(n_groups):
            token_start = eff_start + grp * ratio
            # current group second head
            cur_kv = flat_kv[token_start : token_start + ratio, head_dim:]
            cur_score = flat_score[token_start : token_start + ratio, head_dim:]
            # prev group first head
            if grp > 0:
                prev_start = token_start - ratio
                prev_kv = flat_kv[prev_start : prev_start + ratio, :head_dim]
                prev_score = flat_score[prev_start : prev_start + ratio, :head_dim]
            else:
                if pending > 0:
                    prev_kv = kv_state[
                        slot, :ratio, :head_dim
                    ]  # kv_state[slot, :ratio] = prev-group reference
                    prev_score = score_state[slot, :ratio, :head_dim] - ape
                else:
                    prev_kv = torch.zeros(ratio, head_dim, device=flat_kv.device)
                    prev_score = torch.full(
                        (ratio, head_dim), float("-inf"), device=flat_kv.device
                    )
            # concat and softmax over 2*ratio
            kv_all = torch.cat([prev_kv, cur_kv], dim=0)  # [2*ratio, head_dim]
            score_all = torch.cat([prev_score + ape, cur_score + ape], dim=0)
            weights = score_all.softmax(dim=0)
            compressed = (kv_all * weights).sum(dim=0)  # [head_dim]
            grps.append(compressed)
        if grps:
            out_parts.append(torch.stack(grps, dim=0))
    return (
        torch.cat(out_parts, dim=0)
        if out_parts
        else torch.zeros(0, head_dim, device=flat_kv.device)
    )


def test_csa_compress_matches_reference(compress_backend):
    (
        _backend_name,
        gather_pending_and_new,
        _compress_hca,
        compress_csa,
        _writeback_pending,
    ) = compress_backend
    torch.manual_seed(42)
    device = "cuda"
    ratio = 4
    head_dim = 16
    coff = 2
    D = coff * head_dim  # 32
    max_batch = 4
    max_pending_rows = coff * ratio  # 8

    # Mix of: no pending, partial pending, pending that completes a group
    start_poses = [0, 3, 8, 7]
    seqlens = [5, 7, 4, 9]
    cache_slots = [1, 3, 0, 2]

    total_new = sum(seqlens)
    kv_state = torch.randn(
        max_batch, max_pending_rows, D, device=device, dtype=torch.float32
    )
    score_state = torch.randn(
        max_batch, max_pending_rows, D, device=device, dtype=torch.float32
    )
    kv_cat = torch.randn(total_new, D, device=device, dtype=torch.float32)
    score_cat = torch.randn(total_new, D, device=device, dtype=torch.float32)
    ape = torch.randn(ratio, D, device=device, dtype=torch.float32)

    meta = build_compress_metadata(start_poses, seqlens, ratio, torch.device(device))
    cache_slots_t = torch.tensor(cache_slots, device=device, dtype=torch.int32)

    # Build flat buffer — CSA pending tokens live at kv_state[slot, ratio:ratio+pending]
    flat_kv = torch.empty(meta["total_eff"], D, device=device, dtype=torch.float32)
    flat_score = torch.empty_like(flat_kv)
    gather_pending_and_new(
        kv_state,
        score_state,
        kv_cat,
        score_cat,
        ape,
        flat_kv,
        flat_score,
        meta["cu_eff"],
        meta["cu_new"],
        meta["pending_lens"],
        cache_slots_t,
        ratio,
        pending_offset=ratio,  # CSA: pending tokens at offset ratio in kv_state
    )

    out_kv = torch.zeros(
        meta["total_groups"], head_dim, device=device, dtype=torch.float32
    )
    n_groups_list_t = torch.tensor(
        meta["n_groups_list"], device=device, dtype=torch.int32
    )
    ape_first = ape[:, :head_dim]
    # Snapshot kv_state before compress_csa, because _writeback_csa_prev_kernel
    # overwrites kv_state[slot, :ratio] in-place during the call.
    kv_state_snapshot = kv_state.clone()
    score_state_snapshot = score_state.clone()
    compress_csa(
        flat_kv,
        flat_score,
        ape_first,
        kv_state,
        score_state,
        out_kv,
        meta["cu_eff"],
        meta["cu_groups"],
        meta["group_to_req"],
        meta["pending_lens"],
        cache_slots_t,
        meta["cutoffs"],
        n_groups_list_t,
        ratio,
        head_dim,
    )

    # Reference uses pre-writeback kv_state
    out_ref = _reference_csa_compress(
        flat_kv,
        flat_score,
        ape_first,
        kv_state_snapshot,
        score_state_snapshot,
        start_poses,
        seqlens,
        cache_slots,
        ratio,
        head_dim,
    )

    torch.testing.assert_close(out_kv, out_ref, atol=1e-4, rtol=1e-4)


def test_csa_writeback_prev_group(compress_backend):
    """Verify that compress_csa correctly updates kv_state[:ratio] for next chunk."""
    (
        _backend_name,
        gather_pending_and_new,
        _compress_hca,
        compress_csa,
        _writeback_pending,
    ) = compress_backend
    torch.manual_seed(7)
    device = "cuda"
    ratio = 4
    head_dim = 16
    coff = 2
    D = coff * head_dim
    max_batch = 2
    max_pending_rows = coff * ratio

    start_poses = [
        8,
        0,
    ]  # req0: has a complete group; req1: no groups (effective_len < ratio)
    seqlens = [8, 3]
    cache_slots = [0, 1]

    kv_state = torch.zeros(
        max_batch, max_pending_rows, D, device=device, dtype=torch.float32
    )
    score_state = torch.zeros_like(kv_state)
    kv_cat = torch.randn(sum(seqlens), D, device=device, dtype=torch.float32)
    score_cat = torch.randn_like(kv_cat)
    ape = torch.randn(ratio, D, device=device, dtype=torch.float32)

    meta = build_compress_metadata(start_poses, seqlens, ratio, torch.device(device))
    cache_slots_t = torch.tensor(cache_slots, device=device, dtype=torch.int32)

    flat_kv = torch.empty(meta["total_eff"], D, device=device, dtype=torch.float32)
    flat_score = torch.empty_like(flat_kv)
    gather_pending_and_new(
        kv_state,
        score_state,
        kv_cat,
        score_cat,
        ape,
        flat_kv,
        flat_score,
        meta["cu_eff"],
        meta["cu_new"],
        meta["pending_lens"],
        cache_slots_t,
        ratio,
        pending_offset=ratio,
    )

    kv_state_after = kv_state.clone()
    score_state_after = score_state.clone()
    out_kv = torch.zeros(
        meta["total_groups"], head_dim, device=device, dtype=torch.float32
    )
    n_groups_list_t = torch.tensor(
        meta["n_groups_list"], device=device, dtype=torch.int32
    )
    ape_first = ape[:, :head_dim]
    compress_csa(
        flat_kv,
        flat_score,
        ape_first,
        kv_state_after,
        score_state_after,
        out_kv,
        meta["cu_eff"],
        meta["cu_groups"],
        meta["group_to_req"],
        meta["pending_lens"],
        cache_slots_t,
        meta["cutoffs"],
        n_groups_list_t,
        ratio,
        head_dim,
    )

    # req0: n_groups=4 (pending=0, seqlen=8 → eff=8 → 2 groups)
    # Wait — pending=8%4=0, eff=8, cutoff=8, n_groups=2
    # kv_state[slot=0, :ratio, :head_dim] should equal last group's first head
    ng0 = meta["n_groups_list"][0]
    slot0 = cache_slots[0]
    if ng0 > 0:
        eff_start0 = int(meta["cu_eff"][0].item())
        cutoff0 = int(meta["cutoffs"][0].item())
        last_grp_start = eff_start0 + cutoff0 - ratio
        expected_kv = flat_kv[last_grp_start : last_grp_start + ratio, :head_dim]
        expected_score = (
            flat_score[last_grp_start : last_grp_start + ratio, :head_dim] + ape_first
        )
        torch.testing.assert_close(
            kv_state_after[slot0, :ratio, :head_dim], expected_kv, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            score_state_after[slot0, :ratio, :head_dim],
            expected_score,
            atol=1e-5,
            rtol=1e-5,
        )

    # req1: n_groups=0, kv_state should be unchanged
    slot1 = cache_slots[1]
    torch.testing.assert_close(kv_state_after[slot1], kv_state[slot1], atol=0, rtol=0)


def _exclusive_offsets(lengths: torch.Tensor) -> torch.Tensor:
    offsets = torch.empty(
        lengths.numel() + 1,
        device=lengths.device,
        dtype=lengths.dtype,
    )
    offsets[0] = 0
    offsets[1:] = torch.cumsum(lengths, dim=0, dtype=lengths.dtype)
    return offsets


def test_pack_prefill_kv_matches_reference():
    torch.manual_seed(123)
    device = "cuda"
    D = 13

    history_lens = torch.tensor([0, 3, 2, 5], device=device, dtype=torch.long)
    current_lens = torch.tensor([2, 1, 4, 3], device=device, dtype=torch.long)
    compressed_lens = torch.tensor([1, 0, 3, 2], device=device, dtype=torch.long)

    history_offsets = _exclusive_offsets(history_lens)
    current_offsets = _exclusive_offsets(current_lens)
    compressed_offsets = _exclusive_offsets(compressed_lens)
    out_lens = history_lens + current_lens + compressed_lens
    out_offsets = _exclusive_offsets(out_lens)

    history_kv = torch.randn(
        int(history_offsets[-1].item()), D, device=device, dtype=torch.float32
    )
    current_kv = torch.randn(
        int(current_offsets[-1].item()), D, device=device, dtype=torch.float32
    )
    compressed_kv = torch.randn(
        int(compressed_offsets[-1].item()), D, device=device, dtype=torch.float32
    )
    out_kv = torch.empty(
        int(out_offsets[-1].item()), D, device=device, dtype=torch.float32
    )

    pack_prefill_kv(
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
    )

    ref_parts = []
    for i in range(history_lens.numel()):
        h0, h1 = int(history_offsets[i].item()), int(history_offsets[i + 1].item())
        c0, c1 = int(current_offsets[i].item()), int(current_offsets[i + 1].item())
        p0, p1 = int(compressed_offsets[i].item()), int(
            compressed_offsets[i + 1].item()
        )
        ref_parts.append(
            torch.cat(
                [
                    history_kv[h0:h1],
                    current_kv[c0:c1],
                    compressed_kv[p0:p1],
                ],
                dim=0,
            )
        )
    ref = torch.cat(ref_parts, dim=0)

    torch.testing.assert_close(out_kv, ref, atol=0, rtol=0)
