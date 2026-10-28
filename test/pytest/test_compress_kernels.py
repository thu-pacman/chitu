# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")

from chitu.ops.deepseek_compressor import (
    build_compress_metadata,
    compress_csa,
    compress_hca,
    gather_pending_and_new,
    masked_write_kv_cache,
    pack_prefill_kv,
    writeback_pending,
)

_COMPRESS_IMPLS = ["torch"]
if has_triton:
    _COMPRESS_IMPLS.append("triton")


@pytest.mark.parametrize("impl", _COMPRESS_IMPLS)
def test_hca_gather_compress_and_writeback(impl):
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
    start_positions_t = torch.tensor(start_poses, device=device, dtype=torch.long)
    cache_slots_t = torch.tensor(cache_slots, device=device, dtype=torch.int32)

    def run_flow(impl_name):
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
            start_positions_t,
            cache_slots_t,
            ratio,
            impl=impl_name,
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
            impl=impl_name,
        )
        kv_state_after = kv_state.clone()
        score_state_after = score_state.clone()
        writeback_pending(
            flat_kv,
            flat_score,
            kv_state_after,
            score_state_after,
            ape,
            meta["cu_eff"],
            meta["pending_lens"],
            start_positions_t,
            cache_slots_t,
            pending_offset=0,
            ratio=ratio,
            impl=impl_name,
        )
        return flat_kv, flat_score, out_kv, kv_state_after, score_state_after

    flat_kv, flat_score, out_kv, kv_state_after, score_state_after = run_flow(impl)
    (
        flat_kv_ref,
        flat_score_ref,
        out_kv_ref,
        kv_state_ref,
        score_state_ref,
    ) = run_flow("torch")

    # Gathered KV rows are verbatim copies, so they must match the reference
    # exactly. The score path subtracts `ape` from pending rows, hence the
    # tolerance there.
    assert torch.equal(flat_kv, flat_kv_ref)
    torch.testing.assert_close(flat_score, flat_score_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_kv, out_kv_ref, atol=1e-5, rtol=1e-5)
    # The writeback KV rows are verbatim copies of `flat_kv`; the score rows add
    # `ape` back, hence the tolerance there.
    assert torch.equal(kv_state_after, kv_state_ref)
    torch.testing.assert_close(score_state_after, score_state_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("impl", _COMPRESS_IMPLS)
def test_csa_compress(impl):
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
    start_positions_t = torch.tensor(start_poses, device=device, dtype=torch.long)
    cache_slots_t = torch.tensor(cache_slots, device=device, dtype=torch.int32)
    n_groups_list_t = torch.tensor(
        meta["n_groups_list"], device=device, dtype=torch.int32
    )

    def run_compress(impl_name):
        # Build flat buffer from the logical-position pending-state ring.
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
            start_positions_t,
            cache_slots_t,
            ratio,
            pending_offset=ratio,
            impl=impl_name,
        )
        out_kv = torch.zeros(
            meta["total_groups"], head_dim, device=device, dtype=torch.float32
        )
        compress_csa(
            flat_kv,
            flat_score,
            ape,
            kv_state,
            score_state,
            out_kv,
            meta["cu_eff"],
            meta["cu_groups"],
            meta["group_to_req"],
            meta["pending_lens"],
            start_positions_t,
            cache_slots_t,
            n_groups_list_t,
            ratio,
            head_dim,
            impl=impl_name,
        )
        return flat_kv, flat_score, out_kv

    flat_kv, flat_score, out_kv = run_compress(impl)
    flat_kv_ref, flat_score_ref, out_kv_ref = run_compress("torch")

    # Gathered KV rows are verbatim copies, so they must match the reference
    # exactly. The score path subtracts `ape` from pending rows, hence the
    # tolerance there.
    assert torch.equal(flat_kv, flat_kv_ref)
    torch.testing.assert_close(flat_score, flat_score_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_kv, out_kv_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("impl", _COMPRESS_IMPLS)
def test_csa_writeback(impl):
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
    start_positions_t = torch.tensor(start_poses, device=device, dtype=torch.long)
    cache_slots_t = torch.tensor(cache_slots, device=device, dtype=torch.int32)

    def run_writeback(impl_name):
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
            start_positions_t,
            cache_slots_t,
            ratio,
            pending_offset=ratio,
            impl=impl_name,
        )
        kv_state_after = kv_state.clone()
        score_state_after = score_state.clone()
        writeback_pending(
            flat_kv,
            flat_score,
            kv_state_after,
            score_state_after,
            ape,
            meta["cu_eff"],
            meta["pending_lens"],
            start_positions_t,
            cache_slots_t,
            pending_offset=ratio,
            ratio=ratio,
            impl=impl_name,
        )
        return kv_state_after, score_state_after

    kv_state_after, score_state_after = run_writeback(impl)
    kv_state_ref, score_state_ref = run_writeback("torch")

    # The writeback KV rows are verbatim copies of `flat_kv`; the score rows add
    # `ape` back, hence the tolerance there.
    assert torch.equal(kv_state_after, kv_state_ref)
    torch.testing.assert_close(score_state_after, score_state_ref, atol=1e-5, rtol=1e-5)


def _exclusive_offsets(lengths: torch.Tensor) -> torch.Tensor:
    offsets = torch.empty(
        lengths.numel() + 1,
        device=lengths.device,
        dtype=lengths.dtype,
    )
    offsets[0] = 0
    offsets[1:] = torch.cumsum(lengths, dim=0, dtype=lengths.dtype)
    return offsets


@pytest.mark.parametrize("impl", _COMPRESS_IMPLS)
def test_pack_prefill_kv(impl):
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

    def run_pack(impl_name):
        # Unwritten rows keep their uninitialized garbage, so a row missed by an
        # implementation cannot match the reference by luck.
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
            impl=impl_name,
        )
        return out_kv

    out_kv = run_pack(impl)
    out_kv_ref = run_pack("torch")

    # The op only moves rows around, so the result must match exactly.
    assert torch.equal(out_kv, out_kv_ref)


@pytest.mark.parametrize("impl", _COMPRESS_IMPLS)
@pytest.mark.parametrize("paged", [False, True])
def test_masked_write_kv_cache(paged, impl):
    torch.manual_seed(311)
    device = "cuda"
    n = 6
    D = 5
    seq_len = 8
    page_size = 4

    values = torch.randn(n, D, device=device, dtype=torch.float32)
    cache_slots = torch.tensor([3, 1, 2, 0, 1, 3], device=device, dtype=torch.long)
    positions = torch.tensor([0, 3, 6, 2, 5, 7], device=device, dtype=torch.long)
    cache_seq_ids = torch.tensor([0, 1, 0, 1, 0, 1], device=device, dtype=torch.long)
    valid_mask = torch.tensor(
        [True, True, False, True, False, True], device=device, dtype=torch.bool
    )

    if paged:
        # Two sequences, two 4-token pages each.
        block_table = torch.tensor([[0, 1], [2, 3]], device=device, dtype=torch.long)
        kv_cache_shape = (4, page_size, D)
    else:
        block_table = None
        kv_cache_shape = (4, seq_len, D)

    def run_write(impl_name):
        # Rows that no valid token maps to keep the fill value, so the comparison
        # below also catches spurious writes.
        kv_cache = torch.full(kv_cache_shape, -7.0, device=device, dtype=torch.float32)
        masked_write_kv_cache(
            values,
            kv_cache,
            block_table,
            cache_slots,
            positions,
            cache_seq_ids,
            valid_mask,
            impl=impl_name,
        )
        return kv_cache

    kv_cache = run_write(impl)
    kv_cache_ref = run_write("torch")

    # The op only moves rows around, so the result must match exactly.
    assert torch.equal(kv_cache, kv_cache_ref)


@pytest.mark.parametrize("impl", _COMPRESS_IMPLS)
def test_masked_write_kv_cache_all_invalid_is_noop(impl):
    torch.manual_seed(313)
    device = "cuda"
    n = 4
    D = 3

    values = torch.randn(n, D, device=device, dtype=torch.float32)
    kv_cache = torch.randn(2, 8, D, device=device, dtype=torch.float32)
    cache_slots = torch.tensor([0, 1, 0, 1], device=device, dtype=torch.long)
    positions = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.long)
    cache_seq_ids = torch.arange(n, device=device, dtype=torch.long)
    valid_mask = torch.zeros(n, device=device, dtype=torch.bool)

    before = kv_cache.clone()
    masked_write_kv_cache(
        values,
        kv_cache,
        None,
        cache_slots,
        positions,
        cache_seq_ids,
        valid_mask,
        impl=impl,
    )
    assert torch.equal(kv_cache, before)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton,
    reason="auto-selection test requires CUDA and Triton",
)
def test_compress_ops_auto_select_triton_on_cuda():
    device = torch.device("cuda")

    def f32(*shape):
        return torch.empty(*shape, device=device, dtype=torch.float32)

    def i32(*shape):
        return torch.empty(*shape, device=device, dtype=torch.int32)

    def i64(*shape):
        return torch.empty(*shape, device=device, dtype=torch.long)

    assert (
        gather_pending_and_new.resolve_impl(
            f32(1, 4, 8),
            f32(1, 4, 8),
            f32(2, 8),
            f32(2, 8),
            f32(4, 8),
            f32(4, 8),
            f32(4, 8),
            i32(2),
            i32(2),
            i32(1),
            i64(1),
            i32(1),
            ratio=4,
        )
        == "triton"
    )
    assert (
        compress_hca.resolve_impl(
            f32(4, 8), f32(4, 8), f32(4, 8), f32(1, 8), i32(2), i32(2), i32(1), 4
        )
        == "triton"
    )
    assert (
        compress_csa.resolve_impl(
            f32(4, 8),
            f32(4, 8),
            f32(4, 8),
            f32(1, 4, 8),
            f32(1, 4, 8),
            f32(1, 4),
            i32(2),
            i32(2),
            i32(1),
            i32(1),
            i64(1),
            i32(1),
            i32(1),
            4,
            4,
        )
        == "triton"
    )
    assert (
        writeback_pending.resolve_impl(
            f32(4, 8),
            f32(4, 8),
            f32(1, 4, 8),
            f32(1, 4, 8),
            f32(4, 8),
            i32(2),
            i32(1),
            i64(1),
            i32(1),
            0,
            4,
        )
        == "triton"
    )
    assert (
        pack_prefill_kv.resolve_impl(
            f32(2, 8),
            f32(2, 8),
            f32(2, 8),
            f32(6, 8),
            i64(2),
            i64(2),
            i64(2),
            i64(2),
            i64(1),
            i64(1),
            i64(1),
        )
        == "triton"
    )
    assert (
        masked_write_kv_cache.resolve_impl(
            f32(2, 8),
            f32(4, 8, 8),
            None,
            i64(2),
            i64(2),
            i64(2),
            torch.ones(2, dtype=torch.bool, device=device),
        )
        == "triton"
    )
