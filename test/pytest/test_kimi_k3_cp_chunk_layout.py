# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from chitu.attn_backend import RefAttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.cp_utils import (
    CPContextContiguous,
    build_contiguous_chunk_lengths,
    build_cp_contiguous_reorder_idx,
)


@pytest.fixture(autouse=True)
def _cpu_safe_create_tensor(monkeypatch):
    """Avoid pinned-memory H2D staging in CPU-only CI containers."""

    def _create_tensor(data, device, dtype=None, sync_free=True):
        return torch.tensor(data, dtype=dtype, device=device)

    monkeypatch.setattr("chitu.batched_seq_len.create_tensor", _create_tensor)


class _FakeCPGroup:
    def __init__(self, rank: int, gathered: torch.Tensor):
        self.rank_in_group = rank
        self.is_first_rank = rank == 0
        self.gathered = gathered

    def all_gather_into_tensor(self, out: torch.Tensor, inp: torch.Tensor) -> None:
        assert out.shape == self.gathered.shape
        out.copy_(self.gathered)


def _delta(old_lens, new_lens):
    return BatchedSeqLenDelta(
        list(old_lens),
        list(new_lens),
        device="cpu",
        use_prefix_lens_static_tensor=False,
        use_position_ids_static_tensor=False,
        use_seq_ids_static_tensor=False,
        use_delta_prefix_lens_static_tensor=False,
        use_delta_position_ids_static_tensor=False,
        use_delta_seq_ids_static_tensor=False,
    )


def _context(rank: int, total_tokens: int, gathered: torch.Tensor):
    ctx = CPContextContiguous(4, _FakeCPGroup(rank, gathered))
    ctx._step_active = True
    ctx._orig_num_tokens = total_tokens
    return ctx


def _contiguous_chunk_bounds(total_tokens: int, pcp_size: int, rank: int):
    return (rank * total_tokens) // pcp_size, ((rank + 1) * total_tokens) // pcp_size


def test_contiguous_bounds_partition_global_tokens():
    for total in (7, 10, 16):
        bounds = [_contiguous_chunk_bounds(total, 4, rank) for rank in range(4)]
        assert bounds[0][0] == 0
        assert bounds[-1][1] == total
        assert all(bounds[i][1] == bounds[i + 1][0] for i in range(3))
        lengths = [end - start for start, end in bounds]
        assert max(lengths) - min(lengths) <= 1
        assert sum(lengths) == total


def test_contiguous_chunk_lengths_match_bounds():
    assert build_contiguous_chunk_lengths(10, 4) == [2, 3, 2, 3]
    assert build_contiguous_chunk_lengths(7, 4) == [1, 2, 2, 2]


def test_contiguous_reorder_removes_per_rank_padding():
    lengths = [2, 3, 2, 3]
    received = torch.tensor([0, 1, 0, 2, 3, 4, 5, 6, 0, 7, 8, 9], dtype=torch.long)
    idx = build_cp_contiguous_reorder_idx(lengths, 3, received.device)
    assert torch.equal(received[idx], torch.arange(10))


def test_prepare_mla_chunk_partial_request_across_ranks():
    # Two requests: [0, 3) and [3, 10). Rank 1 owns global delta rows [2, 5).
    delta = _delta([10, 20], [13, 27])
    ctx = _context(1, 10, torch.empty(0))
    layout = ctx.prepare_mla_chunk(delta)

    assert (layout.start, layout.end) == (2, 5)
    assert layout.local_rows == 3
    assert torch.unique_consecutive(layout.cache_seq_ids).tolist() == [0, 1]

    assert layout.seq_len_delta.old.lens_list == [12, 20]
    assert layout.seq_len_delta.new.lens_list == [13, 22]
    assert layout.seq_len_delta.delta_lens_list == [1, 2]
    assert layout.seq_len_delta.delta_prefix_lens_list == [0, 1, 3]
    assert layout.seq_len_delta.new.prefix_lens_list == [0, 13, 35]

    expected_positions = torch.cat(
        [torch.arange(13, dtype=torch.int32), torch.arange(22, dtype=torch.int32)]
    )
    expected_seq_ids = torch.cat(
        [torch.zeros(13, dtype=torch.int32), torch.ones(22, dtype=torch.int32)]
    )
    assert torch.equal(layout.cache_positions, expected_positions)
    assert torch.equal(layout.cache_seq_ids, expected_seq_ids)


def test_prepare_mla_chunk_single_request_continuation():
    delta = _delta([100], [110])
    ctx = _context(1, 10, torch.empty(0))
    layout = ctx.prepare_mla_chunk(delta)

    assert (layout.start, layout.end) == (2, 5)
    assert layout.seq_len_delta.old.lens_list == [102]
    assert layout.seq_len_delta.new.lens_list == [105]
    assert torch.equal(
        layout.cache_positions,
        torch.arange(105, dtype=torch.int32),
    )


def test_allgather_ordered_payload_restores_global_order():
    total = 10
    lengths = build_contiguous_chunk_lengths(total, 4)
    expected_n_local = max(lengths)
    global_payload = torch.arange(total, dtype=torch.float32).view(-1, 1)

    padded = []
    for rank, length in enumerate(lengths):
        start, end = _contiguous_chunk_bounds(total, 4, rank)
        local = global_payload[start:end]
        pad = torch.zeros(expected_n_local - length, 1, dtype=local.dtype)
        padded.append(torch.cat([local, pad], dim=0))
    gathered = torch.cat(padded, dim=0)

    delta = _delta([0], [total])
    for rank in range(4):
        ctx = _context(rank, total, gathered)
        ctx.prepare_mla_chunk(delta)
        start, end = _contiguous_chunk_bounds(total, 4, rank)
        local = global_payload[start:end]
        result = ctx.allgather_ordered_payload(local)
        assert torch.equal(result, global_payload)


@pytest.mark.parametrize(
    "old_lens,delta_lens",
    [
        ([0], [10]),
        ([100], [10]),
        ([10, 20], [3, 7]),
        ([7, 0], [5, 9]),
    ],
)
def test_chunk_metadata_matches_full_causal_reference(old_lens, delta_lens):
    new_lens = [old + delta for old, delta in zip(old_lens, delta_lens)]
    delta = _delta(old_lens, new_lens)
    total_delta = sum(delta_lens)
    total_new = sum(new_lens)

    n_heads = 2
    k_dim = 8
    v_dim = 8
    torch.manual_seed(0)
    q = torch.randn(total_delta, n_heads, k_dim)
    k = torch.randn(total_new, n_heads, k_dim)
    v = torch.randn(total_new, n_heads, v_dim)
    softmax_scale = k_dim**-0.5
    backend = RefAttnBackend()

    ref = backend.prefill_ragged_qkvo(
        q, k, v, delta, causal=True, softmax_scale=softmax_scale
    )

    out = torch.empty_like(ref)
    new_prefix = delta.new.prefix_lens_list
    for rank in range(4):
        ctx = _context(rank, total_delta, torch.empty(0))
        layout = ctx.prepare_mla_chunk(delta)
        q_chunk = q[layout.start : layout.end]
        k_prefix = []
        v_prefix = []
        present_seq_ids = torch.unique_consecutive(layout.cache_seq_ids).tolist()
        for local_seq_idx, seq_id in enumerate(present_seq_ids):
            k_len = layout.seq_len_delta.new.lens_list[local_seq_idx]
            k_start = new_prefix[seq_id]
            k_prefix.append(k[k_start : k_start + k_len])
            v_prefix.append(v[k_start : k_start + k_len])
        out[layout.start : layout.end] = backend.prefill_ragged_qkvo(
            q_chunk,
            torch.cat(k_prefix, dim=0),
            torch.cat(v_prefix, dim=0),
            layout.seq_len_delta,
            causal=True,
            softmax_scale=softmax_scale,
        )

    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
