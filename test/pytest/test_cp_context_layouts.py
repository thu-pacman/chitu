# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.cp_utils import (
    CPContext,
    CPContextContiguous,
    CPContextInterleaved,
    NoOpCPContext,
    create_cp_context,
)


@pytest.fixture(autouse=True)
def _cpu_safe_create_tensor(monkeypatch):
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


def _context(rank: int, gathered: torch.Tensor | None = None):
    if gathered is None:
        gathered = torch.empty(0)
    return CPContextInterleaved(4, _FakeCPGroup(rank, gathered))


def _freqs(total_tokens: int):
    return BatchedFreqsCis(
        torch.arange(total_tokens, dtype=torch.float32).view(-1, 1),
        torch.zeros(total_tokens, 1, dtype=torch.float32),
    )


def test_create_cp_context_layouts():
    assert isinstance(
        create_cp_context(4, _FakeCPGroup(0, torch.empty(0))), CPContextInterleaved
    )
    assert isinstance(
        create_cp_context(4, _FakeCPGroup(0, torch.empty(0)), "contiguous"),
        CPContextContiguous,
    )
    assert isinstance(create_cp_context(1, None), NoOpCPContext)
    with pytest.raises(ValueError, match="Unsupported pcp_layout"):
        create_cp_context(4, _FakeCPGroup(0, torch.empty(0)), "zigzag")


def test_interleaved_is_a_cp_context():
    ctx = _context(0)
    assert isinstance(ctx, CPContext)
    assert ctx.layout == "interleaved"


def test_interleaved_split_prefill_uses_interleaved_rows():
    ctx = _context(1)
    tokens = torch.arange(10)
    tokens, _, freqs = ctx.split_prefill(tokens, None, _freqs(10), 10)

    assert ctx.step_active
    assert ctx.n_local == 3
    assert ctx.expected_n_local == 3
    assert tokens.tolist() == [1, 5, 9]
    assert freqs.cos[:, 0].tolist() == [1.0, 5.0, 9.0]


def test_interleaved_pp_counts():
    ctx = _context(2)
    assert ctx.compute_pp_num_tokens(10) == 2
    assert ctx.compute_pp_num_tokens(9) == 2
    # Below pcp_size there is no split, so the full token count is forwarded.
    assert ctx.compute_pp_num_tokens(2) == 2


def test_interleaved_local_indices_and_slice():
    ctx = _context(1)
    indices, mask = ctx.local_flat_indices(3, 10, torch.device("cpu"))
    assert indices.tolist() == [1, 5, 9]
    assert mask.tolist() == [True, True, True]
    tensor = torch.arange(10)
    assert ctx.slice_for_local(tensor, 3, 10).tolist() == [1, 5, 9]


def test_interleaved_allgather_hidden_states_global_order():
    # T=10, W=4: rank0=[0,4,8], rank1=[1,5,9], rank2=[2,6,pad],
    # rank3=[3,7,pad]. The gathered buffer is rank-concatenated.
    gathered = torch.tensor([0, 4, 8, 1, 5, 9, 2, 6, -1, 3, 7, -1]).view(12, 1)
    ctx = _context(1, gathered)
    ctx.split_prefill(torch.arange(10), None, _freqs(10), 10)

    local = torch.tensor([[1], [5], [9]])
    out = ctx.allgather_hidden_states(local, None, None)

    assert out.flatten().tolist() == list(range(10))


def test_contiguous_split_prefill_uses_contiguous_intervals():
    ctx = CPContextContiguous(4, _FakeCPGroup(1, torch.empty(0)))
    tokens = torch.arange(10)
    tokens, _, freqs = ctx.split_prefill(tokens, None, _freqs(10), 10)

    assert ctx.step_active
    assert ctx.n_local == 3
    assert ctx.expected_n_local == 3
    assert tokens.tolist() == [2, 3, 4]
    assert freqs.cos[:, 0].tolist() == [2.0, 3.0, 4.0]


def test_contiguous_pp_counts_match_intervals():
    ctx = CPContextContiguous(4, _FakeCPGroup(2, torch.empty(0)))
    assert ctx.compute_pp_num_tokens(10) == 2
    assert ctx.compute_pp_num_tokens(9) == 2
    # Below pcp_size there is no split, so the full token count is forwarded.
    assert ctx.compute_pp_num_tokens(2) == 2


def test_contiguous_local_indices_and_slice():
    ctx = CPContextContiguous(4, _FakeCPGroup(1, torch.empty(0)))
    indices, mask = ctx.local_flat_indices(3, 10, torch.device("cpu"))
    assert indices.tolist() == [2, 3, 4]
    assert mask.tolist() == [True, True, True]
    tensor = torch.arange(10)
    assert ctx.slice_for_local(tensor, 3, 10).tolist() == [2, 3, 4]


def test_contiguous_allgather_hidden_states_global_order():
    # T=10, W=4 -> chunk lengths [2, 3, 2, 3], expected_n_local=3.
    # rank1 owns rows 2,3,4; gathered buffer is [rank0 padded, rank1, ...].
    gathered = torch.tensor([0, 1, -1, 2, 3, 4, 5, 6, -1, 7, 8, 9]).view(12, 1)
    ctx = CPContextContiguous(4, _FakeCPGroup(1, gathered))
    ctx.split_prefill(torch.arange(10), None, _freqs(10), 10)

    local = torch.tensor([[2], [3], [4]])
    out = ctx.allgather_hidden_states(local, None, None)

    assert out.flatten().tolist() == list(range(10))


def test_interleaved_position_owner_and_local_index():
    ctx = CPContextInterleaved(4, _FakeCPGroup(0, torch.empty(0)))
    positions = torch.arange(9)
    owner, local = ctx.position_owner_and_local_index(positions, 9)
    assert owner.tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0]
    assert local.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2]


def test_contiguous_position_owner_and_local_index():
    ctx = CPContextContiguous(4, _FakeCPGroup(0, torch.empty(0)))
    positions = torch.arange(9)
    owner, local = ctx.position_owner_and_local_index(positions, 9)
    # T=9, W=4 -> intervals [0,2), [2,4), [4,6), [6,9)
    assert owner.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 3]
    assert local.tolist() == [0, 1, 0, 1, 0, 1, 0, 1, 2]


def test_noop_context_is_identity():
    ctx = NoOpCPContext()
    tokens = torch.arange(10)
    out_tokens, _, out_freqs = ctx.split_prefill(tokens, None, _freqs(10), 10)
    assert out_tokens is tokens
    assert out_freqs.cos[:, 0].tolist() == list(range(10))
    assert ctx.allgather_hidden_states(tokens, None, None) is tokens
    assert ctx.compute_pp_num_tokens(10) == 10
