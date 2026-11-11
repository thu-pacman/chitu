# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
import pytest


import torch

import chitu.batched_seq_len as seq_len_module
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.dsa_indexer_backend import (
    DSAIndexer,
    get_indexer_class,
)
import chitu.dsa_indexer_backend.base as dsa_indexer_module
from chitu.dsa_indexer_backend.hygon_backend import support_indexer_hygon
from chitu.dsa_indexer_backend.triton_backend import has_triton
from chitu.kv_cache import GlobalLocalMap, PagedKVCache, PagedKVCacheAccessor


def _indexer_without_runtime_init(impl: str) -> DSAIndexer:
    indexer = object.__new__(get_indexer_class(impl))
    indexer.impl = impl
    indexer.static_max_n = 8192
    indexer.mtp_size = 1
    indexer.index_topk = 2048
    indexer._indexer_logits_chunk_bytes = None
    indexer.prefill_schedule = None
    return indexer


@pytest.fixture
def cpu_delta(monkeypatch):
    # Keep CPU metadata independent of CUDA allocation and graph state.
    def create_tensor(data, device, dtype=None, sync_free=True):
        return torch.tensor(data, device=device, dtype=dtype)

    monkeypatch.setattr(seq_len_module, "create_tensor", create_tensor)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    delta = BatchedSeqLenDelta(
        [0, 0], [1, 1], device="cpu", max_total_len=64, max_total_delta_len=64
    )
    delta.is_decode_stage = True
    return delta


class _FakeDecodeCPAllGatherGroup:
    """Simulate rank-major AllGather, with absent owners contributing no candidates."""

    def __init__(self, group_size, rank=0, metadata=None):
        self.group_size = group_size
        self.rank = rank
        self.metadata = metadata
        self.local_metadata = None

    def all_gather_into_tensor(self, output, input):
        self.local_metadata = input
        if self.metadata is not None:
            assert torch.equal(input, self.metadata[self.rank])
            output.copy_(self.metadata.flatten(0, 1))
        else:
            gathered = output.view(self.group_size, *input.shape)
            gathered[..., 0].fill_(-8388608)  # FP32 -inf bits.
            gathered[..., 1].fill_(-1)
            gathered[self.rank].copy_(input)


def test_mtp_topk(cpu_delta, global_args, monkeypatch):
    """Draft steps write and select new KV beyond the main model's window."""
    monkeypatch.setitem(global_args.infer, "mtp_size", 3)
    cache = PagedKVCache(
        GlobalLocalMap.from_range(0, 1),
        num_hot_req=1,
        max_seq_len=512,
        num_blocks=4,
        block_size=64,
        dcp_size=2,
        dcp_rank=0,
        shape_per_token_dict={"indexer_k": (4,)},
        dtype_dict={"indexer_k": torch.bfloat16},
        device="cpu",
    )
    cache.paged_kv_cache["indexer_k"].zero_()
    cache.insert_kv_cache_from_transfer("req", [2, 1, 3], 256)
    cache._upd_gpu_block_table(["req"])
    cache.seq_len_delta.copy_from_list([255], [256])
    delta = cache.mtp_seq_len_delta
    delta.copy_from_list([255], [256])
    delta.is_decode_stage = True
    accessor = cache.get_accessor(0, is_mtp=True)
    indexer = _indexer_without_runtime_init("torch_bf16")
    q = torch.ones(1, 2, 4, dtype=torch.bfloat16)
    weights = torch.ones(1, 2)
    for offset in range(2):
        cache.advance_mtp_cache_decode()
        key = torch.full((1, 4), offset + 1, dtype=torch.bfloat16)
        indexer.append_indexer_kv(key, None, delta, accessor)
        assert torch.equal(accessor.kv["indexer_k"][3, offset], key[0])
        result = indexer.dcp_topk(
            q,
            weights,
            delta,
            accessor,
            k=1,
            comm_group=_FakeDecodeCPAllGatherGroup(2),
        )
        assert result.tolist() == [[256 + offset]]
    assert cache.seq_len_delta.new.lens_tensor_device.tolist() == [256]


def test_global_topk(cpu_delta):
    """All owners select the same global IDs, with per-query MTP causal bounds."""
    size, page_size, k = 3, 2, 4
    lengths = [13, 5]
    queries = 3
    rows = len(lengths) * queries
    delta = cpu_delta
    delta.copy_from_list([length - queries for length in lengths], lengths)
    q = torch.ones(rows, 2, 4, dtype=torch.bfloat16)
    weights = torch.ones(rows, 2)
    table = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    indexer = _indexer_without_runtime_init("torch_bf16")
    accessors, metadata = [], []
    for rank in range(size):
        keys = torch.zeros(6, page_size, 4, dtype=torch.bfloat16)
        for request, length in enumerate(lengths):
            for token in range(length):
                if (token // page_size) % size == rank:
                    group_index = token // (page_size * size)
                    keys[table[request, group_index], token % page_size] = token + 1
        accessor = PagedKVCacheAccessor(
            block_table=table,
            kv={"indexer_k": keys},
            block_size=page_size,
            dcp_size=size,
            dcp_rank=rank,
        )
        group = _FakeDecodeCPAllGatherGroup(size, rank)
        indexer.dcp_topk(q, weights, delta, accessor, k, comm_group=group)
        accessors.append(accessor)
        metadata.append(group.local_metadata)

    metadata = torch.stack(metadata)
    results = []
    for rank, accessor in enumerate(accessors):
        result = indexer.dcp_topk(
            q,
            weights,
            delta,
            accessor,
            k,
            comm_group=_FakeDecodeCPAllGatherGroup(size, rank, metadata),
        )
        assert result.shape == (rows, k) and result.dtype == torch.int32
        for row, position in enumerate(delta.delta_position_ids_tensor_device.tolist()):
            expected = sorted(range(position + 1), key=lambda token: (-token, token))[
                :k
            ]
            assert sorted(result[row, result[row] >= 0].tolist()) == sorted(expected)
            assert (result[row, result[row] < 0] == -1).all()
        results.append(result)
    assert all(torch.equal(result, results[0]) for result in results[1:])
    for row, position in enumerate(delta.delta_position_ids_tensor_device.tolist()):
        owner_tokens = [
            token
            for rank, result in enumerate(results)
            for token in result[row].tolist()
            if token >= 0 and token // page_size % size == rank
        ]
        assert len(owner_tokens) == len(set(owner_tokens)) == min(k, position + 1)


def test_shared_topk(monkeypatch, cpu_delta):
    import chitu.models.model_deepseek_v3 as model_module
    from chitu.models.model_glm52 import IndexerGLM52, _IndexerBuffer

    query_rows_per_request = 3
    rows = 2 * query_rows_per_request
    backend = _indexer_without_runtime_init("torch_bf16")
    group = _FakeDecodeCPAllGatherGroup(2)
    monkeypatch.setattr(dsa_indexer_module, "get_dcp_group", lambda: group)
    monkeypatch.setattr(model_module, "get_cp_context", lambda: SimpleNamespace())

    buffer = _IndexerBuffer()
    full = object.__new__(IndexerGLM52)
    torch.nn.Module.__init__(full)
    full.set_indexer_buffer("write", buffer)
    full.indexer_impl = backend
    full.n_heads = 2
    full.softmax_scale = 1.0
    full.index_topk = 2
    full.weights_proj = lambda x: torch.ones(rows, 2)
    monkeypatch.setattr(
        full,
        "_build_index_qk",
        lambda x, q, k, *args, **kwargs: ((q, None), (k, None)),
    )
    delta = cpu_delta
    delta.copy_from_list([9 - query_rows_per_request] * 2, [9, 9])
    keys = torch.zeros(6, 2, 4, dtype=torch.bfloat16)
    keys[:, 0] = 1
    keys[:, 1] = 2
    accessor = PagedKVCacheAccessor(
        block_table=torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32),
        kv={"indexer_k": keys},
        block_size=2,
        dcp_size=2,
        dcp_rank=0,
    )
    inputs = (
        torch.zeros(rows, 4),
        torch.ones(rows, 2, 4, dtype=torch.bfloat16),
        torch.zeros(rows, 4, dtype=torch.bfloat16),
        delta,
        None,
        True,
        accessor,
    )
    result = full.forward(*inputs)
    assert result.dtype == torch.int32 and result.shape == (rows, 2)
    assert all(sorted(row) == [1, 5] for row in result.tolist())

    shared = object.__new__(IndexerGLM52)
    torch.nn.Module.__init__(shared)
    shared.set_indexer_buffer("read", buffer)
    torch.testing.assert_close(shared.forward(*inputs), result)
    reused, page_table = shared.read_reused_topk_for_mla(use_page_table=False)
    torch.testing.assert_close(reused, result)
    assert page_table is None


@pytest.mark.parametrize("impl", ["hygon", "triton_bf16"])
def test_backend_topk(impl):
    """Replay causal TopK as queries cross a page-owner boundary."""

    if impl == "hygon" and not support_indexer_hygon:
        pytest.skip("Decode-CP Hygon indexer requires Hygon paged MQA")
    if impl == "triton_bf16" and (not torch.cuda.is_available() or not has_triton):
        pytest.skip("Decode-CP Triton indexer requires an accelerator and Triton")

    num_heads, head_dim, page_size = 32, 128, 64
    queries = 3
    query_rows = 2 * queries
    q = torch.ones(query_rows, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    weights = torch.ones(query_rows, num_heads, dtype=torch.float32, device="cuda")
    token_values = torch.arange(4 * page_size, device="cuda", dtype=torch.float32)
    k_cache = (
        token_values.to(torch.bfloat16)
        .view(4, page_size, 1)
        .expand(-1, -1, head_dim)
        .contiguous()
    )
    accessor = PagedKVCacheAccessor(
        block_table=torch.tensor([[2, 0], [3, 1]], dtype=torch.int32, device="cuda"),
        kv={"indexer_k": k_cache},
        block_size=page_size,
        dcp_rank=1,
        dcp_size=2,
    )
    delta = BatchedSeqLenDelta(
        [129 - queries, 65 - queries],
        [129, 65],
        device="cuda",
        max_total_delta_len=query_rows,
    )
    delta.is_decode_stage = True
    positions = delta.delta_position_ids_tensor_device
    reference = _indexer_without_runtime_init("torch_bf16").dcp_topk(
        q,
        weights,
        delta,
        accessor,
        k=32,
        comm_group=_FakeDecodeCPAllGatherGroup(2, rank=1),
    )
    indexer = _indexer_without_runtime_init(impl)
    actual = indexer.dcp_topk(
        q,
        weights,
        delta,
        accessor,
        k=32,
        comm_group=_FakeDecodeCPAllGatherGroup(2, rank=1),
    )
    assert torch.equal(actual.sort(dim=-1).values, reference.sort(dim=-1).values)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = indexer.dcp_topk(
            q,
            weights,
            delta,
            accessor,
            k=32,
            comm_group=_FakeDecodeCPAllGatherGroup(2, rank=1),
        )
    positions.add_(1)
    q.mul_(2)
    graph.replay()
    torch.cuda.synchronize()
    updated = _indexer_without_runtime_init("torch_bf16").dcp_topk(
        q,
        weights,
        delta,
        accessor,
        k=32,
        comm_group=_FakeDecodeCPAllGatherGroup(2, rank=1),
    )
    assert not torch.equal(updated.sort(dim=-1).values, reference.sort(dim=-1).values)
    assert torch.equal(replayed.sort(dim=-1).values, updated.sort(dim=-1).values)
