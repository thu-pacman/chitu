# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import chitu.dsa_indexer as dsa_indexer_module
from chitu.dsa_indexer import DSAIndexer
from chitu.kv_cache import DenseKVCacheAccessor, PagedKVCacheAccessor


BACKEND_METHODS = [
    ("deepgemm", "blockfp8_index_score_dsa_deepgemm"),
    ("hygon", "bf16_index_score_dsa_hygon"),
    ("torch_bf16", "bf16_index_score_dsa_torch_bf16"),
    ("triton", "blockfp8_index_score_dsa_triton"),
    ("torch", "blockfp8_index_score_dsa_triton"),
]


def _indexer_without_runtime_init(impl: str) -> DSAIndexer:
    indexer = object.__new__(DSAIndexer)
    indexer.impl = impl
    indexer.static_max_n = 8192
    indexer.mtp_size = 1
    return indexer


def _seq_len_delta(*, max_len: int = 3, is_decode_stage: bool = False):
    return SimpleNamespace(
        is_decode_stage=is_decode_stage,
        delta_position_ids_tensor_device=torch.arange(3, dtype=torch.int32),
        delta_seq_ids_tensor_device=torch.zeros(3, dtype=torch.int32),
        new=SimpleNamespace(
            max_len=max_len,
            position_ids_tensor_device=torch.arange(3, dtype=torch.int32),
            seq_ids_tensor_device=torch.zeros(3, dtype=torch.int32),
        ),
    )


@pytest.mark.parametrize(("impl", "method_name"), BACKEND_METHODS)
def test_prefill_select_all_skips_score_for_every_backend(
    monkeypatch, impl, method_name
):
    indexer = _indexer_without_runtime_init(impl)
    calls = []

    def fake_score(*args, skip_prefill_score=False, **kwargs):
        calls.append(skip_prefill_score)
        return None

    monkeypatch.setattr(indexer, method_name, fake_score)
    q = torch.empty(3, 1)

    indices = indexer.dsa_indexer(
        q,
        torch.empty(3, 1),
        None,
        torch.empty(3, 1),
        _seq_len_delta(),
        object(),
        is_causal=True,
        index_topk=4,
        return_indices=True,
    )

    assert calls == [True]
    # Keep the configured TopK width even when the actual sequence is shorter.
    assert torch.equal(indices, torch.arange(4, dtype=torch.int32).repeat(3, 1))


@pytest.mark.parametrize(("impl", "method_name"), BACKEND_METHODS)
def test_logits_request_keeps_score_path_for_every_backend(
    monkeypatch, impl, method_name
):
    indexer = _indexer_without_runtime_init(impl)
    expected = torch.randn(3, 4)
    calls = []

    def fake_score(*args, skip_prefill_score=False, **kwargs):
        calls.append(skip_prefill_score)
        return expected

    monkeypatch.setattr(indexer, method_name, fake_score)

    actual = indexer.dsa_indexer(
        torch.empty(3, 1),
        torch.empty(3, 1),
        None,
        torch.empty(3, 1),
        _seq_len_delta(),
        object(),
        is_causal=True,
        index_topk=4,
        return_indices=False,
    )

    assert calls == [False]
    assert actual is expected


@pytest.mark.parametrize(
    ("is_decode_stage", "max_len"),
    [(True, 4), (False, 5)],
)
def test_decode_or_long_prefill_keeps_topk_path(monkeypatch, is_decode_stage, max_len):
    indexer = _indexer_without_runtime_init("triton")
    logits = torch.randn(3, max_len)
    expected = torch.full((3, 4), 7, dtype=torch.int64)
    calls = []

    def fake_score(*args, skip_prefill_score=False, **kwargs):
        calls.append(skip_prefill_score)
        return logits

    monkeypatch.setattr(indexer, "blockfp8_index_score_dsa_triton", fake_score)
    monkeypatch.setattr(
        dsa_indexer_module,
        "topk_indices",
        lambda *args, **kwargs: expected,
    )

    actual = indexer.dsa_indexer(
        torch.empty(3, 1),
        torch.empty(3, 1),
        None,
        torch.empty(3, 1),
        _seq_len_delta(max_len=max_len, is_decode_stage=is_decode_stage),
        object(),
        is_causal=True,
        index_topk=4,
        return_indices=True,
    )

    assert calls == [False]
    assert actual is expected


def _paged_accessor(*keys: str) -> PagedKVCacheAccessor:
    return PagedKVCacheAccessor(
        torch.zeros(1, 1, dtype=torch.int32),
        {key: torch.empty(1) for key in keys},
    )


def test_deepgemm_fast_path_appends_cache_without_reading(monkeypatch):
    indexer = _indexer_without_runtime_init("deepgemm")
    appended = []
    monkeypatch.setattr(
        dsa_indexer_module,
        "append_to_paged_kv_cache_blockfp8_deepgemm",
        lambda *args, **kwargs: appended.append(True),
    )
    monkeypatch.setattr(
        dsa_indexer_module,
        "read_from_paged_indexer_kv_cache_deepgemm",
        lambda *args, **kwargs: pytest.fail("fast path must not read the cache"),
    )

    result = indexer.blockfp8_index_score_dsa_deepgemm(
        torch.empty(3, 1),
        torch.empty(3, 1),
        torch.empty(3, 1),
        torch.empty(3, 1),
        _seq_len_delta(),
        _paged_accessor("indexer_k_ks"),
        skip_prefill_score=True,
    )

    assert result is None
    assert appended == [True]


@pytest.mark.parametrize(
    ("impl", "method_name"),
    [
        ("hygon", "bf16_index_score_dsa_hygon"),
        ("torch_bf16", "bf16_index_score_dsa_torch_bf16"),
    ],
)
def test_bf16_fast_path_appends_cache_without_reading(monkeypatch, impl, method_name):
    indexer = _indexer_without_runtime_init(impl)
    appended = []
    monkeypatch.setattr(
        dsa_indexer_module,
        "append_to_paged_kv_cache",
        lambda *args, **kwargs: appended.append(True),
    )
    monkeypatch.setattr(
        dsa_indexer_module,
        "read_from_paged_kv_cache",
        lambda *args, **kwargs: pytest.fail("fast path must not read the cache"),
    )

    result = getattr(indexer, method_name)(
        torch.empty(3, 1),
        torch.empty(3, 1),
        torch.empty(3, 1),
        _seq_len_delta(),
        _paged_accessor("indexer_k"),
        skip_prefill_score=True,
    )

    assert result is None
    assert appended == [True]


@pytest.mark.parametrize("cache_layout", ["paged", "dense"])
def test_triton_fast_path_appends_k_and_scale_without_scoring(
    monkeypatch, cache_layout
):
    indexer = _indexer_without_runtime_init("triton")
    appended = []
    monkeypatch.setattr(
        dsa_indexer_module,
        "get_global_args",
        lambda: SimpleNamespace(
            infer=SimpleNamespace(
                raise_lower_bit_float_to=None,
                max_seq_len=8192,
            )
        ),
    )
    monkeypatch.setattr(
        dsa_indexer_module,
        "blockfp8_index_score_ragged_q_paged_k_dsv32",
        lambda *args, **kwargs: pytest.fail("fast path must not compute scores"),
    )
    monkeypatch.setattr(
        dsa_indexer_module,
        "blockfp8_index_score_ragged_q_dense_k_dsv32",
        lambda *args, **kwargs: pytest.fail("fast path must not compute scores"),
    )

    if cache_layout == "paged":
        monkeypatch.setattr(
            dsa_indexer_module,
            "append_to_paged_kv_cache",
            lambda *args, **kwargs: appended.append(True),
        )
        accessor = _paged_accessor("indexer_k", "indexer_ks")
    else:
        monkeypatch.setattr(
            dsa_indexer_module,
            "append_to_dense_kv_cache",
            lambda *args, **kwargs: appended.append(True),
        )
        accessor = DenseKVCacheAccessor(
            {"indexer_k": torch.empty(1), "indexer_ks": torch.empty(1)}
        )

    result = indexer.blockfp8_index_score_dsa_triton(
        torch.empty(3, 1),
        torch.empty(3, 1),
        torch.empty(3, 1),
        torch.empty(3, 1),
        _seq_len_delta(),
        accessor,
        skip_prefill_score=True,
    )

    assert result is None
    assert appended == [True, True]
