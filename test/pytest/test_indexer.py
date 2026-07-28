# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import re
from types import SimpleNamespace

import pytest
import torch

import chitu.dsa_indexer as dsa_indexer_module
from chitu.dsa_indexer import DSAIndexer
from chitu.kv_cache import DenseKVCacheAccessor, PagedKVCacheAccessor
from chitu.kv_cache.providers.deepseek_v3 import (
    deepseek_v3_indexer_cache_spec,
    deepseek_v3_kv_cache_spec,
)


BACKEND_METHODS = [
    ("deepgemm", "blockfp8_index_score_dsa_deepgemm"),
    ("hygon", "bf16_index_score_dsa_hygon"),
    ("torch_bf16", "bf16_index_score_dsa_torch_bf16"),
    ("triton", "blockfp8_index_score_dsa_torch_or_triton"),
    ("torch", "blockfp8_index_score_dsa_torch_or_triton"),
]


class AttrNamespace(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


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

    monkeypatch.setattr(indexer, "blockfp8_index_score_dsa_torch_or_triton", fake_score)
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


def _dsa_args(
    *,
    kv_quant_type=None,
    main_kv_quant_type=None,
    indexer_type="torch_bf16",
):
    kv_cache_rules = []
    if kv_quant_type is not None:
        kv_cache_rules.append(SimpleNamespace(regex="^indexer_k$", type=kv_quant_type))
    if main_kv_quant_type is not None:
        kv_cache_rules.append(
            SimpleNamespace(regex="^(kv_lora|k_pe)$", type=main_kv_quant_type)
        )
    return SimpleNamespace(
        models=AttrNamespace(
            index_topk=2048,
            index_head_dim=128,
            index_n_heads=32,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            quant_config=SimpleNamespace(
                kv_cache=SimpleNamespace(rules=kv_cache_rules)
            ),
        ),
        infer=SimpleNamespace(
            indexer_type=indexer_type,
            cache_type="paged",
            mtp_size=1,
            mla_absorb="absorb",
            tp_size=1,
        ),
    )


@pytest.mark.parametrize("indexer_type", ["hygon", "torch_bf16"])
def test_bf16_indexer_types_require_unquantized_dsa_indexer_kv(indexer_type):
    args = _dsa_args(kv_quant_type="fp8_pertoken_indexer")

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Unrecognized indexer_type {indexer_type} for FP8 indexer KV quantization."
        ),
    ):
        dsa_indexer_module.validate_indexer_config(args, indexer_type)


@pytest.mark.parametrize("indexer_type", ["deepgemm", "triton", "torch"])
def test_fp8_indexer_types_require_fp8_dsa_indexer_kv(indexer_type):
    args = _dsa_args()

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Unrecognized indexer_type {indexer_type} for BF16 indexer KV quantization."
        ),
    ):
        dsa_indexer_module.validate_indexer_config(args, indexer_type)


@pytest.mark.parametrize(
    ("indexer_type", "expected_keys", "expected_dtypes"),
    [
        (
            "deepgemm",
            {"indexer_k_ks"},
            {"indexer_k_ks": torch.float8_e4m3fn},
        ),
        (
            "triton",
            {"indexer_k", "indexer_ks"},
            {"indexer_k": torch.float8_e4m3fn, "indexer_ks": torch.float32},
        ),
        (
            "torch",
            {"indexer_k", "indexer_ks"},
            {"indexer_k": torch.float8_e4m3fn, "indexer_ks": torch.float32},
        ),
    ],
)
def test_fp8_dsa_indexer_cache_layouts_follow_indexer_type(
    indexer_type, expected_keys, expected_dtypes
):
    spec = deepseek_v3_indexer_cache_spec(
        _dsa_args(kv_quant_type="fp8_pertoken_indexer", indexer_type=indexer_type),
        None,
    )

    assert set(spec.kvargs["shape_per_token_dict"]) == expected_keys
    assert spec.kvargs["dtype_dict"] == expected_dtypes
    assert spec.kvargs["quant_type"] == "fp8_pertoken_indexer"


def test_unquantized_dsa_indexer_cache_uses_bf16_base_tensor():
    spec = deepseek_v3_indexer_cache_spec(_dsa_args(indexer_type="torch_bf16"), None)

    assert spec.kvargs["shape_per_token_dict"] == {"indexer_k": (128,)}
    assert spec.kvargs["dtype_dict"] == {"indexer_k": torch.bfloat16}


def test_fp8_dsa_main_cache_validates_base_tensor_rules_for_packed_layout():
    spec = deepseek_v3_kv_cache_spec(
        _dsa_args(main_kv_quant_type="fp8_pertoken_dsa"), None
    )

    assert spec.kvargs["shape_per_token_dict"] == {"kv_lora_k_pe": (656,)}
    assert spec.kvargs["dtype_dict"] == {"kv_lora_k_pe": torch.float8_e4m3fn}
    assert spec.kv_keys == ["kv_lora", "k_pe"]


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


@pytest.mark.parametrize("impl", ["triton", "torch"])
@pytest.mark.parametrize("cache_layout", ["paged", "dense"])
def test_torch_or_triton_fast_path_appends_k_and_scale_without_scoring(
    monkeypatch, impl, cache_layout
):
    indexer = _indexer_without_runtime_init(impl)
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

    result = indexer.blockfp8_index_score_dsa_torch_or_triton(
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
