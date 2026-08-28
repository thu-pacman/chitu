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
from chitu.ops.quant.blockfp8.index_score import (
    blockfp8_index_score_ragged_q_dense_k_dsv32_torch,
)


class AttrNamespace(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def _indexer_without_runtime_init(impl: str) -> DSAIndexer:
    indexer = object.__new__(DSAIndexer)
    indexer.impl = impl
    indexer.static_max_n = 8192
    indexer.mtp_size = 1
    indexer.index_topk = 2048
    indexer._indexer_logits_chunk_bytes = None
    indexer.prefill_schedule = None
    return indexer


def _seq_len_delta(*, max_len: int = 3, is_decode_stage: bool = False):
    return SimpleNamespace(
        is_decode_stage=is_decode_stage,
        batch_size=1,
        delta_position_ids_tensor_device=torch.arange(3, dtype=torch.int32),
        delta_seq_ids_tensor_device=torch.zeros(3, dtype=torch.int32),
        new=SimpleNamespace(
            max_len=max_len,
            position_ids_tensor_device=torch.arange(3, dtype=torch.int32),
            seq_ids_tensor_device=torch.zeros(3, dtype=torch.int32),
            prefix_lens_tensor_device=torch.zeros(1, dtype=torch.int32),
            lens_tensor_device=torch.tensor([max_len], dtype=torch.int32),
        ),
    )


@pytest.mark.parametrize(
    ("impl", "expected"),
    [
        ("hygon", 3),  # new.max_len
        ("torch_bf16", 8192),  # static_max_n
        ("triton_bf16", 8192),
        ("torch", 8192),
        ("triton", 8192),
        ("deepgemm", 2048),  # align(min(static, max(max_len, index_topk)), 256)
    ],
)
def test_row_width_is_per_backend(impl, expected):
    indexer = _indexer_without_runtime_init(impl)
    assert indexer.row_width(_seq_len_delta(max_len=3)) == expected


def test_chunk_size_none_when_budget_unset():
    indexer = _indexer_without_runtime_init("deepgemm")
    assert indexer._indexer_logits_chunk_bytes is None
    assert indexer.chunk_size(_seq_len_delta()) is None


def test_chunk_size_none_for_decode():
    indexer = _indexer_without_runtime_init("deepgemm")
    indexer._indexer_logits_chunk_bytes = 1 << 20
    assert indexer.chunk_size(_seq_len_delta(is_decode_stage=True)) is None


@pytest.mark.parametrize("impl", ["triton", "torch", "torch_bf16", "triton_bf16"])
def test_chunk_size_derived_for_sliceable_backends(impl):
    indexer = _indexer_without_runtime_init(impl)
    # row_width == static_max_n == 8192 fp32 cols == 32768 bytes/row.
    indexer._indexer_logits_chunk_bytes = 32768 * 5
    assert indexer.chunk_size(_seq_len_delta()) == 5
    # Budget smaller than a single row still yields at least one row.
    indexer._indexer_logits_chunk_bytes = 1
    assert indexer.chunk_size(_seq_len_delta()) == 1


def test_index_score_empty_batch_returns_well_formed_buffer():
    indexer = _indexer_without_runtime_init("torch_bf16")
    out = indexer.index_score(
        torch.empty(0, 1, 128),
        torch.empty(0, 1),
        _seq_len_delta(),
        object(),
    )
    assert out.shape == (0, indexer.static_max_n)
    assert out.dtype == torch.float32


@pytest.mark.parametrize(
    ("impl", "scorer_name"),
    [
        ("deepgemm", "blockfp8_index_score_ragged_qk_dsv32_deepgemm"),
        ("hygon", "bf16_index_score_ragged_qk_dsv32_hygon"),
    ],
)
def test_index_score_prefill_reads_cache_then_method_scores(
    monkeypatch, impl, scorer_name
):
    """Prefill index_score reads full-context K from cache, then scores q."""
    indexer = _indexer_without_runtime_init(impl)
    expected = torch.randn(3, 4)
    read_calls = []
    scorer_kwargs = []
    ke = torch.tensor([1, 2, 3], dtype=torch.int32)
    ks = torch.tensor([0, 4, 8], dtype=torch.int32)

    monkeypatch.setattr(
        dsa_indexer_module,
        "read_from_paged_kv_cache",
        lambda *a, **k: read_calls.append("bf16") or torch.empty(3, 128),
    )
    monkeypatch.setattr(
        dsa_indexer_module,
        "read_from_paged_indexer_kv_cache_deepgemm",
        lambda *a, **k: read_calls.append("deepgemm")
        or (torch.empty(3, 1, 128), torch.empty(3, 1, 1)),
    )

    def fake_method_score(*args, **kwargs):
        scorer_kwargs.append(kwargs)
        return expected

    monkeypatch.setattr(indexer, scorer_name, fake_method_score)

    out = indexer.index_score(
        torch.empty(3, 1, 128),
        torch.empty(3, 1),
        _seq_len_delta(is_decode_stage=False),
        _paged_accessor("indexer_k", "indexer_k_ks"),
        ke=ke,
        ks=ks,
    )

    assert out is expected
    assert len(read_calls) == 1
    assert len(scorer_kwargs) == 1
    if impl == "deepgemm":
        assert scorer_kwargs[0]["ks"] is ks
        assert scorer_kwargs[0]["ke_override"] is ke
    else:
        assert scorer_kwargs[0]["ke"] is ke
        assert scorer_kwargs[0]["ks"] is ks


@pytest.mark.parametrize("impl", ["torch_bf16", "triton_bf16"])
def test_index_score_prefill_reads_cache_then_dispatcher_scores(monkeypatch, impl):
    """Dispatcher-backed bf16 prefill paths call the module-level scorer."""
    indexer = _indexer_without_runtime_init(impl)
    expected = torch.randn(3, 4)
    read_calls = []
    dispatch_calls = []

    monkeypatch.setattr(
        dsa_indexer_module,
        "read_from_paged_kv_cache",
        lambda *a, **k: read_calls.append("bf16") or torch.empty(3, 128),
    )

    def fake_bf16_score(*args, **kwargs):
        dispatch_calls.append(kwargs.get("impl"))
        return expected

    monkeypatch.setattr(
        dsa_indexer_module, "bf16_index_score_ragged_qk_dsv32", fake_bf16_score
    )
    if impl == "triton_bf16":
        monkeypatch.setattr(dsa_indexer_module, "DEFAULT_BLOCK_M", 8, raising=False)
        monkeypatch.setattr(
            dsa_indexer_module,
            "_bucket_max_n",
            lambda actual_max_n: actual_max_n,
            raising=False,
        )
        monkeypatch.setattr(
            dsa_indexer_module,
            "build_qblock_schedule",
            lambda ks, block_m, device: (
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.tensor([ks.shape[0]], dtype=torch.int32, device=device),
                1,
            ),
            raising=False,
        )

    out = indexer.index_score(
        torch.empty(3, 1, 128),
        torch.empty(3, 1),
        _seq_len_delta(is_decode_stage=False),
        _paged_accessor("indexer_k"),
    )

    assert out is expected
    assert read_calls == ["bf16"]
    assert dispatch_calls == [{"torch_bf16": "torch", "triton_bf16": "triton"}[impl]]


@pytest.mark.parametrize(
    ("impl", "scorer_name"),
    [
        ("deepgemm", "blockfp8_index_score_ragged_q_paged_k_dsv32_deepgemm"),
        ("hygon", "bf16_index_score_ragged_q_paged_k_dsv32_hygon"),
        ("torch_bf16", "bf16_index_score_ragged_q_paged_k_dsv32_torch_bf16"),
    ],
)
def test_index_score_decode_uses_paged_scorer_without_read(
    monkeypatch, impl, scorer_name
):
    """Decode index_score scores paged K directly, never reading it back."""
    indexer = _indexer_without_runtime_init(impl)
    expected = torch.randn(1, 4)

    monkeypatch.setattr(
        dsa_indexer_module,
        "read_from_paged_kv_cache",
        lambda *a, **k: pytest.fail("decode must not read the cache"),
    )
    monkeypatch.setattr(
        dsa_indexer_module,
        "read_from_paged_indexer_kv_cache_deepgemm",
        lambda *a, **k: pytest.fail("decode must not read the cache"),
    )
    monkeypatch.setattr(indexer, scorer_name, lambda *a, **k: expected)

    out = indexer.index_score(
        torch.empty(1, 1, 128),
        torch.empty(1, 1),
        _seq_len_delta(is_decode_stage=True),
        _paged_accessor("indexer_k", "indexer_k_ks"),
    )

    assert out is expected


def test_triton_bf16_decode_uses_paged_dispatcher_without_read(monkeypatch):
    indexer = _indexer_without_runtime_init("triton_bf16")
    expected = torch.randn(1, 4)

    monkeypatch.setattr(
        dsa_indexer_module,
        "read_from_paged_kv_cache",
        lambda *a, **k: pytest.fail("decode must not read the cache"),
    )

    dispatch_calls = []

    def fake_bf16_paged_score(*args, **kwargs):
        dispatch_calls.append(kwargs.get("impl"))
        return expected

    monkeypatch.setattr(
        dsa_indexer_module,
        "bf16_index_score_ragged_q_paged_k_dsv32",
        fake_bf16_paged_score,
    )

    out = indexer.index_score(
        torch.empty(1, 1, 128),
        torch.empty(1, 1),
        _seq_len_delta(is_decode_stage=True),
        _paged_accessor("indexer_k"),
    )

    assert out is expected
    assert dispatch_calls == ["triton"]


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


@pytest.mark.parametrize("indexer_type", ["hygon", "torch_bf16", "triton_bf16"])
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


@pytest.mark.parametrize("indexer_type", ["torch_bf16", "triton_bf16"])
def test_unquantized_dsa_indexer_cache_uses_bf16_base_tensor(indexer_type):
    spec = deepseek_v3_indexer_cache_spec(_dsa_args(indexer_type=indexer_type), None)

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


def test_deepgemm_append_indexer_kv_uses_packed_layout(monkeypatch):
    indexer = _indexer_without_runtime_init("deepgemm")
    appended = []
    monkeypatch.setattr(
        dsa_indexer_module,
        "append_to_paged_kv_cache_blockfp8_deepgemm",
        lambda *args, **kwargs: appended.append(True),
    )

    indexer.append_indexer_kv(
        torch.empty(3, 1),
        torch.empty(3, 1),
        _seq_len_delta(),
        _paged_accessor("indexer_k_ks"),
    )

    assert appended == [True]


@pytest.mark.parametrize("impl", ["hygon", "torch_bf16", "triton_bf16"])
def test_bf16_append_indexer_kv_appends_single_tensor(monkeypatch, impl):
    indexer = _indexer_without_runtime_init(impl)
    appended = []
    monkeypatch.setattr(
        dsa_indexer_module,
        "append_to_paged_kv_cache",
        lambda *args, **kwargs: appended.append(True),
    )

    indexer.append_indexer_kv(
        torch.empty(3, 1),
        torch.empty(3, 1),
        _seq_len_delta(),
        _paged_accessor("indexer_k"),
    )

    assert appended == [True]


@pytest.mark.parametrize("impl", ["triton", "torch"])
@pytest.mark.parametrize("cache_layout", ["paged", "dense"])
def test_torch_or_triton_append_indexer_kv_appends_k_and_scale(
    monkeypatch, impl, cache_layout
):
    indexer = _indexer_without_runtime_init(impl)
    appended = []

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

    indexer.append_indexer_kv(
        torch.empty(3, 1),
        torch.empty(3, 1),
        _seq_len_delta(),
        accessor,
    )

    # Both the K tensor and its scale are appended.
    assert appended == [True, True]


def _ragged_delta(q_seq_ids, q_pos_ids, k_lens):
    """Minimal delta stub for the torch ragged dense-K index-score op."""
    b = len(k_lens)
    max_n = max(k_lens)
    k_seq_ids = torch.cat(
        [torch.full((int(l),), s, dtype=torch.long) for s, l in enumerate(k_lens)]
    )
    k_pos_ids = torch.cat([torch.arange(int(l)) for l in k_lens])
    return SimpleNamespace(
        delta_seq_ids_tensor_device=torch.as_tensor(q_seq_ids, dtype=torch.long),
        delta_position_ids_tensor_device=torch.as_tensor(q_pos_ids, dtype=torch.long),
        new=SimpleNamespace(
            seq_ids_tensor_device=k_seq_ids,
            position_ids_tensor_device=k_pos_ids,
            max_len=max_n,
        ),
    )


def test_torch_ragged_index_score_masks_future_and_invalid_k():
    """Per-row-gather torch op masks causal-future and unwritten K to -inf."""
    torch.manual_seed(0)
    b, n, h, d = 2, 6, 4, 8
    k_lens = [4, 5]
    # Two query rows: (seq0,pos1) and (seq1,pos3).
    q_seq_ids = [0, 1]
    q_pos_ids = [1, 3]
    delta = _ragged_delta(q_seq_ids, q_pos_ids, k_lens)

    q = torch.randn(2, h, d)
    q_s = torch.rand(2, h, 1)
    k = torch.randn(b, n, d)
    k_s = torch.rand(b, n, 1)

    out = blockfp8_index_score_ragged_q_dense_k_dsv32_torch(
        q, q_s, k, k_s, delta, causal=True
    )
    assert out.shape == (2, n)

    # Row 0 (seq0,pos1): valid cols are 0..1; 2.. are future or unwritten.
    assert torch.isfinite(out[0, :2]).all()
    assert torch.isinf(out[0, 2:]).all()
    # Row 1 (seq1,pos3): valid cols are 0..3; 4 is future, 5 is unwritten.
    assert torch.isfinite(out[1, :4]).all()
    assert torch.isinf(out[1, 4:]).all()


def test_torch_ragged_index_score_is_query_sliceable():
    """Slicing the query rows yields identical per-row scores (chunk-safe)."""
    torch.manual_seed(1)
    b, n, h, d = 3, 8, 4, 8
    k_lens = [8, 6, 7]
    q_seq_ids = [0, 0, 1, 2, 2]
    q_pos_ids = [2, 7, 5, 0, 6]
    k = torch.randn(b, n, d)
    k_s = torch.rand(b, n, 1)
    q = torch.randn(len(q_seq_ids), h, d)
    q_s = torch.rand(len(q_seq_ids), h, 1)

    full = blockfp8_index_score_ragged_q_dense_k_dsv32_torch(
        q, q_s, k, k_s, _ragged_delta(q_seq_ids, q_pos_ids, k_lens), causal=True
    )

    # Score in two query chunks and stitch; must match the full pass exactly.
    rows = []
    for i, j in [(0, 2), (2, 5)]:
        chunk = blockfp8_index_score_ragged_q_dense_k_dsv32_torch(
            q[i:j],
            q_s[i:j],
            k,
            k_s,
            _ragged_delta(q_seq_ids[i:j], q_pos_ids[i:j], k_lens),
            causal=True,
        )
        rows.append(chunk)
    stitched = torch.cat(rows, dim=0)

    torch.testing.assert_close(stitched, full)
