# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import chitu.dsa_indexer_backend as dsa_indexer_module
from chitu.dsa_indexer_backend import DSAIndexer, get_indexer_class
import chitu.dsa_indexer_backend.base as base_backend
import chitu.dsa_indexer_backend.bf16_backend as bf16_backend
import chitu.dsa_indexer_backend.deepgemm_backend as deepgemm_backend
import chitu.dsa_indexer_backend.hygon_backend as hygon_backend
import chitu.dsa_indexer_backend.nvidia_topk as nvidia_topk_backend
import chitu.dsa_indexer_backend.torch_backend as torch_backend
import chitu.dsa_indexer_backend.triton_backend as triton_backend
import chitu.ops.topk as topk_module
from chitu.kv_cache import DenseKVCacheAccessor, PagedKVCacheAccessor
from chitu.utils import max_alloc_seq_len
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
    indexer = object.__new__(get_indexer_class(impl))
    indexer.impl = impl
    indexer.static_max_n = 8192
    indexer.mtp_size = 1
    indexer.index_topk = 2048
    indexer._indexer_logits_chunk_bytes = None
    indexer.prefill_schedule = None
    return indexer


def _implementation_module(indexer):
    return {
        "deepgemm": deepgemm_backend,
        "hygon": hygon_backend,
        "torch": torch_backend,
        "triton": torch_backend,
        "torch_bf16": bf16_backend,
        "triton_bf16": bf16_backend,
    }[indexer.impl]


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
        (
            bf16_backend
            if indexer.impl == "deepgemm"
            else _implementation_module(indexer)
        ),
        "read_from_paged_kv_cache",
        lambda *a, **k: read_calls.append("bf16") or torch.empty(3, 128),
    )
    monkeypatch.setattr(
        deepgemm_backend,
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
        (
            bf16_backend
            if indexer.impl == "deepgemm"
            else _implementation_module(indexer)
        ),
        "read_from_paged_kv_cache",
        lambda *a, **k: read_calls.append("bf16") or torch.empty(3, 128),
    )

    def fake_bf16_score(*args, **kwargs):
        dispatch_calls.append(kwargs.get("impl"))
        return expected

    monkeypatch.setattr(
        _implementation_module(indexer),
        "bf16_index_score_ragged_qk_dsv32",
        fake_bf16_score,
    )
    if impl == "triton_bf16":
        monkeypatch.setattr(triton_backend, "DEFAULT_BLOCK_M", 8, raising=False)
        monkeypatch.setattr(
            triton_backend,
            "_bucket_max_n",
            lambda actual_max_n: actual_max_n,
            raising=False,
        )
        monkeypatch.setattr(
            triton_backend,
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
        (
            bf16_backend
            if indexer.impl == "deepgemm"
            else _implementation_module(indexer)
        ),
        "read_from_paged_kv_cache",
        lambda *a, **k: pytest.fail("decode must not read the cache"),
    )
    monkeypatch.setattr(
        deepgemm_backend,
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
        (
            bf16_backend
            if indexer.impl == "deepgemm"
            else _implementation_module(indexer)
        ),
        "read_from_paged_kv_cache",
        lambda *a, **k: pytest.fail("decode must not read the cache"),
    )

    dispatch_calls = []

    def fake_bf16_paged_score(*args, **kwargs):
        dispatch_calls.append(kwargs.get("impl"))
        return expected

    monkeypatch.setattr(
        triton_backend,
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
        deepgemm_backend,
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
        _implementation_module(indexer),
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
            _implementation_module(indexer),
            "append_to_paged_kv_cache",
            lambda *args, **kwargs: appended.append(True),
        )
        accessor = _paged_accessor("indexer_k", "indexer_ks")
    else:
        monkeypatch.setattr(
            _implementation_module(indexer),
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


@pytest.mark.parametrize(
    ("impl", "expected_class"),
    [
        ("deepgemm", deepgemm_backend.DeepGEMMIndexer),
        ("hygon", hygon_backend.HygonIndexer),
        ("torch", torch_backend.TorchIndexer),
        ("triton", triton_backend.TritonIndexer),
        ("torch_bf16", torch_backend.TorchBF16Indexer),
        ("triton_bf16", triton_backend.TritonBF16Indexer),
    ],
)
@pytest.mark.parametrize("construction", ["explicit", "auto", "concrete"])
def test_backend_factory_preserves_constructor_and_base_type(
    monkeypatch, impl, expected_class, construction
):
    args = AttrNamespace(
        infer=AttrNamespace(
            indexer_type=impl, max_seq_len=1 << 20, max_batch_size=8, mtp_size=1
        ),
        models=AttrNamespace(index_topk=2048),
    )
    monkeypatch.setattr(base_backend, "get_global_args", lambda: args)
    validated = []
    monkeypatch.setattr(
        dsa_indexer_module,
        "validate_indexer_config",
        lambda actual_args, actual_impl: validated.append((actual_args, actual_impl)),
    )
    monkeypatch.setattr(
        deepgemm_backend, "deep_gemm", SimpleNamespace(get_num_sms=lambda: 80)
    )
    monkeypatch.setattr(hygon_backend, "get_dp_size", lambda: 1)
    if construction == "auto":
        backend = DSAIndexer()
    elif construction == "concrete":
        backend = expected_class()
    else:
        backend = DSAIndexer(impl)
    assert type(backend) is expected_class is get_indexer_class(impl)
    assert isinstance(backend, DSAIndexer)
    assert backend.impl == impl
    # static_max_n 是「可被寻址的最大长度」= max_alloc_seq_len(max_seq_len)：
    # MTP 路径会寻址到 max_seq_len - 1 + 3 * draft_len（见 chitu/utils.max_alloc_seq_len）
    assert backend.static_max_n == max_alloc_seq_len(1 << 20)
    assert validated == [(args, impl)]
    assert not hasattr(backend, "hygon_indexer_topk")
    if impl != "hygon":
        assert not hasattr(backend, "workspace")


def test_backend_factory_rejects_unknown_implementation():
    with pytest.raises(AssertionError, match="Unsupported indexer implementation"):
        DSAIndexer("unknown")


@pytest.mark.parametrize(
    "first_module", ["chitu.dsa_indexer_backend", "chitu.kv_cache.registry"]
)
def test_backend_package_imports_in_a_fresh_process(first_module):
    # conftest imports models before collection; use a fresh interpreter to
    # catch package/registry import cycles hidden by already-loaded modules.
    code = f"""
import importlib
importlib.import_module({first_module!r})
from chitu.dsa_indexer_backend import DSAIndexer
from chitu.dsa_indexer_backend.base import DSAIndexer as BaseIndexer
from chitu.models.model_deepseek_v3 import DSAIndexer as DeepSeekIndexer
from chitu.models.model_glm52 import DSAIndexer as GLMIndexer
assert DSAIndexer is BaseIndexer is DeepSeekIndexer is GLMIndexer
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "impl", ["deepgemm", "torch", "triton", "torch_bf16", "triton_bf16"]
)
def test_backend_topk_fallback_dispatch(monkeypatch, impl):
    indexer = _indexer_without_runtime_init(impl)
    logits = torch.empty(2, 32)
    lengths = torch.tensor([16, 24], dtype=torch.int32)
    starts = torch.tensor([0, 4], dtype=torch.int32)
    expected = torch.empty(2, 8, dtype=torch.int64)
    calls = []

    def generic(actual_logits, k, **kwargs):
        assert actual_logits is logits and k == 8
        assert kwargs["lengths"] is lengths
        assert kwargs["row_starts"] is starts
        assert kwargs["out_dtype"] == torch.int64
        if impl in ("deepgemm", "triton"):
            assert kwargs["impl"] == "torch"
        else:
            assert "impl" not in kwargs
        calls.append(True)
        return expected

    # NVIDIA TopK backends explicitly request exact Torch for unsupported shapes;
    # the remaining backends keep the base implementation's generic dispatch.
    target = nvidia_topk_backend if impl in ("deepgemm", "triton") else base_backend
    monkeypatch.setattr(target, "topk_indices", generic)
    assert (
        indexer.topk_indices(
            logits,
            8,
            object(),
            lengths=lengths,
            row_starts=starts,
            out_dtype=torch.int64,
        )
        is expected
    )
    assert calls == [True]


def test_deepgemm_decode_metadata_reuses_its_static_buffer(monkeypatch):
    indexer = object.__new__(deepgemm_backend.DeepGEMMIndexer)
    indexer.mtp_size = 1
    # This test isolates DeepGEMM's logits metadata; TopK state has its own tests.
    monkeypatch.setattr(nvidia_topk_backend, "has_nvidia_indexer_topk", False)
    calls = []

    def metadata(lengths, page_size, num_sms):
        assert lengths.dtype == torch.int32
        assert page_size == 64 and num_sms == 80
        calls.append(lengths.clone())
        return lengths.clone()

    monkeypatch.setattr(
        deepgemm_backend,
        "deep_gemm",
        SimpleNamespace(get_num_sms=lambda: 80, get_paged_mqa_logits_metadata=metadata),
    )
    indexer._init_backend(SimpleNamespace(infer=SimpleNamespace(max_batch_size=2)))
    indexer.prepare_metadata_for_decode(SimpleNamespace(batch_size=0))
    assert indexer.metadata is None and calls == []
    addresses = []
    for values in ([100, 200], [101, 201]):
        lengths = torch.tensor(values, dtype=torch.int32)
        indexer.prepare_metadata_for_decode(
            SimpleNamespace(
                batch_size=2, new=SimpleNamespace(lens_tensor_device=lengths)
            )
        )
        actual = indexer.metadata.get()
        assert torch.equal(actual, lengths)
        addresses.append(actual.data_ptr())
    assert addresses[0] == addresses[1] and len(calls) == 2


def test_triton_prefill_schedule_is_shared_per_step_and_reset(monkeypatch):
    indexer = object.__new__(triton_backend.TritonBF16Indexer)
    indexer._init_backend(None)
    calls = []

    def build(ks, block_m, device):
        calls.append(ks)
        return torch.zeros(1, dtype=torch.int32), torch.tensor([ks.numel()]), 1

    monkeypatch.setattr(triton_backend, "DEFAULT_BLOCK_M", 8, raising=False)
    monkeypatch.setattr(triton_backend, "_bucket_max_n", lambda n: n, raising=False)
    monkeypatch.setattr(triton_backend, "build_qblock_schedule", build, raising=False)
    ks, ke = torch.tensor([0, 0]), torch.tensor([10, 20])
    first, _, _ = indexer._prefill_schedule(ks, ke, True)
    second, actual_ks, actual_ke = indexer._prefill_schedule(
        ks.clone(), ke.clone(), True
    )
    assert first is second
    assert actual_ks is ks and actual_ke is ke
    assert len(calls) == 1
    indexer.prepare_metadata_for_prefill(object())
    assert indexer.prefill_schedule is None
    third, _, _ = indexer._prefill_schedule(ks, ke + 1, True)
    assert third is not first and third["actual_max_n"] == 21
    assert len(calls) == 2
