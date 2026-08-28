# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import chitu.dsa_indexer as dsa_indexer_module
import chitu.models.model_deepseek_v3 as model_deepseek_v3_module
from chitu.device_type import is_hygon
from chitu.dsa_indexer import DSAIndexer
from chitu.models.model_deepseek_v3 import Indexer


requires_hygon_deepgemm = pytest.mark.skipif(
    not (is_hygon() and dsa_indexer_module.has_hygon_compact_mqa_logits),
    reason="Hygon DeepGEMM compact-capable mqa_logits is not available",
)
requires_installed_hygon_deepgemm = pytest.mark.skipif(
    not (is_hygon() and dsa_indexer_module.has_hygon_deepgemm),
    reason="Hygon DeepGEMM is not installed",
)


def _new_mqa_logits(
    Q=None,
    K=None,
    Weights=None,
    ks=None,
    ke=None,
    kv_scale=None,
    clean_logit=True,
    D_out=None,
    max_seqlen_k=None,
    compact_output=True,
    assume_ks_nondecreasing=False,
):
    pass


def test_compact_mqa_feature_gate_requires_final_gfx936_contract(tmp_path):
    canonical_co = tmp_path / "deepgemm_mqa_logits.co"
    canonical_co.touch()
    common = {"gfx": "gfx936", "DEEPGEMM_ASM_DIR": str(tmp_path)}

    assert dsa_indexer_module._has_hygon_compact_mqa_logits(
        SimpleNamespace(mqa_logits=_new_mqa_logits, **common), True
    )
    assert not dsa_indexer_module._has_hygon_compact_mqa_logits(
        SimpleNamespace(mqa_logits=_new_mqa_logits, **common), False
    )
    assert not dsa_indexer_module._has_hygon_compact_mqa_logits(
        SimpleNamespace(
            gfx="gfx938",
            DEEPGEMM_ASM_DIR=str(tmp_path),
            mqa_logits=_new_mqa_logits,
        ),
        True,
    )

    def old_mqa_logits(*, max_seqlen_k=None, compact_output=True):
        pass

    assert not dsa_indexer_module._has_hygon_compact_mqa_logits(
        SimpleNamespace(mqa_logits=old_mqa_logits, **common), True
    )

    def wrong_default(
        *,
        clean_logit=True,
        D_out=None,
        max_seqlen_k=None,
        compact_output=False,
        assume_ks_nondecreasing=False,
    ):
        pass

    assert not dsa_indexer_module._has_hygon_compact_mqa_logits(
        SimpleNamespace(mqa_logits=wrong_default, **common), True
    )

    canonical_co.unlink()
    assert not dsa_indexer_module._has_hygon_compact_mqa_logits(
        SimpleNamespace(mqa_logits=_new_mqa_logits, **common), True
    )


def test_hygon_prefill_calls_compact_deepgemm_api(monkeypatch):
    indexer = object.__new__(DSAIndexer)

    # Fused QKV projection can leave Q/K with a padded row stride. The dense
    # MQA ABI has no Q/K stride arguments, so only these inputs need packing at
    # this boundary; weights and metadata are already contiguous upstream.
    q_storage = torch.arange(2 * 1 * 256, dtype=torch.float32).view(2, 1, 256)
    q = q_storage[..., ::2]
    k_storage = torch.zeros(4100, 256)
    k = k_storage[:, ::2]
    assert not q.is_contiguous()
    assert not k.is_contiguous()

    weights = torch.zeros(2, 1)
    ks = torch.tensor([0, 2050], dtype=torch.int32)
    lengths = torch.tensor([2050, 2050], dtype=torch.int32)
    logits = torch.zeros(2, 2050)

    def fake_mqa_logits(
        actual_q,
        actual_k,
        actual_weights,
        actual_ks,
        actual_ke,
        *,
        clean_logit,
        max_seqlen_k,
        compact_output,
        assume_ks_nondecreasing,
        D_out=None,
    ):
        assert actual_q.is_contiguous()
        assert actual_q.data_ptr() != q.data_ptr()
        assert torch.equal(actual_q, q)
        assert actual_k.is_contiguous()
        assert actual_k.data_ptr() != k.data_ptr()
        assert torch.equal(actual_k, k)
        assert actual_weights.data_ptr() == weights.data_ptr()
        assert actual_ks.data_ptr() == ks.data_ptr()
        assert torch.equal(actual_ke, ks + lengths)
        assert clean_logit is False
        assert max_seqlen_k == 2050
        assert compact_output is True
        assert assume_ks_nondecreasing is True
        assert D_out is None
        return logits

    monkeypatch.setattr(
        dsa_indexer_module,
        "hygon_deepgemm",
        SimpleNamespace(mqa_logits=fake_mqa_logits),
    )

    result = indexer.bf16_index_score_ragged_qk_dsv32_hygon(
        q,
        weights,
        k,
        SimpleNamespace(new=SimpleNamespace(max_len=2050)),
        causal=True,
        ke=lengths,
        ks=ks,
    )

    assert result is logits
    assert result.shape == (2, 2050)
    assert result.is_contiguous()


def test_model_topk_consumes_compact_local_columns(monkeypatch):
    indexer = object.__new__(Indexer)
    indexer.index_topk = 2
    indexer.indexer_impl = SimpleNamespace(static_max_n=4)
    logits = torch.zeros(2, 4)
    expected = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    delta = SimpleNamespace(
        delta_position_ids_tensor_device=torch.tensor([0, 1], dtype=torch.int32),
        delta_seq_ids_tensor_device=torch.tensor([0, 1], dtype=torch.int64),
        new=SimpleNamespace(lens_tensor_device=torch.tensor([3, 4], dtype=torch.int32)),
    )

    def fake_build_index_score(*args, **kwargs):
        reduce = args[7]
        return reduce(logits, delta)

    indexer._build_index_score = fake_build_index_score

    def fake_topk_indices(actual_logits, k, *, lengths, **kwargs):
        assert actual_logits is logits
        assert k == 2
        assert torch.equal(lengths, torch.tensor([3, 4], dtype=torch.int32))
        assert "row_starts" not in kwargs
        return expected

    monkeypatch.setattr(model_deepseek_v3_module, "topk_indices", fake_topk_indices)

    result = Indexer.forward(
        indexer,
        torch.empty(2, 1),
        torch.empty(2, 1),
        torch.empty(2, 1),
        delta,
        object(),
        False,
        object(),
    )
    assert result is expected


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_cp_noncausal_uses_full_request_lengths_for_single_and_chunked_scores(
    monkeypatch, chunk_size
):
    q = torch.zeros(4, 1, 128)
    k = torch.zeros(8, 128)
    # CP selects an ascending subsequence of the packed query rows. Sequence
    # IDs, and therefore absolute KS, remain nondecreasing on every rank.
    row_seq_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    positions = torch.tensor([0, 2, 1, 3], dtype=torch.int32)
    prefix_lens = torch.tensor([0, 3, 8], dtype=torch.int32)
    request_lens = torch.tensor([3, 5], dtype=torch.int32)
    delta = SimpleNamespace(
        is_decode_stage=False,
        delta_seq_ids_tensor_device=row_seq_ids,
        delta_position_ids_tensor_device=positions,
        new=SimpleNamespace(
            max_len=5,
            prefix_lens_tensor_device=prefix_lens,
            lens_tensor_device=request_lens,
        ),
    )
    calls = []

    def index_score(actual_q, weights, delta_view, cache_accessor, causal, **kwargs):
        calls.append((actual_q.shape[0], kwargs["ks"].clone(), kwargs["ke"].clone()))
        return torch.zeros(actual_q.shape[0], 5)

    impl = SimpleNamespace(
        impl="hygon",
        append_indexer_kv=lambda *args, **kwargs: None,
        chunk_size=lambda unused_delta: chunk_size,
        index_score=index_score,
    )
    indexer = SimpleNamespace(
        _build_index_qk=lambda *args, **kwargs: ((q, None), (k, None)),
        weights_proj=lambda x: torch.zeros(4, 1),
        n_heads=1,
        softmax_scale=1.0,
        indexer_impl=impl,
        index_topk=2,
    )

    def make_view(base, start, stop):
        return SimpleNamespace(
            base_delta=base,
            is_decode_stage=base.is_decode_stage,
            delta_seq_ids_tensor_device=base.delta_seq_ids_tensor_device[start:stop],
            delta_position_ids_tensor_device=base.delta_position_ids_tensor_device[
                start:stop
            ],
            new=base.new,
        )

    monkeypatch.setattr(
        model_deepseek_v3_module,
        "get_cp_context",
        lambda: SimpleNamespace(pcp_size=2, cp_rank=0),
    )
    monkeypatch.setattr(model_deepseek_v3_module, "BatchedSeqLenDeltaView", make_view)

    result = Indexer._build_index_score(
        indexer,
        torch.zeros(4, 1),
        q,
        k,
        delta,
        object(),
        False,
        object(),
        lambda logits, unused_delta: logits,
        allow_select_all=False,
        empty_output=torch.empty(0, 5),
        freqs_cis_k=object(),
    )

    assert result.shape == (4, 5)
    expected_ks = prefix_lens[row_seq_ids]
    expected_ke = request_lens[row_seq_ids]
    assert torch.all(expected_ks[1:] >= expected_ks[:-1])
    if chunk_size is None:
        assert len(calls) == 1
        assert calls[0][0] == 4
        assert torch.equal(calls[0][1], expected_ks)
        assert torch.equal(calls[0][2], expected_ke)
    else:
        assert [rows for rows, _, _ in calls] == [2, 2]
        assert torch.equal(torch.cat([ks for _, ks, _ in calls]), expected_ks)
        assert torch.equal(torch.cat([ke for _, _, ke in calls]), expected_ke)


@requires_hygon_deepgemm
@pytest.mark.parametrize("rows", [17, 129])
def test_hygon_prefill_compact_source_and_co_match_reference(rows):
    torch.manual_seed(37)
    indexer = object.__new__(DSAIndexer)
    heads = 32
    head_dim = 128
    columns = 257
    width = 129

    q_storage = torch.randn(
        rows,
        heads,
        head_dim + 8,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q = q_storage[..., :head_dim]
    assert not q.is_contiguous()
    k = torch.randn(columns, head_dim, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(rows, heads, dtype=torch.float32, device="cuda")
    ks = torch.cat(
        (
            torch.zeros(rows - 1, dtype=torch.int32, device="cuda"),
            torch.tensor([128], dtype=torch.int32, device="cuda"),
        )
    )
    relative_lengths = torch.full((rows,), width, dtype=torch.int32, device="cuda")

    actual = indexer.bf16_index_score_ragged_qk_dsv32_hygon(
        q,
        weights,
        k,
        SimpleNamespace(new=SimpleNamespace(max_len=width)),
        causal=True,
        ke=relative_lengths,
        ks=ks,
    )

    scores = torch.einsum("mhd,nd->hmn", q.float(), k.float())
    global_logits = (scores.relu() * weights.T[:, :, None]).sum(dim=0)
    expected = global_logits[:, :width].clone()
    expected[-1] = global_logits[-1, 128 : 128 + width]

    torch.cuda.synchronize()
    assert actual.shape == (rows, width)
    assert actual.stride() == (width, 1)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@requires_installed_hygon_deepgemm
def test_installed_hygon_mqa_package_is_single_co_and_compact_by_default():
    hygon_deepgemm = dsa_indexer_module.hygon_deepgemm
    assert dsa_indexer_module._has_hygon_compact_mqa_logits(hygon_deepgemm, True)
    co_files = sorted(
        path.name
        for path in Path(hygon_deepgemm.DEEPGEMM_ASM_DIR).glob(
            "deepgemm_mqa_logits*.co"
        )
    )
    assert co_files == ["deepgemm_mqa_logits.co"]
    assert (
        inspect.signature(hygon_deepgemm.mqa_logits)
        .parameters["compact_output"]
        .default
        is True
    )

    q = torch.empty(1, 32, 128, dtype=torch.float8_e4m3fn, device="cuda")
    k = torch.empty(1, 128, dtype=torch.float8_e4m3fn, device="cuda")
    weights = torch.empty(1, 32, dtype=torch.float32, device="cuda")
    bounds = torch.zeros(1, dtype=torch.int32, device="cuda")
    with pytest.raises(NotImplementedError, match="compact_output=True"):
        hygon_deepgemm.mqa_logits(q, k, weights, bounds, bounds)
