import pytest
import torch

import chitu.models.model_hf_qwen3_next as qwen3_next
from chitu.device_type import is_enflame


@pytest.mark.skipif(not is_enflame(), reason="Enflame GCU is required")
@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize("tail_shape", [(1, 32, 4), (1, 8, 16, 16)])
@pytest.mark.parametrize("duplicate_page", [False, True])
def test_qwen3_next_singleton_index_select_is_exact(
    monkeypatch, batch_size, tail_shape, duplicate_page
):
    num_pages = 2 * batch_size + 3
    cache = torch.arange(
        num_pages * torch.tensor(tail_shape).prod().item(),
        dtype=torch.float32,
        device="cuda",
    ).reshape(num_pages, *tail_shape)
    page_ids = torch.arange(
        batch_size - 1, -1, -1, dtype=torch.int32, device="cuda"
    )
    page_ids = (2 * page_ids + 1).reshape(batch_size, 1).contiguous()
    if duplicate_page:
        page_ids[-1].copy_(page_ids[0])

    def unexpected_fallback(*args, **kwargs):
        raise AssertionError("supported GCU singleton read unexpectedly fell back")

    monkeypatch.setattr(
        qwen3_next, "read_from_singleton_paged_kv_cache", unexpected_fallback
    )

    expected = cache[page_ids.squeeze(1), 0]
    cache_before = cache.clone()
    actual = qwen3_next._read_qwen3_next_singleton_state(
        cache,
        page_ids,
        mtp_accept_indices=None,
        use_index_select=True,
    )

    assert actual.is_contiguous()
    assert torch.equal(actual, expected)
    actual.add_(1)
    assert torch.equal(cache, cache_before)


@pytest.mark.parametrize(
    "batch_size,page_dtype,make_noncontiguous,mtp",
    [
        (1, torch.int32, False, False),
        (3, torch.int32, False, False),
        (4, torch.int64, False, False),
        (4, torch.int32, True, False),
        (4, torch.int32, False, True),
    ],
)
def test_qwen3_next_singleton_index_select_fails_closed(
    monkeypatch, batch_size, page_dtype, make_noncontiguous, mtp
):
    cache = torch.arange(16 * 2 * 3, dtype=torch.float32).reshape(16, 1, 2, 3)
    if make_noncontiguous:
        page_ids = torch.arange(2 * batch_size, dtype=page_dtype)[::2].reshape(
            batch_size, 1
        )
        assert not page_ids.is_contiguous()
    else:
        page_ids = torch.arange(batch_size, dtype=page_dtype).reshape(batch_size, 1)
    mtp_accept_indices = (
        torch.zeros(batch_size, dtype=torch.int64) if mtp else None
    )
    calls = []

    def fallback(kv_cache, page_table, mtp_accept_indices=None):
        calls.append((kv_cache, page_table, mtp_accept_indices))
        if mtp_accept_indices is None:
            return kv_cache[page_table.squeeze(1), 0]
        return kv_cache[page_table.squeeze(1), mtp_accept_indices]

    monkeypatch.setattr(
        qwen3_next, "read_from_singleton_paged_kv_cache", fallback
    )
    actual = qwen3_next._read_qwen3_next_singleton_state(
        cache,
        page_ids,
        mtp_accept_indices=mtp_accept_indices,
        use_index_select=True,
    )

    assert len(calls) == 1
    assert torch.equal(
        actual, fallback(cache, page_ids, mtp_accept_indices)
    )


def test_qwen3_next_singleton_index_select_can_be_disabled(monkeypatch):
    cache = torch.arange(8 * 2 * 3, dtype=torch.float32).reshape(8, 1, 2, 3)
    page_ids = torch.tensor([[3], [1], [6], [2]], dtype=torch.int32)
    calls = []

    def fallback(kv_cache, page_table, mtp_accept_indices=None):
        calls.append(True)
        return kv_cache[page_table.squeeze(1), 0]

    monkeypatch.setattr(
        qwen3_next, "read_from_singleton_paged_kv_cache", fallback
    )
    actual = qwen3_next._read_qwen3_next_singleton_state(
        cache,
        page_ids,
        mtp_accept_indices=None,
        use_index_select=False,
    )

    assert calls == [True]
    assert torch.equal(actual, cache[page_ids.squeeze(1), 0])
