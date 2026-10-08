# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

import chitu.batched_seq_len as batched_seq_len_module
from chitu.batched_seq_len import BatchedSeqLenDelta


def test_classic_decode_reuses_seq_ids_and_exact_post_append_lengths(monkeypatch):
    monkeypatch.setattr(batched_seq_len_module, "_CACHE_CLASSIC_DECODE_SEQ_IDS", True)
    monkeypatch.setattr(batched_seq_len_module, "_CACHE_POST_APPEND_LENS", True)
    seq_len_delta = BatchedSeqLenDelta(
        [3, 9],
        [4, 10],
        device="cpu",
        max_batch_size=2,
        max_total_len=32,
        max_total_delta_len=2,
    )

    seq_ids_first = seq_len_delta.delta_seq_ids_tensor_device
    seq_ids_second = seq_len_delta.delta_seq_ids_tensor_device
    post_first = seq_len_delta.post_append_lens_tensor_device
    post_second = seq_len_delta.post_append_lens_tensor_device

    assert torch.equal(seq_ids_first, torch.tensor([0, 1], dtype=torch.int32))
    assert seq_ids_first.data_ptr() == seq_ids_second.data_ptr()
    assert torch.equal(post_first, torch.tensor([4, 10], dtype=torch.int32))
    assert post_first.data_ptr() == post_second.data_ptr()

    seq_len_delta.copy_from_list([4, 10], [5, 11])
    post_third = seq_len_delta.post_append_lens_tensor_device
    assert torch.equal(post_third, torch.tensor([5, 11], dtype=torch.int32))
    assert post_third.data_ptr() == post_first.data_ptr()


def test_non_classic_decode_uses_new_lengths_directly(monkeypatch):
    monkeypatch.setattr(batched_seq_len_module, "_CACHE_POST_APPEND_LENS", True)
    seq_len_delta = BatchedSeqLenDelta(
        [3, 9],
        [5, 12],
        device="cpu",
        max_batch_size=2,
        max_total_len=32,
        max_total_delta_len=5,
    )

    assert (
        seq_len_delta.post_append_lens_tensor_device.data_ptr()
        == seq_len_delta.new.lens_tensor_device.data_ptr()
    )
    assert torch.equal(
        seq_len_delta.post_append_lens_tensor_device,
        torch.tensor([5, 12], dtype=torch.int32),
    )


def test_classic_decode_without_cache_derives_from_live_old_lengths(monkeypatch):
    monkeypatch.setattr(batched_seq_len_module, "_CACHE_POST_APPEND_LENS", False)
    seq_len_delta = BatchedSeqLenDelta(
        [3, 9],
        [4, 10],
        device="cpu",
        max_batch_size=2,
        max_total_len=32,
        max_total_delta_len=2,
    )

    # Reproduce the scheduler's minimal-metadata path: ``new`` changes on the
    # host only, while the live ``old`` device tensor is refreshed.
    seq_len_delta.copy_from_list_classic_decode(
        [4, 10], [5, 11], old_device_tensor_is_used=True
    )

    assert torch.equal(
        seq_len_delta.new.lens_tensor_device,
        torch.tensor([4, 10], dtype=torch.int32),
    )
    assert torch.equal(
        seq_len_delta.post_append_lens_tensor_device,
        torch.tensor([5, 11], dtype=torch.int32),
    )
