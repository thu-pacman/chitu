# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from chitu.batched_seq_len import BatchedSeqLenDelta


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


class TestBatchedSeqLenDelta:
    """BatchedSeqLenDelta 的 is_first_prefill_chunk / is_classic_decoding
    在 copy_from_list 后正确更新的单测

    NOTE: is_first_prefill_chunk 在 __init__ 中被设为 old.total_len == 0，
    但 copy_from_list / copy_from 只更新了 is_classic_decoding 而遗漏了
    is_first_prefill_chunk。当 hybrid prefill 路径依赖该标志判断"是否首次
    prefill chunk"时，续填 chunk（old_len > 0）会错误地进入 prefill-only
    attention，导致 flash attention 读不到历史 KV cache，产生 NaN。
    """

    def test_is_first_prefill_chunk_init_all_zero(self, device):
        """__init__ 传入全零 old_lens 时 is_first_prefill_chunk 应为 True。"""
        sld = BatchedSeqLenDelta(
            [0, 0],
            [128, 128],
            device=device,
            cache_position_ids_tensor_device=False,
            cache_delta_position_ids_tensor_device=False,
        )
        assert sld.is_first_prefill_chunk is True
        assert sld.old.total_len == 0

    def test_is_first_prefill_chunk_init_nonzero(self, device):
        """__init__ 传入非零 old_lens 时 is_first_prefill_chunk 应为 False"""
        sld = BatchedSeqLenDelta(
            [100, 0],
            [200, 100],
            device=device,
            cache_position_ids_tensor_device=False,
            cache_delta_position_ids_tensor_device=False,
        )
        assert sld.is_first_prefill_chunk is False
        assert sld.old.total_len == 100

    def test_copy_from_list_updates_is_first_prefill_chunk(self, device):
        """copy_from_list 从全零 → 非零 old_lens 时，
        is_first_prefill_chunk 必须从 True 变为 False

        hybrid prefill NaN 的原因：copy_from_list 遗漏了
        对 is_first_prefill_chunk 的更新，使得续填 chunk 仍被判定为
        首次 prefill。
        """
        sld = BatchedSeqLenDelta(
            [0, 0],
            [1, 1],
            device=device,
            cache_position_ids_tensor_device=False,
            cache_delta_position_ids_tensor_device=False,
            max_batch_size=2,
        )
        assert sld.is_first_prefill_chunk is True

        sld.copy_from_list([3073, 0], [5119, 2050])
        assert sld.old.total_len == 3073
        assert (
            sld.is_first_prefill_chunk is False
        ), "copy_from_list updated old_lens to [3073, 0] but is_first_prefill_chunk should be False (old.total_len=3073 != 0)"

    def test_copy_from_list_back_to_zero(self, device):
        """copy_from_list 从非零 -> 全零 old_lens 时，
        is_first_prefill_chunk 为 True"""
        sld = BatchedSeqLenDelta(
            [100, 200],
            [200, 300],
            device=device,
            cache_position_ids_tensor_device=False,
            cache_delta_position_ids_tensor_device=False,
            max_batch_size=2,
        )
        assert sld.is_first_prefill_chunk is False

        sld.copy_from_list([0, 0], [4096, 4096])
        assert sld.is_first_prefill_chunk is True

    def test_is_classic_decoding_also_updated(self, device):
        """确认 is_classic_decoding 在 copy_from_list 的正确性"""
        sld = BatchedSeqLenDelta(
            [0, 0],
            [1, 1],
            device=device,
            cache_position_ids_tensor_device=False,
            cache_delta_position_ids_tensor_device=False,
            max_batch_size=2,
        )
        assert sld.is_classic_decoding is False

        sld.copy_from_list([100, 200], [101, 201])
        assert sld.is_classic_decoding is True
        assert sld.is_first_prefill_chunk is False
