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
            use_position_ids_static_tensor=False,
            use_delta_position_ids_static_tensor=False,
        )
        assert sld.is_first_prefill_chunk is True
        assert sld.old.total_len == 0

    def test_is_first_prefill_chunk_init_nonzero(self, device):
        """__init__ 传入非零 old_lens 时 is_first_prefill_chunk 应为 False"""
        sld = BatchedSeqLenDelta(
            [100, 0],
            [200, 100],
            device=device,
            use_position_ids_static_tensor=False,
            use_delta_position_ids_static_tensor=False,
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
            use_position_ids_static_tensor=False,
            use_delta_position_ids_static_tensor=False,
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
            use_position_ids_static_tensor=False,
            use_delta_position_ids_static_tensor=False,
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
            use_position_ids_static_tensor=False,
            use_delta_position_ids_static_tensor=False,
            max_batch_size=2,
        )
        assert sld.is_classic_decoding is False

        sld.copy_from_list([100, 200], [101, 201])
        assert sld.is_classic_decoding is True
        assert sld.is_first_prefill_chunk is False


class TestBatchedSeqLenDeviceAdvance:
    """advance_classic_by_one 只推进 device 侧，host 镜像 stale 时拒绝读取。"""

    @staticmethod
    def _classic_delta(device):
        return BatchedSeqLenDelta(
            [100, 200],
            [101, 201],
            device=device,
            max_batch_size=2,
            use_position_ids_static_tensor=False,
            use_delta_position_ids_static_tensor=False,
        )

    def test_device_advance_poisons_the_host_mirror(self, device):
        sld = self._classic_delta(device)
        assert sld.is_classic_decoding is True

        sld.advance_classic_by_one()
        sld.advance_classic_by_one()

        assert sld.old.lens_tensor_device.tolist() == [102, 202]
        assert sld.new.lens_tensor_device.tolist() == [103, 203]
        assert sld.batch_size == 2
        assert sld.delta_lens_tensor_device.tolist() == [1, 1]
        assert sld.delta_position_ids_tensor_device.tolist() == [102, 202]
        assert sld.delta_seq_ids_tensor_device.tolist() == [0, 1]
        assert sld.delta_max_len == 1
        assert sld.delta_total_len == 2
        assert sld.new.prefix_lens_tensor_device.tolist() == [0, 103, 306]

        for read_stale in (
            lambda: sld.old.lens_list,
            lambda: sld.new.lens_list,
            lambda: sld.new.lens_tensor_cpu,
            lambda: sld.new.prefix_lens_list,
            lambda: sld.new.total_len,
            lambda: sld.new.max_len,
        ):
            with pytest.raises(RuntimeError, match="stale"):
                read_stale()

    def test_copy_from_list_restores_the_host_mirror(self, device):
        sld = self._classic_delta(device)
        sld.advance_classic_by_one()
        with pytest.raises(RuntimeError, match="stale"):
            _ = sld.new.lens_list

        sld.copy_from_list([50, 60], [51, 61])
        assert sld.old.lens_list == [50, 60]
        assert sld.new.total_len == 112
        assert sld.new.max_len == 61
