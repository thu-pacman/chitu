# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from chitu.batched_seq_len import BatchedSeqLenDelta


@pytest.fixture
def device():
    if torch.cuda.is_available():
        # The device a tensor actually lands on, not the name asked for: on Ascend
        # the NPU runtime hands back an `npu:0` tensor for a `cuda:0` request, and
        # the device checks under test compare the two.
        return torch.zeros(1, device="cuda:0").device
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


class TestBatchedSeqLenDeviceSet:
    """`set_lens_device` / `update_host_mirror`: the device-side length update.

    The primitives the KV cache uses to advance a decode step's positions from device
    data (the MTP accept counts) instead of host lists. The tests pin the contract: the
    device tensors move, the host mirrors can be restored by host-side arithmetic, and
    the derived device tensors agree with what `copy_from_list` produces.
    """

    K = 4

    @staticmethod
    def _verify_delta(device, old_lens, new_lens):
        return BatchedSeqLenDelta(
            old_lens,
            new_lens,
            device=device,
            max_batch_size=len(old_lens),
            max_total_len=256,
            max_total_delta_len=64,
            use_position_ids_static_tensor=False,
        )

    def test_set_lens_device_matches_copy_from_list(self, device):
        K = self.K
        old_host = [7, 11]
        new_host = [x + K for x in old_host]
        sld = self._verify_delta(device, old_host, new_host)
        ref = self._verify_delta(device, old_host, new_host)

        for accept in ([0, 3], [2, 1], [3, 3], [0, 2]):
            accept_device = torch.tensor(accept, dtype=torch.int32, device=device)
            # accepted = new + accept + 1 - K: the window the previous step
            # wrote, rewound by the tokens it speculated but did not accept.
            accepted_device = sld.new.lens_tensor_device + (accept_device + 1 - K)
            accepted_host = [x + a + 1 - K for x, a in zip(new_host, accept)]

            sld.set_lens_device(
                accepted_device,
                accepted_device + K,
                is_classic_decoding=False,
                delta_len_host=[K, K],
            )
            sld.old.update_host_mirror(accepted_host)
            sld.new.update_host_mirror([x + K for x in accepted_host])

            old_host, new_host = accepted_host, [x + K for x in accepted_host]
            ref.copy_from_list(old_host, new_host)

            assert sld.old.lens_tensor_device.tolist() == old_host
            assert sld.new.lens_tensor_device.tolist() == new_host
            assert sld.old.lens_list == old_host
            assert sld.new.lens_list == new_host
            assert sld.new.prefix_lens_tensor_device.tolist() == (
                ref.new.prefix_lens_tensor_device.tolist()
            )
            assert sld.delta_lens_tensor_device.tolist() == [K, K]
            assert sld.delta_total_len == ref.delta_total_len == K * 2
            assert sld.delta_max_len == ref.delta_max_len == K
            assert sld.delta_position_ids_tensor_device.tolist() == (
                ref.delta_position_ids_tensor_device.tolist()
            )
            assert sld.delta_seq_ids_tensor_device.tolist() == (
                ref.delta_seq_ids_tensor_device.tolist()
            )
            assert sld.is_classic_decoding is False
            assert sld.is_first_prefill_chunk is False

    def test_set_lens_device_classic_matches_copy_from_list(self, device):
        """The draft window (`accepted .. accepted + 1`) is classic decoding."""
        old_host = [20, 31]
        new_host = [x + 1 for x in old_host]
        sld = self._verify_delta(device, old_host, new_host)
        ref = self._verify_delta(device, old_host, new_host)

        old_device = torch.tensor(old_host, dtype=torch.int32, device=device)
        sld.set_lens_device(
            old_device,
            old_device + 1,
            is_classic_decoding=True,
            delta_len_host=[1, 1],
        )
        sld.old.update_host_mirror(old_host)
        sld.new.update_host_mirror(new_host)

        assert sld.old.lens_tensor_device.tolist() == old_host
        assert sld.new.lens_tensor_device.tolist() == new_host
        assert sld.old.lens_list == old_host
        assert sld.is_classic_decoding is True
        assert sld.delta_lens_tensor_device.tolist() == [1, 1]
        assert sld.delta_total_len == ref.delta_total_len == 2
        assert sld.delta_max_len == ref.delta_max_len == 1
        assert sld.delta_position_ids_tensor_device.tolist() == old_host
        assert sld.delta_seq_ids_tensor_device.tolist() == [0, 1]

    def test_set_lens_device_poisons_the_host_mirror(self, device):
        sld = self._verify_delta(device, [7, 11], [11, 15])
        old_device = torch.tensor([9, 13], dtype=torch.int32, device=device)

        sld.set_lens_device(
            old_device,
            old_device + self.K,
            is_classic_decoding=False,
            delta_len_host=[self.K, self.K],
        )

        # The device state moved...
        assert sld.old.lens_tensor_device.tolist() == [9, 13]
        assert sld.new.lens_tensor_device.tolist() == [13, 17]
        assert sld.batch_size == 2
        # ... the derived device tensors still work off the host-maintained
        # `_delta` mirror ...
        assert sld.delta_total_len == self.K * 2
        assert sld.delta_position_ids_tensor_device.tolist() == [
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
        ]
        # ... and the per-sequence mirrors are poisoned instead of silently stale.
        for read_stale in (
            lambda: sld.old.lens_list,
            lambda: sld.new.lens_list,
            lambda: sld.new.total_len,
            lambda: sld.new.prefix_lens_list,
        ):
            with pytest.raises(RuntimeError, match="stale"):
                read_stale()

    def test_update_host_mirror_restores_reads(self, device):
        sld = self._verify_delta(device, [7, 11], [11, 15])
        old_device = torch.tensor([9, 13], dtype=torch.int32, device=device)
        sld.set_lens_device(
            old_device,
            old_device + self.K,
            is_classic_decoding=False,
            delta_len_host=[self.K, self.K],
        )

        sld.old.update_host_mirror([9, 13])
        sld.new.update_host_mirror([13, 17])

        assert sld.old.lens_list == [9, 13]
        assert sld.new.lens_list == [13, 17]
        assert sld.new.total_len == 30
        assert sld.new.max_len == 17
        assert sld.new.prefix_lens_list == [0, 13, 30]
        assert sld.old.lens_tensor_cpu.tolist() == [9, 13]
        # `_delta` was left alone by these two calls, so its own mirror still
        # holds the host values it was explicitly given.
        assert sld.delta_lens_list == [self.K, self.K]

    def test_update_host_mirror_follows_a_batch_size_change(self, device):
        """The mirror tracks the device batch size, not the previous mirror.

        Consecutive steps may run different batch sizes (warmup walks the batch-size
        ladder), and the device tensors are resized in place by `set_lens_device()`.
        """
        sld = BatchedSeqLenDelta(
            [7, 11],
            [8, 12],
            device=device,
            max_batch_size=4,
            max_total_len=256,
            max_total_delta_len=64,
            use_position_ids_static_tensor=False,
        )
        wide = torch.tensor([3, 4, 5], dtype=torch.int32, device=device)
        sld.set_lens_device(
            wide,
            wide + 1,
            is_classic_decoding=True,
            delta_len_host=[1, 1, 1],
        )
        sld.old.update_host_mirror([3, 4, 5])
        sld.new.update_host_mirror([4, 5, 6])

        assert sld.batch_size == 3
        assert sld.old.lens_list == [3, 4, 5]
        assert sld.new.lens_list == [4, 5, 6]
        assert sld.new.total_len == 15
        assert sld.new.prefix_lens_list == [0, 4, 9, 15]
        assert sld.delta_lens_list == [1, 1, 1]
        assert sld.delta_position_ids_tensor_device.tolist() == [3, 4, 5]

    def test_set_lens_device_resize_keeps_batch_size_on_the_device(self, device):
        """A device resize that the host mirror does not follow still sizes right.

        The deferred draft window (`KVCache.prepare_mtp_cache_decode`) sets the
        device rows for the batch the draft runs on and leaves the mirrors, which
        only the sync phase can correct. When the batch shrank since the previous
        step, the stale mirror also has the wrong *length*, and the derived
        device quantities (`delta_seq_ids_tensor_device`, `delta_lens_*`,
        `delta_prefix_lens_*`) index by row count -- so they have to read the
        device's, not the mirror's.
        """
        sld = self._verify_delta(device, [7, 11], [11, 15])
        # Read the batch size while the mirror is still valid: the value caches,
        # and a resize that only poisons the mirror has to invalidate it.
        assert sld.batch_size == 2
        assert sld.old.batch_size == 2
        narrow = torch.tensor([9], dtype=torch.int32, device=device)
        sld.set_lens_device(
            narrow,
            narrow + self.K,
            is_classic_decoding=True,
            delta_len_host=[self.K],
        )

        assert sld.batch_size == 1
        assert sld.old.batch_size == 1
        assert sld.new.batch_size == 1
        assert sld.delta_seq_ids_tensor_device.tolist() == [0]
        assert sld.delta_lens_tensor_device.tolist() == [1]
        assert sld.delta_prefix_lens_tensor_device.tolist() == [0, 1]
        assert sld.delta_total_len == 1
        assert sld.delta_position_ids_tensor_device.tolist() == [9]
        # The lengths themselves are still the device's to tell, so the mirror
        # keeps raising rather than reporting the batch it used to describe.
        with pytest.raises(RuntimeError, match="stale"):
            sld.old.lens_list

    def test_set_lens_device_rejects_another_device(self, device):
        if device.type == "cpu":
            pytest.skip("needs an accelerator device to have a mismatching one")
        sld = self._verify_delta(device, [7, 11], [11, 15])
        with pytest.raises(ValueError, match="Device mismatch"):
            sld.old.set_lens_device(torch.tensor([1, 2], dtype=torch.int32))

    def test_update_host_mirror_rejects_wrong_length(self, device):
        sld = self._verify_delta(device, [7, 11], [11, 15])
        # `ValueError`, like `set_lens_device()`: the length is the caller's contract, and
        # `python -O` must not be able to strip the check.
        with pytest.raises(ValueError, match="batch size"):
            sld.old.update_host_mirror([1, 2, 3])
