# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import itertools
import functools
import torch

from chitu.static_tensor import StaticTensor
from chitu.cuda_graph import cuda_graph_safe_cached_property
from chitu.utils import invalidate_cached_property


class BatchedSeqLen:
    """
    Lengths of different sequences in one batch, accessible from both CPU and GPU.

    To support CUDA graph, initialize this class only once with `max_batch_size`
    set, and update it in place via `copy_from`.

    Args:
        lens_list (list[int]): The lengths of different sequences in one batch
        device (torch.device): The device of GPU.
        max_batch_size: Reserved max batch size, used for supporting CUDA graph.
            If not set, use `len(lens_list)` by default.
        max_total_len: Reserved max total length, used for supporting CUDA graph.
            If not set, use `sum(lens_list)` by default. `max_total_len` is only
            effective when `cache_position_ids_tensor_device` is True.
        cache_prefix_lens_tensor_device: If true, `prefix_lens_tensor_device` will
            be cached in a CUDA-graph friendly manner, but will occupy extra space
            even if the value is not used. If False, it will be computed on the
            fly. Defaults to True because `prefix_lens_tensor_device` is used for
            multiple times during one step.
        cache_position_ids_tensor_device: If true, `position_ids_tensor_device`
            will be cached in a CUDA-graph friendly manner, but will occupy extra
            space even if the value is not used. If False, it will be computed
            on the fly. Defaults to True because `position_ids_tensor_device` is
            used for multiple times during one step for chunked prefilling.
        cache_seq_ids_tensor_device: If true, `seq_ids_tensor_device` will be cached
            in a CUDA-graph friendly manner, but will occupy extra space even if
            the value is not used. If False, it will be computed on the fly.
            Defaults to True because `seq_ids_tensor_device` is used for multiple
            times during one step for chunked prefilling.
    """

    def __init__(
        self,
        lens_list: list[int],
        device: torch.device | str,
        *,
        max_batch_size: Optional[int] = None,
        max_total_len: Optional[int] = None,
        cache_prefix_lens_tensor_device: bool = True,
        cache_position_ids_tensor_device: bool = True,
        cache_seq_ids_tensor_device: bool = True,
    ) -> None:
        assert all(l >= 0 for l in lens_list)

        if max_batch_size is None:
            max_batch_size = len(lens_list)
        if max_total_len is None:
            max_total_len = sum(lens_list)

        self.lens_list = lens_list
        self.device = device
        self.lens_static_tensor_device = StaticTensor(
            torch.tensor(self.lens_list, device=self.device, dtype=torch.int32),
            max_nelem=max_batch_size,
        )

        self.cache_prefix_lens_tensor_device = cache_prefix_lens_tensor_device
        self._prefix_lens_tensor_device_up_to_date = False
        if self.cache_prefix_lens_tensor_device:
            self._prefix_lens_static_tensor_device = StaticTensor(
                max_nelem=max_batch_size + 1, dtype=torch.int32, device=device
            )

        self.cache_position_ids_tensor_device = cache_position_ids_tensor_device
        self._position_ids_tensor_device_up_to_date = False
        if self.cache_position_ids_tensor_device:
            self._position_ids_static_tensor_device = StaticTensor(
                max_nelem=max_total_len, dtype=torch.int32, device=device
            )

        self.cache_seq_ids_tensor_device = cache_seq_ids_tensor_device
        self._seq_ids_tensor_device_up_to_date = False
        if self.cache_seq_ids_tensor_device:
            self._seq_ids_static_tensor_device = StaticTensor(
                max_nelem=max_total_len, dtype=torch.int32, device=device
            )

    @classmethod
    def from_tokens(
        cls,
        tokens: list[list[int]],
        device: torch.device | str,
        *,
        max_batch_size: Optional[int] = None,
        max_total_len: Optional[int] = None,
        cache_prefix_lens_tensor_device: bool = True,
        cache_position_ids_tensor_device: bool = True,
        cache_seq_ids_tensor_device: bool = True,
    ):
        return cls(
            [len(t) for t in tokens],
            device,
            max_batch_size=max_batch_size,
            max_total_len=max_total_len,
            cache_prefix_lens_tensor_device=cache_prefix_lens_tensor_device,
            cache_position_ids_tensor_device=cache_position_ids_tensor_device,
            cache_seq_ids_tensor_device=cache_seq_ids_tensor_device,
        )

    def copy_from_list(self, lens_list: list[int]):
        self.lens_list = lens_list
        self.lens_static_tensor_device.set(
            torch.tensor(self.lens_list, device=self.device, dtype=torch.int32)
        )

        if self.cache_prefix_lens_tensor_device:
            self._prefix_lens_tensor_device_up_to_date = False
        if self.cache_position_ids_tensor_device:
            self._position_ids_tensor_device_up_to_date = False
        if self.cache_seq_ids_tensor_device:
            self._seq_ids_tensor_device_up_to_date = False

        invalidate_cached_property(self, "lens_tensor_cpu")
        invalidate_cached_property(self, "prefix_lens_list")
        invalidate_cached_property(self, "batch_size")
        invalidate_cached_property(self, "total_len")
        invalidate_cached_property(self, "max_len")

    def copy_from(self, other: "BatchedSeqLen"):
        assert (
            self.device == other.device
        ), f"Device mismatch: {self.device} vs {other.device}"
        self.lens_list = other.lens_list
        self.lens_static_tensor_device.set(other.lens_tensor_device)

        if self.cache_prefix_lens_tensor_device:
            self._prefix_lens_tensor_device_up_to_date = (
                other._prefix_lens_tensor_device_up_to_date
            )
            if self._prefix_lens_tensor_device_up_to_date:
                self._prefix_lens_static_tensor_device.set(
                    other.prefix_lens_tensor_device
                )

        if self.cache_position_ids_tensor_device:
            self._position_ids_tensor_device_up_to_date = (
                other._position_ids_tensor_device_up_to_date
            )
            if self._position_ids_tensor_device_up_to_date:
                self._position_ids_static_tensor_device.set(
                    other.position_ids_tensor_device
                )

        if self.cache_seq_ids_tensor_device:
            self._seq_ids_tensor_device_up_to_date = (
                other._seq_ids_tensor_device_up_to_date
            )
            if self._seq_ids_tensor_device_up_to_date:
                self._seq_ids_static_tensor_device.set(other.seq_ids_tensor_device)

        invalidate_cached_property(self, "lens_tensor_cpu")
        invalidate_cached_property(self, "prefix_lens_list")
        invalidate_cached_property(self, "batch_size")
        invalidate_cached_property(self, "total_len")
        invalidate_cached_property(self, "max_len")

    @functools.cached_property
    def lens_tensor_cpu(self) -> torch.Tensor:
        return torch.tensor(self.lens_list, device="cpu", dtype=torch.int32)

    @property
    def lens_tensor_device(self) -> torch.Tensor:
        return self.lens_static_tensor_device.get()

    @functools.cached_property
    def prefix_lens_list(self) -> list[int]:
        return list(itertools.accumulate(self.lens_list, initial=0))

    @cuda_graph_safe_cached_property(
        static_tensor_name="_prefix_lens_static_tensor_device",
        up_to_date_flag_name="_prefix_lens_tensor_device_up_to_date",
        enable_flag_name="cache_prefix_lens_tensor_device",
    )
    def prefix_lens_tensor_device(self) -> torch.Tensor:
        return torch.cat(
            [
                torch.zeros((1,), device=self.device, dtype=torch.int32),
                torch.cumsum(self.lens_tensor_device, dim=0, dtype=torch.int32),
            ],
            dim=0,
        )

    @cuda_graph_safe_cached_property(
        static_tensor_name="_position_ids_static_tensor_device",
        up_to_date_flag_name="_position_ids_tensor_device_up_to_date",
        enable_flag_name="cache_position_ids_tensor_device",
    )
    def position_ids_tensor_device(self) -> torch.Tensor:
        # Example: lens = [3, 5, 2], total_len = 10

        # Create global positions
        positions = torch.arange(self.total_len, device=self.device, dtype=torch.int32)
        # Example: positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Get sequence IDs for each position
        seq_ids = self.seq_ids_tensor_device
        # Example: seq_ids = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]

        # Get prefix lengths for each position's sequence
        # Use index_select instead of indexing
        prefix_lens = torch.index_select(self.prefix_lens_tensor_device, 0, seq_ids)
        # Example: prefix_lens = [0, 0, 0, 3, 3, 3, 3, 3, 8, 8]

        # Calculate relative position within each sequence
        x = positions - prefix_lens
        # Example: x = [0, 1, 2, 0, 1, 2, 3, 4, 0, 1]

        return x

    @cuda_graph_safe_cached_property(
        static_tensor_name="_seq_ids_static_tensor_device",
        up_to_date_flag_name="_seq_ids_tensor_device_up_to_date",
        enable_flag_name="cache_seq_ids_tensor_device",
    )
    def seq_ids_tensor_device(self) -> torch.Tensor:
        # Example: lens = [3, 5, 2], prefix_lens = [0, 3, 8, 10]

        # Use searchsorted to find which sequence each position belongs to
        positions = torch.arange(self.total_len, device=self.device, dtype=torch.int32)

        # searchsorted returns int64 by default, so  cast to int32
        ret = torch.searchsorted(
            self.prefix_lens_tensor_device[1:], positions, right=True
        ).to(torch.int32)
        # Example: ret = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]

        return ret

    @functools.cached_property
    def batch_size(self) -> int:
        return len(self.lens_list)

    @functools.cached_property
    def max_len(self) -> int:
        return int(self.lens_tensor_cpu.max())

    @functools.cached_property
    def total_len(self) -> int:
        return int(self.lens_tensor_cpu.sum())


class BatchedSeqLenDelta:
    """
    Increment of a BatchedSeqLen, including the old, the new, and the difference.

    Args:
        old_len_list: Optional initial value of the old `lens_list`.
        new_len_list: Optional initial value of the new `lens_list`.
        device (torch.device): The device of GPU.
        max_batch_size: Reserved max batch size, used for supporting CUDA graph.
            If not set, use `len(lens_list)` by default.
        max_total_len: Reserved max total length, used for supporting CUDA graph.
            If not set, use `sum(lens_list)` by default. `max_total_len` is only
            effective when `cache_position_ids_tensor_device` is True.
        max_total_delta_len: Reserved max total length of the extra tokens added
            from the old sequence length to the new sequence length, used for
            supporting CUDA graph. `max_total_delta_len` is only effective only
            when `cache_delta_position_ids_tensor_device` is True, and must be set
            if `cache_delta_position_ids_tensor_device` is True.
        cache_prefix_lens_tensor_device: If true, `prefix_lens_tensor_device` will
            be cached in a CUDA-graph friendly manner, but will occupy extra space
            even if the value is not used. If False, it will be computed on the
            fly. Defaults to True because `prefix_lens_tensor_device` is used for
            multiple times during one step.
        cache_position_ids_tensor_device: If true, `position_ids_tensor_device`
            will be cached in a CUDA-graph friendly manner, but will occupy extra
            space even if the value is not used. If False, it will be computed
            on the fly. Defaults to True because `position_ids_tensor_device` is
            used for multiple times during one step for chunked prefilling.
        cache_delta_prefix_lens_tensor_device: If true, `delta_prefix_lens_tensor_device`
            will be cached in a CUDA-graph friendly manner, but will occupy extra
            space even if the value is not used. If False, it will be computed on
            the fly. Defaults to True.
        cache_delta_position_ids_tensor_device: If true, `delta_position_ids_tensor_device`
            will be cached in a CUDA-graph friendly manner, but will occupy extra
            space even if the value is not used. If False, it will be computed on
            the fly. Defaults to True.
        cache_delta_seq_ids_tensor_device: If true, `delta_seq_ids_tensor_device`
            will be cached in a CUDA-graph friendly manner, but will occupy extra
            space even if the value is not used. If False, it will be computed on
            the fly. Defaults to True.
    """

    def __init__(
        self,
        old_len_list: list[int] = [],
        new_len_list: list[int] = [],
        *,
        device: torch.device | str,
        max_batch_size: Optional[int] = None,
        max_total_len: Optional[int] = None,
        max_total_delta_len: Optional[int] = None,
        cache_prefix_lens_tensor_device: bool = True,
        cache_position_ids_tensor_device: bool = True,
        cache_seq_ids_tensor_device: bool = True,
        cache_delta_prefix_lens_tensor_device: bool = True,
        cache_delta_position_ids_tensor_device: bool = True,
        cache_delta_seq_ids_tensor_device: bool = True,
    ):
        self.device = device

        self.old = BatchedSeqLen(
            old_len_list,
            device=device,
            max_batch_size=max_batch_size,
            max_total_len=max_total_len,
            cache_prefix_lens_tensor_device=cache_prefix_lens_tensor_device,
            cache_position_ids_tensor_device=cache_position_ids_tensor_device,
            cache_seq_ids_tensor_device=cache_seq_ids_tensor_device,
        )
        self.new = BatchedSeqLen(
            new_len_list,
            device=device,
            max_batch_size=max_batch_size,
            max_total_len=max_total_len,
            cache_prefix_lens_tensor_device=cache_prefix_lens_tensor_device,
            cache_position_ids_tensor_device=cache_position_ids_tensor_device,
            cache_seq_ids_tensor_device=cache_seq_ids_tensor_device,
        )
        self._delta = BatchedSeqLen(
            [x - y for x, y in zip(self.new.lens_list, self.old.lens_list)],
            device=device,
            max_batch_size=max_batch_size,
            max_total_len=max_total_delta_len,
            cache_prefix_lens_tensor_device=cache_delta_prefix_lens_tensor_device,
            cache_position_ids_tensor_device=False,  # NOTE: delta_position_ids is NOT _delta.position_ids
            cache_seq_ids_tensor_device=cache_delta_seq_ids_tensor_device,
        )

        self.is_classic_decoding = all(x > 0 for x in self.old.lens_list) and all(
            (x + 1 == y for x, y in zip(self.old.lens_list, self.new.lens_list))
        )

        self.cache_delta_position_ids_tensor_device = (
            cache_delta_position_ids_tensor_device
        )
        self._delta_position_ids_tensor_device_up_to_date = False
        if self.cache_delta_position_ids_tensor_device:
            assert max_total_delta_len is not None
            self._delta_position_ids_static_tensor_device = StaticTensor(
                max_nelem=max_total_delta_len, dtype=torch.int32, device=device
            )

    def copy_from_list(self, old_len_list: list[int], new_len_list: list[int]):
        self.old.copy_from_list(old_len_list)
        self.new.copy_from_list(new_len_list)
        self._delta.copy_from_list(
            [x - y for x, y in zip(self.new.lens_list, self.old.lens_list)]
        )
        self.is_classic_decoding = all(x > 0 for x in self.old.lens_list) and all(
            (x + 1 == y for x, y in zip(self.old.lens_list, self.new.lens_list))
        )
        self._delta_position_ids_tensor_device_up_to_date = False

    def copy_from(self, other_old: BatchedSeqLen, other_new: BatchedSeqLen):
        self.old.copy_from(other_old)
        self.new.copy_from(other_new)
        self._delta.copy_from(
            BatchedSeqLen(
                [x - y for x, y in zip(self.new.lens_list, self.old.lens_list)],
                device=self.device,
                cache_prefix_lens_tensor_device=False,
                cache_position_ids_tensor_device=False,
                cache_seq_ids_tensor_device=False,
            )
        )
        self.is_classic_decoding = all(x > 0 for x in self.old.lens_list) and all(
            (x + 1 == y for x, y in zip(self.old.lens_list, self.new.lens_list))
        )
        self._delta_position_ids_tensor_device_up_to_date = False

    @property
    def batch_size(self):
        ret = self.old.batch_size
        assert self.new.batch_size == ret
        return ret

    @property
    def delta_total_len(self):
        if self.is_classic_decoding:
            return self.batch_size
        else:
            return self._delta.total_len

    @property
    def delta_max_len(self):
        if self.is_classic_decoding:
            return 1
        else:
            return self._delta.max_len

    @property
    def delta_lens_list(self):
        if self.is_classic_decoding:
            return [1] * self.batch_size
        else:
            return self._delta.lens_list

    @property
    def delta_lens_tensor_device(self):
        if self.is_classic_decoding:
            return torch.ones(self.batch_size, device=self.device, dtype=torch.int32)
        else:
            return self._delta.lens_tensor_device

    @property
    def delta_prefix_lens_list(self):
        if self.is_classic_decoding:
            return list(range(self.batch_size + 1))
        else:
            return self._delta.prefix_lens_list

    @property
    def delta_prefix_lens_tensor_device(self):
        if self.is_classic_decoding:
            return torch.arange(
                self.batch_size + 1, device=self.device, dtype=torch.int32
            )
        else:
            return self._delta.prefix_lens_tensor_device

    @cuda_graph_safe_cached_property(
        static_tensor_name="_delta_position_ids_static_tensor_device",
        up_to_date_flag_name="_delta_position_ids_tensor_device_up_to_date",
        enable_flag_name="cache_delta_position_ids_tensor_device",
    )
    def _delta_position_ids_tensor_device_impl(self):
        # Example: old = [10, 20, 30], new = [13, 25, 32]

        delta_lens = self.new.lens_tensor_device - self.old.lens_tensor_device
        # Example: delta_lens = [3, 5, 2]

        prefix_delta_lens = torch.cumsum(delta_lens, dim=0, dtype=torch.int32)
        # Example: prefix_delta_lens = [3, 8, 10]

        delta_total_len = self.new.total_len - self.old.total_len
        positions = torch.arange(
            delta_total_len, device=self.old.device, dtype=torch.int32
        )
        # Example: positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        seq_ids = torch.searchsorted(prefix_delta_lens, positions, right=True).to(
            torch.int32
        )
        # Example: seq_ids = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]

        old_lens = torch.index_select(self.old.lens_tensor_device, 0, seq_ids)
        # Example: old_lens = [10, 10, 10, 20, 20, 20, 20, 20, 30, 30]

        prefix_with_zero = torch.cat(
            [
                torch.zeros(1, device=self.old.device, dtype=torch.int32),
                prefix_delta_lens[:-1],
            ]
        )
        prefix_deltas = torch.index_select(prefix_with_zero, 0, seq_ids)
        # Example: prefix_deltas = [0, 0, 0, 3, 3, 3, 3, 3, 8, 8]

        x = old_lens + (positions - prefix_deltas)
        # Example: x = [10, 11, 12, 20, 21, 22, 23, 24, 30, 31]

        return x

    @property
    def delta_position_ids_tensor_device(self):
        # NOTE: delta_position_ids is NOT _delta.position_ids
        if self.is_classic_decoding:
            return self.old.lens_tensor_device
        else:
            return self._delta_position_ids_tensor_device_impl

    @property
    def delta_seq_ids_tensor_device(self):
        if self.is_classic_decoding:
            return torch.arange(self.batch_size, device=self.device, dtype=torch.int32)
        else:
            return self._delta.seq_ids_tensor_device
