# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import itertools
import functools
import torch

from chitu.static_tensor import StaticTensor


class BatchedSeqLen:
    """
    Lengths of different sequences in one batch, accessible from both CPU and GPU.

    To support CUDA graph, initialize this class only once with `max_batch_size`
    set, and update it in place.

    Args:
        lens_list (List[int]): The lengths of different sequences in one batch
        device (torch.device): The device of GPU.
        max_batch_size: Reserved max batch size, used for supporting CUDA graph.
            If not set, use `len(lens_list)` by default.
        max_total_len: Reserved max total length, used for supporting CUDA graph.
            If not set, use `sum(lens_list)` by default. `max_total_len` is only
            effective when `cache_position_ids_tensor_device` is True.
        cache_prefix_lens_tensor_device: If true, `prefix_lens_tensor_device` will
            be cached in a CUDA-graph friendly manner, but will occupy extra space
            even if the value is not used. If False, it will be computed on the fly.
            Defaults to True because `prefix_lens_tensor_device` is used for multiple
            times during one step.
        cache_position_ids_tensor_device: If true, `position_ids_tensor_device` will
            be cached in a CUDA-graph friendly manner, but will occupy extra space
            (large!) even if the value is not used. If False, it will be computed
            on the fly. Defaults to False because currently `position_ids_tensor_device`
            is used only once for precomputing `freqs_cis`.
    """

    def __init__(
        self,
        lens_list: List[int],
        device: torch.device,
        *,
        max_batch_size: Optional[int] = None,
        max_total_len: Optional[int] = None,
        cache_prefix_lens_tensor_device: bool = True,
        cache_position_ids_tensor_device: bool = False,
    ) -> None:
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
        self.prefix_lens_tensor_device_up_to_date = False
        if self.cache_prefix_lens_tensor_device:
            self.prefix_lens_static_tensor_device = StaticTensor(
                max_nelem=max_batch_size + 1, dtype=torch.int32, device=device
            )

        self.cache_position_ids_tensor_device = cache_position_ids_tensor_device
        self.position_ids_tensor_device_up_to_date = False
        if self.cache_position_ids_tensor_device:
            self.position_ids_static_tensor_device = StaticTensor(
                max_nelem=max_total_len, dtype=torch.int32, device=device
            )

    @classmethod
    def from_tokens(
        cls,
        tokens: List[List[int]],
        device: torch.device,
        *,
        max_batch_size: Optional[int] = None,
        max_total_len: Optional[int] = None,
        cache_prefix_lens_tensor_device: bool = True,
        cache_position_ids_tensor_device: bool = False,
    ):
        return cls(
            [len(t) for t in tokens],
            device,
            max_batch_size=max_batch_size,
            max_total_len=max_total_len,
            cache_prefix_lens_tensor_device=cache_prefix_lens_tensor_device,
            cache_position_ids_tensor_device=cache_position_ids_tensor_device,
        )

    def copy_from(self, other: "BatchedSeqLen"):
        assert (
            self.device == other.device
        ), f"Device mismatch: {self.device} vs {other.device}"
        self.lens_list = other.lens_list
        self.lens_static_tensor_device.set(other.lens_tensor_device)

        if self.cache_prefix_lens_tensor_device:
            self.prefix_lens_tensor_device_up_to_date = (
                other.prefix_lens_tensor_device_up_to_date
            )
            if self.prefix_lens_tensor_device_up_to_date:
                self.prefix_lens_static_tensor_device.set(
                    other.prefix_lens_tensor_device
                )

        if self.cache_position_ids_tensor_device:
            self.position_ids_tensor_device_up_to_date = (
                other.position_ids_tensor_device_up_to_date
            )
            if self.position_ids_tensor_device_up_to_date:
                self.position_ids_static_tensor_device.set(
                    other.position_ids_tensor_device
                )

        # `@cached_property` properties can be invalidated by just deleting them
        # See https://docs.python.org/3/library/functools.html#functools.cached_property
        if hasattr(self, "lens_tensor_cpu"):
            del self.lens_tensor_cpu
        if hasattr(self, "prefix_lens_list"):
            del self.prefix_lens_list
        if hasattr(self, "batch_size"):
            del self.batch_size
        if hasattr(self, "total_len"):
            del self.total_len
        if hasattr(self, "max_len"):
            del self.max_len

    @functools.cached_property
    def lens_tensor_cpu(self) -> torch.Tensor:
        return torch.tensor(self.lens_list, device="cpu", dtype=torch.int32)

    @property
    def lens_tensor_device(self) -> torch.Tensor:
        return self.lens_static_tensor_device.get()

    @functools.cached_property
    def prefix_lens_list(self) -> List[int]:
        return list(itertools.accumulate(self.lens_list, initial=0))

    def _comp_prefix_lens_tensor_device(self):
        return torch.cat(
            [
                torch.zeros((1,), device=self.device, dtype=torch.int32),
                torch.cumsum(self.lens_tensor_device, dim=0, dtype=torch.int32),
            ],
            dim=0,
        )

    @property
    def prefix_lens_tensor_device(self) -> torch.Tensor:
        if self.cache_prefix_lens_tensor_device:
            if not self.prefix_lens_tensor_device_up_to_date:
                self.prefix_lens_static_tensor_device.set(
                    self._comp_prefix_lens_tensor_device()
                )
            return self.prefix_lens_static_tensor_device.get()
        else:
            return self._comp_prefix_lens_tensor_device()

    def _comp_position_ids_tensor_device(self):
        x = torch.ones(self.total_len, device=self.device, dtype=torch.int32)
        # Example: x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        x[self.prefix_lens_tensor_device[1:-1]] = 1 - self.lens_tensor_device[:-1]
        # Example: x = [ 1,  1,  1, -2,  1,  1,  1,  1, -4,  1]

        x = torch.cumsum(x, dim=0, dtype=torch.int32) - 1
        # Example: x = [0, 1, 2, 0, 1, 2, 3, 4, 0, 1]

        return x

    @property
    def position_ids_tensor_device(self) -> torch.Tensor:
        if self.cache_position_ids_tensor_device:
            if not self.position_ids_tensor_device_up_to_date:
                self.position_ids_static_tensor_device.set(
                    self._comp_position_ids_tensor_device()
                )
            return self.position_ids_static_tensor_device.get()
        else:
            return self._comp_position_ids_tensor_device()

    @functools.cached_property
    def batch_size(self) -> int:
        return len(self.lens_list)

    @functools.cached_property
    def max_len(self) -> int:
        return int(self.lens_tensor_cpu.max())

    @functools.cached_property
    def total_len(self) -> int:
        return int(self.lens_tensor_cpu.sum())
