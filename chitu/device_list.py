# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections import OrderedDict, deque
import logging
import functools

logger = logging.getLogger(__name__)


class DeviceList:
    """Amortized O(1) appendable list on device"""

    def __init__(self, data=[], dtype=None, device=None):
        self._data = torch.tensor(data, dtype=dtype, device=device)
        self._len = len(data)
        self.append = functools.partial(self._append, disable_reallocate=False)

    def __len__(self):
        return self._len

    def _append(self, item: int, disable_reallocate: bool):
        if self._len == len(self._data):
            if disable_reallocate:
                raise RuntimeError(
                    f"Cannot append: DeviceList exceeds DeviceList's capacity {len(self._data)}"
                )
            new_data = torch.empty(
                max(2 * self._len, 32), dtype=self._data.dtype, device=self._data.device
            )
            new_data[: self._len] = self._data
            self._data = new_data
        self._data[self._len] = item
        self._len += 1

    def to_tensor(self):
        return self._data[: self._len]

    def __getitem__(self, idx):
        return self.to_tensor()[idx]


class StaticDeviceListManager:
    def __init__(
        self,
        max_num_rows: int,
        max_num_cols: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.buffer = torch.empty(
            [max_num_rows + 1, max_num_cols], dtype=dtype, device=device
        )
        self.max_num_rows = max_num_rows
        self.max_num_cols = max_num_cols

        self._free_rows = deque(range(1, max_num_rows + 1))
        self._lru_queue = OrderedDict()  # {DeviceList:row_idx}

    def push_list(self, data: DeviceList):
        """Push a DeviceList into the static buffer"""
        assert isinstance(
            data, DeviceList
        ), f"data should be instance of DeviceList, got {type(data)}."
        assert (
            data._len <= self.max_num_cols
        ), f"DeviceList length {data._len} exceeds max_num_cols {self.max_num_cols}"
        data.append = functools.partial(data._append, disable_reallocate=True)
        if data in self._lru_queue:
            self._refresh(data)
            return
        row_idx = self._assign_buffer_row(data)
        self.buffer[row_idx][: data._len].copy_(data.to_tensor())
        data._data = self.buffer[row_idx]

    def remove_list(self, key: DeviceList):
        """Remove a DeviceList from the buffer management"""
        if key not in self._lru_queue:
            return
        row_idx = self._lru_queue[key]
        self._free_rows.append(row_idx)
        self._lru_queue.pop(key)

    def evict_list(self, key: DeviceList):
        """Evict a DeviceList from buffer, copying data back to non-static storage"""
        assert (
            key in self._lru_queue
        ), f"Attempt to evict the key that not in _lru_queue"
        logger.warning(
            "Evicting DeviceList from buffer, will cause tensor copy, inefficient"
        )
        row_idx = self._lru_queue[key]
        self._lru_queue.pop(key)
        new_data = torch.empty_like(key._data)
        new_data.copy_(key._data)
        key._data = new_data
        key.append = functools.partial(key._append, disable_reallocate=False)
        return row_idx

    def _assign_buffer_row(self, key: DeviceList):
        if key not in self._lru_queue:
            if self._free_rows:
                row_idx = self._free_rows.popleft()
                self._lru_queue[key] = row_idx
            else:
                key_needed_evict = next(iter(self._lru_queue))
                row_idx = self.evict_list(key_needed_evict)
                self._lru_queue[key] = row_idx
        self._refresh(key)
        return self._lru_queue[key]

    def _refresh(self, key):
        """Refresh this key's position in the lru queue"""
        assert key in self._lru_queue
        self._lru_queue.move_to_end(key)

    def batch_append(self, lists: list[DeviceList], data: torch.Tensor):
        if not lists:
            return
        assert data.device == self.buffer.device
        assert data.dtype == self.buffer.dtype
        assert len(lists) == len(
            data
        ), "Number of lists must match length of data tensor"

        rows = [self._lru_queue.get(l, 0) for l in lists]
        cols = [l._len for l in lists]
        self.buffer[rows, cols] = data

        for l in lists:
            l._len += 1

    def capacity(self):
        return self.max_num_rows
