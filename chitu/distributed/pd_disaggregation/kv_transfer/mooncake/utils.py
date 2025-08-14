# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import threading
from collections import deque

import numpy as np
import numpy.typing as npt


class FastQueue:
    """Fast thread-safe queue implementation"""

    def __init__(self):
        self._buf = deque()
        self._cond = threading.Condition()

    def put(self, item):
        """Put item into queue"""
        with self._cond:
            self._buf.append(item)
            self._cond.notify()

    def get(self):
        """Get item from queue (blocking)"""
        with self._cond:
            while not self._buf:
                self._cond.wait()
            return self._buf.popleft()


def group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int32], dst_indices: npt.NDArray[np.int32]
):
    """Group contiguous indices for concurrent transfer"""
    if src_indices.size == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups
