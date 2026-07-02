# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Transfer buffer abstractions and transfer plan generation for PD disaggregation.

Key types:
  - TransferBuffer:   a single (ptr, length) pair describing one contiguous GPU
    memory region.
  - TransferBuffers:  collection of TransferBuffer entries keyed by a composite
    string key — represents one rank's buffers.
"""

import logging
from dataclasses import dataclass, field
from typing import NamedTuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TransferBuffer
# ---------------------------------------------------------------------------


class TransferBuffer(NamedTuple):
    """A single contiguous GPU memory region for transfer.

    Attributes:
        ptr:     GPU data pointer.
        length:  byte length of the region.
    """

    ptr: int
    length: int


# ---------------------------------------------------------------------------
# TransferBuffers
# ---------------------------------------------------------------------------


@dataclass
class TransferBuffers:
    """Collection of transfer buffers for a single rank.

    Structure::

        buffers: dict[key_str: str, list[TransferBuffer]]

    Keys are composite strings
    ``cache_name,req_id,layer_id,block_id,split_id,split_num,replica_id,replica_size``
    so the whole structure is natively msgpack-serializable.

    Session ownership is tracked externally (e.g. ``TaskInfo.recv_buffers``
    is a ``dict[session_id, TransferBuffers]``).
    """

    buffers: dict[str, list[TransferBuffer]] = field(default_factory=dict)

    # ---- mutation ----------------------------------------------------------

    def add(
        self,
        ptr: int,
        length: int,
        *,
        cache_name: str,
        req_id: str,
        layer_id: int,
        block_id: int,
        split_id: int,
        split_num: int,
        replica_id: int,
        replica_size: int,
    ):
        """Register a physical memory region for transfer.

        Args:
            ptr:           GPU data pointer of the contiguous region.
            length:        byte length of the region.
            cache_name:    key in ``paged_kv_cache`` (e.g. ``"main.k"``).
            req_id:        request identifier.
            layer_id:      **global** layer id.
            block_id:      **logical** position in the block table.
            split_id:      starting index in split-space.
            split_num:     number of split units this region covers.
            replica_id:    which replica group this belongs to.
            replica_size:  total number of replica groups.
        """
        key_str = f"{req_id}[{cache_name}]_L{layer_id}_B{block_id}_S{split_id}+{split_num}_R{replica_id}/{replica_size}"
        logger.debug(f"[PD_KV_TRANSFER] {key_str} ptr={ptr} len={length}")
        self.buffers.setdefault(key_str, []).append(TransferBuffer(ptr, length))

    # ---- query -------------------------------------------------------------

    def total_bytes(self) -> int:
        """Sum of all lengths across all entries."""
        return sum(e.length for entries in self.buffers.values() for e in entries)

    def __bool__(self) -> bool:
        return bool(self.buffers)

    # ---- serialization -----------------------------------------------------

    def to_msgpackable(self) -> dict[str, list[list[int]]]:
        """Return the internal buffers dict as-is for msgpack.

        ``TransferBuffer`` is a NamedTuple → tuple → msgpack array.
        """
        return self.buffers

    @classmethod
    def from_msgpackable(cls, data: dict[str, list[list[int]]]) -> "TransferBuffers":
        """Reconstruct from msgpack round-tripped data.

        Each leaf list ``[ptr, length]`` is converted back to ``TransferBuffer``.
        """
        obj = cls()
        for key_str, entries_raw in data.items():
            obj.buffers[key_str] = [TransferBuffer(*e) for e in entries_raw]
        return obj
