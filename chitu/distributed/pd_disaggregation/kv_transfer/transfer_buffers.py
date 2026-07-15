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
# TransferBufferKey
# ---------------------------------------------------------------------------


class TransferBufferKey(NamedTuple):
    """The identifier for distinguishing different TransferBuffers.
    The same TransferBufferKey corresponds to the same underlying KV cache tensor block.
    Args:
        req_id:        request identifier.
        cache_name:    key in ``paged_kv_cache`` (e.g. ``"main.k"``).
        layer_id:      **global** layer id.
        block_id:      **logical** position in the block table.
        split_id:      starting index in split-space.
        split_len:     number of split units this region covers.
        replica_id:    which replica group this belongs to.
        replica_size:  total number of replica groups.
    """

    req_id: str
    cache_name: str
    layer_id: int
    block_id: int
    split_id: int
    split_len: int
    replica_id: int
    replica_size: int

    @property
    def match_prefix(self) -> tuple:
        """Identity for matching send/recv, ignoring replica placement.

        Two buffers match iff they describe the same (req, cache, layer,
        block, split) region — i.e. everything except the replica id/size.
        """
        return self[:6]


# ---------------------------------------------------------------------------
# TransferBuffers
# ---------------------------------------------------------------------------


@dataclass
class TransferBuffers:
    """Collection of transfer buffers for a single rank.

    Structure::

        buffers: dict[TransferBufferKey, list[TransferBuffer]]

    Keys are :class:`TransferBufferKey` NamedTuples.  Because msgpack has no
    map-key type richer than str/int, the whole structure serializes as a
    *list of ``[key, entries]`` pairs* (see :meth:`to_msgpackable`) rather
    than a dict.

    Session ownership is tracked externally (e.g. ``TaskInfo.recv_buffers``
    is a ``dict[session_id, TransferBuffers]``).
    """

    buffers: dict["TransferBufferKey", list[TransferBuffer]] = field(
        default_factory=dict
    )

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
        split_len: int,
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
            split_len:     number of split units this region covers.
            replica_id:    which replica group this belongs to.
            replica_size:  total number of replica groups.
        """
        key = TransferBufferKey(
            req_id=req_id,
            cache_name=cache_name,
            layer_id=layer_id,
            block_id=block_id,
            split_id=split_id,
            split_len=split_len,
            replica_id=replica_id,
            replica_size=replica_size,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[PD_KV_TRANSFER] %s ptr=%s len=%s", key, ptr, length)
        self.buffers.setdefault(key, []).append(TransferBuffer(ptr, length))

    # ---- query -------------------------------------------------------------

    def total_bytes(self) -> int:
        """Sum of all lengths across all entries."""
        return sum(e.length for entries in self.buffers.values() for e in entries)

    def __bool__(self) -> bool:
        return bool(self.buffers)

    # ---- serialization -----------------------------------------------------

    def to_msgpackable(self) -> list[list]:
        """Serialize as a list of ``[key, entries]`` pairs.

        msgpack cannot use a tuple as a map key, so the dict is flattened to
        a list of pairs.  ``TransferBufferKey`` and ``TransferBuffer`` are both
        NamedTuples → serialized as msgpack arrays.
        """
        return [[list(key), entries] for key, entries in self.buffers.items()]

    @classmethod
    def from_msgpackable(cls, data: list[list]) -> "TransferBuffers":
        """Reconstruct from msgpack round-tripped data.

        Each pair is ``[key_fields, entries]``: ``key_fields`` becomes a
        :class:`TransferBufferKey` and each ``[ptr, length]`` a
        :class:`TransferBuffer`.
        """
        obj = cls()
        for key_raw, entries_raw in data:
            key = TransferBufferKey(*key_raw)
            obj.buffers[key] = [TransferBuffer(*e) for e in entries_raw]
        return obj
