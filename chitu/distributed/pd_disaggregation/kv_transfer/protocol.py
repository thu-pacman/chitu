# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Dataclass-based protocol messages for PD disaggregation control plane.

All messages are serialized as a single msgpack frame.
"""

import dataclasses
import logging
from typing import ClassVar, Any, TYPE_CHECKING, TypeVar

import msgpack

if TYPE_CHECKING:
    from chitu.distributed.pd_disaggregation.kv_transfer.transfer_buffers import (
        TransferBuffers,
    )

logger = logging.getLogger(__name__)

# Registry of all protocol message types, keyed by the string "type" field.
_PROTOCOL_TYPE_REGISTRY: dict[str, type] = {}

T = TypeVar("T")


def _register(cls: T) -> T:
    """Decorator: register a dataclass in the protocol type registry."""
    tp = getattr(cls, "type", None)
    if isinstance(tp, str):
        _PROTOCOL_TYPE_REGISTRY[tp] = cls
    return cls


@dataclasses.dataclass
@_register
class DecodeAllocated:
    """Per-request: Decode sends allocated recv_buffers to Prefill.

    Sent after Decode reserves destination blocks.  *session_id* is the
    decode-side Mooncake session so the prefill can route RDMA writes.
    """

    type: ClassVar[str] = "DecodeAllocated"
    decode_sid: int
    dp_rank: int
    req_id: str
    rank_num: int
    session_id: str
    buffers: "TransferBuffers"

    def to_msgpackable(self) -> dict[str, Any]:
        d = ProtocolSerializer.dataclass_to_dict(self)
        d["buffers"] = self.buffers.to_msgpackable()
        return d

    @classmethod
    def from_msgpackable(cls, data: dict[str, Any]) -> "DecodeAllocated":
        from chitu.distributed.pd_disaggregation.kv_transfer.transfer_buffers import (
            TransferBuffers,
        )

        data = {**data, "buffers": TransferBuffers.from_msgpackable(data["buffers"])}
        return ProtocolSerializer.dict_to_dataclass(data, cls)


@dataclasses.dataclass
@_register
class DecodePrepare:
    """Per-request: Router sends to Decode control rank and broadcast to all ranks."""

    type: ClassVar[str] = "DecodePrepare"
    req_id: str
    prefill_sid: int
    prefix_len: int
    new_cache_ids: dict[str, list[int]]
    dp_rank: int = 0


@dataclasses.dataclass
@_register
class RankTransferDone:
    """Each Prefill rank notifies the control rank that its shard has finished.

    Sent once per TransferPlan execution.  The control rank counts these to
    determine when all transfers for a request are complete.
    """

    type: ClassVar[str] = "RankTransferDone"
    req_id: str
    first_token: int = 0
    rank_bytes: dict[str, int] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
@_register
class PrefillDone:
    """Control rank notifies Decode that all transfers for a request are done.

    Carries first_token and num_hit_tokens, replacing the previous aux buffer.
    """

    type: ClassVar[str] = "PrefillDone"
    req_id: str
    first_token: int
    num_hit_tokens: int
    rank_bytes: dict[str, int] = dataclasses.field(default_factory=dict)


class ProtocolSerializer:
    """Msgpack-based single-frame serialization for protocol messages.

    ``pack`` / ``unpack`` check for ``to_msgpackable()`` /
    ``from_msgpackable()`` on the object/class and delegate when present;
    otherwise they fall back to ``dataclass_to_dict`` / ``dict_to_dataclass``.
    """

    @classmethod
    def dataclass_to_dict(cls, obj: Any) -> dict[str, Any]:
        """Extract each dataclass field into a dict without any transformation.

        The ``type`` ClassVar is explicitly included so the receiver can dispatch.
        """
        result: dict[str, Any] = {}
        type_val = getattr(obj, "type", None)
        if isinstance(type_val, str):
            result["type"] = type_val
        for field in dataclasses.fields(obj):
            result[field.name] = getattr(obj, field.name)
        return result

    @classmethod
    def dict_to_dataclass(cls, data: dict[str, Any], target_cls: type[T]) -> T:
        """Construct *target_cls* from dict values without any transformation."""
        kwargs: dict[str, Any] = {}
        for field in dataclasses.fields(target_cls):
            if field.name in data:
                kwargs[field.name] = data[field.name]
        return target_cls(**kwargs)

    @classmethod
    def pack(cls, obj) -> bytes:
        """Serialize a protocol dataclass to msgpack bytes.

        If *obj* has ``to_msgpackable()``, delegate directly; otherwise
        use ``dataclass_to_dict``.
        """
        d = (
            obj.to_msgpackable()
            if hasattr(obj, "to_msgpackable")
            else cls.dataclass_to_dict(obj)
        )
        return msgpack.packb(d, use_bin_type=True)

    @classmethod
    def unpack(cls, data: bytes):
        """Deserialize msgpack bytes to the appropriate protocol dataclass.

        Dispatches by the "type" field.  If the target class has
        ``from_msgpackable()``, delegates directly; otherwise uses
        ``dict_to_dataclass``.
        """
        if not data:
            raise ValueError("empty protocol message")

        d = msgpack.unpackb(data, raw=False)
        if not isinstance(d, dict):
            raise ValueError(f"protocol message must be a dict, got {type(d)}")

        msg_type = d.get("type")
        if not msg_type:
            raise ValueError("protocol message missing 'type' field")

        target_cls = _PROTOCOL_TYPE_REGISTRY.get(msg_type)
        if target_cls is None:
            raise ValueError(f"unknown protocol message type: {msg_type}")

        if hasattr(target_cls, "from_msgpackable"):
            return target_cls.from_msgpackable(d)
        return cls.dict_to_dataclass(d, target_cls)
