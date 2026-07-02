# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the dataclass-based protocol messages and msgpack serialization.
"""

import pytest
import torch
import msgpack

from chitu.distributed.pd_disaggregation.kv_transfer.protocol import (
    DecodeAllocated,
    RankTransferDone,
    PrefillDone,
    ProtocolSerializer,
)
from chitu.distributed.pd_disaggregation.kv_transfer.transfer_buffers import (
    TransferBuffers,
)

_KW = dict(
    cache_name="main",
    req_id="test",
    layer_id=0,
    block_id=0,
    split_id=0,
    split_num=1,
    replica_id=0,
    replica_size=1,
)


def _make_tensor(shape):
    return torch.randn(shape).contiguous()


def _add(bufs, data, **kw):
    """Shortcut to add a tensor to TransferBuffers."""
    kd = dict(_KW)
    kd.update(kw)
    bufs.add(data.data_ptr(), data.numel() * data.element_size(), **kd)
    return f"{kd['req_id']}[{kd['cache_name']}]_L{kd['layer_id']}_B{kd['block_id']}_S{kd['split_id']}+{kd['split_num']}_R{kd['replica_id']}/{kd['replica_size']}"


class TestDecodeAllocated:
    def test_roundtrip_with_buffers(self):
        bufs = TransferBuffers()
        d1 = _make_tensor((10,))
        d2 = _make_tensor((20,))

        ks1 = _add(bufs, d1, cache_name="main")
        ks2 = _add(bufs, d2, cache_name="linear")

        msg = DecodeAllocated(
            decode_sid=0,
            dp_rank=0,
            req_id="test-request-1",
            rank_num=1,
            session_id="s1",
            buffers=bufs,
        )
        packed = ProtocolSerializer.pack(msg)
        unpacked = ProtocolSerializer.unpack(packed)

        assert isinstance(unpacked, DecodeAllocated)
        assert unpacked.req_id == "test-request-1"
        assert unpacked.session_id == "s1"
        restored = unpacked.buffers
        assert isinstance(restored, TransferBuffers)
        assert ks1 in restored.buffers
        assert ks2 in restored.buffers

    def test_empty_buffers(self):
        msg = DecodeAllocated(
            decode_sid=0,
            dp_rank=0,
            req_id="empty",
            rank_num=1,
            session_id="s1",
            buffers=TransferBuffers(),
        )
        packed = ProtocolSerializer.pack(msg)
        unpacked = ProtocolSerializer.unpack(packed)
        assert unpacked.req_id == "empty"
        assert not unpacked.buffers.buffers

    def test_roundtrip_preserves_ptr_values(self):
        """Verify that pointer values survive serialization."""
        bufs = TransferBuffers()
        d = _make_tensor((5,))
        ks = _add(bufs, d, cache_name="main")

        msg = DecodeAllocated(
            decode_sid=0,
            dp_rank=0,
            req_id="ptr-test",
            rank_num=1,
            session_id="s1",
            buffers=bufs,
        )
        packed = ProtocolSerializer.pack(msg)
        unpacked = ProtocolSerializer.unpack(packed)

        restored = unpacked.buffers
        assert ks in restored.buffers
        entries = restored.buffers[ks]
        assert len(entries) == 1
        assert entries[0].ptr == d.data_ptr()
        assert entries[0].length == d.numel() * d.element_size()


class TestRankTransferDone:
    def test_roundtrip(self):
        msg = RankTransferDone(req_id="req-42", first_token=2048)
        packed = ProtocolSerializer.pack(msg)
        unpacked = ProtocolSerializer.unpack(packed)
        assert isinstance(unpacked, RankTransferDone)
        assert unpacked.req_id == "req-42"
        assert unpacked.first_token == 2048

    def test_zero_prefix_len(self):
        msg = RankTransferDone(req_id="empty", first_token=0)
        packed = ProtocolSerializer.pack(msg)
        unpacked = ProtocolSerializer.unpack(packed)
        assert unpacked.first_token == 0


class TestPrefillDone:
    def test_roundtrip(self):
        msg = PrefillDone(
            req_id="done-1",
            first_token=101,
            num_hit_tokens=512,
        )
        packed = ProtocolSerializer.pack(msg)
        unpacked = ProtocolSerializer.unpack(packed)
        assert isinstance(unpacked, PrefillDone)
        assert unpacked.req_id == "done-1"
        assert unpacked.first_token == 101
        assert unpacked.num_hit_tokens == 512

    def test_zero_values(self):
        msg = PrefillDone(req_id="cold", first_token=0, num_hit_tokens=0)
        packed = ProtocolSerializer.pack(msg)
        unpacked = ProtocolSerializer.unpack(packed)
        assert unpacked.first_token == 0
        assert unpacked.num_hit_tokens == 0


class TestProtocolDispatcher:
    """Verify that unpack dispatches to the correct type."""

    def test_dispatch_all_types(self):
        msgs = [
            DecodeAllocated(
                decode_sid=0,
                dp_rank=0,
                req_id="r1",
                rank_num=1,
                session_id="s1",
                buffers=TransferBuffers(),
            ),
            RankTransferDone(req_id="r2", first_token=100),
            PrefillDone(req_id="r3", first_token=5, num_hit_tokens=10),
        ]
        for msg in msgs:
            packed = ProtocolSerializer.pack(msg)
            unpacked = ProtocolSerializer.unpack(packed)
            assert type(unpacked) is type(msg), f"{type(msg).__name__} dispatch failed"
            assert unpacked.req_id == msg.req_id

    def test_unknown_type(self):
        import msgpack as _msgpack

        data = _msgpack.packb({"type": "UnknownType", "req_id": "x"}, use_bin_type=True)
        with pytest.raises(ValueError, match="unknown protocol message type"):
            ProtocolSerializer.unpack(data)

    def test_missing_type_field(self):
        import msgpack as _msgpack

        data = _msgpack.packb({"req_id": "x"}, use_bin_type=True)
        with pytest.raises(ValueError, match="missing.*type"):
            ProtocolSerializer.unpack(data)

    def test_empty_data(self):
        with pytest.raises(ValueError, match="empty"):
            ProtocolSerializer.unpack(b"")

    def test_corrupted_data(self):
        with pytest.raises((ValueError, msgpack.exceptions.UnpackException)):
            ProtocolSerializer.unpack(b"\xff\xfe\xfd")
