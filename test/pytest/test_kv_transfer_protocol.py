# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the dataclass-based protocol messages and msgpack serialization.
"""

import pytest
import msgpack
import numpy as np

from chitu.distributed.pd_disaggregation.kv_transfer.protocol import (
    DecodeAllocated,
    DecodePrepare,
    RankTransferDone,
    PrefillDone,
    ProtocolSerializer,
)
from chitu.distributed.pd_disaggregation.kv_transfer.transfer_buffers import (
    TransferBuffers,
)


class TestDecodeAllocated:
    def test_roundtrip_with_buffers(self):
        bufs = TransferBuffers()
        bufs.cache_block_ids["main"] = np.array([1, 2, 3], dtype=np.int32)
        bufs.cache_block_ids["linear"] = np.array([4, 5], dtype=np.int32)

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
        assert "main" in restored.cache_block_ids
        assert "linear" in restored.cache_block_ids
        assert list(restored.cache_block_ids["main"]) == [1, 2, 3]
        assert list(restored.cache_block_ids["linear"]) == [4, 5]

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
        assert not unpacked.buffers.cache_block_ids

    def test_roundtrip_preserves_block_ids(self):
        """Verify that block IDs survive serialization."""
        bufs = TransferBuffers()
        bufs.cache_block_ids["main"] = np.array([7, 8, 9, 10], dtype=np.int32)

        msg = DecodeAllocated(
            decode_sid=0,
            dp_rank=0,
            req_id="id-test",
            rank_num=1,
            session_id="s1",
            buffers=bufs,
        )
        packed = ProtocolSerializer.pack(msg)
        unpacked = ProtocolSerializer.unpack(packed)

        restored = unpacked.buffers
        assert "main" in restored.cache_block_ids
        ids = list(restored.cache_block_ids["main"])
        assert ids == [7, 8, 9, 10]
        assert restored.cache_block_ids["main"].dtype == np.int32


class TestDecodePrepare:
    def test_roundtrip_preserves_hit_block_counts(self):
        msg = DecodePrepare(
            req_id="prepare-1",
            prefill_sid=0,
            prefix_len=1024,
            new_cache_ids={"main": [1, 2, 3, 4], "linear": [8]},
            dp_rank=1,
            cache_manager_hit_block_counts={"main": 2, "linear": 0},
        )
        packed = ProtocolSerializer.pack(msg)
        unpacked = ProtocolSerializer.unpack(packed)

        assert isinstance(unpacked, DecodePrepare)
        assert unpacked.req_id == "prepare-1"
        assert unpacked.new_cache_ids["main"] == [1, 2, 3, 4]
        assert unpacked.cache_manager_hit_block_counts == {"main": 2, "linear": 0}


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
        data = msgpack.packb({"type": "UnknownType", "req_id": "x"}, use_bin_type=True)
        with pytest.raises(ValueError, match="unknown protocol message type"):
            ProtocolSerializer.unpack(data)

    def test_missing_type_field(self):
        data = msgpack.packb({"req_id": "x"}, use_bin_type=True)
        with pytest.raises(ValueError, match="missing.*type"):
            ProtocolSerializer.unpack(data)

    def test_empty_data(self):
        with pytest.raises(ValueError, match="empty"):
            ProtocolSerializer.unpack(b"")

    def test_corrupted_data(self):
        with pytest.raises((ValueError, msgpack.exceptions.UnpackException)):
            ProtocolSerializer.unpack(b"\xff\xfe\xfd")
