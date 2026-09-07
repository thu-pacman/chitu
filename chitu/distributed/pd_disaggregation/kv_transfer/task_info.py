# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Per-request transfer state for PD disaggregation.
"""

from __future__ import annotations
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, TYPE_CHECKING

from chitu.trace import Trace

if TYPE_CHECKING:
    from chitu.distributed.pd_disaggregation.kv_transfer.transfer_buffers import (
        TransferBuffers,
    )


@dataclass
class TransferStatus:
    """Record the transfer status and other info associated with first token in the Decode Main Rank"""

    done: bool = False
    first_token: int = 0
    num_hit_tokens: int = 0
    trace: Optional[Trace] = None

    def __bool__(self):
        return self.done


@dataclass
class TaskInfo:
    """Per-request state for a single PD disaggregation transfer."""

    # ===============================
    #     Prefill & Decode shared
    # ===============================

    req_id: str = ""
    first_token: int = 0
    num_hit_tokens: int = 0

    # ===============================
    #            Prefill
    # ===============================

    decode_sid: int = -1
    decode_dp_rank: int = -1

    recv_buffers: dict[str, TransferBuffers] = field(default_factory=dict)
    """Per decode session_id (P side, from DecodeAllocated)."""
    decode_allocated_cnt: int = 0
    is_decode_allocated: bool = False

    done_count: int = 0
    """RankTransferDone counter."""

    rank_bytes: dict[str, int] = field(default_factory=dict)
    """Per-session sent bytes, accumulated from RankTransferDone (P side)."""

    # ===============================
    #            Decode
    # ===============================

    prefill_sid: Optional[int] = None

    cache_new_block_ids: dict[str, list[int]] = field(default_factory=dict)
    """cache name -> full prefix block ids to install in the decode block table."""

    cache_transfer_block_ids: dict[str, list[int]] = field(default_factory=dict)
    """cache name -> block ids that are written by this RDMA transfer."""

    cache_manager_new_block_ids: dict[str, list[int]] = field(default_factory=dict)
    """cache manager name -> list of new block ids."""

    cache_manager_hit_block_counts: dict[str, int] = field(default_factory=dict)
    """cache manager name -> number of decode-side prefix blocks already ready."""

    is_prefill_done: bool = False
    prefill_done_event: threading.Event = field(default_factory=threading.Event)
    """Set when PrefillDone is received; used for thread-safe synchronization
    between the ZMQ recv thread and the compute thread."""
    prefill_failed: bool = False
    """Set when the Prefill side reports a request-level failure: the
    compute thread's recv_kv_cache_and_insert must wake and fail this request
    as request-level instead of timing out and crashing."""
    is_decode_prepare_received: bool = False
    is_decode_allocated_sent: bool = False

    prefix_len: int = 0
    dp_rank: int = -1
    """dp_rank that owns this request on the decode side."""

    recv_bytes: int = 0
    """Expected total recv bytes, set during prepare_kv_transfer (D side)."""

    trace: Optional[Trace] = None
