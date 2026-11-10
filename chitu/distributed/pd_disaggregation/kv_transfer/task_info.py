# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Per-request transfer state for PD disaggregation.
"""

from __future__ import annotations
import threading
import time
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
    input_cached_tokens: Optional[int] = None

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

    decode_sid: int | None = None
    """Decode scheduler slot that owns this request, or None before DecodeAllocated."""
    decode_generation: int = 0
    decode_dp_rank: int | None = None

    recv_buffers: dict[str, TransferBuffers] = field(default_factory=dict)
    """Per decode session_id (P side, from DecodeAllocated)."""
    decode_allocated_cnt: int = 0
    is_decode_allocated: bool = False

    done_count: int = 0
    """RankTransferDone counter."""

    rank_bytes: dict[str, int] = field(default_factory=dict)
    """Per-session sent bytes, accumulated from RankTransferDone (P side)."""

    # Progress clocks for the P-side stall sweep. All three are monotonic.
    # `submitted_ts` and `started_ts` stay 0 until that step really happened, so
    # the sweep can tell "not submitted yet" (a long prefill is healthy) from
    # "submitted and silent".
    submitted_ts: float = 0.0
    """When send_kv_cache handed this request to a transfer worker."""
    started_ts: float = 0.0
    """When the transfer worker began the RDMA write (diagnostics only)."""
    last_progress_ts: float = field(default_factory=time.monotonic)
    """Last sign of progress: submit, worker start, or a RankTransferDone.

    Initialized to "now" at creation: if it were 0, every request would look
    stalled (now - 0 exceeds any threshold) before its transfer even starts.
    """
    transfer_grace_s: float = 150.0
    """How long without progress before the sweep retires this transfer.

    The value comes from the manager: `_info(create=True)` passes its own grace
    in, so this dataclass default only applies to a TaskInfo built by hand. The
    Prefill manager requires transfer timeout plus release grace to finish below
    `decode_wait_timeout_s` and refuses to start otherwise, so the Prefill side
    always gives up before Decode frees the receive blocks.
    """

    # ===============================
    #            Decode
    # ===============================

    prefill_sid: Optional[int] = None
    prefill_generation: int = 0

    cache_new_block_ids: dict[str, list[int]] = field(default_factory=dict)
    """cache name -> all allocated block ids, including MTP lookahead pages."""

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
    dp_rank: int | None = None
    """dp_rank that owns this request on the decode side, or None before DecodePrepare."""

    recv_bytes: int = 0
    """Expected total recv bytes, set during prepare_kv_transfer (D side)."""

    trace: Optional[Trace] = None
