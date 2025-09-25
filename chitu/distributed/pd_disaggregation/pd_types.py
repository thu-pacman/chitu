# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Data types and enum definitions for PD disaggregation
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import torch


class PDRequestStatus(Enum):
    """PD request status"""

    PENDING = "pending"  # waiting
    DISPATCHED = "dispatched"  # dispatched to P and D
    PREFILL_RUNNING = "prefill_running"  # prefill running
    PREFILL_COMPLETE = "prefill_complete"  # prefill completed
    KV_TRANSFERRING = "kv_transferring"  # KV cache transferring
    DECODE_RUNNING = "decode_running"  # decode running
    COMPLETED = "completed"  # completed
    FAILED = "failed"  # failed


class SchedulerType(Enum):
    """Scheduler type"""

    PREFILL = "prefill"
    DECODE = "decode"
    UNIFIED = "unified"  # traditional Prefill+Decode


class KVTransferStatus(Enum):
    """KV transfer status"""

    WAITING = "waiting"
    TRANSFERRING = "transferring"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PDPairInfo:
    """P-D pairing information"""

    request_id: str
    prefill_scheduler_id: int
    decode_scheduler_id: int
    created_time: float = field(default_factory=time.time)
    status: PDRequestStatus = PDRequestStatus.PENDING


@dataclass
class PendingPDRequest:
    """Pending PD request"""

    request_id: str
    original_request: Any  # ChatRequest type, avoid circular import
    prefill_scheduler_id: int
    decode_scheduler_id: int
    status: PDRequestStatus = PDRequestStatus.PENDING
    created_time: float = field(default_factory=time.time)
    prefill_start_time: Optional[float] = None
    prefill_complete_time: Optional[float] = None
    decode_start_time: Optional[float] = None
    decode_complete_time: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class KVTransferMetadata:
    """KV transfer metadata"""

    request_id: str
    prefill_scheduler_id: int
    decode_scheduler_id: int
    kv_cache_shape: tuple
    first_token_logits_shape: tuple
    transfer_session_id: str
    prefill_endpoint: str
    decode_endpoint: str
    status: KVTransferStatus = KVTransferStatus.WAITING
    created_time: float = field(default_factory=time.time)


@dataclass
class PrefillCompleteMessage:
    """Prefill completion message"""

    request_id: str
    prefill_scheduler_id: int
    decode_scheduler_id: int
    first_token_logits: Optional[torch.Tensor] = None
    kv_transfer_metadata: Optional[dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class DecodeReadyMessage:
    """Decode ready message"""

    request_id: str
    prefill_scheduler_info: dict[str, Any]
    kv_transfer_metadata: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class PDCoordinationMessage:
    """PD coordination message"""

    message_type: str  # "prefill_complete", "decode_ready", "kv_transfer_complete"
    request_id: str
    sender_type: SchedulerType
    sender_id: int
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SchedulerInfo:
    """Scheduler information"""

    scheduler_id: int
    scheduler_type: SchedulerType
    host: str
    port: int
    status: str = "online"  # online, offline, busy
    load_info: dict[str, Any] = field(default_factory=dict)
    last_heartbeat: float = field(default_factory=time.time)


@dataclass
class PDStats:
    """PD disaggregation statistics"""

    total_requests: int = 0
    pending_requests: int = 0
    prefill_running: int = 0
    decode_running: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    avg_prefill_time: float = 0.0
    avg_decode_time: float = 0.0
    avg_kv_transfer_time: float = 0.0
    last_update_time: float = field(default_factory=time.time)


@dataclass
class BatchInfo:
    """Batch information"""

    batch_id: str
    request_ids: list[str]
    prefill_scheduler_id: int
    decode_scheduler_ids: list[int]  # might be dispatched to multiple decode schedulers
    batch_size: int
    total_tokens: int
    created_time: float = field(default_factory=time.time)
    status: str = "pending"  # pending, processing, completed
