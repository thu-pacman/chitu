# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
PD disaggregation coordination service
Responsible for metadata synchronization and coordination between Prefill (P) and Decode (D)
"""

import logging

from chitu.distributed.pd_disaggregation.pd_types import (
    KVTransferMetadata,
    PDPairInfo,
    PDRequestStatus,
    SchedulerInfo,
    SchedulerType,
)

logger = logging.getLogger(__name__)


class PDCoordinationService:
    """PD coordination service"""

    def __init__(self):
        # State management
        self.pd_pairs: dict[str, PDPairInfo] = {}  # request_id -> PDPairInfo
        self.kv_transfer_metadata: dict[str, KVTransferMetadata] = (
            {}
        )  # request_id -> metadata
        self.prefill_schedulers: dict[int, SchedulerInfo] = (
            {}
        )  # local_instance_id -> info
        self.decode_schedulers: dict[int, SchedulerInfo] = (
            {}
        )  # local_instance_id -> info

        # Running state
        self.running = False

    async def start(self):
        """Start coordination service"""
        logger.info("starting pd coordination service...")
        self.running = True
        logger.info("pd coordination service started")

    async def stop(self):
        """Stop coordination service"""
        logger.info("stopping pd coordination service...")
        self.running = False

    async def register_pd_pair(
        self, request_id: str, prefill_scheduler_id: int, decode_scheduler_id: int
    ):
        """Register P-D pair"""
        pair_info = PDPairInfo(
            request_id=request_id,
            prefill_scheduler_id=prefill_scheduler_id,
            decode_scheduler_id=decode_scheduler_id,
            status=PDRequestStatus.DISPATCHED,
        )

        self.pd_pairs[request_id] = pair_info
        logger.debug(
            f"registered pd pair: {request_id} -> P{prefill_scheduler_id}-D{decode_scheduler_id}"
        )

    async def register_scheduler(
        self,
        local_instance_id: int,
        scheduler_type: SchedulerType,
        host: str,
        port: int,
    ):
        """Register Scheduler"""
        scheduler_info = SchedulerInfo(
            local_instance_id=local_instance_id,
            scheduler_type=scheduler_type,
            host=host,
            port=port,
        )

        if scheduler_type == SchedulerType.PREFILL:
            self.prefill_schedulers[local_instance_id] = scheduler_info
            logger.info(
                f"registered prefill instance: {local_instance_id} at {host}:{port}"
            )
        elif scheduler_type == SchedulerType.DECODE:
            self.decode_schedulers[local_instance_id] = scheduler_info
            logger.info(
                f"registered decode instance: {local_instance_id} at {host}:{port}"
            )

    def get_pd_stats(self) -> dict:
        """Get PD disaggregation statistics"""
        stats = {
            "total_pairs": len(self.pd_pairs),
            "status_counts": {},
            "prefill_schedulers": len(self.prefill_schedulers),
            "decode_schedulers": len(self.decode_schedulers),
            "kv_transfers": len(self.kv_transfer_metadata),
        }

        # Count request numbers for each status
        for pair in self.pd_pairs.values():
            status = pair.status.value
            stats["status_counts"][status] = stats["status_counts"].get(status, 0) + 1

        return stats
