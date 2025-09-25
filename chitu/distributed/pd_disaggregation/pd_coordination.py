# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
PD disaggregation coordination service
Responsible for metadata synchronization and coordination between Prefill (P) and Decode (D)
"""

import asyncio
import logging

import zmq
import zmq.asyncio
import msgpack

from chitu.distributed.pd_disaggregation.pd_types import (
    KVTransferMetadata,
    KVTransferStatus,
    PDCoordinationMessage,
    PDPairInfo,
    PDRequestStatus,
    SchedulerInfo,
    SchedulerType,
)

logger = logging.getLogger(__name__)


class PDCoordinationService:
    """PD coordination service"""

    def __init__(self, coordination_port: int, metadata_sync_port: int):
        self.coordination_port = coordination_port
        self.metadata_sync_port = metadata_sync_port

        # State management
        self.pd_pairs: dict[str, PDPairInfo] = {}  # request_id -> PDPairInfo
        self.kv_transfer_metadata: dict[str, KVTransferMetadata] = (
            {}
        )  # request_id -> metadata
        self.prefill_schedulers: dict[int, SchedulerInfo] = {}  # scheduler_id -> info
        self.decode_schedulers: dict[int, SchedulerInfo] = {}  # scheduler_id -> info

        # ZMQ related
        self.context = zmq.asyncio.Context()
        self.coordination_socket = None  # coordination message socket
        self.metadata_socket = None  # metadata sync socket

        # Running state
        self.running = False
        self.coordination_task = None
        self.metadata_task = None

    async def start(self):
        """Start coordination service"""
        logger.info("starting pd coordination service...")

        try:
            # Create sockets
            self.coordination_socket = self.context.socket(zmq.PULL)
            self.coordination_socket.bind(f"tcp://*:{self.coordination_port}")

            self.metadata_socket = self.context.socket(zmq.REP)
            self.metadata_socket.bind(f"tcp://*:{self.metadata_sync_port}")

            self.running = True

            # Start async tasks
            self.coordination_task = asyncio.create_task(self._coordination_handler())
            self.metadata_task = asyncio.create_task(self._metadata_sync_handler())

            logger.info(
                f"pd coordination service started, coordination port: {self.coordination_port}, metadata port: {self.metadata_sync_port}"
            )

        except Exception as e:
            logger.error(f"failed to start pd coordination service: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop coordination service"""
        logger.info("stopping pd coordination service...")

        self.running = False

        # Cancel tasks
        if self.coordination_task:
            self.coordination_task.cancel()
        if self.metadata_task:
            self.metadata_task.cancel()

        # Close sockets
        if self.coordination_socket:
            self.coordination_socket.close()
        if self.metadata_socket:
            self.metadata_socket.close()

        # Close context
        self.context.term()

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
        self, scheduler_id: int, scheduler_type: SchedulerType, host: str, port: int
    ):
        """Register Scheduler"""
        scheduler_info = SchedulerInfo(
            scheduler_id=scheduler_id,
            scheduler_type=scheduler_type,
            host=host,
            port=port,
        )

        if scheduler_type == SchedulerType.PREFILL:
            self.prefill_schedulers[scheduler_id] = scheduler_info
            logger.info(
                f"registered prefill scheduler: {scheduler_id} at {host}:{port}"
            )
        elif scheduler_type == SchedulerType.DECODE:
            self.decode_schedulers[scheduler_id] = scheduler_info
            logger.info(f"registered decode scheduler: {scheduler_id} at {host}:{port}")

    async def handle_prefill_complete(
        self, request_id: str, prefill_scheduler_id: int, kv_metadata: dict
    ):
        """Handle Prefill-complete notification"""
        if request_id not in self.pd_pairs:
            logger.warning(f"pd pair info not found for request: {request_id}")
            return

        pair_info = self.pd_pairs[request_id]
        pair_info.status = PDRequestStatus.PREFILL_COMPLETE

        # Create KV transfer metadata
        transfer_metadata = KVTransferMetadata(
            request_id=request_id,
            prefill_scheduler_id=prefill_scheduler_id,
            decode_scheduler_id=pair_info.decode_scheduler_id,
            kv_cache_shape=kv_metadata.get("kv_cache_shape", ()),
            first_token_logits_shape=kv_metadata.get("first_token_logits_shape", ()),
            transfer_session_id=kv_metadata.get("transfer_session_id", ""),
            prefill_endpoint=kv_metadata.get("prefill_endpoint", ""),
            decode_endpoint=kv_metadata.get("decode_endpoint", ""),
        )

        self.kv_transfer_metadata[request_id] = transfer_metadata

        logger.info(
            f"prefill complete: {request_id}, preparing to notify decode scheduler {pair_info.decode_scheduler_id}"
        )

        # Notify corresponding Decode Scheduler
        await self._notify_decode_scheduler(
            pair_info.decode_scheduler_id, transfer_metadata
        )

    async def handle_kv_transfer_ready(
        self, request_id: str, prefill_info: dict, decode_info: dict
    ):
        """Handle KV transfer ready notification"""
        if request_id not in self.pd_pairs:
            logger.warning(f"pd pair info not found for request: {request_id}")
            return

        # Update transfer metadata
        if request_id in self.kv_transfer_metadata:
            metadata = self.kv_transfer_metadata[request_id]
            metadata.prefill_endpoint = prefill_info.get("endpoint", "")
            metadata.decode_endpoint = decode_info.get("endpoint", "")
            metadata.transfer_session_id = prefill_info.get("session_id", "")
            metadata.status = KVTransferStatus.TRANSFERRING

            logger.info(f"kv transfer ready for request: {request_id}")
        else:
            logger.warning(f"kv transfer metadata not found for request: {request_id}")

    async def handle_kv_transfer_complete(self, request_id: str):
        """Handle KV transfer completion notification"""
        if request_id in self.kv_transfer_metadata:
            self.kv_transfer_metadata[request_id].status = KVTransferStatus.COMPLETED

        if request_id in self.pd_pairs:
            self.pd_pairs[request_id].status = PDRequestStatus.DECODE_RUNNING

        logger.info(f"kv transfer complete: {request_id}")

    async def handle_decode_complete(self, request_id: str):
        """Handle Decode completion notification"""
        if request_id in self.pd_pairs:
            self.pd_pairs[request_id].status = PDRequestStatus.COMPLETED

        logger.info(f"decode complete: {request_id}")

    async def _coordination_handler(self):
        """Coordination message handler"""
        logger.info("starting coordination message handler")

        while self.running:
            try:
                # Receive coordination message
                message_bytes = await self.coordination_socket.recv()
                message_data = msgpack.unpackb(message_bytes, raw=False)

                message = PDCoordinationMessage(**message_data)
                await self._process_coordination_message(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"error processing coordination message: {e}")

    async def _metadata_sync_handler(self):
        """Metadata synchronization handler"""
        logger.info("starting metadata sync handler")

        while self.running:
            try:
                # Receive metadata request
                request_bytes = await self.metadata_socket.recv()
                request_data = msgpack.unpackb(request_bytes, raw=False)

                response = await self._process_metadata_request(request_data)

                # Send response
                response_bytes = msgpack.packb(response)
                await self.metadata_socket.send(response_bytes)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"error processing metadata request: {e}")

    async def _process_coordination_message(self, message: PDCoordinationMessage):
        """Process coordination message"""
        try:
            if message.message_type == "prefill_complete":
                await self.handle_prefill_complete(
                    message.request_id, message.sender_id, message.payload
                )
            elif message.message_type == "kv_transfer_complete":
                await self.handle_kv_transfer_complete(message.request_id)
            elif message.message_type == "decode_complete":
                await self.handle_decode_complete(message.request_id)
            else:
                logger.warning(
                    f"unknown coordination message type: {message.message_type}"
                )

        except Exception as e:
            logger.error(f"failed to process coordination message: {e}")

    async def _process_metadata_request(self, request_data: dict) -> dict:
        """Process metadata request"""
        request_type = request_data.get("type")

        if request_type == "get_kv_transfer_metadata":
            request_id = request_data.get("request_id")
            if request_id in self.kv_transfer_metadata:
                metadata = self.kv_transfer_metadata[request_id]
                return {
                    "status": "success",
                    "metadata": {
                        "request_id": metadata.request_id,
                        "prefill_scheduler_id": metadata.prefill_scheduler_id,
                        "decode_scheduler_id": metadata.decode_scheduler_id,
                        "kv_cache_shape": metadata.kv_cache_shape,
                        "first_token_logits_shape": metadata.first_token_logits_shape,
                        "transfer_session_id": metadata.transfer_session_id,
                        "prefill_endpoint": metadata.prefill_endpoint,
                        "decode_endpoint": metadata.decode_endpoint,
                    },
                }
            else:
                return {
                    "status": "not_found",
                    "message": f"metadata not found for request {request_id}",
                }

        elif request_type == "get_scheduler_info":
            scheduler_type = request_data.get("scheduler_type")
            scheduler_id = request_data.get("scheduler_id")

            if scheduler_type == "prefill" and scheduler_id in self.prefill_schedulers:
                info = self.prefill_schedulers[scheduler_id]
                return {
                    "status": "success",
                    "info": {
                        "host": info.host,
                        "port": info.port,
                        "status": info.status,
                    },
                }
            elif scheduler_type == "decode" and scheduler_id in self.decode_schedulers:
                info = self.decode_schedulers[scheduler_id]
                return {
                    "status": "success",
                    "info": {
                        "host": info.host,
                        "port": info.port,
                        "status": info.status,
                    },
                }
            else:
                return {
                    "status": "not_found",
                    "message": f"scheduler {scheduler_type}:{scheduler_id} not found",
                }

        else:
            return {
                "status": "error",
                "message": f"unknown request type: {request_type}",
            }

    async def _notify_decode_scheduler(
        self, decode_scheduler_id: int, metadata: KVTransferMetadata
    ):
        """Notify Decode Scheduler to prepare for receiving KV cache"""
        if decode_scheduler_id not in self.decode_schedulers:
            logger.error(f"decode scheduler {decode_scheduler_id} not found")
            return

        # Here we should notify the Decode Scheduler via ZMQ
        # The concrete implementation will be added in later stages
        logger.info(
            f"notifying decode scheduler {decode_scheduler_id} to prepare for kv cache: {metadata.request_id}"
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
