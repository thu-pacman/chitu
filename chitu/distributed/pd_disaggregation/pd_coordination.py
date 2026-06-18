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

from chitu.boot.tcp_ip import get_local_ip
from chitu.distributed.coordinator import set_endpoint
from chitu.distributed.pd_disaggregation.pd_types import (
    KVTransferMetadata,
    KVTransferStatus,
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
        # Prefill control-plane endpoints per engine_rank (dp_id).
        # Used by Prefill PP/TP ranks to discover their control rank's ZMQ ports
        # 之前是 Prefill 启动的时候通过 torch broadcast 告诉其他 Prefill rank main rank 的端口，现在改用 coordination service 来同步
        self.prefill_ctrl_endpoints: dict[int, dict] = (
            {}
        )  # engine_rank -> endpoint dict

        # Decode prepare endpoints (control plane) per (decode_scheduler_id, dp_rank).
        # Used by Decode scheduler rank0 to send "prepare transfer" commands to the
        # owner dp_rank without coupling to PP/TP ports.
        self.decode_prepare_endpoints: dict[tuple[int, int], dict] = {}

        # Decode status endpoints (control plane) per (decode_scheduler_id, dp_rank).
        # Prefill sends final Success to the owner dp_rank's status endpoint.
        # Decode scheduler (rank0) need to discover dp_rank0 endpoint for
        # status aggregation. The same owner record publish an
        # internal broadcast port for decode-local ZMQ broadcast.
        self.decode_status_endpoints: dict[tuple[int, int], dict] = {}

        # ZMQ related
        self.context = zmq.asyncio.Context()
        self.metadata_socket = None  # metadata sync socket

        # Running state
        self.running = False
        self.metadata_task = None

    async def start(self):
        """Start coordination service"""
        logger.info("starting pd coordination service...")

        # Create sockets
        self.metadata_socket = self.context.socket(zmq.REP)
        metadata_sync_ip = get_local_ip()
        metadata_sync_port = self.metadata_socket.bind_to_random_port(
            f"tcp://{metadata_sync_ip}"
        )
        # Publish the metadata sync endpoint so workers can discover the
        # router's non-wildcard ip and port.
        set_endpoint(
            "router", "metadata_sync_port", metadata_sync_ip, metadata_sync_port
        )

        self.running = True

        # Start async tasks
        self.metadata_task = asyncio.create_task(self._metadata_sync_handler())

        logger.info(
            f"pd coordination service started, metadata endpoint: {metadata_sync_ip}:{metadata_sync_port}"
        )

    async def stop(self):
        """Stop coordination service"""
        logger.info("stopping pd coordination service...")

        self.running = False

        # Cancel tasks
        if self.metadata_task:
            self.metadata_task.cancel()

        # Close sockets
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

    async def _process_metadata_request(self, request_data: dict) -> dict:
        """Process metadata request"""
        request_type = request_data.get("type")

        if request_type == "set_prefill_ctrl_endpoint":
            engine_rank = request_data.get("engine_rank", None)
            ip = request_data.get("ip", "")
            port = request_data.get("port", None)
            internal_port = request_data.get("internal_port", None)
            broadcast_port = request_data.get("broadcast_port", None)
            er = int(engine_rank)
            p = int(port)
            ipt = str(ip)
            ipt = ipt.strip()
            internal_p = int(internal_port) if internal_port is not None else 0
            broadcast_p = int(broadcast_port) if broadcast_port is not None else 0
            if not ipt or p <= 0:
                return {
                    "status": "error",
                    "message": f"invalid endpoint: ip={ipt!r} port={p}",
                }
            self.prefill_ctrl_endpoints[er] = {
                "ip": ipt,
                "port": p,
                "internal_port": internal_p,
                "broadcast_port": broadcast_p,
            }
            return {"status": "success"}

        if request_type == "get_prefill_ctrl_endpoint":
            engine_rank = request_data.get("engine_rank", None)
            er = int(engine_rank)
            ep = self.prefill_ctrl_endpoints.get(er, None)
            if not ep:
                return {
                    "status": "not_found",
                    "message": f"prefill ctrl endpoint not found for engine_rank={er}",
                }
            return {"status": "success", "endpoint": dict(ep)}

        if request_type == "set_decode_prepare_endpoint":
            decode_scheduler_id = int(request_data.get("decode_scheduler_id", 0) or 0)
            dp_rank = int(request_data.get("dp_rank", 0) or 0)
            ip = str(request_data.get("ip", "") or "").strip()
            port = int(request_data.get("port", 0) or 0)
            if not ip or port <= 0:
                return {
                    "status": "error",
                    "message": f"invalid endpoint: ip={ip!r} port={port}",
                }
            self.decode_prepare_endpoints[(decode_scheduler_id, dp_rank)] = {
                "ip": ip,
                "port": port,
            }
            return {"status": "success"}

        if request_type == "get_decode_prepare_endpoint":
            decode_scheduler_id = int(request_data.get("decode_scheduler_id", 0) or 0)
            dp_rank = int(request_data.get("dp_rank", 0) or 0)
            ep = self.decode_prepare_endpoints.get((decode_scheduler_id, dp_rank), None)
            if not ep:
                return {
                    "status": "not_found",
                    "message": f"decode prepare endpoint not found for decode_scheduler_id={decode_scheduler_id} dp_rank={dp_rank}",
                }
            return {"status": "success", "endpoint": dict(ep)}

        if request_type == "set_decode_status_endpoint":
            decode_scheduler_id = int(request_data.get("decode_scheduler_id", 0) or 0)
            dp_rank = int(request_data.get("dp_rank", 0) or 0)
            ip = str(request_data.get("ip", "") or "").strip()
            port = int(request_data.get("port", 0) or 0)
            broadcast_port = request_data.get("broadcast_port", 0)
            if not ip or port <= 0:
                return {
                    "status": "error",
                    "message": f"invalid endpoint: ip={ip!r} port={port}",
                }
            self.decode_status_endpoints[(decode_scheduler_id, dp_rank)] = {
                "ip": ip,
                "port": port,
                "broadcast_port": broadcast_port,
            }
            return {"status": "success"}

        if request_type == "get_decode_status_endpoint":
            decode_scheduler_id = int(request_data.get("decode_scheduler_id", 0) or 0)
            dp_rank = int(request_data.get("dp_rank", 0) or 0)
            ep = self.decode_status_endpoints.get((decode_scheduler_id, dp_rank), None)
            if not ep:
                return {
                    "status": "not_found",
                    "message": f"decode status endpoint not found for decode_scheduler_id={decode_scheduler_id} dp_rank={dp_rank}",
                }
            return {"status": "success", "endpoint": dict(ep)}

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
            local_instance_id = request_data.get("local_instance_id")

            if (
                scheduler_type == "prefill"
                and local_instance_id in self.prefill_schedulers
            ):
                info = self.prefill_schedulers[local_instance_id]
                return {
                    "status": "success",
                    "info": {
                        "host": info.host,
                        "port": info.port,
                        "status": info.status,
                    },
                }
            elif (
                scheduler_type == "decode"
                and local_instance_id in self.decode_schedulers
            ):
                info = self.decode_schedulers[local_instance_id]
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
                    "message": f"scheduler {scheduler_type}:{local_instance_id} not found",
                }

        else:
            return {
                "status": "error",
                "message": f"unknown request type: {request_type}",
            }

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
