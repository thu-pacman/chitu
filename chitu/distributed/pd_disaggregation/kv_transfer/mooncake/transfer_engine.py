# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import logging
import asyncio
import threading
import os
from aiohttp import web
from mooncake.engine import TransferEngine
from chitu.device_type import is_ascend

logger = logging.getLogger(__name__)


class MooncakeTransferEngine:
    """Mooncake transfer engine for high-speed KV cache transfer"""

    def __init__(self, hostname: str, ib_device: Optional[str] = None):
        self.hostname = hostname
        self.ib_device = ib_device
        # Check MOONCAKE_MOCK_MODE env var
        self.mock_mode = os.environ.get("MOONCAKE_MOCK_MODE", "0") == "1"
        self.protocol = os.environ.get("MOONCAKE_PROTOCOL", "rdma")

        self.engine = TransferEngine()

        if is_ascend():
            # Build Ascend-compliant local_server_name: ip:port:npu_<phy_id>
            phy_id_str = os.environ.get("ASCEND_PHY_ID")
            if phy_id_str is None:
                # Fallback: first device from ASCEND_RT_VISIBLE_DEVICES
                visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
                phy_id_str = visible.split(",")[0].strip() if visible else "-1"
            try:
                phy_id = int(phy_id_str)
            except Exception:
                phy_id = -1

            # Choose a deterministic unique port per card
            if phy_id >= 0:
                local_port = 10000 + phy_id
            else:
                local_port = 12000 + (os.getpid() % 1000)

            local_server_name = f"{self.hostname}:{local_port}:npu_{phy_id}"
            self.initialize(
                hostname=local_server_name,
                device_name=self.ib_device,
                protocol="hccl",
                metadata_server="P2PHANDSHAKE",
            )
        else:
            # Non-Ascend: keep original behavior (RDMA with default hostname)
            # If protocol is forced to something else (e.g. tcp) via env var, use it.

            self.initialize(
                hostname=self.hostname,
                device_name=self.ib_device,
                protocol=self.protocol,
                metadata_server="P2PHANDSHAKE",
            )

        self.session_id = f"{self.hostname}:{self.engine.get_rpc_port()}"
        logger.info("mooncake transfer engine initialized successfully")

    def register(self, ptr, length):
        """Register memory for RDMA transfer"""
        if self.mock_mode:
            logger.debug(f"mock: register memory ptr={ptr}, length={length}")
            return  # Success in mock mode

        logger.info(
            f"Mooncake attempting to register memory: length={length/1024/1024:.2f} MB"
        )
        ret_value = self.engine.register_memory(ptr, length)
        if ret_value != 0:
            logger.error("mooncake memory registration failed")
            raise RuntimeError("mooncake memory registration failed")

    def deregister(self, ptr):
        """Deregister memory from RDMA transfer"""
        if self.mock_mode:
            logger.debug(f"mock: deregister memory ptr={ptr}")
            return  # Success in mock mode

        ret_value = self.engine.unregister_memory(ptr)
        if ret_value != 0:
            logger.error("mooncake memory deregistration failed")
            raise RuntimeError("mooncake memory deregistration failed")

    def initialize(
        self,
        hostname: str,
        device_name: Optional[str],
        protocol: str = "rdma",
        metadata_server: str = "P2PHANDSHAKE",
    ) -> None:
        """Initialize the mooncake instance"""
        if self.mock_mode:
            logger.debug(
                f"mock: initialize hostname={hostname}, device={device_name}, protocol={protocol}"
            )
            return  # Success in mock mode

        ret_value = self.engine.initialize(
            hostname,
            metadata_server,
            protocol,
            device_name if device_name is not None else "",
        )
        if ret_value != 0:
            logger.error("mooncake transfer engine initialization failed")
            raise RuntimeError("mooncake transfer engine initialization failed")

    def transfer_sync(
        self, session_id: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        """Synchronously transfer data to the specified address"""
        if self.mock_mode:
            logger.debug(
                f"mock: transfer {buffer} -> {session_id}:{peer_buffer_address}, length={length}"
            )
            return 0  # Success in mock mode

        ret = self.engine.transfer_sync_write(
            session_id, buffer, peer_buffer_address, length
        )

        # Currently we assume some failures should be accepted
        if ret < 0:
            logger.debug(
                f"failed to transfer data from {buffer} to {session_id} - {peer_buffer_address}"
            )
        return ret

    def get_session_id(self):
        """Get session ID for this transfer engine"""
        return self.session_id


class MooncakeBootstrapServer:
    """
    Minimal Bootstrap server for Mooncake handshake.
    Exposes HTTP endpoints:
      - PUT /route: register prefill rank {role, rank_ip, rank_port, engine_rank}
      - GET /route?engine_rank=-1: return dp_size (fallback to 1)
      - GET /route?engine_rank=<rank>: return {rank_ip, rank_port}
      - GET /health: health check
    """

    def __init__(self, port: int):
        self.port = port
        self.dp_size = 1  # fallback for 1P1D; can be extended to config-driven
        self.prefill_port_table: dict[int, dict[str, str | int]] = {}
        self._loop = None
        self._runner = None
        self._lock = threading.Lock()

    def _setup_routes(self, app):

        async def handle_health(request):
            return web.Response(text="OK", status=200)

        async def handle_route(request):
            method = request.method
            if method == "PUT":
                data = await request.json()
                role = data.get("role")
                rank_ip = data.get("rank_ip")
                rank_port = int(data.get("rank_port", 0))
                engine_rank = int(data.get("engine_rank", -1))
                if role == "Prefill" and engine_rank >= 0 and rank_ip and rank_port > 0:
                    with self._lock:
                        self.prefill_port_table[engine_rank] = {
                            "rank_ip": rank_ip,
                            "rank_port": rank_port,
                        }
                    return web.Response(text="OK", status=200)
                return web.Response(text="Bad Request", status=400)
            elif method == "GET":
                engine_rank = request.query.get("engine_rank")
                if engine_rank is None:
                    return web.Response(
                        text="Missing inputs for bootstrap server.", status=400
                    )
                er = int(engine_rank)
                if er == -1:
                    return web.json_response({"dp_size": self.dp_size}, status=200)
                with self._lock:
                    info = self.prefill_port_table.get(er)
                if info is not None:
                    return web.json_response(info, status=200)
                return web.Response(text="Bootstrap info not Found", status=404)
            else:
                return web.Response(text="Method not allowed", status=405)

        app.router.add_get("/health", handle_health)
        app.router.add_route("*", "/route", handle_route)

    def _run_server(self):

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        app = web.Application()
        self._setup_routes(app)
        self._runner = web.AppRunner(app)
        self._loop.run_until_complete(self._runner.setup())
        site = web.TCPSite(self._runner, port=self.port)
        self._loop.run_until_complete(site.start())
        logger.info(f"Mooncake Bootstrap HTTP server started on port {self.port}")
        self._loop.run_forever()

    def start_in_background(self):
        t = threading.Thread(target=self._run_server, daemon=True)
        t.start()
        return t
