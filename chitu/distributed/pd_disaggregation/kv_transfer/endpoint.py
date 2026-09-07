# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0


import threading
import zmq

from chitu.boot.tcp_ip import get_local_ip
from chitu.distributed.coordinator import get_endpoint, set_endpoint
from chitu.distributed.parallel_state import get_world_group
from chitu.serve.crash import report_and_exit
import logging

logger = logging.getLogger(__name__)

TCP_ENDPOINT = "tcp://{}:{}"
ZMQ_RCVHWM = 200000
ZMQ_RCVBUF = 4 * 1024 * 1024  # 4MB
ZMQ_SNDHWM = 200000
ZMQ_SNDBUF = 4 * 1024 * 1024  # 4MB


class KVManagerEndpoint:
    def __init__(self, role: str, name: str):
        self.role = role
        self.name = name

        self.zmq_ctx = zmq.Context.instance()
        self.ip = get_local_ip()
        # ZMQ sockets are not thread-safe: the send_socket may be accessed
        # concurrently from ThreadPoolExecutor workers (prefill transfer_worker)
        # and from the asyncio event loop (stop_request -> remove_request_all_rank).
        # Serialize all sends through a single lock.
        self._send_lock = threading.Lock()

    def init_master(self, has_slave: bool):
        """init listener and receiver on master rank"""
        bind_addr = f"tcp://{self.ip}"
        if not has_slave:
            self.socket = self.zmq_ctx.socket(zmq.PULL)
            self.socket.setsockopt(zmq.RCVHWM, ZMQ_RCVHWM)
            self.socket.setsockopt(zmq.RCVBUF, ZMQ_RCVBUF)
            set_endpoint(
                self.role,
                self.name,
                self.ip,
                self.socket.bind_to_random_port(bind_addr),
            )
            return

        self._relay_socket = self.zmq_ctx.socket(zmq.PULL)
        self._relay_socket.setsockopt(zmq.RCVHWM, ZMQ_RCVHWM)
        self._relay_socket.setsockopt(zmq.RCVBUF, ZMQ_RCVBUF)
        set_endpoint(
            self.role,
            self.name,
            self.ip,
            self._relay_socket.bind_to_random_port(bind_addr),
        )

        self._push_sockets = []
        for rank in range(get_world_group().group_size - 1, -1, -1):
            _pub_socket = self.zmq_ctx.socket(zmq.PUSH)
            _pub_socket.setsockopt(zmq.SNDHWM, ZMQ_SNDHWM)
            _pub_socket.setsockopt(zmq.SNDBUF, ZMQ_SNDBUF)
            _pub_socket.setsockopt(zmq.SNDTIMEO, 60 * 1000)
            _pub_socket.setsockopt(zmq.LINGER, 0)
            pub_port = _pub_socket.bind_to_random_port(bind_addr)
            set_endpoint(
                self.role,
                f"{self.name}_pub_{rank}",
                self.ip,
                pub_port,
            )
            self._push_sockets.append(_pub_socket)

        threading.Thread(target=self.relay_thread, daemon=True).start()

        self.socket = self.zmq_ctx.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVHWM, ZMQ_RCVHWM)
        self.socket.setsockopt(zmq.RCVBUF, ZMQ_RCVBUF)
        endpoint = TCP_ENDPOINT.format(
            *get_endpoint(
                self.role,
                f"{self.name}_pub_{get_world_group().global_rank}",
            )
        )
        self.socket.connect(endpoint)

    def init_slave(self):
        """init receiver on slave rank"""
        self.socket = self.zmq_ctx.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVHWM, ZMQ_RCVHWM)
        self.socket.setsockopt(zmq.RCVBUF, ZMQ_RCVBUF)
        endpoint = TCP_ENDPOINT.format(
            *get_endpoint(
                self.role,
                f"{self.name}_pub_{get_world_group().global_rank}",
            )
        )
        self.socket.connect(endpoint)

    def init_remote(self):
        """init sender on any rank"""
        endpoint = TCP_ENDPOINT.format(*get_endpoint(self.role, self.name))
        self.send_socket = self.zmq_ctx.socket(zmq.PUSH)
        self.send_socket.setsockopt(zmq.SNDHWM, ZMQ_SNDHWM)
        self.send_socket.setsockopt(zmq.SNDBUF, ZMQ_SNDBUF)
        self.send_socket.setsockopt(zmq.SNDTIMEO, 60 * 1000)
        self.send_socket.setsockopt(zmq.LINGER, 0)
        self.send_socket.connect(endpoint)

    def launch_recv_thread(self, handler):
        threading.Thread(target=self.recv_thread, args=(handler,), daemon=True).start()

    def relay_thread(self):
        try:
            while True:
                msg = self._relay_socket.recv()
                for sock in self._push_sockets:
                    sock.send(msg)
        except Exception:
            logger.exception(
                f"{self.role}:{self.name} relay_thread fatal error, entering crash protocol"
            )
            report_and_exit(f"KV relay thread crashed ({self.role}:{self.name})")

    def recv_thread(self, handler):
        while True:
            try:
                raw = self.socket.recv()
            except Exception:
                logger.exception(
                    f"{self.role}:{self.name} recv_thread socket error, entering crash protocol"
                )
                report_and_exit(
                    f"KV recv_thread socket crashed ({self.role}:{self.name})"
                )
            try:
                handler(raw)
            except Exception:
                logger.exception(
                    f"{self.role}:{self.name} recv_thread handler error, entering crash protocol"
                )
                report_and_exit(
                    f"KV recv_thread handler crashed ({self.role}:{self.name})"
                )

    def send(self, data: bytes):
        with self._send_lock:
            return self.send_socket.send(data)


class DecodeEndpoints:
    def __init__(self, decode_sid: int):
        role = f"decode{decode_sid}"
        self.decode_prepare = KVManagerEndpoint(role, "decode_prepare")
        self.prefill_done = KVManagerEndpoint(role, "prefill_done")


class PrefillEndpoints:
    def __init__(self, prefill_sid: int):
        role = f"prefill{prefill_sid}"
        self.decode_allocated = KVManagerEndpoint(role, "decode_allocated")
        self.rank_transfer_done = KVManagerEndpoint(role, "rank_transfer_done")
