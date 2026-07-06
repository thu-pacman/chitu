# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0


import os
import threading
import zmq

from chitu.boot.tcp_ip import get_local_ip
from chitu.distributed.coordinator import get_endpoint, set_endpoint
import logging

logger = logging.getLogger(__name__)

TCP_ENDPOINT = "tcp://{}:{}"


class KVManagerEndpoint:
    def __init__(self, role: str, name: str):
        self.role = role
        self.name = name

        self.zmq_ctx = zmq.Context.instance()
        self.ip = get_local_ip()

    def init_master(self, has_slave: bool):
        """init listener and receiver on master rank"""
        bind_addr = f"tcp://{self.ip}"
        if not has_slave:
            self.socket = self.zmq_ctx.socket(zmq.PULL)
            set_endpoint(
                self.role,
                self.name,
                self.ip,
                self.socket.bind_to_random_port(bind_addr),
            )
            return

        self._relay_socket = self.zmq_ctx.socket(zmq.PULL)
        set_endpoint(
            self.role,
            self.name,
            self.ip,
            self._relay_socket.bind_to_random_port(bind_addr),
        )

        self._pub_socket = self.zmq_ctx.socket(zmq.PUB)
        pub_port = self._pub_socket.bind_to_random_port(bind_addr)
        set_endpoint(self.role, self.name + "_pub", self.ip, pub_port)
        self._pub_socket.setsockopt(zmq.LINGER, 0)
        threading.Thread(target=self.relay_thread, daemon=True).start()

        self.socket = self.zmq_ctx.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.connect(TCP_ENDPOINT.format(self.ip, pub_port))

    def init_slave(self):
        """init receiver on slave rank"""
        endpoint = TCP_ENDPOINT.format(*get_endpoint(self.role, self.name + "_pub"))
        self.socket = self.zmq_ctx.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.connect(endpoint)

    def init_remote(self):
        """init sender on any rank"""
        endpoint = TCP_ENDPOINT.format(*get_endpoint(self.role, self.name))
        self.send_socket = self.zmq_ctx.socket(zmq.PUSH)
        self.send_socket.setsockopt(zmq.SNDTIMEO, 60 * 1000)
        self.send_socket.setsockopt(zmq.LINGER, 0)
        self.send_socket.connect(endpoint)

    def launch_recv_thread(self, handler):
        threading.Thread(target=self.recv_thread, args=(handler,), daemon=True).start()

    def relay_thread(self):
        try:
            zmq.proxy(self._relay_socket, self._pub_socket)
        except Exception:
            logger.exception(
                f"{self.role}:{self.name} relay_thread fatal error, exiting process"
            )
            os._exit(1)

    def recv_thread(self, handler):
        while True:
            try:
                raw = self.socket.recv()
            except Exception:
                logger.exception(
                    f"{self.role}:{self.name} recv_thread socket error, exiting process"
                )
                os._exit(1)
            try:
                handler(raw)
            except Exception:
                logger.exception(
                    f"{self.role}:{self.name} recv_thread handler error, exiting process"
                )
                os._exit(1)

    def send(self, data: bytes):
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
