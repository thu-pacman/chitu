# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import netifaces
import socket
import zmq


class FreeTCPPortHolder:
    """
    This class holds a TCP port via keeping a socket open.

    The TCP port can later be used to create another socket. This can
    be done by calling `.pop()` on this class, which closes the holder
    socket and returns the TCP port.
    """

    def __init__(self):
        try:
            # try ipv4
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.bind(("", 0))
        except OSError:
            # try ipv6
            self.sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            self.sock.bind(("", 0))
        self.alive = True

    def __del__(self):
        self.sock.close()

    @property
    def port(self):
        return self.sock.getsockname()[1]

    def pop(self):
        if not self.alive:
            raise RuntimeError("TCPPortHolder is already closed")
        port = self.port
        self.sock.close()
        self.alive = False
        return port


def get_free_port():
    """
    Get free TCP port, but may be used by other process later
    """

    return FreeTCPPortHolder().pop()


def is_port_available(port: int):
    """
    Test whether the port is available, but may be used by other process later
    """

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", port))
            return True
    except Exception:
        return False


def get_port_from_zmq_socket(zmq_socket):
    return int(zmq_socket.getsockopt(zmq.LAST_ENDPOINT).decode().split(":")[-1])


def is_localhost(host: str):
    return host in {"localhost", "127.0.0.1", "::1"}


@functools.cache
def get_local_ip() -> str:
    try:
        try:
            ifaces = netifaces.interfaces()
            gateways = netifaces.gateways()
            default_gateway = gateways.get("default", {}).get(netifaces.AF_INET, None)
            if len(ifaces) > 0 and default_gateway:
                _, main_nic_name = default_gateway
                for iface in ifaces:
                    if iface == main_nic_name:
                        iface_addrs = netifaces.ifaddresses(iface).get(
                            netifaces.AF_INET, []
                        )
                        if iface_addrs:
                            return iface_addrs[0]["addr"]
            raise RuntimeError("Not network interface found")
        except Exception as e_netifaces:
            raise RuntimeError("Unable to get local IP via netifaces") from e_netifaces

    except Exception as e_method1:
        try:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
                    return s.getsockname()[0]
            except Exception as e_socket_connect:
                raise RuntimeError(
                    "Unable to get local IP via socket.connect"
                ) from e_socket_connect

        except Exception as e_method2:
            try:
                hostname = socket.gethostname()
                ip = socket.gethostbyname(hostname)
                if ip and ip != "127.0.0.1" and ip != "0.0.0.0":
                    return ip
                raise RuntimeError(
                    f"No IP found for hostname {hostname}"
                ) from e_method2
            except Exception as e_socket_gethostname:
                raise RuntimeError(
                    "Unable to get local IP via socket.gethostname"
                ) from e_socket_gethostname
