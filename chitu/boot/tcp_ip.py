# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import netifaces
import socket


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


#: Number of ports reserved to each job inside its port block (see
#: `job_port_base`). This is enough for the coordinator port, plus one torchrun
#: master port and one rendezvous port per instance (see `MAX_INSTS_PER_JOB`).
PORTS_PER_JOB = 20

#: Maximum number of instances of one job that fit into its port block.
#: `multi_inst.n_insts` must not exceed this, otherwise instances of one job
#: would step into the port block reserved to another job.
MAX_INSTS_PER_JOB = (PORTS_PER_JOB - 1) // 2

#: Per-job port blocks are allocated from `[PORT_BLOCK_BASE, PORT_BLOCK_LIMIT)`.
#: This range must stay outside the kernel's local port range
#: (`net.ipv4.ip_local_port_range`, 32768-60999 by default): a port inside that
#: range can be occupied by an outbound TCP connection of another process on the
#: same node before we bind it, and our `bind()` then fails with EADDRINUSE.
#: It must also stay clear of the port ranges used by other parts of chitu on
#: the same node: the transfer engine ports of the KV cache transfer
#: (`mooncake/transfer_engine.py`: 10000-10031, 12000-12999, and 12001 by
#: default) and the random port range that CI cases use for `serve.port`
#: (20000-32767).
PORT_BLOCK_BASE = 13000
PORT_BLOCK_LIMIT = 20000
PORT_BLOCK_COUNT = (PORT_BLOCK_LIMIT - PORT_BLOCK_BASE) // PORTS_PER_JOB


def job_port_base(slurm_job_id: int) -> int:
    """
    Get the first port of the port block reserved to the given Slurm job.

    Ports that must be agreed on by all nodes and all instances of a job (e.g.
    the coordinator port, and the torchrun master and rendezvous ports of
    instances spanning multiple nodes) are derived from the Slurm job ID, so
    that all the nodes of the job compute the same ports. Concurrent jobs get
    consecutive job IDs, hence disjoint port blocks, so that they don't collide
    on the same node.
    """

    return PORT_BLOCK_BASE + (slurm_job_id % PORT_BLOCK_COUNT) * PORTS_PER_JOB


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
