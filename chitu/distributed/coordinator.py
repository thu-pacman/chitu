# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# This file manages a TCPStore that records TCP port allocation on each host.
#
# Store format: key="<host_role>:<connection_name>", value=<non-wildcard-ip>:<port>. E.g.,
# key="router:token_port", value=1.2.3.4:12345.
#
# Only the host knows its non-wildcard ip (not something like 0.0.0.0), and
# clients only know its role (currently all use cases are "router").
#
# Steps for establishing a new TCP server:
# 1. Bind TCP server to a random port (port=0).
# 2. Set the (ip, port) in the TCPStore, where ip is the host's non-wildcard ip
#    obtained via `get_local_ip`.
#
# Steps for connect to a TCP server:
# 1. Get the (ip, port) from the TCPStore.
# 2. Connect to the TCP server with that ip and port.

import torch

_coordinator = None


def init_coordinator(host: str, port: int, is_coordinator_host: bool):
    global _coordinator
    if host in {"0.0.0.0", "::", ""}:
        raise ValueError(
            f"The coordinator host must be recognized from all nodes, and therefore "
            f"it must NOT be wildcard address {host}. Please set concrete IP addresses."
        )
    _coordinator = torch.distributed.TCPStore(
        host, port, is_master=is_coordinator_host, wait_for_workers=False
    )


def set_endpoint(host_role: str, connection_name: str, ip: str, port: int):
    """Register a server endpoint under the given host role.

    `ip` should be the host's non-wildcard ip (e.g. from `get_local_ip`),
    not a wildcard address like "0.0.0.0".
    """
    _coordinator.set(f"{host_role}:{connection_name}", f"{ip}:{port}")


def get_endpoint(host_role: str, connection_name: str) -> tuple[str, int]:
    """Get a server endpoint registered under the given host role.

    Returns a (ip, port) tuple.
    """
    value = _coordinator.get(f"{host_role}:{connection_name}")
    assert isinstance(value, bytes)
    ip, port = value.decode().rsplit(":", 1)
    return ip, int(port)
