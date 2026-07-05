# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# This file manages a TCPStore that records TCP port allocation on each host.
#
# Store format:
#   Endpoint keys:  key="endpoint:<host_role>:<connection_name>", value=<ip>:<port>
#                   E.g., key="endpoint:router:token_port", value=1.2.3.4:12345.
#   Generic keys:   arbitrary keys stored via set_value/get_value.
#
# Endpoint keys use a fixed "endpoint:" prefix to prevent collision with
# generic keys (e.g. "inst0:cache_dists").
#
# Only the host knows its non-wildcard ip (not something like 0.0.0.0), and
# clients only know its role (e.g., "router", "instance_0").
#
# Steps for establishing a new TCP server:
# 1. Bind TCP server to a random port (port=0).
# 2. Set the (ip, port) in the TCPStore, where ip is the host's non-wildcard ip
#    obtained via `get_local_ip`.
#
# Steps for connect to a TCP server:
# 1. Get the (ip, port) from the TCPStore.
# 2. Connect to the TCP server with that ip and port.

from typing import Optional
from logging import getLogger
import os
import torch
import datetime

from chitu.boot.tcp_ip import is_localhost

logger = getLogger(__name__)

_coordinator = None
_coordinator_host = None
_coordinator_override_existing = False

#: Prefix applied to all endpoint keys to prevent collision with generic
#: coordinator keys (e.g. ``"inst0:cache_dists"``).
ENDPOINT_PREFIX = "endpoint:"


def init_coordinator(
    host: Optional[str],
    port: Optional[int],
    is_coordinator_host: bool,
    *,
    reuse_from_torchrun: bool = False,
    override_existing: bool = False,
):
    """
    Initialize the TCPStore that records TCP endpoints allocation on each host.

    Args:
        host (Optional[str]): The host where the TCPStore is running. Ignored when
            `reuse_from_torchrun` is true.
        port (Optional[int]): The port where the TCPStore is running. Ignored when
            `reuse_from_torchrun` is true.
        is_coordinator_host (bool): If true, this process hosts the TCPStore service.
        reuse_from_torchrun (bool): If true, ignore `host` and `port`. Reuse from
            torchrun's key-value store with a `torch.distributed.PrefixStore` instead.
        override_existing (bool): If true, `set_endpoint` will overwrite the possibly
            existing endpoint of the same name. You are NOT expected to override any
            endpoint in production, but this is useful for running multiple test cases
            in one process. If false, `set_endpoint` will raise a `RuntimeError` if
            the endpoint has already been registered, to catch role/connection name
            conflicts.
    """

    global _coordinator
    global _coordinator_host
    global _coordinator_override_existing

    if reuse_from_torchrun:
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "reuse_from_torchrun=True requires torch.distributed to be initialized"
            )
        default_store = torch.distributed.distributed_c10d._get_default_store()
        _coordinator = torch.distributed.PrefixStore("chitu_coordinator", default_store)
        _coordinator_host = os.environ["MASTER_ADDR"]
        _coordinator_override_existing = override_existing
        return

    if host in {"0.0.0.0", "::", ""}:
        raise ValueError(
            f"The coordinator host must be recognized from all nodes, and therefore "
            f"it must NOT be wildcard address {host}. Please set concrete IP addresses."
        )
    _coordinator = torch.distributed.TCPStore(
        host, port, is_master=is_coordinator_host, wait_for_workers=False
    )
    _coordinator_host = host
    _coordinator_override_existing = override_existing


def set_endpoint(host_role: str, connection_name: str, ip: str, port: int):
    """Register a server endpoint under the given host role.

    Args:
        host_role (str): The role of the host, e.g., "router", "instance_0".
        connection_name (str): The name of the connection, e.g., "token_port".
        ip (str): The host's non-wildcard ip, e.g., from `get_local_ip()`. not a wildcard
            address like "0.0.0.0".
        port (int): The TCP port ID.
    """

    if not is_localhost(_coordinator_host) and is_localhost(ip):
        logger.warning(
            f"The coordinator hosts at {_coordinator_host} indicating there might be "
            f"inter-node connection, but the endpoint {ip}:{port} is on localhost, which "
            f"may cause the inter-node connection to fail."
        )
    key = f"{ENDPOINT_PREFIX}{host_role}:{connection_name}"
    if not _coordinator_override_existing and _coordinator.check([key]):
        existing = _coordinator.get(key).decode()
        raise RuntimeError(
            f"Endpoint {key!r} is already registered (existing value: "
            f"{existing!r}, new value: {ip}:{port}). This indicates a "
            f"host_role/connection_name conflict."
        )
    _coordinator.set(key, f"{ip}:{port}")


def get_endpoint(
    host_role: str, connection_name: str, timeout: float = None
) -> tuple[str, int]:
    """Get a server endpoint registered under the given host role.

    If `timeout` (in seconds) is given, wait at most that long for the endpoint
    to be registered. Otherwise the store's default timeout is used.

    Returns a (ip, port) tuple.
    """

    if timeout is not None:
        # Temporarily override the store's wait timeout for this lookup.
        prev_timeout = _coordinator.timeout
        _coordinator.set_timeout(datetime.timedelta(seconds=timeout))
        try:
            value = _coordinator.get(f"{ENDPOINT_PREFIX}{host_role}:{connection_name}")
        finally:
            _coordinator.set_timeout(prev_timeout)
    else:
        value = _coordinator.get(f"{ENDPOINT_PREFIX}{host_role}:{connection_name}")
    assert isinstance(value, bytes)
    ip, port = value.decode().rsplit(":", 1)
    return ip, int(port)


def set_value(key: str, value: bytes, *, override: bool = False) -> None:
    """Store a raw byte payload under *key* in the coordinator store.

    If *override* is ``False`` (the default) and *key* already exists,
    raises ``RuntimeError`` to catch accidental key collisions.  Set
    *override* to ``True`` to deliberately overwrite an existing value.
    """
    if not override and _coordinator.check([key]):
        raise RuntimeError(
            f"Coordinator key {key!r} already exists. If you intend to "
            f"overwrite, pass override=True."
        )
    _coordinator.set(key, value)


def get_value(key: str) -> bytes:
    """Retrieve a raw byte payload for *key* from the coordinator store."""
    return _coordinator.get(key)
