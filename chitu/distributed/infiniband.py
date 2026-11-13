# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Detect and configure InfiniBand environment variables.
"""

from typing import Optional
from logging import getLogger
from pathlib import Path
import functools
import os
import re
import subprocess
import sys

logger = getLogger(__name__)
_IB_SYSFS_PATH = Path("/sys/class/infiniband")


@functools.cache
def detect_ib_devices() -> Optional[str]:
    """Detect active IB devices with highest rate, return comma-separated list (e.g. ``mlx5_0,mlx5_3``)."""
    try:
        output = subprocess.check_output(
            ["ibstat"], stderr=subprocess.DEVNULL, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        logger.warning("No `ibstat` command found for detecting IB devices")
        return None

    ca_name: Optional[str] = None
    active: dict[str, int] = {}

    for line in output.splitlines():
        m = re.match(r"^CA '(\S+)'", line)
        if m:
            ca_name = m.group(1)
            continue
        if ca_name is None:
            continue
        if "State:" in line and "Active" in line:
            active.setdefault(ca_name, 0)
        m_rate = re.search(r"Rate:\s*(\d+)", line)
        if m_rate and ca_name in active:
            active[ca_name] = max(active[ca_name], int(m_rate.group(1)))

    if not active:
        logger.warning("No active IB devices found")
        return None

    max_rate = max(active.values())
    devices = sorted(k for k, v in active.items() if v == max_rate)
    return ",".join(devices) if devices else None


def detect_mooncake_ib_devices() -> Optional[str]:
    """Resolve HCA filters to active device names for Mooncake's RDMA port.

    NCCL_IB_HCA takes precedence over NVSHMEM_HCA_LIST. Without either,
    retain the highest-rate autodetection policy. Explicit filters include
    every matching active device, regardless of link rate.
    """
    for env_name in ("NCCL_IB_HCA", "NVSHMEM_HCA_LIST"):
        selection = os.environ.get(env_name, "").strip()
        if selection:
            break
    else:
        return detect_ib_devices()

    exclude = selection.startswith("^")
    if exclude:
        selection = selection[1:]
    exact = env_name == "NVSHMEM_HCA_LIST" or selection.startswith("=")
    if env_name == "NCCL_IB_HCA" and selection.startswith("="):
        selection = selection[1:]

    selectors: list[tuple[str, Optional[int]]] = []
    for entry in selection.split(","):
        # NCCL's optional rail and plane fields describe its topology; only
        # the device and port fields constrain Mooncake's device selection.
        pattern = (
            r"([\w.-]+)(?::(\d*)(?::\d*(?::\d*)?)?)?"
            if env_name == "NCCL_IB_HCA"
            else r"([\w.-]+)(?::(\d+))?"
        )
        match = re.fullmatch(pattern, entry.strip())
        if match is None or (match[2] and not 1 <= int(match[2]) <= 255):
            raise ValueError(f"Invalid {env_name} selector: {entry!r}")
        selectors.append((match[1], int(match[2]) if match[2] else None))

    port_value = os.environ.get("MC_IB_PORT", "1")
    if not port_value.isdecimal() or not 1 <= int(port_value) <= 255:
        raise ValueError(
            f"MC_IB_PORT must be an integer in [1, 255], got {port_value!r}"
        )
    port = int(port_value)
    devices = []
    for device in sorted(_IB_SYSFS_PATH.glob("*")):
        try:
            state = (device / "ports" / str(port) / "state").read_text().strip()
        except FileNotFoundError:
            continue
        if not state.startswith("4:"):
            continue
        matches = any(
            (device.name == name if exact else device.name.startswith(name))
            and (selected_port is None or selected_port == port)
            for name, selected_port in selectors
        )
        if matches != exclude:
            devices.append(device.name)
    if not devices:
        # None/empty lets Mooncake autodetect, which could reintroduce a NIC
        # explicitly excluded by the user.
        raise ValueError(
            f"{env_name}={os.environ[env_name]!r} selects no active IB devices "
            f"for Mooncake port {port}; check {_IB_SYSFS_PATH} and MC_IB_PORT"
        )
    return ",".join(devices)


@functools.cache
def detect_ib_network_interface() -> Optional[str]:
    """Detect active (Up) IB network interface. Prioritize bond interface."""
    try:
        output = subprocess.check_output(
            ["ibdev2netdev"], stderr=subprocess.DEVNULL, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        logger.warning(
            "No `ibdev2netdev` command found for detecting IB network interface"
        )
        return None

    up_ifaces: list[str] = []
    for line in output.splitlines():
        if "Up" not in line:
            continue
        parts = line.split()
        if len(parts) >= 5:
            up_ifaces.append(parts[4])

    if not up_ifaces:
        logger.warning("No active IB network interface found")
        return None

    for iface in up_ifaces:
        if iface.startswith("bond"):
            return iface
    return up_ifaces[0]


@functools.cache
def get_recommended_ib_envs() -> dict[str, str]:
    """Detect and configure InfiniBand environment variables, return the variables in a dict."""
    envs: dict[str, str] = {}

    ib_devices = detect_ib_devices()
    if ib_devices:
        logger.info(f"Detected IB devices: {ib_devices}")
        # NCCL otherwise treats mlx5_1 as a prefix matching mlx5_10 too.
        envs["NCCL_IB_HCA"] = f"={ib_devices}"
        envs["NVSHMEM_HCA_LIST"] = ib_devices

    nvshmem_hcas = os.environ.get("NVSHMEM_HCA_LIST", "").strip()
    if nvshmem_hcas and not os.environ.get("NCCL_IB_HCA", "").strip():
        # Preserve the explicit NVSHMEM filter when automatic setup fills
        # NCCL_IB_HCA, which takes precedence in Mooncake selection.
        envs["NCCL_IB_HCA"] = (
            f"^={nvshmem_hcas[1:]}"
            if nvshmem_hcas.startswith("^")
            else f"={nvshmem_hcas}"
        )

    ib_iface = detect_ib_network_interface()
    if ib_iface:
        logger.info(f"Detected IB network interface: {ib_iface}")
        for key in (
            "GLOO_SOCKET_IFNAME",
            "NCCL_SOCKET_IFNAME",
            "HCCL_SOCKET_IFNAME",
            "NVSHMEM_IB_DEVICE",
        ):
            envs[key] = ib_iface

    return envs


def auto_set_ib_envs():
    envs = get_recommended_ib_envs()
    for name, value in envs.items():
        if name not in os.environ or not os.environ[name]:
            os.environ[name] = value
            logger.info(f"Automatically set {name} to {value}")
        elif os.environ[name] != value:
            logger.info(
                f"Keep user's explicit {name}={os.environ[name]} setting, while the recommended value is {value}"
            )
