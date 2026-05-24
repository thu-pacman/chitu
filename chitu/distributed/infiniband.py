# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Detect and configure InfiniBand environment variables.
"""

from typing import Optional
from logging import getLogger
import os
import re
import subprocess
import sys

logger = getLogger(__name__)


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


def get_recommended_ib_envs() -> dict[str, str]:
    """Detect and configure InfiniBand environment variables, return the variables in a dict."""
    envs: dict[str, str] = {}

    ib_devices = detect_ib_devices()
    if ib_devices:
        logger.info(f"Detected IB devices: {ib_devices}")
        for key in ("NCCL_IB_HCA", "NVSHMEM_HCA_LIST"):
            envs[key] = ib_devices

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
