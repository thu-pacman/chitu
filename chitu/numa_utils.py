# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
import subprocess
import os

import torch

from chitu.device_type import is_nvidia
from chitu.utils import try_import_opt_dep

numa, has_numa = try_import_opt_dep("numa", "cpu")


logger = getLogger(__name__)


def _get_numa_of_gpu(gpu_id: int):
    if not is_nvidia():
        raise NotImplementedError(
            "Detecting NUMA node near GPU is only supported on NVIDIA GPUs"
        )

    # Get PCI bus ID from nvidia-smi
    result = subprocess.run(
        f"nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i {gpu_id}",
        shell=True,
        capture_output=True,
        text=True,
        check=True,
    )
    pci_bus_id = result.stdout.strip()

    # Convert format (e.g., 00000000:3B:00.0 -> 0000:3b:00.0)
    pci_parts = pci_bus_id.split(":")
    if len(pci_parts) != 3:
        raise RuntimeError(f"Invalid PCI bus ID: {pci_bus_id}")
    pci_bus_id = f"{pci_parts[0][-4:]}:{pci_parts[1]}:{pci_parts[2]}"
    pci_bus_id = pci_bus_id.lower()

    # Read from /sys/bus/pci/devices/0000:3b:00.0/numa_node
    with open(f"/sys/bus/pci/devices/{pci_bus_id}/numa_node") as f:
        return int(f.read().strip())

    raise RuntimeError("NUMA node not found")


def _bind_process_to_numa_id(numa_id: int):
    if numa_id > numa.get_max_node():
        raise RuntimeError(
            f"NUMA node {numa_id} out of range [0, {numa.get_max_node()}]"
        )
    numa.bind({numa_id})
    logger.info(f"Bound process to NUMA node {numa_id}")


def bind_process_to_numa(binding_type: str):
    if binding_type == "none":
        pass

    elif binding_type == "one_numa_per_rank":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        _bind_process_to_numa_id(local_rank)

    elif binding_type == "numa_near_device":
        device_id = torch.cuda.current_device()
        try:
            numa_id = _get_numa_of_gpu(device_id)
            _bind_process_to_numa_id(numa_id)
        except Exception as e:
            logger.warning(
                f"Failed to find a NUMA node near GPU {device_id} and bind to it: {e}"
            )

    else:
        raise ValueError(f"Unknown NUMA binding type: {binding_type}")
