# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import functools
import logging
import os

from chitu.device_type import is_hygon
from chitu.import_utils import try_import_opt_dep, try_import_platform_dep

pynvml, has_pynvml = try_import_opt_dep("pynvml", "nvidia-ml-py")
amdsmi, has_amdsmi = try_import_platform_dep("amdsmi")

logger = logging.getLogger(__name__)


@functools.cache
def get_nvml():
    if not has_pynvml:
        raise RuntimeError("nvidia-ml-py is not available")
    try:
        pynvml.nvmlInit()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize NVML: {e}") from e
    atexit.register(_shutdown_nvml)
    return pynvml


@functools.cache
def get_amdsmi():
    if not has_amdsmi:
        raise RuntimeError("AMD SMI is not available")
    try:
        amdsmi.amdsmi_init()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize AMD SMI: {e}") from e
    atexit.register(_shutdown_amdsmi)
    return amdsmi


@functools.cache
def get_nvml_handle(device_index: int):
    nvml = get_nvml()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        if cuda_visible_devices:
            device_tokens = [
                token.strip()
                for token in cuda_visible_devices.split(",")
                if token.strip()
            ]
            if device_index < len(device_tokens):
                token = device_tokens[device_index]
                if token.startswith(("GPU-", "MIG-")):
                    return nvml.nvmlDeviceGetHandleByUUID(token)
                try:
                    return nvml.nvmlDeviceGetHandleByIndex(int(token))
                except ValueError as e:
                    raise RuntimeError(
                        f"Invalid CUDA_VISIBLE_DEVICES token {token!r}: {e}"
                    ) from e
        return nvml.nvmlDeviceGetHandleByIndex(device_index)
    except Exception as e:
        raise RuntimeError(
            f"Failed to get NVML handle for device {device_index}: {e}"
        ) from e


@functools.cache
def get_amdsmi_handle(device_index: int):
    smi = get_amdsmi()
    try:
        handles = smi.amdsmi_get_processor_handles()
    except Exception as e:
        raise RuntimeError(f"Failed to get AMD SMI processor handles: {e}") from e

    for visible_devices_env in (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        visible_devices = os.environ.get(visible_devices_env)
        if not visible_devices:
            continue
        device_tokens = [
            token.strip() for token in visible_devices.split(",") if token.strip()
        ]
        if device_index >= len(device_tokens):
            continue
        token = device_tokens[device_index]
        try:
            return handles[int(token)]
        except (ValueError, IndexError) as e:
            raise RuntimeError(
                f"Failed to map device {device_index} from {visible_devices_env} token {token!r}: {e}"
            ) from e

    try:
        return handles[device_index]
    except IndexError as e:
        raise RuntimeError(
            f"AMD SMI handle index {device_index} is out of range: {e}"
        ) from e


def check_accelerator_fully_connected(physical_device_ids: list[int]):
    topology_name = "HSW" if is_hygon() else "NVLink"
    try:
        if is_hygon():
            return _check_full_hsw(physical_device_ids), topology_name
        return _check_full_nvlink(physical_device_ids), topology_name
    except Exception as e:
        logger.warning("Failed to verify %s topology: %s", topology_name, e)
        return False, topology_name


def get_accelerator_memory_bytes(device_index: int, pid: int | None = None):
    """Return accelerator memory usage and total bytes for a device.

    When ``pid`` is provided, process-specific memory usage is returned when the
    device management library can identify that process. In Docker containers,
    use ``--pid=host`` so NVML and AMD SMI report process IDs in the same PID
    namespace as Python. Without ``--pid=host``, process-specific lookup may
    fail and the implementation may fall back to device-level memory usage.
    """
    try:
        if is_hygon():
            return _get_amdsmi_memory_bytes(device_index, pid)
        return _get_nvml_memory_bytes(device_index, pid)
    except Exception as e:
        logger.warning("Failed to get accelerator memory bytes: %s", e)
        return None


def _check_full_hsw(physical_device_ids: list[int]) -> bool:
    smi = get_amdsmi()
    try:
        handles = smi.amdsmi_get_processor_handles()
        if len(set(physical_device_ids)) != len(physical_device_ids):
            raise RuntimeError(f"Duplicate device IDs: {physical_device_ids}")
        if any(
            device_id < 0 or device_id >= len(handles)
            for device_id in physical_device_ids
        ):
            raise RuntimeError(
                f"Device IDs {physical_device_ids} are out of range for {len(handles)} AMD SMI handles"
            )

        hsw_type = smi.AmdSmiIoLinkType.XGMI.value
        for i, src_id in enumerate(physical_device_ids):
            for dst_id in physical_device_ids[i + 1 :]:
                link = smi.amdsmi_topo_get_link_type(handles[src_id], handles[dst_id])
                link_type = getattr(link["type"], "value", link["type"])
                if int(link["hops"]) != 1 or int(link_type) != hsw_type:
                    return False
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to verify HSW topology: {e}") from e


def _check_full_nvlink(physical_device_ids: list[int]) -> bool:
    nvml = get_nvml()
    try:
        handles = [nvml.nvmlDeviceGetHandleByIndex(int(i)) for i in physical_device_ids]
        for i, h_i in enumerate(handles):
            for j, h_j in enumerate(handles):
                if i >= j:
                    continue
                try:
                    st = nvml.nvmlDeviceGetP2PStatus(
                        h_i, h_j, nvml.NVML_P2P_CAPS_INDEX_NVLINK
                    )
                except nvml.NVMLError as e:
                    raise RuntimeError(
                        f"Failed to query NVLink P2P status for device pair ({i}, {j}): {e}"
                    ) from e
                if st != nvml.NVML_P2P_STATUS_OK:
                    return False
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to verify NVLink topology: {e}") from e


def _get_nvml_memory_bytes(device_index: int, pid: int | None):
    nvml = get_nvml()
    try:
        handle = get_nvml_handle(device_index)
        mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
        total = int(mem_info.total)
        used = int(mem_info.used)
        if pid is not None:
            try:
                processes = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except Exception as e:
                try:
                    processes = nvml.nvmlDeviceGetComputeRunningProcesses_v2(handle)
                except Exception as e2:
                    raise RuntimeError(
                        "Failed to get NVML compute running processes "
                        f"with both APIs: {e}; fallback error: {e2}"
                    ) from e2
            for proc in processes:
                if proc.pid != pid:
                    continue
                proc_used = getattr(proc, "usedGpuMemory", None)
                if proc_used is not None and proc_used > 0:
                    used = int(proc_used)
                break
        return used, total
    except Exception as e:
        raise RuntimeError(
            f"Failed to get NVML memory bytes for device {device_index}: {e}"
        ) from e


def _extract_amdsmi_int(mapping, keys):
    if not isinstance(mapping, dict):
        raise TypeError(
            f"Expected dict while extracting AMD SMI value, got {type(mapping).__name__}"
        )
    last_error = None
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, dict):
            try:
                return _extract_amdsmi_int(value, keys)
            except Exception as e:
                last_error = e
                continue
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            last_error = e
            continue
    if last_error is not None:
        raise KeyError(
            f"None of AMD SMI keys {keys} yielded an integer; last error: {last_error}"
        ) from last_error
    raise KeyError(f"None of AMD SMI keys {keys} were found")


def _get_amdsmi_process_memory_bytes(smi, handle, pid: int):
    try:
        process_ids = smi.amdsmi_get_gpu_process_list(handle)
    except Exception as e:
        raise RuntimeError(f"Failed to get AMD SMI GPU process list: {e}") from e

    for process in process_ids:
        try:
            process_pid = _extract_amdsmi_int(process, ("pid", "process_id"))
        except Exception as e:
            try:
                process_pid = int(process)
            except (TypeError, ValueError) as e2:
                raise RuntimeError(
                    f"Failed to extract AMD SMI process pid from {process!r}: {e}; fallback error: {e2}"
                ) from e2
        if process_pid != pid:
            continue

        try:
            used = _extract_amdsmi_int(
                process,
                (
                    "vram_usage",
                    "memory_usage",
                    "mem_usage",
                    "memory_used",
                    "used_memory",
                    "vram_mem",
                    "mem",
                ),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract AMD SMI memory usage for pid {pid}: {e}"
            ) from e
        if used > 0:
            return used
        raise RuntimeError(
            f"AMD SMI memory usage for pid {pid} is not positive: {used}"
        )
    raise RuntimeError(
        f"AMD SMI process list does not contain pid {pid}. In Docker containers, "
        "use --pid=host for process-specific memory accounting."
    )


def _get_amdsmi_memory_bytes(device_index: int, pid: int | None):
    smi = get_amdsmi()
    try:
        handle = get_amdsmi_handle(device_index)
        mem_type = smi.AmdSmiMemoryType.VRAM
        total = int(smi.amdsmi_get_gpu_memory_total(handle, mem_type))
        used = int(smi.amdsmi_get_gpu_memory_usage(handle, mem_type))
        if pid is not None:
            try:
                used = _get_amdsmi_process_memory_bytes(smi, handle, pid)
            except Exception as e:
                logger.warning("Failed to get AMD SMI process memory usage: %s", e)
        return used, total
    except Exception as e:
        raise RuntimeError(
            f"Failed to get AMD SMI memory bytes for device {device_index}: {e}"
        ) from e


def _shutdown_nvml():
    try:
        pynvml.nvmlShutdown()
    except Exception as e:
        logger.debug("Failed to shut down NVML cleanly: %s", e)


def _shutdown_amdsmi():
    try:
        amdsmi.amdsmi_shut_down()
    except Exception as e:
        logger.debug("Failed to shut down AMD SMI cleanly: %s", e)
