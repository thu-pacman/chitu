# SPDX-FileCopyrightText: 2025 Qingcheng.AI
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from typing import List, Optional, Union
from logging import getLogger
import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from chitu.device_type import is_hygon
from chitu.import_utils import try_import_platform_dep, try_import_opt_dep

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
pynvml, has_pynvml = try_import_opt_dep("pynvml", "nvidia-ml-py")

logger = getLogger(__name__)

custom_ar = False
_backend_checked = False


def _hygon_custom_ar_enabled() -> bool:
    return os.environ.get("CHITU_HYGON_CUSTOM_AR", "1").strip() == "1"


def _init_backend():
    global custom_ar, _backend_checked
    if _backend_checked:
        return
    _backend_checked = True
    try:
        if has_chitu_backend:
            chitu_backend.meta_size()
            custom_ar = True
        else:
            custom_ar = False
    except Exception:
        custom_ar = False


MiB = 1024 * 1024
CUSTOM_ALL_REDUCE_MAX_SIZES = {
    "9.0": {
        2: 64 * MiB,
        4: 32 * MiB,
        6: MiB // 2,
        8: MiB // 4,
    },
    "10.0": {
        2: 2 * MiB,
        4: 2 * MiB,
        6: 1 * MiB,
        8: 1 * MiB,
    },
    # Full-HSW TP2/4/6/8 CUDA-graph measurements validated custom AR through
    # a BF16 [256, 7168] payload (3.5 MiB).
    "hygon": {
        2: 7 * MiB // 2,
        4: 7 * MiB // 2,
        6: 7 * MiB // 2,
        8: 7 * MiB // 2,
    },
}


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


def _check_p2p_access(local_device_id: int, local_device_ids: List[int]) -> bool:
    for peer_device_id in local_device_ids:
        if peer_device_id == local_device_id:
            continue
        try:
            if not torch.cuda.can_device_access_peer(local_device_id, peer_device_id):
                return False
        except Exception:
            return False
    return True


def _check_full_hsw(physical_device_ids: List[int]) -> bool:
    """Return whether every Hygon device pair has a direct HSW link."""
    initialized = False
    try:
        amdsmi, has_amdsmi = try_import_platform_dep("amdsmi")
        if not has_amdsmi:
            logger.warning(
                "amdsmi is not available; cannot verify HSW topology. "
                "Disabling custom AR."
            )
            return False

        amdsmi.amdsmi_init()
        initialized = True
        handles = amdsmi.amdsmi_get_processor_handles()
        if len(set(physical_device_ids)) != len(physical_device_ids) or any(
            device_id < 0 or device_id >= len(handles)
            for device_id in physical_device_ids
        ):
            return False

        hsw_type = amdsmi.AmdSmiIoLinkType.XGMI.value
        for i, src_id in enumerate(physical_device_ids):
            for dst_id in physical_device_ids[i + 1 :]:
                link = amdsmi.amdsmi_topo_get_link_type(
                    handles[src_id], handles[dst_id]
                )
                link_type = getattr(link["type"], "value", link["type"])
                if int(link["hops"]) != 1 or int(link_type) != hsw_type:
                    return False
        return True
    except Exception as e:
        logger.warning("Failed to verify HSW topology: %s", e)
        return False
    finally:
        if initialized:
            try:
                amdsmi.amdsmi_shut_down()
            except Exception as e:
                logger.debug("Failed to shut down amdsmi cleanly: %s", e)


def _check_full_nvlink(physical_device_ids) -> bool:
    """Every pair of GPUs is connected by NVLink (1 hop).

    Matches vLLM's `is_fully_connected` (vllm/platforms/cuda.py): query NVML
    via nvidia-ml-py for each pair with `NVML_P2P_CAPS_INDEX_NVLINK`. Only returns
    True if every pair reports `NVML_P2P_STATUS_OK`. PCIe-only hosts will see
    `NOT_SUPPORTED` and correctly yield False — which is the condition the
    custom AR spin-barrier kernel relies on.
    """
    if not has_pynvml:
        logger.warning(
            "nvidia-ml-py not available; cannot verify NVLink. Disabling custom AR."
        )
        return False
    try:
        pynvml.nvmlInit()
        try:
            handles = [
                pynvml.nvmlDeviceGetHandleByIndex(int(i)) for i in physical_device_ids
            ]
            for i, h_i in enumerate(handles):
                for j, h_j in enumerate(handles):
                    if i >= j:
                        continue
                    try:
                        st = pynvml.nvmlDeviceGetP2PStatus(
                            h_i, h_j, pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                        )
                        if st != pynvml.NVML_P2P_STATUS_OK:
                            return False
                    except pynvml.NVMLError:
                        return False
            return True
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return False


class ChituCustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size=8192 * 1024,
        symm_mem_enabled=False,
    ) -> None:
        _init_backend()
        self._IS_CAPTURING = False
        self.disabled = False
        self._ptr = 0
        self.group = group
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)

        if not custom_ar:
            logger.info("Custom allreduce is disabled: missing library.")
            self.disabled = True
            return

        if is_hygon() and not _hygon_custom_ar_enabled():
            logger.info("Hygon custom allreduce disabled by CHITU_HYGON_CUSTOM_AR=0.")
            self.disabled = True
            return

        assert (
            dist.get_backend(group) != dist.Backend.NCCL
        ), "ChituCustomAllreduce should be attached to a non-NCCL group."

        if self.world_size == 1:
            self.disabled = True
            return

        if self.world_size not in ChituCustomAllreduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                f"Custom allreduce disabled: unsupported world size {self.world_size}. "
                f"Supported: {ChituCustomAllreduce._SUPPORTED_WORLD_SIZES}"
            )
            self.disabled = True
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        # HIP graph-pool allocations are not reliably visible through peer IPC
        # mappings on Hygon. Stage into the pre-registered uncached buffer.
        self._use_staging_buffer_in_graph = is_hygon()

        try:
            local_index = device.index
            if local_index is None:
                local_index = torch.cuda.current_device()
            visible_devices = next(
                (
                    value
                    for name in (
                        "CUDA_VISIBLE_DEVICES",
                        "HIP_VISIBLE_DEVICES",
                        "ROCR_VISIBLE_DEVICES",
                    )
                    if (value := os.environ.get(name))
                ),
                None,
            )
            if visible_devices:
                device_ids = list(map(int, visible_devices.split(",")))
                physical_device_id = device_ids[local_index]
            else:
                physical_device_id = local_index

            topology = torch.tensor(
                [local_index, physical_device_id], dtype=torch.int, device="cpu"
            )
            gather_list = [torch.zeros_like(topology) for _ in range(self.world_size)]
            dist.all_gather(gather_list, topology, group=self.group)

            local_device_ids = [int(t[0].item()) for t in gather_list]
            physical_device_ids = [int(t[1].item()) for t in gather_list]

            if len(set(local_device_ids)) != self.world_size:
                logger.warning(
                    "Custom allreduce disabled: group is not contained on one "
                    "host with unique local device IDs: %s",
                    local_device_ids,
                )
                self.disabled = True
                return

            if not _check_p2p_access(local_index, local_device_ids):
                logger.warning(
                    f"Rank {self.rank}: P2P access check failed. Custom AR disabled."
                )
                self.disabled = True
                return

            if is_hygon():
                self.fully_connected = _check_full_hsw(physical_device_ids)
                topology_name = "HSW"
            else:
                self.fully_connected = _check_full_nvlink(physical_device_ids)
                topology_name = "NVLink"

            if not self.fully_connected:
                # vLLM's upstream check is `world_size > 2 and not fully_connected`
                # because its 2-GPU 1stage kernel is meant to work on PCIe. In
                # practice the cross-device release/acquire.sys barrier is still
                # unreliable on PCIe-only hosts (see vllm_custom_all_reduce.cuh's
                # multi_gpu_barrier), so we keep the stricter rule: any group
                # without a fully connected device fabric disables custom AR.
                logger.warning(
                    "Custom allreduce disabled: not full %s between all GPUs "
                    "in this group (world_size=%d). P2P alone cannot guarantee the "
                    "cross-device atomic / release-acquire visibility the barrier "
                    "kernel depends on.",
                    topology_name,
                    self.world_size,
                )
                self.disabled = True
                return

        except Exception as e:
            logger.warning(
                f"Topology detection failed: {e}. Disabling custom AR for safety."
            )
            self.disabled = True
            return

        if is_hygon():
            max_size = min(
                max_size,
                CUSTOM_ALL_REDUCE_MAX_SIZES["hygon"][self.world_size],
            )
        self.max_size = max_size

        try:
            # Metadata buffer
            self.meta_ptrs = self.create_shared_buffer(
                chitu_backend.meta_size() + max_size, group=group, uncached=True
            )

            # Data buffer (IPC)
            self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)

            # Rank data buffer
            self.rank_data = torch.empty(
                8 * 1024 * 1024, dtype=torch.uint8, device=self.device
            )

            # <--- 再次检查防止空指针传入
            if any(p == 0 for p in self.meta_ptrs) or any(
                p == 0 for p in self.buffer_ptrs
            ):
                raise RuntimeError(
                    "Found invalid (0) pointers in shared buffer initialization"
                )

            self._ptr = chitu_backend.init_custom_ar(
                self.meta_ptrs, self.rank_data, self.rank, self.fully_connected
            )

            if self._ptr == 0:
                logger.warning("chitu_backend.init_custom_ar returned 0 handle.")
                self.disabled = True
            else:
                chitu_backend.register_buffer(self._ptr, self.buffer_ptrs)
                logger.info("ChituCustomAllreduce initialized successfully.")

        except Exception as e:
            logger.warning(f"Failed to initialize custom allreduce buffers: {e}")
            self.disabled = True
            self.close()
            self._ptr = 0

    @contextmanager
    def capture(self):
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled and not self._use_staging_buffer_in_graph:
                self.register_graph_buffers()

    def _register_for_cuda_graph_capture(self):
        from chitu.cuda_graph import add_post_hook_for_currently_capturing_graph_object

        if torch.cuda.is_current_stream_capturing():
            self._IS_CAPTURING = True
            if self._use_staging_buffer_in_graph:
                return
            has_run = [False]

            def post_hook():
                if not has_run[0]:
                    if not self.disabled and self._ptr != 0:
                        self.register_graph_buffers()
                    has_run[0] = True

            add_post_hook_for_currently_capturing_graph_object(post_hook)

    def register_graph_buffers(self):
        if self.disabled or self._ptr == 0:
            return

        handle, offset = chitu_backend.get_graph_buffer_ipc_meta(self._ptr)

        local_data = [handle, offset]
        all_data = [None for _ in range(self.world_size)]
        dist.all_gather_object(all_data, local_data, group=self.group)

        handles = [d[0] for d in all_data]
        offsets = [d[1] for d in all_data]

        if any(h is None for h in handles):
            logger.error("Failed to gather graph buffers metadata")
            return

        chitu_backend.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False

        inp_size = inp.numel() * inp.element_size()

        if inp_size % 16 != 0:
            return False

        if not is_weak_contiguous(inp):
            return False

        if self.world_size == 2 or self.fully_connected:
            if is_hygon():
                return inp_size <= self.max_size
            return inp_size < self.max_size

        return False

    def all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor = None, registered: bool = False
    ):
        if out is None:
            out = torch.empty_like(inp)

        if registered:
            chitu_backend.all_reduce(self._ptr, inp, out, 0, 0)
        else:
            chitu_backend.all_reduce(
                self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size
            )
        return out

    def custom_all_reduce(
        self, input: torch.Tensor, *, maybe_inplace: bool = True
    ) -> Optional[torch.Tensor]:

        if torch.cuda.is_current_stream_capturing():
            self._register_for_cuda_graph_capture()

        if input.numel() == 0:
            return input

        if self.disabled or not self.should_custom_ar(input):
            return None

        registered = (
            self._IS_CAPTURING
            and torch.cuda.is_current_stream_capturing()
            and not self._use_staging_buffer_in_graph
        )
        out = torch.empty_like(input)

        self.all_reduce(input, out=out, registered=registered)
        return out

    def close(self):
        try:
            if self.disabled:
                return

            if self._ptr:
                if has_chitu_backend:
                    dispose_fn = getattr(chitu_backend, "dispose", None)
                    if callable(dispose_fn):
                        dispose_fn(self._ptr)
                self._ptr = 0

            if hasattr(self, "meta_ptrs"):
                self.free_shared_buffer(self.meta_ptrs, rank=self.rank)
            if hasattr(self, "buffer_ptrs"):
                self.free_shared_buffer(self.buffer_ptrs, rank=self.rank)

            self.disabled = True
        except Exception as e:
            print(f"Error checking custom_ar_chitu close: {e}")

    def __del__(self):
        self.close()

    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int,
        group: ProcessGroup = None,
        uncached: bool = False,
    ) -> List[int]:
        try:
            if hasattr(chitu_backend, "allocate_shared_buffer_and_handle"):
                pointer, handle = chitu_backend.allocate_shared_buffer_and_handle(
                    size_in_bytes
                )

                world_size = dist.get_world_size(group=group)
                rank = dist.get_rank(group=group)

                handles = [None] * world_size
                dist.all_gather_object(handles, handle, group=group)

                pointers = []
                for i, h in enumerate(handles):
                    if i == rank:
                        pointers.append(pointer)
                    else:
                        if hasattr(chitu_backend, "open_mem_handle") and h is not None:
                            ptr = chitu_backend.open_mem_handle(h)
                            if ptr == 0:
                                raise RuntimeError(
                                    f"Rank {rank}: Failed to open IPC handle from Rank {i}"
                                )
                            pointers.append(ptr)
                        else:
                            logger.warning(
                                f"Missing open_mem_handle or invalid handle from Rank {i}"
                            )
                            pointers.append(0)
                return pointers
            else:
                logger.warning(
                    "chitu_backend missing allocate_shared_buffer_and_handle"
                )
                return [0] * dist.get_world_size(group=group)

        except Exception as e:
            logger.warning(f"Failed to create shared buffer: {e}")
            return [0] * dist.get_world_size(group=group)

    @staticmethod
    def free_shared_buffer(
        pointers: List[int],
        group: ProcessGroup = None,
        rank: int = None,
    ) -> None:
        if rank is None:
            rank = dist.get_rank(group=group)
        if has_chitu_backend:
            try:
                if pointers[rank] != 0:
                    free_fn = getattr(chitu_backend, "free_shared_buffer", None)
                    if callable(free_fn):
                        free_fn(pointers[rank])
                    pointers[rank] = 0
            except Exception:
                pass


def create_chitu_custom_allreduce(
    group: ProcessGroup,
    device: torch.device,
    max_size: int = 8192 * 1024,
    symm_mem_enabled: bool = False,
) -> Optional[ChituCustomAllreduce]:
    try:
        return ChituCustomAllreduce(
            group=group,
            device=device,
            max_size=max_size,
            symm_mem_enabled=symm_mem_enabled,
        )
    except Exception as e:
        logger.warning(f"Failed to create ChituCustomAllreduce: {e}")
        return None
