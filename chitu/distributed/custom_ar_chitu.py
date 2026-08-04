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
from chitu.import_utils import try_import_platform_dep
from chitu.accelerator_monitor import check_accelerator_fully_connected

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")

logger = getLogger(__name__)

custom_ar = False
_backend_checked = False
HYGON_VARLEN_COLLECTIVE_ABI_VERSION = 2


def _hygon_custom_ar_enabled() -> bool:
    return os.environ.get("CHITU_HYGON_CUSTOM_AR", "1").strip() == "1"


def _has_hygon_varlen_collective_api() -> bool:
    version = getattr(chitu_backend, "hygon_varlen_collective_abi_version", None)
    try:
        abi_version = version() if callable(version) else None
        return (
            callable(getattr(chitu_backend, "varlen_all_gather", None))
            and callable(getattr(chitu_backend, "varlen_reduce_scatter", None))
            and isinstance(abi_version, int)
            and abi_version == HYGON_VARLEN_COLLECTIVE_ABI_VERSION
        )
    except Exception:
        return False


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


def _all_ranks_true(group: ProcessGroup, local_value: bool) -> bool:
    """Return True only when every rank in the CPU process group agrees."""

    flag = torch.tensor([int(local_value)], dtype=torch.int32, device="cpu")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


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
        self._supports_varlen_collectives = False
        self._ptr = 0
        self.group = group
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        assert (
            dist.get_backend(group) != dist.Backend.NCCL
        ), "ChituCustomAllreduce should be attached to a non-NCCL group."
        self.meta_ptrs = []
        self.buffer_ptrs = []

        local_is_hygon = is_hygon()
        all_hygon = _all_ranks_true(self.group, local_is_hygon)
        all_non_hygon = _all_ranks_true(self.group, not local_is_hygon)
        local_reason = None
        if not all_hygon and not all_non_hygon:
            local_reason = "ranks disagree on the accelerator platform"
        elif not custom_ar:
            local_reason = "missing custom allreduce library"
        elif all_hygon and not _hygon_custom_ar_enabled():
            local_reason = "CHITU_HYGON_CUSTOM_AR=0"
        elif self.world_size == 1:
            local_reason = "world size is 1"
        elif self.world_size not in ChituCustomAllreduce._SUPPORTED_WORLD_SIZES:
            local_reason = (
                f"unsupported world size {self.world_size}; supported sizes are "
                f"{ChituCustomAllreduce._SUPPORTED_WORLD_SIZES}"
            )

        if not _all_ranks_true(self.group, local_reason is None):
            if local_reason is not None:
                logger.warning("Custom allreduce disabled: %s.", local_reason)
            else:
                logger.warning(
                    "Custom allreduce disabled because another rank failed "
                    "the prerequisite checks."
                )
            self.disabled = True
            return

        self._is_hygon = all_hygon
        if self._is_hygon:
            self._supports_varlen_collectives = _all_ranks_true(
                self.group, _has_hygon_varlen_collective_api()
            )
            if not self._supports_varlen_collectives:
                logger.warning(
                    "Hygon custom allreduce remains available for ordinary AR, "
                    "but varlen AG/RS is unavailable on at least one rank."
                )

        device_error = None
        local_index = None
        physical_device_id = None
        try:
            if isinstance(device, int):
                device = torch.device(f"cuda:{device}")
            elif isinstance(device, str):
                device = torch.device(device)
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
        except Exception as e:
            device_error = e

        if not _all_ranks_true(self.group, device_error is None):
            if device_error is not None:
                logger.warning(
                    "Custom allreduce device mapping failed on rank %d: %s",
                    self.rank,
                    device_error,
                )
            else:
                logger.warning(
                    "Custom allreduce disabled because another rank could not "
                    "resolve its physical device."
                )
            self.disabled = True
            return

        assert isinstance(device, torch.device)
        assert local_index is not None
        assert physical_device_id is not None
        self.device = device
        # HIP graph-pool allocations are not reliably visible through peer IPC
        # mappings on Hygon. Stage into the pre-registered uncached buffer.
        self._use_staging_buffer_in_graph = self._is_hygon

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

        local_p2p_ok = _check_p2p_access(local_index, local_device_ids)
        if not local_p2p_ok:
            logger.warning(
                "Rank %d: P2P access check failed. Custom AR will be disabled.",
                self.rank,
            )

        topology_name = "HSW" if self._is_hygon else "NVLink"
        # The fabric query is host-global. Run it once and fold its result into
        # the same all-rank decision as each rank's local P2P check.
        local_fabric_ok = True
        if self.rank == 0:
            local_fabric_ok, topology_name = check_accelerator_fully_connected(
                physical_device_ids
            )

        self.fully_connected = _all_ranks_true(
            self.group, local_p2p_ok and local_fabric_ok
        )
        if not self.fully_connected:
            # vLLM's upstream check is `world_size > 2 and not fully_connected`
            # because its 2-GPU 1stage kernel is meant to work on PCIe. The
            # cross-device release/acquire barrier still requires a fully
            # connected device fabric here.
            logger.warning(
                "Custom allreduce disabled: not full %s/P2P between all GPUs "
                "in this group (world_size=%d).",
                topology_name,
                self.world_size,
            )
            self.disabled = True
            return

        self.custom_ar_max_size = max_size
        if is_hygon():
            self.custom_ar_max_size = min(
                max_size,
                CUSTOM_ALL_REDUCE_MAX_SIZES["hygon"][self.world_size],
            )
        self.max_size = max_size

        rank_data_error = None
        try:
            self.rank_data = torch.empty(
                8 * 1024 * 1024, dtype=torch.uint8, device=self.device
            )
        except Exception as e:
            rank_data_error = e

        if not _all_ranks_true(self.group, rank_data_error is None):
            if rank_data_error is not None:
                logger.warning(
                    "Rank %d failed to allocate custom AR rank data: %s",
                    self.rank,
                    rank_data_error,
                )
            else:
                logger.warning(
                    "Custom allreduce disabled because another rank could not "
                    "allocate rank data."
                )
            self.disabled = True
            return

        self.meta_ptrs = self.create_shared_buffer(
            chitu_backend.meta_size() + max_size, group=group, uncached=True
        )
        self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)
        local_buffers_ok = not any(p == 0 for p in self.meta_ptrs) and not any(
            p == 0 for p in self.buffer_ptrs
        )
        if not _all_ranks_true(self.group, local_buffers_ok):
            logger.warning(
                "Custom allreduce disabled because shared-buffer setup failed "
                "on at least one rank."
            )
            self.close()
            return

        init_error = None
        try:
            self._ptr = chitu_backend.init_custom_ar(
                self.meta_ptrs, self.rank_data, self.rank, self.fully_connected
            )
        except Exception as e:
            init_error = e

        if not _all_ranks_true(self.group, init_error is None and self._ptr != 0):
            if init_error is not None:
                logger.warning(
                    "Rank %d failed to initialize custom AR: %s",
                    self.rank,
                    init_error,
                )
            else:
                logger.warning(
                    "Custom allreduce initialization failed on at least one rank."
                )
            self.close()
            return

        register_error = None
        try:
            chitu_backend.register_buffer(self._ptr, self.buffer_ptrs)
        except Exception as e:
            register_error = e

        if not _all_ranks_true(self.group, register_error is None):
            if register_error is not None:
                logger.warning(
                    "Rank %d failed to register the custom AR staging buffer: %s",
                    self.rank,
                    register_error,
                )
            else:
                logger.warning(
                    "Custom allreduce staging-buffer registration failed on "
                    "at least one rank."
                )
            self.close()
            return

        logger.info("ChituCustomAllreduce initialized successfully.")

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
        gathered_data = []
        for data in all_data:
            if data is None:
                logger.error("Failed to gather graph buffers metadata")
                return
            gathered_data.append(data)

        handles = [d[0] for d in gathered_data]
        offsets = [d[1] for d in gathered_data]

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
                return inp_size <= self.custom_ar_max_size
            return inp_size < self.max_size

        return False

    @property
    def supports_varlen_collectives(self) -> bool:
        """Whether this manager can run the Hygon variable-length AG/RS API."""

        return (
            not self.disabled
            and getattr(self, "_is_hygon", False)
            and getattr(self, "_supports_varlen_collectives", False)
        )

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

    def varlen_all_gather(
        self,
        inp: torch.Tensor,
        local_count: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Compact DP-to-ETP all-gather driven by a device int32 count.

        ``inp`` and ``out`` share one fixed global row capacity. Every rank
        calls the collective, including ranks whose logical row count is zero.
        Input is first copied to the manager's persistent uncached staging
        allocation.
        """
        if not self.supports_varlen_collectives or self._ptr == 0:
            raise RuntimeError(
                "custom all-reduce manager does not support Hygon varlen AG/RS"
            )
        chitu_backend.varlen_all_gather(
            self._ptr,
            inp,
            out,
            local_count,
            self.buffer_ptrs[self.rank],
            self.max_size,
        )
        return out

    def varlen_reduce_scatter(
        self,
        inp: torch.Tensor,
        local_count: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Compact ETP-to-DP reduce-scatter with one global row capacity."""
        if not self.supports_varlen_collectives or self._ptr == 0:
            raise RuntimeError(
                "custom all-reduce manager does not support Hygon varlen AG/RS"
            )
        chitu_backend.varlen_reduce_scatter(
            self._ptr,
            inp,
            out,
            local_count,
            self.buffer_ptrs[self.rank],
            self.max_size,
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
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        zeros = [0] * world_size
        pointer = 0
        handle = None
        allocation_error = None

        try:
            if not hasattr(chitu_backend, "allocate_shared_buffer_and_handle"):
                raise RuntimeError(
                    "chitu_backend missing allocate_shared_buffer_and_handle"
                )
            pointer, handle = chitu_backend.allocate_shared_buffer_and_handle(
                size_in_bytes
            )
        except Exception as e:
            allocation_error = e

        if not _all_ranks_true(
            group, allocation_error is None and pointer != 0 and handle is not None
        ):
            if allocation_error is not None:
                logger.warning(
                    "Rank %d failed to allocate a custom AR shared buffer: %s",
                    rank,
                    allocation_error,
                )
            free_fn = getattr(chitu_backend, "free_shared_buffer", None)
            if pointer != 0 and callable(free_fn):
                try:
                    free_fn(pointer)
                except Exception:
                    pass
            return zeros

        handles = [None] * world_size
        # A process-group failure is not a safe local fallback: let it raise so
        # the caller does not split ranks between custom AR and RCCL.
        dist.all_gather_object(handles, handle, group=group)

        pointers = [0] * world_size
        pointers[rank] = pointer
        open_error = None
        for i, peer_handle in enumerate(handles):
            if i == rank:
                continue
            try:
                if not hasattr(chitu_backend, "open_mem_handle"):
                    raise RuntimeError("chitu_backend missing open_mem_handle")
                if peer_handle is None:
                    raise RuntimeError(f"Rank {i} returned an invalid IPC handle")
                peer_pointer = chitu_backend.open_mem_handle(peer_handle)
                if peer_pointer == 0:
                    raise RuntimeError(f"failed to open IPC handle from rank {i}")
                pointers[i] = peer_pointer
            except Exception as e:
                open_error = e
                break

        if not _all_ranks_true(
            group, open_error is None and all(pointer != 0 for pointer in pointers)
        ):
            if open_error is not None:
                logger.warning(
                    "Rank %d failed to open a custom AR peer buffer: %s",
                    rank,
                    open_error,
                )
            free_fn = getattr(chitu_backend, "free_shared_buffer", None)
            if pointer != 0 and callable(free_fn):
                try:
                    free_fn(pointer)
                except Exception:
                    pass
            return zeros

        return pointers

    @staticmethod
    def free_shared_buffer(
        pointers: List[int],
        group: Optional[ProcessGroup] = None,
        rank: Optional[int] = None,
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
