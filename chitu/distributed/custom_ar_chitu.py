# SPDX-FileCopyrightText: 2025 Qingcheng.AI
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from typing import List, Optional, Union
import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from logging import getLogger

from chitu.import_utils import try_import_platform_dep

ops = None
has_chitu_backend = False
custom_ar = False
_backend_checked = False


def _init_backend():
    global ops, has_chitu_backend, custom_ar, _backend_checked
    if _backend_checked:
        return
    _backend_checked = True
    ops, has_chitu_backend = try_import_platform_dep("chitu_backend")
    try:
        if has_chitu_backend:
            ops.meta_size()
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
}

logger = getLogger(__name__)


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


def _check_p2p_access(rank: int, world_size: int) -> bool:
    for i in range(world_size):
        if i == rank:
            continue
        try:
            if not torch.cuda.can_device_access_peer(rank, i):
                return False
        except Exception:
            return False
    return True


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

        try:
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible_devices:
                device_ids = list(map(int, cuda_visible_devices.split(",")))
                physical_device_id = device_ids[device.index]
            else:
                physical_device_id = device.index

            tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
            gather_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(gather_list, tensor, group=self.group)

            self.fully_connected = True

            if not _check_p2p_access(self.rank, self.world_size):
                logger.warning(
                    f"Rank {self.rank}: P2P access check failed. Custom AR disabled."
                )
                self.disabled = True
                return

            if self.world_size > 2 and not self.fully_connected:
                logger.warning(
                    "Custom allreduce disabled: >2 GPUs without full interconnect."
                )
                self.disabled = True
                return

        except Exception as e:
            logger.warning(
                f"Topology detection failed: {e}. Defaulting to fully_connected=True"
            )
            self.fully_connected = True

        self.max_size = max_size

        try:
            # Metadata buffer
            self.meta_ptrs = self.create_shared_buffer(
                ops.meta_size() + max_size, group=group, uncached=True
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

            self._ptr = ops.init_custom_ar(
                self.meta_ptrs, self.rank_data, self.rank, self.fully_connected
            )

            if self._ptr == 0:
                logger.warning("ops.init_custom_ar returned 0 handle.")
                self.disabled = True
            else:
                ops.register_buffer(self._ptr, self.buffer_ptrs)
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
            if not self.disabled:
                self.register_graph_buffers()

    def _register_for_cuda_graph_capture(self):
        from chitu.cuda_graph import add_post_hook_for_currently_capturing_graph_object

        if torch.cuda.is_current_stream_capturing():
            has_run = [False]

            def post_hook():
                if not has_run[0]:
                    if not self.disabled and self._ptr != 0:
                        self.register_graph_buffers()
                    has_run[0] = True

            add_post_hook_for_currently_capturing_graph_object(post_hook)
            self._IS_CAPTURING = True

    def register_graph_buffers(self):
        if self.disabled or self._ptr == 0:
            return

        handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)

        local_data = [handle, offset]
        all_data = [None for _ in range(self.world_size)]
        dist.all_gather_object(all_data, local_data, group=self.group)

        handles = [d[0] for d in all_data]
        offsets = [d[1] for d in all_data]

        if any(h is None for h in handles):
            logger.error("Failed to gather graph buffers metadata")
            return

        ops.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False

        inp_size = inp.numel() * inp.element_size()

        if inp_size % 16 != 0:
            return False

        if not is_weak_contiguous(inp):
            return False

        if self.world_size == 2 or self.fully_connected:
            return inp_size < self.max_size

        return False

    def all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor = None, registered: bool = False
    ):
        if out is None:
            out = torch.empty_like(inp)

        if registered:
            ops.all_reduce(self._ptr, inp, out, 0, 0)
        else:
            ops.all_reduce(
                self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size
            )
        return out

    def custom_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:

        if torch.cuda.is_current_stream_capturing():
            self._register_for_cuda_graph_capture()

        if input.numel() == 0:
            return input

        if self.disabled or not self.should_custom_ar(input):
            return None

        if self._IS_CAPTURING and torch.cuda.is_current_stream_capturing():
            registered = True
        else:
            registered = False

        self.all_reduce(input, out=input, registered=registered)
        return input

    def close(self):
        try:
            if self.disabled:
                return

            if self._ptr:
                if ops is not None:
                    dispose_fn = getattr(ops, "dispose", None)
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
            if hasattr(ops, "allocate_shared_buffer_and_handle"):
                pointer, handle = ops.allocate_shared_buffer_and_handle(size_in_bytes)

                world_size = dist.get_world_size(group=group)
                rank = dist.get_rank(group=group)

                handles = [None] * world_size
                dist.all_gather_object(handles, handle, group=group)

                pointers = []
                for i, h in enumerate(handles):
                    if i == rank:
                        pointers.append(pointer)
                    else:
                        if hasattr(ops, "open_mem_handle") and h is not None:
                            ptr = ops.open_mem_handle(h)
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
                logger.warning("ops missing allocate_shared_buffer_and_handle")
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
        if ops is not None:
            try:
                if pointers[rank] != 0:
                    free_fn = getattr(ops, "free_shared_buffer", None)
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
