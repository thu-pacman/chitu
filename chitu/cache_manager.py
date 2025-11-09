# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence, Optional, Callable, Iterable
from typing_extensions import override
from dataclasses import dataclass
from logging import getLogger
import torch
from collections import deque
import functools

from chitu.cuda_graph import cuda_graph_safe_cached_property
from chitu.global_vars import get_slot_handle, get_timers, get_global_args
from chitu.static_tensor import StaticTensor
from chitu.batched_seq_len import BatchedSeqLen, BatchedSeqLenDelta
from chitu.utils import ceil_div

logger = getLogger(__name__)


class GlobalLocalMap:
    """
    A mapping between global indices (0..N) and local offsets for one instance.

    Supports two modes:
      1. range mode: [begin_idx, end_idx)
      2. list mode: arbitrary list of global indices (may be non-contiguous)
    """

    __slots__ = ("_mode", "_begin", "_end", "_list", "_map")

    def __init__(
        self,
        begin_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        idx_list: Optional[Iterable[int]] = None,
    ):
        # Validate mode selection
        if idx_list is not None and (begin_idx is not None or end_idx is not None):
            raise ValueError("Cannot provide both range and idx_list.")

        if idx_list is not None:
            # list mode
            self._mode = "list"
            lst: list[int] = list(idx_list)
            if len(lst) != len(set(lst)):
                raise ValueError("Duplicate global indices in idx_list.")
            self._list = lst
            self._map = {g: i for i, g in enumerate(lst)}  # global -> local
            self._begin = None
            self._end = None

        elif begin_idx is not None and end_idx is not None:
            # range mode
            if not (0 <= begin_idx <= end_idx):
                raise ValueError(
                    "Invalid range: must satisfy 0 <= begin_idx <= end_idx."
                )
            self._mode = "range"
            self._begin = int(begin_idx)
            self._end = int(end_idx)
            self._list = None
            self._map = None
        else:
            raise ValueError("Must provide either [begin_idx, end_idx) or idx_list.")

    @classmethod
    def from_range(cls, begin_idx: int, end_idx: int) -> "GlobalLocalMap":
        return cls(begin_idx=begin_idx, end_idx=end_idx)

    @classmethod
    def from_list(cls, idx_list: Iterable[int]) -> "GlobalLocalMap":
        return cls(idx_list=idx_list)

    def to_local(self, global_idx: int) -> int:
        """
        Convert a global index to its local offset (0-based).
        Raises KeyError if the global index does not belong to this instance.
        """
        if self._mode == "range":
            if self._begin <= global_idx < self._end:
                return global_idx - self._begin
            raise KeyError(
                f"global idx {global_idx} not in range [{self._begin}, {self._end})"
            )
        else:
            try:
                return self._map[global_idx]  # type: ignore[index]
            except KeyError:
                raise KeyError(f"global idx {global_idx} not in list")

    def size(self) -> int:
        """Return number of local elements."""
        if self._mode == "range":
            return self._end - self._begin  # type: ignore[operator]
        else:
            return len(self._list)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return self.size()

    def __contains__(self, global_idx: int) -> bool:
        if self._mode == "range":
            return self._begin <= global_idx < self._end  # type: ignore[operator]
        else:
            return global_idx in self._map  # type: ignore[union-attr]

    def to_global(self, local_offset: int) -> int:
        """
        Reverse lookup: convert a local offset back to its global index.
        Raises IndexError if out of bounds.
        """
        if not (0 <= local_offset < self.size()):
            raise IndexError("local offset out of range")

        if self._mode == "range":
            return self._begin + local_offset  # type: ignore[operator]
        else:
            return self._list[local_offset]  # type: ignore[index]


class KVCacheAccessor:
    """
    Base class for KV cache accessors

    A KV cache accessor locates specific tokens of a specific layer in a KV cache. Data
    can be read from the accessor, and updates to the accessor apply to the KV cache.
    """

    pass


@dataclass
class PagedKVCacheAccessor(KVCacheAccessor):
    block_table: torch.Tensor
    kv: dict[str, torch.Tensor]
    get_page_ids: Optional[Callable[[], torch.Tensor]] = None
    get_offs_in_page: Optional[Callable[[], torch.Tensor]] = None

    @property
    def k(self):  # Legacy interface
        return self.kv["k"]

    @property
    def v(self):  # Legacy interface
        return self.kv["v"]


@dataclass
class DenseKVCacheAccessor(KVCacheAccessor):
    kv: dict[str, torch.Tensor]  # shape: [num_req, max_seqlen + 1, shape_per_token...]

    @property
    def k(self):  # Legacy interface
        return self.kv["k"]

    @property
    def v(self):  # Legacy interface
        return self.kv["v"]


class KVCacheManagerBase:
    def __init__(
        self,
        layer_id_map: GlobalLocalMap,
        *,
        num_hot_req: int,
        max_seq_len: int,
        shape_per_token_dict: Optional[dict[str, torch.Size | Sequence[int]]] = None,
        dtype_dict: Optional[dict[str, torch.dtype]] = None,
        n_local_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        device="cuda",
    ):
        """
        Base class for KV cache managers

        Args:
            layer_id_map: Mapping from global layer id to local layer id.
            num_hot_req: Max batch size.
            max_seq_len: Max sequence length.
            shape_per_token_dict: Shape per token of each KV cache tensor. If not provided, it will be derived
                from `n_local_kv_heads` and `head_dim`.
            dtype_dict: Data type per KV cache tensor. If not provided, it will use `torch.get_default_dtype()`.
            n_local_kv_heads: Number of KV cache heads, used if `shape_per_token_dict` is not provided. This
                is only useful for Llama-like models.
            head_dim: KV cache head dimension, used if `shape_per_token_dict` is not provided. This is only
                useful for Llama-like models.
        """

        self.layer_id_map = layer_id_map
        self.num_layers = layer_id_map.size()

        self.num_hot_req = num_hot_req
        self.max_seq_len = max_seq_len

        self.device = torch.device(device)

        if shape_per_token_dict is None:
            if n_local_kv_heads is None:
                raise ValueError(
                    "`n_local_kv_heads` must be set if `shape_per_token_dict` is None"
                )
            if head_dim is None:
                raise ValueError(
                    "`head_dim` must be set if `shape_per_token_dict` is None"
                )
            shape_per_token_dict = {
                "k": (n_local_kv_heads, head_dim),
                "v": (n_local_kv_heads, head_dim),
            }
        self.shape_per_token_dict = shape_per_token_dict

        if dtype_dict is None:
            dtype_dict = {
                key: torch.get_default_dtype() for key in shape_per_token_dict
            }
        self.dtype_dict = dtype_dict

        self.req_id_to_seq_len: dict[str, int] = {}

        prefill_chunk_size = get_global_args().infer.prefill_chunk_size
        self.max_total_len = num_hot_req * max_seq_len
        self.max_total_delta_len = max(
            (
                prefill_chunk_size
                if prefill_chunk_size is not None
                else num_hot_req * max_seq_len
            ),  # prefill
            num_hot_req,  # decode
        )
        self.seq_len_delta = BatchedSeqLenDelta(
            device=self.device,
            max_batch_size=num_hot_req,
            max_total_len=self.max_total_len,
            max_total_delta_len=self.max_total_delta_len,
            cache_prefix_lens_tensor_device=True,
            cache_position_ids_tensor_device=True,
            cache_seq_ids_tensor_device=True,
            cache_delta_position_ids_tensor_device=True,
            cache_delta_seq_ids_tensor_device=True,
        )

        self.curr_req_ids: Optional[list[str]] = None

        self.timers = get_timers()

    def prepare_cache_prefill(self, req_ids: list[str], delta_seq_len: list[int]):
        self.curr_req_ids = req_ids

        prev_seq_len = BatchedSeqLen(
            [self.req_id_to_seq_len.get(req_id, 0) for req_id in req_ids],
            device=self.device,
            cache_prefix_lens_tensor_device=False,
            cache_position_ids_tensor_device=False,
            cache_seq_ids_tensor_device=False,
        )
        next_seq_len = BatchedSeqLen(
            [
                self.req_id_to_seq_len.get(req_id, 0) + d
                for req_id, d in zip(req_ids, delta_seq_len)
            ],
            device=self.device,
            cache_prefix_lens_tensor_device=False,
            cache_position_ids_tensor_device=False,
            cache_seq_ids_tensor_device=False,
        )
        self.seq_len_delta.copy_from(prev_seq_len, next_seq_len)

        for req_id, seq_len in zip(req_ids, next_seq_len.lens_list):
            self.req_id_to_seq_len[req_id] = seq_len

    def finalize_cache_all_prefill(self):
        self.curr_req_ids = None

    def prepare_cache_decode(self, req_ids: list[str]):
        self.curr_req_ids = req_ids

        self.seq_len_delta.copy_from_list(
            [self.req_id_to_seq_len[req_id] for req_id in req_ids],
            [self.req_id_to_seq_len[req_id] + 1 for req_id in req_ids],
        )

        for req_id in req_ids:
            self.req_id_to_seq_len[req_id] += 1

    def get_block_size(self):
        """Return the number of tokens that a block can accommodate"""
        raise NotImplementedError()

    def get_max_num_blocks(self):
        """Return maximun number of total blocks, which is greater than or equal to self.get_num_blocks()"""
        raise NotImplementedError()

    def get_num_blocks(self):
        """Return number of total blocks"""
        raise NotImplementedError()

    @property
    def num_free_blocks(self):
        """Return number of free blocks"""
        raise NotImplementedError()

    @property
    def num_used_blocks(self):
        """Renturn number of blocks that has reserved for reqs to use."""
        raise NotImplementedError()

    def get_accessor(self, layer_id: int) -> KVCacheAccessor:
        raise NotImplementedError()

    def finalize_cache_single_decode(self, req_ids: list[str]):
        self.curr_req_ids = None

    def finalize_cache_all_decode(self, req_id: str):
        del self.req_id_to_seq_len[req_id]

    def get_gpu_block_table(self):
        return None


class PagedKVCacheManager(KVCacheManagerBase):
    def __init__(
        self,
        layer_id_map: GlobalLocalMap,
        *,
        num_hot_req: int,
        max_seq_len: int,
        shape_per_token_dict: Optional[dict[str, torch.Size | Sequence[int]]] = None,
        dtype_dict: Optional[dict[str, torch.dtype]] = None,
        n_local_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        device="cuda",
        block_size: int = 512,  # must be a multiple of 256 for FlashAttention
        num_blocks: int = -1,
        lazy_mode=False,
    ):
        super().__init__(
            layer_id_map,
            num_hot_req=num_hot_req,
            max_seq_len=max_seq_len,
            shape_per_token_dict=shape_per_token_dict,
            dtype_dict=dtype_dict,
            n_local_kv_heads=n_local_kv_heads,
            head_dim=head_dim,
            device=device,
        )

        self.max_blocks_per_req = ceil_div(max_seq_len, block_size)
        self.max_num_blocks = self.max_blocks_per_req * num_hot_req
        if num_blocks == -1:  # Being warmed-up
            # Should be consistent with `_warmup_via_taskpool` in `chitu_main.py`
            if get_global_args().infer.prefill_chunk_size is not None:
                self.num_blocks = (
                    ceil_div(
                        get_global_args().infer.prefill_chunk_size // num_hot_req + 1,
                        block_size,
                    )
                    * num_hot_req
                )
            else:
                self.num_blocks = num_hot_req
        else:
            self.num_blocks = num_blocks

        self.block_size = block_size

        self.block_table: dict[str, list[int]] = {}  # (seq_id, block_idx)
        self.gpu_block_table = StaticTensor(
            max_nelem=self.max_num_blocks, dtype=torch.int32, device=self.device
        )
        self.free_blocks = deque(range(self.num_blocks))
        self.paged_kv_cache: dict[str, torch.Tensor] = {}
        logger.info(
            f"Allocating KV cache to {self.num_blocks} blocks, each of size {self.block_size}"
        )

        for key in self.shape_per_token_dict:
            self.paged_kv_cache[key] = torch.zeros(
                (self.num_layers, self.num_blocks, block_size)
                + tuple(self.shape_per_token_dict[key]),
                dtype=self.dtype_dict[key],
                device=device,
            )

        self._page_ids_static_tensor = StaticTensor(
            max_nelem=self.max_total_delta_len, device=device, dtype=torch.int32
        )
        self._offs_in_page_static_tensor = StaticTensor(
            max_nelem=self.max_total_delta_len, device=device, dtype=torch.int32
        )
        self._page_ids_up_to_date = False
        self._offs_in_page_up_to_date = False
        self.lazy_mode = lazy_mode

    def get_max_blocks_per_req(self) -> int:
        """Return the maximum number of blocks a single request can occupy."""
        return self.max_blocks_per_req

    def reserve_blocks_for_transfer(self, req_id: str, num_blocks: int) -> list[int]:
        """Reserve a number of free blocks for an incoming transfer on decode side.

        The reserved blocks are removed from the free list immediately to avoid
        collision and are recorded in `block_table[req_id]`.
        """
        reserved: list[int] = []
        num_blocks = int(num_blocks)
        if num_blocks <= 0:
            return reserved
        for _ in range(min(num_blocks, len(self.free_blocks))):
            reserved.append(self.get_free_block())
        # Record the reservation for this request
        if reserved:
            self.block_table[req_id] = list(reserved)
        return reserved

    def realloc(self, num_blocks):
        self.num_blocks = min(num_blocks, self.max_num_blocks)
        logger.info(
            f"Reallocating KV cache to {self.num_blocks} blocks, each of size {self.block_size}"
        )

        self.free_blocks = deque(range(self.num_blocks))
        keys = list(self.paged_kv_cache.keys())
        self.paged_kv_cache.clear()  # Clear first before allocating new tensors, to reduce peak memory usage
        for key in keys:
            self.paged_kv_cache[key] = torch.zeros(
                (self.num_layers, self.num_blocks, self.block_size)
                + tuple(self.shape_per_token_dict[key]),
                device=self.device,
            )

    @override
    def get_block_size(self):
        """Return the number of tokens that a block can accommodate"""
        return self.block_size

    @override
    def get_max_num_blocks(self):
        """Return maximun number of total blocks, which is greater than or equal to self.get_num_blocks()"""
        return self.max_num_blocks

    @override
    def get_num_blocks(self):
        """Return number of total blocks"""
        return self.num_blocks

    @override
    @property
    def num_free_blocks(self):
        """Return number of free blocks"""
        return len(self.free_blocks)

    @override
    @property
    def num_used_blocks(self):
        """Renturn number of blocks that has reserved for reqs to use."""
        return self.num_blocks - len(self.free_blocks)

    @cuda_graph_safe_cached_property("_page_ids_static_tensor", "_page_ids_up_to_date")
    def page_ids(self):
        return self.gpu_block_table.get()[
            self.seq_len_delta.delta_seq_ids_tensor_device,
            self.seq_len_delta.delta_position_ids_tensor_device // self.block_size,
        ]

    @cuda_graph_safe_cached_property(
        "_offs_in_page_static_tensor", "_offs_in_page_up_to_date"
    )
    def offs_in_page(self):
        return self.seq_len_delta.delta_position_ids_tensor_device % self.block_size

    def _upd_gpu_block_table(self, req_ids: list[str]):
        if get_global_args().infer.use_cuda_graph:
            max_block_num = self.max_blocks_per_req
        else:
            max_block_num = max(len(self.block_table[req_id]) for req_id in req_ids)

        all_block_ids = [
            # pad the block ids to max_block_num
            self.block_table[req_id]
            + [0] * (max_block_num - len(self.block_table[req_id]))
            for req_id in req_ids
        ]
        cpu_block_table_tensor = torch.tensor(all_block_ids, dtype=torch.int32)
        self.gpu_block_table.set_shape(cpu_block_table_tensor.shape)
        self.gpu_block_table.get().copy_(cpu_block_table_tensor, non_blocking=True)

        self._page_ids_up_to_date = False
        self._offs_in_page_up_to_date = False

    @override
    def prepare_cache_prefill(self, req_ids: list[str], delta_seq_len: list[int]):
        super().prepare_cache_prefill(req_ids, delta_seq_len)

        for req_id, new_seq_len in zip(req_ids, self.seq_len_delta.new.lens_list):
            if req_id not in self.block_table:
                self.block_table[req_id] = []

            # Allocate blocks for the request
            if self.lazy_mode:
                self.block_table[req_id].append(self.get_free_block())
            else:
                needs_blocks = self.num_additional_blocks_req_need(req_id, new_seq_len)
                self.block_table[req_id].extend(
                    [self.get_free_block() for _ in range(needs_blocks)]
                )
        self._upd_gpu_block_table(req_ids)

    def num_additional_blocks_req_need(self, req_id: str, target_seq_len: int) -> int:
        """Calculates the number of additional blocks needed to store tokens up to the target sequence length.
            Computes the difference between the blocks required for the target sequence length and the blocks
            currently allocated to the request. The result represents how many new blocks need to be allocated
            beyond what the request already has.
        Args:
            req_id: Unique id of the request
            target_seq_len: Desired total sequence length including existing tokens
        Return:
            Number of additional kv_cache blocks required to reach the target sequence length
        """
        if req_id in self.block_table:
            return max(
                0,
                ceil_div(target_seq_len, self.block_size)
                - len(self.block_table[req_id]),
            )
        return max(0, ceil_div(target_seq_len, self.block_size))

    @override
    def prepare_cache_decode(self, req_ids: list[str]):
        # Prepare enough block table for next decoding. When decoding, AttnBackend will fill new kv into
        # paged kv cache in place.
        super().prepare_cache_decode(req_ids)
        if not self.lazy_mode:
            for i, req_id in enumerate(req_ids):
                num_additional_blocks = self.num_additional_blocks_req_need(
                    req_id, self.req_id_to_seq_len[req_id]
                )
                self.block_table[req_id].extend(
                    [self.get_free_block() for _ in range(num_additional_blocks)]
                )
        self._upd_gpu_block_table(req_ids)

    def get_free_block(self):
        self.timers("get_free_block").start()
        if len(self.free_blocks) == 0:
            raise Exception(
                f"No more free blocks: cache manager has total {self.get_num_blocks()} blocks, {self.num_used_blocks} blocks has been used."
            )
        idx = self.free_blocks.popleft()
        self.timers("get_free_block").stop()
        return idx

    @override
    def get_gpu_block_table(self):
        return self.gpu_block_table.get()

    @override
    def get_accessor(self, layer_id: int) -> PagedKVCacheAccessor:
        local_layer_id = self.layer_id_map.to_local(layer_id)
        ret_kv = {
            key: cache[local_layer_id] for key, cache in self.paged_kv_cache.items()
        }
        return PagedKVCacheAccessor(
            self.get_gpu_block_table(),
            ret_kv,
            lambda: self.page_ids,
            lambda: self.offs_in_page,
        )

    def free_req_cache_blocks(self, req_id: str):
        self.timers("free_req_cache_blocks").start()
        for block in self.block_table[req_id]:
            self.free_blocks.append(block)
        del self.block_table[req_id]
        self.timers("free_req_cache_blocks").stop()

    @override
    def finalize_cache_all_decode(self, req_id: str):
        self.timers("finalize_cache_all_decode").start()
        if req_id not in self.req_id_to_seq_len:
            return
        # assert req_id in self.req_id_to_seq_len
        # assert req_id in self.block_table
        self.free_req_cache_blocks(req_id)
        super().finalize_cache_all_decode(req_id)
        self.timers("finalize_cache_all_decode").stop()

    # --- PD disaggregation support ---
    def get_contiguous_buf_infos(self):
        """
        Return contiguous buffer info for RDMA registration.
        For each layer, provide base pointer, total length (bytes), and per-item length (bytes) of one page.
        """
        kv_data_ptrs = []
        kv_data_lens = []
        kv_item_lens = []
        for key in self.paged_kv_cache:
            item_len = (
                int(self.block_size)
                * functools.reduce(
                    lambda x, y: x * y, self.shape_per_token_dict[key], 1
                )
                * self.paged_kv_cache[key].element_size()
            )
            total_len = int(self.num_blocks) * item_len
            for layer in range(self.num_layers):
                layer_ptr = self.paged_kv_cache[key][layer].data_ptr()
                kv_data_ptrs.append(layer_ptr)
                kv_data_lens.append(total_len)
                kv_item_lens.append(item_len)
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_page_indices(self, req_id):
        """Return current allocated page indices for a request, empty if not found."""
        return self.block_table.get(req_id, [])

    def insert_kv_cache_from_transfer(
        self, req_id: str, page_indices: list[int], prefix_length: int
    ):
        """
        Register transferred KV pages into block table and set the sequence length.
        Assumes data has been copied into corresponding pages via RDMA.
        """
        # validate indices are within total blocks
        for idx in page_indices:
            assert 0 <= int(idx) < self.num_blocks, f"invalid page index: {idx}"
        self.block_table[req_id] = list(int(x) for x in page_indices)
        self.req_id_to_seq_len[req_id] = int(prefix_length)


class DenseKVCacheManager(KVCacheManagerBase):
    def __init__(
        self,
        layer_id_map: GlobalLocalMap,
        *,
        num_hot_req: int,
        max_seq_len: int,
        shape_per_token_dict: Optional[dict[str, torch.Size | Sequence[int]]] = None,
        dtype_dict: Optional[dict[str, torch.dtype]] = None,
        n_local_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        device="cuda",
    ):
        super().__init__(
            layer_id_map,
            num_hot_req=num_hot_req,
            max_seq_len=max_seq_len,
            shape_per_token_dict=shape_per_token_dict,
            dtype_dict=dtype_dict,
            n_local_kv_heads=n_local_kv_heads,
            head_dim=head_dim,
            device=device,
        )

        self.slot_availability = [True] * num_hot_req
        self.hot_reqs: list[Optional[str]] = [None] * num_hot_req
        self.req2slot: dict[str, int] = {}

        self.kv_buffer: dict[str, torch.Tensor] = {}
        for key in self.shape_per_token_dict:
            self.kv_buffer[key] = torch.zeros(
                (
                    self.num_layers,
                    self.num_hot_req,
                    self.max_seq_len,
                )
                + tuple(self.shape_per_token_dict[key]),
                dtype=self.dtype_dict[key],
                device=self.device,
            )

        self.prepared_cache: dict[str, torch.Tensor] = {}

        self.slot_handle = get_slot_handle()

    @override
    def get_block_size(self):
        """Return the number of tokens that a block can accommodate"""
        return self.max_seq_len

    @override
    def get_max_num_blocks(self):
        """Return maximun number of total blocks, which is greater than or equal to self.get_num_blocks()"""
        return self.get_num_blocks()

    @override
    def get_num_blocks(self):
        """Return number of total blocks"""
        return self.num_hot_req

    @override
    @property
    def num_free_blocks(self):
        """Return number of free blocks"""
        return sum(1 for is_available in self.slot_availability if is_available == True)

    @override
    @property
    def num_used_blocks(self):
        """Renturn number of blocks that has reserved for reqs to use."""
        return sum(
            1 for is_available in self.slot_availability if is_available == False
        )

    def get_start_and_end_idx(self):
        if self.slot_handle:
            start_idx, end_idx = self.slot_handle.get_current_slot_start_end_idx()
        else:
            start_idx, end_idx = 0, self.num_hot_req
        return start_idx, end_idx

    @override
    def prepare_cache_prefill(self, req_ids: list[str], delta_seq_len: list[int]):
        super().prepare_cache_prefill(req_ids, delta_seq_len)

        # get start_idx and end_idx of current slot_group
        start_idx, end_idx = self.get_start_and_end_idx()

        # Only allocate slots in current slot_group
        slot_id = start_idx
        for it, req_id in enumerate(req_ids):
            if req_id not in self.req2slot:
                allocated = False
                while slot_id < end_idx:
                    if self.slot_availability[slot_id]:
                        self.req2slot[req_id] = slot_id
                        self.slot_availability[slot_id] = False
                        self.hot_reqs[slot_id] = req_id
                        allocated = True
                        slot_id += 1
                        break
                    slot_id += 1
                assert allocated, f"Failed to allocate slot for {req_id}"

        start_pos = self.req2slot[req_ids[0]]
        self._prepare_cache(req_ids, start_pos)

    @override
    def prepare_cache_decode(self, req_ids: list[str]):
        self.timers("cache_prepare").start()
        super().prepare_cache_decode(req_ids)
        start_pos = self.get_start_and_end_idx()[0]
        self._prepare_cache(req_ids, start_pos)
        self.timers("cache_prepare").stop()

    def _prepare_cache(self, req_ids: list[str], start_pos: int):
        assert (
            start_pos + len(req_ids) <= self.num_hot_req
        ), f"start_pos:{start_pos}, number of req:{len(req_ids)}, num_hot_req:{self.num_hot_req}"
        for key in self.kv_buffer:
            self.prepared_cache[key] = self.kv_buffer[key][
                :, start_pos : start_pos + len(req_ids)
            ]

    @override
    def get_accessor(self, layer_id: int) -> DenseKVCacheAccessor:
        local_layer_id = self.layer_id_map.to_local(layer_id)
        ret_kv = {
            key: cache[local_layer_id] for key, cache in self.prepared_cache.items()
        }
        return DenseKVCacheAccessor(ret_kv)

    @override
    def finalize_cache_all_decode(self, req_id: str):
        if req_id not in self.hot_reqs:
            return
        slot_id = self.hot_reqs.index(req_id)
        if slot_id is None:  # not in the hot slot
            return

        if self.slot_handle:
            # get end_idx in req_id slot
            end_idx = 0
            slot_end_idx = self.slot_handle.slot_end_idx
            for idx in slot_end_idx:
                if slot_id < idx:
                    end_idx = idx
                    break
            assert end_idx > slot_id, "get the wrong id in skewkvcache"
            slot_last_id = None
            for idx in range(end_idx - 1, slot_id, -1):
                if not self.slot_availability[idx]:
                    slot_last_id = idx
                    break
        else:
            slot_last_id = next(
                (
                    i
                    for i in range(slot_id + 1, self.num_hot_req)
                    if (
                        not self.slot_availability[i]
                        and (i + 1 >= self.num_hot_req or self.slot_availability[i + 1])
                    )
                ),
                None,
            )

        if slot_last_id is not None:
            for key in self.kv_buffer:
                self.kv_buffer[key][:, slot_id] = self.kv_buffer[key][:, slot_last_id]
            req_key = next(
                (k for k, v in self.req2slot.items() if v == slot_last_id), None
            )
            if req_key is not None:
                self.req2slot[req_key] = slot_id
                self.hot_reqs[slot_id] = req_key
            self.hot_reqs[slot_last_id] = None
            self.slot_availability[slot_last_id] = True
            if req_id in self.req2slot:
                self.req2slot.pop(req_id)
            for key in self.kv_buffer:
                self.kv_buffer[key][:, slot_last_id].zero_()
        else:
            self.hot_reqs[slot_id] = None
            self.slot_availability[slot_id] = True
            self.req2slot.pop(req_id)
            for key in self.kv_buffer:
                self.kv_buffer[key][:, slot_id].zero_()

        super().finalize_cache_all_decode(req_id)
