# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Sequence, Optional
from typing_extensions import override
from dataclasses import dataclass
from logging import getLogger
import torch
from collections import deque

from chitu.global_vars import get_slot_handle, get_timers, get_global_args
from chitu.static_tensor import StaticTensor
from chitu.batched_seq_len import BatchedSeqLen, BatchedSeqLenDelta
from chitu.utils import ceil_div

logger = getLogger(__name__)


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
    k: Optional[torch.Tensor]
    v: Optional[torch.Tensor]


@dataclass
class DenseKVCacheAccessor(KVCacheAccessor):
    k: Optional[torch.Tensor]  # shape: [num_req, max_seqlen + 1, n_kv_heads, head_dim]
    v: Optional[torch.Tensor]  # shape: [num_req, max_seqlen + 1, n_kv_heads, head_dim]


class KVCacheManagerBase:
    def __init__(
        self,
        begin_layer_id,
        end_layer_id,
        *,
        num_hot_req: int,
        max_seq_len: int,
        k_shape_per_sample: Optional[torch.Size | Sequence[int]] = None,
        v_shape_per_sample: Optional[torch.Size | Sequence[int]] = None,
        kv_shape_per_sample: Optional[torch.Size | Sequence[int]] = None,
        n_local_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        device="cuda",
    ):
        """
        Base class for KV cache managers

        Note for KV cache shapes:
        - You can either set `k_shae_per_sample` and `v_shape_per_sample`, or `n_local_kv_heads` and `head_dim`.
        - Otherwise, you can set `kv_shape_per_sample`, which means a holistic shape for both K and V, which
          internally uses only K and disables V.
        """

        self.begin_layer_id = begin_layer_id
        self.end_layer_id = end_layer_id
        self.num_layers = end_layer_id - begin_layer_id

        self.num_hot_req = num_hot_req
        self.max_seq_len = max_seq_len

        self.device = torch.device(device)

        self.k_shape_per_sample: Optional[torch.Size | Sequence[int]]
        self.v_shape_per_sample: Optional[torch.Size | Sequence[int]]
        if kv_shape_per_sample is None:
            if k_shape_per_sample is not None:
                self.k_shape_per_sample = k_shape_per_sample
            else:
                if n_local_kv_heads is None:
                    raise ValueError(
                        "`n_local_kv_heads` must be set if both `kv_shape_per_sample` and `k_shape_per_sample` are None"
                    )
                if head_dim is None:
                    raise ValueError(
                        "`head_dim` must be set if both `kv_shape_per_sample` and `k_shape_per_sample` are None"
                    )
                self.k_shape_per_sample = (n_local_kv_heads, head_dim)
            if v_shape_per_sample is not None:
                self.v_shape_per_sample = v_shape_per_sample
            else:
                if n_local_kv_heads is None:
                    raise ValueError(
                        "`n_local_kv_heads` must be set if both `kv_shape_per_sample` and `k_shape_per_sample` are None"
                    )
                if head_dim is None:
                    raise ValueError(
                        "`head_dim` must be set if both `kv_shape_per_sample` and `k_shape_per_sample` are None"
                    )
                self.v_shape_per_sample = (n_local_kv_heads, head_dim)
        else:
            self.k_shape_per_sample = kv_shape_per_sample
            self.v_shape_per_sample = None

        self.req_id_to_seq_len: Dict[str, int] = {}

        prefill_chunk_size = get_global_args().infer.prefill_chunk_size
        self.seq_len_delta = BatchedSeqLenDelta(
            device=self.device,
            max_batch_size=num_hot_req,
            max_total_len=num_hot_req * max_seq_len,
            max_total_delta_len=(
                prefill_chunk_size
                if prefill_chunk_size is not None
                else num_hot_req * max_seq_len
            ),
            cache_prefix_lens_tensor_device=True,
            cache_position_ids_tensor_device=True,
            cache_seq_ids_tensor_device=True,
            cache_delta_position_ids_tensor_device=True,
            cache_delta_seq_ids_tensor_device=True,
        )

        self.curr_req_ids: Optional[List[str]] = None

        self.timers = get_timers()

    def prepare_cache_prefill(self, req_ids: List[str], delta_seq_len: List[int]):
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

    def is_block_full_for_req(self, req_id):
        """Check if the KV cache blocks for the given request are fully utilized. Called by the scheduler to determine whether a new KV cache block needs to be allocated for the specified request."""
        raise NotImplementedError()

    def prepare_cache_decode(self, req_ids: List[str]):
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

    def finalize_cache_single_decode(self, req_ids: List[str]):
        self.curr_req_ids = None

    def finalize_cache_all_decode(self, req_id: str):
        del self.req_id_to_seq_len[req_id]

    def get_gpu_block_table(self):
        return None


class PagedKVCacheManager(KVCacheManagerBase):
    def __init__(
        self,
        begin_layer_id: int,
        end_layer_id: int,
        *,
        num_hot_req: int,
        max_seq_len: int,
        k_shape_per_sample: Optional[torch.Size | Sequence[int]] = None,
        v_shape_per_sample: Optional[torch.Size | Sequence[int]] = None,
        kv_shape_per_sample: Optional[torch.Size | Sequence[int]] = None,
        n_local_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        device="cuda",
        block_size: int = 512,  # must be a multiple of 256 for FlashAttention
        num_blocks: int = -1,
    ):
        """
        Paged KV cache manager

        Note for KV cache shapes:
        - You can either set `k_shae_per_sample` and `v_shape_per_sample`, or `n_local_kv_heads` and `head_dim`.
        - Otherwise, you can set `kv_shape_per_sample`, which means a holistic shape for both K and V, which
          internally uses only K and disables V.
        """

        super().__init__(
            begin_layer_id,
            end_layer_id,
            num_hot_req=num_hot_req,
            max_seq_len=max_seq_len,
            k_shape_per_sample=k_shape_per_sample,
            v_shape_per_sample=v_shape_per_sample,
            kv_shape_per_sample=kv_shape_per_sample,
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

        self.block_table: Dict[str, List[int]] = {}  # (seq_id, block_idx)
        self.gpu_block_table = StaticTensor(
            max_nelem=self.max_num_blocks, dtype=torch.int32, device=self.device
        )
        self.free_blocks = deque(range(self.num_blocks))
        self.paged_k_cache: Optional[torch.Tensor]
        self.paged_v_cache: Optional[torch.Tensor]
        if self.k_shape_per_sample is not None:
            self.paged_k_cache = torch.zeros(
                (self.num_layers, self.num_blocks, block_size)
                + tuple(self.k_shape_per_sample),
                device=device,
            )
        else:
            self.paged_k_cache = None
        if self.v_shape_per_sample is not None:
            self.paged_v_cache = torch.zeros(
                (self.num_layers, self.num_blocks, block_size)
                + tuple(self.v_shape_per_sample),
                device=device,
            )
        else:
            self.paged_v_cache = None

    def get_max_blocks_per_req(self) -> int:
        """Return the maximum number of blocks a single request can occupy."""
        return self.max_blocks_per_req

    def reserve_blocks_for_transfer(self, req_id: str, num_blocks: int) -> List[int]:
        """Reserve a number of free blocks for an incoming transfer on decode side.

        The reserved blocks are removed from the free list immediately to avoid
        collision and are recorded in `block_table[req_id]`.
        """
        reserved: List[int] = []
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
        has_k_cache = self.paged_k_cache is not None
        has_v_cache = self.paged_v_cache is not None
        if has_k_cache:
            del self.paged_k_cache
            self.paged_k_cache = torch.zeros(
                (self.num_layers, self.num_blocks, self.block_size)
                + self.k_shape_per_sample,
                device=self.device,
            )
        if has_v_cache:
            del self.paged_v_cache
            self.paged_v_cache = torch.zeros(
                (self.num_layers, self.num_blocks, self.block_size)
                + self.v_shape_per_sample,
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

    def _upd_gpu_block_table(self, req_ids: List[str]):
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

    @override
    def prepare_cache_prefill(self, req_ids: List[str], delta_seq_len: List[int]):
        super().prepare_cache_prefill(req_ids, delta_seq_len)

        for req_id, new_seq_len in zip(req_ids, self.seq_len_delta.new.lens_list):
            if req_id not in self.block_table:
                self.block_table[req_id] = []

            # Allocate blocks for the request
            while len(self.block_table[req_id]) * self.block_size < new_seq_len:
                self.block_table[req_id].append(self.get_free_block())

        self._upd_gpu_block_table(req_ids)

    @override
    def is_block_full_for_req(self, req_id):
        """Check if the KV cache blocks for the given request are fully utilized. Called by the scheduler to determine whether a new KV cache block needs to be allocated for the specified request."""
        return self.req_id_to_seq_len[req_id] == self.block_size * len(
            self.block_table[req_id]
        )

    @override
    def prepare_cache_decode(self, req_ids: List[str]):
        # Prepare enough block table for next decoding. When decoding, AttnBackend will fill new kv into
        # paged kv cache in place.
        for i, req_id in enumerate(req_ids):
            # if self.seq_len_delta.old.lens_list[i] % self.block_size == 0:
            if self.is_block_full_for_req(req_id):
                self.block_table[req_id].append(self.get_free_block())

        super().prepare_cache_decode(req_ids)
        self._upd_gpu_block_table(req_ids)

    def get_free_block(self):
        # TODO: When run out of free blocks, use scheduling and preemption in paper instead of exception
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
        ret_k = (
            self.paged_k_cache[layer_id - self.begin_layer_id]
            if self.paged_k_cache is not None
            else None
        )
        ret_v = (
            self.paged_v_cache[layer_id - self.begin_layer_id]
            if self.paged_v_cache is not None
            else None
        )
        return PagedKVCacheAccessor(self.get_gpu_block_table(), ret_k, ret_v)

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
        Note: Use K buffer if available; if V buffer only, use it.
        """
        kv_data_ptrs = []
        kv_data_lens = []
        kv_item_lens = []

        has_k = self.paged_k_cache is not None
        has_v = self.paged_v_cache is not None
        if not has_k and not has_v:
            return [], [], []

        # choose a reference buffer for sizing
        ref_buf = self.paged_k_cache if has_k else self.paged_v_cache
        elem_size = ref_buf.element_size()

        if self.k_shape_per_sample is not None:
            other_dims = 1
            for d in self.k_shape_per_sample:
                other_dims *= int(d)
        else:
            other_dims = 1

        item_len = int(self.block_size) * int(other_dims) * int(elem_size)
        total_len = int(self.num_blocks) * int(item_len)

        for layer in range(self.num_layers):
            layer_ptr = (
                self.paged_k_cache[layer].data_ptr()
                if has_k
                else self.paged_v_cache[layer].data_ptr()
            )
            kv_data_ptrs.append(layer_ptr)
            kv_data_lens.append(total_len)
            kv_item_lens.append(item_len)

        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_page_indices(self, req_id):
        """Return current allocated page indices for a request, empty if not found."""
        return self.block_table.get(req_id, [])

    def insert_kv_cache_from_transfer(
        self, req_id: str, page_indices: List[int], prefix_length: int
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
        begin_layer_id,
        end_layer_id,
        *,
        num_hot_req: int,
        max_seq_len: int,
        k_shape_per_sample: Optional[torch.Size | Sequence[int]] = None,
        v_shape_per_sample: Optional[torch.Size | Sequence[int]] = None,
        kv_shape_per_sample: Optional[torch.Size | Sequence[int]] = None,
        n_local_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        device="cuda",
    ):
        """
        Non-paged KV cache manager

        Note for KV cache shapes:
        - You can either set `k_shae_per_sample` and `v_shape_per_sample`, or `n_local_kv_heads` and `head_dim`.
        - Otherwise, you can set `kv_shape_per_sample`, which means a holistic shape for both K and V, which
          internally uses only K and disables V.
        """

        super().__init__(
            begin_layer_id,
            end_layer_id,
            num_hot_req=num_hot_req,
            max_seq_len=max_seq_len,
            k_shape_per_sample=k_shape_per_sample,
            v_shape_per_sample=v_shape_per_sample,
            kv_shape_per_sample=kv_shape_per_sample,
            n_local_kv_heads=n_local_kv_heads,
            head_dim=head_dim,
            device=device,
        )

        self.slot_availability = [True] * num_hot_req
        self.hot_reqs: List[Optional[str]] = [None] * num_hot_req
        self.req2slot: Dict[str, int] = {}

        self.k_buffer: Optional[torch.Tensor] = None
        self.v_buffer: Optional[torch.Tensor] = None
        if self.k_shape_per_sample is not None:
            self.k_buffer = torch.zeros(
                (
                    self.num_layers,
                    self.num_hot_req,
                    self.max_seq_len,
                )
                + tuple(self.k_shape_per_sample),
                device=self.device,
            )
        if self.v_shape_per_sample is not None:
            self.v_buffer = torch.zeros(
                (
                    self.num_layers,
                    self.num_hot_req,
                    self.max_seq_len,
                )
                + tuple(self.v_shape_per_sample),
                device=self.device,
            )

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
    def prepare_cache_prefill(self, req_ids: List[str], delta_seq_len: List[int]):
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
    def is_block_full_for_req(self, req_id):
        """Check if the KV cache blocks for the given request are fully utilized. Called by the scheduler to determine whether a new KV cache block needs to be allocated for the specified request."""
        return False

    @override
    def prepare_cache_decode(self, req_ids: List[str]):
        self.timers("cache_prepare").start()
        super().prepare_cache_decode(req_ids)
        start_pos = self.get_start_and_end_idx()[0]
        self._prepare_cache(req_ids, start_pos)
        self.timers("cache_prepare").stop()

    def _prepare_cache(self, req_ids: List[str], start_pos: int):
        assert (
            start_pos + len(req_ids) <= self.num_hot_req
        ), f"start_pos:{start_pos}, number of req:{len(req_ids)}, num_hot_req:{self.num_hot_req}"

        self.k_prepared_cache = (
            None
            if self.k_buffer is None
            else self.k_buffer[:, start_pos : start_pos + len(req_ids)]
        )
        self.v_prepared_cache = (
            None
            if self.v_buffer is None
            else self.v_buffer[:, start_pos : start_pos + len(req_ids)]
        )

    @override
    def get_accessor(self, layer_id: int) -> DenseKVCacheAccessor:
        ret_k = (
            self.k_prepared_cache[layer_id - self.begin_layer_id]
            if self.k_prepared_cache is not None
            else None
        )
        ret_v = (
            self.v_prepared_cache[layer_id - self.begin_layer_id]
            if self.v_prepared_cache is not None
            else None
        )
        return DenseKVCacheAccessor(ret_k, ret_v)

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
            if self.k_buffer is not None:
                self.k_buffer[:, slot_id] = self.k_buffer[:, slot_last_id]
            if self.v_buffer is not None:
                self.v_buffer[:, slot_id] = self.v_buffer[:, slot_last_id]
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
            if self.k_buffer is not None:
                self.k_buffer[:, slot_last_id].zero_()
            if self.v_buffer is not None:
                self.v_buffer[:, slot_last_id].zero_()
        else:
            self.hot_reqs[slot_id] = None
            self.slot_availability[slot_id] = True
            self.req2slot.pop(req_id)
            if self.k_buffer is not None:
                self.k_buffer[:, slot_id].zero_()
            if self.v_buffer is not None:
                self.v_buffer[:, slot_id].zero_()

        super().finalize_cache_all_decode(req_id)
