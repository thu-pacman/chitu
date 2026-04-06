# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Sequence, Optional, Callable, Iterable, TYPE_CHECKING
from typing_extensions import override
from dataclasses import dataclass
from logging import getLogger
import torch
import functools
from enum import Enum
from collections import deque, defaultdict

from chitu.cuda_graph import cuda_graph_safe_cached_property
from chitu.global_vars import get_slot_handle, get_global_args
from chitu.static_tensor import StaticTensor
from chitu.batched_seq_len import BatchedSeqLen, BatchedSeqLenDelta
from chitu.utils import ceil_div
from chitu.ops import fp8_pertensor_kvcache_quant, fp8_pertoken_kvcache_quant_dsa

if TYPE_CHECKING:
    from chitu.task import PackedTasksBase


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
    use_i64_offsets: bool = False

    @property
    def k(self):  # Legacy interface
        return self.kv["k"]

    @property
    def v(self):  # Legacy interface
        return self.kv["v"]


@dataclass
class DenseKVCacheAccessor(KVCacheAccessor):
    kv: dict[str, torch.Tensor]  # shape: [num_req, max_seqlen + 1, shape_per_token...]
    use_i64_offsets: bool = False

    @property
    def k(self):  # Legacy interface
        return self.kv["k"]

    @property
    def v(self):  # Legacy interface
        return self.kv["v"]


class KVCacheQuantType(Enum):
    # Add KV Cache quant here
    NONE = "None"
    FP8_PERTENSOR = "fp8_pertensor"
    FP8_PERTOKEN_DSA = "fp8_pertoken_dsa"

    @property
    def needs_kv_scales(self) -> bool:
        return self in {KVCacheQuantType.FP8_PERTENSOR}


class KVCacheBase:
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
        quant_type: str = None,
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

        self.quant_type = KVCacheQuantType(str(quant_type))

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

        self.tid_to_cached_len: dict[str, int] = (
            {}
        )  # Map from task_id to cached kvcache length of the task
        self.curr_tids: Optional[list[str]] = None  # current task ids in model run.

        prefill_chunk_size_global = get_global_args().infer.prefill_chunk_size
        prefill_chunk_size_per_dp = (
            ceil_div(prefill_chunk_size_global, get_global_args().infer.dp_size)
            if prefill_chunk_size_global is not None
            else None
        )
        _mtp_size = get_global_args().infer.mtp_size
        _mtp_extra = _mtp_size if _mtp_size > 1 else 0
        self.max_total_len = num_hot_req * (max_seq_len + _mtp_extra)
        self.max_total_delta_len = max(
            (
                prefill_chunk_size_per_dp
                if prefill_chunk_size_per_dp is not None
                else num_hot_req * max_seq_len
            ),  # prefill
            num_hot_req * _mtp_size,  # decode
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

        self.mtp_size = get_global_args().infer.mtp_size
        if self.mtp_size > 1:
            self.mtp_seq_len_delta = BatchedSeqLenDelta(
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

    @property
    def is_quant_kv(self):
        return self.quant_type is not KVCacheQuantType.NONE

    def kvcache_quant(  # should support various kv quantization patterns
        self,
        q: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        n_local_kv_heads: Optional[int] = None,
        kv_lora_rank: Optional[int] = None,
    ):
        # NOTE: Currently KVCacheQuantType.NONE (no quantization) is not handled here,
        # because this function is not versatile enough to handle different attention
        # patterns.
        assert self.quant_type is not KVCacheQuantType.NONE
        if self.quant_type is KVCacheQuantType.FP8_PERTENSOR:
            assert q_scale is None and n_local_kv_heads is not None
            return fp8_pertensor_kvcache_quant(
                q,
                k,
                v,
                k_scale,
                v_scale,
                self.seq_len_delta.batch_size,
                n_local_kv_heads,
            )
        elif self.quant_type is KVCacheQuantType.FP8_PERTOKEN_DSA:
            assert k is not None and kv_lora_rank is not None
            return fp8_pertoken_kvcache_quant_dsa(k, kv_lora_rank)
        else:
            raise NotImplementedError(
                f"Unsupported kv_cache quant type: {self.quant_type}"
            )

    def estimate_bytes_per_block(self) -> int:
        raise NotImplementedError()

    def prepare_cache_prefill(self, tasks: "PackedTasksBase"):
        if tasks.hit_token_lens:
            cached_token_lens: list[int] = [
                self.tid_to_cached_len.get(tid, 0) + tasks.hit_token_lens[i]
                for i, tid in enumerate(tasks.task_ids)
            ]
        else:
            cached_token_lens: list[int] = [
                self.tid_to_cached_len.get(tid, 0) for tid in tasks.task_ids
            ]
        delta_seq_len: list[int] = [len(t) for t in tasks.tokens]
        self.curr_tids = tasks.task_ids

        prev_seq_len = BatchedSeqLen(
            cached_token_lens,
            device=self.device,
            cache_prefix_lens_tensor_device=False,
            cache_position_ids_tensor_device=False,
            cache_seq_ids_tensor_device=False,
        )
        next_seq_len = BatchedSeqLen(
            [cached + delta for cached, delta in zip(cached_token_lens, delta_seq_len)],
            device=self.device,
            cache_prefix_lens_tensor_device=False,
            cache_position_ids_tensor_device=False,
            cache_seq_ids_tensor_device=False,
        )
        self.seq_len_delta.copy_from(prev_seq_len, next_seq_len)
        if self.mtp_size > 1:
            self.mtp_seq_len_delta.copy_from(prev_seq_len, next_seq_len)

        for tid, seq_len in zip(tasks.task_ids, next_seq_len.lens_list):
            self.tid_to_cached_len[tid] = seq_len

    def prepare_cache_decode(self, tasks: "PackedTasksBase"):
        cached_token_lens: list[int] = [
            self.tid_to_cached_len.get(tid, 0) for tid in tasks.task_ids
        ]
        self.curr_tids = tasks.task_ids

        self.seq_len_delta.copy_from_list(
            cached_token_lens,
            [cached + self.mtp_size for cached in cached_token_lens],
        )
        for tid in tasks.task_ids:
            self.tid_to_cached_len[tid] += self.mtp_size

    def update_mtp_cache_decode(self, mtp_offset: list[int]):
        task_ids = self.curr_tids
        for tid, off_set in zip(task_ids, mtp_offset):
            self.tid_to_cached_len[tid] += off_set - self.mtp_size

    def prepare_mtp_cache_decode(self, mtp_offset: int):
        task_ids = self.curr_tids
        self.mtp_seq_len_delta.copy_from_list(
            [
                self.tid_to_cached_len[tid] - self.mtp_size + mtp_offset
                for tid in task_ids
            ],
            [
                self.tid_to_cached_len[tid] - self.mtp_size + mtp_offset + 1
                for tid in task_ids
            ],
        )

    def finalize_cache_all_decode(self, tasks: "PackedTasksBase"):
        for tid in tasks.task_ids:
            if tid in self.tid_to_cached_len:
                self.tid_to_cached_len.pop(tid)

    def get_gpu_block_table(self):
        return None


class PagedKVCache(KVCacheBase):
    def __init__(
        self,
        layer_id_map: GlobalLocalMap,
        *,
        num_hot_req: int,
        max_seq_len: int,
        num_blocks: int,
        shape_per_token_dict: Optional[dict[str, torch.Size | Sequence[int]]] = None,
        dtype_dict: Optional[dict[str, torch.dtype]] = None,
        n_local_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        quant_type: str = None,
        device="cuda",
        block_size: int = 512,  # must be a multiple of 256 for FlashAttention
        is_singleton: bool = False,
    ):
        super().__init__(
            layer_id_map,
            num_hot_req=num_hot_req,
            max_seq_len=max_seq_len,
            shape_per_token_dict=shape_per_token_dict,
            dtype_dict=dtype_dict,
            n_local_kv_heads=n_local_kv_heads,
            head_dim=head_dim,
            quant_type=quant_type,
            device=device,
        )
        mtp_extra = self.mtp_size if (self.mtp_size > 1 and not is_singleton) else 0
        self.max_blocks_per_req = ceil_div(max_seq_len + mtp_extra, block_size)
        self.page_table_max_num_blocks = self.max_blocks_per_req * num_hot_req
        self.max_num_blocks = self.page_table_max_num_blocks

        if get_global_args().infer.enable_prefix_caching:
            self.allocatable_max_num_blocks = 1 << 60
        else:
            self.allocatable_max_num_blocks = self.page_table_max_num_blocks

        self.num_blocks = num_blocks
        self.block_size = block_size

        self.block_table: dict[str, list[int]] = defaultdict(
            list
        )  # {seq_id: block_ids}
        self.gpu_block_table = StaticTensor(
            max_nelem=self.max_num_blocks, dtype=torch.int32, device=self.device
        )
        self.paged_kv_cache: dict[str, torch.Tensor] = {}
        logger.info(
            f"Allocating KV cache of {','.join(self.shape_per_token_dict.keys())} with "
            f"{self.num_blocks} blocks, each of size {self.block_size}"
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
        self.use_i64_offsets = self.needs_i64_kv_offsets()

    def get_allocatable_max_num_blocks(self) -> int:
        return int(getattr(self, "allocatable_max_num_blocks", self.max_num_blocks))

    def realloc(self, num_blocks):
        requested_num_blocks = int(num_blocks)
        allocatable_cap = self.get_allocatable_max_num_blocks()

        logger.info(
            f"Requested realloc to {requested_num_blocks} KV blocks. "
            f"page_table_max_num_blocks={self.page_table_max_num_blocks}, "
            f"allocatable_max_num_blocks={allocatable_cap}, "
            f"infer.max_reqs={get_global_args().infer.max_reqs}, "
            f"infer.max_seq_len={get_global_args().infer.max_seq_len}, "
            f"prefix_caching={get_global_args().infer.enable_prefix_caching}"
        )

        if requested_num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {requested_num_blocks}")

        self.num_blocks = min(requested_num_blocks, allocatable_cap)

        logger.info(
            f"Reallocating KV cache to {self.num_blocks} blocks, "
            f"each of size {self.block_size}"
        )

        keys = list(self.paged_kv_cache.keys())
        self.paged_kv_cache.clear()
        for key in keys:
            self.paged_kv_cache[key] = torch.zeros(
                (self.num_layers, self.num_blocks, self.block_size)
                + tuple(self.shape_per_token_dict[key]),
                dtype=self.dtype_dict[key],
                device=self.device,
            )
        self.use_i64_offsets = self.needs_i64_kv_offsets()

    def needs_i64_kv_offsets(self) -> bool:
        for _, t in self.paged_kv_cache.items():
            kv = t[0].view(
                t.shape[1], t.shape[2], -1
            )  # (num_blocks, block_size, other_dims_flat)
            max_kv_off = (
                (kv.shape[0] - 1) * kv.stride(0)
                + (kv.shape[1] - 1) * kv.stride(1)
                + (kv.shape[2] - 1) * kv.stride(2)
            )
            if max_kv_off > (1 << 31) - 1:
                return True
        return False

    @cuda_graph_safe_cached_property("_page_ids_static_tensor", "_page_ids_up_to_date")
    def page_ids(self):
        return self.gpu_block_table.get()[
            self.seq_len_delta.delta_seq_ids_tensor_device,
            self.seq_len_delta.delta_position_ids_tensor_device // self.block_size,
        ]

    @cuda_graph_safe_cached_property("_page_ids_static_tensor", "_page_ids_up_to_date")
    def page_ids_mtp(self):
        return self.gpu_block_table.get()[
            self.mtp_seq_len_delta.delta_seq_ids_tensor_device,
            self.mtp_seq_len_delta.delta_position_ids_tensor_device // self.block_size,
        ]

    @cuda_graph_safe_cached_property(
        "_offs_in_page_static_tensor", "_offs_in_page_up_to_date"
    )
    def offs_in_page(self):
        return self.seq_len_delta.delta_position_ids_tensor_device % self.block_size

    @cuda_graph_safe_cached_property(
        "_offs_in_page_static_tensor", "_offs_in_page_up_to_date"
    )
    def offs_in_page_mtp(self):
        return self.mtp_seq_len_delta.delta_position_ids_tensor_device % self.block_size

    def _upd_gpu_block_table(self, task_ids: list[str]):
        block_lists = [list(self.block_table[tid]) for tid in task_ids]
        max_len = max(len(blocks) for blocks in block_lists)
        if get_global_args().infer.use_cuda_graph:
            if max_len > self.max_blocks_per_req:
                logger.warning(
                    "block_table length exceeds per-request page-table limit; "
                    "decode max_seq_len may be too small. "
                    f"max_len={max_len} max_blocks_per_req={self.max_blocks_per_req} "
                    f"max_seq_len={get_global_args().infer.max_seq_len} block_size={self.block_size}"
                )
            max_block_num = max(self.max_blocks_per_req, max_len)
        else:
            max_block_num = max_len

        all_block_ids = [
            # pad the block ids to max_block_num
            blocks + [0] * (max_block_num - len(blocks))
            for blocks in block_lists
        ]
        cpu_block_table_tensor = torch.tensor(all_block_ids, dtype=torch.int32)
        self.gpu_block_table.set_shape(cpu_block_table_tensor.shape)
        self.gpu_block_table.get().copy_(cpu_block_table_tensor, non_blocking=True)

        self.update_page_offs()
        # self._page_ids_up_to_date = False
        # self._offs_in_page_up_to_date = False

    def update_page_offs(self):
        self._page_ids_up_to_date = False
        self._offs_in_page_up_to_date = False

    @override
    def prepare_cache_prefill(self, tasks: "PackedTasksBase"):
        super().prepare_cache_prefill(tasks)
        if tasks.new_cache_ids_list:
            for tid, new_cache_ids in zip(tasks.task_ids, tasks.new_cache_ids_list):
                self.block_table[tid].extend(new_cache_ids)
        self._upd_gpu_block_table(tasks.task_ids)

    @override
    def prepare_cache_decode(self, tasks: "PackedTasksBase"):
        # Prepare enough block table for next decoding. When decoding, AttnBackend will fill new kv into
        # paged kv cache in place.
        super().prepare_cache_decode(tasks)
        if tasks.new_cache_ids_list:
            for tid, new_cache_ids in zip(tasks.task_ids, tasks.new_cache_ids_list):
                self.block_table[tid].extend(new_cache_ids)
        self._upd_gpu_block_table(tasks.task_ids)

    def estimate_bytes_per_block(self) -> int:
        """
        Estimate additional bytes required to allocate 1 more KV page/block.

        For paged KV cache, each key is stored as a tensor shaped:
            (num_layers, num_blocks, block_size, *shape_per_token)

        Increasing num_blocks by 1 adds:
            num_layers * block_size * prod(shape_per_token) * element_size(dtype)
        bytes for that key.
        """
        bs = int(self.block_size)
        n_layers = int(self.num_layers)

        total = 0
        for key, shape in self.shape_per_token_dict.items():
            n_elem_per_token = 1
            for d in tuple(shape):
                n_elem_per_token *= int(d)

            dtype = self.dtype_dict[key]
            elem_size = torch.empty((), dtype=dtype).element_size()

            bytes_per_layer_per_block = bs * n_elem_per_token * elem_size
            total += n_layers * bytes_per_layer_per_block

        return int(total)

    def finalize_cache_all_decode(self, tasks: "PackedTasksBase"):
        super().finalize_cache_all_decode(tasks)
        for tid in tasks.task_ids:
            if tid in self.block_table:
                self.block_table.pop(tid)

    @override
    def get_accessor(self, layer_id: int, is_mtp: bool = False) -> PagedKVCacheAccessor:
        local_layer_id = self.layer_id_map.to_local(layer_id)
        ret_kv = {
            key: cache[local_layer_id] for key, cache in self.paged_kv_cache.items()
        }

        if is_mtp:
            return PagedKVCacheAccessor(
                self.get_gpu_block_table(),
                ret_kv,
                lambda: self.page_ids_mtp,
                lambda: self.offs_in_page_mtp,
                self.use_i64_offsets,
            )

        else:
            return PagedKVCacheAccessor(
                self.get_gpu_block_table(),
                ret_kv,
                lambda: self.page_ids,
                lambda: self.offs_in_page,
                self.use_i64_offsets,
            )

    @override
    def get_gpu_block_table(self):
        return self.gpu_block_table.get()

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
            logger.info(f"Getting contiguous buffer info for key: {key}")
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
        self, tid: str, page_indices: list[int], prefix_length: int
    ):
        """
        Register transferred KV pages into block table and set the sequence length.
        Assumes data has been copied into corresponding pages via RDMA.
        """
        # validate indices are within total blocks
        for idx in page_indices:
            assert 0 <= int(idx) < self.num_blocks, f"invalid page index: {idx}"
        assert (
            len(self.block_table[tid]) == 0
        ), f"tid:{tid}, self.block_table[tid]:{self.block_table[tid]}"
        self.block_table[tid] = list(int(x) for x in page_indices)
        self.tid_to_cached_len[tid] = int(prefix_length)


class SingletonPagedKVCache(PagedKVCache):
    """
    1-token-per-request specialization of PagedKVCache

    For linear attention or RNN states, instead of transformer states.
    """

    def __init__(
        self,
        layer_id_map: GlobalLocalMap,
        *,
        num_hot_req: int,
        shape_per_token_dict: Optional[dict[str, torch.Size | Sequence[int]]] = None,
        dtype_dict: Optional[dict[str, torch.dtype]] = None,
        n_local_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        device="cuda",
    ):
        self.mtp_size = get_global_args().infer.mtp_size
        super().__init__(
            layer_id_map,
            num_hot_req=num_hot_req,
            max_seq_len=self.mtp_size,
            shape_per_token_dict=shape_per_token_dict,
            dtype_dict=dtype_dict,
            n_local_kv_heads=n_local_kv_heads,
            head_dim=head_dim,
            device=device,
            block_size=self.mtp_size,
            num_blocks=num_hot_req,
            is_singleton=True,
        )
        self.max_blocks_per_req = 1
        self.max_num_blocks = self.max_blocks_per_req * num_hot_req
        self.free_blocks = deque(range(self.num_blocks))

        if self.mtp_size > 1:
            self.tid_to_mtp_offset: dict[str, int] = {}
            self.mtp_offset = []
            self.is_mtp_decode_stage = False
            self.mtp_offset_tensor = StaticTensor(
                max_nelem=self.num_hot_req, device=device, dtype=torch.int32
            )

    def realloc(self, num_blocks):
        super().realloc(num_blocks)
        self.free_blocks = deque(range(self.num_blocks))

    @property
    def num_free_blocks(self):
        """Return number of free blocks"""
        return len(self.free_blocks)

    @override
    def update_mtp_cache_decode(self, mtp_offset: list[int]):
        super().update_mtp_cache_decode(mtp_offset)

        if self.mtp_size > 1:
            task_ids = self.curr_tids
            for task_id, off_set in zip(task_ids, mtp_offset):
                self.tid_to_mtp_offset[task_id] = off_set - 1

    @override
    def prepare_cache_prefill(self, tasks: "PackedTasksBase"):
        KVCacheBase.prepare_cache_prefill(self, tasks)

        for task_id in tasks.task_ids:
            if task_id not in self.block_table:
                self.block_table[task_id] = [self.get_free_block()]
        self._upd_gpu_block_table(tasks.task_ids)

        if self.mtp_size > 1:
            self.mtp_offset.clear()
            for task_id in tasks.task_ids:
                self.tid_to_mtp_offset[task_id] = -1
                self.mtp_offset.append(-1)
            self.is_mtp_decode_stage = False
            self.mtp_offset_tensor.set(
                torch.tensor(self.mtp_offset, dtype=torch.int32, device=self.device)
            )

    @override
    def prepare_cache_decode(self, tasks: "PackedTasksBase"):
        KVCacheBase.prepare_cache_decode(self, tasks)
        self._upd_gpu_block_table(tasks.task_ids)

        if self.mtp_size > 1:
            self.mtp_offset = list(self.tid_to_mtp_offset.values())
            self.is_mtp_decode_stage = True
            self.mtp_offset_tensor.set(
                torch.tensor(self.mtp_offset, dtype=torch.int32, device=self.device)
            )

    @override
    def finalize_cache_all_decode(self, tasks: "PackedTasksBase"):
        KVCacheBase.finalize_cache_all_decode(self, tasks)
        for tid in tasks.task_ids:
            if tid not in self.block_table:
                continue
            self.free_req_cache_blocks(tid)

        if self.mtp_size > 1:
            for tid in tasks.task_ids:
                if tid in self.tid_to_mtp_offset:
                    self.tid_to_mtp_offset.pop(tid)

    def free_req_cache_blocks(self, tid: str):
        for block in self.block_table[tid]:
            self.free_blocks.append(block)
        del self.block_table[tid]

    @override
    def get_free_block(self):
        if len(self.free_blocks) == 0:
            raise Exception(
                f"No more free blocks: cache manager has total {self.num_blocks} blocks, {self.num_blocks - len(self.free_blocks)} blocks has been used."
            )
        idx = self.free_blocks.popleft()
        for key in self.paged_kv_cache:
            self.paged_kv_cache[key][:, idx] = 0
        return idx

    @override
    @cuda_graph_safe_cached_property("_page_ids_static_tensor", "_page_ids_up_to_date")
    def page_ids(self):
        return self.gpu_block_table.get().squeeze(1)

    @override
    @cuda_graph_safe_cached_property("_page_ids_static_tensor", "_page_ids_up_to_date")
    def page_ids_mtp(self):
        return self.gpu_block_table.get().squeeze(1)

    @override
    @cuda_graph_safe_cached_property(
        "_offs_in_page_static_tensor", "_offs_in_page_up_to_date"
    )
    def offs_in_page(self):
        return torch.zeros_like(self.seq_len_delta.delta_lens_tensor_device)

    @override
    @cuda_graph_safe_cached_property(
        "_offs_in_page_static_tensor", "_offs_in_page_up_to_date"
    )
    def offs_in_page_mtp(self):
        return torch.zeros_like(self.mtp_seq_len_delta.delta_lens_tensor_device)

    # NOTE: get_contiguous_buf_infos is inherited from PagedKVCache.
    # The implementation is generic and works correctly for singleton cache
    # (block_size=1, num_blocks=num_hot_req).

    def reserve_blocks_for_transfer(self, tid: str, num_blocks: int) -> list[int]:
        """Reserve a number of free blocks for an incoming transfer on decode side.
        The reserved blocks are removed from the free list immediately to avoid
        collision and are recorded in `block_table[req_id]`.
        SingletonPagedKVCache没有对应的SingletonPagedKVCacheManager，因此需要自行分配和管理kv cache block索引
        """
        reserved: list[int] = []
        num_blocks = int(num_blocks)
        if num_blocks <= 0:
            return reserved

        # NOTE:
        # PD disaggregation relies on destination block_table having enough blocks
        # to cover prefix_length. Partially reserving blocks will trigger
        # CUDA device-side asserts when indexing page table by position_ids//block_size.
        if num_blocks > len(self.free_blocks):
            raise RuntimeError(
                f"Not enough free KV blocks for transfer: req_id={tid} "
                f"need={num_blocks} free={len(self.free_blocks)} "
                f"total={self.num_blocks} used={self.num_blocks - len(self.free_blocks)}"
            )

        for _ in range(num_blocks):
            reserved.append(self.get_free_block())

        self.block_table[tid] = list(reserved)
        return reserved

    def insert_linear_state_from_transfer(
        self, tid: str, page_index: int, prefix_length: int
    ):
        """
        Register transferred linear attention state into block table.
        For linear attention, each request uses exactly one block.
        """
        assert (
            0 <= int(page_index) < self.num_blocks
        ), f"invalid page index: {page_index}"
        self.block_table[tid] = [int(page_index)]
        self.tid_to_cached_len[tid] = prefix_length


class MMPagedKVCache(PagedKVCache):
    """Paged KV cache  with multimodal chunk-consumption helpers.

    This class is based on `PagedKVCache` and adds the extra APIs for
    Qwen3-VL & Qwen-3.5 multimodal cache flow.

    The seq_multimodal_len_delta tracks the consumption progress of multimodal tokens in a batched manner.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.free_blocks = deque(range(self.num_blocks))
        self.req_len: dict[str, dict[str, dict[str, int]]] = {}
        self.request_metadata: dict[str, dict[str, Any]] = {}
        self.tid_to_multimodal_len: dict[str, int] = {}
        self.seq_multimodal_len_delta = BatchedSeqLenDelta(
            device=self.device,
            max_batch_size=self.num_hot_req,
            max_total_len=self.max_total_len,
            max_total_delta_len=self.max_total_delta_len,
            cache_prefix_lens_tensor_device=True,
            cache_position_ids_tensor_device=True,
            cache_seq_ids_tensor_device=True,
            cache_delta_position_ids_tensor_device=True,
            cache_delta_seq_ids_tensor_device=True,
        )

    def realloc(self, num_blocks):
        super().realloc(num_blocks)
        self.free_blocks = deque(range(self.num_blocks))

    def register_tensor_for_consumption(
        self,
        tid: str,
        tensor_key: str,
        total_tokens: int,
    ) -> None:
        if tid not in self.req_len:
            self.req_len[tid] = {}
        self.req_len[tid][tensor_key] = {
            "total": int(total_tokens),
        }

    def prepare_multimodal_cache_prefill(
        self, task_ids: list[str], delta_seq_len: list[int]
    ):
        self.curr_tids = task_ids
        tensor_key = "vision_embeds"

        prev_seq_len = BatchedSeqLen(
            [self.tid_to_multimodal_len.get(tid, 0) for tid in task_ids],
            device=self.device,
            cache_prefix_lens_tensor_device=False,
            cache_position_ids_tensor_device=False,
            cache_seq_ids_tensor_device=False,
        )
        next_seq_len = BatchedSeqLen(
            [
                min(
                    self.tid_to_multimodal_len.get(tid, 0) + d,
                    self.req_len.get(tid, {}).get(tensor_key, {}).get("total", 0),
                )
                for tid, d in zip(task_ids, delta_seq_len)
            ],
            device=self.device,
            cache_prefix_lens_tensor_device=False,
            cache_position_ids_tensor_device=False,
            cache_seq_ids_tensor_device=False,
        )
        self.seq_multimodal_len_delta.copy_from(prev_seq_len, next_seq_len)

        for tid, seq_len in zip(task_ids, next_seq_len.lens_list):
            self.tid_to_multimodal_len[tid] = seq_len

    def get_consumption_progress(self, tid: str, tensor_key: str) -> int:
        info = self.req_len.get(tid, {}).get(tensor_key)
        if info is None:
            return 0
        return int(info["total"])

    def prepare_cache_for_pre_layers_prefill(
        self,
        task_ids: list[str],
        chunk_sizes: list[int],
    ):
        """Prepare for tracking the consumption progress of multimodal tokens."""
        self.prepare_multimodal_cache_prefill(task_ids, chunk_sizes)
        self._upd_gpu_block_table(task_ids)

    def batched_consume_next_chunk(
        self,
        task_ids: list[str],
        tensor_keys: list[str],
        auto_free: bool = False,
    ) -> tuple[list[list[torch.Tensor]], list[bool]]:

        delta_pos = self.seq_multimodal_len_delta.delta_position_ids_tensor_device
        delta_seq = self.seq_multimodal_len_delta.delta_seq_ids_tensor_device
        gpu_block_table = self.gpu_block_table.get()

        local_layer_id = self.layer_id_map.to_local(0)
        kv_data_list = []
        for tensor_key in tensor_keys:
            kv_data = self.paged_kv_cache[tensor_key][local_layer_id]
            kv_data_list.append(kv_data)

        delta_lens = self.seq_multimodal_len_delta._delta.lens_list

        if delta_pos.numel() > 0:
            block_indices = delta_pos // self.block_size
            offset_indices = delta_pos % self.block_size

            page_ids = gpu_block_table[delta_seq.long(), block_indices.long()]
            flat_indices = page_ids.long() * self.block_size + offset_indices.long()

            gathered_list = [
                kv_data.view(-1, *kv_data.shape[2:])[flat_indices]
                for kv_data in kv_data_list
            ]
            results = [torch.split(gathered, delta_lens) for gathered in gathered_list]
        else:
            results = []
            for tensor_key in tensor_keys:
                per_token_shape = tuple(self.shape_per_token_dict[tensor_key])
                dtype = self.dtype_dict[tensor_key]
                results.append(
                    [
                        torch.empty(
                            0, *per_token_shape, dtype=dtype, device=self.device
                        )
                        for _ in task_ids
                    ]
                )

        complete_flags = []
        for i, tid in enumerate(task_ids):
            req_info = self.req_len.get(tid, {}).get(tensor_keys[0], {})
            total = req_info.get("total", 0)
            consumed = self.tid_to_multimodal_len[tid]

            is_complete = consumed >= total
            complete_flags.append(is_complete)

            if is_complete and auto_free:
                for tensor_key in tensor_keys:
                    self.req_len.get(tid, {}).pop(tensor_key, None)

        return results, complete_flags

    def get_free_block(self):
        if len(self.free_blocks) == 0:
            raise Exception(
                f"No more free blocks: cache manager has total {self.num_blocks} blocks, all blocks has been used."
            )
        idx = self.free_blocks.popleft()
        for key in self.paged_kv_cache:
            self.paged_kv_cache[key][:, idx] = 0
        return idx

    def num_additional_blocks_req_need(self, tid: str, target_seq_len: int) -> int:
        """Calculates the number of additional blocks needed to store tokens up to the target sequence length.
            Computes the difference between the blocks required for the target sequence length and the blocks
            currently allocated to the request. The result represents how many new blocks need to be allocated
            beyond what the request already has.
        Args:
            tid: Unique id of the request
            target_seq_len: Desired total sequence length including existing tokens
        Return:
            Number of additional kv_cache blocks required to reach the target sequence length
        """
        if tid in self.block_table:
            return max(
                0,
                ceil_div(target_seq_len, self.block_size) - len(self.block_table[tid]),
            )
        return max(0, ceil_div(target_seq_len, self.block_size))

    def allocate_block_for_cache(self, tasks: "PackedTasksBase"):
        """Allocate blocks for the requests to write into cache."""
        KVCacheBase.prepare_cache_prefill(self, tasks)
        for tid, new_seq_len in zip(tasks.task_ids, self.seq_len_delta.new.lens_list):
            if tid not in self.block_table:
                self.block_table[tid] = []

            # Allocate blocks for the request
            needs_blocks = self.num_additional_blocks_req_need(tid, new_seq_len)
            self.block_table[tid].extend(
                [self.get_free_block() for _ in range(needs_blocks)]
            )
        self._upd_gpu_block_table(tasks.task_ids)

    @override
    def prepare_cache_prefill(self, tasks: "PackedTasksBase"):
        pass

    @override
    def prepare_cache_decode(self, tasks: "PackedTasksBase"):
        pass

    def free_req_cache_blocks(self, tid: str):
        for block_id in self.block_table[tid]:
            self.free_blocks.append(block_id)
        del self.block_table[tid]

    @override
    def finalize_cache_all_decode(self, tasks: "PackedTasksBase"):
        KVCacheBase.finalize_cache_all_decode(self, tasks)
        for tid in tasks.task_ids:
            if tid not in self.tid_to_cached_len:
                return
            if tid not in self.tid_to_multimodal_len:
                return
            if tid in self.block_table:
                self.free_req_cache_blocks(tid)
            self.req_len.pop(tid, None)
            self.request_metadata.pop(tid, None)
            self.finalize_cache_all_decode_multimodal(tid)

    def finalize_cache_all_decode_multimodal(self, tid: str):
        del self.tid_to_multimodal_len[tid]


class DenseKVCache(KVCacheBase):
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
        quant_type: str = None,
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
            quant_type=quant_type,
            device=device,
        )
        self.block_size = self.max_seq_len
        self.num_blocks = num_hot_req
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
        self.use_i64_offsets = self.needs_i64_kv_offsets()

    def needs_i64_kv_offsets(self) -> bool:
        for _, t in self.kv_buffer.items():
            kv_cache = t[0]
            kv = kv_cache.view(kv_cache.shape[0], kv_cache.shape[1], -1)

            max_kv_off = (
                (kv.shape[0] - 1) * kv.stride(0)
                + (kv.shape[1] - 1) * kv.stride(1)
                + (kv.shape[2] - 1) * kv.stride(2)
            )
            if max_kv_off > (1 << 31) - 1:
                return True
        return False

    def update_page_offs(self):
        pass

    @property
    def num_free_blocks(self):
        """Return number of free blocks"""
        return sum(1 for is_available in self.slot_availability if is_available == True)

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
    def prepare_cache_prefill(self, tasks: "PackedTasksBase"):
        task_ids = tasks.task_ids
        super().prepare_cache_prefill(tasks)

        # get start_idx and end_idx of current slot_group
        start_idx, end_idx = self.get_start_and_end_idx()

        # Only allocate slots in current slot_group
        slot_id = start_idx
        for it, tid in enumerate(task_ids):
            if tid not in self.req2slot:
                allocated = False
                while slot_id < end_idx:
                    if self.slot_availability[slot_id]:
                        self.req2slot[tid] = slot_id
                        self.slot_availability[slot_id] = False
                        self.hot_reqs[slot_id] = tid
                        allocated = True
                        slot_id += 1
                        break
                    slot_id += 1
                assert allocated, f"Failed to allocate slot for {tid}"

        start_pos = self.req2slot[task_ids[0]]
        self._prepare_cache(task_ids, start_pos)

    @override
    def prepare_cache_decode(self, tasks: "PackedTasksBase"):
        super().prepare_cache_decode(tasks)
        task_ids = tasks.task_ids
        start_pos = self.get_start_and_end_idx()[0]
        self._prepare_cache(task_ids, start_pos)

    def _prepare_cache(self, task_ids: list[str], start_pos: int):
        assert (
            start_pos + len(task_ids) <= self.num_hot_req
        ), f"start_pos:{start_pos}, number of req:{len(task_ids)}, num_hot_req:{self.num_hot_req}"
        for key in self.kv_buffer:
            self.prepared_cache[key] = self.kv_buffer[key][
                :, start_pos : start_pos + len(task_ids)
            ]

    @override
    def get_accessor(self, layer_id: int, is_mtp: bool = False) -> DenseKVCacheAccessor:
        local_layer_id = self.layer_id_map.to_local(layer_id)
        ret_kv = {
            key: cache[local_layer_id] for key, cache in self.prepared_cache.items()
        }
        return DenseKVCacheAccessor(ret_kv, self.use_i64_offsets)

    @override
    def finalize_cache_all_decode(self, tasks: "PackedTasksBase"):
        for tid in tasks.task_ids:
            if tid not in self.hot_reqs:
                continue
            slot_id = self.hot_reqs.index(tid)
            if slot_id is None:  # not in the hot slot
                continue

            # get end_idx in tid slot
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

            if slot_last_id is not None:
                for key in self.kv_buffer:
                    self.kv_buffer[key][:, slot_id] = self.kv_buffer[key][
                        :, slot_last_id
                    ]
                req_key = next(
                    (k for k, v in self.req2slot.items() if v == slot_last_id), None
                )
                if req_key is not None:
                    self.req2slot[req_key] = slot_id
                    self.hot_reqs[slot_id] = req_key
                self.hot_reqs[slot_last_id] = None
                self.slot_availability[slot_last_id] = True
                if tid in self.req2slot:
                    self.req2slot.pop(tid)
                for key in self.kv_buffer:
                    self.kv_buffer[key][:, slot_last_id].zero_()
            else:
                self.hot_reqs[slot_id] = None
                self.slot_availability[slot_id] = True
                self.req2slot.pop(tid)
                for key in self.kv_buffer:
                    self.kv_buffer[key][:, slot_id].zero_()
