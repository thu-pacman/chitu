# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Sequence, Optional, Callable, Iterable, TYPE_CHECKING
from typing_extensions import override
from dataclasses import dataclass
from logging import getLogger
import torch
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
    from chitu.distributed.pd_disaggregation.kv_transfer.cache_info import (
        CacheDistributions,
    )
    from chitu.distributed.pd_disaggregation.kv_transfer.transfer_buffers import (
        TransferBuffers,
    )


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
    """Lightweight descriptor passed to attention kernels to locate cached KV states.

    An accessor is a view into a specific layer's KV cache.  It carries the
    tensors and metadata that the attention backend needs to read (and
    optionally update) cached key/value data without coupling to the underlying
    cache storage layout (paged vs. dense).

    Subclasses add layout-specific fields: ``PagedKVCacheAccessor`` carries
    block tables for page-to-physical-address translation, while
    ``DenseKVCacheAccessor`` carries contiguous tensors indexed directly by
    request and position.
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
        # prefill_chunk_size is the GLOBAL budget across all DP and CP ranks. The
        # seq_len_delta buffers hold the per-step delta a single rank admits in
        # prefill (global-across-CP for that DP rank), so the worst-case per-DP
        # delta is global // dp_size (no pcp_size factor: pcp ranks each handle
        # only their own local slice, sized by CP at split time).
        prefill_chunk_size_per_dp = (
            ceil_div(
                prefill_chunk_size_global,
                get_global_args().infer.dp_size,
            )
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
            assert q_scale is None
            return fp8_pertensor_kvcache_quant(
                q,
                k,
                v,
                k_scale,
                v_scale,
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
        if tasks.inc_hit_tokens_list:
            cached_token_lens: list[int] = [
                self.tid_to_cached_len.get(tid, 0) + tasks.inc_hit_tokens_list[i]
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
            self.mtp_seq_len_delta.is_decode_stage = False

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

    def update_mtp_cache_accept(
        self, tasks: "PackedTasksBase", mtp_accept_indices: list[int]
    ):
        for tid, accept_index in zip(tasks.task_ids, mtp_accept_indices):
            if accept_index >= 0:
                self.tid_to_cached_len[tid] += -self.mtp_size + accept_index + 1

    def prepare_mtp_cache_decode(self, draft_offset: int):
        task_ids = self.curr_tids
        self.mtp_seq_len_delta.copy_from_list(
            [
                self.tid_to_cached_len[tid] - self.mtp_size + draft_offset
                for tid in task_ids
            ],
            [
                self.tid_to_cached_len[tid] - self.mtp_size + draft_offset + 1
                for tid in task_ids
            ],
        )
        self.mtp_seq_len_delta.is_decode_stage = True

    def prepare_cache_prefill_dllm(
        self,
        tasks: "PackedTasksBase",
        prefilling_lengths: list[int],
    ):
        """Prepare cache for DLLM prefill.

        DLLM prefill uses truncated prompt lengths (aligned to block_length).
        This method sets up seq_len_delta based on prefilling_lengths.

        Args:
            tasks: PackedTasks containing task information
            prefilling_lengths: Actual prefilling lengths for each task (truncated to block boundary)
        """
        self.curr_tids = tasks.task_ids

        # DLLM prefill starts from 0, ends at prefilling_lengths
        prev_seq_len = BatchedSeqLen(
            [0] * len(tasks.task_ids),
            device=self.device,
            cache_prefix_lens_tensor_device=False,
            cache_position_ids_tensor_device=False,
            cache_seq_ids_tensor_device=False,
        )
        next_seq_len = BatchedSeqLen(
            prefilling_lengths,
            device=self.device,
            cache_prefix_lens_tensor_device=False,
            cache_position_ids_tensor_device=False,
            cache_seq_ids_tensor_device=False,
        )
        self.seq_len_delta.copy_from(prev_seq_len, next_seq_len)

        for tid, seq_len in zip(tasks.task_ids, prefilling_lengths):
            self.tid_to_cached_len[tid] = seq_len

    def prepare_cache_decode_dllm(
        self,
        tasks: "PackedTasksBase",
        decoding_start: torch.Tensor,
        block_length: int,
    ):
        """Prepare cache for DLLM decode.

        Args:
            tasks: PackedTasks containing task information
            decoding_start: Tensor of starting positions for each sequence
            block_length: Length of decode block
        """
        self.curr_req_ids = tasks.task_ids

        # Set up seq_len_delta based on decoding_start + block_length
        new_lens = decoding_start + block_length
        self.seq_len_delta.copy_from_tensor(decoding_start, new_lens)

    def finalize_cache_single_decode_dllm(
        self, req_ids: list[str], block_finished: torch.Tensor, block_length: int
    ):
        """Finalize DLLM decode: update tid_to_cached_len for finished blocks."""
        finished_indices = block_finished.nonzero(as_tuple=True)[0].tolist()
        for idx in finished_indices:
            self.tid_to_cached_len[req_ids[idx]] += block_length
        self.curr_req_ids = None

    def finalize_cache_all_decode(self, tasks: "PackedTasksBase"):
        for tid in tasks.task_ids:
            if tid in self.tid_to_cached_len:
                self.tid_to_cached_len.pop(tid)

    def get_gpu_block_table(self):
        return None

    def get_seq_len_delta(self, is_mtp=False):
        return self.mtp_seq_len_delta if is_mtp else self.seq_len_delta


class PagedKVCache(KVCacheBase):
    """Paged KV cache with virtual-to-physical block mapping.

    PagedKVCache divides the KV cache into fixed-size blocks (pages) and maps
    each request's logical positions to physical blocks via a block table.

    Key design points:
    - ``block_table``: maps each request's logical block offsets to physical
      block indices.
    - Prefix-cache sharing, when enabled for a cache type, is managed by the
      cache manager rather than the cache itself.
    - ``allocatable_max_num_blocks`` records the effective upper bound used by
      warmup-time reallocation.
    """

    def __init__(
        self,
        layer_id_map: GlobalLocalMap,
        *,
        num_hot_req: int,
        max_seq_len: int,
        num_blocks: int,
        page_table_max_seq_len: Optional[int] = None,
        shape_per_token_dict: Optional[dict[str, torch.Size | Sequence[int]]] = None,
        dtype_dict: Optional[dict[str, torch.dtype]] = None,
        n_local_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        quant_type: str = None,
        device="cuda",
        block_size: int = 512,  # must be a multiple of 256 for FlashAttention
        is_singleton: bool = False,
        manager_name: str = "main",
        split_size: int = 0,
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
        if page_table_max_seq_len is None:
            page_table_max_seq_len = max_seq_len + mtp_extra
        else:
            page_table_max_seq_len = int(page_table_max_seq_len)
            if page_table_max_seq_len < 0:
                raise ValueError(
                    "page_table_max_seq_len must be >= 0, "
                    f"got {page_table_max_seq_len}"
                )
        self.page_table_max_seq_len = page_table_max_seq_len
        self.max_blocks_per_req = ceil_div(page_table_max_seq_len, block_size)
        self.page_table_max_num_blocks = self.max_blocks_per_req * num_hot_req
        self.max_num_blocks = self.page_table_max_num_blocks

        if get_global_args().infer.enable_prefix_caching:
            self.allocatable_max_num_blocks = 1 << 60
        else:
            self.allocatable_max_num_blocks = self.page_table_max_num_blocks

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.manager_name = manager_name
        self.split_size = split_size

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
            f"infer.max_batch_size={get_global_args().infer.max_batch_size}, "
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
            for tid, item in zip(tasks.task_ids, tasks.new_cache_ids_list):
                new_cache_ids = item.get(self.manager_name, [])
                self.block_table[tid].extend(new_cache_ids)
        self._upd_gpu_block_table(tasks.task_ids)

    def prepare_cache_prefill_dllm(
        self,
        tasks: "PackedTasksBase",
        prefilling_lengths: list[int],
    ):
        """Prepare cache for DLLM prefill.

        Similar to prepare_cache_prefill, but uses prefilling_lengths for seq_len_delta.
        DLLM prefill truncates prompts to block boundaries.

        Args:
            tasks: PackedTasks containing new_cache_ids_list
            prefilling_lengths: Actual prefilling lengths for each task
        """
        # Call base class to set up seq_len_delta and tid_to_cached_len
        super().prepare_cache_prefill_dllm(tasks, prefilling_lengths)

        # Receive pre-allocated block indices from scheduler
        if tasks.new_cache_ids_list:
            for tid, item in zip(tasks.task_ids, tasks.new_cache_ids_list):
                new_cache_ids = item.get(self.manager_name, [])
                self.block_table[tid].extend(new_cache_ids)
        self._upd_gpu_block_table(tasks.task_ids)

    @override
    def prepare_cache_decode(self, tasks: "PackedTasksBase"):
        # Prepare enough block table for next decoding. When decoding, AttnBackend will fill new kv into
        # paged kv cache in place.
        super().prepare_cache_decode(tasks)
        if tasks.new_cache_ids_list:
            for tid, item in zip(tasks.task_ids, tasks.new_cache_ids_list):
                new_cache_ids = item.get(self.manager_name, [])
                self.block_table[tid].extend(new_cache_ids)
        self._upd_gpu_block_table(tasks.task_ids)

    def prepare_cache_decode_dllm(
        self,
        tasks: "PackedTasksBase",
        decoding_start: torch.Tensor,
        block_length: int,
    ):
        """Prepare cache for DLLM decode.

        Similar to prepare_cache_decode, this receives new_cache_ids_list from scheduler
        and updates block_table accordingly.
        """
        # Call base class to set curr_req_ids and seq_len_delta
        super().prepare_cache_decode_dllm(tasks, decoding_start, block_length)

        # Receive pre-allocated block indices from scheduler (via new_cache_ids_list)
        if tasks.new_cache_ids_list:
            for tid, item in zip(tasks.task_ids, tasks.new_cache_ids_list):
                new_cache_ids = item.get(self.manager_name, [])
                self.block_table[tid].extend(new_cache_ids)
        self._upd_gpu_block_table(tasks.task_ids)

    def estimate_bytes_per_block(self) -> int:
        """Estimate additional bytes required to allocate 1 more KV page/block.

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

    def get_kv_transfer_buffers(
        self,
        buffers: "TransferBuffers",
        req_id: str,
        block_indices: list[int],
        *,
        local_dists: "CacheDistributions",
        remote_dists: "CacheDistributions",
    ):
        """Register block entries for send or recv into TransferBuffers."""
        if not block_indices:
            return

        global_layers = [
            self.layer_id_map.to_global(local_layer)
            for local_layer in range(self.num_layers)
        ]

        for key, cache in self.paged_kv_cache.items():
            assert (
                cache.is_contiguous()
            ), f"kv cache {key} must be contiguous for transfer"
            ld = local_dists.dists[key]
            rd = remote_dists.dists[key]
            n_chunks, split_len = ld.calc_chunking(rd)

            elem_size = cache.element_size()
            layer_stride_bytes = cache.stride(0) * elem_size
            block_stride_bytes = cache.stride(1) * elem_size
            assert (
                block_stride_bytes % n_chunks == 0
            ), f"block bytes {block_stride_bytes} not divisible by n_chunks {n_chunks}"
            chunk_bytes = block_stride_bytes // n_chunks
            base_ptr = cache.data_ptr()

            for i, block_id in enumerate(block_indices):
                for local_layer, global_layer in enumerate(global_layers):
                    layer_block_base_ptr = (
                        base_ptr
                        + local_layer * layer_stride_bytes
                        + block_id * block_stride_bytes
                    )
                    for j in range(n_chunks):
                        buffers.add(
                            layer_block_base_ptr + chunk_bytes * j,
                            chunk_bytes,
                            cache_name=key,
                            req_id=req_id,
                            layer_id=global_layer,
                            block_id=i,
                            split_id=ld.split_id + j * split_len,
                            split_len=split_len,
                            replica_id=ld.replica_id,
                            replica_size=ld.replica_size,
                        )

    def kv_recv_reorder(
        self,
        new_block_ids: list[int],
        *,
        local_dists: "CacheDistributions",
        remote_dists: "CacheDistributions",
    ):
        """Permute chunk-interleaved layout back to T-major."""
        for key, cache in self.paged_kv_cache.items():
            if self.split_size == 0:
                continue
            ld = local_dists.dists[key]
            rd = remote_dists.dists[key]
            n_chunks, _ = ld.calc_chunking(rd)
            if n_chunks <= 1:
                continue

            for block_id in new_block_ids:
                for layer in range(cache.shape[0]):
                    block = cache[layer, block_id].contiguous()
                    cache[layer, block_id] = (
                        block.view(n_chunks, block.shape[0], -1)
                        .permute(1, 0, 2)
                        .reshape(block.shape)
                        .contiguous()
                    )


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
        split_size: int = 0,
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
            manager_name="singleton",
            is_singleton=True,
            split_size=split_size,
        )
        self.max_blocks_per_req = 1
        self.max_num_blocks = self.max_blocks_per_req * num_hot_req

        if self.mtp_size > 1:
            self.is_mtp_decode_stage = False

    @override
    def prepare_cache_prefill(self, tasks: "PackedTasksBase"):
        KVCacheBase.prepare_cache_prefill(self, tasks)

        if self.mtp_size > 1:
            self.is_mtp_decode_stage = False

        if not tasks.new_cache_ids_list:
            self._upd_gpu_block_table(tasks.task_ids)
            return

        for tid, item in zip(tasks.task_ids, tasks.new_cache_ids_list):
            new_cache_ids = item.get(self.manager_name, [])
            if new_cache_ids:
                self.block_table[tid] = list(new_cache_ids)
                # Zero the block tensor so linear recurrent state
                # starts fresh.  Without this, stale state from a
                # previous request can corrupt generation quality.
                for cache_id in new_cache_ids:
                    for key in self.paged_kv_cache:
                        self.paged_kv_cache[key][:, cache_id] = 0
        self._upd_gpu_block_table(tasks.task_ids)

    @override
    def prepare_cache_decode(self, tasks: "PackedTasksBase"):
        KVCacheBase.prepare_cache_decode(self, tasks)
        self._upd_gpu_block_table(tasks.task_ids)

        if self.mtp_size > 1:
            self.is_mtp_decode_stage = True

    @override
    def finalize_cache_all_decode(self, tasks: "PackedTasksBase"):
        KVCacheBase.finalize_cache_all_decode(self, tasks)
        for tid in tasks.task_ids:
            if tid in self.block_table:
                del self.block_table[tid]

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

    def insert_mtp_state_from_transfer(
        self, tid: str, page_index: int, prefix_length: int
    ):
        """
        Register transferred MTP hidden state into block table.
        For MTP, each request uses exactly one block.
        Unlike insert_kv_cache_from_transfer, this doesn't require empty block_table
        since SingletonPagedKVCache may have allocated a block during prepare_cache_prefill.
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

    @override
    def prepare_mtp_cache_decode(self, draft_offset: int):
        pass

    @override
    def update_mtp_cache_accept(
        self, tasks: "PackedTasksBase", mtp_accept_indices: list[int]
    ):
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
    """Contiguous (non-paged) KV cache — one fixed-size buffer per request.

    DenseKVCache is designed for dense (a.k.a. skew) KV cache. Additionally,
    it is used for some auxiliary fixed-length KV caches in some models.
    """

    def __init__(
        self,
        layer_id_map: GlobalLocalMap,
        *,
        num_hot_req: int,
        max_seq_len: int,
        storage_max_seq_len: Optional[int] = None,
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
        self.storage_max_seq_len = (
            self.max_seq_len
            if storage_max_seq_len is None
            else int(storage_max_seq_len)
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
                    self.storage_max_seq_len,
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

    def estimate_bytes_per_block(self) -> int:
        total = 0
        for key, shape in self.shape_per_token_dict.items():
            n_elem_per_token = 1
            for dim in tuple(shape):
                n_elem_per_token *= int(dim)
            elem_size = torch.empty((), dtype=self.dtype_dict[key]).element_size()
            total += (
                int(self.num_layers)
                * int(self.storage_max_seq_len)
                * n_elem_per_token
                * elem_size
            )
        return int(total)

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

    def _copy_cache_slot(self, dst_slot: int, src_slot: int):
        for key in self.kv_buffer:
            self.kv_buffer[key][:, dst_slot] = self.kv_buffer[key][:, src_slot]

    def _zero_cache_slot(self, slot: int):
        for key in self.kv_buffer:
            self.kv_buffer[key][:, slot].zero_()

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
                self._copy_cache_slot(slot_id, slot_last_id)
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
                self._zero_cache_slot(slot_last_id)
            else:
                self.hot_reqs[slot_id] = None
                self.slot_availability[slot_id] = True
                self.req2slot.pop(tid)
                self._zero_cache_slot(slot_id)


class DeepSeekV4DenseKVCache(DenseKVCache):
    def __init__(self, *args, **kwargs):
        self.request_shape_dict = kwargs.pop("request_shape_dict", {})
        self.request_dtype_dict = kwargs.pop("request_dtype_dict", {})
        super().__init__(*args, **kwargs)
        self.request_buffer: dict[str, torch.Tensor] = {}
        self.prepared_request_cache: dict[str, torch.Tensor] = {}
        for key, shape in self.request_shape_dict.items():
            dtype = self.request_dtype_dict.get(key, torch.get_default_dtype())
            self.request_buffer[key] = torch.zeros(
                (self.num_layers, self.num_hot_req) + tuple(shape),
                dtype=dtype,
                device=self.device,
            )

    def estimate_bytes_per_block(self) -> int:
        total = super().estimate_bytes_per_block()
        for key, shape in self.request_shape_dict.items():
            n_elem_per_req = 1
            for dim in tuple(shape):
                n_elem_per_req *= int(dim)
            dtype = self.request_dtype_dict.get(key, torch.get_default_dtype())
            elem_size = torch.empty((), dtype=dtype).element_size()
            total += int(self.num_layers) * n_elem_per_req * elem_size
        return int(total)

    def _prepare_cache(self, task_ids: list[str], start_pos: int):
        super()._prepare_cache(task_ids, start_pos)
        for key in self.request_buffer:
            self.prepared_request_cache[key] = self.request_buffer[key][
                :, start_pos : start_pos + len(task_ids)
            ]

    def get_accessor(self, layer_id: int, is_mtp: bool = False):
        accessor = super().get_accessor(layer_id, is_mtp)
        local_layer_id = self.layer_id_map.to_local(layer_id)
        accessor.kv.update(
            {
                key: cache[local_layer_id]
                for key, cache in self.prepared_request_cache.items()
            }
        )
        return accessor

    def _copy_cache_slot(self, dst_slot: int, src_slot: int):
        super()._copy_cache_slot(dst_slot, src_slot)
        for key in self.request_buffer:
            self.request_buffer[key][:, dst_slot] = self.request_buffer[key][
                :, src_slot
            ]

    def _zero_cache_slot(self, slot: int):
        super()._zero_cache_slot(slot)
        for key in self.request_buffer:
            self.request_buffer[key][:, slot].zero_()

    def get_deepseek_v4_cache_slots(self, seq_delta) -> list[int]:
        curr_tids = self.curr_tids
        if curr_tids is None or len(curr_tids) != seq_delta.batch_size:
            return list(range(seq_delta.batch_size))
        return [self.req2slot[tid] for tid in curr_tids]


class DeepSeekV4PagedKVCache(PagedKVCache):
    """Paged KV cache with block-backed DeepSeek-V4 compressor request state."""

    def __init__(self, *args, **kwargs):
        self.request_shape_dict = kwargs.pop("request_shape_dict", {})
        self.request_dtype_dict = kwargs.pop("request_dtype_dict", {})
        super().__init__(*args, **kwargs)
        self.request_buffer: dict[str, torch.Tensor] = {}
        self._allocate_request_buffer()

    def _allocate_request_buffer(self):
        self.request_buffer.clear()
        for key, shape in self.request_shape_dict.items():
            dtype = self.request_dtype_dict.get(key, torch.get_default_dtype())
            # Pending compressor state follows the same block lifecycle as the
            # sliding-window KV page. The first block of each request is the
            # anchor slot used by model_deepseek_v4.py.
            self.request_buffer[key] = torch.zeros(
                (self.num_layers, self.num_blocks) + tuple(shape),
                dtype=dtype,
                device=self.device,
            )

    def _zero_request_blocks(self, block_ids):
        if not self.request_buffer:
            return
        block_ids = [int(block_id) for block_id in block_ids]
        if not block_ids:
            return
        block_ids_tensor = torch.tensor(block_ids, dtype=torch.long, device=self.device)
        for key in self.request_buffer:
            self.request_buffer[key].index_fill_(1, block_ids_tensor, 0)

    def _zero_new_request_blocks(self, tasks: "PackedTasksBase"):
        if not tasks.new_cache_ids_list:
            return
        new_block_ids = []
        for item in tasks.new_cache_ids_list:
            new_block_ids.extend(item.get(self.manager_name, []))
        self._zero_request_blocks(new_block_ids)

    @override
    def prepare_cache_prefill(self, tasks: "PackedTasksBase"):
        super().prepare_cache_prefill(tasks)
        self._zero_new_request_blocks(tasks)

    @override
    def prepare_cache_decode(self, tasks: "PackedTasksBase"):
        super().prepare_cache_decode(tasks)
        self._zero_new_request_blocks(tasks)

    @override
    def realloc(self, num_blocks):
        super().realloc(num_blocks)
        self._allocate_request_buffer()

    @override
    def estimate_bytes_per_block(self) -> int:
        total = super().estimate_bytes_per_block()
        for key, shape in self.request_shape_dict.items():
            n_elem_per_block = 1
            for dim in tuple(shape):
                n_elem_per_block *= int(dim)
            dtype = self.request_dtype_dict.get(key, torch.get_default_dtype())
            elem_size = torch.empty((), dtype=dtype).element_size()
            total += int(self.num_layers) * n_elem_per_block * elem_size
        return int(total)

    @override
    def finalize_cache_all_decode(self, tasks: "PackedTasksBase"):
        block_ids_to_zero = []
        for tid in tasks.task_ids:
            block_ids_to_zero.extend(self.block_table.get(tid, []))
        super().finalize_cache_all_decode(tasks)
        self._zero_request_blocks(block_ids_to_zero)

    @override
    def get_accessor(self, layer_id: int, is_mtp: bool = False) -> PagedKVCacheAccessor:
        accessor = super().get_accessor(layer_id, is_mtp)
        local_layer_id = self.layer_id_map.to_local(layer_id)
        accessor.kv.update(
            {key: cache[local_layer_id] for key, cache in self.request_buffer.items()}
        )
        return accessor

    def get_deepseek_v4_cache_slots(self, seq_delta) -> list[int]:
        curr_tids = self.curr_tids
        if curr_tids is None or len(curr_tids) != seq_delta.batch_size:
            if self.request_buffer:
                raise RuntimeError(
                    "DeepSeek-V4 paged request state requires current task ids "
                    "to resolve block-backed slots"
                )
            return list(range(seq_delta.batch_size))
        slots = []
        for tid in curr_tids:
            blocks = self.block_table.get(tid, [])
            if not blocks:
                raise RuntimeError(f"DeepSeek-V4 paged cache has no block for {tid}")
            slots.append(int(blocks[0]))
        return slots


class DeepSeekV4SlidingWindowPagedKVCache(DeepSeekV4PagedKVCache):
    """One-page-per-request paged cache for DeepSeek-V4 sliding-window KV.

    This is the DeepSeek-V4 sliding-window variant of singleton paged cache:
    each active request owns exactly one physical page, and that page is a
    ring buffer with ``window_size`` token slots. ``window_size`` here is the
    physical page width used by the cache manager; the model's logical visible
    window may be smaller. Unlike ``SingletonPagedKVCache`` for MTP/linear
    states, the in-page offset is not always zero; it is
    ``logical_position % window_size``.

    The cache still consumes block ids from Chitu's normal ``main`` paged cache
    manager, because the scheduler currently requires a ``main`` manager for
    capacity accounting and request lifecycle metadata.
    """

    def __init__(
        self,
        *args,
        num_hot_req: int,
        max_seq_len: int,
        window_size: int,
        num_blocks: Optional[int] = None,
        block_size: Optional[int] = None,
        **kwargs,
    ):
        self.window_size = int(window_size)
        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        if block_size is not None and int(block_size) != self.window_size:
            raise ValueError(
                "DeepSeek-V4 sliding-window paged cache requires "
                f"block_size == window_size, got block_size={block_size}, "
                f"window_size={self.window_size}"
            )

        fixed_num_blocks = int(num_hot_req)
        if num_blocks is not None and int(num_blocks) != fixed_num_blocks:
            logger.info(
                "DeepSeek-V4 sliding-window paged cache uses one page per hot "
                "request; ignoring num_blocks=%s and using num_hot_req=%s",
                int(num_blocks),
                fixed_num_blocks,
            )

        super().__init__(
            *args,
            num_hot_req=num_hot_req,
            max_seq_len=max_seq_len,
            num_blocks=fixed_num_blocks,
            page_table_max_seq_len=self.window_size,
            block_size=self.window_size,
            is_singleton=True,
            **kwargs,
        )

        # The logical sequence can be longer than the window, but the page table
        # has exactly one entry per request. The parent sees page_table_max_seq_len
        # as one window, so it already builds the correct one-entry table.
        self.allocatable_max_num_blocks = self.page_table_max_num_blocks
        self.fixed_num_blocks = True

    @override
    def realloc(self, num_blocks):
        if int(num_blocks) != self.num_hot_req:
            logger.info(
                "DeepSeek-V4 sliding-window paged cache keeps one page per hot "
                "request; requested num_blocks=%s, using num_hot_req=%s",
                int(num_blocks),
                self.num_hot_req,
            )
        super().realloc(self.num_hot_req)

    @override
    @cuda_graph_safe_cached_property("_page_ids_static_tensor", "_page_ids_up_to_date")
    def page_ids(self):
        return self.gpu_block_table.get()[
            self.seq_len_delta.delta_seq_ids_tensor_device, 0
        ]

    @override
    @cuda_graph_safe_cached_property("_page_ids_static_tensor", "_page_ids_up_to_date")
    def page_ids_mtp(self):
        return self.gpu_block_table.get()[
            self.mtp_seq_len_delta.delta_seq_ids_tensor_device, 0
        ]

    @override
    @cuda_graph_safe_cached_property(
        "_offs_in_page_static_tensor", "_offs_in_page_up_to_date"
    )
    def offs_in_page(self):
        return self.seq_len_delta.delta_position_ids_tensor_device % self.window_size

    @override
    @cuda_graph_safe_cached_property(
        "_offs_in_page_static_tensor", "_offs_in_page_up_to_date"
    )
    def offs_in_page_mtp(self):
        return (
            self.mtp_seq_len_delta.delta_position_ids_tensor_device % self.window_size
        )
