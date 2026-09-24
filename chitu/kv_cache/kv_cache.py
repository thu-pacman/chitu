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
from itertools import accumulate

from chitu.cuda_graph import cuda_graph_safe_cached_property
from chitu.global_vars import get_slot_handle, get_global_args
from chitu.kv_cache.manager_names import MAIN_CACHE_NAME, MTP_CACHE_NAME
from chitu.static_tensor import StaticTensor
from chitu.batched_seq_len import BatchedSeqLen, BatchedSeqLenDelta
from chitu.utils import ceil_div, create_tensor, max_alloc_seq_len
from chitu.ops import fp8_pertensor_kvcache_quant, fp8_pertoken_kvcache_quant_dsa

if TYPE_CHECKING:
    from chitu.task import PackedTasksBase
    from chitu.distributed.pd_disaggregation.kv_transfer.cache_info import (
        RankCacheInfos,
        InstanceCacheInfos,
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
class SingletonPagedKVCacheAccessor(KVCacheAccessor):
    kv: dict[str, torch.Tensor]
    get_write_page_ids: Callable[[], torch.Tensor]
    get_read_page_ids: Callable[[], torch.Tensor]
    get_ckpt_write_pages: Callable[[], Optional[torch.Tensor]]
    get_ckpt_cu_starts: Callable[[], Optional[torch.Tensor]]
    use_i64_offsets: bool = False


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
    FP8_PERTOKEN_INDEXER = "fp8_pertoken_indexer"
    FP8_E5M2 = "fp8_e5m2"

    @property
    def needs_kv_scales(self) -> bool:
        return self in {KVCacheQuantType.FP8_PERTENSOR}


BlockTableUpdate = tuple[int, int, Sequence[int]]  # batch_index, start, block_ids


@dataclass
class GPUBlockTableSyncState:
    """Previous GPU synchronization state indexed by batch index."""

    batch_states: list[Optional[tuple[str, int]]]
    batch_size: int = 0

    @classmethod
    def create(cls, capacity: int) -> "GPUBlockTableSyncState":
        return cls(batch_states=[None] * capacity)

    def get(self, batch_index: int) -> Optional[tuple[str, int]]:
        return self.batch_states[batch_index]

    def commit(self, task_ids: list[str], lengths: list[int]) -> None:
        for batch_index, (task_id, length) in enumerate(zip(task_ids, lengths)):
            self.batch_states[batch_index] = (task_id, length)
        for batch_index in range(len(task_ids), self.batch_size):
            self.batch_states[batch_index] = None
        self.batch_size = len(task_ids)


class KVCacheBase:
    # Tokens per KV-cache block: the page size of `PagedKVCache`, the whole
    # sequence for `DenseKVCache`.
    block_size: int

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
        # 每个请求可能写入 kv cache 的最大长度（decode 最后一步可能多算 mtp_size 个 token）
        self.max_total_len = num_hot_req * max_alloc_seq_len(max_seq_len)
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
            use_prefix_lens_static_tensor=True,
            use_position_ids_static_tensor=False,
            use_seq_ids_static_tensor=False,
            use_delta_position_ids_static_tensor=True,
            use_delta_seq_ids_static_tensor=True,
        )

        self.mtp_size = get_global_args().infer.mtp_size
        if self.mtp_size > 1:
            self.mtp_seq_len_delta = BatchedSeqLenDelta(
                device=self.device,
                max_batch_size=num_hot_req,
                max_total_len=self.max_total_len,
                max_total_delta_len=self.max_total_delta_len,
                use_prefix_lens_static_tensor=True,
                use_position_ids_static_tensor=False,
                use_seq_ids_static_tensor=False,
                use_delta_position_ids_static_tensor=True,
                use_delta_seq_ids_static_tensor=True,
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
            use_prefix_lens_static_tensor=False,
            use_position_ids_static_tensor=False,
            use_seq_ids_static_tensor=False,
        )
        next_seq_len = BatchedSeqLen(
            [cached + delta for cached, delta in zip(cached_token_lens, delta_seq_len)],
            device=self.device,
            use_prefix_lens_static_tensor=False,
            use_position_ids_static_tensor=False,
            use_seq_ids_static_tensor=False,
        )
        self.seq_len_delta.copy_from(prev_seq_len, next_seq_len)
        if self.mtp_size > 1:
            self.mtp_seq_len_delta.copy_from(prev_seq_len, next_seq_len)
            self.mtp_seq_len_delta.is_decode_stage = False

        for tid, seq_len in zip(tasks.task_ids, next_seq_len.lens_list):
            self.tid_to_cached_len[tid] = seq_len

    def prepare_cache_decode(self, tasks: "PackedTasksBase"):
        # A fully-cached task is converted to Decode by the scheduler and skips
        # prefill_step, so prepare_cache_prefill never ran for it and
        # tid_to_cached_len is still 0. Preset it here using the hit length from
        # inc_hit_tokens_list (N-1 when fully cached) so that decode counts the
        # N-1 already-cached prefix KV into old seq len and computes the Nth token
        # at position N-1.
        for tid, inc_hit in zip(tasks.task_ids, tasks.inc_hit_tokens_list):
            if inc_hit > self.tid_to_cached_len.get(tid, 0):
                self.tid_to_cached_len[tid] = inc_hit

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
                assert tid in self.tid_to_cached_len
                self.tid_to_cached_len[tid] += -self.mtp_size + accept_index + 1

    def prepare_mtp_cache_decode(self, tasks: "PackedTasksBase", draft_offset: int):
        self.mtp_seq_len_delta.copy_from_list(
            [self.tid_to_cached_len[tid] + draft_offset - 1 for tid in tasks.task_ids],
            [self.tid_to_cached_len[tid] + draft_offset for tid in tasks.task_ids],
        )
        self.mtp_seq_len_delta.is_decode_stage = True
        self.update_page_offs()

    def advance_mtp_cache_decode(self) -> None:
        """Advance the MTP decode position by one step, on device.

        Device-side counterpart of ``prepare_mtp_cache_decode(tasks, offset)``
        for the sequential MTP draft loop: it advances the MTP sequence-length
        delta by exactly one token without rebuilding CPU lists, so the whole
        draft loop can be captured into a single CUDA graph.

        The per-step GPU block table stays unchanged: blocks covering the full
        ``max_seq_len + mtp_size`` span are reserved up front (see
        ``page_table_max_seq_len``), so only positions move, not block rows.
        Only the cached page ids/offsets are dropped, so that they are
        recomputed for the new position -- inside a captured region, that
        recomputation is what gets recorded into the graph.
        """
        self.mtp_seq_len_delta.advance_classic_by_one()
        self.update_page_offs()

    def update_page_offs(self) -> None:
        """Drop cached page ids / in-page offsets.

        Called whenever the decode position moves in place -- the MTP walks
        above do it themselves -- so that the next read re-derives them from
        the new position. Caches without pages have nothing to drop.
        """

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
            use_prefix_lens_static_tensor=False,
            use_position_ids_static_tensor=False,
            use_seq_ids_static_tensor=False,
        )
        next_seq_len = BatchedSeqLen(
            prefilling_lengths,
            device=self.device,
            use_prefix_lens_static_tensor=False,
            use_position_ids_static_tensor=False,
            use_seq_ids_static_tensor=False,
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

    #: Whether a request maps its tokens to physical blocks through ``block_table``.
    has_token_block_table: bool = True

    def __init__(
        self,
        layer_id_map: GlobalLocalMap,
        *,
        num_hot_req: int,
        max_seq_len: int,
        num_blocks: int,
        max_blocks_per_req: Optional[int] = None,
        shape_per_token_dict: Optional[dict[str, torch.Size | Sequence[int]]] = None,
        dtype_dict: Optional[dict[str, torch.dtype]] = None,
        n_local_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        quant_type: str = None,
        device="cuda",
        block_size: int = 512,  # must be a multiple of 256 for FlashAttention
        manager_name: str = MAIN_CACHE_NAME,
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

        if max_blocks_per_req is None:
            max_seq_len_consider_mtp = max_alloc_seq_len(max_seq_len)
            max_blocks_per_req = ceil_div(max_seq_len_consider_mtp, block_size)
        assert max_blocks_per_req > 0, f"max_blocks_per_req should bigger than 0"
        self.max_blocks_per_req = max_blocks_per_req
        # Page-table capacity: `gpu_block_table` has num_hot_req rows of
        # max_blocks_per_req entries. This is *not* the physical block pool size
        # (`num_blocks`), which may exceed it when prefix caching keeps cached
        # blocks alive beyond what active requests can index.
        self.max_num_blocks = self.max_blocks_per_req * num_hot_req

        if get_global_args().infer.enable_prefix_caching:
            self.allocatable_max_num_blocks = 1 << 60
        else:
            self.allocatable_max_num_blocks = self.max_num_blocks

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.manager_name = manager_name
        self.split_size = split_size

        if self.has_token_block_table:
            self.block_table: dict[str, list[int]] = defaultdict(
                list
            )  # {seq_id: block_ids}
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
        self.use_i64_offsets = self.needs_i64_kv_offsets()
        self.init_metadata_buffer()

    def init_metadata_buffer(self):
        self.gpu_block_table = StaticTensor(
            max_nelem=self.max_num_blocks, dtype=torch.int32, device=self.device
        )
        self._gpu_block_table_sync_state = GPUBlockTableSyncState.create(
            self.num_hot_req
        )
        self._page_ids_static_tensor = StaticTensor(
            max_nelem=self.max_total_delta_len, device=self.device, dtype=torch.int32
        )
        self._offs_in_page_static_tensor = StaticTensor(
            max_nelem=self.max_total_delta_len, device=self.device, dtype=torch.int32
        )
        self._page_ids_up_to_date = False
        self._offs_in_page_up_to_date = False

    def get_allocatable_max_num_blocks(self) -> int:
        return int(getattr(self, "allocatable_max_num_blocks", self.max_num_blocks))

    def realloc(self, num_blocks):
        requested_num_blocks = int(num_blocks)
        allocatable_cap = self.get_allocatable_max_num_blocks()

        logger.info(
            f"Requested realloc to {requested_num_blocks} KV blocks. "
            f"max_num_blocks={self.max_num_blocks}, "
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

    def _copy_gpu_block_table_updates(
        self,
        gpu_block_table: torch.Tensor,
        updates: list[BlockTableUpdate],
    ) -> None:
        """Copy planned valid ranges through one flat CPU staging tensor.

        This avoids materializing a padded Python ``batch x table_width`` list.
        Values outside each request's valid block-table prefix remain unspecified.
        """
        flat_block_ids = [
            block_id for _, _, block_ids in updates for block_id in block_ids
        ]
        staging = create_tensor(
            flat_block_ids,
            device="cpu",
            dtype=torch.int32,
            sync_free=self.device.type != "cpu",
        )

        staging_offset = 0
        for batch_index, start, block_ids in updates:
            num_blocks = len(block_ids)
            gpu_block_table[batch_index, start : start + num_blocks].copy_(
                staging[staging_offset : staging_offset + num_blocks],
                non_blocking=self.device.type != "cpu",
            )
            staging_offset += num_blocks

    def _validate_gpu_block_table_batch(self, task_ids: list[str]) -> None:
        if len(task_ids) > self.num_hot_req:
            raise ValueError(
                "block-table batch exceeds configured hot-request capacity: "
                f"batch_size={len(task_ids)} num_hot_req={self.num_hot_req}"
            )

    def _upd_gpu_block_table(
        self,
        task_ids: list[str],
        incremental: bool = False,
    ):
        """Synchronize CPU block-table rows into the GPU table.

        ``self.block_table`` is authoritative. ``incremental=True`` is reserved
        for the append-only scheduler path: an unchanged batch-index binding copies
        only ``blocks[previous_len:]``. Rebinds and other callers refresh the
        complete valid prefix. Consumers must use sequence lengths to bound every
        row; padding beyond ``len(self.block_table[task_id])`` is unspecified.
        """
        self._validate_gpu_block_table_batch(task_ids)
        block_rows = [self.block_table[tid] for tid in task_ids]
        max_len = max((len(blocks) for blocks in block_rows), default=0)

        if max_len > self.max_blocks_per_req:
            raise ValueError(
                "block-table row exceeds configured per-request capacity: "
                f"max_len={max_len} max_blocks_per_req={self.max_blocks_per_req} "
                f"max_seq_len={get_global_args().infer.max_seq_len} block_size={self.block_size}"
            )

        # Keep the row stride stable so synchronization history remains valid
        # regardless of whether CUDA Graph is enabled.
        self.gpu_block_table.set_shape((len(task_ids), self.max_blocks_per_req))
        gpu_block_table = self.gpu_block_table.get()
        state = self._gpu_block_table_sync_state
        updates: list[BlockTableUpdate] = []

        for batch_index, (task_id, blocks) in enumerate(zip(task_ids, block_rows)):
            binding = state.get(batch_index)
            previous_task_id, previous_len = binding or (None, 0)
            # Only an unchanged batch-index binding with monotonic growth has a dirty suffix.
            is_valid_append = (
                incremental
                and previous_task_id == task_id
                and previous_len <= len(blocks)
            )

            if is_valid_append:
                if previous_len < len(blocks):
                    updates.append((batch_index, previous_len, blocks[previous_len:]))
            else:
                if blocks:
                    updates.append((batch_index, 0, blocks))

        if updates:
            self._copy_gpu_block_table_updates(gpu_block_table, updates)

        state.commit(task_ids, [len(blocks) for blocks in block_rows])
        self.update_page_offs()

    def _update_block_table_from_scheduler(
        self,
        tasks: "PackedTasksBase",
        incremental: bool = False,
    ) -> None:
        """Apply scheduler allocations, then synchronize the active GPU rows.

        Owning both operations here makes ``incremental=True`` an append-only
        contract instead of trusting a caller-provided list of dirty block IDs.
        """
        task_ids = tasks.task_ids
        self._validate_gpu_block_table_batch(task_ids)

        new_cache_ids_list = tasks.new_cache_ids_list
        if new_cache_ids_list and len(new_cache_ids_list) != len(task_ids):
            raise ValueError(
                "new_cache_ids_list must be empty or match task_ids: "
                f"new_cache_ids={len(new_cache_ids_list)} task_ids={len(task_ids)}"
            )

        if new_cache_ids_list:
            for task_id, item in zip(task_ids, new_cache_ids_list):
                new_block_ids = item.get(self.manager_name, [])
                new_length = len(self.block_table.get(task_id, ())) + len(new_block_ids)
                if new_length > self.max_blocks_per_req:
                    raise ValueError(
                        "block-table row exceeds configured per-request capacity: "
                        f"task_id={task_id} new_len={new_length} "
                        f"max_blocks_per_req={self.max_blocks_per_req}"
                    )

            for task_id, item in zip(task_ids, new_cache_ids_list):
                self.block_table[task_id].extend(item.get(self.manager_name, []))

        self._upd_gpu_block_table(task_ids, incremental=incremental)

    def update_page_offs(self) -> None:
        self._page_ids_up_to_date = False
        self._offs_in_page_up_to_date = False

    @override
    def prepare_cache_prefill(self, tasks: "PackedTasksBase"):
        super().prepare_cache_prefill(tasks)
        self._update_block_table_from_scheduler(tasks)

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

        self._update_block_table_from_scheduler(tasks)

    @override
    def prepare_cache_decode(self, tasks: "PackedTasksBase"):
        # Prepare enough block table for next decoding. When decoding, AttnBackend will fill new kv into
        # paged kv cache in place.
        super().prepare_cache_decode(tasks)
        self._update_block_table_from_scheduler(tasks, incremental=True)

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

        self._update_block_table_from_scheduler(tasks, incremental=True)

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

    def kv_recv_reorder(
        self,
        block_ids: list[int],
        *,
        local_dists: "RankCacheInfos",
        remote_dists: "InstanceCacheInfos",
    ):
        """Permute chunk-interleaved layout back to T-major."""
        for key, cache in self.paged_kv_cache.items():
            if self.split_size == 0:
                continue
            ld = local_dists.get(key)
            rd = remote_dists.get_any(key)
            if ld is None or rd is None:
                continue
            n_chunks, _ = ld.calc_chunking(rd)
            if n_chunks <= 1:
                continue

            for block_id in block_ids:
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
    In-place-update recurrent-state cache (linear attention / RNN / MTP hidden state).

    A request owns two kinds of blocks, which this class stores separately:
        * ``inplace_block_ids``: the ``mtp_size`` in-place state blocks per request
          (``spec_0, ..., spec_{mtp-1}`` in the layout below), written in place every
          decode step;
        * ``ckpt_block_ids``: checkpoint blocks, appended incrementally and prefix-shared.

    The scheduler delivers both kinds together in ``new_cache_ids`` (per batch item,
    under ``manager_name``), in this order:
        1. [ spec_0, spec_1, ..., spec_{mtp-1} | ckpt_0(cached), ..., ckpt_k(fresh), ... ]
           when ``checkpoint_interval`` is not None,
        2. [ spec_0, spec_1, ..., spec_{mtp-1} ]
           when ``checkpoint_interval`` is None.
    """

    def __init__(
        self,
        layer_id_map: GlobalLocalMap,
        *,
        num_hot_req: int,
        num_blocks: int,
        shape_per_token_dict: Optional[dict[str, torch.Size | Sequence[int]]] = None,
        dtype_dict: Optional[dict[str, torch.dtype]] = None,
        n_local_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        device="cuda",
        split_size: int = 0,
        manager_name: str = MTP_CACHE_NAME,
        checkpoint_interval: Optional[int] = None,
    ):
        self.mtp_size = get_global_args().infer.mtp_size
        max_seq_len = int(get_global_args().infer.max_seq_len)
        self.checkpoint_interval = checkpoint_interval

        assert num_blocks > 0, f"num_blocks should bigger than 0, got {num_blocks}"
        # 每请求 block 预算：checkpoint_interval 不为 None 时为 mtp_size 个 spec block
        # 加上 ceil_div(max_alloc_seq_len(max_seq_len), checkpoint_interval) 个 ckpt block；
        # 为 None 时请求只持有 spec blocks。
        # 可寻址上界与 main cache 统一取 max_alloc_seq_len（见 chitu/utils.max_alloc_seq_len）：
        # decode 最后一步的推测段会让 alloc_seq_len 超过 max_seq_len，多出来的长度一旦跨过 C
        # 的整数倍就要多一个 ckpt block
        max_blocks_per_req = self.mtp_size
        if checkpoint_interval is not None:
            max_blocks_per_req = (
                ceil_div(max_alloc_seq_len(max_seq_len), checkpoint_interval)
                + self.mtp_size
            )

        self.inplace_block_ids: dict[str, list[int]] = {}
        # 上一 step 每个请求被接受的最后一个 draft token 在 inplace_block_ids 中的下标，
        # 由 update_mtp_cache_accept 写入（executor 的 _prepare_accept_indices），
        # _upd_gpu_block_table 据此算出 _read_page_ids
        self.tid_to_accept_index: dict[str, int] = {}
        # 每请求的 ckpt block ids，下标 j ↔ 覆盖位置 [jC, (j+1)C) 的那一块（C 是
        # checkpoint_interval）。本类没有基类那种 token → 页的 block_table（见下面的
        # block_table property），所以这份 dict 只能走 ckpt_block_ids 这个名字
        self.ckpt_block_ids: dict[str, list[int]] = defaultdict(list)

        super().__init__(
            layer_id_map,
            num_hot_req=num_hot_req,
            max_seq_len=max_seq_len,
            max_blocks_per_req=max_blocks_per_req,
            shape_per_token_dict=shape_per_token_dict,
            dtype_dict=dtype_dict,
            n_local_kv_heads=n_local_kv_heads,
            head_dim=head_dim,
            device=device,
            block_size=1,
            num_blocks=num_blocks,
            manager_name=manager_name,
            split_size=split_size,
        )

        self.is_decode_stage = False

    def init_metadata_buffer(self):
        assert (
            self.mtp_size >= 1
        ), f"mtp_size shouldn't less than 1, got {self.mtp_size}"
        self._write_page_ids = StaticTensor(
            max_nelem=self.num_hot_req * self.mtp_size,
            dtype=torch.int32,
            device=self.device,
        )  # 原地写缓存ids
        self._read_page_ids = StaticTensor(
            max_nelem=self.num_hot_req,
            dtype=torch.int32,
            device=self.device,
        )  # 每请求本 step 要读的 page id（上一 step 的 _write_page_ids 第 accept_index 列，
        # 该列是 checkpoint 边界时是 ckpt block）
        self._read_page_ids_written: list[int] = (
            []
        )  # _read_page_ids 当前step中的内容，用于去重
        self._ckpt_write_pages: Optional[torch.Tensor] = (
            None  # 写checkpoint缓存, 写checkpoint在cuda graph外，无需StaticTensor
        )
        # 见 ckpt_cu_starts
        self._ckpt_cu_starts: Optional[torch.Tensor] = None

    def _is_ckpt_pos(self, pos: int | torch.Tensor) -> bool:
        """位置 ``pos`` 的 state 是否落在 ckpt block 上。

        位置是从 seq[0] 起算的绝对位置，checkpoint 是每 C 个 token 的最后一个位置，所以
        判据就是 ``pos % C == C - 1``；其余位置的 state 都在 in-place block 的某一列上。

        没有 checkpoint_interval 时任何位置都不是 checkpoint 位置。
        """
        C = self.checkpoint_interval
        assert C is not None
        return (pos % C) == C - 1

    def _ckpt_idx(self, pos: int | torch.Tensor) -> int | torch.Tensor:
        """位置 ``pos``的 state 落在 ``ckpt_block_ids`` 的第几块上。"""
        C = self.checkpoint_interval
        assert C is not None
        return pos // C

    def _ckpt_page_id(self, tid: str, pos: int) -> int:
        """位置 ``pos`` 的 state 所在的 ckpt page id（``pos`` 必须是 checkpoint 位置）。"""
        assert self._is_ckpt_pos(pos), f"position {pos} is not a checkpoint position"
        idx = self._ckpt_idx(pos)
        ckpt_blocks = self.ckpt_block_ids[tid]
        assert idx < len(ckpt_blocks), (
            f"tid:{tid} checkpoint block {idx} not allocated yet "
            f"(pos={pos}, C={self.checkpoint_interval}, n_blocks={len(ckpt_blocks)}); "
            "the checkpoint state would be lost"
        )
        return ckpt_blocks[idx]

    def _upd_read_page_ids(
        self, task_ids: Sequence[str], consumed_lens: Sequence[int]
    ) -> None:
        """每个请求本 step 该从哪个 block 读 state。

        要读的 state 是「本 chunk / decode 窗口的第一个 token 之前那个位置」计算出的
        state，也就是位置 ``consumed_lens[i] - 1`` 的 state，它有两处可能的存放地：

        * 那个位置是 checkpoint 位置时（``_is_ckpt_pos(consumed - 1)``），落在
          ``_ckpt_page_id`` 上：prefill 续算时是上一个 chunk 的算子写进去的，或是命中的
          共享块（disaggregation 场景另见 insert_kv_cache_from_transfer）；decode 时是上一步
          被重定向的那一列（见 _update_write_page_ids）
        * 其余情况（chunk 尾、mtp_size=1 等）都在 in-place block 的第 accept_index
          列上，prefill / mtp_size=1 / 没走 MTP 的请求取第 0 列
        """
        if not task_ids:
            return
        C = self.checkpoint_interval
        page_ids = []
        for tid, n_consumed in zip(task_ids, consumed_lens):
            pos = n_consumed - 1
            if C is not None and n_consumed > 0 and self._is_ckpt_pos(pos):
                page_ids.append(self._ckpt_page_id(tid, pos))
                continue
            col = max(self.tid_to_accept_index.get(tid, 0), 0)
            page_ids.append(self.inplace_block_ids[tid][col])
        if page_ids == self._read_page_ids_written:
            return
        self._read_page_ids_written = page_ids
        self._read_page_ids.set(
            torch.tensor(page_ids, dtype=torch.int32, device=self.device)
        )

    def _update_write_page_ids(self, task_ids: Sequence[str]) -> None:
        """每个请求本 step 的 ``mtp_size`` 列各写哪个 block（``_write_page_ids``）。

        第 j 列是位置 ``start_len + j`` 的 state，默认写 in-place block 的第 j 列；decode 时
        窗口 ``[start_len, start_len + mtp_size)`` 里落在 checkpoint 位置（``_is_ckpt_pos``）
        的那几列直接写 ``_ckpt_page_id`` 给出的 ckpt block，省掉事后再从 in-place block 拷一次
        的搬运。
        """
        start_lens = self.seq_len_delta.old.lens_list
        C = self.checkpoint_interval
        pages = []
        for i, tid in enumerate(task_ids):
            row = self.inplace_block_ids[tid]
            if self.is_decode_stage and C is not None:
                row = list(row)  # 要改页时才复制，其余情况直接用 in-place block ids
                start_len = start_lens[i]
                pos = (start_len // C + 1) * C - 1  # 窗口内第一个 checkpoint 位置
                while pos < start_len + self.mtp_size:
                    row[pos - start_len] = self._ckpt_page_id(tid, pos)
                    pos += C
            pages.append(row)
        self._write_page_ids.set(
            torch.tensor(pages, dtype=torch.int32, device=self.device)
        )

    def _upd_gpu_block_table(
        self,
        task_ids: list[str],
        incremental: bool = False,
    ):
        self._update_write_page_ids(task_ids)

        self._upd_read_page_ids(task_ids, self.seq_len_delta.old.lens_list)

        self._upd_ckpt_write_pages(task_ids)

    def _upd_ckpt_write_pages(self, task_ids: Sequence[str]) -> None:
        """算出本 step 要写的 checkpoint：``_ckpt_write_pages`` 与 ``_ckpt_cu_starts``。

        只有 prefill 的 chunk 算子会写 checkpoint（decode 的 checkpoint 由
        ``_update_write_page_ids`` 直接把写页重定向到 ckpt block）.

        - ``_ckpt_write_pages``：每个 checkpoint 要写的 ckpt block，算子算出的 state 按顺序写过去
        - ``_ckpt_cu_starts``：每个 seq 有几个 checkpoint（累加形式），见 ``ckpt_cu_starts``
        """
        C = self.checkpoint_interval
        delta_pos = self.seq_len_delta.delta_position_ids_tensor_device
        if C is None or self.is_decode_stage or delta_pos.shape[0] == 0:
            self._ckpt_write_pages = None
            self._ckpt_cu_starts = None
            return

        # 每个 checkpoint 落在哪个页由 _ckpt_page_id 给出，顺便数出每个 seq 有几个
        is_ckpt = self._is_ckpt_pos(delta_pos)
        ckpt_seqs = self.seq_len_delta.delta_seq_ids_tensor_device[is_ckpt].tolist()
        ckpt_pos = delta_pos[is_ckpt].tolist()
        write_pages = []
        counts = [0] * len(task_ids)
        for seq_id, pos in zip(ckpt_seqs, ckpt_pos):
            write_pages.append(self._ckpt_page_id(task_ids[seq_id], pos))
            counts[seq_id] += 1
        self._ckpt_write_pages = torch.tensor(
            write_pages, dtype=torch.int32, device=self.device
        )

        # 算子按 chunk 起点每 C 个 token 取一个 checkpoint（接口对齐 flashinfer）。
        # scheduler 把续算长度和 chunk 大小都向下取整到 C 的倍数，保证 state 不会写
        # 到错位的页上。
        for tid, start, count in zip(
            task_ids, self.seq_len_delta.old.lens_list, counts
        ):
            assert start % C == 0 or count == 0, (
                f"tid:{tid} chunk starts at {start}, which is not a multiple of the "
                f"checkpoint interval {C}, but this chunk has {count} checkpoint(s); "
                "the chunk operators count checkpoint positions from the chunk start, "
                "so they would write those states to the wrong pages"
            )
        self._ckpt_cu_starts = torch.tensor(
            [0, *accumulate(counts)], dtype=torch.int64, device=self.device
        )

    def _update_block_table_from_scheduler(
        self,
        tasks: "PackedTasksBase",
        incremental: bool = False,
    ) -> None:
        """Apply scheduler allocations, then synchronize the active GPU rows.

        Owning both operations here makes ``incremental=True`` an append-only
        contract instead of trusting a caller-provided list of dirty block IDs.
        """
        task_ids = tasks.task_ids
        self._validate_gpu_block_table_batch(task_ids)

        new_cache_ids_list = tasks.new_cache_ids_list
        if new_cache_ids_list and len(new_cache_ids_list) != len(task_ids):
            raise ValueError(
                "new_cache_ids_list must be empty or match task_ids: "
                f"new_cache_ids={len(new_cache_ids_list)} task_ids={len(task_ids)}"
            )

        if new_cache_ids_list:
            for i, item in enumerate(tasks.new_cache_ids_list):
                new_cache_ids = item.get(self.manager_name, [])
                task_id = task_ids[i]

                # Update inplace_block_ids(store in-place state block ids)
                is_first_alloc = not self.inplace_block_ids.get(task_id, [])
                if is_first_alloc:
                    # 首次为task分配block时，会将inplace_block_ids放在new_cache_ids头部
                    assert (
                        len(new_cache_ids) >= self.mtp_size
                    ), f"len({new_cache_ids}) should bigger than mtp_size({self.mtp_size})."
                    self.inplace_block_ids[task_id] = new_cache_ids[: self.mtp_size]

                    # zero freshly-allocated blocks
                    for key in self.paged_kv_cache:
                        self.paged_kv_cache[key][:, self.inplace_block_ids[task_id]] = 0

                if not new_cache_ids:
                    continue

                # Update ckpt_block_ids(only store checkpoint block ids). 首次分配时
                # new_cache_ids = [spec | fresh ckpt]，spec 部分已计入 inplace_block_ids；
                # 其余情况下 new_cache_ids 只含 ckpt block ids
                new_ckpt_ids = (
                    new_cache_ids[self.mtp_size :] if is_first_alloc else new_cache_ids
                )
                new_length = (
                    len(self.ckpt_block_ids[task_id])
                    + len(self.inplace_block_ids[task_id])
                    + len(new_ckpt_ids)
                )
                assert new_length <= self.max_blocks_per_req, (
                    "block-table row exceeds configured per-request capacity: "
                    f"task_id={task_id} new_len={new_length} "
                    f"max_blocks_per_req={self.max_blocks_per_req}"
                )
                self.ckpt_block_ids[task_id].extend(new_ckpt_ids)

                # zero freshly-allocated blocks(exclude cached blocks).
                # checkpoint_interval 为 None 时请求没有 ckpt blocks，无需清零
                if self.checkpoint_interval is not None:
                    n_cached_tokens = self.seq_len_delta.old.lens_list[i]
                    # 只有被前缀完全覆盖的 ckpt block 才是「已算好的」，即 j <
                    # n_cached_tokens / C，故向下取整；用 ceil_div 等于隐含假设
                    # n_cached_tokens 一定落在 C 网格上，一旦不落在网格上（chunk 起点没对齐，
                    # 见 scheduler 的守卫）就会漏清零一个其实要写的 ckpt block
                    n_cached_blocks = n_cached_tokens // self.checkpoint_interval
                    fresh_blocks = self.ckpt_block_ids[task_id][n_cached_blocks:]
                    for key in self.paged_kv_cache:
                        self.paged_kv_cache[key][:, fresh_blocks] = 0

        self._upd_gpu_block_table(task_ids=task_ids)

    @override
    def prepare_cache_prefill(self, tasks: "PackedTasksBase"):
        # is_decode_stage 是 step 级状态：prefill 置 False、decode 置 True，保留到
        # 下一个 batch 覆盖为止（model 侧据此判断 MTP decode）
        self.is_decode_stage = False
        super().prepare_cache_prefill(tasks)

    @override
    def prepare_cache_decode(self, tasks: "PackedTasksBase"):
        KVCacheBase.prepare_cache_decode(self, tasks)
        self.is_decode_stage = True
        self._update_block_table_from_scheduler(tasks, incremental=True)

    @override
    def update_mtp_cache_accept(
        self, tasks: "PackedTasksBase", mtp_accept_indices: list[int]
    ):
        KVCacheBase.update_mtp_cache_accept(self, tasks, mtp_accept_indices)
        # 记下本 step 每个请求的接受长度对应的 in-place block 下标，供 _upd_read_page_ids
        # 用。只能在这里取：部分 rank 上 task.mtp_accept_index 是 -1，真正的值由
        # executor dispatch/broadcast 得到
        for tid, accept_index in zip(tasks.task_ids, mtp_accept_indices):
            assert -1 <= accept_index < self.mtp_size, (
                f"tid:{tid}, invalid mtp accept index {accept_index}, "
                f"expect in [-1, {self.mtp_size - 1}]"
            )
            self.tid_to_accept_index[tid] = accept_index
        # 刚应用的 accept 直到这里才记进tid_to_cached_len/tid_to_accept_index。
        # last stage 上是在 postprocess_generate_draft 里调该函数，紧接着
        # model.read_mtp_hidden_states(is_draft=False)拿 _read_page_ids 得到
        # draft 链的起点，此处需重算_read_page_ids
        if self.curr_tids:
            self._upd_read_page_ids(
                self.curr_tids,
                [self.tid_to_cached_len.get(tid, 0) for tid in self.curr_tids],
            )

    @override
    def finalize_cache_all_decode(self, tasks: "PackedTasksBase"):
        KVCacheBase.finalize_cache_all_decode(self, tasks)
        for tid in tasks.task_ids:
            self.tid_to_accept_index.pop(tid, None)
            self.inplace_block_ids.pop(tid, None)
            self.ckpt_block_ids.pop(tid, None)

    @override
    def get_accessor(
        self, layer_id: int, is_mtp: bool = False
    ) -> SingletonPagedKVCacheAccessor:
        local_layer_id = self.layer_id_map.to_local(layer_id)
        ret_kv = {
            key: cache[local_layer_id] for key, cache in self.paged_kv_cache.items()
        }
        return SingletonPagedKVCacheAccessor(
            kv=ret_kv,
            get_write_page_ids=lambda: self._write_page_ids.get(),
            get_read_page_ids=lambda: self._read_page_ids.get(),
            get_ckpt_write_pages=lambda: self._ckpt_write_pages,
            get_ckpt_cu_starts=lambda: self._ckpt_cu_starts,
            use_i64_offsets=self.use_i64_offsets,
        )

    # 本类没有基类那种 token → 物理页 的页表（见 init_metadata_buffer），下面这几个接口一律
    # 直接报错，而不是让调用方拿到一个别的东西（比如 ckpt block）当页表/页 id 用：
    has_token_block_table = False

    _PAGE_IDS_MSG = (
        "SingletonPagedKVCache has no token -> page mapping; use "
        "get_accessor().get_read_page_ids() / get_write_page_ids() instead"
    )
    _GPU_BLOCK_TABLE_MSG = (
        "SingletonPagedKVCache has no gpu_block_table: it does not map tokens to blocks, "
        "so there is no per-request block row table to hand to attention metadata; the "
        "page ids it does have come from get_accessor().get_read_page_ids() / "
        "get_write_page_ids()"
    )
    _BLOCK_TABLE_MSG = (
        "SingletonPagedKVCache has no block_table: it does not map tokens to blocks. A "
        "request here owns mtp_size in-place state blocks (inplace_block_ids) and some "
        "checkpoint blocks (ckpt_block_ids), neither of which is a token -> page mapping"
    )

    @property
    @override
    def page_ids(self):
        raise NotImplementedError(self._PAGE_IDS_MSG)

    @property
    @override
    def page_ids_mtp(self):
        raise NotImplementedError(self._PAGE_IDS_MSG)

    @property
    @override
    def offs_in_page(self):
        raise NotImplementedError(self._PAGE_IDS_MSG)

    @property
    @override
    def offs_in_page_mtp(self):
        raise NotImplementedError(self._PAGE_IDS_MSG)

    @override
    def get_gpu_block_table(self):
        raise NotImplementedError(self._GPU_BLOCK_TABLE_MSG)

    @property
    def block_table(self):
        raise NotImplementedError(self._BLOCK_TABLE_MSG)

    @block_table.setter
    def block_table(self, value):
        raise NotImplementedError(self._BLOCK_TABLE_MSG)

    @property
    def ckpt_cu_starts(self) -> Optional[torch.Tensor]:
        """本 step 每个 seq 有几个 checkpoint 的累加计数（int64, [bsz + 1]），用于 chunk linear_attn/conv算子。
        本 step 没有 checkpoint 要写时为 None（不设 checkpoint_interval、decode、
        本 step 没有新增 token）：decode 的 checkpoint 由 ``_update_write_page_ids`` 把写页
        直接重定向到 ckpt block，不需要算子额外输出。
        """
        return self._ckpt_cu_starts

    @override
    def insert_kv_cache_from_transfer(
        self, tid: str, inplace_page_ids: Sequence[int], prefix_length: int
    ):
        """
        Register the in-place state pages transferred from the prefill instance.

        Decode only needs the state at the end of the prompt, which lives in the in-place
        pages, so ckpt pages are never transferred: a decode-only instance consumes no
        prefix-cache hit and never rolls back further than the mtp window. ``ckpt_block_ids``
        (ckpt blocks) is therefore left untouched and stays empty until this instance
        allocates its own, which is why this cache must not be configured with a
        ``checkpoint_interval`` (no linear prefix caching on the decode side).
        Assumes the state has been copied into these pages via RDMA.
        """
        assert self.checkpoint_interval is None, (
            f"tid:{tid}: this cache has checkpoint_interval="
            f"{self.checkpoint_interval}, but the transferred state only covers the "
            "in-place pages; a cache that keeps checkpoints must receive them too"
        )
        inplace_page_ids = [int(x) for x in inplace_page_ids]
        assert len(inplace_page_ids) == self.mtp_size, (
            f"tid:{tid}, expect {self.mtp_size} in-place pages, "
            f"got {len(inplace_page_ids)}: {inplace_page_ids}"
        )
        for idx in inplace_page_ids:
            assert 0 <= idx < self.num_blocks, f"invalid page index: {idx}"
        # 必须在首次 _upd_gpu_block_table/清零之前登记，否则会把传过来的 state 清掉
        self.inplace_block_ids[tid] = inplace_page_ids
        self.tid_to_cached_len[tid] = int(prefix_length)


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
            use_prefix_lens_static_tensor=True,
            use_position_ids_static_tensor=True,
            use_seq_ids_static_tensor=True,
            use_delta_position_ids_static_tensor=True,
            use_delta_seq_ids_static_tensor=True,
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
            use_prefix_lens_static_tensor=False,
            use_position_ids_static_tensor=False,
            use_seq_ids_static_tensor=False,
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
            use_prefix_lens_static_tensor=False,
            use_position_ids_static_tensor=False,
            use_seq_ids_static_tensor=False,
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
    def prepare_mtp_cache_decode(self, tasks: "PackedTasksBase", offset: int):
        self.update_page_offs()

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
            max_alloc_seq_len(self.max_seq_len)
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
        max_blocks_per_req = ceil_div(self.window_size, self.window_size)
        super().__init__(
            *args,
            num_hot_req=num_hot_req,
            max_seq_len=max_seq_len,
            num_blocks=fixed_num_blocks,
            max_blocks_per_req=max_blocks_per_req,
            block_size=self.window_size,
            **kwargs,
        )

        # The logical sequence can be longer than the window, but the page table
        # has exactly one entry per request.
        self.allocatable_max_num_blocks = self.max_num_blocks
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
