# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Sequence, Optional
from typing_extensions import override
from logging import getLogger
import torch
from collections import deque

from chitu.global_vars import get_slot_handle, get_timers, get_global_args
from chitu.static_tensor import StaticTensor
from chitu.batched_seq_len import BatchedSeqLen

logger = getLogger(__name__)


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
        self.prev_seq_len = BatchedSeqLen([], self.device, max_batch_size=num_hot_req)
        self.next_seq_len = BatchedSeqLen([], self.device, max_batch_size=num_hot_req)

        self.curr_req_ids: Optional[List[str]] = None

        self.timers = get_timers()

    def prepare_cache_prefill(self, req_ids: List[str], next_seq_len: BatchedSeqLen):
        self.next_seq_len.copy_from(next_seq_len)
        self.curr_req_ids = req_ids

    def finalize_cache_bylayer_prefill(
        self,
        xk: Optional[torch.Tensor],
        xv: Optional[torch.Tensor],
        req_ids: List[str],
        next_seq_len: BatchedSeqLen,
        layer_id: int,
    ):
        pass

    def finalize_cache_all_prefill(self):
        self.curr_req_ids = None

    def prepare_cache_decode(self, req_ids: List[str]):
        self.curr_req_ids = req_ids

        self.prev_seq_len.copy_from(
            BatchedSeqLen(
                [self.req_id_to_seq_len[req_id] for req_id in req_ids],
                device=self.device,
            )
        )
        self.next_seq_len.copy_from(
            BatchedSeqLen(
                [self.req_id_to_seq_len[req_id] + 1 for req_id in req_ids],
                device=self.device,
            )
        )

    def finalize_cache_single_decode(self, req_ids: List[str]):
        for req_id in req_ids:
            self.req_id_to_seq_len[req_id] += 1
        self.curr_req_ids = None

    def finalize_cache_all_decode(self, req_id: str):
        pass

    def get_gpu_block_table(self):
        return None

    def get_block_size(self):
        return 0


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

        self.max_blocks_per_req = (max_seq_len + block_size - 1) // block_size
        self.max_num_blocks = self.max_blocks_per_req * num_hot_req
        self.num_blocks = num_blocks if num_blocks != -1 else num_hot_req

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
        return self.block_size

    def get_num_blocks(self):
        return self.max_num_blocks

    @override
    def prepare_cache_prefill(self, req_ids: List[str], next_seq_len: BatchedSeqLen):
        super().prepare_cache_prefill(req_ids, next_seq_len)

        block_idxs = []
        indices_in_block = []
        for req_id, seq_len in zip(req_ids, next_seq_len.lens_list):
            # 设置其他函数会用到的变量
            self.req_id_to_seq_len[req_id] = seq_len

            # 为请求分配blocks
            num_blocks_prepared = (seq_len + self.block_size - 1) // self.block_size
            block_ids = [self.get_free_block() for _ in range(num_blocks_prepared)]
            self.block_table[req_id] = block_ids

            # 计算每个元素对应的block id 和 block内的索引
            num_full_block, remainder = divmod(seq_len, self.block_size)
            if num_full_block > 0:
                block_idxs.extend(
                    [
                        block_id
                        for block_id in block_ids[:num_full_block]
                        for _ in range(self.block_size)
                    ]
                )
                indices_in_block.extend(
                    [i for i in range(self.block_size) for _ in range(num_full_block)]
                )
            if remainder > 0:
                block_idxs.extend([block_ids[num_full_block]] * remainder)
                indices_in_block.extend([i for i in range(remainder)])

        # 在不同layer间共享
        self.new_tokens_block_indices = torch.tensor(
            block_idxs,
            dtype=torch.int32,
            device=self.device,
        )
        self.new_tokens_indices_in_block = torch.tensor(
            indices_in_block,
            dtype=torch.int32,
            device=self.device,
        )

    # Init block table and kv cache with kv generated during prefill
    @override
    def finalize_cache_bylayer_prefill(
        self,
        xk: Optional[torch.Tensor],
        xv: Optional[torch.Tensor],
        req_ids: List[str],
        next_seq_len: BatchedSeqLen,
        layer_id: int,
    ):
        self.timers("finalize_cache_bylayer_prefill").start()

        layer_idx = layer_id - self.begin_layer_id

        if (
            get_global_args().infer.attn_type == "npu"
            and self.k_shape_per_sample is not None
            and len(self.k_shape_per_sample) == 1
            and get_global_args().models.type != "deepseek-v3"
        ):
            # NPU BSH layout
            xk = xk.view(xk.shape[0], -1).contiguous() if xk is not None else None
            xv = xv.view(xv.shape[0], -1).contiguous() if xv is not None else None

        if xk is not None:
            assert self.paged_k_cache is not None
            self.paged_k_cache[layer_idx].index_put_(
                (self.new_tokens_block_indices, self.new_tokens_indices_in_block), xk
            )
        if xv is not None:
            assert self.paged_v_cache is not None
            self.paged_v_cache[layer_idx].index_put_(
                (self.new_tokens_block_indices, self.new_tokens_indices_in_block), xv
            )

        self.timers("finalize_cache_bylayer_prefill").stop()

    @override
    def prepare_cache_decode(self, req_ids: List[str]):
        super().prepare_cache_decode(req_ids)

        # Prepare enough block table for next decoding. When decoding, AttnBackend will fill new kv into
        # paged kv cache in place.
        for i, req_id in enumerate(req_ids):
            if self.prev_seq_len.lens_list[i] % self.block_size == 0:
                self.block_table[req_id].append(self.get_free_block())

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

    def get_free_block(self):
        # TODO: When run out of free blocks, use scheduling and preemption in paper instead of exception
        self.timers("get_free_block").start()
        if len(self.free_blocks) == 0:
            raise Exception("No more free blocks.")
        idx = self.free_blocks.popleft()
        self.timers("get_free_block").stop()
        return idx

    @override
    def get_gpu_block_table(self):
        return self.gpu_block_table.get()

    def get_paged_kv_cache(self, layer_id: int):
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
        return ret_k, ret_v

    def free_req_cache_blocks(self, req_id: str):
        self.timers("free_req_cache_blocks").start()
        for block in self.block_table[req_id]:
            self.free_blocks.append(block)
        del self.block_table[req_id]
        del self.req_id_to_seq_len[req_id]
        self.timers("free_req_cache_blocks").stop()

    @override
    def finalize_cache_all_decode(self, req_id: str):
        self.timers("finalize_cache_all_decode").start()
        if req_id not in self.req_id_to_seq_len:
            return
        # assert req_id in self.req_id_to_seq_len
        # assert req_id in self.block_table
        self.free_req_cache_blocks(req_id)
        self.timers("finalize_cache_all_decode").stop()


class KVCacheManagerSkewAware(KVCacheManagerBase):
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

        self.prepared_reqs: List[str] = []
        self.rounded_max_seq = -1
        self.slot_handle = get_slot_handle()

    @override
    def prepare_cache_prefill(self, req_ids: List[str], next_seq_len: BatchedSeqLen):
        super().prepare_cache_prefill(req_ids, next_seq_len)

        if self.slot_handle:
            start_idx, _ = self.slot_handle.get_current_slot_start_end_idx()
        else:
            start_idx = 0
        for it, req_id in enumerate(req_ids):
            self.req_id_to_seq_len[req_id] = next_seq_len.lens_list[it]
            for i in range(start_idx, self.num_hot_req):
                if self.slot_availability[i]:
                    self.req2slot[req_id] = i
                    self.slot_availability[i] = False
                    self.hot_reqs[i] = req_id
                    break
            assert (
                req_id in self.req2slot
            ), f"Cannot allocate slot: {req_id} {self.req2slot}"

    # Prefill:
    @override
    def finalize_cache_bylayer_prefill(
        self,
        xk: Optional[torch.Tensor],
        xv: Optional[torch.Tensor],
        req_ids: List[str],
        next_seq_len: BatchedSeqLen,
        layer_id: int,
    ):
        self.timers("finalize_cache_bylayer_prefill").start()

        if (
            get_global_args().infer.attn_type == "npu"
            and self.k_shape_per_sample is not None
            and len(self.k_shape_per_sample) == 1
        ):
            # NPU BSH layout
            xk = xk.view(xk.shape[0], -1).contiguous() if xk is not None else None
            xv = xv.view(xv.shape[0], -1).contiguous() if xv is not None else None

        start = 0
        for it, req_id in enumerate(req_ids):
            end = start + next_seq_len.lens_list[it]
            if xk is not None:
                assert self.k_buffer is not None
                self.k_buffer[layer_id - self.begin_layer_id][self.req2slot[req_id]][
                    : next_seq_len.lens_list[it]
                ] = xk[start:end]
            if xv is not None:
                assert self.v_buffer is not None
                self.v_buffer[layer_id - self.begin_layer_id][self.req2slot[req_id]][
                    : next_seq_len.lens_list[it]
                ] = xv[start:end]
            start = end

        self.timers("finalize_cache_bylayer_prefill").stop()

    # Decode:
    @override
    def prepare_cache_decode(self, req_ids: List[str]):
        self.timers("cache_prepare").start()

        super().prepare_cache_decode(req_ids)
        max_seq = self.prev_seq_len.max_len

        args = get_global_args()
        if args.infer.pp_size == 1:  # Non-PP
            self.k_prepared_cache = (
                None if self.k_buffer is None else self.k_buffer[:, : len(req_ids)]
            )
            self.v_prepared_cache = (
                None if self.v_buffer is None else self.v_buffer[:, : len(req_ids)]
            )

        else:  # PP
            if args.infer.use_cuda_graph:
                raise NotImplementedError(
                    "Setting infer.cache_type=skew and infer.use_cuda_graph=True "
                    "simultaneously is not supported when using pipeline parallelism"
                )

            start_pos = self.hot_reqs.index(req_ids[0])
            assert start_pos + len(req_ids) <= self.num_hot_req

            limit = 16
            rounded_max_seq = (max_seq + 1 + limit - 1) // limit * limit
            if (
                self.rounded_max_seq >= rounded_max_seq
                and self.prepared_reqs == req_ids
            ):
                # prepared cache is long enough
                self.timers("cache_prepare").stop()
                return

            self.rounded_max_seq = rounded_max_seq
            self.prepared_reqs = req_ids

            if self.k_buffer is not None:
                k_prepared_cache_shape = list(self.k_buffer.shape)
                k_prepared_cache_stride = list(self.k_buffer.stride())
                k_prepared_cache_stride[0] = (
                    k_prepared_cache_shape[1] * k_prepared_cache_stride[1]
                )
                k_prepared_cache_shape[1] = len(req_ids)
                k_prepared_cache_shape[2] = rounded_max_seq
                k_prepared_cache_offset = start_pos * k_prepared_cache_stride[1]
                self.k_prepared_cache = torch.as_strided(
                    self.k_buffer,
                    k_prepared_cache_shape,
                    k_prepared_cache_stride,
                    k_prepared_cache_offset,
                )
            else:
                self.k_prepared_cache = None

            if self.v_buffer is not None:
                v_prepared_cache_shape = list(self.v_buffer.shape)
                v_prepared_cache_stride = list(self.v_buffer.stride())
                v_prepared_cache_stride[0] = (
                    v_prepared_cache_shape[1] * v_prepared_cache_stride[1]
                )
                v_prepared_cache_shape[1] = len(req_ids)
                v_prepared_cache_shape[2] = rounded_max_seq
                v_prepared_cache_offset = start_pos * v_prepared_cache_stride[1]
                self.v_prepared_cache = torch.as_strided(
                    self.v_buffer,
                    v_prepared_cache_shape,
                    v_prepared_cache_stride,
                    v_prepared_cache_offset,
                )
            else:
                self.v_prepared_cache = None

        self.timers("cache_prepare").stop()

    # Decode:
    # return [2, num_req, max_seqlen + 1, n_local_kv_heads, head_dim]
    def get_cache_decode(self, layer_id: int):
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
        return ret_k, ret_v

    # Decode:
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
