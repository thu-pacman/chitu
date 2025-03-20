from logging import getLogger

import torch

from chitu.global_vars import get_slot_handle, get_timers, get_global_args
from chitu.static_tensor import StaticTensor

logger = getLogger(__name__)
_BLOCK_SIZE = 512  # _BLOCK_SIZE must be a multiple of 256 for FlashAttention
_MAX_SEQ_LEN = 2048


class PagedKVCacheManager:
    def __init__(
        self,
        begin_layer_id,
        end_layer_id,
        num_hot_req=16,
        block_size=_BLOCK_SIZE,
        max_seq_len=_MAX_SEQ_LEN,
        device="cuda",
        *,
        k_shape_per_sample=None,
        v_shape_per_sample=None,
        kv_shape_per_sample=None,
        n_local_kv_heads=None,
        head_dim=None,
    ):
        """
        You can optionally set `k_shae_per_sample` and `v_shape_per_sample`, or `n_local_kv_heads` and `head_dim`.
        """

        self.max_blocks_per_req = max_seq_len // block_size + 1
        self.num_blocks = self.max_blocks_per_req * num_hot_req

        self.begin_layer_id = begin_layer_id
        self.end_layer_id = end_layer_id
        self.num_layers = end_layer_id - begin_layer_id
        self.k_shape_per_sample = (
            k_shape_per_sample
            if k_shape_per_sample is not None
            else (n_local_kv_heads, head_dim)
        )
        self.v_shape_per_sample = (
            v_shape_per_sample
            if v_shape_per_sample is not None
            else (n_local_kv_heads, head_dim)
        )
        self.kv_shape_per_sample = kv_shape_per_sample
        # assert block_size % 256 == 0
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)

        self.seq_lens = {}
        self.timers = get_timers()
        self.block_table = {}  # (seq_id, block_idx)
        self.curr_seq_lens_gpu_excl_this_decode = StaticTensor(
            max_nelem=num_hot_req, dtype=torch.int32, device=self.device
        )
        self.curr_seq_lens_gpu_incl_this_decode = StaticTensor(
            max_nelem=num_hot_req, dtype=torch.int32, device=self.device
        )
        self.gpu_block_table = StaticTensor(
            max_nelem=self.num_blocks, dtype=torch.int32, device=self.device
        )
        # TODO: For better performance, use list instead of set for free_blocks
        self.free_blocks = set(range(self.num_blocks))
        if self.kv_shape_per_sample is not None:
            self.paged_kv_cache = torch.zeros(
                (self.num_layers, self.num_blocks, block_size)
                + self.kv_shape_per_sample,
                device=device,
            )
        else:
            self.paged_k_cache = torch.zeros(
                (self.num_layers, self.num_blocks, block_size)
                + self.k_shape_per_sample,
                device=device,
            )
            self.paged_v_cache = torch.zeros(
                (self.num_layers, self.num_blocks, block_size)
                + self.v_shape_per_sample,
                device=device,
            )

    def get_block_size(self):
        return self.block_size

    def get_num_blocks(self):
        return self.num_blocks

    # Init block table and kv cache with kv generated during prefill
    def finalize_cache_bylayer_prefill(self, xk, xv, req_ids, varlen, layer_id):
        self.timers("cache_finalize_cache_all_prefill").start()
        for idx, req_id in enumerate(req_ids):
            num_blocks_prepared = (
                varlen.cpu_lens[idx] + self.block_size - 1
            ) // self.block_size

            if layer_id == self.begin_layer_id:
                self.seq_lens[req_id] = varlen.cpu_lens[idx]
                block_ids = []
                for chunck_id in range(num_blocks_prepared):
                    block_idx = self.get_free_block()
                    block_ids.append(block_idx)
                self.block_table[req_id] = block_ids
            else:
                block_ids = self.block_table[req_id]

            start_pos = varlen.cpu_prefix_lens[idx]
            end_pos = varlen.cpu_prefix_lens[idx + 1]
            for chunck_id in range(num_blocks_prepared):
                block_idx = block_ids[chunck_id]
                if self.kv_shape_per_sample is not None:
                    if chunck_id != num_blocks_prepared - 1:
                        self.paged_kv_cache[layer_id - self.begin_layer_id][
                            block_idx
                        ] = xk[start_pos : (start_pos + self.block_size)].clone()
                        start_pos += self.block_size
                    else:
                        tmp_len = end_pos - start_pos
                        self.paged_kv_cache[layer_id - self.begin_layer_id][block_idx][
                            :tmp_len
                        ] = xk[start_pos:end_pos].clone()
                else:
                    if chunck_id != num_blocks_prepared - 1:
                        self.paged_k_cache[layer_id - self.begin_layer_id][
                            block_idx
                        ] = xk[start_pos : (start_pos + self.block_size)].clone()
                        self.paged_v_cache[layer_id - self.begin_layer_id][
                            block_idx
                        ] = xv[start_pos : (start_pos + self.block_size)].clone()
                        start_pos += self.block_size
                    else:
                        tmp_len = end_pos - start_pos
                        self.paged_k_cache[layer_id - self.begin_layer_id][block_idx][
                            :tmp_len
                        ] = xk[start_pos:end_pos].clone()
                        self.paged_v_cache[layer_id - self.begin_layer_id][block_idx][
                            :tmp_len
                        ] = xv[start_pos:end_pos].clone()
        self.timers("cache_finalize_cache_all_prefill").stop()

    def finalize_cache_all_prefill(self, req_ids, varlen):
        self.curr_varlens = None
        self.curr_req_ids = None

    def prepare_cache_decode(self, req_ids):
        seq_lens = []
        for req_id in req_ids:
            seq_len = self.seq_lens[req_id]
            seq_lens.append(seq_len)
        max_seq = max(seq_lens)
        self.curr_seq_lens = seq_lens
        self.curr_seq_lens_gpu_excl_this_decode.set(
            torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
        )
        self.curr_seq_lens_gpu_incl_this_decode.set(
            self.curr_seq_lens_gpu_excl_this_decode.get() + 1
        )

    def get_free_block(self):
        # TODO: When run out of free blocks, use scheduling and preemption in paper instead of exception
        self.timers("get_free_block").start()
        if len(self.free_blocks) == 0:
            raise Exception("No more free blocks.")
        idx = list(self.free_blocks)[0]
        self.free_blocks.remove(idx)
        self.timers("get_free_block").stop()
        return idx

    def get_gpu_block_table(self):
        return self.gpu_block_table.get()

    def get_gpu_seq_lens_excl_this_decode(self):
        return self.curr_seq_lens_gpu_excl_this_decode.get()

    def get_gpu_seq_lens_incl_this_decode(self):
        return self.curr_seq_lens_gpu_incl_this_decode.get()

    def get_paged_kv_cache(self, layer_id):
        if self.kv_shape_per_sample is not None:
            return self.paged_kv_cache[layer_id - self.begin_layer_id]
        else:
            return (
                self.paged_k_cache[layer_id - self.begin_layer_id],
                self.paged_v_cache[layer_id - self.begin_layer_id],
            )

    def free_req_cache_blocks(self, req_id):
        self.timers("free_req_cache_blocks").start()
        for block in self.block_table[req_id]:
            self.free_blocks.add(block)
        del self.block_table[req_id]
        self.timers("free_req_cache_blocks").stop()

    # Prepare enough block table for next decoding. When decoding, flash attention will fill new kv into paged kv cache (inplace).
    def prepare_block_table_for_decode(self, req_ids):
        for req_id in req_ids:
            if self.seq_lens[req_id] % self.block_size == 0:
                self.block_table[req_id].append(self.get_free_block())

        if get_global_args().infer.use_cuda_graph:
            max_block_num = self.max_blocks_per_req
        else:
            max_block_num = max(len(self.block_table[req_id]) for req_id in req_ids)
        self.gpu_block_table.set(
            torch.zeros((len(req_ids), max_block_num), dtype=torch.int32, device="cuda")
        )
        for idx, req_id in enumerate(req_ids):
            block_ids = self.block_table[req_id]
            self.gpu_block_table.get()[idx, : len(block_ids)] = torch.tensor(
                block_ids, dtype=torch.int32
            )

    def finalize_cache_single_decode(self, req_ids):
        for req_id in req_ids:
            self.seq_lens[req_id] = self.seq_lens[req_id] + 1
        self.curr_varlens = None
        self.curr_req_ids = None

    def finalize_cache_all_decode(self, req_id):
        self.timers("finalize_cache_all_decode").start()
        assert req_id in self.seq_lens
        assert req_id in self.block_table
        del self.seq_lens[req_id]
        self.free_req_cache_blocks(req_id)
        self.curr_varlens = None
        self.curr_req_ids = None
        self.timers("finalize_cache_all_decode").stop()


class KVCacheManager:
    def __init__(
        self, begin_layer_id, end_layer_id, n_local_kv_heads, head_dim, device="cuda"
    ):
        self.cache = {}
        self.prepared_cache = []
        self.begin_layer_id = begin_layer_id
        self.end_layer_id = end_layer_id
        self.num_layers = end_layer_id - begin_layer_id
        self.n_local_kv_heads = n_local_kv_heads
        self.head_dim = head_dim
        self.tmp_storage = []
        self.seq_lens = {}
        self.timers = get_timers()
        self.device = torch.device(device)

    # Prefill:
    def finalize_cache_bylayer_prefill(
        self, cache_k, cache_v, req_ids, varlen, layer_id
    ):
        self.tmp_storage.append([cache_k, cache_v])

    # Prefill:
    # return for every req [layer, seq, n_local_kv_heads, head_dim] * 2 (for k and v)
    def finalize_cache_all_prefill(self, req_ids, varlen):
        self.timers("cache_finalize_cache_all_prefill").start()
        assert (
            len(self.tmp_storage) == self.num_layers
        ), f"{len(self.tmp_storage)} {self.num_layers}"
        assert len(varlen.cpu_lens) == len(req_ids)
        assert sum(varlen.cpu_lens) == self.tmp_storage[0][0].shape[0]
        for it, req_id in enumerate(req_ids):
            self.seq_lens[req_id] = varlen.cpu_lens[it]
            self.cache[req_id] = [None] * self.num_layers
        for local_layer_id in range(self.num_layers):
            start = 0
            for it, req_id in enumerate(req_ids):
                end = start + varlen.cpu_lens[it]
                self.cache[req_id][local_layer_id] = [
                    self.tmp_storage[local_layer_id][0][
                        start:end
                    ],  # [seq, n_local_kv_heads, head_dim]
                    self.tmp_storage[local_layer_id][1][
                        start:end
                    ],  # [seq, n_local_kv_heads, head_dim]
                ]
                start = end
        self.tmp_storage = []
        self.curr_varlens = None
        self.curr_req_ids = None
        self.timers("cache_finalize_cache_all_prefill").stop()

    # Decode:
    # return [layer, num_req, 2, max_seqlen + 1, n_local_kv_heads, head_dim]
    def prepare_cache_decode(self, req_ids):
        self.timers("cache_prepare").start()
        max_seq = 0
        seq_lens = []
        for req_id in req_ids:
            seq_len = self.seq_lens[req_id]
            seq_lens.append(seq_len)
        self.curr_seq_lens = seq_lens
        self.curr_seq_lens_gpu_excl_this_decode = torch.tensor(
            seq_lens, dtype=torch.int32, device=self.device
        )
        self.curr_seq_lens_gpu_incl_this_decode = (
            self.curr_seq_lens_gpu_excl_this_decode + 1
        )
        max_seq = max(seq_lens)
        n_local_kv_heads = self.cache[req_ids[0]][0][0].shape[-2]
        head_dim = self.cache[req_ids[0]][0][0].shape[-1]
        prepared_cache = torch.zeros(
            [
                self.num_layers,  # layers
                2,
                len(req_ids),  # batch_size
                max_seq + 1,  # seq_len
                n_local_kv_heads,  # n_local_kv_heads
                head_dim,  # head_dim
            ],
            device=self.device,
        )
        # hkz-comment: Very similar to matrix transpose;
        for local_layer_id in range(self.num_layers):
            for it, req_id in enumerate(req_ids):
                prepared_cache[local_layer_id][0][it][: seq_lens[it]] = self.cache[
                    req_id
                ][local_layer_id][0]
                prepared_cache[local_layer_id][1][it][: seq_lens[it]] = self.cache[
                    req_id
                ][local_layer_id][1]
        self.prepared_cache = prepared_cache
        self.timers("cache_prepare").stop()

    # Decode:
    # return [2, num_req, max_seqlen + 1, n_local_kv_heads, head_dim]
    def get_cache_decode(self, layer_id):
        return self.prepared_cache[layer_id - self.begin_layer_id]

    def get_gpu_block_table(self):
        return None

    def get_block_size(self):
        return 0

    # Decode:
    # return [# of current req_ids]
    def get_gpu_seq_lens_excl_this_decode(self):
        return self.curr_seq_lens_gpu_excl_this_decode

    # Decode:
    # return [# of current req_ids]
    def get_gpu_seq_lens_incl_this_decode(self):
        return self.curr_seq_lens_gpu_incl_this_decode

    # Decode:
    # return for every req [layer, seq + 1, n_local_kv_heads, head_dim] * 2 (for k and v)
    def finalize_cache_single_decode(self, req_ids):
        self.timers("cache_finalize_cache_single_decode").start()
        assert len(self.prepared_cache) > 0
        for it, req_id in enumerate(req_ids):
            self.cache[req_id] = [None] * self.num_layers
            self.seq_lens[req_id] += 1
        for local_layer_id in range(self.num_layers):
            for it, req_id in enumerate(req_ids):
                self.cache[req_id][local_layer_id] = [
                    self.prepared_cache[local_layer_id][0][it][
                        : self.curr_seq_lens[it] + 1
                    ],  # [seq + 1, n_local_kv_heads, head_dim]
                    self.prepared_cache[local_layer_id][1][it][
                        : self.curr_seq_lens[it] + 1
                    ],  # [seq + 1, n_local_kv_heads, head_dim]
                ]
        self.prepared_cache = []
        self.timers("cache_finalize_cache_single_decode").stop()
        self.curr_varlens = None
        self.curr_req_ids = None

    # Decode:
    def finalize_cache_all_decode(self, req_id):
        self.curr_varlens = None
        self.curr_req_ids = None
        pass


class KVCache:
    def __init__(self):
        self.cache_k = None
        self.cache_v = None
        self.inited = False

    def check_shape(self, cache):
        # (bsz * seqlen, self.n_local_heads, self.head_dim)
        assert len(cache.shape) == 3, cache.shape
        assert cache.device != torch.device("cpu")

    def check_shapes(self, cache_k, cache_v):
        self.check_shape(cache_k)
        self.check_shape(cache_v)
        assert cache_k.shape == cache_v.shape

    def init(self, cache_k, cache_v):
        assert not self.inited
        # self.check_shapes(cache_k, cache_v)

        self.cache_k = cache_k
        self.cache_v = cache_v
        self.inited = True

    def extend(self, new_cache_k, new_cache_v):
        assert self.inited
        self.cache_k = torch.cat([self.cache_k, new_cache_k], dim=0)
        self.cache_v = torch.cat([self.cache_v, new_cache_v], dim=0)


class KVCacheManagerSkewAware:
    def __init__(
        self,
        begin_layer_id,
        end_layer_id,
        num_hot_req,
        max_seq_len=2048,
        device="cuda",
        *,
        k_shape_per_sample=None,
        v_shape_per_sample=None,
        n_local_kv_heads=None,
        head_dim=None,
    ):
        """
        You can optionally set `k_shae_per_sample` and `v_shape_per_sample`, or `n_local_kv_heads` and `head_dim`.
        """

        self.begin_layer_id = begin_layer_id
        self.end_layer_id = end_layer_id
        self.num_layers = end_layer_id - begin_layer_id
        self.k_shape_per_sample = (
            k_shape_per_sample
            if k_shape_per_sample is not None
            else (n_local_kv_heads, head_dim)
        )
        self.v_shape_per_sample = (
            v_shape_per_sample
            if v_shape_per_sample is not None
            else (n_local_kv_heads, head_dim)
        )
        self.num_hot_req = num_hot_req
        self.slot_availability = [True] * num_hot_req
        self.hot_reqs = [-1] * num_hot_req
        self.req2slot = {}
        self.seq_lens = {}
        self.max_seq_len = max_seq_len
        self.tmp_storage = []
        self.device = torch.device(device)
        self.k_buffer = torch.zeros(
            (
                self.num_layers,
                self.num_hot_req,
                self.max_seq_len,
            )
            + self.k_shape_per_sample,
            device=self.device,
        )
        self.v_buffer = torch.zeros(
            (
                self.num_layers,
                self.num_hot_req,
                self.max_seq_len,
            )
            + self.v_shape_per_sample,
            device=self.device,
        )
        self.timers = get_timers()
        self.prepared_reqs = []
        self.rounded_max_seq = -1
        self.slot_handle = get_slot_handle()

    # Prefill:
    # return for every req [layer, seq, n_local_kv_heads, head_dim] * 2 (for k and v)
    def finalize_cache_bylayer_prefill(
        self, cache_k, cache_v, req_ids, varlen, layer_id
    ):
        self.timers("cache_finalize_cache_all_prefill").start()
        if self.slot_handle:
            start_idx, _ = self.slot_handle.get_current_slot_start_end_idx()
        else:
            start_idx = 0
        if layer_id == self.begin_layer_id:
            for it, req_id in enumerate(req_ids):
                self.seq_lens[req_id] = varlen.cpu_lens[it]
                for i in range(start_idx, self.num_hot_req):
                    if self.slot_availability[i]:
                        self.req2slot[req_id] = i
                        self.slot_availability[i] = False
                        self.hot_reqs[i] = req_id
                        break
                assert (
                    req_id in self.req2slot
                ), f"Cannot allocate slot: {req_id} {self.req2slot}"

        start = 0
        for it, req_id in enumerate(req_ids):
            end = start + varlen.cpu_lens[it]
            self.k_buffer[layer_id - self.begin_layer_id][self.req2slot[req_id]][
                : varlen.cpu_lens[it]
            ] = cache_k[start:end]
            self.v_buffer[layer_id - self.begin_layer_id][self.req2slot[req_id]][
                : varlen.cpu_lens[it]
            ] = cache_v[start:end]
            start = end
        self.timers("cache_finalize_cache_all_prefill").stop()

    # Prefill:
    def finalize_cache_all_prefill(self, req_ids, varlen):
        pass

    # Decode:
    def prepare_cache_decode(self, req_ids):
        self.timers("cache_prepare").start()
        start_pos = self.hot_reqs.index(req_ids[0])
        assert start_pos + len(req_ids) <= self.num_hot_req
        # assert (
        #     self.hot_reqs[start_pos : start_pos + len(req_ids)] == req_ids
        # ), f"{self.hot_reqs} {req_ids}"

        seq_lens = []
        for req_id in req_ids:
            seq_len = self.seq_lens[req_id]
            seq_lens.append(seq_len)
        max_seq = max(seq_lens)
        self.curr_seq_lens = seq_lens
        self.curr_seq_lens_gpu_excl_this_decode = torch.tensor(
            seq_lens, dtype=torch.int32, device=self.device
        )
        self.curr_seq_lens_gpu_incl_this_decode = (
            self.curr_seq_lens_gpu_excl_this_decode + 1
        )

        limit = 16
        rounded_max_seq = (max_seq + 1 + limit - 1) // limit * limit
        if self.rounded_max_seq >= rounded_max_seq and self.prepared_reqs == req_ids:
            # prepared cache is long enough
            self.timers("cache_prepare").stop()
            return

        self.rounded_max_seq = rounded_max_seq
        self.prepared_reqs = req_ids

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

        self.timers("cache_prepare").stop()

    # Decode:
    # return [2, num_req, max_seqlen + 1, n_local_kv_heads, head_dim]
    def get_cache_decode(self, layer_id):
        return (
            self.k_prepared_cache[layer_id - self.begin_layer_id],
            self.v_prepared_cache[layer_id - self.begin_layer_id],
        )

    def get_gpu_block_table(self):
        return None

    def get_block_size(self):
        return 0

    # Decode:
    # return [# of current req_ids]
    def get_gpu_seq_lens_excl_this_decode(self):
        return self.curr_seq_lens_gpu_excl_this_decode

    # Decode:
    # return [# of current req_ids]
    def get_gpu_seq_lens_incl_this_decode(self):
        return self.curr_seq_lens_gpu_incl_this_decode

    # Decode:
    def finalize_cache_single_decode(self, req_ids):
        for item in req_ids:
            self.seq_lens[item] += 1
        self.curr_varlens = None
        self.curr_req_ids = None

    # Decode:
    def finalize_cache_all_decode(self, req_id):
        slot_id = self.hot_reqs.index(req_id)
        if slot_id == -1:  # not in the hot slot
            return

        if self.slot_handle:
            # get end_idx in req_id slot
            end_idx = 0
            slot_end_idx = self.slot_handle.slot_end_idx
            for idx in slot_end_idx:
                if slot_id < idx:
                    end_idx = idx
                    break
            assert end_idx > slot_id, f"get the wrong id in skewkvcache"
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
            self.k_buffer[:, slot_id] = self.k_buffer[:, slot_last_id]
            self.v_buffer[:, slot_id] = self.v_buffer[:, slot_last_id]
            req_key = next(
                (k for k, v in self.req2slot.items() if v == slot_last_id), None
            )
            if req_key is not None:
                self.req2slot[req_key] = slot_id
                self.hot_reqs[slot_id] = req_key
            self.hot_reqs[slot_last_id] = -1
            self.slot_availability[slot_last_id] = True
            if req_id in self.req2slot:
                self.req2slot.pop(req_id)
            self.k_buffer[:, slot_last_id].zero_()
            self.v_buffer[:, slot_last_id].zero_()
        else:
            self.hot_reqs[slot_id] = -1
            self.slot_availability[slot_id] = True
            self.req2slot.pop(req_id)
            self.k_buffer[:, slot_id].zero_()
            self.v_buffer[:, slot_id].zero_()


class KVCacheManagerNop:
    def __init__(
        self,
        begin_layer_id,
        end_layer_id,
        n_local_kv_heads,
        head_dim,
        num_hot_req=16,
        max_seq_len=2048,
        device="cuda",
    ):
        self.begin_layer_id = begin_layer_id
        self.end_layer_id = end_layer_id
        self.num_layers = end_layer_id - begin_layer_id
        self.n_local_kv_heads = n_local_kv_heads
        self.head_dim = head_dim
        self.num_hot_req = num_hot_req
        self.slot_availability = [True] * num_hot_req
        self.hot_reqs = [-1] * num_hot_req
        self.req2slot = {}
        self.max_seq_len = max_seq_len
        self.tmp_storage = []
        self.device = torch.device(device)
        self.buffer = torch.zeros(
            [
                self.num_layers,
                2,
                self.num_hot_req,
                self.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ],
            device=self.device,
        )
        pass

    # Prefill:
    # return for every req [layer, seq, n_local_kv_heads, head_dim] * 2 (for k and v)
    def finalize_cache_bylayer_prefill(self, cache_k, cache_v, req_ids, varlen):
        pass

    # Prefill:
    def finalize_cache_all_prefill(self, req_ids, varlen):
        pass

    # Decode:
    def prepare_cache_decode(self, req_ids):
        pass

    def get_gpu_block_table(self):
        return None

    def get_block_size(self):
        return 0

    # Decode:
    # return [# of current req_ids]
    def get_gpu_seq_lens_excl_this_decode(self):
        pass

    # Decode:
    # return [# of current req_ids]
    def get_gpu_seq_lens_incl_this_decode(self):
        pass

    # Decode:
    # return [2, num_req, max_seqlen + 1, n_local_kv_heads, head_dim]
    def get_cache_decode(self, layer_id):
        pass

    # Decode:
    def finalize_cache_single_decode(self, req_ids):
        self.curr_varlens = None
        self.curr_req_ids = None

    # Decode:
    def finalize_cache_all_decode(self, req_id):
        pass
