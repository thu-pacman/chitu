import torch
from .global_vars import get_timers, get_dtype

from logging import getLogger

from .ops import move_data

import os

try:
    os.environ["PAGED_SIZE"]
    paged_size = int(os.environ["PAGED_SIZE"])
except KeyError:
    paged_size = 16
print("PAGED_SIZE : ", paged_size)


logger = getLogger(__name__)
_BLOCK_SIZE = 512  # _BLOCK_SIZE must be a multiple of 256 for FlashAttention
_MAX_SEQ_LEN = 2048
_MAX_NUM_BLOCKS_PER_LAYER = (
    _MAX_SEQ_LEN // _BLOCK_SIZE
) * paged_size  # TODO: make  this dynamic


class PagedKVCacheManager:
    def __init__(
        self,
        num_layers,
        n_local_kv_heads,
        head_dim,
        block_size=_BLOCK_SIZE,
        max_seq_len=_MAX_SEQ_LEN,
        num_blocks_per_layer=_MAX_NUM_BLOCKS_PER_LAYER,
        device="cuda",
    ):
        print("Init PagedKVCacheManager")

        self.num_layers = num_layers
        self.n_local_kv_heads = n_local_kv_heads
        self.head_dim = head_dim
        assert block_size % 256 == 0
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        self.num_blocks = num_blocks_per_layer * num_layers
        self.device = torch.device(device)

        self.seq_lens = {}
        self.timers = get_timers()
        self.block_table = {}  # (seq_id, layer_id, block_idx)
        # TODO: For better performance, use list instead of set for free_blocks
        self.free_blocks = set(range(self.num_blocks))
        use_half = get_dtype()
        self.paged_k_cache = torch.zeros(
            self.num_blocks,
            block_size,
            n_local_kv_heads,
            head_dim,
            device=device,
            dtype=torch.float16 if use_half else torch.bfloat16,
        )
        self.paged_v_cache = torch.zeros(
            self.num_blocks,
            block_size,
            n_local_kv_heads,
            head_dim,
            device=device,
            dtype=torch.float16 if use_half else torch.bfloat16,
        )

        # seq_lens = (tokens != -1).sum(1)
        # for i, t in enumerate(seq_lens.tolist()):
        #     num_blocks_to_reserve = math.ceil(t / block_size)
        #     num_filled_positions = t % block_size
        #     for b in range(num_blocks_to_reserve):
        #         index = self.get_free_block()
        #         if b == num_blocks_to_reserve-1:
        #             self.block_table[i].append((index, num_filled_positions))
        #         else:
        #             self.block_table[i].append((index, block_size))

    # Init block table and kv cache with kv generated during prefill
    def finalize_cache_bylayer_prefill(self, xk, xv, req_ids, varlen, layer_id):
        self.timers("cache_finalize_cache_all_prefill").start()
        for idx, req_id in enumerate(req_ids):
            if layer_id == 0:
                self.seq_lens[req_id] = varlen.cpu_lens[idx]
                # logger.warning(f"Prefill: seq_lens[{req_id}] = {self.seq_lens[req_id]}")
            num_blocks_prepared = (
                varlen.cpu_lens[idx] + self.block_size - 1
            ) // self.block_size
            block_ids = []
            start_pos = varlen.cpu_prefix_lens[idx]
            end_pos = varlen.cpu_prefix_lens[idx + 1]
            for chunck_id in range(num_blocks_prepared):
                block_idx = self.get_free_block()
                block_ids.append(block_idx)
                if chunck_id != num_blocks_prepared - 1:
                    self.paged_k_cache[block_idx] = xk[
                        start_pos : (start_pos + self.block_size)
                    ].clone()
                    self.paged_v_cache[block_idx] = xv[
                        start_pos : (start_pos + self.block_size)
                    ].clone()
                    start_pos += self.block_size
                else:
                    tmp_len = end_pos - start_pos
                    self.paged_k_cache[block_idx][:tmp_len] = xk[
                        start_pos:end_pos
                    ].clone()
                    self.paged_v_cache[block_idx][:tmp_len] = xv[
                        start_pos:end_pos
                    ].clone()
            if req_id in self.block_table:
                assert len(self.block_table[req_id]) == layer_id
            else:
                self.block_table[req_id] = []
            self.block_table[req_id].append(block_ids.copy())
            # logger.warning(f"block table[{req_id}][{layer_id}] = {block_ids}")
        self.timers("cache_finalize_cache_all_prefill").stop()

    def finalize_cache_all_prefill(self, req_ids, varlen):
        self.curr_varlens = None
        self.curr_req_ids = None
        pass

    def prepare_cache_decode(self, req_ids):
        seq_lens = []
        for req_id in req_ids:
            seq_len = self.seq_lens[req_id]
            seq_lens.append(seq_len)
        max_seq = max(seq_lens)
        self.curr_seq_lens = seq_lens
        self.curr_seq_lens_gpu = torch.tensor(
            seq_lens, dtype=torch.int32, device=self.device
        )
        pass

    def get_free_block(self):
        # TODO: When run out of free blocks, use scheduling and preemption in paper instead of exception
        self.timers("get_free_block").start()
        if len(self.free_blocks) == 0:
            raise Exception("No more free blocks.")
        idx = list(self.free_blocks)[0]
        self.free_blocks.remove(idx)
        self.timers("get_free_block").stop()
        return idx

    def get_gpu_block_table(self, req_ids, layer_id):
        self.timers("get_gpu_block_table").start()
        max_block_num = max(
            len(self.block_table[req_id][layer_id]) for req_id in req_ids
        )
        gpu_block_table = [[0] * max_block_num for _ in range(len(req_ids))]
        for idx, req_id in enumerate(req_ids):
            block_ids = self.block_table[req_id][layer_id]
            gpu_block_table[idx][: len(block_ids)] = block_ids
        # logger.warning(gpu_block_table)
        output = torch.tensor(gpu_block_table, dtype=torch.int32, device=self.device)
        self.timers("get_gpu_block_table").stop()
        return output

    def get_gpu_seq_lens(self):
        return self.curr_seq_lens_gpu

    def get_paged_kv_cache(self):
        return self.paged_k_cache, self.paged_v_cache

    def free_req_cache_blocks(self, req_id):
        self.timers("free_req_cache_blocks").start()
        for layer_blocks in self.block_table[req_id]:
            for block in layer_blocks:
                self.free_blocks.add(block)
        del self.block_table[req_id]
        self.timers("free_req_cache_blocks").stop()

    # Prepare enough block table for next decoding. When decoding, flash attention will fill new kv into paged kv cache (inplace).
    def prepare_block_table_for_decode(self, req_ids, layer_id):
        self.timers("prepare_block_table_for_decode").start()
        for req_id in req_ids:
            if self.seq_lens[req_id] % self.block_size == 0:
                self.block_table[req_id][layer_id].append(self.get_free_block())
        self.timers("prepare_block_table_for_decode").stop()

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
    def __init__(self, num_layers, n_local_kv_heads, head_dim, device="cuda"):
        self.cache = {}
        self.prepared_cache = []
        self.num_layers = num_layers
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
        for layer_id in range(self.num_layers):
            start = 0
            for it, req_id in enumerate(req_ids):
                end = start + varlen.cpu_lens[it]
                self.cache[req_id][layer_id] = [
                    self.tmp_storage[layer_id][0][
                        start:end
                    ],  # [seq, n_local_kv_heads, head_dim]
                    self.tmp_storage[layer_id][1][
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
        self.curr_seq_lens_gpu = torch.tensor(
            seq_lens, dtype=torch.int32, device=self.device
        )
        max_seq = max(seq_lens)
        n_local_kv_heads = self.cache[req_ids[0]][0][0].shape[-2]
        head_dim = self.cache[req_ids[0]][0][0].shape[-1]
        use_half = get_dtype()
        prepared_cache = torch.zeros(
            [
                self.num_layers,  # layers
                2,
                len(req_ids),  # batch_size
                max_seq + 1,  # seq_len
                n_local_kv_heads,  # n_local_kv_heads
                head_dim,  # head_dim
            ],
            dtype=torch.float16 if use_half else torch.bfloat16,
            device=self.device,
        )
        # hkz-comment: Very similar to matrix transpose;
        for layer_id in range(self.num_layers):
            for it, req_id in enumerate(req_ids):
                prepared_cache[layer_id][0][it][: seq_lens[it]] = self.cache[req_id][
                    layer_id
                ][0]
                prepared_cache[layer_id][1][it][: seq_lens[it]] = self.cache[req_id][
                    layer_id
                ][1]
        self.prepared_cache = prepared_cache
        self.timers("cache_prepare").stop()

    # Decode:
    # return [num_req, 2, max_seqlen + 1, n_local_kv_heads, head_dim]
    def update_cache_decode(self, xk, xv, layer_id):
        assert len(self.prepared_cache) > 0
        output = self.prepared_cache[layer_id]
        for it in range(xk.shape[0]):
            output[0][it][self.curr_seq_lens[it]] = xk[it]
            output[1][it][self.curr_seq_lens[it]] = xv[it]
        return output

    # Decode:
    # return [num_req, 2, max_seqlen + 1, n_local_kv_heads, head_dim]
    def get_cache_decode(self, layer_id):
        return self.prepared_cache[layer_id]

    # Decode:
    # return [# of current req_ids]
    def get_gpu_seq_lens(self):
        return self.curr_seq_lens_gpu

    # Decode:
    # return for every req [layer, seq + 1, n_local_kv_heads, head_dim] * 2 (for k and v)
    def finalize_cache_single_decode(self, req_ids):
        self.timers("cache_finalize_cache_single_decode").start()
        assert len(self.prepared_cache) > 0
        for it, req_id in enumerate(req_ids):
            self.cache[req_id] = [None] * self.num_layers
            self.seq_lens[req_id] += 1
        for layer_id in range(self.num_layers):
            for it, req_id in enumerate(req_ids):
                self.cache[req_id][layer_id] = [
                    self.prepared_cache[layer_id][0][it][
                        : self.curr_seq_lens[it] + 1
                    ],  # [seq + 1, n_local_kv_heads, head_dim]
                    self.prepared_cache[layer_id][1][it][
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
        num_layers,
        n_local_kv_heads,
        head_dim,
        num_hot_req,
        max_seq_len=2048,
        device="cuda",
    ):
        self.num_layers = num_layers
        self.n_local_kv_heads = n_local_kv_heads
        self.head_dim = head_dim
        self.num_hot_req = num_hot_req
        self.slot_availability = [True] * num_hot_req
        self.hot_reqs = [-1] * num_hot_req
        self.req2slot = {}
        self.seq_lens = {}
        self.max_seq_len = max_seq_len
        self.tmp_storage = []
        self.device = torch.device(device)
        use_half = get_dtype()
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
            dtype=torch.float16 if use_half else torch.bfloat16,
        )
        self.timers = get_timers()
        self.prepared_reqs = []
        self.rounded_max_seq = -1

    # Prefill:
    # return for every req [layer, seq, n_local_kv_heads, head_dim] * 2 (for k and v)
    def finalize_cache_bylayer_prefill(
        self, cache_k, cache_v, req_ids, varlen, layer_id
    ):
        self.timers("cache_finalize_cache_all_prefill").start()
        if layer_id == 0:
            # logger.warning(req_ids)
            for it, req_id in enumerate(req_ids):
                self.seq_lens[req_id] = varlen.cpu_lens[it]
                for i in range(self.num_hot_req):
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
            self.buffer[layer_id][0][self.req2slot[req_id]][: varlen.cpu_lens[it]] = (
                cache_k[start:end]
            )
            self.buffer[layer_id][1][self.req2slot[req_id]][: varlen.cpu_lens[it]] = (
                cache_v[start:end]
            )
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
        self.curr_seq_lens_gpu = torch.tensor(
            seq_lens, dtype=torch.int32, device=self.device
        )

        limit = 16
        rounded_max_seq = (max_seq + 1 + limit - 1) // limit * limit
        if self.rounded_max_seq >= rounded_max_seq and self.prepared_reqs == req_ids:
            # prepared cache is long enough
            self.timers("cache_prepare").stop()
            return

        self.rounded_max_seq = rounded_max_seq
        self.prepared_reqs = req_ids

        # fmt: off
        self.prepared_cache = torch.as_strided(
            self.buffer,
            (
                self.num_layers, # num layer
                2,  # k & v
                len(req_ids),  # 
                rounded_max_seq,  # head_dim * n_local_kv_heads
                self.n_local_kv_heads, # head_dim
                self.head_dim, # 1
            ),
            (
                self.head_dim * self.n_local_kv_heads * self.max_seq_len * self.num_hot_req * 2,
                self.head_dim * self.n_local_kv_heads * self.max_seq_len * self.num_hot_req,
                self.head_dim * self.n_local_kv_heads * self.max_seq_len,
                self.head_dim * self.n_local_kv_heads,
                self.head_dim,
                1,
            ),
            start_pos * self.head_dim * self.n_local_kv_heads * self.max_seq_len,
        )
        # fmt: on
        self.timers("cache_prepare").stop()

    # Decode:
    # return [num_req, 2, max_seqlen + 1, n_local_kv_heads, head_dim]
    def update_cache_decode(self, xk, xv, layer_id):
        # self.timers("cache_update").start()
        output = self.prepared_cache[layer_id]
        # self.timers("cache_update").stop()
        # return output
        # for it in range(xk.shape[0]):
        #     output[0][it][self.curr_seq_lens[it]] = xk[it]
        #     output[1][it][self.curr_seq_lens[it]] = xv[it]
        # if layer_id == 0:
        #     logger.warning(f"Update: {self.curr_seq_lens[0]} {xk.shape}")
        move_data(output, xk, xv, self.curr_seq_lens_gpu, self.max_seq_len)
        return output

    # Decode:
    # return [num_req, 2, max_seqlen + 1, n_local_kv_heads, head_dim]
    def get_cache_decode(self, layer_id):
        return self.prepared_cache[layer_id]

    # Decode:
    # return [# of current req_ids]
    def get_gpu_seq_lens(self):
        return self.curr_seq_lens_gpu

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
        self.hot_reqs[slot_id] = -1
        self.slot_availability[slot_id] = True
        self.req2slot.pop(req_id)
        self.buffer[:, :, slot_id, :, :, :].zero_()


class KVCacheManagerNop:
    def __init__(
        self,
        num_layers,
        n_local_kv_heads,
        head_dim,
        num_hot_req=16,
        max_seq_len=2048,
        device="cuda",
    ):
        self.num_layers = num_layers
        self.n_local_kv_heads = n_local_kv_heads
        self.head_dim = head_dim
        self.num_hot_req = num_hot_req
        self.slot_availability = [True] * num_hot_req
        self.hot_reqs = [-1] * num_hot_req
        self.req2slot = {}
        self.max_seq_len = max_seq_len
        self.tmp_storage = []
        self.device = torch.device(device)
        use_half = get_dtype()
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
            dtype=torch.float16 if use_half else torch.bfloat16,
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

    # Decode:
    # return [num_req, 2, max_seqlen + 1, n_local_kv_heads, head_dim]
    def update_cache_decode(self, xk, xv):
        pass

    # Decode:
    # return [# of current req_ids]
    def get_gpu_seq_lens(self):
        pass

    # Decode:
    # return [num_req, 2, max_seqlen + 1, n_local_kv_heads, head_dim]
    def get_cache_decode(self, layer_id):
        pass

    # Decode:
    def finalize_cache_single_decode(self, req_ids):
        self.curr_varlens = None
        self.curr_req_ids = None

    # Decode:
    def finalize_cache_all_decode(self, req_id):
        pass
