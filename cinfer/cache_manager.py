import torch
from .global_vars import get_timers

from logging import getLogger


logger = getLogger(__name__)


class KVCacheManager:
    def __init__(self, num_layers, n_local_kv_heads, head_dim):
        self.cache = {}
        self.prepared_cache = []
        self.num_layers = num_layers
        self.n_local_kv_heads = n_local_kv_heads
        self.head_dim = head_dim
        self.tmp_storage = []
        self.lengths = {}
        self.timers = get_timers()

    # Prefill:
    def finalize_cache_bylayer_prefill(self, cache_k, cache_v, req_ids, varlen):
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
            self.lengths[req_id] = varlen.cpu_lens[it]
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
        self.timers("cache_finalize_cache_all_prefill").stop()

    # Decode:
    # return [layer, num_req, 2, max_seqlen + 1, n_local_kv_heads, head_dim]
    def prepare_cache_decode(self, req_ids):
        self.timers("cache_prepare").start()
        self.layer_id = 0
        max_seq = 0
        seq_lens = []

        seq_lens = []
        for req_id in req_ids:
            seq_len = self.lengths[req_id]
            seq_lens.append(seq_len)
        self.seq_lens = seq_lens
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
            dtype=torch.bfloat16,
            device="cuda",
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
    def update_cache_decode(self, xk, xv):
        assert len(self.prepared_cache) > 0
        output = self.prepared_cache[self.layer_id]
        self.layer_id += 1
        for it in range(xk.shape[0]):
            output[0][it][self.seq_lens[it]] = xk[it]
            output[1][it][self.seq_lens[it]] = xv[it]
        return output

    # Decode:
    # return for every req [layer, seq + 1, n_local_kv_heads, head_dim] * 2 (for k and v)
    def finalize_cache_single_decode(self, req_ids):
        self.timers("cache_finalize_cache_single_decode").start()
        assert len(self.prepared_cache) > 0
        for it, req_id in enumerate(req_ids):
            self.cache[req_id] = [None] * self.num_layers
            self.lengths[req_id] += 1
        for layer_id in range(self.num_layers):
            for it, req_id in enumerate(req_ids):
                self.cache[req_id][layer_id] = [
                    self.prepared_cache[layer_id][0][it][
                        : self.seq_lens[it] + 1
                    ],  # [seq + 1, n_local_kv_heads, head_dim]
                    self.prepared_cache[layer_id][1][it][
                        : self.seq_lens[it] + 1
                    ],  # [seq + 1, n_local_kv_heads, head_dim]
                ]
        self.prepared_cache = []
        self.timers("cache_finalize_cache_single_decode").stop()

    # Decode:
    def finalize_cache_all_decode(self, req_id):
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
        num_hot_req=16,
        max_seq_length=2048,
    ):
        self.num_layers = num_layers
        self.n_local_kv_heads = n_local_kv_heads
        self.head_dim = head_dim
        self.num_hot_req = num_hot_req
        self.slot_availability = [True] * num_hot_req
        self.hot_reqs = [-1] * num_hot_req
        self.req2slot = {}
        self.lengths = {}
        self.max_seq_length = max_seq_length
        self.tmp_storage = []
        self.buffer = torch.zeros(
            [
                self.num_layers,
                2,
                self.num_hot_req,
                self.max_seq_length,
                self.n_local_kv_heads,
                self.head_dim,
            ],
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.timers = get_timers()
        self.prepared_reqs = []
        self.rounded_max_seq = -1
        self.layer_id = 0

    # Prefill:
    # return for every req [layer, seq, n_local_kv_heads, head_dim] * 2 (for k and v)
    def finalize_cache_bylayer_prefill(self, cache_k, cache_v, req_ids, varlen):
        self.timers("cache_finalize_cache_all_prefill").start()
        if self.layer_id == 0:
            for it, req_id in enumerate(req_ids):
                self.lengths[req_id] = varlen.cpu_lens[it]
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
            self.buffer[self.layer_id][0][self.req2slot[req_id]][
                : varlen.cpu_lens[it]
            ] = cache_k[start:end]
            self.buffer[self.layer_id][1][self.req2slot[req_id]][
                : varlen.cpu_lens[it]
            ] = cache_v[start:end]
            start = end
        self.timers("cache_finalize_cache_all_prefill").stop()
        self.layer_id += 1

    # Prefill:
    def finalize_cache_all_prefill(self, req_ids, varlen):
        self.layer_id = 0

    # Decode:
    def prepare_cache_decode(self, req_ids):
        self.timers("cache_prepare").start()
        self.layer_id = 0
        start_pos = self.hot_reqs.index(req_ids[0])
        assert start_pos + len(req_ids) <= self.num_hot_req
        # assert (
        #     self.hot_reqs[start_pos : start_pos + len(req_ids)] == req_ids
        # ), f"{self.hot_reqs} {req_ids}"

        seq_lens = []
        for req_id in req_ids:
            seq_len = self.lengths[req_id]
            seq_lens.append(seq_len)
        self.seq_lens = seq_lens
        max_seq = max(seq_lens)

        limit = 16
        rounded_max_seq = (max_seq + 1 + limit - 1) // limit * limit
        if self.rounded_max_seq >= rounded_max_seq and self.prepared_reqs == req_ids:
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
                self.head_dim * self.n_local_kv_heads * self.max_seq_length * self.num_hot_req * 2,
                self.head_dim * self.n_local_kv_heads * self.max_seq_length * self.num_hot_req,
                self.head_dim * self.n_local_kv_heads * self.max_seq_length,
                self.head_dim * self.n_local_kv_heads,
                self.head_dim,
                1,
            ),
            start_pos * self.head_dim * self.n_local_kv_heads * self.max_seq_length,
        )
        # fmt: on
        self.timers("cache_prepare").stop()

    # Decode:
    # return [num_req, 2, max_seqlen + 1, n_local_kv_heads, head_dim]
    def update_cache_decode(self, xk, xv):
        output = self.prepared_cache[self.layer_id]
        self.layer_id += 1
        for it in range(xk.shape[0]):
            output[0][it][self.seq_lens[it]] = xk[it]
            output[1][it][self.seq_lens[it]] = xv[it]
        return output

    # Decode:
    def finalize_cache_single_decode(self, req_ids):
        for item in req_ids:
            self.lengths[item] += 1
        self.layer_id = 0

    # Decode:
    def finalize_cache_all_decode(self, req_id):
        slot_id = self.hot_reqs.index(req_id)
        if slot_id == -1:  # not in the hot slot
            return
        self.hot_reqs[slot_id] = -1
        self.slot_availability[slot_id] = True
        self.req2slot.pop(req_id)
        self.lengths.pop(req_id)
