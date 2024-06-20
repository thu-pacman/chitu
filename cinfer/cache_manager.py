import torch
from .global_vars import get_timers


class KVCacheManager:
    def __init__(self, num_layers, n_local_kv_heads, head_dim):
        self.cache = {}
        self.prepared_cache = []
        self.num_layers = num_layers
        self.n_local_kv_heads = n_local_kv_heads
        self.head_dim = head_dim
        self.tmp_storage = []
        self.timers = get_timers()

    # return [layer, num_req, 2, max_seqlen + 1, n_local_kv_heads, head_dim]
    def prepare(self, req_ids):
        self.timers("cache_prepare").start()
        self.layer_id = 0
        max_seq = 0
        seq_lens = []
        for req_id in req_ids:
            seq_len = self.cache[req_id][self.layer_id][0].shape[0]
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
            ]
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

    # return for every req [layer, seq, n_local_kv_heads, head_dim] * 2 (for k and v)
    def finalize_prefill(self, req_ids, varlen):
        self.timers("cache_finalize_prefill").start()
        assert (
            len(self.tmp_storage) == self.num_layers
        ), f"{len(self.tmp_storage)} {self.num_layers}"
        assert len(varlen.cpu_lens) == len(req_ids)
        assert sum(varlen.cpu_lens) == self.tmp_storage[0][0].shape[0]
        for req_id in req_ids:
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
        self.timers("cache_finalize_prefill").stop()

    # return for every req [layer, seq + 1, n_local_kv_heads, head_dim] * 2 (for k and v)
    def finalize_decode(self, req_ids):
        self.timers("cache_finalize_decode").start()
        assert len(self.prepared_cache) > 0
        for it, req_id in enumerate(req_ids):
            self.cache[req_id] = [None] * self.num_layers
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
        self.timers("cache_finalize_decode").stop()

    def get(self, key):
        return self.cache.get(key, None)

    def set(self, key, value):
        self.cache[key] = value

    def delete(self, key):
        if key in self.cache:
            del self.cache[key]

    def clear(self):
        self.cache.clear()

    def tmp_store(self, cache_k, cache_v):
        self.tmp_storage.append([cache_k, cache_v])

    # return [num_req, 2, max_seqlen + 1, n_local_kv_heads, head_dim]
    def update_prepare_cache(self, xk, xv):
        assert len(self.prepared_cache) > 0
        output = self.prepared_cache[self.layer_id]
        self.layer_id += 1
        for it in range(xk.shape[0]):
            output[0][it][self.seq_lens[it]] = xk[it]
            output[1][it][self.seq_lens[it]] = xv[it]
        return output


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
        print("cache init:", cache_k.shape)

        self.cache_k = cache_k
        self.cache_v = cache_v
        self.inited = True

    def extend(self, new_cache_k, new_cache_v):
        print("cache extend:", new_cache_k.shape)
        assert self.inited
        self.cache_k = torch.cat([self.cache_k, new_cache_k], dim=0)
        self.cache_v = torch.cat([self.cache_v, new_cache_v], dim=0)
