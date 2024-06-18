import torch


class KVCacheManager:
    def __init__(self, num_layers):
        self.cache = {}
        self.prepared = False
        self.prepared_cache = []
        self.num_layers = num_layers
        self.tmp_storage = []

    def prepare(self, req_ids):
        self.prepared = True
        self.layer_id = 0
        for layer_id in range(self.num_layers):
            self.prepared_cache.append([])
            for req_id in req_ids:
                self.prepared_cache[layer_id].append(
                    [self.cache[req_id][layer_id][0], self.cache[req_id][layer_id][1]]
                )
                # TODO: Pad

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

    def use_prepare_cache(self):
        return (
            self.prepared_cache[self.layer_id][0],
            self.prepared_cache[self.layer_id][1],
        )

    def finalize_prefill(self, req_ids, varlen):
        assert len(self.tmp_storage) == self.num_layers
        assert len(varlen) == self.num_layers
        assert sum(varlen.cpu_lens) == self.tmp_storage[0][0].shape[0]
        for req_id in req_ids:
            self.cache[req_id] = [None] * self.num_layers
        for layer_id in range(self.num_layers):
            start = 0
            for req_id in req_ids:
                end = start + varlen.cpu_lens[layer_id]
                self.cache[req_id][layer_id] = [
                    self.tmp_storage[layer_id][0][start:end],
                    self.tmp_storage[layer_id][1][start:end],
                ]
                start = end

    def update_cache(self, layer_id, req_id, it):
        self.cache[req_id][layer_id][0] = torch.cat(
            [self.cache[req_id][layer_id][0], self.tmp_storage[layer_id][0][it]], dim=0
        )
        self.cache[req_id][layer_id][1] = torch.cat(
            [self.cache[req_id][layer_id][1], self.tmp_storage[layer_id][1][it]], dim=0
        )

    def finalize_decode(self, req_ids):
        assert len(self.tmp_storage) == self.num_layers
        assert len(self.prepared_cache) > 0
        for layer_id in range(self.num_layers):
            for it, req_id in enumerate(req_ids):
                self.update_cache(layer_id, req_id, it)


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
