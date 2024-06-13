import torch


class KVCacheManager:
    def __init__(self):
        self.cache = {}

    def get(self, key):
        return self.cache.get(key, None)

    def set(self, key, value):
        self.cache[key] = value

    def delete(self, key):
        if key in self.cache:
            del self.cache[key]

    def clear(self):
        self.cache.clear()


class KVCache:
    def __init__(self):
        self.cache_k = None
        self.cache_v = None
        self.inited = False

    def check_shape(self, cache):
        # (bsz * seqlen, self.n_local_heads, self.head_dim)
        assert len(cache.shape) == 3
        assert cache.device == torch.device("cuda")

    def check_shapes(self, cache_k, cache_v):
        self.check_shape(cache_k)
        self.check_shape(cache_v)
        assert cache_k.shape == cache_v.shape

    def init(self, cache_k, cache_v):
        assert not self.inited
        self.check_shapes(cache_k, cache_v)

        self.cache_k = cache_k
        self.cache_v = cache_v
        self.inited = True

    def extend(self, new_cache_k, new_cache_v):
        assert self.inited
        raise NotImplementedError
