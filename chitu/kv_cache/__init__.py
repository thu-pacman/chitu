# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
KV cache configuration in Chitu is modularized via registries.

This package separates KV cache concerns into two layers:

1) KV cache spec providers (layout/quantization):
   - Define the KV cache tensor layout for a model (e.g. k/v vs. kv_lora/k_pe),
     including shape_per_token_dict or (n_local_kv_heads, head_dim) style specs.
   - Apply optional KV-cache quantization rules (dtype_dict) consistently.

2) Cache manager builders:
   - Decide which cache managers to create for a given model/config (e.g. main,
     linear-attn state, indexer cache), and how to assign layers to each manager.
   - Keep backend.py free of model-specific logic.

Registration is done via @register_* decorators. Importing this package triggers
side-effect imports that populate registries, so callers should import
`chitu.kv_cache` (or the specific submodules) before building cache managers.

To add support for a new model:
- Implement a spec provider under `chitu/kv_cache/providers/` and register it.
- Implement a builder under `chitu/kv_cache/builders.py` if the model needs
  non-default cache-manager topology.
"""

from chitu.kv_cache.prefix_caching import (
    TokenBlock,
    BlockIdentity,
    BlockIdentityChainBuilder,
    BlockRuntime,
    KVBlockState,
    NONE_BLK_HASH,
)
from chitu.kv_cache.kv_cache import (
    GlobalLocalMap,
    KVCacheBase,
    PagedKVCache,
    SingletonPagedKVCache,
    MMPagedKVCache,
    DenseKVCache,
    DeepSeekV4DenseKVCache,
    DeepSeekV4PagedKVCache,
    DeepSeekV4SlidingWindowPagedKVCache,
    KVCacheAccessor,
    PagedKVCacheAccessor,
    DenseKVCacheAccessor,
)
from chitu.kv_cache.cache_manager import (
    KVCacheManagerBase,
    PagedKVCacheManager,
    DeepSeekV4SlidingKVCacheManager,
    DeepSeekV4CompressedKVCacheManager,
    SingletonPagedKVCacheManager,
)
