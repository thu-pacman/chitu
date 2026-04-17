# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""kv cache statics for metrics collection."""

from chitu.backend import Backend
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chitu.kv_cache import PagedKVCacheManager


def paged_kvcache_stats(dp_id):
    if Backend.cache_managers is None:
        return 0, 0, 0
    kv_cache_manager: PagedKVCacheManager = Backend.cache_managers[dp_id]["main"]
    total_blocks = kv_cache_manager.num_blocks
    used_blocks = kv_cache_manager.num_active_blocks
    kvcache_usage = used_blocks / total_blocks
    return used_blocks, total_blocks, kvcache_usage
