# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""kv cache statics for metrics collection."""

from chitu.backend import Backend
from chitu.utils import ceil_div
from typing import TYPE_CHECKING, Optional

try:
    from chitu.distributed.pd_disaggregation.pd_scheduler import (
        get_pd_scheduler_instance,
    )
except Exception:  # pragma: no cover - optional PD dependency
    get_pd_scheduler_instance = None

if TYPE_CHECKING:
    from chitu.kv_cache import PagedKVCacheManager


def paged_kvcache_stats() -> dict[int, tuple[int, int, float]]:
    if Backend.cache_managers is None:
        return {}
    kvcache_stats = {}
    for dp_id, cache_managers in enumerate(Backend.cache_managers):
        kv_cache_manager: PagedKVCacheManager = cache_managers["main"]
        total_blocks = kv_cache_manager.num_blocks
        used_blocks = kv_cache_manager.num_active_blocks
        if total_blocks > 0:
            kvcache_usage = used_blocks / total_blocks
        else:
            kvcache_usage = -1
        kvcache_stats[dp_id] = (total_blocks, used_blocks, kvcache_usage)
    return kvcache_stats


def per_dp_kvcache_stats(dp_id: int) -> dict[int, tuple[int, int, float]]:
    total_blocks = Backend.cache_dict["main"].num_blocks
    used_blocks = Backend.cache_dict["main"].num_used_blocks
    if total_blocks > 0:
        kvcache_usage = used_blocks / total_blocks
    else:
        kvcache_usage = -1
    return {dp_id: (total_blocks, used_blocks, kvcache_usage)}


def kvcache_stats(is_main_rank: bool, dp_id: int) -> dict[int, tuple[int, int, float]]:
    from chitu.kv_cache import PagedKVCache

    if Backend.cache_dict is None:
        return {}
    elif isinstance(Backend.cache_dict["main"], PagedKVCache):
        return paged_kvcache_stats() if is_main_rank else {}
    else:
        return per_dp_kvcache_stats(dp_id)


def get_prealloc_blocks(dp_size: int) -> Optional[dict[int, int]]:
    """Get prealloc KV blocks per DP rank from PD scheduler."""
    if get_pd_scheduler_instance is None:
        return None
    scheduler = get_pd_scheduler_instance()
    if scheduler is None:
        return None
    if Backend.cache_dict["main"] is None:
        return None
    if not hasattr(Backend.cache_dict["main"], "block_size"):
        return None
    block_size = Backend.cache_dict["main"].block_size
    if block_size <= 0:
        return None
    tokens_by_dp = getattr(scheduler, "_decode_prealloc_tokens_inflight_by_dp", None)
    if not tokens_by_dp:
        return None
    prealloc_blocks: dict[int, int] = {}
    for dp_id in range(dp_size):
        tokens = 0
        if dp_id < len(tokens_by_dp):
            tokens = int(tokens_by_dp[dp_id])
        prealloc_blocks[dp_id] = int(ceil_div(tokens, block_size)) if tokens > 0 else 0
    return prealloc_blocks
