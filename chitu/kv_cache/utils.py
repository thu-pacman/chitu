# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import gc
import math
import torch
import psutil
from logging import getLogger
from typing import Callable, Iterable

from chitu.kv_cache import GlobalLocalMap
from chitu.kv_cache.manager_names import MAIN_CACHE_NAME
from chitu.distributed.parallel_state import get_pp_group
from chitu.distributed.partition import compute_layer_dist_in_pp
from chitu.utils import ceil_div, get_global_args

logger = getLogger(__name__)


def cleanup_cuda_if_needed():
    if get_global_args().infer.op_impl != "cpu":
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def clamp_int(x, low, high):
    x = int(x)
    low = int(low)
    high = int(high)
    return min(max(x, low), high)


def _bytes_to_gb(x: int) -> float:
    return float(x) / 1e9


def build_layer_id_map(
    args,
    layer_filter_fn: Callable[[Iterable[int]], Iterable[int]] = lambda x: x,
) -> GlobalLocalMap:
    pp_size = int(args.infer.pp_size)

    mtp_size = int(getattr(get_global_args().infer, "mtp_size", 1))
    total_n_layers = int(args.models.n_layers) + (1 if mtp_size > 1 else 0)

    if pp_size > 1:
        layer_dist = compute_layer_dist_in_pp(pp_size)
        pp_rank = get_pp_group().rank_in_group
        local_begin = sum(layer_dist[:pp_rank])
        local_end = local_begin + layer_dist[pp_rank]
    else:
        local_begin = 0
        local_end = total_n_layers

    local_layers = list(layer_filter_fn(range(local_begin, local_end)))
    return GlobalLocalMap.from_list(local_layers)


# SingletonPagedKVCache and its scheduler-side manager. Both describe the same
# per-request state blocks, so the sizing helpers below take either one.
_SINGLETON_KV_CACHE_TYPES = {"SingletonPagedKVCache", "SingletonPagedKVCacheManager"}


def is_singleton_kv_cache(cache) -> bool:
    """``cache`` 是不是按请求独占状态的 SingletonPagedKVCache（或其 manager）。"""
    return type(cache).__name__ in _SINGLETON_KV_CACHE_TYPES


def kv_cache_block_layout(cache) -> tuple[int, int]:
    """cache（或对应的 manager）的 block 数构成：``(固定块数, 一块覆盖多少 token)``。

    * 普通 paged cache：``(0, block_size)``；
    * 带 checkpoint 的 SingletonPagedKVCache：``(num_hot_req * mtp_size, C)``
    """
    if is_singleton_kv_cache(cache):
        checkpoint_interval = getattr(cache, "checkpoint_interval", None)
        if checkpoint_interval is not None:
            return (
                int(cache.num_hot_req) * int(cache.mtp_size),
                int(checkpoint_interval),
            )
    return 0, int(cache.block_size)


def kv_cache_num_blocks_for_tokens(cache, max_tokens: int) -> int:
    """给定 token 容量，cache（或对应的 manager）应有的 block 数。"""
    fixed_blocks, tokens_per_block = kv_cache_block_layout(cache)
    return fixed_blocks + ceil_div(int(max_tokens), int(tokens_per_block))


def is_reallocable_kv_cache(cache) -> bool:
    """Whether a KV cache participates in the token-capacity block sizing.

    A cache (for singletons, its manager too) is reallocable only if its block
    count scales with sequence length (tokens):

    * ``SingletonPagedKVCache`` with a ``checkpoint_interval``: one checkpoint
      block per C tokens, so it does scale;
    * any other paged cache: one block per ``block_size`` tokens.

    Fixed-capacity caches MUST be excluded — a ``SingletonPagedKVCache`` without
    a ``checkpoint_interval`` (MTP hidden state, decode-side linear-attention
    state: just ``mtp_size`` in-place blocks per hot request) and any cache
    flagged ``fixed_num_blocks`` (e.g. DeepSeek-V4 sliding-window ring buffers)
    own a per-request number of blocks unrelated to token capacity.
    """
    if not hasattr(cache, "realloc") or not hasattr(cache, "num_blocks"):
        return False
    if not hasattr(cache, "manager_name") or not hasattr(cache, "max_num_blocks"):
        return False
    if is_singleton_kv_cache(cache):
        return getattr(cache, "checkpoint_interval", None) is not None
    if bool(getattr(cache, "fixed_num_blocks", False)):
        return False
    return True


def build_layer_id_map_lastlayer(args):
    layer_ids = []
    if get_pp_group().is_last_rank:
        mtp_size = int(getattr(get_global_args().infer, "mtp_size", 1))
        total_n_layers = int(args.models.n_layers) + (1 if mtp_size > 1 else 0)
        layer_ids = [total_n_layers - 1]
    return GlobalLocalMap.from_list(layer_ids)


def get_peak_live_and_target_bytes(
    memory_utilization=0.98,
    reserve_bytes=0,
) -> tuple[int, int, dict[str, int]]:
    """
    Return:
        live_bytes: peak memory bytes until now (including activation memory
            observed during warming-up)
        target_budget_bytes: target live-memory budget after applying
            memory_utilization and reserve_bytes
        allocator_debug_bytes: allocator peak/current debug fields
    """

    if get_global_args().infer.op_impl == "cpu":
        process = psutil.Process(os.getpid())
        live_bytes = int(process.memory_info().vms)
        target_budget_bytes = int(
            psutil.virtual_memory().total * memory_utilization
        ) - int(reserve_bytes)
        return (
            live_bytes,
            target_budget_bytes,
            {
                "reserved_peak_gap_bytes": 0,
            },
        )

    current_device = torch.cuda.current_device()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    memory_stats = torch.cuda.memory_stats(current_device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(current_device)

    ###################################################################
    # Memory stats part 1: Peak torch allocated bytes
    #
    # We use peak becuase it includes the activation memory observed during
    # warming-up
    peak_torch_allocated_bytes = memory_stats["allocated_bytes.all.peak"]
    peak_torch_reserved_bytes = memory_stats["reserved_bytes.all.peak"]
    # Memory stats part 1 ends.
    ###################################################################

    ###################################################################
    # Memory stats part 2: Current non-torch allocated bytes
    #
    # We fallback to current instead of peak, beacuse we have no way to
    # track peak non-torch allocated bytes.

    current_cuda_allocated_bytes = total_bytes - free_bytes
    # current_cuda_allocated_bytes includes torch allocated bytes, torch unused
    # bytes, and non-torch allocated bytes

    current_torch_reserved_bytes = memory_stats["reserved_bytes.all.current"]
    # current_torch_reserved_bytes includes torch allocated bytes and torch
    # unused bytes

    current_non_torch_allocations = max(
        0, current_cuda_allocated_bytes - current_torch_reserved_bytes
    )
    # Memory stats part 2 ends.
    ###################################################################

    live_bytes = int(peak_torch_allocated_bytes + current_non_torch_allocations)
    target_budget_bytes = int(total_bytes * memory_utilization) - int(reserve_bytes)
    allocator_debug_bytes = {
        "reserved_peak_gap_bytes": max(
            0, int(peak_torch_reserved_bytes) - int(peak_torch_allocated_bytes)
        ),
    }

    logger.debug(
        "Current live KV budget: total_bytes=%d, live_bytes=%d, reserve_bytes=%d, "
        "target_budget_bytes=%d",
        total_bytes,
        live_bytes,
        reserve_bytes,
        target_budget_bytes,
    )
    return live_bytes, target_budget_bytes, allocator_debug_bytes


def reduce_num_block_plan_across_ranks(plan: dict[str, int]) -> dict[str, int]:
    plan = {k: int(v) for k, v in plan.items()}

    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() == 1
    ):
        return plan

    keys = [MAIN_CACHE_NAME] + [k for k in sorted(plan.keys()) if k != MAIN_CACHE_NAME]

    if get_global_args().infer.op_impl == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", torch.cuda.current_device())

    t = torch.tensor([int(plan[k]) for k in keys], device=device, dtype=torch.long)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)

    return {k: int(v) for k, v in zip(keys, t.tolist())}


def allreduce_min_int(value: int) -> int:
    value = int(value)
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() == 1
    ):
        return value

    if get_global_args().infer.op_impl == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", torch.cuda.current_device())

    t = torch.tensor([value], device=device, dtype=torch.long)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
    return int(t.item())
