# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import math
import torch
import psutil
from logging import getLogger
import torch.distributed as dist

from typing import Callable, Iterable, Tuple, Dict, Optional

from chitu.kv_cache import GlobalLocalMap
from chitu.distributed.parallel_state import get_pp_group
from chitu.distributed.partition import compute_layer_dist_in_pp
from chitu.utils import get_global_args


logger = getLogger(__name__)


def build_layer_id_map(
    args,
    layer_filter_fn: Callable[[Iterable[int]], Iterable[int]] = lambda x: x,
) -> GlobalLocalMap:
    pp_size = int(args.infer.pp_size)

    mtp_size = int(getattr(get_global_args().infer, "mtp_size", 1))
    total_n_layers = int(args.models.n_layers) + (1 if mtp_size > 1 else 0)

    if pp_size > 1:
        layer_dist = compute_layer_dist_in_pp(args.models.n_layers, pp_size)
        pp_rank = get_pp_group().rank_in_group
        local_begin = sum(layer_dist[:pp_rank])
        local_end = local_begin + layer_dist[pp_rank]
    else:
        local_begin = 0
        local_end = total_n_layers

    local_layers = list(layer_filter_fn(range(local_begin, local_end)))
    return GlobalLocalMap.from_list(local_layers)


def plan_kv_cache_blocks_after_warmup(args, cache_managers):
    """
    Structure-driven pre-plan:

    - main is kept at current size in pre-plan
    - indexer is shrunk to the minimum size needed to cover CURRENT main blocks
    - final main/indexer growth will be solved after shrink using current live memory
    """
    assert "main" in cache_managers, "main cache manager is required"

    infos = {}
    for name, cm in cache_managers.items():
        infos[name] = {
            "cm": cm,
            "block_mem": cm.estimate_bytes_per_block(),
            "current_blocks": cm.num_blocks,
            "max_num_blocks": cm.max_num_blocks,
        }

    main_cm = cache_managers["main"]
    main_cur = main_cm.num_blocks

    plan = {"main": main_cur}

    if "indexer" in cache_managers:
        indexer_cm = cache_managers["indexer"]
        indexer_cur = indexer_cm.num_blocks
        indexer_cap = indexer_cm.max_num_blocks

        desired_indexer_blocks = estimate_indexer_blocks_from_main(
            main_cm, indexer_cm, main_cur
        )
        desired_indexer_blocks = min(max(0, desired_indexer_blocks), indexer_cap)

        # shrink-only in pre-plan
        indexer_target = min(indexer_cur, desired_indexer_blocks)
        plan["indexer"] = indexer_target

        logger.info(
            "KV cache pre-plan: main_cur=%d, indexer_cur=%d, desired_indexer_blocks=%d, indexer_target=%d",
            main_cur,
            indexer_cur,
            desired_indexer_blocks,
            indexer_target,
        )
    # TODO: add mm plan

    for name, info in infos.items():
        if name in plan:
            continue
        plan[name] = min(info["current_blocks"], info["max_num_blocks"])

    return plan


def get_additional_block_num(cache_manager, memory_utilization=0.98) -> int:
    block_mem = cache_manager.estimate_bytes_per_block()
    if block_mem <= 0:
        return 0
    additional_bytes = get_additional_kv_cache_memory_bytes(memory_utilization)
    return max(0, additional_bytes // block_mem)


def get_additional_kv_cache_memory_bytes(memory_utilization=0.98) -> int:
    if get_global_args().infer.op_impl == "cpu":
        process = psutil.Process(os.getpid())
        current_process_mem = process.memory_info().vms
        additional_memory = (
            psutil.virtual_memory().total * memory_utilization - current_process_mem
        )
        return additional_memory

    current_device = torch.cuda.current_device()
    torch.cuda.synchronize()

    _, total_memory = torch.cuda.mem_get_info(current_device)
    peak_memory = torch.cuda.memory_stats(current_device)["allocated_bytes.all.peak"]

    torch.cuda.empty_cache()

    torch_allocated_bytes = torch.cuda.memory_stats(current_device)[
        "allocated_bytes.all.current"
    ]
    total_allocated_bytes = (
        torch.cuda.mem_get_info(current_device)[1]
        - torch.cuda.mem_get_info(current_device)[0]
    )
    non_torch_allocations = total_allocated_bytes - torch_allocated_bytes
    if non_torch_allocations > 0:
        peak_memory += non_torch_allocations

    additional_kv_cache_memory = total_memory * memory_utilization - peak_memory
    logger.debug(
        f"{additional_kv_cache_memory} bytes of memory available on this rank for additional KV "
        f"cache after warming-up."
    )
    return additional_kv_cache_memory


def reduce_num_block_plan_across_ranks(plan: dict[str, int]) -> dict[str, int]:
    plan = {k: int(v) for k, v in plan.items()}

    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() == 1
    ):
        return plan

    keys = ["main"] + [k for k in sorted(plan.keys()) if k != "main"]

    if get_global_args().infer.op_impl == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", torch.cuda.current_device())

    t = torch.tensor([int(plan[k]) for k in keys], device=device, dtype=torch.long)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)

    return {k: int(v) for k, v in zip(keys, t.tolist())}


def get_additional_block_num_from_current(
    cache_manager,
    memory_utilization=0.98,
    reserve_bytes=0,
) -> int:
    block_mem = cache_manager.estimate_bytes_per_block()
    if block_mem <= 0:
        return 0

    additional_bytes = get_current_available_kv_cache_memory_bytes(
        memory_utilization=memory_utilization,
        reserve_bytes=reserve_bytes,
    )
    return max(0, additional_bytes // block_mem)


def get_current_available_kv_cache_memory_bytes(
    memory_utilization=0.98,
    reserve_bytes=0,
) -> int:
    """
    Estimate additional bytes available for KV cache based on CURRENT live memory,
    not historical peak memory.

    This is intended for the 2nd-stage grow-after-shrink path.
    """
    if get_global_args().infer.op_impl == "cpu":
        process = psutil.Process(os.getpid())
        current_process_mem = process.memory_info().vms
        available = (
            psutil.virtual_memory().total * memory_utilization
            - current_process_mem
            - reserve_bytes
        )
        return max(0, available)

    current_device = torch.cuda.current_device()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    free_bytes, total_bytes = torch.cuda.mem_get_info(current_device)

    torch_allocated_bytes = torch.cuda.memory_stats(current_device)[
        "allocated_bytes.all.current"
    ]

    total_allocated_bytes = total_bytes - free_bytes
    non_torch_allocations = max(0, total_allocated_bytes - torch_allocated_bytes)

    live_bytes = torch_allocated_bytes + non_torch_allocations

    target_live_bytes = int(total_bytes * memory_utilization)
    available = target_live_bytes - live_bytes - int(reserve_bytes)

    logger.debug(
        "Current live KV headroom: total_bytes=%d, live_bytes=%d, reserve_bytes=%d, "
        "target_live_bytes=%d, available=%d",
        total_bytes,
        live_bytes,
        reserve_bytes,
        target_live_bytes,
        available,
    )
    return max(0, available)


def estimate_indexer_blocks_from_main(main_cm, indexer_cm, main_blocks: int) -> int:
    """
    Derive how many indexer blocks are needed to cover the same token range as main_blocks.

    If both caches use the same block_size (most likely here), this becomes 1:1.
    Otherwise use token-coverage ratio.
    """
    # TODO: mm cache
    if indexer_cm.block_size <= 0:
        return 0
    return int(math.ceil(main_blocks * main_cm.block_size / indexer_cm.block_size))


def solve_main_target_after_shrink(
    args,
    main_cm,
    indexer_cm,
    reserve_bytes: int,
) -> tuple[int, int]:
    """
    After non-main managers have been shrunk, solve the largest main_target such that:
        extra_main_bytes + extra_indexer_bytes <= current_headroom_bytes
    """
    main_cur = main_cm.num_blocks
    main_cap = main_cm.max_num_blocks
    main_bpb = main_cm.estimate_bytes_per_block()

    indexer_cur = indexer_cm.num_blocks
    indexer_cap = indexer_cm.max_num_blocks
    indexer_bpb = indexer_cm.estimate_bytes_per_block()

    headroom_bytes = int(
        get_current_available_kv_cache_memory_bytes(
            memory_utilization=args.infer.memory_utilization,
            reserve_bytes=reserve_bytes,
        )
    )

    lo, hi = main_cur, main_cap
    best_main = main_cur
    best_indexer = indexer_cur

    while lo <= hi:
        mid = (lo + hi) // 2

        derived_indexer = estimate_indexer_blocks_from_main(main_cm, indexer_cm, mid)
        derived_indexer = min(max(0, derived_indexer), indexer_cap)

        delta_main_bytes = max(0, mid - main_cur) * main_bpb
        delta_indexer_bytes = max(0, derived_indexer - indexer_cur) * indexer_bpb
        total_extra_bytes = delta_main_bytes + delta_indexer_bytes

        if total_extra_bytes <= headroom_bytes:
            best_main = mid
            best_indexer = derived_indexer
            lo = mid + 1
        else:
            hi = mid - 1

    logger.info(
        "KV joint solve after shrink: headroom_bytes=%d, "
        "main_cur=%d, indexer_cur=%d, best_main=%d, best_indexer=%d",
        headroom_bytes,
        main_cur,
        indexer_cur,
        best_main,
        best_indexer,
    )
    return best_main, best_indexer


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
