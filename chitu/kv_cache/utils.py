# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import gc
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


def get_current_available_kv_cache_memory_bytes(
    memory_utilization=0.98,
    reserve_bytes=0,
) -> int:
    """
    Estimate additional bytes available for KV cache based on current live memory.
    """
    live_bytes, target_budget_bytes = get_current_live_and_target_bytes(
        memory_utilization=memory_utilization,
        reserve_bytes=reserve_bytes,
    )
    return max(0, int(target_budget_bytes) - int(live_bytes))


def get_current_live_and_target_bytes(
    memory_utilization=0.98,
    reserve_bytes=0,
) -> tuple[int, int]:
    """
    Return:
      live_bytes: current live memory bytes
      target_budget_bytes: target live-memory budget after applying
                           memory_utilization and reserve_bytes
    """
    if get_global_args().infer.op_impl == "cpu":
        process = psutil.Process(os.getpid())
        live_bytes = int(process.memory_info().vms)
        target_budget_bytes = int(
            psutil.virtual_memory().total * memory_utilization
        ) - int(reserve_bytes)
        return live_bytes, target_budget_bytes

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

    live_bytes = int(torch_allocated_bytes + non_torch_allocations)
    target_budget_bytes = int(total_bytes * memory_utilization) - int(reserve_bytes)

    logger.debug(
        "Current live KV budget: total_bytes=%d, live_bytes=%d, reserve_bytes=%d, "
        "target_budget_bytes=%d",
        total_bytes,
        live_bytes,
        reserve_bytes,
        target_budget_bytes,
    )
    return live_bytes, target_budget_bytes


def solve_main_target_from_current(
    args,
    main_cm,
    reserve_bytes: int,
) -> int:
    """
    Solve the FINAL feasible main target from current live memory.
    This may shrink or grow main.
    """
    main_cur = int(main_cm.num_blocks)
    main_cap = int(main_cm.max_num_blocks)
    main_bpb = int(main_cm.estimate_bytes_per_block())

    if main_bpb <= 0:
        return 0

    live_bytes, target_budget_bytes = get_current_live_and_target_bytes(
        memory_utilization=args.infer.memory_utilization,
        reserve_bytes=reserve_bytes,
    )

    # baseline = everything except current main cache
    baseline_bytes = max(0, int(live_bytes) - int(main_cur) * int(main_bpb))

    allowed_main_bytes = max(0, int(target_budget_bytes) - int(baseline_bytes))
    target_main_blocks = allowed_main_bytes // int(main_bpb)

    logger.info(
        "KV main-only solve: live_bytes=%d target_budget_bytes=%d "
        "baseline_bytes=%d main_cur=%d target_main_blocks=%d",
        int(live_bytes),
        int(target_budget_bytes),
        int(baseline_bytes),
        int(main_cur),
        int(target_main_blocks),
    )

    return max(0, min(int(target_main_blocks), int(main_cap)))


def solve_main_target_after_shrink(
    args,
    main_cm,
    indexer_cm,
    reserve_bytes: int,
) -> int:
    """
    Return the largest feasible FINAL main target after shrink,
    accounting for coupled indexer size.
    This may shrink or grow main.
    """
    main_cur = int(main_cm.num_blocks)
    main_cap = int(main_cm.max_num_blocks)
    main_bpb = int(main_cm.estimate_bytes_per_block())

    indexer_cur = int(indexer_cm.num_blocks)
    indexer_cap = int(indexer_cm.max_num_blocks)
    indexer_bpb = int(indexer_cm.estimate_bytes_per_block())

    live_bytes, target_budget_bytes = get_current_live_and_target_bytes(
        memory_utilization=args.infer.memory_utilization,
        reserve_bytes=reserve_bytes,
    )

    # baseline = everything except current main + current indexer
    baseline_bytes = max(
        0,
        int(live_bytes)
        - int(main_cur) * int(main_bpb)
        - int(indexer_cur) * int(indexer_bpb),
    )

    lo, hi = 0, int(main_cap)
    best_main = 0
    best_indexer = 0

    while lo <= hi:
        mid = (lo + hi) // 2

        derived_indexer = estimate_indexer_blocks_from_main(main_cm, indexer_cm, mid)
        derived_indexer = min(max(0, int(derived_indexer)), int(indexer_cap))

        total_live_if_mid = (
            int(baseline_bytes)
            + int(mid) * int(main_bpb)
            + int(derived_indexer) * int(indexer_bpb)
        )

        if int(total_live_if_mid) <= int(target_budget_bytes):
            best_main = int(mid)
            best_indexer = int(derived_indexer)
            lo = mid + 1
        else:
            hi = mid - 1

    logger.info(
        "KV joint solve after shrink: live_bytes=%d target_budget_bytes=%d "
        "baseline_bytes=%d main_cur=%d indexer_cur=%d best_main=%d best_indexer=%d",
        int(live_bytes),
        int(target_budget_bytes),
        int(baseline_bytes),
        int(main_cur),
        int(indexer_cur),
        int(best_main),
        int(best_indexer),
    )
    return int(best_main)


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
