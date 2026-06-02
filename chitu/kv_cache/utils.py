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


def build_layer_id_map_lastlayer(args):
    layer_ids = []
    if get_pp_group().is_last_rank:
        mtp_size = int(getattr(get_global_args().infer, "mtp_size", 1))
        total_n_layers = int(args.models.n_layers) + (1 if mtp_size > 1 else 0)
        layer_ids = [total_n_layers - 1]
    return GlobalLocalMap.from_list(layer_ids)


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
            "max_num_blocks": cm.get_allocatable_max_num_blocks(),
        }

    main_cm = cache_managers["main"]
    main_cur = int(main_cm.num_blocks)

    plan = {"main": main_cur}

    if "indexer" in cache_managers:
        indexer_cm = cache_managers["indexer"]
        indexer_cur = int(indexer_cm.num_blocks)
        indexer_cap = indexer_cm.get_allocatable_max_num_blocks()

        desired_indexer_blocks = estimate_indexer_blocks_from_main(
            main_cm, indexer_cm, main_cur
        )
        desired_indexer_blocks = min(
            max(0, int(desired_indexer_blocks)), int(indexer_cap)
        )

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

    for name, info in infos.items():
        if name in plan:
            continue
        plan[name] = min(int(info["current_blocks"]), int(info["max_num_blocks"]))

    # mtp cache only exists in pp last stage, causing stuck in `reduce_num_block_plan_across_ranks`
    # mtp cache size only depends on request num, shrinking is not required
    plan.pop("mtp", None)
    return plan


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
    main_cap = main_cm.get_allocatable_max_num_blocks()
    main_bpb = int(main_cm.estimate_bytes_per_block())

    if main_bpb <= 0:
        return 0

    live_bytes, target_budget_bytes, allocator_debug_bytes = (
        get_peak_live_and_target_bytes(
            memory_utilization=args.infer.memory_utilization,
            reserve_bytes=reserve_bytes,
        )
    )

    # baseline = everything except current main cache
    baseline_bytes = max(0, int(live_bytes) - int(main_cur) * int(main_bpb))

    allowed_main_bytes = max(0, int(target_budget_bytes) - int(baseline_bytes))
    target_main_blocks = allowed_main_bytes // int(main_bpb)

    logger.info(
        "KV main-only solve: live=%.2fGB target_budget=%.2fGB "
        "baseline=%.2fGB main_cur=%d target_main_blocks=%d main_cap=%d "
        "reserved_peak_gap=%.2fGB",
        _bytes_to_gb(live_bytes),
        _bytes_to_gb(target_budget_bytes),
        _bytes_to_gb(baseline_bytes),
        int(main_cur),
        int(target_main_blocks),
        int(main_cap),
        _bytes_to_gb(allocator_debug_bytes["reserved_peak_gap_bytes"]),
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
    main_cap = main_cm.get_allocatable_max_num_blocks()
    main_bpb = int(main_cm.estimate_bytes_per_block())

    indexer_cur = int(indexer_cm.num_blocks)
    indexer_cap = indexer_cm.get_allocatable_max_num_blocks()
    indexer_bpb = int(indexer_cm.estimate_bytes_per_block())

    live_bytes, target_budget_bytes, allocator_debug_bytes = (
        get_peak_live_and_target_bytes(
            memory_utilization=args.infer.memory_utilization,
            reserve_bytes=reserve_bytes,
        )
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
        derived_indexer = clamp_int(derived_indexer, 1, indexer_cap)

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
        "KV joint solve after shrink: live=%.2fGB target_budget=%.2fGB "
        "baseline=%.2fGB main_cur=%d indexer_cur=%d best_main=%d best_indexer=%d "
        "main_cap=%d indexer_cap=%d reserved_peak_gap=%.2fGB",
        _bytes_to_gb(live_bytes),
        _bytes_to_gb(target_budget_bytes),
        _bytes_to_gb(baseline_bytes),
        int(main_cur),
        int(indexer_cur),
        int(best_main),
        int(best_indexer),
        int(main_cap),
        int(indexer_cap),
        _bytes_to_gb(allocator_debug_bytes["reserved_peak_gap_bytes"]),
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
