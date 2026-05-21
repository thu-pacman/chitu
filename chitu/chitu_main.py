# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import time
import traceback
from logging import getLogger
from typing import Optional
import random
import re
import traceback
from tqdm import tqdm

import torch
import torch.distributed
import zmq
import zmq.asyncio
import msgpack

from chitu.backend import Backend, BackendState
from chitu.device_type import is_nvidia, has_accelerator
from chitu.executor import Executor
from chitu.global_vars import (
    get_global_args,
    get_slot_handle,
    set_global_variables,
    set_quant_variables,
    set_backend_variables,
)
from chitu.models.registry import ModelType
from chitu.moe.load_balancer import get_moe_load_planner
from chitu.scheduler import Scheduler
from chitu.task import (
    PackedTasks,
    PackedTasksBase,
    SerializedPackedTasksPayloadType,
    Task,
    TaskPool,
    TaskType,
    UserRequest,
    TaskCollector,
    DPTaskCollector,
)
from chitu.utils import (
    gen_req_id,
    try_import_opt_dep,
    try_import_and_setup_torch_npu,
    ceil_div,
    gather_str_to_dst_rank,
    get_chitu_bool_env,
)
from chitu.schemas.utils import ModelConfigResolver
from chitu.distributed.parallel_state import get_pp_group, get_world_group
from chitu.logging_utils import setup_chitu_logging
from chitu.metrics import (
    PrometheusMetricsCollector,
    start_prometheus_server_and_metrics_monitor,
    stop_metrics_monitor,
)
from chitu.ops.utils import (
    clear_observed_op_impl_selections,
    emit_observed_op_impl_summary,
)
from chitu.distributed.comm_group import SingletonGroupPlaceholder
from chitu.dp_token_sender import get_dp_token_manager, start_dp_token_manager
from chitu.kv_cache.utils import (
    plan_kv_cache_blocks_after_warmup,
    reduce_num_block_plan_across_ranks,
    estimate_indexer_blocks_from_main,
    solve_main_target_after_shrink,
    solve_main_target_from_current,
    get_peak_live_and_target_bytes,
    cleanup_cuda_if_needed,
    allreduce_min_int,
    clamp_int,
)

numa, has_numa = try_import_opt_dep("numa", "cpu")
cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
deep_ep, has_deep_ep = try_import_opt_dep("deep_ep", "deep_ep")


logger = getLogger(__name__)


_last_step_task_type: Optional[TaskType] = None


def get_last_step_task_type() -> Optional[TaskType]:
    """Return the TaskType (Prefill / Decode) of the most recent local step."""
    return _last_step_task_type


def init_logger():
    setup_chitu_logging()


def init_cache_static():
    if has_accelerator():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _deepseek_v4_compressed_ratio_from_manager_name(manager_name: str) -> Optional[int]:
    if manager_name == "compressed_csa":
        return 4
    if manager_name == "compressed_hca":
        return 128
    match = re.fullmatch(r"compressed_(\d+)", str(manager_name))
    if match:
        return int(match.group(1))
    return None


def _deepseek_v4_compressed_blocks_for_seq_len(
    seq_len: int,
    *,
    ratio: int,
    block_size: int,
    num_reqs: int,
) -> int:
    compressed_len = max(0, int(seq_len)) // int(ratio)
    if compressed_len == 0:
        return 0
    return int(num_reqs) * ceil_div(compressed_len, int(block_size))


def _deepseek_v4_effective_seq_len_for_targets(
    max_seq_len: int,
    compressed_groups: dict[str, dict],
    targets: dict[str, int],
    *,
    num_reqs: int,
) -> int:
    if not compressed_groups:
        return int(max_seq_len)

    lo, hi = 0, int(max_seq_len)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        enough_blocks = True
        for manager_name, group in compressed_groups.items():
            needed = _deepseek_v4_compressed_blocks_for_seq_len(
                mid,
                ratio=int(group["ratio"]),
                block_size=int(group["block_size"]),
                num_reqs=int(num_reqs),
            )
            if int(needed) > int(targets[manager_name]):
                enough_blocks = False
                break
        if enough_blocks:
            best = int(mid)
            lo = mid + 1
        else:
            hi = mid - 1
    return int(min(best, int(max_seq_len)))


def _solve_deepseek_v4_kv_targets_from_seq_len(
    args,
    group_infos: dict[str, dict],
    allowed_group_bytes: int,
) -> tuple[dict[str, int], int, dict[str, int], int]:
    """Solve DeepSeek-V4 KV blocks by logical sequence length.

    The solver keeps sliding-window groups fixed at one page per hot request and
    sizes compressed groups from a single effective logical sequence length.
    It also enforces a one-request max_seq_len floor for each compressed group.
    """

    max_seq_len = int(args.infer.max_seq_len)
    num_hot_req = ceil_div(args.infer.max_batch_size, args.infer.dp_size)

    min_targets: dict[str, int] = {}
    compressed_groups: dict[str, dict] = {}

    for manager_name, info in group_infos.items():
        cap = int(info["cap"])
        ratio = info.get("compress_ratio")
        if ratio is None:
            ratio = _deepseek_v4_compressed_ratio_from_manager_name(manager_name)
        if ratio is not None:
            block_size = int(info["block_size"])
            one_req_blocks = _deepseek_v4_compressed_blocks_for_seq_len(
                max_seq_len,
                ratio=int(ratio),
                block_size=block_size,
                num_reqs=1,
            )
            # PagedKVCache cannot be reallocated to zero blocks. Even when
            # max_seq_len < ratio, keep a minimal physical allocation.
            min_targets[manager_name] = clamp_int(max(1, one_req_blocks), 1, cap)
            compressed_groups[manager_name] = {
                "ratio": int(ratio),
                "block_size": block_size,
            }
            continue

        if info["fixed_num_blocks"] or manager_name == "main":
            min_targets[manager_name] = cap
            continue

        logger.warning(
            "DeepSeek-V4 KV solve found unknown non-compressed manager group %s; "
            "keeping its current blocks as the minimum target",
            manager_name,
        )
        min_targets[manager_name] = clamp_int(int(info["blocks"]), 1, cap)

    def target_for_seq_len(seq_len: int) -> dict[str, int]:
        targets = dict(min_targets)
        for manager_name, group in compressed_groups.items():
            info = group_infos[manager_name]
            batch_blocks = _deepseek_v4_compressed_blocks_for_seq_len(
                seq_len,
                ratio=int(group["ratio"]),
                block_size=int(group["block_size"]),
                num_reqs=int(num_hot_req),
            )
            targets[manager_name] = clamp_int(
                max(int(min_targets[manager_name]), int(batch_blocks)),
                1,
                int(info["cap"]),
            )
        return targets

    def target_bytes(targets: dict[str, int]) -> int:
        return sum(
            int(targets[manager_name])
            * int(group_infos[manager_name]["bytes_per_block"])
            for manager_name in group_infos
        )

    targets = dict(min_targets)
    if target_bytes(targets) <= int(allowed_group_bytes):
        lo, hi = 0, max_seq_len
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = target_for_seq_len(mid)
            if target_bytes(candidate) <= int(allowed_group_bytes):
                targets = candidate
                lo = mid + 1
            else:
                hi = mid - 1
    else:
        logger.warning(
            "DeepSeek-V4 KV warmup budget %.2fGB is below one-request max_seq_len "
            "floor %.2fGB; using one-request floor targets",
            float(allowed_group_bytes) / 1e9,
            float(target_bytes(targets)) / 1e9,
        )

    effective_max_seq_len = _deepseek_v4_effective_seq_len_for_targets(
        max_seq_len,
        compressed_groups,
        targets,
        num_reqs=1,
    )
    batch_effective_max_seq_len = _deepseek_v4_effective_seq_len_for_targets(
        max_seq_len,
        compressed_groups,
        targets,
        num_reqs=num_hot_req,
    )

    return (
        targets,
        int(effective_max_seq_len),
        min_targets,
        int(batch_effective_max_seq_len),
    )


def _auto_set_deepseek_v4_num_blocks_after_warmup(
    args,
    paged_caches,
):
    reserve_bytes = 512 << 20
    cache_groups = {}
    for name, cache in paged_caches.items():
        manager_name = getattr(cache, "manager_name", name)
        cache_groups.setdefault(manager_name, []).append((name, cache))

    group_infos = {}
    num_hot_req = ceil_div(args.infer.max_batch_size, args.infer.dp_size)
    for manager_name, caches in cache_groups.items():
        block_counts = {int(cache.num_blocks) for _, cache in caches}
        if len(block_counts) != 1:
            logger.warning(
                "DeepSeek-V4 manager group %s has mismatched cache block counts: %s; "
                "using the minimum for warmup solve",
                manager_name,
                sorted(block_counts),
            )
        cap = min(int(cache.get_allocatable_max_num_blocks()) for _, cache in caches)
        block_sizes = {int(cache.block_size) for _, cache in caches}
        if len(block_sizes) != 1:
            logger.warning(
                "DeepSeek-V4 manager group %s has mismatched block sizes: %s; "
                "using the maximum for warmup solve",
                manager_name,
                sorted(block_sizes),
            )
        group_infos[manager_name] = {
            "caches": caches,
            "blocks": min(block_counts),
            "bytes_per_block": sum(
                int(cache.estimate_bytes_per_block()) for _, cache in caches
            ),
            "cap": int(cap),
            "block_size": max(block_sizes),
            "compress_ratio": _deepseek_v4_compressed_ratio_from_manager_name(
                manager_name
            ),
            "fixed_num_blocks": any(
                bool(getattr(cache, "fixed_num_blocks", False)) for _, cache in caches
            ),
            "max_blocks_per_req": max(1, ceil_div(int(cap), int(num_hot_req))),
        }

    if not group_infos:
        logger.warning(
            "skip DeepSeek-V4 KV resize because no paged cache groups were found"
        )
        return

    for manager_name, info in group_infos.items():
        if info["bytes_per_block"] <= 0:
            logger.warning(
                "skip DeepSeek-V4 KV resize because manager group %s has zero bytes_per_block",
                manager_name,
            )
            return

    live_bytes, target_budget_bytes, allocator_debug_bytes = (
        get_peak_live_and_target_bytes(
            memory_utilization=args.infer.memory_utilization,
            reserve_bytes=reserve_bytes,
        )
    )
    current_group_bytes = sum(
        info["blocks"] * info["bytes_per_block"] for info in group_infos.values()
    )
    baseline_bytes = max(0, int(live_bytes) - int(current_group_bytes))
    allowed_group_bytes = max(0, int(target_budget_bytes) - int(baseline_bytes))

    targets, effective_max_seq_len, min_targets, batch_effective_max_seq_len = (
        _solve_deepseek_v4_kv_targets_from_seq_len(
            args,
            group_infos,
            allowed_group_bytes,
        )
    )

    targets = reduce_num_block_plan_across_ranks(targets)
    effective_max_seq_len = allreduce_min_int(int(effective_max_seq_len))
    batch_effective_max_seq_len = allreduce_min_int(int(batch_effective_max_seq_len))

    logger.info(
        "DeepSeek-V4 KV solve: live=%.2fGB target_budget=%.2fGB "
        "baseline=%.2fGB allowed_group=%.2fGB effective_max_seq_len=%d "
        "batch_effective_max_seq_len=%d num_hot_req=%d targets=%s "
        "min_targets=%s caps=%s bytes_per_block=%s reserved_peak_gap=%.2fGB",
        float(live_bytes) / 1e9,
        float(target_budget_bytes) / 1e9,
        float(baseline_bytes) / 1e9,
        float(allowed_group_bytes) / 1e9,
        int(effective_max_seq_len),
        int(batch_effective_max_seq_len),
        int(num_hot_req),
        targets,
        min_targets,
        {name: info["cap"] for name, info in group_infos.items()},
        {name: info["bytes_per_block"] for name, info in group_infos.items()},
        float(allocator_debug_bytes["reserved_peak_gap_bytes"]) / 1e9,
    )

    for manager_name, info in group_infos.items():
        target_blocks = int(targets[manager_name])
        for cache_name, cache in info["caches"]:
            current_blocks = int(cache.num_blocks)
            if current_blocks != target_blocks:
                cache.realloc(target_blocks)
                logger.info(
                    "DeepSeek-V4 %s cache resized from %d to %d blocks for manager %s",
                    cache_name,
                    current_blocks,
                    target_blocks,
                    manager_name,
                )
                cleanup_cuda_if_needed()

    get_global_args().infer.num_blocks = int(
        targets.get("main", next(iter(targets.values())))
    )
    if Backend.cache_managers:
        for dp_rank, dp_rank_managers in enumerate(Backend.cache_managers):
            for manager_name, target_blocks in targets.items():
                manager = dp_rank_managers.get(manager_name)
                if manager is None:
                    continue
                if int(manager.num_blocks) != int(target_blocks):
                    manager.realloc(int(target_blocks))
                    logger.info(
                        "scheduler dp_rank=%d DeepSeek-V4 %s cache manager synced to %d blocks",
                        dp_rank,
                        manager_name,
                        int(target_blocks),
                    )

    if torch.distributed.get_rank() == 0:
        for scheduler in Backend.schedulers:
            scheduler.reset_kvcache_block_threshold()


def _auto_set_num_blocks_after_warmup(args):
    if not (args.infer.cache_type == "paged" and args.infer.num_blocks == -1):
        logger.info(
            f"skip auto set num blocks after warmup because {args.infer.num_blocks=}"
        )
        return

    pd_cfg = args.dp_config.router.pd_disaggregation
    if pd_cfg.enabled:
        sched_type = args.scheduler.type.lower()
        is_pd_decode_only = "decode_only" in sched_type
    else:
        is_pd_decode_only = False
    full_warmup = args.infer.full_warmup

    paged_caches = {}
    for name, cache in Backend.cache_dict.items():
        if (
            hasattr(cache, "realloc")
            and hasattr(cache, "num_blocks")
            and hasattr(cache, "max_num_blocks")
        ):
            paged_caches[name] = cache

    if "main" not in paged_caches:
        logger.warning(
            "skip auto set num blocks after warmup because main cache manager is missing"
        )
        return

    if is_pd_decode_only and not full_warmup:
        if args.models.type == ModelType.DEEPSEEK_V4:
            current_targets = {}
            for name, cache in paged_caches.items():
                manager_name = getattr(cache, "manager_name", name)
                current_targets[manager_name] = min(
                    int(cache.num_blocks),
                    current_targets.get(manager_name, int(cache.num_blocks)),
                )
            current_main_blocks = int(
                current_targets.get("main", next(iter(current_targets.values())))
            )
        else:
            current_targets = {"main": int(paged_caches["main"].num_blocks)}
            current_main_blocks = int(current_targets["main"])
        get_global_args().infer.num_blocks = int(current_main_blocks)
        if Backend.cache_managers:
            for dp_rank_managers in Backend.cache_managers:
                for manager_name, target_blocks in current_targets.items():
                    manager = dp_rank_managers.get(manager_name)
                    if manager is None:
                        continue
                    mgr_current = int(manager.num_blocks)
                    if int(target_blocks) != int(mgr_current):
                        manager.realloc(int(target_blocks))
                        logger.info(
                            "scheduler %s cache manager synced to %d blocks after warmup skip",
                            manager_name,
                            int(target_blocks),
                        )
        if torch.distributed.get_rank() == 0:
            for scheduler in Backend.schedulers:
                scheduler.reset_kvcache_block_threshold()
        logger.warning(
            "skip auto set num blocks after warmup for PD decode-only without "
            "full_warmup; direct warmup only exercises batch_size=1 and seq_len=1, "
            "so keeping current KV blocks=%s",
            current_targets,
        )
        return

    for name, cache in paged_caches.items():
        bytes_per_block = (
            int(cache.estimate_bytes_per_block())
            if hasattr(cache, "estimate_bytes_per_block")
            else -1
        )
        effective_cap = cache.get_allocatable_max_num_blocks()
        logger.info(
            "%s warmup stats before planning: bytes_per_block=%s current_blocks=%s "
            "max_num_blocks=%s effective_cap=%s",
            name,
            bytes_per_block,
            int(cache.num_blocks),
            int(cache.max_num_blocks),
            int(effective_cap),
        )

    if args.models.type == ModelType.DEEPSEEK_V4 and len(paged_caches) > 1:
        _auto_set_deepseek_v4_num_blocks_after_warmup(
            args,
            paged_caches,
        )
        return

    plan = plan_kv_cache_blocks_after_warmup(args, paged_caches)
    plan = reduce_num_block_plan_across_ranks(plan)
    logger.info("KV cache reduced pre-plan after warmup: %s", plan)

    # shrink non-main managers first
    for name, cm in paged_caches.items():
        if name == "main":
            continue

        current_blocks = int(cm.num_blocks)
        target_blocks = int(plan.get(name, current_blocks))
        target_blocks = clamp_int(target_blocks, 1, cm.get_allocatable_max_num_blocks())

        if target_blocks < current_blocks:
            cm.realloc(int(target_blocks))
            logger.info(
                "%s cache manager shrunk to %d blocks before joint solve",
                name,
                int(target_blocks),
            )

    cleanup_cuda_if_needed()

    main_cm = paged_caches["main"]
    indexer_cm = paged_caches.get("indexer")

    main_current = int(main_cm.num_blocks)
    main_cap = main_cm.get_allocatable_max_num_blocks()
    reserve_bytes = 512 << 20  # 512 MiB
    min_decode_blocks = 1
    if is_pd_decode_only:
        min_decode_blocks = int(
            getattr(main_cm, "max_blocks_per_req", 0)
            or ceil_div(int(args.infer.max_seq_len), int(main_cm.block_size))
        )
        min_decode_blocks = clamp_int(min_decode_blocks, 1, int(main_cap))

    if indexer_cm is not None:
        solved_main = solve_main_target_after_shrink(
            args=args,
            main_cm=main_cm,
            indexer_cm=indexer_cm,
            reserve_bytes=reserve_bytes,
        )

        solved_main = allreduce_min_int(int(solved_main))
        solved_main = clamp_int(int(solved_main), 1, int(main_cap))

        final_indexer_target = estimate_indexer_blocks_from_main(
            main_cm, indexer_cm, int(solved_main)
        )
        final_indexer_target = clamp_int(
            final_indexer_target,
            1,
            indexer_cm.get_allocatable_max_num_blocks(),
        )
        final_indexer_target = allreduce_min_int(int(final_indexer_target))

        final_main_target = int(solved_main)

        logger.info(
            "KV final joint targets: main_current=%d final_main_target=%d "
            "final_indexer_target=%d reserve_bytes=%d",
            int(main_current),
            int(final_main_target),
            int(final_indexer_target),
            int(reserve_bytes),
        )
    else:
        final_main_target = solve_main_target_from_current(
            args=args,
            main_cm=main_cm,
            reserve_bytes=reserve_bytes,
        )
        final_main_target = allreduce_min_int(int(final_main_target))
        final_main_target = clamp_int(int(final_main_target), 1, int(main_cap))
        final_indexer_target = None

    if is_pd_decode_only and int(final_main_target) < int(min_decode_blocks):
        logger.warning(
            "PD decode-only warmup solve requested %d main KV blocks, but at least "
            "%d are required to hold one max_seq_len request; clamping upward",
            int(final_main_target),
            int(min_decode_blocks),
        )
        final_main_target = int(min_decode_blocks)

    # If main needs shrink, do it early to release memory before any later growth.
    main_current = int(main_cm.num_blocks)
    if int(final_main_target) < int(main_current):
        main_cm.realloc(int(final_main_target))
        logger.info(
            "main cache manager pre-shrunk from %d to %d blocks before final placement",
            int(main_current),
            int(final_main_target),
        )
        cleanup_cuda_if_needed()

    # Place indexer to the final target if present. Allow both shrink and grow.
    if indexer_cm is not None:
        indexer_current = int(indexer_cm.num_blocks)
        final_indexer_target = clamp_int(
            final_indexer_target,
            1,
            indexer_cm.get_allocatable_max_num_blocks(),
        )
        if int(final_indexer_target) != int(indexer_current):
            indexer_cm.realloc(int(final_indexer_target))
            logger.info(
                "indexer cache manager resized from %d to %d blocks before main safe solve",
                int(indexer_current),
                int(final_indexer_target),
            )
            cleanup_cuda_if_needed()

    # Recheck main safe target from current memory after indexer is in place
    main_current = int(main_cm.num_blocks)
    safe_main_target = solve_main_target_from_current(
        args=args,
        main_cm=main_cm,
        reserve_bytes=reserve_bytes,
    )
    safe_main_target = min(int(safe_main_target), int(final_main_target))
    safe_main_target = clamp_int(
        safe_main_target,
        1,
        main_cm.get_allocatable_max_num_blocks(),
    )
    safe_main_target = allreduce_min_int(int(safe_main_target))
    if is_pd_decode_only and int(safe_main_target) < int(min_decode_blocks):
        logger.warning(
            "PD decode-only safe main KV target %d is smaller than the one-request "
            "floor %d; clamping upward",
            int(safe_main_target),
            int(min_decode_blocks),
        )
        safe_main_target = int(min_decode_blocks)

    logger.info(
        "KV safe main target before final resize: current=%d final_main_target=%d safe_main_target=%d",
        int(main_current),
        int(final_main_target),
        int(safe_main_target),
    )

    # Final resize main
    main_current = int(main_cm.num_blocks)
    if int(safe_main_target) != int(main_current):
        main_cm.realloc(int(safe_main_target))
        logger.info(
            "main cache manager resized from %d to %d blocks after warmup",
            int(main_current),
            int(safe_main_target),
        )
    else:
        logger.info(
            "main cache manager keeps %d blocks after warmup",
            int(main_current),
        )

    cleanup_cuda_if_needed()

    final_main_blocks = int(main_cm.num_blocks)
    final_main_blocks = allreduce_min_int(int(final_main_blocks))
    get_global_args().infer.num_blocks = int(final_main_blocks)

    if Backend.cache_managers:
        for dp_rank, dp_rank_managers in enumerate(Backend.cache_managers):
            for name, cache in paged_caches.items():
                manager = dp_rank_managers.get(name)
                if manager is None:
                    continue

                final_blocks = int(cache.num_blocks)
                mgr_current = int(manager.num_blocks)
                if int(final_blocks) != int(mgr_current):
                    manager.realloc(int(final_blocks))
                    logger.info(
                        "scheduler dp_rank=%d %s cache manager synced to %d blocks after warmup",
                        dp_rank,
                        name,
                        int(final_blocks),
                    )

    # finalize other non-main managers
    for name, cm in paged_caches.items():
        if name == "main":
            continue

        logger.info(
            "%s cache keeps %d blocks after warmup",
            name,
            int(cm.num_blocks),
        )

    if torch.distributed.get_rank() == 0:
        for scheduler in Backend.schedulers:
            scheduler.reset_kvcache_block_threshold()


def _emit_observed_op_impl_summary_after_warmup():
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() != 0
    ):
        return False
    return emit_observed_op_impl_summary(target_logger=logger)


def _warmup_via_taskpool(args):
    rank = torch.distributed.get_rank()

    # Turn ON MoE planner warmup mode on all ranks
    planner = get_moe_load_planner()
    if planner is not None:
        planner.set_warmup_mode(True)

    logger.info("Starting inference system warmup...")

    init_cache_static()
    num_warmup_reqs = args.infer.max_batch_size
    prefill_chunk_size = args.infer.prefill_chunk_size
    _mtp_size = get_global_args().infer.mtp_size
    _n_decode_steps = 2 if get_global_args().infer.schedule_overlap else 1
    _warmup_max_new_tokens = (1 + _n_decode_steps) * _mtp_size
    if prefill_chunk_size is not None:
        warmup_seq_len = max(
            min(
                prefill_chunk_size // num_warmup_reqs,
                args.infer.max_seq_len - _warmup_max_new_tokens,
            ),
            1,
        )
    else:
        logger.warning(
            "infer.prefill_chunk_size is not set, GPU memory usage estimation may be incorrect (may cause OOM)"
        )
        warmup_seq_len = 1
        prefill_chunk_size = args.infer.max_seq_len * args.infer.max_batch_size
    if rank == 0:
        for i in range(num_warmup_reqs):
            req = UserRequest.create_mock(
                input_len=warmup_seq_len,
                request_id=f"{gen_req_id()}",
                max_new_tokens=_warmup_max_new_tokens,
                temperature=0.7,
                top_k=1,
            )
            task = Task(f"{req.request_id}", req, stop_with_eos=False)
            TaskPool.add(task)
        logger.info(f"Added {num_warmup_reqs} warmup requests to TaskPool")
        for scheduler in Backend.schedulers:
            scheduler.start_warmup()

    # Prefill phase
    # In DP chunk prefill, each schedule processes approximately `prefill_chunk_size` tokens across the whole DP group.
    # Due to per-rank budget constraints and uneven task distribution, some tokens may be left unprocessed.
    # Example: DP2, chunk=16, max_batch_size=5 (创建 5 个 warmup 任务), 每任务 3 tokens
    #   - Budget: Rank0=8, Rank1=8 (chunk_size 均分给各 rank)
    #   - Tasks: Rank0 分到 3 个任务 (round robin), Rank1 分到 2 个任务
    #   - Actual: Rank0 处理 8 tokens (3+3+2, task4 剩 1 token), Rank1 处理 6 tokens (3+3)
    #   - Result: 需要 2 轮迭代来处理完所有 15 tokens
    total_tokens = warmup_seq_len * num_warmup_reqs

    # Calculate required iterations considering DP task distribution
    # NOTE: required iterations maybe wrong when pp_size > 1
    if get_global_args().infer.dp_size > 1:
        # In DP mode, tasks are distributed via round-robin, and each rank has limited budget
        # Worst case: most tasks go to one rank, requiring more iterations
        # Example:
        # dp_size = 2 prefill_chunk_size = 16
        # per_rank_budget = 16 // 2 = 8
        # max_tasks_per_rank = ceil_div(5, 2) = 3  # round-robin 最多分到 3 个任务
        # max_tokens_per_rank = 3 × 3 = 9
        # num_required = ceil_div(9, 8) = 2
        dp_size = get_global_args().infer.dp_size
        per_rank_budget = prefill_chunk_size // dp_size
        max_tasks_per_rank = ceil_div(num_warmup_reqs, dp_size)
        max_tokens_per_rank = max_tasks_per_rank * warmup_seq_len
        # Iterations needed for the busiest rank (usually rank 0)
        num_required_prefill_schedules = ceil_div(max_tokens_per_rank, per_rank_budget)
    else:
        num_required_prefill_schedules = ceil_div(total_tokens, prefill_chunk_size)
    num_required_decode_schedules = _n_decode_steps * _mtp_size

    logger.info(
        f"Warmup: total_tokens={total_tokens}, chunk_size={prefill_chunk_size}, "
        f"prefill_iters={num_required_prefill_schedules}"
    )

    if rank == 0:
        prefill_iter = 0
        while True:
            status = chitu_run()
            if status != SerializedPackedTasksPayloadType.NoneType:
                prefill_iter += 1
            prefill_remaining = sum(
                1
                for task in TaskPool.pool.values()
                if task.task_type == TaskType.Prefill
            )
            if prefill_remaining == 0:
                break
        if prefill_iter != num_required_prefill_schedules:
            logger.warning(
                f"Warmup incomplete: # of prefill iterations is {prefill_iter}, "
                f"which is excepted to be {num_required_prefill_schedules}."
            )
        assert (
            len(TaskPool.pool) == num_warmup_reqs
        ), f"Expected {num_warmup_reqs} tasks after prefill, found {len(TaskPool.pool)}"
        # End warmup prefill phase before starting decode phase
        for scheduler in Backend.schedulers:
            scheduler.end_warmup()

        decode_iter = 0
        while not TaskPool.all_finished():
            status = chitu_run()
            if status != SerializedPackedTasksPayloadType.NoneType:
                decode_iter += 1
        if decode_iter != num_required_decode_schedules:
            logger.warning(
                f"Warmup incomplete: # of decode iterations is {decode_iter}, "
                f"which is excepted to be {num_required_decode_schedules}."
            )
        # stop other ranks
        Backend.executor.step(
            PackedTasksBase(
                num_tasks=0,
                payload_type=SerializedPackedTasksPayloadType.TerminateBackend,
            )
        )
    else:
        while not chitu_is_terminated():
            chitu_run()
    # restart backend
    chitu_start()

    logger.info("Inference system warmup completed")

    planner = get_moe_load_planner()
    if planner is not None:
        planner.set_warmup_mode(False)


def _warmup_backend_direct(
    args,
    local_max_bs=1,
    decode_steps=2,
    bs_descend=0,
    *,
    skip_model_prefill: bool = False,
    skip_model_decode: bool = False,
):
    logger.info(
        f"Starting local backend warmup (direct) with local max batch size {local_max_bs}..."
    )
    init_cache_static()

    req_ids = [f"__warmup_{i}__" for i in range(local_max_bs)]
    is_pp_first_rank = get_pp_group() is None or get_pp_group().is_first_rank
    is_deepseek_v4 = args.models.type == ModelType.DEEPSEEK_V4
    if is_pp_first_rank:
        tokens = torch.randint(
            1,
            args.models.vocab_size,
            size=(local_max_bs,),
            device="cuda",
            dtype=torch.int64,
        )
    elif is_deepseek_v4:
        tokens = torch.randn(
            local_max_bs,
            args.models.hc_mult,
            args.models.dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
    else:
        tokens = torch.randn(
            local_max_bs,
            args.models.dim,
            device="cuda",
            dtype=torch.get_default_dtype(),
        )

    all_tasks = PackedTasksBase(local_max_bs, task_ids=req_ids)
    if Backend.cache_managers:
        new_cache_ids = {}
        for name, manager in Backend.cache_managers[0].items():
            new_cache_ids[name] = [random.randrange(manager.num_blocks)]
        all_tasks.new_cache_ids_list = [new_cache_ids for _ in range(local_max_bs)]
    all_tasks.tokens = [[1] for _ in range(local_max_bs)]

    # Prefill
    for cache in Backend.cache_dict.values():
        cache.prepare_cache_prefill(all_tasks)
    PrometheusMetricsCollector.update_GPU_usage()

    # decode_only 下，Decode 不需要跑 prefill；但需要把 cache 的 seq_len
    # 和 block_table 初始化到可 decode 的状态（否则后续 prepare_cache_decode 会找不到 req_id）
    # 仅做 cache prepare，避免 prefill 算子在 Decode 进程里被执行，从而触发所谓的“illegal memory access”
    # MoE 的 task_type 需要设置在 moe_impl 上，否则 decode_only
    # 的 warmup 会在 MoE layer 里因为 task_type=None 触发 KeyError(None)

    output_token_offsets = torch.arange(
        local_max_bs, dtype=torch.int32, device=tokens.device
    )
    if not skip_model_prefill:
        Backend.model.prefill(tokens, output_token_offsets)

    # Decode steps
    if not skip_model_decode:
        for i in tqdm(
            range(max(1, decode_steps)), desc="finished warmup decode iterations"
        ):
            curr_bs = local_max_bs - i * bs_descend
            curr_req_ids = req_ids[:curr_bs]

            cur_tasks = PackedTasksBase(curr_bs, task_ids=curr_req_ids)
            cur_tasks.new_cache_ids_list = []
            cur_tasks.tokens = [[1] for _ in range(curr_bs)]

            for cache in Backend.cache_dict.values():
                cache.prepare_cache_decode(cur_tasks)
            PrometheusMetricsCollector.update_GPU_usage()

            # direct warmup 绕过了 executor，因此必须在这里显式设置
            if (
                hasattr(Backend.model, "moe_impl")
                and Backend.model.moe_impl is not None
            ):
                Backend.model.moe_impl.prepare(TaskType.Decode, curr_bs)

            if is_pp_first_rank:
                step_token = torch.randint(
                    1,
                    args.models.vocab_size,
                    size=(curr_bs,),
                    device="cuda",
                    dtype=torch.int64,
                )
            elif is_deepseek_v4:
                step_token = torch.randn(
                    curr_bs,
                    args.models.hc_mult,
                    args.models.dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
            else:
                step_token = torch.randn(
                    curr_bs,
                    args.models.dim,
                    device="cuda",
                    dtype=torch.get_default_dtype(),
                )
            _ = Backend.model.decode(step_token, curr_bs)

    # Clean KV for this request
    for cache in Backend.cache_dict.values():
        cache.finalize_cache_all_decode(all_tasks)
    PrometheusMetricsCollector.update_GPU_usage()
    logger.info("Local backend warmup (direct) completed")


def warmup_engine(args):
    # Router 进程不做 warmup
    if args.dp_config.router.is_router:
        return

    clear_observed_op_impl_selections()

    # PD分离→direct，非PD→taskpool
    pd_enabled = args.dp_config.router.pd_disaggregation.enabled

    runner = "direct" if pd_enabled else "taskpool"
    sched_type = str(args.scheduler.type).lower()
    skip_model_prefill = "decode_only" in sched_type
    skip_model_decode = "prefill_only" in sched_type
    full_warmup = args.infer.full_warmup

    def _log_skip_prefill():
        if skip_model_prefill:
            logger.info(
                "[warmup][direct] decode_only detected, skipping model.prefill in warmup"
            )

    if runner == "taskpool":
        _warmup_via_taskpool(args)
    elif not full_warmup:
        _log_skip_prefill()
        _warmup_backend_direct(
            args,
            decode_steps=2,
            skip_model_prefill=bool(skip_model_prefill),
            skip_model_decode=bool(skip_model_decode),
        )

    if full_warmup:
        if runner == "direct":
            logger.info("[warmup] full_warmup enabled, skip base warmup")
        _log_skip_prefill()
        max_reqs_per_dp = ceil_div(args.infer.max_batch_size, args.infer.dp_size)
        if args.infer.pp_size > 1:
            if (
                args.scheduler.pp_config.pp_micro_batch_size_decode == "max"
                or get_slot_handle()
            ):
                local_max_bs = ceil_div(max_reqs_per_dp, args.infer.pp_size)
            else:
                local_max_bs = args.scheduler.pp_config.pp_micro_batch_size_decode
        else:
            local_max_bs = max_reqs_per_dp
        _warmup_backend_direct(
            args,
            local_max_bs=local_max_bs,
            decode_steps=local_max_bs,
            bs_descend=1,
            skip_model_prefill=bool(skip_model_prefill),
            skip_model_decode=bool(skip_model_decode),
        )

    _auto_set_num_blocks_after_warmup(args)
    _emit_observed_op_impl_summary_after_warmup()


def check_checkpoint_path(args):
    if args.models.ckpt_dir is None:
        if not getattr(args.models, "is_pro", False):
            raise ValueError(
                f"No checkpoint path provided. You can set it in command line by adding "
                f"`models.ckpt_dir=<path>`. The model {args.models.name} can be downloaded "
                f"from {args.models.source}"
            )
        else:
            raise ValueError(
                f"No checkpoint path provided. You can set it in command line by adding "
                f"`models.ckpt_dir=<path>`. The model {args.models.name} is part of "
                f"chitu-pro, which may be obtained by concatting solution@chitu.ai"
            )
    if args.models.tokenizer_path is None:
        logger.info(
            f"Using {args.models.ckpt_dir} as the path to tokenizer. If the tokenizer has a different path, please set in command line by adding `models.tokenizer_path=<path>`"
        )
        args.models.tokenizer_path = args.models.ckpt_dir
    if hasattr(args.models, "processor_path") and args.models.processor_path is None:
        logger.info(
            f"Using {args.models.ckpt_dir} as the path to processor. If the processor has a different path, please set in command line by adding `models.processor_path=<path>`"
        )
        args.models.processor_path = args.models.ckpt_dir


def _has_cpu_layer(args) -> bool:
    if (backend_config := args.models.get("backend_config")) is not None:
        for config in backend_config.get("backend", []):
            if (pattern := config.get("model")) is not None:
                if re.match(pattern, args.models.name.lower()):
                    for rule in config.rules:
                        if rule.get("backend") == "cpuinfer":
                            return True
    return False


def chitu_init(args):
    debug = get_chitu_bool_env("CHITU_DEBUG", False)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    if (
        is_nvidia()
        and torch.distributed.is_nccl_available()
        and torch.cuda.nccl.version() <= (2, 21, 5)
    ):
        os.environ["NCCL_NVLS_NCHANNELS"] = "32"

    init_logger()

    ###################################################################
    # Deal with legacy arguments
    if hasattr(args.infer, "soft_fp8") and args.infer.soft_fp8:
        logger.warning(
            "Argument `infer.soft_fp8=True` is deprecated. Use `infer.raise_lower_bit_float_to=bfloat16` instead."
        )
        args.infer.raise_lower_bit_float_to = "bfloat16"
    if hasattr(args, "dtype") and args.dtype is not None:
        logger.warning(
            "Argument `dtype` is deprecated. Use `float_16bit_variant` instead."
        )
        args.float_16bit_variant = args.dtype
    if hasattr(args.infer, "do_load") and not args.infer.do_load:
        logger.warning(
            "Argument `infer.do_load=False` is deprecated. Use `debug.skip_model_load=True` instead."
        )
        args.debug.skip_model_load = True
    if hasattr(args.infer, "max_reqs") and args.infer.max_reqs is not None:
        args.infer.max_batch_size = args.infer.max_reqs
        logger.warning(
            f"Argument `infer.max_reqs={args.infer.max_reqs}` is deprecated. Use `infer.max_batch_size={args.infer.max_batch_size}` instead."
        )
    # max_concurrent_requests default: max_batch_size * 2
    if getattr(args.infer, "max_concurrent_requests", None) is not None:
        pass
    else:
        args.infer.max_concurrent_requests = args.infer.max_batch_size * 2
        logger.info(
            f"infer.max_concurrent_requests not set, defaulting to max_batch_size * 2 ({args.infer.max_concurrent_requests})"
        )

    if (
        hasattr(args.scheduler.pp_config, "prefill_num_tasks_divided_by_pp")
        and not args.scheduler.pp_config.prefill_num_tasks_divided_by_pp
    ):
        logger.warning(
            "Argument `scheduler.pp_config.prefill_num_tasks_divided_by_pp=False` is deprecated. Use `scheduler.pp_config.pp_micro_batch_size_prefill=<num>` instead."
        )
        assert (
            hasattr(args.scheduler.pp_config, "prefill_num_tasks")
            and args.scheduler.pp_config.prefill_num_tasks
        )
        args.scheduler.pp_config.pp_micro_batch_size_prefill = (
            args.scheduler.pp_config.prefill_num_tasks
        )
    if (
        hasattr(args.scheduler.pp_config, "enforce_decode_num_tasks_max")
        and not args.scheduler.pp_config.enforce_decode_num_tasks_max
    ):
        logger.warning(
            "Argument `scheduler.pp_config.enforce_decode_num_tasks_max=False` is deprecated. Use `scheduler.pp_config.pp_micro_batch_size_decode=<num>` instead."
        )
        assert (
            hasattr(args.scheduler.pp_config, "decode_num_tasks")
            and args.scheduler.pp_config.decode_num_tasks
        )
        args.scheduler.pp_config.pp_micro_batch_size_decode = (
            args.scheduler.pp_config.decode_num_tasks
        )

    ###################################################################
    # Deal with automatic arguments

    if args.infer.device_ids is None:
        args.infer.device_ids = [i % local_world_size for i in range(world_size)]
    if len(args.infer.device_ids) != world_size:
        raise ValueError(
            f"len(infer.device_ids) ({len(args.infer.device_ids)}) must be equalt to world_size ({world_size})"
        )

    # prefill_chunk_size default value: 4096 * dp_size
    if args.infer.prefill_chunk_size == "auto":
        args.infer.prefill_chunk_size = 4096 * args.infer.dp_size

    if (
        args.infer.prefill_chunk_size is not None
        and args.infer.prefill_chunk_size
        > args.infer.max_batch_size * args.infer.max_seq_len
    ):
        logger.warning(
            f"infer.prefill_chunk_size ({args.infer.prefill_chunk_size}) is larger than "
            f"infer.max_batch_size ({args.infer.max_batch_size}) * infer.max_seq_len "
            f"({args.infer.max_seq_len}), which has no effect. Reducing it to "
            f"infer.max_batch_size * infer.max_seq_len."
        )
        args.infer.prefill_chunk_size = (
            args.infer.max_batch_size * args.infer.max_seq_len
        )

    if args.infer.prefill_chunk_size is not None:
        if args.infer.pp_size > 1 and args.infer.cache_type == "skew":
            logger.warning(
                "Disabling infer.prefill_chunk_size because it is not compatible with PP+skew yet"
            )
            args.infer.prefill_chunk_size = None

    # Auto setting for binding process to CPU NUMA
    if args.infer.bind_process_to_cpu == "auto":
        if not has_numa:
            logger.warning(
                "Optional dependency '[numa]' is mising. Disabling NUMA binding."
            )
            args.infer.bind_process_to_cpu = "none"
        elif not numa.available():
            logger.warning(
                "NUMA is not support on this OS or hardware platform. Disabling NUMA binding."
            )
            args.infer.bind_process_to_cpu = "none"
        elif _has_cpu_layer(args):
            if numa.get_max_node() + 1 < local_world_size:
                logger.warning(
                    "Disable NUMA binding due to insufficient NUMA nodes. Is is an inefficient setting of CPU inference."
                )
                args.infer.bind_process_to_cpu = "none"
            else:
                args.infer.bind_process_to_cpu = "one_numa_per_rank"
        else:
            args.infer.bind_process_to_cpu = "numa_near_device"

    if args.infer.use_cuda_graph == "auto":
        if args.models.name in [
            "Mixtral-8x7B-Instruct-v0.1",
            "Qwen3-30B-A3B-mix-fp4-fp8",
            "Qwen3-Next-80B-A3B-Instruct",
        ]:
            args.infer.use_cuda_graph = False
        elif (
            args.infer.ep_size > 1
            and args.infer.dp_size > 1
            and (args.infer.tp_size > 1 or not has_deep_ep)
        ):
            args.infer.use_cuda_graph = False
        elif args.infer.attn_type == "ref":
            args.infer.use_cuda_graph = False
        elif args.infer.op_impl is not None and args.infer.op_impl == "cpu":
            args.infer.use_cuda_graph = False
        elif (
            args.models is not None
            and str(args.models).find("'backend': 'cpuinfer'") != -1
        ):
            args.infer.use_cuda_graph = False
        else:
            args.infer.use_cuda_graph = True

    if args.infer.mtp_size <= 0:
        args.infer.mtp_size = 1

    if args.infer.full_warmup == "auto":
        if args.infer.pp_size > 1 and args.infer.use_cuda_graph:
            args.infer.full_warmup = True
        else:
            args.infer.full_warmup = False

    if args.scheduler.pp_config.pp_micro_batch_size_prefill == "auto":
        args.scheduler.pp_config.pp_micro_batch_size_prefill = "max"

    if args.scheduler.pp_config.pp_micro_batch_size_decode == "auto":
        args.scheduler.pp_config.pp_micro_batch_size_decode = "max"

    if args.infer.embed_tokens_lm_head_tp_size == "auto":
        args.infer.embed_tokens_lm_head_tp_size = args.infer.tp_size
    else:
        assert (
            args.infer.embed_tokens_lm_head_tp_size.isdigit()
        ), "embed_tokens_lm_head_tp_size must be auto or an integer"

    if args.infer.mla_absorb == "auto":
        if args.models.type == ModelType.DEEPSEEK_V3:
            args.infer.mla_absorb = "absorb-without-precomp"
        else:
            args.infer.mla_absorb = "none"

    if args.infer.dp_size > args.infer.max_batch_size:
        raise ValueError(
            f"infer.dp_size ({args.infer.dp_size}) cannot be greater than infer.max_batch_size ({args.infer.max_batch_size})"
        )

    # Check checkpoint exists
    check_checkpoint_path(args)

    # Parse model configuration, supporting dynamic reading from config.json files
    # Uses $(config.json:field_name) syntax, e.g., n_heads: "$(config.json:head_dim)"
    model_resolver = ModelConfigResolver()
    args.models = model_resolver.process_config_dict(args.models, args.models.ckpt_dir)

    if (
        args.models.type == ModelType.DEEPSEEK_V3
        and args.models.get("index_topk", None) is not None
    ):
        assert args.infer.indexer_type in ("auto", "deepgemm", "hygon", "triton")
        from chitu.dsa_indexer import (
            support_indexer_deepgemm,
            support_indexer_hygon,
            validate_indexer_config,
        )

        if args.infer.indexer_type == "auto":
            if (
                support_indexer_hygon
                and args.infer.cache_type == "paged"
                and args.infer.mtp_size < 3
                and int(args.models.index_head_dim) == 128
                and int(args.models.index_n_heads) in (32, 64)
            ):
                args.infer.indexer_type = "hygon"
            elif (
                support_indexer_deepgemm
                and args.infer.cache_type == "paged"
                and args.infer.mtp_size < 3
            ):
                args.infer.indexer_type = "deepgemm"
            else:
                args.infer.indexer_type = "triton"
        else:
            validate_indexer_config(args, args.infer.indexer_type)

    set_quant_variables(args)
    set_backend_variables(args)
    set_global_variables(args, debug=debug)
    logger.debug(f"Auto setting configs done. Full configs are: {args}")

    ###################################################################
    # Initialize backend

    try:
        args = get_global_args()
        Backend.build(args)
        rank = torch.distributed.get_rank()
        if rank == 0:
            Backend.schedulers = [
                Scheduler.build(args.scheduler, args.infer, dp_rank=i)
                for i in range(args.infer.dp_size)
            ]
        executor = Executor.build(args)
        Backend.executor = executor
        PackedTasks.configure(max_num_tasks=args.infer.max_batch_size)
        logger.info("Chitu has been initialized")

        collector = PrometheusMetricsCollector.get_instance(is_create=True)

        collector_addrs = [collector.addr]
        if type(get_world_group().gpu_group) != SingletonGroupPlaceholder:
            try:
                collector_addrs = gather_str_to_dst_rank(
                    collector.addr, dst=0, group=get_world_group().gpu_group
                )
            except Exception as e:
                logger.error(
                    f"An error occurred while gathering collector addresses to rank 0. Prometheus will monitor metrics only on rank 0: {e}"
                )

        logger.debug(f"collector_addrs:{collector_addrs}")

        # Only rank 0 monitors (it has all TaskPool data)
        should_start_monitor = rank == 0
        if should_start_monitor:
            start_prometheus_server_and_metrics_monitor(collector_addrs)
    except Exception as e:
        if not torch.distributed.is_initialized():
            raise e
        rank = torch.distributed.get_rank()
        # Prepend rank ID before the message
        msg = "\n".join(
            [f"[Rank {rank}] {line}" for line in traceback.format_exc().split("\n")]
        )
        raise Exception(
            msg
        ) from None  # `msg` already contains traceback, so raise from None


def _update_tasks_preferred_dp_rank():
    """Update the preferred DP rank for each schedulable but unscheduled prefill task.
    The preferred DP rank selection considers:
        - Prefix cache hit rate
        - Load of each DP rank
    """
    args = get_global_args()
    dp_size = int(args.infer.dp_size)
    enable_prefix_caching = bool(args.infer.enable_prefix_caching)

    if dp_size <= 1:
        return

    # weight of preifx cache hit rate
    hit_rate_weight = float(
        getattr(args.infer, "dp_prefix_caching_hit_rate_weight", 0.01)
    )

    # penalty weight of the number of running tasks in the DP rank
    running_tasks_penalty_weight = float(
        getattr(args.infer, "dp_prefix_caching_running_penalty_weight", 0.01)
    )

    hit_rate_weight = max(min(hit_rate_weight, 1), 0)
    running_tasks_penalty_weight = max(min(running_tasks_penalty_weight, 1), 0)

    if Backend.cache_managers is not None:
        # paged kv cache
        projected_running_tasks_per_dp = [
            len(
                Backend.schedulers[dp_rank].cache_manager_dict["main"].task_to_cache_ids
            )
            for dp_rank in range(dp_size)
        ]
    else:
        # dense kv cache
        projected_running_tasks_per_dp = [0 for dp_rank in range(dp_size)]
        for task in TaskPool.pool.values():
            if task.dp_rank is not None and 0 <= task.dp_rank <= dp_size:
                projected_running_tasks_per_dp[task.dp_rank] += 1

    for task in TaskPool.pool.values():
        if (
            task.dp_rank is not None
            or task.task_type != TaskType.Prefill
            or not task.can_schedule()
        ):
            continue

        best_preference_score = float("-inf")
        best_dp_rank = None

        for dp_rank in range(dp_size):
            if enable_prefix_caching:
                cache_managers = list(Backend.cache_managers[dp_rank].values())
                cached_tokens = task.prefix_tokens_len
                for cache_manager in cache_managers:
                    manager_cached_tokens = (
                        cache_manager.num_cached_blocks(task) * cache_manager.block_size
                    )
                    cached_tokens = min(cached_tokens, manager_cached_tokens)
                if task.prefix_tokens_len > 0:
                    cached_rate = cached_tokens / task.prefix_tokens_len
                else:
                    cached_rate = 0.0
            else:
                cached_rate = 0.0

            preference_score = (
                cached_rate * hit_rate_weight
                - running_tasks_penalty_weight * projected_running_tasks_per_dp[dp_rank]
            )

            if preference_score > best_preference_score:
                best_preference_score = preference_score
                best_dp_rank = dp_rank

        task.preferred_dp_rank = best_dp_rank
        if best_dp_rank is not None:
            projected_running_tasks_per_dp[best_dp_rank] += 1


@torch.inference_mode()
def chitu_run_main_rank():
    # 1. Schedule
    global _last_step_task_type
    for scheduler in Backend.schedulers:
        scheduler.prepare_for_schedule()
    if Backend.args.infer.dp_size == 1:
        assert len(Backend.schedulers) == 1
        task_ids = Backend.schedulers[0].schedule()
    else:
        _update_tasks_preferred_dp_rank()  # 按击中率和负载均衡路由

        # New prefill tasks are routed by DP preference assignment above.
        id_and_scheduler_list = list(enumerate(Backend.schedulers))
        task_ids_list = [[]] * len(id_and_scheduler_list)
        for task_type in (TaskType.Prefill, TaskType.Decode):
            for i, scheduler in id_and_scheduler_list:
                task_ids = scheduler.schedule(strict_allowed_task_type={task_type})
                task_ids_list[i] = task_ids
            if any((len(task_ids) > 0 for task_ids in task_ids_list)):
                DPTaskCollector.prepare_dp_tasks(task_ids_list)
                for task_ids in task_ids_list:
                    for task_id in task_ids:
                        task = TaskPool.pool.get(task_id)
                        if task is None:
                            continue
                        if getattr(task, "task_type", None) != TaskType.Decode:
                            continue
                        if getattr(task, "pd_sched_wait_end_logged", False):
                            continue
                        task.pd_sched_wait_end_logged = True
                        req_id = getattr(
                            getattr(task, "req", None), "request_id", task_id
                        )
                        logger.debug(
                            f"[PD_STAGE][decode.sched_wait.end] req_id={req_id}"
                        )
                break
        if all(len(task_ids) == 0 for task_ids in task_ids_list):
            DPTaskCollector.prepare_dp_tasks([])
        task_ids = task_ids_list[0]
    all_rank_task_ids = (
        task_ids
        if Backend.args.infer.dp_size == 1
        else [task_id for task_ids in task_ids_list for task_id in task_ids]
    )
    _last_step_task_type = (
        TaskType.Special
        if len(all_rank_task_ids) == 0
        else TaskPool.pool[all_rank_task_ids[0]].task_type
    )

    # 2. Run
    if task_ids or DPTaskCollector.has_available_tasks():
        logger.debug(f"Processing {task_ids}")
        if task_ids:
            for task_id in task_ids:
                task = TaskPool.pool.get(task_id)
                if task is None:
                    continue
                if getattr(task, "task_type", None) != TaskType.Decode:
                    continue
                if getattr(task, "pd_sched_wait_end_logged", False):
                    continue
                task.pd_sched_wait_end_logged = True
                req_id = getattr(getattr(task, "req", None), "request_id", task_id)
                logger.debug(f"[PD_STAGE][decode.sched_wait.end] req_id={req_id}")
        tasks = PackedTasks(task_ids)
    else:
        tasks = PackedTasksBase(
            num_tasks=len(task_ids),
            task_ids=task_ids,
            task_type=TaskType.Special,
            payload_type=SerializedPackedTasksPayloadType.Empty,
        )
    backend_payload_type = Backend.executor.step(tasks)
    _last_step_task_type = (
        tasks.task_type
        if tasks.payload_type != SerializedPackedTasksPayloadType.Empty
        and tasks.task_type in (TaskType.Prefill, TaskType.Decode)
        else None
    )

    # 3. Update TaskPool
    task_ids = TaskCollector.get_update_task_ids()
    task_ids = [task_id for task_id in task_ids if TaskPool.pool.get(task_id)]
    # tasks w/o dp_size are evicted and already removed
    task_ids = [
        task_id for task_id in task_ids if TaskPool.pool[task_id].dp_rank is not None
    ]
    DPTaskCollector.clear_last_packedtasks()

    # Collect tasks by DP rank. All tasks that have run should have dp_rank.
    task_ids_per_dp = [[] for _ in range(Backend.args.infer.dp_size)]
    for task_id in task_ids:
        dp_rank = TaskPool.pool[task_id].dp_rank
        task_ids_per_dp[dp_rank].append(task_id)

    # tasks from the model-running tasks (task_ids) which will not run anymore
    removed_task_ids = []
    for i in range(Backend.args.infer.dp_size):
        dp_local_removed_task_ids = Backend.schedulers[i].update(task_ids_per_dp[i])
        removed_task_ids += dp_local_removed_task_ids
    Backend.executor.special_step(removed_task_ids, type="EndTask")
    return backend_payload_type


@torch.inference_mode()
def chitu_run():
    rank = torch.distributed.get_rank()
    try:
        global _last_step_task_type
        check_alloc_retries()
        if rank != 0:
            payload_type = Backend.executor.step(None)
            _last_step_task_type = None
            return payload_type
        return chitu_run_main_rank()
    except Exception as e:
        # Prepend rank ID before the message
        msg = "\n".join(
            [f"[Rank {rank}] {line}" for line in traceback.format_exc().split("\n")]
        )
        raise Exception(
            msg
        ) from None  # `msg` already contains traceback, so raise from None


_last_alloc_retries = 0


def check_alloc_retries():
    global _last_alloc_retries
    cur_alloc_retries = torch.cuda.memory_stats(torch.cuda.current_device())[
        "num_alloc_retries"
    ]
    if cur_alloc_retries > _last_alloc_retries:
        logger.warning(
            f"{cur_alloc_retries - _last_alloc_retries} allocations succeeded only "
            f"after retrying (freeing memory from PyTorch allocator to CUDA and then "
            f"allocating them back). This will significantly reduce the performance. "
            f"Please try reducing memory usage, for example by lowering "
            f"`infer.memory_utilization`."
        )
    _last_alloc_retries = cur_alloc_retries


async def start_enhanced_scheduler_service(rank: int, dp_config, args):
    # only main rank of dp group start enhanced scheduler service
    dp_id = args.dp_config.dp_id
    if rank != 0:
        logger.warning(
            f"[Enhanced Scheduler {dp_id}] only main rank of dp group start Enhanced Scheduler service"
        )
        return

    logger.warning(f"[Enhanced Scheduler {dp_id}] Starting...")

    # Initialize ZMQ
    context = zmq.asyncio.Context()

    # Receive request socket
    request_socket = context.socket(zmq.PULL)
    request_port = dp_config.scheduler_base_port
    request_address = f"tcp://{dp_config.scheduler_base_host}:{request_port}"
    request_socket.bind(request_address)
    logger.warning(
        f"[Enhanced Scheduler {dp_id}] Listening to requests: {request_address}"
    )

    # Send statistics socket
    stats_socket = context.socket(zmq.PUSH)
    stats_address = f"tcp://{dp_config.router.host}:{dp_config.router.stats_port}"  # Router stats port
    stats_socket.connect(stats_address)
    logger.warning(
        f"[Enhanced Scheduler {dp_id}] connected to stats service: {stats_address}"
    )

    # Start DP Token Manager
    try:
        dp_id = get_global_args().dp_config.dp_id
        router_token_address = f"tcp://{dp_config.router.host}:{dp_config.router.token_port}"  # Token Router listen address

        logger.warning(
            f"[Enhanced Scheduler {dp_id}] Starting DP Token Manager, group ID={dp_id}"
        )
        await start_dp_token_manager(dp_id, router_token_address)
        logger.warning(
            f"[Enhanced Scheduler {dp_id}] DP Token Manager started successfully"
        )
    except Exception as e:
        logger.exception(
            f"[Enhanced Scheduler {dp_id}] DP Token Manager failed to start"
        )
        return

    # Performance statistics
    processed_requests = 0
    start_time = time.time()

    logger.warning(
        f"[Enhanced Scheduler {dp_id}] Starting to process requests, scheduler listening to requests on {request_address}"
    )
    last_num_blocks = 0
    last_block_size = 0
    try:
        while True:
            # Check if there are requests
            if await request_socket.poll(timeout=100):  # 100ms timeout
                try:
                    # Receive request
                    data = await request_socket.recv()
                    request_data = msgpack.unpackb(data, raw=False)

                    logger.info(
                        f"[Enhanced Scheduler {dp_id}] Received request: {request_data.get('request_id', 'unknown')}"
                    )

                    # Process request
                    await process_scheduler_request(rank, request_data)
                    processed_requests += 1

                except Exception as e:
                    logger.error(
                        f"[Enhanced Scheduler {dp_id}] Failed to process request: {e}"
                    )

            # Send statistics periodically
            current_time = time.time()
            # Send statistics every second
            if (current_time - start_time) >= 1.0:
                elapsed = current_time - start_time
                throughput = processed_requests / elapsed

                # Router load stats for this scheduler process.
                from chitu.metrics.task_stats import count_router_load

                running_requests, waiting_requests = count_router_load()
                evicted_blk_hashes = []
                num_blocks = 0
                block_size = None
                try:
                    num_managers = len(Backend.cache_managers)
                    for cache_manager_dict in Backend.cache_managers:
                        if block_size is None:
                            block_size = cache_manager_dict["main"].block_size
                        assert (
                            block_size == cache_manager_dict["main"].block_size
                        ), f"The block size of all main cache managers in the same instance should be the same. "

                        num_blocks += cache_manager_dict["main"].num_blocks

                        # Send only incremental evictions
                        if hasattr(
                            cache_manager_dict["main"], "pop_evicted_blk_hashes"
                        ):
                            evicted_blk_hashes.extend(
                                cache_manager_dict["main"].pop_evicted_blk_hashes(
                                    max_items=(512 // num_managers)
                                )
                            )

                except Exception as cache_e:
                    logger.error(
                        f"[Enhanced Scheduler {dp_id}] collect cache stats failed: {cache_e}"
                    )

                try:
                    stats = {
                        "scheduler_id": dp_config.dp_id,
                        "running_requests": int(running_requests),
                        "waiting_requests": int(waiting_requests),
                        "pending_tokens": 0,  # TODO: calculate pending tokens
                        "throughput_tokens_per_sec": throughput,
                        "last_update_time": current_time,
                        "heartbeat": True,
                    }

                    # num_blocks, block_size, evicted_blk_hashes这3个参数有变动时才传输
                    if num_blocks != last_num_blocks:
                        stats["num_blocks"] = num_blocks
                        last_num_blocks = num_blocks
                    if block_size != last_block_size:
                        stats["block_size"] = block_size
                        last_block_size = block_size
                    if evicted_blk_hashes:
                        stats["evicted_blk_hashes"] = evicted_blk_hashes

                    stats_data = msgpack.packb(stats)
                    await stats_socket.send(stats_data)
                    logger.debug(
                        f"[Enhanced Scheduler {dp_id}] throughput: {throughput:.2f}"
                    )
                except Exception as e:
                    logger.error(
                        f"[Enhanced Scheduler {dp_id}] throughput send failed: {e}"
                    )

                # Reset counter
                processed_requests = 0
                start_time = current_time

    except KeyboardInterrupt:
        logger.warning(f"[Enhanced Scheduler {dp_id}] Received interrupt signal")
    except Exception as e:
        logger.error(f"[Enhanced Scheduler {dp_id}] Service exception: {e}")
    finally:
        # Clean up resources
        request_socket.close()
        stats_socket.close()
        context.term()
        logger.warning(f"[Enhanced Scheduler {dp_id}] Service stopped")


async def process_scheduler_request(rank: int, request_data: dict):
    """Handle scheduling requests from Router"""
    try:
        # Create UserRequest
        user_request = UserRequest.from_dict(request_data)
        request_id = user_request.request_id

        # Create Task, honoring stop/ignore_eos semantics from request_data
        stop_with_eos = True
        if request_data.get("ignore_eos"):
            stop_with_eos = False
        elif not request_data.get("stop_with_eos"):
            stop_with_eos = False

        task = Task(task_id=request_id, req=user_request, stop_with_eos=stop_with_eos)

        try:
            dp_id = get_global_args().dp_config.dp_id
            token_manager = get_dp_token_manager(dp_id)
            # ensure token manager started
            await token_manager.start()
            if token_manager is not None:
                # Wrap Task to enable token sending
                wrapped_task = token_manager.wrap_task(task)
                TaskPool.add(wrapped_task)
            else:
                # If Token Manager not initialized, add the original Task directly
                TaskPool.add(task)

        except Exception as e:
            # If DP Token Manager acquisition fails, fall back to original Task
            logger.error(
                f"[Enhanced Scheduler {dp_id}] Failed to get DP Token Manager: {e}"
            )
            TaskPool.add(task)
            logger.warning(
                f"[Enhanced Scheduler {dp_id}] Fallback to original task: {request_id}"
            )

        logger.debug(f"[Enhanced Scheduler {dp_id}] Request handled: {request_id}")

    except Exception as e:
        logger.error(f"[Enhanced Scheduler {dp_id}] Failed to process request: {e}")
        logger.error(
            f"[Enhanced Scheduler {dp_id}] Error details: {traceback.format_exc()}"
        )


def chitu_start():
    Backend.state = BackendState.Running


def chitu_terminate():
    if torch.distributed.get_rank() == 0:
        Backend.state = BackendState.Terminated
        terminated_task = PackedTasksBase(
            num_tasks=0,
            payload_type=SerializedPackedTasksPayloadType.TerminateBackend,
        )
        Backend.executor.step(terminated_task)
    stop_metrics_monitor()
    PrometheusMetricsCollector.stop_instance()


def chitu_is_terminated():
    return Backend.state == BackendState.Terminated
