# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import time
import traceback
from logging import getLogger
from typing import Optional
import re
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
    is_classic_pd_disagg,
    is_independent_multi_inst,
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
    try_import_and_setup_torch_npu,
    ceil_div,
    gather_str_to_dst_rank,
    get_chitu_bool_env,
)
from chitu.cp_utils import get_cp_context
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
from chitu.distributed.coordinator import get_endpoint, set_endpoint
from chitu.boot.tcp_ip import get_local_ip
from chitu.dp_token_sender import get_dp_token_manager, start_dp_token_manager
from chitu.dp_request_router import is_terminate_engine_message
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

try_import_and_setup_torch_npu()


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


def _direct_warmup_target_lens(
    local_max_bs: int,
    decode_steps: int,
    bs_descend: int,
    skip_model_decode: bool,
) -> list[int]:
    local_max_bs = int(local_max_bs)
    if local_max_bs <= 0:
        return []

    # _warmup_backend_direct always runs prepare_cache_prefill with one token per
    # request, even when model.prefill itself is skipped for decode-only workers.
    target_lens = [1 for _ in range(local_max_bs)]
    if skip_model_decode:
        return target_lens

    mtp_size = int(get_global_args().infer.mtp_size)
    for i in range(max(1, int(decode_steps))):
        curr_bs = local_max_bs - i * int(bs_descend)
        if curr_bs <= 0:
            break
        curr_bs = min(curr_bs, local_max_bs)
        for req_idx in range(curr_bs):
            target_lens[req_idx] += mtp_size
    return target_lens


def _cache_consumes_direct_warmup_new_cache_ids(cache) -> bool:
    if not all(
        hasattr(cache, attr)
        for attr in ("block_table", "block_size", "num_blocks", "manager_name")
    ):
        return False

    # These cache types manage their own block allocation and do not consume
    # PackedTasks.new_cache_ids_list in prepare_cache_prefill/decode.
    if type(cache).__name__ in {"SingletonPagedKVCache", "MMPagedKVCache"}:
        return False

    return True


def _direct_warmup_num_blocks_for_cache(
    cache,
    manager_name: str,
    target_len: int,
) -> int:
    target_len = int(target_len)
    if target_len <= 0:
        return 0

    # DeepSeek-V4 sliding-window cache is scheduler-managed even though each
    # request owns exactly one ring-buffer page.
    if bool(getattr(cache, "fixed_num_blocks", False)):
        return 1

    ratio = _deepseek_v4_compressed_ratio_from_manager_name(manager_name)
    storage_len = target_len // int(ratio) if ratio is not None else target_len
    if storage_len <= 0:
        return 0
    return ceil_div(storage_len, int(cache.block_size))


def _build_direct_warmup_new_cache_ids_list(
    local_max_bs: int,
    decode_steps: int,
    bs_descend: int,
    skip_model_decode: bool,
) -> list[dict[str, list[int]]]:
    """Build scheduler-style cache block metadata for direct warmup.

    Direct warmup bypasses the scheduler, but PagedKVCache.prepare_cache_* still
    expects the scheduler-produced request -> block ids mapping.  Allocate that
    metadata deterministically here, grouped by manager_name, so all caches owned
    by the same logical manager share the same block ids.
    """

    local_max_bs = int(local_max_bs)
    target_lens = _direct_warmup_target_lens(
        local_max_bs,
        decode_steps,
        bs_descend,
        skip_model_decode,
    )
    new_cache_ids_list: list[dict[str, list[int]]] = [{} for _ in range(local_max_bs)]

    manager_groups = {}
    for cache in Backend.cache_dict.values():
        if not _cache_consumes_direct_warmup_new_cache_ids(cache):
            continue
        manager_name = str(cache.manager_name)
        manager_groups.setdefault(manager_name, []).append(cache)

    for manager_name, caches in manager_groups.items():
        capacity = min(int(cache.num_blocks) for cache in caches)
        blocks_per_req = [0 for _ in range(local_max_bs)]
        for cache in caches:
            for req_idx, target_len in enumerate(target_lens):
                blocks_per_req[req_idx] = max(
                    blocks_per_req[req_idx],
                    _direct_warmup_num_blocks_for_cache(
                        cache, manager_name, target_len
                    ),
                )

        total_blocks = sum(blocks_per_req)
        if total_blocks > capacity:
            raise RuntimeError(
                "direct warmup needs "
                f"{total_blocks} blocks for cache manager {manager_name}, "
                f"but only {capacity} are allocated; "
                f"blocks_per_req={blocks_per_req}, target_lens={target_lens}"
            )

        next_block = 0
        for req_idx, num_blocks in enumerate(blocks_per_req):
            if num_blocks > 0:
                new_cache_ids_list[req_idx][manager_name] = list(
                    range(next_block, next_block + num_blocks)
                )
            next_block += num_blocks

    return new_cache_ids_list


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

    if is_classic_pd_disagg():
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
    # In PCP mode, the effective per-step budget is `prefill_chunk_size * pcp_size` because each CP
    # rank only processes 1/pcp_size of the global tokens.
    # Due to per-rank budget constraints and uneven task distribution, some tokens may be left unprocessed.
    # Example: DP2, chunk=16, max_batch_size=5 (创建 5 个 warmup 任务), 每任务 3 tokens
    #   - Budget: Rank0=8, Rank1=8 (chunk_size 均分给各 rank)
    #   - Tasks: Rank0 分到 3 个任务 (round robin), Rank1 分到 2 个任务
    #   - Actual: Rank0 处理 8 tokens (3+3+2, task4 剩 1 token), Rank1 处理 6 tokens (3+3)
    #   - Result: 需要 2 轮迭代来处理完所有 15 tokens
    _cp_context = get_cp_context()
    effective_prefill_chunk_size = prefill_chunk_size * _cp_context.pcp_size
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
        per_rank_budget = effective_prefill_chunk_size // dp_size
        max_tasks_per_rank = ceil_div(num_warmup_reqs, dp_size)
        max_tokens_per_rank = max_tasks_per_rank * warmup_seq_len
        # Iterations needed for the busiest rank (usually rank 0)
        num_required_prefill_schedules = ceil_div(max_tokens_per_rank, per_rank_budget)
    else:
        num_required_prefill_schedules = ceil_div(
            total_tokens, effective_prefill_chunk_size
        )
    num_required_decode_schedules = _n_decode_steps * _mtp_size

    logger.info(
        f"Warmup: total_tokens={total_tokens}, chunk_size={prefill_chunk_size}, "
        f"effective_chunk_size={effective_prefill_chunk_size}, "
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
    _cp_context = get_cp_context()
    if _cp_context.is_active:
        local_max_bs = max(local_max_bs, _cp_context.pcp_size)
        # Round up to multiple of pcp_size for clean interleaved split
        local_max_bs = (
            (local_max_bs + _cp_context.pcp_size - 1) // _cp_context.pcp_size
        ) * _cp_context.pcp_size
    init_cache_static()

    req_ids = [f"__warmup_{i}__" for i in range(local_max_bs)]
    tokens = torch.randint(
        1,
        args.models.vocab_size,
        size=(local_max_bs,),
        device="cuda",
        dtype=torch.int64,
    )
    hiddens = None
    if not get_pp_group().is_first_rank:
        hiddens = torch.randn(
            Backend.executor.get_payload_shape(local_max_bs),
            device="cuda",
            dtype=Backend.executor.get_payload_dtype(),
        )
    all_tasks = PackedTasksBase(local_max_bs, task_ids=req_ids)
    all_tasks.new_cache_ids_list = _build_direct_warmup_new_cache_ids_list(
        local_max_bs,
        decode_steps,
        bs_descend,
        bool(skip_model_decode),
    )
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
    if args.infer.mtp_size > 1:
        mtp_accept_indices = torch.zeros(
            local_max_bs, dtype=torch.int64, device=tokens.device
        )
        Backend.model.mtp_accept_indices.set(mtp_accept_indices)

    if not skip_model_prefill:
        Backend.model.prefill(tokens, hiddens, output_token_offsets)

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

            if args.infer.mtp_size > 1:
                mtp_accept_indices = torch.zeros(
                    curr_bs, dtype=torch.int64, device=tokens.device
                )
                Backend.model.mtp_accept_indices.set(mtp_accept_indices)

            if get_pp_group().is_first_rank:
                payload = tokens[:curr_bs]
            else:
                payload = hiddens[:curr_bs]

            _ = Backend.model.decode(payload)

    # Clean KV for this request
    for cache in Backend.cache_dict.values():
        cache.finalize_cache_all_decode(all_tasks)
    PrometheusMetricsCollector.update_GPU_usage()
    logger.info("Local backend warmup (direct) completed")


def warmup_engine(args):
    # Router 进程不做 warmup
    if args.multi_inst.router.is_router:
        return

    clear_observed_op_impl_selections()

    # 告知 DeepGEMM 预热范围，使其在首次遇到某个 (n,k) 时提前编译所有可能的 m 值对应的 kernel，
    # 避免推理过程中因 m 变化触发 JIT 编译导致延迟波动。
    # DG_WARMUP_MAX_M 若已由外部设置则不覆盖。
    if "DG_WARMUP_MAX_M" not in os.environ and getattr(
        args.infer, "full_warmup", False
    ):
        _pcs = getattr(args.infer, "prefill_chunk_size", None)
        if not (_pcs and isinstance(_pcs, int) and _pcs > 0):
            _pcs = getattr(args.infer, "max_seq_len", 0) * getattr(
                args.infer, "max_batch_size", 0
            )
        if _pcs and isinstance(_pcs, int) and _pcs > 0:
            _dp_size = max(getattr(args.infer, "dp_size", 1), 1)
            _warmup_max_m = _pcs // _dp_size
            os.environ["DG_WARMUP_MAX_M"] = str(_warmup_max_m)
            logger.info(
                f"[warmup] Set DG_WARMUP_MAX_M={_warmup_max_m} "
                f"(prefill_chunk_size={_pcs} / dp_size={_dp_size})"
            )

    if is_classic_pd_disagg():
        runner = "direct"
    elif is_independent_multi_inst():
        runner = "taskpool"
    else:
        raise NotImplementedError(
            "Mixing prefill_and_decode with prefill/decode roles is not supported"
        )

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


def chitu_init(args):
    """
    Initialize the computation thread of Chitu.

    Args:
        args: Hydra config.

    Returns:
        Preprocessed config.
    """

    debug = get_chitu_bool_env("CHITU_DEBUG", False)

    if (
        is_nvidia()
        and torch.distributed.is_nccl_available()
        and torch.cuda.nccl.version() <= (2, 21, 5)
    ):
        os.environ["NCCL_NVLS_NCHANNELS"] = "32"

    init_logger()

    set_quant_variables(args)
    set_backend_variables(args)
    set_global_variables(args, debug=debug)
    args = get_global_args()  # Get the pre-processed global args

    ###################################################################
    # Initialize backend

    try:
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

        collector_addrs = PrometheusMetricsCollector.addrs
        if type(get_world_group().gpu_group) != SingletonGroupPlaceholder:
            try:
                collector_addrs = gather_str_to_dst_rank(
                    collector_addrs[0], dst=0, group=get_world_group().gpu_group
                )
            except Exception as e:
                logger.error(
                    f"An error occurred while gathering collector addresses to rank 0. Prometheus will monitor metrics only on rank 0: {e}"
                )
        PrometheusMetricsCollector.addrs = collector_addrs

        logger.info(f"Prometheus collector addresses:{collector_addrs}")

        # Only rank 0 monitors (it has all TaskPool data)
        should_start_monitor = rank == 0 and args.multi_inst.n_insts == 1
        if should_start_monitor:
            start_prometheus_server_and_metrics_monitor(collector_addrs)
    except Exception as e:
        if not torch.distributed.is_initialized():
            raise e
        rank = torch.distributed.get_rank()
        # Prepend rank ID before the message
        header = ""
        if args.multi_inst.router.is_router:
            header += "[Router]"
        else:
            header += f"[Inst {args.multi_inst.inst_id}]"
        header += f" [Rank {rank}]"
        msg = "\n".join(
            [f"{header} {line}" for line in traceback.format_exc().split("\n")]
        )
        raise Exception(
            msg
        ) from None  # `msg` already contains traceback, so raise from None

    return args


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


async def start_enhanced_scheduler_service(rank: int, multi_inst, args):
    # only main rank of dp group start enhanced scheduler service
    instance_id = args.multi_inst.inst_id
    if rank != 0:
        logger.warning(
            f"[Enhanced Scheduler {instance_id}] only main rank of dp group start Enhanced Scheduler service"
        )
        return

    logger.warning(f"[Enhanced Scheduler {instance_id}] Starting...")

    # Initialize ZMQ
    context = zmq.asyncio.Context()

    # Receive request socket
    request_socket = context.socket(zmq.PULL)
    request_socket.setsockopt(zmq.LINGER, 0)
    # Bind the TCP server to a random port on the non-wildcard ip, then
    # register it in the coordinator under role `instance_<id>`.
    request_ip = get_local_ip()
    request_port = request_socket.bind_to_random_port(f"tcp://{request_ip}")
    set_endpoint(f"instance_{instance_id}", "request_port", request_ip, request_port)
    request_address = f"tcp://{request_ip}:{request_port}"
    logger.warning(
        f"[Enhanced Scheduler {instance_id}] Listening to requests: {request_address}"
    )

    # Send statistics socket
    stats_socket = context.socket(zmq.PUSH)
    stats_socket.setsockopt(zmq.LINGER, 0)
    # Get the router stats endpoint from the coordinator, then connect to it.
    stats_ip, stats_port = get_endpoint("router", "stats_port")
    stats_address = f"tcp://{stats_ip}:{stats_port}"  # Router stats endpoint
    stats_socket.connect(stats_address)
    logger.warning(
        f"[Enhanced Scheduler {instance_id}] connected to stats service: {stats_address}"
    )

    # Start DP Token Manager
    try:
        logger.warning(
            f"[Enhanced Scheduler {instance_id}] Starting DP Token Manager, group ID={instance_id}"
        )
        await start_dp_token_manager(instance_id)
        logger.warning(
            f"[Enhanced Scheduler {instance_id}] DP Token Manager started successfully"
        )
    except Exception as e:
        logger.exception(
            f"[Enhanced Scheduler {instance_id}] DP Token Manager failed to start"
        )
        return

    # Performance statistics
    processed_requests = 0
    start_time = time.time()

    logger.warning(
        f"[Enhanced Scheduler {instance_id}] Starting to process requests, scheduler listening to requests on {request_address}"
    )
    last_num_blocks = 0
    last_block_size = 0
    try:
        while True:
            if Backend.state == BackendState.Terminated:
                break

            # Check if there are requests
            if await request_socket.poll(timeout=100):  # 100ms timeout
                try:
                    # Receive request
                    data = await request_socket.recv()
                    request_data = msgpack.unpackb(data, raw=False)

                    logger.info(
                        f"[Enhanced Scheduler {instance_id}] Received request: {request_data.get('request_id', 'unknown')}"
                    )

                    # Process request
                    await process_scheduler_request(rank, request_data)
                    processed_requests += 1

                except Exception as e:
                    logger.error(
                        f"[Enhanced Scheduler {instance_id}] Failed to process request: {e}"
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
                        f"[Enhanced Scheduler {instance_id}] collect cache stats failed: {cache_e}"
                    )

                try:
                    stats = {
                        "local_instance_id": instance_id,
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
                        f"[Enhanced Scheduler {instance_id}] throughput: {throughput:.2f}"
                    )
                except Exception as e:
                    logger.error(
                        f"[Enhanced Scheduler {instance_id}] throughput send failed: {e}"
                    )

                # Reset counter
                processed_requests = 0
                start_time = current_time

    except KeyboardInterrupt:
        logger.warning(f"[Enhanced Scheduler {instance_id}] Received interrupt signal")
    except Exception as e:
        logger.error(f"[Enhanced Scheduler {instance_id}] Service exception: {e}")
    finally:
        # Notify the router that this instance has fully drained before tearing
        # down sockets. This lets /terminate_engine keep token/request routers
        # alive until late tokens and finish messages have been received.
        try:
            terminated_stats = {
                "local_instance_id": instance_id,
                "running_requests": 0,
                "waiting_requests": 0,
                "pending_tokens": 0,
                "throughput_tokens_per_sec": 0.0,
                "last_update_time": time.time(),
                "heartbeat": False,
                "terminated": True,
            }
            await stats_socket.send(msgpack.packb(terminated_stats))
        except Exception:
            logger.exception(
                f"[Enhanced Scheduler {instance_id}] failed to send termination ack"
            )

        # Clean up resources
        request_socket.close(0)
        stats_socket.close(0)
        context.destroy(linger=0)
        logger.warning(f"[Enhanced Scheduler {instance_id}] Service stopped")


async def process_scheduler_request(rank: int, request_data: dict):
    """Handle scheduling requests from Router"""
    try:
        if is_terminate_engine_message(request_data):
            Backend.state = BackendState.Terminating
            logger.info("Terminate_engine received. Draining in-flight requests")
            return

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
            instance_id = get_global_args().multi_inst.inst_id
            token_manager = get_dp_token_manager(instance_id)
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
                f"[Enhanced Scheduler {instance_id}] Failed to get DP Token Manager: {e}"
            )
            TaskPool.add(task)
            logger.warning(
                f"[Enhanced Scheduler {instance_id}] Fallback to original task: {request_id}"
            )

        logger.debug(
            f"[Enhanced Scheduler {instance_id}] Request handled: {request_id}"
        )

    except Exception as e:
        instance_id = get_global_args().multi_inst.inst_id
        logger.error(
            f"[Enhanced Scheduler {instance_id}] Failed to process request: {e}"
        )
        logger.error(
            f"[Enhanced Scheduler {instance_id}] Error details: {traceback.format_exc()}"
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

        try:
            from chitu.dp_token_sender import close_dp_token_managers

            close_dp_token_managers()
        except Exception:
            logger.exception("Failed to close DP token managers")
    stop_metrics_monitor()
    PrometheusMetricsCollector.stop_instance()


def chitu_is_terminated():
    return Backend.state == BackendState.Terminated
