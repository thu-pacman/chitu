# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
DP Scheduler service module for Chitu serve.
Contains the Enhanced Scheduler service logic for DP mode.
"""

import asyncio
import logging
from logging import getLogger
import threading

import torch
import torch.distributed

from chitu.chitu_main import chitu_init, warmup_engine, start_enhanced_scheduler_service
from chitu.serve.common import start_worker
from chitu.task import TaskPool

logger = getLogger(__name__)


def init_dp_scheduler(args, rank):
    """Initialize DP Enhanced Scheduler"""
    logger.info(f"[SCHEDULER] Starting DP Enhanced Scheduler...")

    # Initialize torch.distributed (if not already initialized)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")

    world_size = torch.distributed.get_world_size()

    logger.info(f"[SCHEDULER] Starting DP Enhanced Scheduler, world_size={world_size}")

    chitu_init(args, logging_level=logging.INFO)
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    # Router process will skip warmup in unified
    try:
        warmup_engine(args)
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.warning(f"[SCHEDULER] unified warmup skipped/failed: {e}")

    # Check if PD disaggregation is enabled
    pd_enabled = (
        hasattr(args.dp_config.router, "pd_disaggregation")
        and args.dp_config.router.pd_disaggregation.enabled
    )

    if pd_enabled:
        logger.info("[SCHEDULER] PD disaggregation enabled, using PD Scheduler")
        # Use PD disaggregation scheduler
        from chitu.distributed.pd_disaggregation.pd_service import init_pd_scheduler

        init_pd_scheduler(args, rank)
        return

    # Traditional DP scheduler
    logger.info("[SCHEDULER] Using traditional DP Enhanced Scheduler")

    logger.info(
        f"[WARMUP] Unified warmup done earlier; task pool size: {len(TaskPool.pool)}"
    )

    # Determine actual distributed rank
    actual_rank = (
        torch.distributed.get_rank() if torch.distributed.is_initialized() else rank
    )

    # For non-zero ranks (TP peers), block on compute loop in main thread so that
    # chitu_run() triggers Backend.executor.step(None) and keeps TP comm alive.
    if actual_rank != 0:
        logger.info(
            f"[SCHEDULER] rank={actual_rank} running process_queue on main thread (no ZMQ service)"
        )
        start_worker()
        return

    logger.info(
        f"[SCHEDULER] rank=0 starting process_queue in background thread and Enhanced Scheduler service(for zmq service) on main loop..."
    )

    # Start the compute loop in a dedicated thread/event loop to avoid blocking asyncio
    t = threading.Thread(target=start_worker, daemon=True)
    t.start()

    # Run Enhanced Scheduler ZMQ service on the main asyncio loop
    asyncio.run(start_enhanced_scheduler_service(actual_rank, args.dp_config, args))
