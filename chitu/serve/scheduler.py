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

import torch
import torch.distributed

from chitu.backend import Backend
from chitu.chitu_main import chitu_init, warmup_engine_unified
from chitu.global_vars import get_global_args
from chitu.serve.common import process_queue
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
    torch.distributed.barrier()

    # 统一 warmup（Router 进程在 unified 内会自动跳过）
    try:
        warmup_engine_unified(args)
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

    logger.info(
        f"[SCHEDULER] Starting parallel tasks: process_queue + Enhanced Scheduler service..."
    )

    # Run both tasks in the same event loop
    async def run_dp_scheduler_services():
        # Import here to avoid circular dependency
        from chitu.chitu_main import start_enhanced_scheduler_service

        dp_config = args.dp_config

        # Run process_queue and Enhanced Scheduler service in parallel
        await asyncio.gather(
            process_queue(),  # Inference loop queue, processes TaskPool
            start_enhanced_scheduler_service(rank, dp_config, args),  # ZMQ service
        )

    asyncio.run(run_dp_scheduler_services())
