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
from chitu.chitu_main import chitu_init, warmup_engine
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

    logger.info(f"[WARMUP] Starting warmup...")
    warmup_engine(args)
    # warmup - DP compatible: each DP group's local master rank needs to do warmup
    logger.info(f"[WARMUP] Warmup done, task pool size: {len(TaskPool.pool)}")

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
