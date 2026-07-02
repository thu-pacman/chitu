# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
DP Scheduler service module for Chitu serve.
Contains the Enhanced Scheduler service logic for DP mode.
"""

import asyncio
import os
from datetime import timedelta
from logging import getLogger
import threading

import torch
import torch.distributed

from chitu.chitu_main import chitu_init, warmup_engine, start_enhanced_scheduler_service
from chitu.serve.common import start_worker
from chitu.task import TaskPool
from chitu.distributed.infiniband import auto_set_ib_envs
from chitu.global_vars import (
    is_classic_pd_disagg,
    is_independent_multi_inst,
    set_cuda_device,
)

logger = getLogger(__name__)

from chitu.distributed.pd_disaggregation.pd_service import (
    init_pd_scheduler,
    init_pd_worker,
)


def init_dp_scheduler(args):
    """Initialize DP Enhanced Scheduler"""
    logger.info(f"[SCHEDULER] Starting DP Enhanced Scheduler...")

    # Initialize torch.distributed (if not already initialized)
    if not torch.distributed.is_initialized():
        auto_set_ib_envs()
        torch.distributed.init_process_group("nccl")

    world_size = torch.distributed.get_world_size()

    logger.info(f"[SCHEDULER] Starting DP Enhanced Scheduler, world_size={world_size}")

    args = chitu_init(args)
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    # Router process will skip warmup in unified
    warmup_engine(args)

    rank = torch.distributed.get_rank()

    if is_classic_pd_disagg():
        logger.info("[SCHEDULER] Using classic PD Scheduler")
        # Use PD disaggregation scheduler
        if rank == 0:
            if init_pd_scheduler is None:
                raise RuntimeError("PD scheduler service not available")
            init_pd_scheduler(args, rank)
        else:
            if init_pd_worker is None:
                raise RuntimeError("PD worker service not available")
            init_pd_worker(args, rank)
        return

    if not is_independent_multi_inst():
        raise NotImplementedError(
            "Mixing prefill_and_decode with prefill/decode roles is not supported"
        )

    # Traditional DP scheduler
    logger.info("[SCHEDULER] Using traditional DP Enhanced Scheduler")

    logger.info(
        f"[WARMUP] Unified warmup done earlier; task pool size: {len(TaskPool.pool)}"
    )

    # For non-zero ranks (TP peers), block on compute loop in main thread so that
    # chitu_run() triggers Backend.executor.step(None) and keeps TP comm alive.
    if rank != 0:
        logger.info(
            f"[SCHEDULER] rank={rank} running process_queue on main thread (no ZMQ service)"
        )
        start_worker()
        return

    logger.info(
        f"[SCHEDULER] rank=0 starting process_queue in background thread and Enhanced Scheduler service(for zmq service) on main loop..."
    )

    def set_device_id_again_and_start_worker():
        set_cuda_device()
        start_worker()

    t = threading.Thread(target=set_device_id_again_and_start_worker)
    t.start()

    # Run Enhanced Scheduler ZMQ service on the main asyncio loop.
    # The compute loop owns CUDA/NCCL teardown; do not leave it as a daemon
    # thread during interpreter shutdown, otherwise native destructors can abort
    # with "terminate called without an active exception".
    try:
        asyncio.run(start_enhanced_scheduler_service(rank, args.multi_inst, args))
    finally:
        logger.info(
            f"[SCHEDULER] rank={rank} waiting for compute worker thread to stop"
        )
        t.join()
        logger.info(f"[SCHEDULER] rank={rank} compute worker thread stopped")
