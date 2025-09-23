# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Common service functions for Chitu serve module.
"""

import asyncio
import logging
from logging import getLogger

import torch
import torch.distributed

from chitu.backend import Backend
from chitu.chitu_main import chitu_run
from chitu.task import TaskPool, PackedTasksBase, SerializedPackedTasksPayloadType

logger = getLogger(__name__)

# Global variables for serve module
min_batch_size = 1


def set_min_batch_size(value: int):
    global min_batch_size
    min_batch_size = value


async def process_queue():
    """Process the task queue - common function used by both normal and DP modes"""
    # DP compatible: each DP group's local master rank needs to start heartbeat
    rank = torch.distributed.get_rank()
    if rank == 0:
        asyncio.create_task(heartbeat_timer(60))
    global min_batch_size
    while True:
        if (len(TaskPool.pool) >= min_batch_size) or rank != 0:
            min_batch_size = 1
            chitu_run()
        else:
            await asyncio.sleep(0.01)


async def propagate_heartbeat():
    """Add heartbeat tasks"""
    heartbeat_task = PackedTasksBase(
        num_tasks=0,
        payload_type=SerializedPackedTasksPayloadType.Heartbeat,
    )
    Backend.executor.step(heartbeat_task)


async def heartbeat_timer(interval=60):
    """Timer for sending heartbeat tasks"""
    while True:
        await asyncio.sleep(interval)
        await propagate_heartbeat()


def start_worker():
    """Start worker for processing queue in a new event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(process_queue())
