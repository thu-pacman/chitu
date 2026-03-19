# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Common service functions for Chitu serve module.
"""

import asyncio
from logging import getLogger

import torch
import torch.distributed

from chitu.task import (
    TaskPool,
    SerializedPackedTasksPayloadType,
    TaskCollector,
)

logger = getLogger(__name__)

# Global variables for serve module
min_batch_size = 1


def set_min_batch_size(value: int):
    global min_batch_size
    min_batch_size = value


async def process_queue():
    """Process the task queue - common function used by both normal and DP modes"""
    from chitu.chitu_main import chitu_run

    rank = torch.distributed.get_rank()
    global min_batch_size
    while True:
        TaskPool.add_all_queued()
        if (len(TaskPool.pool) >= min_batch_size) or rank != 0:
            min_batch_size = 1
            status = chitu_run()
            if status == SerializedPackedTasksPayloadType.NoneType:
                await asyncio.sleep(0.01)
        elif TaskCollector.has_batch_results():
            TaskCollector.process_last_batch_results()
        else:
            await asyncio.sleep(0.01)


def start_worker():
    """Start worker for processing queue in a new event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(process_queue())
