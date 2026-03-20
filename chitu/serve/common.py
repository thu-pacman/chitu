# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Common service functions for Chitu serve module.
"""

import asyncio
from typing import Any
from logging import getLogger

import torch
import torch.distributed
from fastapi import HTTPException

from chitu.task import (
    Task,
    TaskPool,
    UserRequest,
    SerializedPackedTasksPayloadType,
    TaskCollector,
)
from chitu.async_response import AsyncResponse
from chitu.global_vars import get_global_args

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


def get_priority_from_api_key(api_key: str) -> int:
    args = get_global_args()
    for item in args.serve.api_keys:
        if item.key == api_key:
            return item.priority
    if args.serve.validate_api_key == True:
        raise HTTPException(status_code=503, detail="Unauthorized api key")
    return 1


def build_chat_template_kwargs(enable_thinking: bool) -> dict[str, Any]:
    chat_template_kwargs = {}
    if "DeepSeek-V3.1" in get_global_args().models.name:
        # DeepSeek-V3.1 tokenizer uses `thinking` instead of `enable_thinking`
        chat_template_kwargs["thinking"] = enable_thinking
    else:
        chat_template_kwargs["enable_thinking"] = enable_thinking
    return chat_template_kwargs


def submit_request(req: UserRequest) -> AsyncResponse:
    task = Task(
        req.request_id, req, stop_with_eos=req.stop_with_eos, priority=req.priority
    )
    TaskPool.enqueue(task)
    return AsyncResponse(req)
