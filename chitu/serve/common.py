# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Common service functions for Chitu serve module.
"""

import asyncio
import os
import queue
from typing import Any, List, Optional
from logging import getLogger

import torch
import torch.distributed
from fastapi import HTTPException

from chitu.global_vars import get_global_args
from chitu.profiler import MemoryRecorder, ProfileManager
from chitu.task import (
    Task,
    TaskPool,
    UserRequest,
    SerializedPackedTasksPayloadType,
    TaskCollector,
)
from chitu.task_type import TaskType
from chitu.dp_router import get_request_router, get_token_router

logger = getLogger(__name__)

# Global variables for serve module
min_batch_size = 1
_profile_manager: Optional[ProfileManager] = None
_profile_cmd_queue: queue.Queue = queue.Queue()
_pending_profile_payload: Optional[dict] = None
_pending_profile_applied = False


def _put_to_profile_queue(payload: dict) -> None:
    _profile_cmd_queue.put(payload)


def set_min_batch_size(value: int):
    global min_batch_size
    min_batch_size = value


def get_profile_output_root() -> str:
    output_root = os.getenv("CHITU_TORCH_PROFILER_OUTPUT_ROOT", "trace/chitu")
    return os.path.abspath(os.path.normpath(output_root))


def resolve_profile_output_dir(output_dir: str) -> str:
    shared_root = get_profile_output_root()
    requested = os.path.normpath(output_dir)

    if not os.path.isabs(requested):
        runtime_relative_shared_root = os.path.normpath(
            os.path.relpath(shared_root, start=os.getcwd())
        )
        if requested in ("", "."):
            resolved = shared_root
        elif requested == runtime_relative_shared_root or requested.startswith(
            runtime_relative_shared_root + os.sep
        ):
            resolved = os.path.join(os.getcwd(), requested)
        else:
            resolved = os.path.join(shared_root, requested)
    else:
        shared_root_with_sep = shared_root + os.sep
        if requested == shared_root or requested.startswith(shared_root_with_sep):
            resolved = requested
        else:
            resolved = os.path.join(shared_root, os.path.basename(requested))

    return os.path.abspath(os.path.normpath(resolved))


def _resolve_activities(
    activities: Optional[List[str]],
    profile_memory: bool,
) -> List[str]:
    if activities is not None:
        result = list(activities)
    else:
        result = ["CPU", "GPU"]
    if profile_memory and "MEM" not in result:
        result.append("MEM")
    return result


def queue_profile_start(
    output_dir: str = "trace/chitu",
    start_step: int = 0,
    num_steps: int = 10,
    with_stack: bool = False,
    profile_by_stage: bool = False,
    activities: Optional[List[str]] = None,
    profile_memory: bool = False,
    memory_max_entries: int = 100000,
) -> str:
    output_dir = resolve_profile_output_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    resolved_activities = _resolve_activities(activities, profile_memory)

    payload: dict[str, Any] = {
        "action": "start",
        "output_dir": output_dir,
        "start_step": max(start_step, 0),
        "num_steps": max(num_steps, 1),
        "with_stack": bool(with_stack),
        "profile_by_stage": bool(profile_by_stage),
        "activities": resolved_activities,
        "memory_max_entries": max(memory_max_entries, 1),
    }
    _put_to_profile_queue(payload)
    return output_dir


def queue_profile_stop() -> None:
    _put_to_profile_queue({"action": "stop"})


def queue_mem_dump() -> None:
    _put_to_profile_queue({"action": "dump_memory"})


def enqueue_profile_payload(payload: dict) -> None:
    """Enqueue a pre-built profile command payload into the local queue.

    Used by PD scheduler ZMQ handlers to forward commands received from the
    Router into the inference loop's queue. The payload should already be in
    the same shape as what queue_profile_start/queue_profile_stop/queue_mem_dump
    produce (i.e. a dict with an "action" key).

    The actual application happens later in the inference thread via
    process_queue() draining the queue, preserving the
    "torch.profiler runs on the inference thread" invariant.
    """
    _put_to_profile_queue(payload)


# memory profiler
def _dump_memory_snapshot():
    rec = MemoryRecorder.get()
    if not rec.enabled and not rec.recording:
        logger.warning("dump_memory: memory recording not active, skipping")
        return
    path = rec.dump_snapshot(rec.snapshot_dir, tag="api")
    if path:
        logger.warning("dump_memory: snapshot saved -> %s", path)
    else:
        logger.warning("dump_memory: snapshot dump failed")


def _start_local_profiler(
    output_dir: str,
    activities: List[str],
    start_step: int = 0,
    num_steps: int = 10,
    with_stack: bool = False,
    profile_by_stage: bool = False,
    memory_max_entries: int = 100000,
):
    global _profile_manager

    if _profile_manager is not None and _profile_manager.active:
        logger.warning("Restarting an already active profiler.")
        _stop_local_profiler()

    mgr = ProfileManager()
    mgr.configure(
        output_dir=output_dir,
        activities=activities,
        start_step=start_step,
        num_steps=num_steps,
        with_stack=with_stack,
        profile_by_stage=profile_by_stage,
        memory_max_entries=memory_max_entries,
    )
    _profile_manager = mgr


def _stop_local_profiler():
    global _profile_manager
    mgr = _profile_manager
    _profile_manager = None
    if mgr is not None and mgr.active:
        mgr.stop()


def _drain_profile_queue() -> Optional[dict]:
    """Pop all queued payloads and return only the newest one (or None)."""
    drained: Optional[dict] = None
    while True:
        try:
            drained = _profile_cmd_queue.get_nowait()
        except queue.Empty:
            break
    return drained


def _apply_profile_command(payload: dict):
    """Apply a profile command on the local inference thread."""
    action = payload.get("action")
    logger.info("_apply_profile_command action=%s", action)
    if action == "start":
        activities = _resolve_activities(
            payload.get("activities"),
            bool(payload.get("profile_memory", False)),
        )
        _start_local_profiler(
            output_dir=payload["output_dir"],
            activities=activities,
            start_step=int(payload.get("start_step", 0)),
            num_steps=int(payload.get("num_steps", 10)),
            with_stack=bool(payload.get("with_stack", False)),
            profile_by_stage=bool(payload.get("profile_by_stage", False)),
            memory_max_entries=int(payload.get("memory_max_entries", 100000)),
        )
    elif action == "stop":
        _stop_local_profiler()
    elif action == "dump_memory":
        _dump_memory_snapshot()
    else:
        logger.warning("Ignoring unknown profiler action: %s", action)


def _drain_profile_queue_to_pending() -> None:
    global _pending_profile_payload, _pending_profile_applied

    payload = _drain_profile_queue()
    if payload is not None:
        _pending_profile_payload = payload
        _pending_profile_applied = False


def has_pending_profile_payload() -> bool:
    return _pending_profile_payload is not None


def get_pending_profile_payload() -> Optional[dict]:
    return _pending_profile_payload


def clear_pending_profile_payload() -> None:
    global _pending_profile_payload, _pending_profile_applied

    _pending_profile_payload = None
    _pending_profile_applied = False


def apply_pending_profile_command(clear_after_apply: bool) -> None:
    """Apply the newest queued profile command once on the inference thread."""
    global _pending_profile_payload, _pending_profile_applied

    _drain_profile_queue_to_pending()
    if _pending_profile_payload is None or _pending_profile_applied:
        return

    _apply_profile_command(_pending_profile_payload)
    _pending_profile_applied = True
    if clear_after_apply:
        _pending_profile_payload = None
        _pending_profile_applied = False


def receive_profile_payload(payload: dict) -> None:
    global _pending_profile_payload, _pending_profile_applied

    if _pending_profile_payload != payload:
        _pending_profile_payload = payload
        _pending_profile_applied = False
    apply_pending_profile_command(clear_after_apply=False)


def apply_profile_command(payload: dict) -> None:
    _apply_profile_command(payload)


def step_profiler(task_type: Optional[TaskType] = None):
    global _profile_manager

    mgr = _profile_manager
    if mgr is None or not mgr.active:
        return

    stage: Optional[str] = None
    if task_type == TaskType.Prefill:
        stage = "Prefill"
    elif task_type == TaskType.Decode:
        stage = "Decode"

    mgr.step(stage)

    if mgr.completed:
        _profile_manager = None


async def process_queue():
    """Process the task queue - common function used by both normal and DP modes"""
    from chitu.backend import Backend, BackendState
    from chitu.chitu_main import (
        chitu_run,
        get_last_step_task_type,
        chitu_is_terminated,
        chitu_terminate,
    )

    rank = torch.distributed.get_rank()
    global min_batch_size
    while True:
        if chitu_is_terminated():
            break
        if rank == 0:
            can_forward_profile_payload = any(
                getattr(dispatcher, "supports_profile_payload", False)
                for dispatcher in Backend.executor.task_dispatchers
            )
            apply_pending_profile_command(
                clear_after_apply=not can_forward_profile_payload
            )
            if Backend.state == BackendState.Terminating and TaskPool.all_finished():
                chitu_terminate()
                break

        has_profile_command = rank == 0 and (
            not _profile_cmd_queue.empty() or has_pending_profile_payload()
        )
        TaskPool.add_all_queued()
        if (
            rank != 0
            or Backend.state == BackendState.Terminating
            or (len(TaskPool.pool) >= min_batch_size)
            or (len(TaskPool.pool) == 0 and not TaskPool.all_finished())
            or has_profile_command
        ):
            min_batch_size = 1
            status = chitu_run()
            if status != SerializedPackedTasksPayloadType.NoneType:
                if _profile_manager is not None:
                    task_type = get_last_step_task_type()
                    if task_type is not None:
                        step_profiler(task_type=task_type)
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
    try:
        loop.run_until_complete(process_queue())
    finally:
        loop.close()


def get_priority_from_api_key(api_key: str) -> int:
    args = get_global_args()
    for item in args.serve.api_keys:
        if item.key == api_key:
            return item.priority
    if args.serve.validate_api_key == True:
        raise HTTPException(status_code=503, detail="Unauthorized api key")
    return 1


def parse_api_key_from_headers(
    authorization: Optional[str], x_api_key: Optional[str] = None
) -> str:
    if x_api_key:
        return x_api_key
    if authorization is None:
        return ""

    value = authorization.strip()
    if value == "":
        return ""

    if not value.lower().startswith("bearer"):
        raise HTTPException(
            status_code=400, detail="Authorization header must start with 'Bearer'"
        )
    return value[len("bearer") :].strip()


def build_chat_template_kwargs(
    enable_thinking: bool,
    reasoning_effort: Optional[str] = None,
) -> dict[str, Any]:
    chat_template_kwargs = {}
    if "DeepSeek-V3.1" in get_global_args().models.name:
        # DeepSeek-V3.1 tokenizer uses `thinking` instead of `enable_thinking`
        chat_template_kwargs["thinking"] = enable_thinking
    else:
        chat_template_kwargs["enable_thinking"] = enable_thinking
    if reasoning_effort is not None:
        chat_template_kwargs["reasoning_effort"] = reasoning_effort
    return chat_template_kwargs


async def submit_request(req: UserRequest):
    if get_global_args().multi_inst.n_insts > 1:
        logger.debug(f"[HTTP] Using DP mode for request: {req.request_id}")
        token_router = get_token_router()
        request_router = get_request_router()
        await request_router.add_request(req)
        await token_router.register_request(req)
    else:
        task = Task(
            req.request_id,
            req,
            stop_with_eos=req.stop_with_eos,
            priority=req.priority,
        )
        TaskPool.enqueue(task)
