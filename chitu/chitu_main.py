import logging
from logging import getLogger

import torch
import torch.distributed

from chitu.backend import Backend, BackendState
from chitu.executor import Executor
from chitu.global_vars import set_global_variables
from chitu.scheduler import Scheduler
from chitu.task import (
    PackedTasks,
    PackedTasksBase,
    SerializedPackedTasksPayloadType,
    TaskPool,
    TaskType,
    req_encode,
)
from chitu.tensor_parallel import get_tp_group

logger = getLogger(__name__)


def init_logger(logging_level=logging.INFO):
    base_name = __name__.split(".")[0]
    base_logger = getLogger(base_name)
    base_logger.setLevel(logging.INFO)

    def add_rank_to_msg(record):
        if torch.distributed.is_initialized():
            record.msg = f"[Rank {torch.distributed.get_rank()}] {record.msg}"
        return True

    def add_filter_to_all_parent_handlers(cur_logger):
        for handler in cur_logger.handlers:
            handler.addFilter(add_rank_to_msg)
        if cur_logger.parent:
            add_filter_to_all_parent_handlers(cur_logger.parent)

    # If there is no handlers, create a new handler to hold the filter. If not (very likely
    # because we launch from Hydra, and Hydra setup the root logger), a new handler will only
    # duplicate the logs. In this case, we should add the filter to all the existing handlers.
    if base_logger.hasHandlers():  # Including handlers from the parents
        add_filter_to_all_parent_handlers(base_logger)
    else:
        handler = logging.StreamHandler()
        handler.addFilter(add_rank_to_msg)
        base_logger.addHandler(handler)


def chitu_init(args, logging_level=logging.INFO):
    init_logger(logging_level)
    set_global_variables(args)
    Backend.build(args)
    rank = torch.distributed.get_rank()
    if rank == 0:
        scheduler = Scheduler.build(args.scheduler)
        Backend.scheduler = scheduler
    executor = Executor.build(args)
    Backend.executor = executor
    PackedTasks.configure(max_num_tasks=args.infer.max_reqs)


def remove_task_other_device(remove_task_ids):
    # TODO handle in executor?
    if len(remove_task_ids) == 0:
        return
    # Since we are removing, any task type is fine
    task_tensor = PackedTasksBase(
        len(remove_task_ids), remove_task_ids, remove_task_ids, TaskType.Decode
    ).serialize(payload_type=SerializedPackedTasksPayloadType.EndTask, device=0)
    if Backend.args.infer.pp_size > 1:
        torch.distributed.isend(tensor=task_tensor, dst=Backend.args.infer.tp_size)
        if Backend.args.infer.tp_size > 1:
            torch.distributed.broadcast(
                tensor=task_tensor, src=Backend.pp_main_rank, group=get_tp_group()
            )
    elif Backend.args.infer.tp_size > 1:
        torch.distributed.broadcast(tensor=task_tensor, src=0)


def update_ongoing_tasks():
    to_remove = []
    unwait_tasks = []
    logits_list = []
    for ogr in Backend.ongoing_reqs:
        if ogr.handle.is_completed():
            to_remove.append(ogr)
            unwait_tasks.append(ogr.waiting_task)
            logits_list.append(ogr.logits.view(-1, ogr.logits.shape[-1]))
            for task in ogr.waiting_task.tasks:
                task.unwait()
    for tr in to_remove:
        Backend.ongoing_reqs.remove(tr)
    return unwait_tasks, logits_list


def chitu_update(task_ids, rank, world_size):
    if rank == 0:
        TaskPool.display()
    if world_size == 1:
        Backend.scheduler.update(task_ids)
    else:
        unwait_tasks, logits = update_ongoing_tasks()
        for idx, task in enumerate(unwait_tasks):
            Backend.executor.update_response(task, logits[idx])
        unwait_task_ids = [t.task_id for task in unwait_tasks for t in task.tasks]
        removed_decode_task_ids = Backend.scheduler.update(task_ids, unwait_task_ids)
        remove_task_other_device(removed_decode_task_ids)


@torch.inference_mode()
def chitu_run():
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if rank == 0:
        task_ids = Backend.scheduler.schedule()
        if len(task_ids) == 0:  # no tasks to do, but some tasks are waiting
            chitu_update(task_ids, rank, world_size)
            return
        if rank == 0:
            logger.debug(f"Processing {task_ids}")
        tasks = PackedTasks(task_ids, rank)
    else:
        tasks = None
    Backend.executor.step(tasks)

    if Backend.args.infer.pp_size > 1 and rank == 0:
        chitu_update(task_ids, rank, world_size)
    elif rank == 0:
        TaskPool.display()
        removed_decode_task_ids = Backend.scheduler.update(task_ids)
        if world_size != 1:
            remove_task_other_device(removed_decode_task_ids)


def chitu_terminate():
    if torch.distributed.get_rank() == 0:
        Backend.state = BackendState.Terminating
        Backend.executor.step(None)


def chitu_is_terminated():
    return Backend.state == BackendState.Terminated
