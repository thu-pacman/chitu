import torch.distributed
from .executor import Executor
from .scheduler import Scheduler
from .task import (
    SerializedPackedTasksPayloadType,
    PackedTasksBase,
    PackedTasks,
    req_encode,
    TaskPool,
    TaskType,
)
from .backend import Backend, BackendState
from .tensor_parallel import get_tp_group
import torch
from logging import getLogger

logger = getLogger(__name__)


def cinfer_init(args):
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


def cinfer_update(task_ids, rank, world_size):
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
def cinfer_run():
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if rank == 0:
        task_ids = Backend.scheduler.schedule()
        if len(task_ids) == 0:  # no tasks to do, but some tasks are waiting
            cinfer_update(task_ids, rank, world_size)
            return
        # if rank == 0:
        #     logger.warning(f"Processing {task_ids}")
        tasks = PackedTasks(task_ids, rank)
    else:
        tasks = None
    Backend.executor.step(tasks)

    if Backend.args.infer.pp_size > 1 and rank == 0:
        cinfer_update(task_ids, rank, world_size)
    elif rank == 0:
        TaskPool.display()
        removed_decode_task_ids = Backend.scheduler.update(task_ids)
        if world_size != 1:
            remove_task_other_device(removed_decode_task_ids)


def cinfer_terminate():
    if torch.distributed.get_rank() == 0:
        Backend.state = BackendState.Terminating
        Backend.executor.step(None)


def cinfer_is_terminated():
    return Backend.state == BackendState.Terminated
