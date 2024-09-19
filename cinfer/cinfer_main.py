import torch.distributed
from .executor import Executor
from .scheduler import Scheduler
from .task import PackedTasks, req_encode, TaskPool
from .backend import Backend
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
    PackedTasks.max_num_tasks = args.infer.max_reqs


def remove_task_other_device(remove_task_ids):
    if len(remove_task_ids) == 0:
        return
    encoded = [0] * (PackedTasks.max_num_tasks * 2)
    encoded[PackedTasks.max_num_tasks] = -1  # Flag to indicate ending these tasks
    for it, tid in enumerate(remove_task_ids):
        encoded[it] = req_encode(tid)
    task_tensor = torch.tensor(
        encoded,
        dtype=torch.int64,
        device=0,
    )
    torch.distributed.isend(tensor=task_tensor, dst=1)


def update_ongoing_reqs():
    to_remove = []
    unwait_task_ids = []
    for ogr in Backend.ongoing_reqs:
        if ogr.handle.is_completed():
            to_remove.append(ogr)
            for task in ogr.tasks:
                task.unwait()
                unwait_task_ids.append(task.task_id)
    for tr in to_remove:
        Backend.ongoing_reqs.remove(tr)
    return unwait_task_ids


def update_ongoing_tasks():
    unwait_task_ids = update_ongoing_reqs()
    return unwait_task_ids


def cinfer_update(task_ids, rank, world_size):
    if rank == 0:
        TaskPool.display()
    removed_decode_task_ids = Backend.scheduler.update(
        task_ids, update_ongoing_tasks() if world_size > 1 else []
    )
    if world_size > 1:
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
    if Backend.parallel_type == "pipe" and rank == 0:
        cinfer_update(task_ids, rank, world_size)
    elif rank == 0:
        TaskPool.display()
        Backend.scheduler.update(task_ids)
