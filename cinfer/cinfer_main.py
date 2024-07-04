from .executor import Executor
from .scheduler import Scheduler
from .task import PackedTasks
from .model import Backend
import torch


def cinfer_init(args):
    rank = torch.distributed.get_rank()
    if rank == 0:
        scheduler = Scheduler.build(args.scheduler)
        Backend.scheduler = scheduler
    executor = Executor.build(args.executor)
    Backend.executor = executor
    PackedTasks.max_num_tasks = 32


def cinfer_run():
    rank = torch.distributed.get_rank()
    if rank == 0:
        task_ids = Backend.scheduler.schedule()
        if len(task_ids) == 0:
            Backend.update_ongoing_reqs()
            return
        tasks = PackedTasks(task_ids, rank)
    else:
        tasks = None
    logits = Backend.executor.step(tasks)
    if rank == 0:
        Backend.update_ongoing_reqs()
        Backend.scheduler.update()
