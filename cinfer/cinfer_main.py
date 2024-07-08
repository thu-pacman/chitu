import torch.distributed
from .executor import Executor
from .scheduler import Scheduler
from .task import PackedTasks
from .model import Backend
import torch
from logging import getLogger

logger = getLogger(__name__)


def cinfer_init(args):
    rank = torch.distributed.get_rank()
    if rank == 0:
        scheduler = Scheduler.build(args.scheduler)
        Backend.scheduler = scheduler
    executor = Executor.build(args.executor)
    Backend.executor = executor
    PackedTasks.max_num_tasks = 32


def update_ongoing_reqs():
    logger.warning(
        f"rank {torch.distributed.get_rank()} update_ongoing_reqs: {len(Backend.ongoing_reqs)} "
    )
    to_remove = []
    unwait_task_ids = []
    for ogr in Backend.ongoing_reqs:
        if ogr.handle.is_completed():
            logger.warning(
                f"rank {torch.distributed.get_rank()} {ogr.logits.shape} logit completed"
            )
            # logits = ogr.logits
            # for i, req in enumerate(ogr.reqs):
            #     req.add_data(
            #         Backend.tokenizer.decode([torch.argmax(logits[i], dim=-1).item()])
            #     )
            for task in ogr.tasks:
                task.unwait()
                unwait_task_ids.append(task.task_id)
            to_remove.append(ogr)
    for tr in to_remove:
        Backend.ongoing_reqs.remove(tr)
    return unwait_task_ids


def cinfer_run():
    rank = torch.distributed.get_rank()
    if rank == 0:
        task_ids = Backend.scheduler.schedule()
        if len(task_ids) == 0:
            unwait_task_ids = update_ongoing_reqs()
            Backend.scheduler.update(unwait_task_ids)
            return
        tasks = PackedTasks(task_ids, rank)
    else:
        tasks = None
    logits = Backend.executor.step(tasks)
    if rank == 0:
        unwait_task_ids = update_ongoing_reqs()
        Backend.scheduler.update(unwait_task_ids)
