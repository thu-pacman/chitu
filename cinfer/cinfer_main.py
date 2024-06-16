from .executor import Executor
from .scheduler import Scheduler
from .task import PackedTasks
from .model import Backend


def cinfer_init(args):
    executor = Executor.build(args.executor)
    scheduler = Scheduler.build(args.scheduler)
    Backend.executor = executor
    Backend.scheduler = scheduler


def cinfer_run():
    task_ids = Backend.scheduler.schedule()
    logits, new_tasks = Backend.executor.step(PackedTasks(task_ids))
    Backend.scheduler.update(new_tasks)
