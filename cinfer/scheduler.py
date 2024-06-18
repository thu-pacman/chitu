from .task import TaskPool, TaskType


class Scheduler:
    @staticmethod
    def build(args):
        if args.type.lower() == "fifo":
            return FifoScheduler(args.fifo.num_tasks)
        if args.type.lower() == "prefill_first":
            return PrefillFirstScheduler(args.prefill_first.num_tasks)
        else:
            raise NotImplementedError(f"Scheduler {args.type} not implemented")

    def __init__(self):
        self.ret_task_ids = []

    def schedule(self) -> list[str]:
        assert len(self.ret_task_ids) == 0
        return self.ret_task_ids

    def update(self):
        for task_id in self.ret_task_ids:
            if TaskPool.pool[task_id].need_remove():
                assert TaskPool.remove(task_id), "Task not found in pool"
        self.ret_task_ids = []  # reset scheduled tasks

    def is_done(self):
        return len(TaskPool.pool) == 0


class FifoScheduler(Scheduler):
    def __init__(self, num_tasks):
        super().__init__()
        assert num_tasks > 0, "num_tasks must be greater than 0"
        self.num_tasks = num_tasks

    def schedule(self) -> list[str]:
        super().schedule()
        self.ret_task_ids = TaskPool.id_list[: self.num_tasks]
        return self.ret_task_ids


class PrefillFirstScheduler(Scheduler):
    def __init__(self, num_tasks):
        super().__init__()
        assert num_tasks > 0, "num_tasks must be greater than 0"
        self.num_tasks = num_tasks

    def schedule(self) -> list[str]:
        super().schedule()
        tasks_to_run = []
        # select at most num_tasks prefill tasks
        for task_id in TaskPool.id_list:
            if TaskPool.pool[task_id].task_type == TaskType.Prefill:
                tasks_to_run.append[task_id]
                if len(tasks_to_run) == self.num_tasks:
                    break
        # if no prefill tasks, select at most num_tasks decode tasks
        if len(tasks_to_run) == 0:
            tasks_to_run = TaskPool.id_list[: self.num_tasks]
        # no hybrid tasks, as hybrid executor is not implemented yet
        return tasks_to_run
        