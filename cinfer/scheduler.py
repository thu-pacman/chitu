from .task import TaskPool


class Scheduler:
    @staticmethod
    def build(args):
        if args.type.lower() == "fifo":
            return FifoScheduler(args.num_tasks)
        else:
            raise NotImplementedError(f"Scheduler {args.type} not implemented")

    def __init__(self):
        self.ret_task_ids = []

    def schedule(self) -> list[int]:
        assert len(self.ret_task_ids) == 0
        return self.ret_task_ids

    def update(self, new_tasks):
        for task in self.ret_task_ids:
            if task.need_remove():
                assert TaskPool.remove(task.task_id), "Task not found in pool"
        for task in new_tasks:
            assert TaskPool.add(task)
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
