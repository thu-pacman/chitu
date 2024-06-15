import torch
from .task import TaskPool


def simulation_driver():
    sched = TaskSchedulerFCFS()
    sched.meta["max_task_batchsize"] = 4
    uid = "u"
    for i in range(10):
        rid = f"r0{i}"
        prompt = "This is a test prompt"
        sched.append_user_request(UserPromptRequest(uid, rid, prompt, 128))
    tids_to_run = []
    while ScheduleHelper.global_timer_counter < 1000:
        tids_to_run = sched.schedule(tids_to_run)
        # execute
        ScheduleHelper.timer_tick()


class Scheduler:
    def __init__(self):
        self.ret_task_ids = []

    def schedule(self) -> list[int]:
        assert len(self.ret_task_ids) == 0
        return self.ret_task_ids

    def update(self, tids_to_run):
        pass


class FifoScheduler(Scheduler):
    def __init__(self, num_tasks):
        super().__init__()
        assert num_tasks > 0, "num_tasks must be greater than 0"
        self.num_tasks = num_tasks

    def schedule(self) -> list[str]:
        super().schedule()
        self.ret_task_ids = TaskPool.id_list[: self.num_tasks]
        return self.ret_task_ids

    def update(self, new_tasks):
        for task in self.ret_task_ids:
            assert TaskPool.remove(task), "Task not found in pool"
        for task in new_tasks:
            assert TaskPool.add(task)
