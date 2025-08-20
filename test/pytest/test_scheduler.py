from typing_extensions import override
from omegaconf import OmegaConf

from chitu.task import Task, TaskPool, MockFixedLengthedUserRequest
from chitu.scheduler import Scheduler
from chitu.global_vars import set_global_args


def test_prefill_first():
    set_global_args(
        OmegaConf.create({"infer": {"max_seq_len": 1024, "op_impl": "torch"}}),
        need_ensure=False,
    )

    tasks = []
    for i in range(9):
        req = MockFixedLengthedUserRequest(
            input_len=10, request_id=f"req_{i}", enable_reasoning=False
        )
        task = Task(f"{req.request_id}", req)
        tasks.append(task)
    tasks[2].start_decoding()
    tasks[5].start_decoding()
    tasks[6].start_decoding()

    TaskPool.add(tasks[7])
    TaskPool.add(tasks[2])
    TaskPool.add(tasks[1])
    TaskPool.add(tasks[5])
    TaskPool.add(tasks[3])
    TaskPool.add(tasks[8])
    TaskPool.add(tasks[0])
    TaskPool.add(tasks[4])
    TaskPool.add(tasks[6])

    scheduler = Scheduler(4, 2, "prefill_first")

    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_7", "req_1", "req_3", "req_8"])
    for task_id in batch1_ids:
        TaskPool.remove(task_id)

    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_0", "req_4"])
    for task_id in batch2_ids:
        TaskPool.remove(task_id)

    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(["req_2", "req_5"])
    for task_id in batch3_ids:
        TaskPool.remove(task_id)

    batch4_ids = scheduler.schedule()
    assert sorted(batch4_ids) == sorted(["req_6"])
    for task_id in batch4_ids:
        TaskPool.remove(task_id)

    batch5_ids = scheduler.schedule()
    assert len(batch5_ids) == 0


def test_fcfs():
    set_global_args(
        OmegaConf.create({"infer": {"max_seq_len": 1024, "op_impl": "torch"}}),
        need_ensure=False,
    )

    tasks = []
    for i in range(9):
        req = MockFixedLengthedUserRequest(
            input_len=10, request_id=f"req_{i}", enable_reasoning=False
        )
        task = Task(f"{req.request_id}", req)
        tasks.append(task)
    tasks[6].start_decoding()
    tasks[7].start_decoding()
    tasks[8].start_decoding()

    TaskPool.add(tasks[7])
    TaskPool.add(tasks[2])
    TaskPool.add(tasks[1])
    TaskPool.add(tasks[5])
    TaskPool.add(tasks[3])
    TaskPool.add(tasks[8])
    TaskPool.add(tasks[0])
    TaskPool.add(tasks[4])
    TaskPool.add(tasks[6])

    scheduler = Scheduler(4, 4, "fcfs")

    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_0", "req_1", "req_2", "req_3"])
    for task_id in batch1_ids:
        TaskPool.remove(task_id)

    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_4", "req_5"])
    for task_id in batch2_ids:
        TaskPool.remove(task_id)

    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(["req_6", "req_7", "req_8"])
    for task_id in batch3_ids:
        TaskPool.remove(task_id)

    batch4_ids = scheduler.schedule()
    assert len(batch4_ids) == 0


def test_request_preset_over_prefill_first():
    set_global_args(
        OmegaConf.create({"infer": {"max_seq_len": 1024, "op_impl": "torch"}}),
        need_ensure=False,
    )

    tasks = []
    for i in range(9):
        req = MockFixedLengthedUserRequest(
            input_len=10, request_id=f"req_{i}", enable_reasoning=False
        )
        task = Task(f"{req.request_id}", req, priority=2 if i in [0, 3, 4, 6] else 1)
        tasks.append(task)
    tasks[2].start_decoding()
    tasks[5].start_decoding()
    tasks[6].start_decoding()

    TaskPool.add(tasks[7])
    TaskPool.add(tasks[2])
    TaskPool.add(tasks[1])
    TaskPool.add(tasks[5])
    TaskPool.add(tasks[3])
    TaskPool.add(tasks[8])
    TaskPool.add(tasks[0])
    TaskPool.add(tasks[4])
    TaskPool.add(tasks[6])

    scheduler = Scheduler(4, 2, "request_preset,prefill_first")

    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_7", "req_3", "req_0", "req_4"])
    for task_id in batch1_ids:
        TaskPool.remove(task_id)

    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_2", "req_6"])
    for task_id in batch2_ids:
        TaskPool.remove(task_id)

    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(["req_1", "req_8"])
    for task_id in batch3_ids:
        TaskPool.remove(task_id)

    batch4_ids = scheduler.schedule()
    assert sorted(batch4_ids) == sorted(["req_5"])
    for task_id in batch4_ids:
        TaskPool.remove(task_id)

    batch5_ids = scheduler.schedule()
    assert len(batch5_ids) == 0
