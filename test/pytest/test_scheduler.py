from typing_extensions import override
from omegaconf import OmegaConf

from chitu.task import Task, TaskPool, MockFixedLengthedUserRequest
from chitu.scheduler import Scheduler
from chitu.global_vars import set_global_args


def test_prefill_first():
    set_global_args(
        OmegaConf.create({"infer": {"max_seq_len": 1024}}),
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
    assert len(batch1_ids) == 4
    assert "req_7" in batch1_ids
    assert "req_1" in batch1_ids
    assert "req_3" in batch1_ids
    assert "req_8" in batch1_ids
    for task_id in batch1_ids:
        TaskPool.remove(task_id)

    batch2_ids = scheduler.schedule()
    assert len(batch2_ids) == 2
    assert "req_0" in batch2_ids
    assert "req_4" in batch2_ids
    for task_id in batch2_ids:
        TaskPool.remove(task_id)

    batch3_ids = scheduler.schedule()
    assert len(batch3_ids) == 2
    assert "req_2" in batch3_ids
    assert "req_5" in batch3_ids
    for task_id in batch3_ids:
        TaskPool.remove(task_id)

    batch4_ids = scheduler.schedule()
    assert len(batch4_ids) == 1
    assert "req_6" in batch4_ids
    for task_id in batch4_ids:
        TaskPool.remove(task_id)

    batch5_ids = scheduler.schedule()
    assert len(batch5_ids) == 0


def test_fcfs():
    set_global_args(
        OmegaConf.create({"infer": {"max_seq_len": 1024}}),
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
    assert len(batch1_ids) == 4
    assert "req_0" in batch1_ids
    assert "req_1" in batch1_ids
    assert "req_2" in batch1_ids
    assert "req_3" in batch1_ids
    for task_id in batch1_ids:
        TaskPool.remove(task_id)

    batch2_ids = scheduler.schedule()
    assert len(batch2_ids) == 2
    assert "req_4" in batch2_ids
    assert "req_5" in batch2_ids
    for task_id in batch2_ids:
        TaskPool.remove(task_id)

    batch3_ids = scheduler.schedule()
    assert len(batch3_ids) == 3
    assert "req_6" in batch3_ids
    assert "req_7" in batch3_ids
    assert "req_8" in batch3_ids
    for task_id in batch3_ids:
        TaskPool.remove(task_id)

    batch4_ids = scheduler.schedule()
    assert len(batch4_ids) == 0
