from omegaconf import OmegaConf

from chitu.task import Task, TaskPool, UserRequest
from chitu.kv_cache import PagedKVCacheManager
from chitu.scheduler import Scheduler, SkewScheduler
from chitu.global_vars import set_global_args, set_slot_handle, get_global_args
from chitu.backend import Backend
import pytest
from chitu.task_type import TaskType


class MockExecutor:
    def __init__(self):
        pass

    def step(self, tasks):
        pass

    def special_step(self, task_ids, type):
        pass


class MockTokenizer:
    def __init__(self):
        self.stop_tokens = [2]


def test_prefix_cache_probe_does_not_fill_single_slot_scheduler():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 2048,
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    TaskPool.reset()
    cache_manager = PagedKVCacheManager(
        num_blocks=100,
        num_hot_req=1,
        max_seq_len=2048,
        dp_rank=0,
        block_size=512,
        enable_prefix_caching=True,
    )
    Backend.cache_managers = [{"main": cache_manager}]
    Backend.executor = MockExecutor()

    req = UserRequest.create_mock(
        input_len=600, request_id="req_prefix_probe", enable_thinking=False
    )
    task = Task(req.request_id, req)
    TaskPool.add(task)

    assert cache_manager.num_cached_blocks(task) == 0
    assert task.task_id not in cache_manager.task_to_cache_ids

    scheduler = Scheduler(
        max_running_tasks=1,
        prefill_num_tasks=1,
        decode_num_tasks=1,
        scheduler_type="prefill_first",
        cache_manager_dict=Backend.cache_managers[0],
        num_scheduler_groups=1,
        prefill_chunk_size=None,
    )

    scheduler.prepare_for_schedule()
    assert scheduler.schedule() == [task.task_id]
    TaskPool.reset()


def test_chunked_prefill():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 32768,
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    TaskPool.reset()
    Backend.cache_managers = [
        {
            "main": PagedKVCacheManager(
                num_blocks=10000,
                num_hot_req=10,
                max_seq_len=32768,
                dp_rank=0,
                block_size=5120,
            )
        }
    ]

    Backend.executor = MockExecutor()

    for i in range(4):
        req = UserRequest.create_mock(
            input_len=1000 * (i + 1), request_id=f"req_{i}", enable_thinking=False
        )
        task = Task(f"{req.request_id}", req)
        TaskPool.add(task)

    scheduler = Scheduler(
        max_running_tasks=100,
        prefill_num_tasks=4,
        decode_num_tasks=4,
        scheduler_type="prefill_first",
        cache_manager_dict=Backend.cache_managers[0],
        num_scheduler_groups=1,
        prefill_chunk_size=4096,
    )

    # Prefill:

    # Remaining: [1000, 2000, 3000, 4000]

    scheduler.prepare_for_schedule()
    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_0", "req_1", "req_2"])
    for task_id in batch1_ids:
        TaskPool.pool[task_id].consume_req_tokens()

    scheduler.update(batch1_ids)
    # Remaining: [0, 0, 1904, 4000]

    scheduler.prepare_for_schedule()
    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_2", "req_3"])
    for task_id in batch2_ids:
        TaskPool.pool[task_id].consume_req_tokens()

    scheduler.update(batch2_ids)
    # Remaining: [0, 0, 0, 1808]

    scheduler.prepare_for_schedule()
    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(["req_3"])
    for task_id in batch3_ids:
        TaskPool.pool[task_id].consume_req_tokens()

    scheduler.update(batch3_ids)
    # Remaining: [0, 0, 0, 0]

    # Decode:

    scheduler.prepare_for_schedule()
    batch4_ids = scheduler.schedule()
    assert sorted(batch4_ids) == sorted(["req_0", "req_1", "req_2", "req_3"])


def test_chunked_prefill_skew():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 32768,
                    "max_batch_size": 256,
                    "op_impl": "torch",
                    "cache_type": "skew",
                    "pp_size": 2,
                    "dp_size": 1,
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )

    infer_args = get_global_args().infer
    set_slot_handle(
        infer_args.max_batch_size,
        infer_args.pp_size,
    )

    TaskPool.reset()
    Backend.cache_managers = None

    for i in range(4):
        req = UserRequest.create_mock(
            input_len=1000 * (i + 1), request_id=f"req_{i}", enable_thinking=False
        )
        task = Task(f"{req.request_id}", req)
        TaskPool.add(task)

    scheduler = SkewScheduler(
        max_batch_size=infer_args.max_batch_size,
        cache_manager_dict=None,
        original_scheduler_type="prefill_first",
        prefill_chunk_size=4096,
    )

    # Prefill:

    # Slot groups: [[],[]], sgroup head at: 0
    # Task length remaining: [1000, 2000, 3000, 4000]
    scheduler.prepare_for_schedule()
    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_0", "req_1", "req_2"])
    for task_id in batch1_ids:
        TaskPool.pool[task_id].consume_req_tokens()
    scheduler.update(batch1_ids)

    # Slot groups: [[], ["req_0", "req_1", "req_2", 'req_3']], sgroup head at: 1
    # Task length remaining:: [0, 0, 1904, 4000]
    scheduler.prepare_for_schedule()
    empty_ids = scheduler.schedule()
    assert empty_ids == []  # skewScheduler doesn't change task's slot group
    scheduler.update([])

    # Slot groups: [[], ["req_0", "req_1", "req_2", 'req_3'],[]], sgroup head at: 0
    # Task length remaining: [0, 0, 1904, 4000]
    scheduler.prepare_for_schedule()
    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_2", "req_3"])
    for task_id in batch2_ids:
        TaskPool.pool[task_id].consume_req_tokens()
    scheduler.update(batch2_ids)

    # Slot groups: [[], ["req_0", "req_1", "req_2", 'req_3']], sgroup head at: 1
    # Task length remaining: [0, 0, 0, 1808]
    scheduler.prepare_for_schedule()
    empty_ids = scheduler.schedule()
    assert empty_ids == []
    scheduler.update([])

    # Slot groups: [[], ["req_0", "req_1", "req_2", 'req_3']], sgroup head at: 0
    # Task length remaining: [0, 0, 0, 1808]
    scheduler.prepare_for_schedule()
    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(["req_3"])
    for task_id in batch3_ids:
        TaskPool.pool[task_id].consume_req_tokens()
    scheduler.update(batch3_ids)

    # Slot groups: [["req_0", "req_1", "req_2", 'req_3'],[]], sgroup head at: 1
    # Remaining: [0, 0, 0, 0]
    scheduler.prepare_for_schedule()
    empty_ids = scheduler.schedule()
    assert empty_ids == []
    scheduler.update([])

    # Decode:
    # Slot groups: [[], ["req_0", "req_1", "req_2", 'req_3']], sgroup head at: 0
    # Remaining: [0, 0, 0, 0]
    scheduler.prepare_for_schedule()
    batch4_ids = scheduler.schedule()
    assert sorted(batch4_ids) == sorted(["req_0", "req_1", "req_2", "req_3"])


def test_priority_prefill_first():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 1024,
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    TaskPool.reset()
    Backend.cache_managers = [
        {
            "main": PagedKVCacheManager(
                num_blocks=10,
                num_hot_req=100,
                max_seq_len=1024,
                dp_rank=0,
                block_size=512,
            )
        }
    ]

    scheduler = Scheduler(
        max_running_tasks=100,
        prefill_num_tasks=4,
        decode_num_tasks=2,
        cache_manager_dict=Backend.cache_managers[0],
        scheduler_type="prefill_first",
        num_scheduler_groups=1,
    )

    main_manager: PagedKVCacheManager = Backend.cache_managers[0]["main"]

    tasks = []
    for i in range(9):
        req = UserRequest.create_mock(
            input_len=10, request_id=f"req_{i}", enable_thinking=False
        )
        task = Task(f"{req.request_id}", req)
        tasks.append(task)

    # 让task_2, task_5, task_6为decode状态
    for task in [tasks[2], tasks[5], tasks[6]]:
        task.dp_rank = 0
        assert main_manager.num_cached_blocks(task) == 0
        task.set_prefill_chunk_size_for_one_step(task.prefix_tokens_len)
        scheduler._prepare_prefill_metadata(task, cached_len=0)
        task.consume_req_tokens()

    TaskPool.add(tasks[7])
    TaskPool.add(tasks[2])
    TaskPool.add(tasks[1])
    TaskPool.add(tasks[5])
    TaskPool.add(tasks[3])
    TaskPool.add(tasks[8])
    TaskPool.add(tasks[0])
    TaskPool.add(tasks[4])
    TaskPool.add(tasks[6])

    scheduler.prepare_for_schedule()
    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_7", "req_1", "req_3", "req_8"])
    for task_id in batch1_ids:
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch1_ids)

    scheduler.prepare_for_schedule()
    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_0", "req_4"])
    for task_id in batch2_ids:
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch2_ids)

    scheduler.prepare_for_schedule()
    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(["req_2", "req_5"])
    for task_id in batch3_ids:
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch3_ids)

    scheduler.prepare_for_schedule()
    batch4_ids = scheduler.schedule()
    assert sorted(batch4_ids) == sorted(["req_6"])
    for task_id in batch4_ids:
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch4_ids)

    scheduler.prepare_for_schedule()
    batch5_ids = scheduler.schedule()
    assert len(batch5_ids) == 0


def test_priority_prefill_first_skew():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 1024,
                    "max_batch_size": 4,
                    "op_impl": "torch",
                    "cache_type": "skew",
                    "pp_size": 1,
                    "dp_size": 1,
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )

    infer_args = get_global_args().infer
    set_slot_handle(
        infer_args.max_batch_size,
        infer_args.pp_size,
    )

    TaskPool.reset()
    Backend.cache_managers = None
    Backend.tokenizer = MockTokenizer()

    tasks = []
    for i in range(9):
        req = UserRequest.create_mock(
            input_len=10, request_id=f"req_{i}", enable_thinking=False
        )
        task = Task(f"{req.request_id}", req)
        tasks.append(task)
    tasks[2].consume_req_tokens()
    tasks[5].consume_req_tokens()
    tasks[6].consume_req_tokens()

    TaskPool.add(tasks[7])
    TaskPool.add(tasks[2])
    TaskPool.add(tasks[1])
    TaskPool.add(tasks[5])
    TaskPool.add(tasks[3])
    TaskPool.add(tasks[8])
    TaskPool.add(tasks[0])
    TaskPool.add(tasks[4])
    TaskPool.add(tasks[6])

    scheduler = SkewScheduler(
        infer_args.max_batch_size,
        cache_manager_dict=None,
        original_scheduler_type="prefill_first",
        prefill_chunk_size=None,
    )

    # slot_groups: [[]], free_sgroups: [0]
    # TaskPool: ['req_7', 'req_2', 'req_1', 'req_5', 'req_3', 'req_8', 'req_0', 'req_4', 'req_6']
    scheduler.prepare_for_schedule()
    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_7", "req_1", "req_3", "req_8"])
    for task_id in batch1_ids:
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch1_ids)

    # slot_groups: [[]], free_sgroup: deque([0])
    # TaskPool: ['req_2', 'req_5', 'req_0', 'req_4', 'req_6']
    scheduler.prepare_for_schedule()
    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_0", "req_4"])
    for task_id in batch2_ids:
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch2_ids)

    # slot_group: [[]], free_sgroups: deque([0])
    # TaskPool: ['req_2', 'req_5', 'req_6']
    scheduler.prepare_for_schedule()
    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(
        ["req_2", "req_5", "req_6"]
    )  # skewScheduler's decode_mbs == prefill_mbs == 4
    for task_id in batch3_ids:
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch3_ids)

    # slot_group: [[]], free_sgroups: deque([0])
    # TaskPool: []
    scheduler.prepare_for_schedule()
    batch4_ids = scheduler.schedule()
    assert len(batch4_ids) == 0


def test_priority_fcfs():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 1024,
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    TaskPool.reset()

    Backend.cache_managers = [
        {
            "main": PagedKVCacheManager(
                num_blocks=10,
                num_hot_req=100,
                max_seq_len=1024,
                dp_rank=0,
                block_size=512,
            )
        }
    ]

    main_manager: PagedKVCacheManager = Backend.cache_managers[0]["main"]

    scheduler = Scheduler(
        max_running_tasks=100,
        prefill_num_tasks=4,
        decode_num_tasks=4,
        scheduler_type="fcfs",
        cache_manager_dict=Backend.cache_managers[0],
        num_scheduler_groups=1,
        dp_rank=0,
    )

    tasks = []
    for i in range(9):
        req = UserRequest.create_mock(
            input_len=10, request_id=f"req_{i}", enable_thinking=False
        )
        task = Task(f"{req.request_id}", req)
        tasks.append(task)

    # 让task_6, task_7, task_8为decode状态
    for task in [tasks[6], tasks[7], tasks[8]]:
        task.dp_rank = 0
        main_manager.num_cached_blocks(task) == 0
        task.set_prefill_chunk_size_for_one_step(task.prefix_tokens_len)
        scheduler._prepare_prefill_metadata(task, cached_len=0)
        task.consume_req_tokens()

    TaskPool.add(tasks[7])
    TaskPool.add(tasks[2])
    TaskPool.add(tasks[1])
    TaskPool.add(tasks[5])
    TaskPool.add(tasks[3])
    TaskPool.add(tasks[8])
    TaskPool.add(tasks[0])
    TaskPool.add(tasks[4])
    TaskPool.add(tasks[6])

    scheduler.prepare_for_schedule()
    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_0", "req_1", "req_2", "req_3"])
    for task_id in batch1_ids:
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch1_ids)

    scheduler.prepare_for_schedule()
    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_4", "req_5"])
    for task_id in batch2_ids:
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch2_ids)

    scheduler.prepare_for_schedule()
    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(["req_6", "req_7", "req_8"])
    for task_id in batch3_ids:
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch3_ids)

    scheduler.prepare_for_schedule()
    batch4_ids = scheduler.schedule()
    assert len(batch4_ids) == 0


def test_priority_fcfs_skew():

    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 1024,
                    "max_batch_size": 4,
                    "op_impl": "torch",
                    "cache_type": "skew",
                    "pp_size": 1,
                    "dp_size": 1,
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    infer_args = get_global_args().infer
    set_slot_handle(
        infer_args.max_batch_size,
        infer_args.pp_size,
    )

    TaskPool.reset()
    Backend.cache_managers = None
    tasks = []
    for i in range(9):
        req = UserRequest.create_mock(
            input_len=10, request_id=f"req_{i}", enable_thinking=False
        )
        task = Task(f"{req.request_id}", req)
        tasks.append(task)
    tasks[6].consume_req_tokens()
    tasks[7].consume_req_tokens()
    tasks[8].consume_req_tokens()

    TaskPool.add(tasks[7])
    TaskPool.add(tasks[2])
    TaskPool.add(tasks[1])
    TaskPool.add(tasks[5])
    TaskPool.add(tasks[3])
    TaskPool.add(tasks[8])
    TaskPool.add(tasks[0])
    TaskPool.add(tasks[4])
    TaskPool.add(tasks[6])

    scheduler = SkewScheduler(
        infer_args.max_batch_size,
        cache_manager_dict=None,
        original_scheduler_type="fcfs",
        prefill_chunk_size=None,
    )

    # slot_group: [[]], free_sgroup: deque([0]), slot_capacity: 4
    # TaskPool: ['req_7', 'req_2', 'req_1', 'req_5', 'req_3', 'req_8', 'req_0', 'req_4', 'req_6']
    scheduler.prepare_for_schedule()
    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_0", "req_1", "req_2", "req_3"])
    for task_id in batch1_ids:
        # Make task need_move
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch1_ids)

    # slot_group: [[]], free_sgroup: deque([0])
    # TaskPool: ['req_7', 'req_5', 'req_8', 'req_4', 'req_6']
    scheduler.prepare_for_schedule()
    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_4", "req_5"])
    for task_id in batch2_ids:
        # Make task need_move
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch2_ids)

    # slot_group: [[]], free_sgroup: deque([0])
    # TaskPool: ['req_7', 'req_8', 'req_6']
    scheduler.prepare_for_schedule()
    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(["req_6", "req_7", "req_8"])

    for task_id in batch3_ids:
        # Make task need_move
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch3_ids)

    # slot_group: [[]], free_sgroup: deque([0])
    # TaskPool: []
    scheduler.prepare_for_schedule()
    batch4_ids = scheduler.schedule()
    assert len(batch4_ids) == 0


def test_priority_request_preset_over_prefill_first():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 1024,
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    TaskPool.reset()
    Backend.cache_managers = [
        {
            "main": PagedKVCacheManager(
                num_blocks=10,
                num_hot_req=10,
                max_seq_len=1024,
                dp_rank=0,
                block_size=512,
            )
        }
    ]

    main_manager = Backend.cache_managers[0]["main"]

    scheduler = Scheduler(
        100,
        4,
        2,
        "request_preset,prefill_first",
        cache_manager_dict=Backend.cache_managers[0],
        num_scheduler_groups=1,
    )

    tasks = []
    for i in range(9):
        req = UserRequest.create_mock(
            input_len=10, request_id=f"req_{i}", enable_thinking=False
        )
        task = Task(f"{req.request_id}", req, priority=2 if i in [0, 3, 4, 6] else 1)
        tasks.append(task)

    # 让task_2, task_5, task_6为decode状态
    for task in [tasks[2], tasks[5], tasks[6]]:
        task.dp_rank = 0
        main_manager.num_cached_blocks(task) == 0
        task.set_prefill_chunk_size_for_one_step(task.prefix_tokens_len)
        scheduler._prepare_prefill_metadata(task, cached_len=0)
        task.consume_req_tokens()

    TaskPool.add(tasks[7])
    TaskPool.add(tasks[2])
    TaskPool.add(tasks[1])
    TaskPool.add(tasks[5])
    TaskPool.add(tasks[3])
    TaskPool.add(tasks[8])
    TaskPool.add(tasks[0])
    TaskPool.add(tasks[4])
    TaskPool.add(tasks[6])

    # ['req_7', 'req_2':Decode, 'req_1', 'req_5':Decode, 'req_3', 'req_8', 'req_0', 'req_4', 'req_6':Decode]
    scheduler.prepare_for_schedule()
    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_7", "req_3", "req_0", "req_4"])
    for task_id in batch1_ids:
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch1_ids)

    # TaskPool: ['req_2':Decode, 'req_1', 'req_5':Decode, 'req_8', 'req_6':Decode]
    scheduler.prepare_for_schedule()
    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_2", "req_6"])
    for task_id in batch2_ids:
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch2_ids)

    # TaskPool: ['req_1', 'req_5':Decode, 'req_8']
    scheduler.prepare_for_schedule()
    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(["req_1", "req_8"])
    for task_id in batch3_ids:
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch3_ids)

    # TaskPool: ['req_5':Decode]
    scheduler.prepare_for_schedule()
    batch4_ids = scheduler.schedule()
    assert sorted(batch4_ids) == sorted(["req_5"])
    for task_id in batch4_ids:
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch4_ids)

    scheduler.prepare_for_schedule()
    batch5_ids = scheduler.schedule()
    assert len(batch5_ids) == 0


def test_priority_request_preset_over_prefill_first_skew():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 1024,
                    "max_batch_size": 4,
                    "op_impl": "torch",
                    "cache_type": "skew",
                    "pp_size": 1,
                    "dp_size": 1,
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    infer_args = get_global_args().infer
    set_slot_handle(
        infer_args.max_batch_size,
        infer_args.pp_size,
    )

    TaskPool.reset()
    Backend.cache_managers = None

    tasks = []
    for i in range(9):
        req = UserRequest.create_mock(
            input_len=10, request_id=f"req_{i}", enable_thinking=False
        )
        task = Task(f"{req.request_id}", req, priority=2 if i in [0, 3, 4, 6] else 1)
        tasks.append(task)
    tasks[2].consume_req_tokens()  # Decode task
    tasks[5].consume_req_tokens()
    tasks[6].consume_req_tokens()

    TaskPool.add(tasks[7])
    TaskPool.add(tasks[2])
    TaskPool.add(tasks[1])
    TaskPool.add(tasks[5])
    TaskPool.add(tasks[3])
    TaskPool.add(tasks[8])
    TaskPool.add(tasks[0])
    TaskPool.add(tasks[4])
    TaskPool.add(tasks[6])

    # skewScheduler's prefill_mbs == decode_mbs == 4
    scheduler = SkewScheduler(
        infer_args.max_batch_size,
        cache_manager_dict=None,
        original_scheduler_type="request_preset,prefill_first",
        prefill_chunk_size=None,
    )

    # TaskPool: ['req_7', 'req_2', 'req_1', 'req_5', 'req_3', 'req_8', 'req_0', 'req_4', 'req_6']
    # slot_group: [[]], free_sgroup: deque([0])
    scheduler.prepare_for_schedule()
    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_7", "req_3", "req_0", "req_4"])
    for task_id in batch1_ids:
        # Make task need_move
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch1_ids)

    # TaskPool: ['req_2', 'req_1', 'req_5', 'req_8', 'req_6']
    # slot_group: [[]], free_sgroup: deque([0])
    scheduler.prepare_for_schedule()
    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_2", "req_6", "req_5"])
    for task_id in batch2_ids:
        # Make task need_move
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch2_ids)

    # TaskPool: ['req_1', 'req_8']
    # slot_group: [[]], free_sgroup: deque([0])
    scheduler.prepare_for_schedule()
    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(["req_1", "req_8"])
    for task_id in batch3_ids:
        # Make task need_move
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch3_ids)

    # TaskPool: []
    # slot_group: [[]], free_sgroup: deque([0])
    scheduler.prepare_for_schedule()
    batch4_ids = scheduler.schedule()
    assert len(batch4_ids) == 0


def test_max_running_tasks():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 1024,
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    TaskPool.reset()
    Backend.cache_managers = [
        {
            "main": PagedKVCacheManager(
                num_blocks=10,
                num_hot_req=10,
                max_seq_len=1024,
                dp_rank=0,
                block_size=512,
            )
        }
    ]

    tasks = []
    for i in range(9):
        req = UserRequest.create_mock(
            input_len=10, request_id=f"req_{i}", enable_thinking=False
        )
        task = Task(f"{req.request_id}", req)
        tasks.append(task)

    TaskPool.add(tasks[0])
    TaskPool.add(tasks[1])
    TaskPool.add(tasks[2])
    TaskPool.add(tasks[3])
    TaskPool.add(tasks[4])
    TaskPool.add(tasks[5])
    TaskPool.add(tasks[6])
    TaskPool.add(tasks[7])
    TaskPool.add(tasks[8])

    scheduler = Scheduler(
        4,
        4,
        4,
        "prefill_first",
        cache_manager_dict=Backend.cache_managers[0],
        num_scheduler_groups=1,
    )

    scheduler.prepare_for_schedule()
    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_0", "req_1", "req_2", "req_3"])

    # Now req_[0-3] are decoding
    tasks[0].consume_req_tokens()
    tasks[1].consume_req_tokens()
    tasks[2].consume_req_tokens()
    tasks[3].consume_req_tokens()
    scheduler.update(batch1_ids)

    # No to schedule req_[4-8] although they are prefill (higher priority),
    # because reaching max_running_tasks=4
    scheduler.prepare_for_schedule()
    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(["req_0", "req_1", "req_2", "req_3"])
    scheduler.update(batch3_ids)


def test_single_prompt_seq_bigger_than_scheduler_capacity():
    """test when single prompt length is bigger than scheduler capacity, which equals NUM_BLOCKS*BLOCK_SIZE"""
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 2048,
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    TaskPool.reset()

    NUM_BLOCKS = 2
    BLOCK_SIZE = 512
    Backend.cache_managers = [
        {
            "main": PagedKVCacheManager(
                num_blocks=NUM_BLOCKS,
                num_hot_req=2,
                max_seq_len=2048,
                dp_rank=0,
                block_size=BLOCK_SIZE,
            )
        }
    ]  # kv_cache capacity = 1024

    req = UserRequest.create_mock(
        input_len=NUM_BLOCKS * BLOCK_SIZE + 1,
        request_id=f"req_0",
        enable_thinking=False,
    )
    task = Task(f"{req.request_id}", req)
    TaskPool.add(task)

    scheduler = Scheduler(
        100,
        4,
        2,
        "request_preset,prefill_first",
        cache_manager_dict=Backend.cache_managers[0],
        num_scheduler_groups=1,
    )
    scheduler.prepare_for_schedule()
    task_ids = scheduler.schedule()
    # The task should be terminated, so it should be removed from TaskPool
    assert task.task_id not in TaskPool.pool
    assert task_ids == []
    scheduler.cache_manager_dict["main"].finalize_metadata_all_decode(task)


def test_single_decode_prompt_seq_bigger_than_kvcache_capacity():
    """test when single task's prefix length(prompt length + decoded tokens length)
    is bigger than scheduler capacity(NUM_BLOCKS*BLOCK_SIZE).
    """
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 1024000,  # Larger than kv_cache capacity
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    TaskPool.reset()

    NUM_BLOCKS = 2
    BLOCK_SIZE = 512
    DIFF = 5

    Backend.cache_managers = [
        {
            "main": PagedKVCacheManager(
                num_blocks=NUM_BLOCKS,
                num_hot_req=2,
                max_seq_len=1024000,
                dp_rank=0,
                block_size=512,
            )
        }
    ]  # kv_cache capacity = 1024

    req = UserRequest.create_mock(
        input_len=NUM_BLOCKS * BLOCK_SIZE - DIFF,
        request_id=f"req_0",
        enable_thinking=False,
    )
    task = Task(f"{req.request_id}", req)
    TaskPool.add(task)

    scheduler = Scheduler(
        100,
        4,
        2,
        "prefill_first",
        cache_manager_dict=Backend.cache_managers[0],
        num_scheduler_groups=1,
    )
    scheduler.prepare_for_schedule()
    task_ids = scheduler.schedule()
    task.consume_req_tokens()
    task.prefix_tokens.append(1)
    scheduler.update(task_ids)

    for step in range(DIFF):
        scheduler.prepare_for_schedule()
        task_ids = scheduler.schedule()
        assert task_ids == ["req_0"]
        task.prefix_tokens.append(1)
        scheduler.update(task_ids)

    scheduler.prepare_for_schedule()
    task_ids = scheduler.schedule()
    # The task should be terminated when it exceeds capacity
    assert task.task_id not in TaskPool.pool
    assert task_ids == []
    scheduler.cache_manager_dict["main"].finalize_metadata_all_decode(task)


def test_evict_task():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 5123,
                    "cache_type": "paged",
                    "op_impl": "torch",
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    TaskPool.reset()

    NUM_BLOCKS = 4
    BLOCK_SIZE = 512
    DECODE_NUM_TASKS = 4

    Backend.executor = MockExecutor()

    Backend.cache_managers = [
        {
            "main": PagedKVCacheManager(
                num_blocks=NUM_BLOCKS,
                num_hot_req=4,
                max_seq_len=5123,
                dp_rank=0,
                block_size=BLOCK_SIZE,
            )
        }
    ]  # kv_cache capacity = 5120

    main_manager = Backend.cache_managers[0]["main"]
    scheduler = Scheduler(
        100,
        4,
        DECODE_NUM_TASKS,
        "prefill_first,fcfs",
        cache_manager_dict=Backend.cache_managers[0],
        num_scheduler_groups=1,
    )

    Backend.tokenizer = MockTokenizer()
    tasks = []
    task_ids = []
    assert len(TaskPool.pool) == 0

    # add 4 decoding tasks into TaskPool, allocate kv_cache for them according to their prefix length
    for i in range(NUM_BLOCKS):
        req = UserRequest.create_mock(
            input_len=(
                BLOCK_SIZE if i != 3 else BLOCK_SIZE * 2
            ),  # prompt_lens: [BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, 2*BLOCK_SIZE]
            request_id=f"req_{i}",
            enable_thinking=False,
        )
        task = Task(f"{req.request_id}", req)
        tasks.append(task)
        task_ids.append(task.task_id)
        TaskPool.add(task)  # pool: ['req_0', 'req_1', 'req_2', 'req_3']

        task.dp_rank = 0
        main_manager.num_cached_blocks(task) == 0
        task.set_prefill_chunk_size_for_one_step(BLOCK_SIZE)
        scheduler._prepare_prefill_metadata(task, cached_len=0)
        task.consume_req_tokens()
        if i != 3:
            task.prefix_tokens.append(1)

    # TaskPool: ['req_0':Decode, 'req_1':Decode, 'req_2':Decode, 'req_3':Prefill]
    # num_free_blocks: 0
    # evict low priority tasks('req_2':Decode, 'req_3':Prefill) when cache manager has no more blocks for decoding
    req_2_prefix_tokens = tasks[-2].prefix_tokens
    req_3_prefix_tokens = tasks[-1].prefix_tokens

    assert (
        scheduler.kvcache_block_threshold
        == Backend.cache_managers[0]["main"].num_blocks
    )
    scheduler.prepare_for_schedule()
    task_ids = scheduler.schedule()
    scheduler.update(task_ids)
    assert (
        scheduler.kvcache_block_threshold
        == (Backend.cache_managers[0]["main"].num_blocks // 2) // 2
    )
    assert tasks[-1].task_type == TaskType.Prefill
    assert tasks[-2].task_type == TaskType.Prefill

    # TaskPool: ['req_0', 'req_1', 'req_2':Prefill, 'req_3':Prefill]
    # num_free_blocks: 0
    # evicted tasks will not be rescheduled in the short term due to the congestion control
    scheduler.prepare_for_schedule()
    task_ids = scheduler.schedule()
    assert task_ids == [
        "req_0",
        "req_1",
    ]  # req_2 or req_3 will not be rescheduled before other decoding tasks release kv cache blocks

    # two tasks finishes decoding
    for i in range(2):
        task = TaskPool.pool[f"req_{i}"]
        task.next_token = 2
        task.num_new_tokens = 1
        task.set_stopped()
    task_ids = [task.task_id for task in tasks]
    removed_task_ids = scheduler.update(task_ids)
    assert len(removed_task_ids) == 2
    assert len(TaskPool.pool) == 2
    assert (
        scheduler.kvcache_block_threshold
        == Backend.cache_managers[0]["main"].num_blocks
    )

    # TaskPool: ['req_2':Prefill, 'req_3':Prefill]
    # num_free_blocks: 4
    scheduler.prepare_for_schedule()
    task_ids = scheduler.schedule()
    assert len(task_ids) == 2
    assert TaskPool.pool[task_ids[-2]].prefix_tokens == req_2_prefix_tokens
    assert TaskPool.pool[task_ids[-1]].prefix_tokens == req_3_prefix_tokens


def test_scheduler_group():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 1024,
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "pp_size": 2,
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    TaskPool.reset()

    Backend.cache_managers = [
        {
            "main": PagedKVCacheManager(
                num_blocks=100,
                num_hot_req=100,
                max_seq_len=1024,
                dp_rank=0,
                block_size=512,
            )
        }
    ]
    Backend.tokenizer = MockTokenizer()

    scheduler = Scheduler(
        100,
        4,
        4,
        "prefill_first",
        cache_manager_dict=Backend.cache_managers[0],
        num_scheduler_groups=2,
    )

    tasks = []
    for i in range(9):
        req = UserRequest.create_mock(
            input_len=10, request_id=f"req_{i}", enable_thinking=False
        )
        task = Task(f"{req.request_id}", req)
        tasks.append(task)

    # 让task_2, task_5为decode状态
    for task in [tasks[2], tasks[5]]:
        task.dp_rank = 0
        scheduler.cache_manager_dict["main"].num_cached_blocks(task) == 0
        task.set_prefill_chunk_size_for_one_step(task.prefix_tokens_len)
        scheduler._prepare_prefill_metadata(task, cached_len=0)
        task.consume_req_tokens()

    # Add 4 tasks
    TaskPool.add(tasks[7])
    TaskPool.add(tasks[2])
    TaskPool.add(tasks[1])
    TaskPool.add(tasks[5])

    # TaskPool: ['req_7', 'req_2', 'req_1', 'req_5']
    # sgroup head at: 0, empty sgroup: [0, 1]
    scheduler.prepare_for_schedule()
    batch1_ids = scheduler.schedule()
    assert batch1_ids == [
        "req_7",
        "req_1",
    ]  # scheduler prefill tasks, num_tasks <= prefill_mbs == 4

    # 在下一次调度前，executor.step中会更新task.consumed_req_tokens, 此处需模拟executor中的更新, cache_managers和scheduler依赖此值
    for task_id in batch1_ids:
        TaskPool.pool[task_id].consume_req_tokens()

    # TaskPool: {'req_7':waiting, 'req_2', 'req_1':waiting, 'req_5'}
    # sgroup head at: 1, empty sgroup: [0]
    scheduler.prepare_for_schedule()
    batch2_ids = scheduler.schedule()
    assert batch2_ids == [
        "req_2",
        "req_5",
    ]  # scheduler decode tasks, num_tasks <=decode_mbs == 4

    # 在下一次调度前，executor.step中会更新task.consumed_req_tokens, 此处需模拟executor中的更新,  cache_managers和scheduler依赖此值
    for task_id in batch2_ids:
        TaskPool.pool[task_id].consume_req_tokens()

    # Add another 5 decode tasks
    for i in [3, 8, 0, 4, 6]:
        TaskPool.add(tasks[i])
        tasks[i].dp_rank = 0
        assert scheduler.cache_manager_dict["main"].num_cached_blocks(tasks[i]) == 0
        tasks[i].set_prefill_chunk_size_for_one_step(tasks[i].prefix_tokens_len)
        scheduler._prepare_prefill_metadata(tasks[i], cached_len=0)
        tasks[i].consume_req_tokens()
        assert tasks[i].task_type == TaskType.Decode

    # sgroup head at: 0, empty sgroup: []
    scheduler.update(batch1_ids)
    assert len(scheduler.sgroup_list._sgroup_list[1]) == 0

    # TaskPool: ['req_7', 'req_2':waiting, 'req_1', 'req_5':waiting, 'req_3', 'req_8', 'req_0', 'req_4', 'req_6']
    # sgroup head at: 0, empty sgroup: [1]
    scheduler.prepare_for_schedule()
    batch3_ids = scheduler.schedule()
    assert batch3_ids == ["req_7", "req_1", "req_3", "req_8"]

    # Set all tasks in scheduler_group_1 are unwait, release scheduler_group_1
    for task_id in batch2_ids:
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id].set_stopped()
    # sgroup head at: 1, empty sgroup: []
    scheduler.update(batch2_ids)

    # TaskPool: TaskPool: ['req_0', 'req_4', 'req_6']
    # sgroup head at: 1, empty sgroup: [0]
    # sgroup_0 release earlier than sgroup_1, so it will be scheduled earlier than sgroup_1
    scheduler.prepare_for_schedule()
    batch4_ids = scheduler.schedule()
    assert batch4_ids == ["req_0", "req_4", "req_6"]
    # remove tasks in scheduler_group_0, release scheduler_group_0
    for task_id in batch3_ids:
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id].set_stopped()
    # sgroup head at: 0, empty sgroup: []
    scheduler.update(batch3_ids)

    for task_id in batch4_ids:
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch4_ids)

    # TaskPool: []
    scheduler.prepare_for_schedule()
    empty_ids = scheduler.schedule()
    assert len(empty_ids) == 0


def test_slot_group_skew():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 1024,
                    "max_batch_size": 8,
                    "op_impl": "torch",
                    "cache_type": "skew",
                    "pp_size": 2,
                    "dp_size": 1,
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    infer_args = get_global_args().infer
    set_slot_handle(
        infer_args.max_batch_size,
        infer_args.pp_size,
    )

    TaskPool.reset()
    Backend.cache_managers = None
    Backend.tokenizer = MockTokenizer()

    tasks = []
    for i in range(9):
        req = UserRequest.create_mock(
            input_len=10, request_id=f"req_{i}", enable_thinking=False
        )
        task = Task(f"{req.request_id}", req)
        tasks.append(task)

    # Set 3 decode tasks: 2,5,6
    tasks[2].consume_req_tokens()  # Decode
    tasks[5].consume_req_tokens()
    tasks[6].consume_req_tokens()

    # Add 4 tasks
    TaskPool.add(tasks[7])
    TaskPool.add(tasks[2])
    TaskPool.add(tasks[1])
    TaskPool.add(tasks[5])

    # skewScheduler's prefill_mbs == decode_mbs == 4
    scheduler = SkewScheduler(
        infer_args.max_batch_size,
        cache_manager_dict=None,
        original_scheduler_type="request_preset,prefill_first",
        prefill_chunk_size=None,
    )

    # TaskPool: ['req_7', 'req_2', 'req_1', 'req_5']
    # slot_group: [[], []], sgroup head at: 0
    scheduler.prepare_for_schedule()
    batch1_ids = scheduler.schedule()
    assert batch1_ids == [
        "req_7",
        "req_1",
    ]  # scheduler prefill tasks, num_tasks <= prefill_mbs == 4

    # TaskPool: {'req_7':waiting, 'req_2', 'req_1':waiting, 'req_5'}
    # slot_group: [[], ['req_7', 'req_1']], sgroup head at: 1
    scheduler.prepare_for_schedule()
    batch2_ids = scheduler.schedule()
    assert batch2_ids == [
        "req_2",
        "req_5",
    ]  # scheduler decode tasks, num_tasks <=decode_mbs == 4

    # Add another 5 tasks
    TaskPool.add(tasks[3])
    TaskPool.add(tasks[8])
    TaskPool.add(tasks[0])
    TaskPool.add(tasks[4])
    TaskPool.add(tasks[6])

    # Set all tasks in scheduler_group_0 are unwait, release scheduler_group_0
    # slot_group: [['req_2', 'req_5'], ['req_7', 'req_1']], sgroup head at: 0
    scheduler.update(batch1_ids)

    # TaskPool: ['req_7', 'req_2':waiting, 'req_1', 'req_5':waiting, 'req_3', 'req_8', 'req_0', 'req_4', 'req_6']
    # slot_group: [['req_2', 'req_5'], []], sgroup head at: 0
    scheduler.prepare_for_schedule()
    batch3_ids = scheduler.schedule()
    assert batch3_ids == ["req_7", "req_1", "req_3", "req_8"]

    # Set all tasks in scheduler_group_1 are unwait, release scheduler_group_1
    for task_id in batch2_ids:
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id].set_stopped()
    # slot_group: [['req_2', 'req_5'], ['req_7', 'req_1', 'req_3', 'req_8']], sgroup head at: 1
    scheduler.update(batch2_ids)

    # TaskPool: TaskPool: ['req_0', 'req_4', 'req_6']
    # slot_group: [[], ['req_7', 'req_1', 'req_3', 'req_8']], sgroup head at: 1
    # sgroup_0 release earlier than sgroup_1, so it will be scheduled earlier than sgroup_1
    scheduler.prepare_for_schedule()
    batch4_ids = scheduler.schedule()
    assert batch4_ids == ["req_0", "req_4"]

    # remove tasks in scheduler_group_0, release scheduler_group_0
    for task_id in batch3_ids:
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id].set_stopped()
    # slot_group: [['req_0', 'req_4'], ['req_7', 'req_1', 'req_3', 'req_8']], sgroup head at: 0
    scheduler.update(batch3_ids)

    # TaskPool: ['req_0':waiting, 'req_4':waiting, 'req_6']
    # slot_group: [['req_0', 'req_4'], []], sgroup head at: 0
    scheduler.prepare_for_schedule()
    batch5_ids = scheduler.schedule()
    assert batch5_ids == ["req_6"]

    for task_id in batch4_ids + batch5_ids:
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id].set_stopped()
    scheduler.update(batch4_ids)
    scheduler.update(batch5_ids)

    # TaskPool: []
    scheduler.prepare_for_schedule()
    empty_ids = scheduler.schedule()
    assert len(empty_ids) == 0


def test_pp_chunked_prefill():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 1024,
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    TaskPool.reset()
    Backend.cache_managers = [
        {
            "main": PagedKVCacheManager(
                num_blocks=10000,
                num_hot_req=10000,
                max_seq_len=1024,
                dp_rank=0,
                block_size=5120,
            )
        }
    ]

    Backend.executor = MockExecutor()

    for i in range(5):
        req = UserRequest.create_mock(
            input_len=192, request_id=f"req_{i}", enable_thinking=False
        )
        task = Task(f"{req.request_id}", req)
        TaskPool.add(task)

    for i in range(5):
        req = UserRequest.create_mock(
            input_len=96, request_id=f"req_{i + 5}", enable_thinking=False
        )
        task = Task(f"{req.request_id}", req)
        TaskPool.add(task)

    scheduler = Scheduler(
        100,
        12,
        12,
        "prefill_first",
        cache_manager_dict=Backend.cache_managers[0],
        num_scheduler_groups=3,
        prefill_chunk_size=200,
    )

    # group 1: [0, 1], [1, 6]
    # group 2: [2, 3], [3, 7]
    # group 3: [4, 5], [5, 8, 9]
    expected_batch_ids_list = [
        ["req_0", "req_1"],
        ["req_2", "req_3"],
        ["req_4", "req_5"],
        ["req_1", "req_6"],
        ["req_3", "req_7"],
        ["req_5", "req_8", "req_9"],
    ]

    for i in range(len(expected_batch_ids_list)):
        scheduler.prepare_for_schedule()
        batch_ids = scheduler.schedule()
        assert batch_ids == expected_batch_ids_list[i]
        for task_id in batch_ids:
            TaskPool.pool[task_id].consume_req_tokens()

        scheduler.update(batch_ids)


def test_prepare_prefill_metadata_multi_cache_managers():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 2048,
                    "cache_type": "paged",
                    "op_impl": "torch",
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )

    main = PagedKVCacheManager(
        num_hot_req=8,
        max_seq_len=2048,
        dp_rank=0,
        block_size=128,
        num_blocks=16,
        enable_prefix_caching=True,
    )
    indexer = PagedKVCacheManager(
        num_hot_req=8,
        max_seq_len=2048,
        dp_rank=0,
        block_size=64,
        num_blocks=32,
        enable_prefix_caching=True,
    )
    scheduler = Scheduler(
        100,
        12,
        12,
        "prefill_first",
        cache_manager_dict={"main": main, "indexer": indexer},
        num_scheduler_groups=1,
        prefill_chunk_size=20000,
    )

    task0 = Task("req_prefill_0", UserRequest.create_mock(512, "req_prefill_0"))

    for manager in (main, indexer):
        assert manager.num_cached_blocks(task0) == 0
    task0.set_prefill_chunk_size_for_one_step(128)

    scheduler._prepare_prefill_metadata(task0, cached_len=0)
    task0.consume_req_tokens()

    assert len(task0.new_cache_ids["main"]) == 1
    assert len(task0.new_cache_ids["indexer"]) == 2
    assert task0.inc_hit_tokens == 0
    assert task0.consumed_req_tokens == 128
    assert task0.kv_cache_len_used_in_completed_steps == 128

    # 测试task1的prompt被main和indexer manager全部击中
    task1 = Task("req_prefill_1", UserRequest.create_mock(128, "req_prefill_1"))
    assert main.num_cached_blocks(task1) == 1
    assert indexer.num_cached_blocks(task1) == 2

    num_cached_tokens = scheduler._num_prefill_cached_tokens(task1)
    assert num_cached_tokens == task1.prefix_tokens_len
    task1.set_prefill_chunk_size_for_one_step(1)
    scheduler._prepare_prefill_metadata(task1, cached_len=num_cached_tokens)

    assert len(task1.new_cache_ids["main"]) == 1
    assert len(task1.new_cache_ids["indexer"]) == 2
    assert task1.inc_hit_tokens == 127
    assert task1.consumed_req_tokens == 127
    assert task1.prefill_chunk_size == 1

    # 测试test1的prepare_decode_metadata过程
    task1.consume_req_tokens()
    task1.prefix_tokens.append(1)
    assert task1.task_type == TaskType.Decode
    assert task1.prefix_tokens_len == 129
    assert task1.kv_cache_len_used_in_completed_steps == 128

    assert main.task_to_cache_ids[task1.task_id] == {0}
    assert indexer.task_to_cache_ids[task1.task_id] == {0, 1}
    scheduler._prepare_decode_metadata(task1)
    assert len(task1.new_cache_ids["main"]) == 1
    assert len(task1.new_cache_ids["indexer"]) == 1


def test_check_prefill_capacity_requires_all_cache_managers():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 2048,
                    "cache_type": "paged",
                    "op_impl": "torch",
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    task = Task("req_capacity", UserRequest.create_mock(1024, "req_capacity"))
    task.set_prefill_chunk_size_for_one_step(256)

    # main manager has enough blocks, indexer manager does not.
    main = PagedKVCacheManager(
        num_hot_req=4,
        max_seq_len=2048,
        dp_rank=0,
        block_size=256,
        num_blocks=8,
        enable_prefix_caching=False,
    )
    indexer = PagedKVCacheManager(
        num_hot_req=4,
        max_seq_len=2048,
        dp_rank=0,
        block_size=128,
        num_blocks=10,
        enable_prefix_caching=False,
    )
    main.active_blocks = {i: object() for i in range(6)}
    indexer.active_blocks = {i: object() for i in range(9)}
    scheduler = Scheduler(
        100,
        12,
        12,
        "prefill_first",
        cache_manager_dict={"main": main, "indexer": indexer},
        num_scheduler_groups=1,
        prefill_chunk_size=20000,
    )
    scheduler.kvcache_block_threshold = 8
    from chitu.scheduler import KVCacheCapacityStatus

    assert (
        scheduler._check_prefill_capacity(task, cached_len=512)
        is KVCacheCapacityStatus.CONGESTED
    )


def test_check_prefill_capacity_exceeds_physical_blocks():
    from chitu.scheduler import KVCacheCapacityStatus

    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 2048,
                    "cache_type": "paged",
                    "op_impl": "torch",
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    task = Task("req_exceeds", UserRequest.create_mock(4096, "req_exceeds"))
    task.set_prefill_chunk_size_for_one_step(4096)
    main = PagedKVCacheManager(
        num_hot_req=4,
        max_seq_len=2048,
        dp_rank=0,
        block_size=256,
        num_blocks=4,
        enable_prefix_caching=False,
    )
    scheduler = Scheduler(
        100,
        12,
        12,
        "prefill_first",
        cache_manager_dict={"main": main},
        num_scheduler_groups=1,
        prefill_chunk_size=20000,
    )
    scheduler.kvcache_block_threshold = 4
    assert (
        scheduler._check_prefill_capacity(task, cached_len=0)
        is KVCacheCapacityStatus.EXCEEDS_CAPACITY
    )


def test_check_decode_capacity_requires_all_cache_managers():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 2048,
                    "cache_type": "paged",
                    "op_impl": "torch",
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    task = Task(
        "req_decode_capacity", UserRequest.create_mock(400, "req_decode_capacity")
    )
    task.task_type = TaskType.Decode

    main = PagedKVCacheManager(
        num_blocks=10,
        num_hot_req=8,
        max_seq_len=2048,
        dp_rank=0,
        block_size=100,
        enable_prefix_caching=False,
    )
    main.active_blocks = {i: object() for i in range(5)}
    main.task_to_cache_ids[task.task_id] = {0, 1}
    indexer = PagedKVCacheManager(
        num_blocks=8,
        num_hot_req=8,
        max_seq_len=2048,
        dp_rank=0,
        block_size=128,
        enable_prefix_caching=False,
    )
    indexer.active_blocks = {i: object() for i in range(7)}
    indexer.task_to_cache_ids[task.task_id] = {0, 1}
    scheduler = Scheduler(
        100,
        12,
        12,
        "prefill_first",
        cache_manager_dict={"main": main, "indexer": indexer},
        num_scheduler_groups=1,
        prefill_chunk_size=20000,
    )

    from chitu.scheduler import KVCacheCapacityStatus

    assert scheduler._check_decode_capacity(task) is KVCacheCapacityStatus.CONGESTED


def _make_partial_prefill_task(
    scheduler: Scheduler,
    *,
    request_id: str,
    prompt_len: int,
    consumed: int,
) -> Task:
    """构造正在prefill的任务: 已consumed了部分prompt(consumed_req_tokens=consumed)、持有KV块、
    状态为AvailableForSchedule的Prefill任务。
    """
    req = UserRequest.create_mock(
        input_len=prompt_len,
        request_id=request_id,
    )
    task = Task(request_id, req)
    TaskPool.add(task)
    task.dp_rank = 0
    # Drive a single prefill chunk to allocate KV blocks for `consumed` tokens.
    task.set_prefill_chunk_size_for_one_step(consumed)
    scheduler._prepare_prefill_metadata(task, cached_len=0)
    task.consume_req_tokens()  # consumed_req_tokens = consumed, prefill_chunk_size=None
    assert task.task_type == TaskType.Prefill
    assert task.consumed_req_tokens == consumed
    return task


def test_schedule_prefill_tasks_eviction_breaks_deadlock():
    """模拟死锁场景：
    cache_manager容量=2*block，chunk=1*block，两个长为2-block的prompt各完成1个block长度处于
    在途返回状态，互相占块导致谁都长不到 prompt 末尾。
    期望:
    调度器逐出低优先级在途任务，高优先级任务在本轮就被调度出去，且阈值未被折半（让位式逐出不施加
    拥塞控制，避免临时拥塞变永久死锁）。
    """
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 4096,
                    "cache_type": "paged",
                    "op_impl": "torch",
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    TaskPool.reset()

    NUM_BLOCKS = 2
    BLOCK_SIZE = 1024
    Backend.executor = MockExecutor()
    Backend.tokenizer = MockTokenizer()
    Backend.cache_managers = [
        {
            "main": PagedKVCacheManager(
                num_blocks=NUM_BLOCKS,
                num_hot_req=4,
                max_seq_len=4096,
                dp_rank=0,
                block_size=BLOCK_SIZE,
                enable_prefix_caching=False,
            )
        }
    ]
    main = Backend.cache_managers[0]["main"]

    scheduler = Scheduler(
        max_running_tasks=4,
        prefill_num_tasks=4,
        decode_num_tasks=4,
        scheduler_type="prefill_first,fcfs",
        cache_manager_dict=Backend.cache_managers[0],
        num_scheduler_groups=1,
        prefill_chunk_size=BLOCK_SIZE,
    )
    initial_threshold = scheduler.kvcache_block_threshold

    # task1 arrives first, task2 after, both are 2-block prompts.
    # Both have already prefilled 1 block.
    task1 = _make_partial_prefill_task(
        scheduler, request_id="req_high", prompt_len=2 * BLOCK_SIZE, consumed=BLOCK_SIZE
    )
    task2 = _make_partial_prefill_task(
        scheduler, request_id="req_low", prompt_len=2 * BLOCK_SIZE, consumed=BLOCK_SIZE
    )
    # Both tasks now hold 1 KV block each; manager is full.
    assert len(main.task_to_cache_ids[task1.task_id]) == 1
    assert len(main.task_to_cache_ids[task2.task_id]) == 1
    assert main.num_active_blocks == NUM_BLOCKS

    scheduler.prepare_for_schedule()
    sched_ids = scheduler.schedule()

    # The high-priority task should be scheduled this very round (no empty schedule).
    assert sched_ids == [task1.task_id]
    # The low-priority in-flight task was evicted: rolled back to fresh prefill state,
    # its KV block freed.
    assert task2.task_id in TaskPool.pool  # not removed, just evicted
    assert task2.task_type == TaskType.Prefill
    assert task2.consumed_req_tokens == 0
    assert task2.task_id not in main.task_to_cache_ids
    # kvcache_block_threshold must not be halved — otherwise the freed capacity would be re-blocked and the task we
    # tried to unblock would still be congested next round.
    assert scheduler.kvcache_block_threshold == initial_threshold

    TaskPool.reset()


def test_prefill_capacity_reserves_for_inflight_prefill():
    """准入预留：当存在持有块的在途prefill任务时，新prefill任务的容量检查需为它们
    完成所需的容量预留，否则会被准入并触发互相占块的死锁。
    """
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 4096,
                    "cache_type": "paged",
                    "op_impl": "torch",
                    "schedule_overlap": True,
                    "mtp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    TaskPool.reset()

    NUM_BLOCKS = 2
    BLOCK_SIZE = 1024
    Backend.executor = MockExecutor()
    Backend.tokenizer = MockTokenizer()
    Backend.cache_managers = [
        {
            "main": PagedKVCacheManager(
                num_blocks=NUM_BLOCKS,
                num_hot_req=4,
                max_seq_len=4096,
                dp_rank=0,
                block_size=BLOCK_SIZE,
                enable_prefix_caching=False,
            )
        }
    ]
    main = Backend.cache_managers[0]["main"]

    scheduler = Scheduler(
        max_running_tasks=4,
        prefill_num_tasks=4,
        decode_num_tasks=4,
        scheduler_type="prefill_first,fcfs",
        cache_manager_dict=Backend.cache_managers[0],
        num_scheduler_groups=1,
        prefill_chunk_size=BLOCK_SIZE,
    )

    # An in-flight prefill holding 1 block, still needing 1 more block to finish.
    inflight = _make_partial_prefill_task(
        scheduler,
        request_id="req_inflight",
        prompt_len=2 * BLOCK_SIZE,
        consumed=BLOCK_SIZE,
    )

    # A new prefill candidate; manager has 1 free block, which would naively look
    # like enough room for 1 chunk — but reserving for the in-flight task should make
    # the new task be marked CONGESTED.
    new_task = Task("req_new", UserRequest.create_mock(2 * BLOCK_SIZE, "req_new"))
    new_task.set_prefill_chunk_size_for_one_step(BLOCK_SIZE)

    from chitu.scheduler import KVCacheCapacityStatus

    # is_new_task为True时，表示新prefill任务，需要考虑为正在prefill的任务预留容量，防止进入自锁状态
    assert (
        scheduler._check_prefill_capacity(new_task, cached_len=0, is_new_task=True)
        is KVCacheCapacityStatus.CONGESTED
    )

    # is_new_task为False时，表示非新prefill任务，容量足够（不扣除为正在prefill任务预留的容量）
    assert (
        scheduler._check_prefill_capacity(new_task, cached_len=0, is_new_task=False)
        is KVCacheCapacityStatus.OK
    )
    assert (
        scheduler._check_prefill_capacity(
            inflight, cached_len=BLOCK_SIZE, is_new_task=False
        )
        is KVCacheCapacityStatus.OK
    )

    TaskPool.reset()
