from typing_extensions import override
from omegaconf import OmegaConf

from chitu.task import Task, TaskPool, MockFixedLengthedUserRequest
from chitu.scheduler import Scheduler
from chitu.global_vars import set_global_args, get_global_args
from chitu.backend import Backend
import pytest
from chitu.task_type import TaskType


class MockCacheManager:
    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._num_used_blocks = 0

    def get_block_size(self):
        """Return the number of tokens that a block can accommodate"""
        return self.block_size

    def get_num_blocks(self):
        """Return number of total blocks"""
        return self.num_blocks

    @property
    def num_free_blocks(self):
        """Return number of free blocks"""
        return self.num_blocks - self._num_used_blocks

    @property
    def num_used_blocks(self):
        """Renturn number of blocks that has reserved for reqs to use."""
        return self._num_used_blocks

    def prepare_cache_prefill(self, task_id):
        prefill_len = TaskPool.pool[task_id].prefix_tokens_len
        needed_blocks = (prefill_len + self.block_size - 1) // self.block_size
        self._num_used_blocks += needed_blocks

    def req_needs_new_block(self, task_id):
        prefix_len_old = TaskPool.pool[task_id].prefix_tokens_len
        if prefix_len_old % self.block_size == 0:
            return True
        return False

    def prepare_cache_decode(self, task_id):
        if self.req_needs_new_block(task_id):
            self._num_used_blocks += 1

    def finalize_cache_all_decode(self, task_id):
        prefix_len = TaskPool.pool[task_id].prefix_tokens_len
        release_blocks = (prefix_len + self.block_size - 1) // self.block_size
        self._num_used_blocks -= release_blocks


class MockTokenizer:
    def __init__(self):
        self.stop_tokens = [2]


def test_chunked_prefill():
    set_global_args(
        OmegaConf.create({"infer": {"max_seq_len": 32768, "op_impl": "torch"}}),
        need_ensure=False,
    )
    TaskPool.reset()
    Backend.cache_manager = MockCacheManager(num_blocks=10000, block_size=512)

    for i in range(4):
        req = MockFixedLengthedUserRequest(
            input_len=1000 * (i + 1), request_id=f"req_{i}", enable_reasoning=False
        )
        task = Task(f"{req.request_id}", req)
        TaskPool.add(task)

    scheduler = Scheduler(4, 4, "prefill_first", prefill_chunk_size=4096)

    # Prefill:

    # Remaining: [1000, 2000, 3000, 4000]

    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_0", "req_1", "req_2"])
    for task_id in batch1_ids:
        TaskPool.pool[task_id].consume_req_tokens()

    # Remaining: [0, 0, 1904, 4000]

    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_2", "req_3"])
    for task_id in batch2_ids:
        TaskPool.pool[task_id].consume_req_tokens()

    # Remaining: [0, 0, 0, 1808]

    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(["req_3"])
    for task_id in batch3_ids:
        TaskPool.pool[task_id].consume_req_tokens()

    # Remaining: [0, 0, 0, 0]

    # Decode:

    batch4_ids = scheduler.schedule()
    assert sorted(batch4_ids) == sorted(["req_0", "req_1", "req_2", "req_3"])


def test_priority_prefill_first():
    set_global_args(
        OmegaConf.create({"infer": {"max_seq_len": 1024, "op_impl": "torch"}}),
        need_ensure=False,
    )
    TaskPool.reset()
    Backend.cache_manager = MockCacheManager(num_blocks=10, block_size=512)

    tasks = []
    for i in range(9):
        req = MockFixedLengthedUserRequest(
            input_len=10, request_id=f"req_{i}", enable_reasoning=False
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


def test_priority_fcfs():
    set_global_args(
        OmegaConf.create({"infer": {"max_seq_len": 1024, "op_impl": "torch"}}),
        need_ensure=False,
    )
    TaskPool.reset()
    Backend.cache_manager = MockCacheManager(num_blocks=10, block_size=512)

    tasks = []
    for i in range(9):
        req = MockFixedLengthedUserRequest(
            input_len=10, request_id=f"req_{i}", enable_reasoning=False
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


def test_priority_request_preset_over_prefill_first():
    set_global_args(
        OmegaConf.create({"infer": {"max_seq_len": 1024, "op_impl": "torch"}}),
        need_ensure=False,
    )
    TaskPool.reset()
    Backend.cache_manager = MockCacheManager(num_blocks=10, block_size=512)

    tasks = []
    for i in range(9):
        req = MockFixedLengthedUserRequest(
            input_len=10, request_id=f"req_{i}", enable_reasoning=False
        )
        task = Task(f"{req.request_id}", req, priority=2 if i in [0, 3, 4, 6] else 1)
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


def test_single_prompt_seq_bigger_than_scheduler_capacity():
    """test when single prompt length is bigger than scheduler capacity, which equals NUM_BLOCKS*BLOCK_SIZE"""
    set_global_args(
        OmegaConf.create({"infer": {"max_seq_len": 2048, "op_impl": "torch"}}),
        need_ensure=False,
    )
    TaskPool.reset()

    NUM_BLOCKS = 2
    BLOCK_SIZE = 512
    Backend.cache_manager = MockCacheManager(
        num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE
    )  # kv_cache capacity = 1024

    req = MockFixedLengthedUserRequest(
        input_len=NUM_BLOCKS * BLOCK_SIZE + 1,
        request_id=f"req_0",
        enable_reasoning=False,
    )
    task = Task(f"{req.request_id}", req)
    TaskPool.add(task)

    scheduler = Scheduler(4, 2, "request_preset,prefill_first")
    with pytest.raises(Exception) as exc_info:
        scheduler.schedule()
    assert "KV_cache capacity is insufficient to support prefilling" in str(exc_info)

    TaskPool.remove(task.task_id)


def test_single_decode_prompt_seq_bigger_than_scheduler_capacity():
    """test when single task's prefix length(prompt length + decoded tokens length)
    is bigger than scheduler capacity(NUM_BLOCKS*BLOCK_SIZE).
    """
    set_global_args(
        OmegaConf.create({"infer": {"max_seq_len": 1024, "op_impl": "torch"}}),
        need_ensure=False,
    )
    TaskPool.reset()

    NUM_BLOCKS = 2
    BLOCK_SIZE = 512
    DIFF = 5
    Backend.cache_manager = MockCacheManager(
        num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE
    )  # kv_cache capacity = 1024
    req = MockFixedLengthedUserRequest(
        input_len=NUM_BLOCKS * BLOCK_SIZE - DIFF,
        request_id=f"req_0",
        enable_reasoning=False,
    )
    task = Task(f"{req.request_id}", req)
    TaskPool.add(task)

    scheduler = Scheduler(4, 2, "prefill_first")
    task_ids = scheduler.schedule()
    Backend.cache_manager.prepare_cache_prefill(task_ids[0])
    task._prefix_tokens.append(1)

    task.consume_req_tokens()
    for step in range(DIFF - 1):
        task_ids = scheduler.schedule()
        assert task_ids == [
            "req_0",
        ]
        Backend.cache_manager.prepare_cache_decode(task_ids[0])
        task._prefix_tokens.append(1)

    with pytest.raises(Exception) as exc_info:
        scheduler.schedule()
    assert "KV_cache capacity is insufficient to support decoding completion" in str(
        exc_info
    )
    TaskPool.remove(task.task_id)


def test_evict_decode_task():
    set_global_args(
        OmegaConf.create(
            {"infer": {"max_seq_len": 5123, "cache_type": "paged", "op_impl": "torch"}}
        ),
        need_ensure=False,
    )
    TaskPool.reset()

    NUM_BLOCKS = 10
    BLOCK_SIZE = 512
    DECODE_NUM_TASKS = 4

    Backend.cache_manager = MockCacheManager(
        num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE
    )  # kv_cache capacity = 5120
    Backend.tokenizer = MockTokenizer()
    tasks = []
    assert len(TaskPool.pool) == 0

    # add 10 decoding tasks into TaskPool, allocate kv_cache for them according to their prefix length
    for i in range(NUM_BLOCKS):
        req = MockFixedLengthedUserRequest(
            input_len=BLOCK_SIZE - 1 if i >= 2 else BLOCK_SIZE,
            request_id=f"req_{i}",
            enable_reasoning=False,
        )
        task = Task(f"{req.request_id}", req)
        tasks.append(task)
        TaskPool.add(task)  # pool: [req_0,req_1,...,req_9]
        Backend.cache_manager.prepare_cache_prefill(task.task_id)
        task.consume_req_tokens()

    # evict low priority tasks(req_8,req_9) when cache manager has no more blocks for decoding
    req_8_prefix_tokens = tasks[-2].prefix_tokens
    req_9_prefix_tokens = tasks[-1].prefix_tokens
    scheduler = Scheduler(4, DECODE_NUM_TASKS, "prefill_first,fcfs")
    assert scheduler.kvcache_block_threshold == Backend.cache_manager.get_num_blocks()
    task_ids = scheduler.schedule()
    assert (
        scheduler.kvcache_block_threshold
        == (Backend.cache_manager.get_num_blocks() // 2) // 2
    )
    assert tasks[-1].task_type == TaskType.Prefill
    assert tasks[-2].task_type == TaskType.Prefill

    # evicted tasks will not be rescheduled in the short term due to the congestion control
    task_ids = scheduler.schedule()
    assert task_ids == [
        "req_0",
        "req_1",
        "req_2",
        "req_3",
    ]  # req_8 or req_9 will not be rescheduled before other decoding tasks release kv cache blocks

    # after two tasks finished decoding, reschedule req_8,req_9 / or one task finished decoding ,reschedule req_8
    for i in range(3):
        task = TaskPool.pool[f"req_{i}"]
        task.next_token = 2
        task.num_new_tokens = 1
        Backend.cache_manager.finalize_cache_all_decode(f"req_{i}")
    task_ids = [task.task_id for task in tasks]
    removed_task_ids = scheduler.update(task_ids)
    assert len(removed_task_ids) == 3
    assert len(TaskPool.pool) == 7
    assert scheduler.kvcache_block_threshold == Backend.cache_manager.get_num_blocks()

    task_ids = scheduler.schedule()
    assert len(task_ids) == 2
    assert TaskPool.pool[task_ids[-1]].prefix_tokens == req_9_prefix_tokens
    assert TaskPool.pool[task_ids[-2]].prefix_tokens == req_8_prefix_tokens
