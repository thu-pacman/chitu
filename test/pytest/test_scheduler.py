from typing_extensions import override
from omegaconf import OmegaConf

from chitu.task import Task, TaskPool, MockFixedLengthedUserRequest
from chitu.scheduler import Scheduler, SkewScheduler
from chitu.global_vars import set_global_args, _set_slot_handle, get_global_args
from chitu.backend import Backend
import pytest
from chitu.task_type import TaskType, TaskDecodeType
from chitu.utils import ceil_div


class MockExecutor:
    def __init__(self):
        pass

    def step(self, tasks):
        task_ids = [
            task_id
            for task_id in tasks.task_ids
            if task_id in Backend.cache_manager.block_table
        ]
        Backend.cache_manager.finalize_cache_all_decode(task_ids)


class MockCacheManager:
    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.block_table = {}

    def get_block_size(self):
        """Return the number of tokens that a block can accommodate"""
        return self.block_size

    def get_num_blocks(self):
        """Return number of total blocks"""
        return self.num_blocks

    @property
    def num_free_blocks(self):
        """Return number of free blocks"""
        return self.num_blocks - self.num_used_blocks

    @property
    def num_used_blocks(self):
        """Renturn number of blocks that has reserved for reqs to use."""
        return sum(self.block_table.values())

    def prepare_cache_prefill(self, task_ids):
        for task_id in task_ids:
            prefill_len = TaskPool.pool[task_id].prefix_tokens_len
            needed_blocks = (prefill_len + self.block_size - 1) // self.block_size
            if task_id not in self.block_table:
                self.block_table[task_id] = 0
            self.block_table[task_id] += needed_blocks

    def num_additional_blocks_req_need(self, task_id, target_seq_len):
        if task_id in self.block_table:
            return max(
                0, ceil_div(target_seq_len, self.block_size) - self.block_table[task_id]
            )
        return max(0, ceil_div(target_seq_len, self.block_size))

    def prepare_cache_decode(self, task_ids):
        for task_id in task_ids:
            prefix_len = TaskPool.pool[task_id].prefix_tokens_len
            needed_blocks = self.num_additional_blocks_req_need(task_id, prefix_len)
            self.block_table[task_id] += needed_blocks

    def finalize_cache_all_decode(self, task_ids):
        for task_id in task_ids:
            self.block_table.pop(task_id)


class MockTokenizer:
    def __init__(self):
        self.stop_tokens = [2]


def test_chunked_prefill():
    set_global_args(
        OmegaConf.create(
            {"infer": {"max_seq_len": 32768, "op_impl": "torch", "cache_type": "paged"}}
        ),
        need_ensure=False,
    )
    TaskPool.reset()
    Backend.cache_manager = MockCacheManager(num_blocks=10000, block_size=5120)
    Backend.executor = MockExecutor()

    for i in range(4):
        req = MockFixedLengthedUserRequest(
            input_len=1000 * (i + 1), request_id=f"req_{i}", enable_reasoning=False
        )
        task = Task(f"{req.request_id}", req)
        TaskPool.add(task)

    scheduler = Scheduler(4, 4, "prefill_first", 1, prefill_chunk_size=4096)

    # Prefill:

    # Remaining: [1000, 2000, 3000, 4000]

    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_0", "req_1", "req_2"])
    for task_id in batch1_ids:
        TaskPool.pool[task_id].consume_req_tokens()

    scheduler.update(batch1_ids)
    # Remaining: [0, 0, 1904, 4000]

    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_2", "req_3"])
    for task_id in batch2_ids:
        TaskPool.pool[task_id].consume_req_tokens()

    scheduler.update(batch2_ids)
    # Remaining: [0, 0, 0, 1808]

    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(["req_3"])
    for task_id in batch3_ids:
        TaskPool.pool[task_id].consume_req_tokens()

    scheduler.update(batch3_ids)
    # Remaining: [0, 0, 0, 0]

    # Decode:

    batch4_ids = scheduler.schedule()
    assert sorted(batch4_ids) == sorted(["req_0", "req_1", "req_2", "req_3"])


def test_chunked_prefill_skew():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 32768,
                    "max_reqs": 256,
                    "op_impl": "torch",
                    "cache_type": "skew",
                    "pp_size": 2,
                    "dp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )

    infer_args = get_global_args().infer
    _set_slot_handle(
        infer_args.max_reqs,
        infer_args.pp_size,
        infer_args.dp_size,
        infer_args.cache_type,
    )

    TaskPool.reset()
    Backend.cache_manager = MockCacheManager(num_blocks=10000, block_size=512)

    for i in range(4):
        req = MockFixedLengthedUserRequest(
            input_len=1000 * (i + 1), request_id=f"req_{i}", enable_reasoning=False
        )
        task = Task(f"{req.request_id}", req)
        TaskPool.add(task)

    scheduler = SkewScheduler(
        infer_args.max_reqs, "prefill_first", prefill_chunk_size=4096
    )

    # Prefill:

    # Slot groups: [[],[]], free_sgroup: [0, 1]
    # Task length remaining: [1000, 2000, 3000, 4000]
    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_0", "req_1", "req_2"])
    for task_id in batch1_ids:
        TaskPool.pool[task_id].consume_req_tokens()
    scheduler.update(batch1_ids)

    # Slot groups: [["req_0", "req_1", "req_2", 'req_3'],[]], free_sgroup: [1, 0]
    # Task length remaining:: [0, 0, 1904, 4000]
    empty_ids = scheduler.schedule()
    assert empty_ids == []  # skewScheduler doesn't change task's slot group
    scheduler.update([])

    # Slot groups: [["req_0", "req_1", "req_2", 'req_3'],[]], free_sgroup: [0, 1]
    # Task length remaining: [0, 0, 1904, 4000]
    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_2", "req_3"])
    for task_id in batch2_ids:
        TaskPool.pool[task_id].consume_req_tokens()
    scheduler.update(batch2_ids)

    # Slot groups: [["req_0", "req_1", "req_2", 'req_3'],[]], free_sgroup: [1, 0]
    # Task length remaining: [0, 0, 0, 1808]
    empty_ids = scheduler.schedule()
    assert empty_ids == []
    scheduler.update([])

    # Slot groups: [["req_0", "req_1", "req_2", 'req_3'],[]], free_sgroup: [0, 1]
    # Task length remaining: [0, 0, 0, 1808]
    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(["req_3"])
    for task_id in batch3_ids:
        TaskPool.pool[task_id].consume_req_tokens()
    scheduler.update(batch3_ids)

    # Slot groups: [["req_0", "req_1", "req_2", 'req_3'],[]], free_sgroup: [1, 0]
    # Remaining: [0, 0, 0, 0]
    empty_ids = scheduler.schedule()
    assert empty_ids == []
    scheduler.update([])

    # Decode:
    # Slot groups: [["req_0", "req_1", "req_2", 'req_3'],[]], free_sgroup: [1, 0]
    # Remaining: [0, 0, 0, 0]
    batch4_ids = scheduler.schedule()
    assert sorted(batch4_ids) == sorted(["req_0", "req_1", "req_2", "req_3"])


def test_priority_prefill_first():
    set_global_args(
        OmegaConf.create(
            {"infer": {"max_seq_len": 1024, "op_impl": "torch", "cache_type": "paged"}}
        ),
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

    scheduler = Scheduler(4, 2, "prefill_first", 1)

    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_7", "req_1", "req_3", "req_8"])
    scheduler.update(batch1_ids)
    for task_id in batch1_ids:
        TaskPool.remove(task_id)

    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_0", "req_4"])
    scheduler.update(batch2_ids)
    for task_id in batch2_ids:
        TaskPool.remove(task_id)

    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(["req_2", "req_5"])
    scheduler.update(batch3_ids)
    for task_id in batch3_ids:
        TaskPool.remove(task_id)

    batch4_ids = scheduler.schedule()
    assert sorted(batch4_ids) == sorted(["req_6"])
    scheduler.update(batch4_ids)
    for task_id in batch4_ids:
        TaskPool.remove(task_id)

    batch5_ids = scheduler.schedule()
    assert len(batch5_ids) == 0


def test_priority_prefill_first_skew():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 1024,
                    "max_reqs": 4,
                    "op_impl": "torch",
                    "cache_type": "skew",
                    "pp_size": 1,
                    "dp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )

    infer_args = get_global_args().infer
    _set_slot_handle(
        infer_args.max_reqs,
        infer_args.pp_size,
        infer_args.dp_size,
        infer_args.cache_type,
    )

    TaskPool.reset()
    Backend.cache_manager = MockCacheManager(num_blocks=10, block_size=512)
    Backend.tokenizer = MockTokenizer()

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

    scheduler = SkewScheduler(
        infer_args.max_reqs, "prefill_first", prefill_chunk_size=None
    )

    # slot_groups: [[]], free_sgroups: [0]
    # TaskPool: ['req_7', 'req_2', 'req_1', 'req_5', 'req_3', 'req_8', 'req_0', 'req_4', 'req_6']
    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_7", "req_1", "req_3", "req_8"])
    for task_id in batch1_ids:
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id]._decode_status = TaskDecodeType.Stopped
    scheduler.update(batch1_ids)

    # slot_groups: [[]], free_sgroup: deque([0])
    # TaskPool: ['req_2', 'req_5', 'req_0', 'req_4', 'req_6']
    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_0", "req_4"])
    for task_id in batch2_ids:
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id]._decode_status = TaskDecodeType.Stopped
    scheduler.update(batch2_ids)

    # slot_group: [[]], free_sgroups: deque([0])
    # TaskPool: ['req_2', 'req_5', 'req_6']
    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(
        ["req_2", "req_5", "req_6"]
    )  # skewScheduler's decode_mbs == prefill_mbs == 4
    for task_id in batch3_ids:
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id]._decode_status = TaskDecodeType.Stopped
    scheduler.update(batch3_ids)

    # slot_group: [[]], free_sgroups: deque([0])
    # TaskPool: []
    batch4_ids = scheduler.schedule()
    assert len(batch4_ids) == 0


def test_priority_fcfs():
    set_global_args(
        OmegaConf.create(
            {"infer": {"max_seq_len": 1024, "op_impl": "torch", "cache_type": "paged"}}
        ),
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

    scheduler = Scheduler(4, 4, "fcfs", 1)

    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_0", "req_1", "req_2", "req_3"])
    scheduler.update(batch1_ids)
    for task_id in batch1_ids:
        TaskPool.remove(task_id)

    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_4", "req_5"])
    scheduler.update(batch2_ids)
    for task_id in batch2_ids:
        TaskPool.remove(task_id)

    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(["req_6", "req_7", "req_8"])
    scheduler.update(batch3_ids)
    for task_id in batch3_ids:
        TaskPool.remove(task_id)

    batch4_ids = scheduler.schedule()
    assert len(batch4_ids) == 0


def test_priority_fcfs_skew():

    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 1024,
                    "max_reqs": 4,
                    "op_impl": "torch",
                    "cache_type": "skew",
                    "pp_size": 1,
                    "dp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    infer_args = get_global_args().infer
    _set_slot_handle(
        infer_args.max_reqs,
        infer_args.pp_size,
        infer_args.dp_size,
        infer_args.cache_type,
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

    # scheduler = Scheduler(4, 4, "fcfs", 1)
    scheduler = SkewScheduler(infer_args.max_reqs, "fcfs", prefill_chunk_size=None)

    # slot_group: [[]], free_sgroup: deque([0]), slot_capacity: 4
    # TaskPool: ['req_7', 'req_2', 'req_1', 'req_5', 'req_3', 'req_8', 'req_0', 'req_4', 'req_6']
    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_0", "req_1", "req_2", "req_3"])
    for task_id in batch1_ids:
        # Make task need_move
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id]._decode_status = TaskDecodeType.Stopped
    scheduler.update(batch1_ids)

    # slot_group: [[]], free_sgroup: deque([0])
    # TaskPool: ['req_7', 'req_5', 'req_8', 'req_4', 'req_6']
    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_4", "req_5"])
    for task_id in batch2_ids:
        # Make task need_move
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id]._decode_status = TaskDecodeType.Stopped
    scheduler.update(batch2_ids)

    # slot_group: [[]], free_sgroup: deque([0])
    # TaskPool: ['req_7', 'req_8', 'req_6']
    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(["req_6", "req_7", "req_8"])

    # slot_group: [["req_6", "req_7", "req_8"]], free_sgroup: deque([])
    # TaskPool: ['req_7', 'req_8', 'req_6']
    batch4_ids = scheduler.schedule()
    assert len(batch4_ids) == 0  # No avalible slot group, return empty task_ids
    for task_id in batch3_ids:
        # Make task need_move
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id]._decode_status = TaskDecodeType.Stopped
    scheduler.update(batch3_ids)

    # slot_group: [[]], free_sgroup: deque([0])
    # TaskPool: []
    batch4_ids = scheduler.schedule()
    assert len(batch4_ids) == 0


def test_priority_request_preset_over_prefill_first():
    set_global_args(
        OmegaConf.create(
            {"infer": {"max_seq_len": 1024, "op_impl": "torch", "cache_type": "paged"}}
        ),
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

    scheduler = Scheduler(4, 2, "request_preset,prefill_first", 1)

    # ['req_7', 'req_2':Decode, 'req_1', 'req_5':Decode, 'req_3', 'req_8', 'req_0', 'req_4', 'req_6':Decode]
    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_7", "req_3", "req_0", "req_4"])
    scheduler.update(batch1_ids)
    for task_id in batch1_ids:
        TaskPool.remove(task_id)

    # TaskPool: ['req_2':Decode, 'req_1', 'req_5':Decode, 'req_8', 'req_6':Decode]
    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_2", "req_6"])
    scheduler.update(batch2_ids)
    for task_id in batch2_ids:
        TaskPool.remove(task_id)

    # TaskPool: ['req_1', 'req_5':Decode, 'req_8']
    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(["req_1", "req_8"])
    scheduler.update(batch3_ids)
    for task_id in batch3_ids:
        TaskPool.remove(task_id)

    # TaskPool: ['req_5':Decode]
    batch4_ids = scheduler.schedule()
    assert sorted(batch4_ids) == sorted(["req_5"])
    scheduler.update(batch4_ids)
    for task_id in batch4_ids:
        TaskPool.remove(task_id)

    batch5_ids = scheduler.schedule()
    assert len(batch5_ids) == 0


def test_priority_request_preset_over_prefill_first_skew():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 1024,
                    "max_reqs": 4,
                    "op_impl": "torch",
                    "cache_type": "skew",
                    "pp_size": 1,
                    "dp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    infer_args = get_global_args().infer
    _set_slot_handle(
        infer_args.max_reqs,
        infer_args.pp_size,
        infer_args.dp_size,
        infer_args.cache_type,
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

    # scheduler = Scheduler(4, 2, "request_preset,prefill_first", 1) x
    # skewScheduler's prefill_mbs == decode_mbs == 4
    scheduler = SkewScheduler(
        infer_args.max_reqs, "request_preset,prefill_first", prefill_chunk_size=None
    )

    # TaskPool: ['req_7', 'req_2', 'req_1', 'req_5', 'req_3', 'req_8', 'req_0', 'req_4', 'req_6']
    # slot_group: [[]], free_sgroup: deque([0])
    batch1_ids = scheduler.schedule()
    assert sorted(batch1_ids) == sorted(["req_7", "req_3", "req_0", "req_4"])
    for task_id in batch1_ids:
        # Make task need_move
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id]._decode_status = TaskDecodeType.Stopped
    scheduler.update(batch1_ids)

    # TaskPool: ['req_2', 'req_1', 'req_5', 'req_8', 'req_6']
    # slot_group: [[]], free_sgroup: deque([0])
    batch2_ids = scheduler.schedule()
    assert sorted(batch2_ids) == sorted(["req_2", "req_6", "req_5"])
    for task_id in batch2_ids:
        # Make task need_move
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id]._decode_status = TaskDecodeType.Stopped
    scheduler.update(batch2_ids)

    # TaskPool: ['req_1', 'req_8']
    # slot_group: [[]], free_sgroup: deque([0])
    batch3_ids = scheduler.schedule()
    assert sorted(batch3_ids) == sorted(["req_1", "req_8"])
    for task_id in batch3_ids:
        # Make task need_move
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id]._decode_status = TaskDecodeType.Stopped
    scheduler.update(batch3_ids)

    # TaskPool: []
    # slot_group: [[]], free_sgroup: deque([0])
    batch4_ids = scheduler.schedule()
    assert len(batch4_ids) == 0


def test_single_prompt_seq_bigger_than_scheduler_capacity():
    """test when single prompt length is bigger than scheduler capacity, which equals NUM_BLOCKS*BLOCK_SIZE"""
    set_global_args(
        OmegaConf.create(
            {"infer": {"max_seq_len": 2048, "op_impl": "torch", "cache_type": "paged"}}
        ),
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

    scheduler = Scheduler(4, 2, "request_preset,prefill_first", 1)
    with pytest.raises(Exception) as exc_info:
        scheduler.schedule()
    assert "KV_cache capacity is insufficient to support prefilling" in str(exc_info)

    TaskPool.remove(task.task_id)


def test_single_decode_prompt_seq_bigger_than_scheduler_capacity():
    """test when single task's prefix length(prompt length + decoded tokens length)
    is bigger than scheduler capacity(NUM_BLOCKS*BLOCK_SIZE).
    """
    set_global_args(
        OmegaConf.create(
            {"infer": {"max_seq_len": 1024, "op_impl": "torch", "cache_type": "paged"}}
        ),
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

    scheduler = Scheduler(4, 2, "prefill_first", 1)
    task_ids = scheduler.schedule()
    Backend.cache_manager.prepare_cache_prefill(task_ids)
    task._prefix_tokens.append(1)
    scheduler.update(task_ids)

    task.consume_req_tokens()
    for step in range(DIFF):
        task_ids = scheduler.schedule()
        assert task_ids == [
            "req_0",
        ]
        Backend.cache_manager.prepare_cache_decode(task_ids)
        task._prefix_tokens.append(1)
        scheduler.update(task_ids)

    with pytest.raises(Exception) as exc_info:
        scheduler.schedule()
    assert "KV_cache capacity is insufficient to support decoding completion" in str(
        exc_info
    )
    TaskPool.remove(task.task_id)


def test_evict_decode_task():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 5123,
                    "cache_type": "paged",
                    "op_impl": "torch",
                    "cache_type": "paged",
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
    Backend.cache_manager = MockCacheManager(
        num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE
    )  # kv_cache capacity = 5120
    Backend.tokenizer = MockTokenizer()
    tasks = []
    task_ids = []
    assert len(TaskPool.pool) == 0

    # add 4 decoding tasks into TaskPool, allocate kv_cache for them according to their prefix length
    for i in range(NUM_BLOCKS):
        req = MockFixedLengthedUserRequest(
            input_len=BLOCK_SIZE,
            request_id=f"req_{i}",
            enable_reasoning=False,
        )
        task = Task(f"{req.request_id}", req)
        tasks.append(task)
        task_ids.append(task.task_id)
        TaskPool.add(task)  # pool: ['req_0', 'req_1', 'req_2', 'req_3']
        task.consume_req_tokens()
        task._prefix_tokens.append(1)
    Backend.cache_manager.prepare_cache_prefill(task_ids)

    # TaskPool: ['req_0', 'req_1', 'req_2', 'req_3']
    # num_free_blocks: 0
    # evict low priority tasks('req_2', 'req_3') when cache manager has no more blocks for decoding
    req_2_prefix_tokens = tasks[-2].prefix_tokens
    req_3_prefix_tokens = tasks[-1].prefix_tokens
    scheduler = Scheduler(4, DECODE_NUM_TASKS, "prefill_first,fcfs", 1)
    assert scheduler.kvcache_block_threshold == Backend.cache_manager.get_num_blocks()
    task_ids = scheduler.schedule()
    Backend.cache_manager.prepare_cache_decode(task_ids)
    scheduler.update(task_ids)
    assert (
        scheduler.kvcache_block_threshold
        == (Backend.cache_manager.get_num_blocks() // 2) // 2
    )
    assert tasks[-1].task_type == TaskType.Prefill
    assert tasks[-2].task_type == TaskType.Prefill

    # TaskPool: ['req_0', 'req_1', 'req_2':Prefill, 'req_3':Prefill]
    # num_free_blocks: 0
    # evicted tasks will not be rescheduled in the short term due to the congestion control
    task_ids = scheduler.schedule()
    assert task_ids == [
        "req_0",
        "req_1",
    ]  # req_2 or req_3 will not be rescheduled before other decoding tasks release kv cache blocks

    # after two tasks finished decoding, reschedule req_8,req_9 / or one task finished decoding ,reschedule req_8
    for i in range(2):
        task = TaskPool.pool[f"req_{i}"]
        task.next_token = 2
        task.num_new_tokens = 1
        task._decode_status = TaskDecodeType.Stopped
    task_ids = [task.task_id for task in tasks]
    removed_task_ids = scheduler.update(task_ids)
    Backend.cache_manager.finalize_cache_all_decode(removed_task_ids)
    assert len(removed_task_ids) == 2
    assert len(TaskPool.pool) == 2
    assert scheduler.kvcache_block_threshold == Backend.cache_manager.get_num_blocks()

    # TaskPool: ['req_2':Prefill, 'req_3':Prefill]
    # num_free_blocks: 4
    task_ids = scheduler.schedule()
    assert len(task_ids) == 2
    assert TaskPool.pool[task_ids[-1]].prefix_tokens == req_2_prefix_tokens
    assert TaskPool.pool[task_ids[-2]].prefix_tokens == req_3_prefix_tokens


def test_scheduler_group():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 1024,
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "pp_size": 2,
                }
            }
        ),
        need_ensure=False,
    )
    TaskPool.reset()
    Backend.cache_manager = MockCacheManager(num_blocks=10, block_size=512)
    Backend.tokenizer = MockTokenizer()

    tasks = []
    for i in range(9):
        req = MockFixedLengthedUserRequest(
            input_len=10, request_id=f"req_{i}", enable_reasoning=False
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

    scheduler = Scheduler(4, 2, "prefill_first", num_scheduler_groups=2)

    # TaskPool: ['req_7', 'req_2', 'req_1', 'req_5']
    # free_sgroups: deque([0, 1]), used_sgroups: set()
    batch1_ids = scheduler.schedule()
    assert batch1_ids == [
        "req_7",
        "req_1",
    ]  # scheduler prefill tasks, num_tasks <= prefill_mbs == 4
    for task_id in batch1_ids:
        TaskPool.pool[task_id].wait(None)

    # TaskPool: {'req_7':waiting, 'req_2', 'req_1':waiting, 'req_5'}
    # free_sgroups: deque([1]), used_sgroups: {0}
    batch2_ids = scheduler.schedule()
    assert batch2_ids == [
        "req_2",
        "req_5",
    ]  # scheduler decode tasks, num_tasks <=decode_mbs == 4
    for task_id in batch2_ids:
        TaskPool.pool[task_id].wait(None)

    # Add another 5 tasks
    TaskPool.add(tasks[3])
    TaskPool.add(tasks[8])
    TaskPool.add(tasks[0])
    TaskPool.add(tasks[4])
    TaskPool.add(tasks[6])

    # TaskPool: ['req_7':waiting, 'req_2':waiting, 'req_1':waiting, 'req_5':waiting, 'req_3', 'req_8', 'req_0', 'req_4', 'req_6']
    # free_sgroups: deque([]), used_sgroups: {0, 1}
    empty_ids = scheduler.schedule()
    assert len(empty_ids) == 0  # No avilable scheduler groups, empty task_ids

    # Set all tasks in scheduler_group_0 are unwait, release scheduler_group_0
    for task_id in batch1_ids:
        TaskPool.pool[task_id].unwait()
    scheduler.update(batch1_ids)

    # TaskPool: ['req_7', 'req_2':waiting, 'req_1', 'req_5':waiting, 'req_3', 'req_8', 'req_0', 'req_4', 'req_6']
    # free_sgroups: deque([0]), used_sgroups: {1}
    batch3_ids = scheduler.schedule()
    assert batch3_ids == ["req_7", "req_1", "req_3", "req_8"]

    # remove tasks in scheduler_group_0, release scheduler_group_0
    for task_id in batch3_ids:
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id]._decode_status = TaskDecodeType.Stopped
    scheduler.update(batch3_ids)

    # Set all tasks in scheduler_group_1 are unwait, release scheduler_group_1
    for task_id in batch2_ids:
        TaskPool.pool[task_id].unwait()
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id]._decode_status = TaskDecodeType.Stopped
    scheduler.update(batch2_ids)

    # TaskPool: TaskPool: ['req_0', 'req_4', 'req_6']
    # free_sgroups: deque([1, 0]), used_sgroups: set()
    # sgroup_0 release earlier than sgroup_1, so it will be scheduled earlier than sgroup_1
    batch4_ids = scheduler.schedule()
    assert batch4_ids == ["req_0", "req_4"]
    for task_id in batch4_ids:
        TaskPool.pool[task_id].wait(None)

    # TaskPool: ['req_0':waiting, 'req_4':waiting, 'req_6']
    # free_sgroups: deque([1]), used_sgroups: {0}
    batch5_ids = scheduler.schedule()
    assert batch5_ids == ["req_6"]

    for task_id in batch5_ids:
        TaskPool.pool[task_id].wait(None)
    for task_id in batch4_ids + batch5_ids:
        TaskPool.pool[task_id].unwait()
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id]._decode_status = TaskDecodeType.Stopped
    scheduler.update(batch5_ids, batch4_ids)

    # TaskPool: []
    empty_ids = scheduler.schedule()
    assert len(empty_ids) == 0
    assert len(scheduler.free_sgroups) == 2
    assert len(scheduler.used_sgroups) == 0


def test_slot_group_skew():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 1024,
                    "max_reqs": 8,
                    "op_impl": "torch",
                    "cache_type": "skew",
                    "pp_size": 2,
                    "dp_size": 1,
                }
            }
        ),
        need_ensure=False,
    )
    infer_args = get_global_args().infer
    _set_slot_handle(
        infer_args.max_reqs,
        infer_args.pp_size,
        infer_args.dp_size,
        infer_args.cache_type,
    )

    TaskPool.reset()
    Backend.cache_manager = MockCacheManager(num_blocks=8, block_size=512)
    Backend.tokenizer = MockTokenizer()

    tasks = []
    for i in range(9):
        req = MockFixedLengthedUserRequest(
            input_len=10, request_id=f"req_{i}", enable_reasoning=False
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
        infer_args.max_reqs, "request_preset,prefill_first", prefill_chunk_size=None
    )

    # TaskPool: ['req_7', 'req_2', 'req_1', 'req_5']
    # slot_group: [[], []], free_sgroup: deque([0, 1])
    batch1_ids = scheduler.schedule()
    assert batch1_ids == [
        "req_7",
        "req_1",
    ]  # scheduler prefill tasks, num_tasks <= prefill_mbs == 4
    for task_id in batch1_ids:
        TaskPool.pool[task_id].wait(None)

    # TaskPool: {'req_7':waiting, 'req_2', 'req_1':waiting, 'req_5'}
    # slot_group: [['req_7', 'req_1'], []], free_sgroup: deque([1])
    batch2_ids = scheduler.schedule()
    assert batch2_ids == [
        "req_2",
        "req_5",
    ]  # scheduler decode tasks, num_tasks <=decode_mbs == 4
    for task_id in batch2_ids:
        TaskPool.pool[task_id].wait(None)

    # Add another 5 tasks
    TaskPool.add(tasks[3])
    TaskPool.add(tasks[8])
    TaskPool.add(tasks[0])
    TaskPool.add(tasks[4])
    TaskPool.add(tasks[6])

    # TaskPool: ['req_7':waiting, 'req_2':waiting, 'req_1':waiting, 'req_5':waiting, 'req_3', 'req_8', 'req_0', 'req_4', 'req_6']
    # slot_group: [['req_7', 'req_1'], ['req_2', 'req_5']], free_sgroups: deque([])
    empty_ids = scheduler.schedule()
    assert len(empty_ids) == 0  # No avilable scheduler groups, empty task_ids
    assert len(scheduler.free_sgroups) == 0

    # Set all tasks in scheduler_group_0 are unwait, release scheduler_group_0
    for task_id in batch1_ids:
        TaskPool.pool[task_id].unwait()
    scheduler.update(batch1_ids)

    # TaskPool: ['req_7', 'req_2':waiting, 'req_1', 'req_5':waiting, 'req_3', 'req_8', 'req_0', 'req_4', 'req_6']
    # slot_group: [['req_7', 'req_1'], ['req_2', 'req_5']], free_sgroup: deque([0])
    batch3_ids = scheduler.schedule()
    assert batch3_ids == ["req_7", "req_1", "req_3", "req_8"]

    # remove tasks in scheduler_group_0, release scheduler_group_0
    for task_id in batch3_ids:
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id]._decode_status = TaskDecodeType.Stopped
    scheduler.update(batch3_ids)

    # Set all tasks in scheduler_group_1 are unwait, release scheduler_group_1
    for task_id in batch2_ids:
        TaskPool.pool[task_id].unwait()
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id]._decode_status = TaskDecodeType.Stopped
    scheduler.update(batch2_ids)

    # TaskPool: TaskPool: ['req_0', 'req_4', 'req_6']
    # slot_group: [[], []], free_sgroup: deque([0, 1])
    # sgroup_0 release earlier than sgroup_1, so it will be scheduled earlier than sgroup_1
    batch4_ids = scheduler.schedule()
    assert batch4_ids == ["req_0", "req_4"]
    for task_id in batch4_ids:
        TaskPool.pool[task_id].wait(None)

    # TaskPool: ['req_0':waiting, 'req_4':waiting, 'req_6']
    # slot_group: [['req_0', 'req_4'], []], free_sgroup: deque([1])
    batch5_ids = scheduler.schedule()
    assert batch5_ids == ["req_6"]

    for task_id in batch5_ids:
        TaskPool.pool[task_id].wait(None)
    for task_id in batch4_ids + batch5_ids:
        TaskPool.pool[task_id].unwait()
        TaskPool.pool[task_id].num_new_tokens = 1025
        TaskPool.pool[task_id].next_token = 2
        TaskPool.pool[task_id].task_type = TaskType.Decode
        TaskPool.pool[task_id]._decode_status = TaskDecodeType.Stopped
    scheduler.update(batch5_ids, batch4_ids)

    # TaskPool: []
    empty_ids = scheduler.schedule()
    assert len(empty_ids) == 0
    assert len(scheduler.free_sgroups) == 2
    assert len(scheduler.used_sgroups) == 0
