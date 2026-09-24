import pytest
from chitu.kv_cache import (
    DeepSeekV4CompressedKVCacheManager,
    DeepSeekV4SlidingKVCacheManager,
    PagedKVCacheManager,
    SingletonPagedKVCacheManager,
    TokenBlock,
    BlockIdentityChainBuilder,
    BlockRuntime,
    NONE_BLK_HASH,
)
from weakref import WeakValueDictionary
from collections import deque
from chitu.task import UserRequest, Task
from chitu.task_type import TaskType
from omegaconf import OmegaConf
from chitu.global_vars import set_global_args
from chitu.backend import Backend
from chitu.utils import ceil_div
from chitu.scheduler import Scheduler, KVCacheCapacityStatus


@pytest.fixture(autouse=True)
def setup_global_args():
    """Set up global arguments for tests."""
    BlockIdentityChainBuilder.clear_registry()
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 2048,
                    "max_batch_size": 10,
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "schedule_overlap": True,
                    "prefill_chunk_size": None,
                    "mtp_size": 1,
                    "dp_size": 1,
                },
                "multi_inst": {
                    "n_insts": 2,
                    "router": {},
                },
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )


class TestPagedKVCacheManager:
    """测试PagedKVCacheManager类"""

    def test_initialization(self):
        """测试类初始化"""
        cache_manager = PagedKVCacheManager(
            num_blocks=100,
            num_hot_req=100,
            max_seq_len=2048,
            dp_rank=0,
            block_size=512,
            enable_prefix_caching=False,
        )
        assert cache_manager.num_blocks == 100
        assert cache_manager.block_size == 512
        assert cache_manager.max_num_blocks == 400
        assert cache_manager.dp_rank == 0
        assert len(cache_manager.free_cache_ids) == 100
        assert len(cache_manager.active_blocks) == 0
        assert len(cache_manager.cached_idle_blocks) == 0
        assert cache_manager.enable_prefix_caching is False

    def test_prefix_cache_probe_does_not_register_unscheduled_task(self):
        cache_manager = PagedKVCacheManager(
            num_blocks=100,
            num_hot_req=100,
            max_seq_len=2048,
            dp_rank=0,
            block_size=512,
            enable_prefix_caching=True,
        )
        req = UserRequest.create_mock(
            input_len=600, request_id="unscheduled", enable_thinking=False
        )
        task = Task(task_id=req.request_id, req=req)

        assert task.task_id not in cache_manager.task_to_cache_ids
        assert cache_manager.num_cached_blocks(task) == 0
        assert task.task_id not in cache_manager.task_to_cache_ids
        assert cache_manager.num_cached_idle_blocks(task) == 0
        assert task.task_id not in cache_manager.task_to_cache_ids

    def test_deepseek_v4_sliding_manager_uses_window_length(self):
        manager = DeepSeekV4SlidingKVCacheManager(
            num_blocks=4,
            num_hot_req=2,
            max_seq_len=4096,
            window_size=384,
            dp_rank=0,
            block_size=384,
            enable_prefix_caching=False,
        )

        assert manager.max_blocks_per_req == 1
        assert manager.max_num_blocks == 2
        assert manager.num_blocks == 2
        assert manager.num_blocks_for_seq_len(0) == 0
        assert manager.num_blocks_for_seq_len(1) == 1
        assert manager.num_blocks_for_seq_len(128) == 1
        assert manager.num_blocks_for_seq_len(129) == 1
        assert manager.num_blocks_for_seq_len(384) == 1
        assert manager.num_blocks_for_seq_len(4096) == 1

        manager.realloc(100)
        assert manager.num_blocks == 2

    def test_deepseek_v4_sliding_manager_allocates_one_metadata_block(self):
        manager = DeepSeekV4SlidingKVCacheManager(
            num_blocks=2,
            num_hot_req=2,
            max_seq_len=4096,
            window_size=384,
            dp_rank=0,
            block_size=384,
            enable_prefix_caching=False,
        )
        req = UserRequest.create_mock(
            input_len=4096, request_id="req", enable_thinking=False
        )
        task = Task(task_id=req.request_id, req=req)

        new_cache_ids = manager.prepare_metadata_before_prefill(task)

        assert len(new_cache_ids) == 1
        assert len(manager.task_to_cache_ids[task.task_id]) == 1
        assert len(manager.task_to_token_blocks[task.task_id]) == 1

    def test_deepseek_v4_compressed_manager_uses_compressed_length(self):
        manager = DeepSeekV4CompressedKVCacheManager(
            num_blocks=8,
            num_hot_req=2,
            max_seq_len=4096,
            compress_ratio=4,
            dp_rank=0,
            block_size=128,
            enable_prefix_caching=False,
        )

        assert manager.max_blocks_per_req == 8
        assert manager.max_num_blocks == 16
        assert manager.num_blocks_for_seq_len(0) == 0
        assert manager.num_blocks_for_seq_len(3) == 0
        assert manager.num_blocks_for_seq_len(4) == 1
        assert manager.num_blocks_for_seq_len(512) == 1
        assert manager.num_blocks_for_seq_len(516) == 2
        assert manager.num_blocks_for_seq_len(4096) == 8

        manager.realloc(100)
        assert manager.num_blocks == 16

    def test_deepseek_v4_compressed_manager_allocates_compressed_metadata_blocks(self):
        manager = DeepSeekV4CompressedKVCacheManager(
            num_blocks=8,
            num_hot_req=2,
            max_seq_len=4096,
            compress_ratio=4,
            dp_rank=0,
            block_size=128,
            enable_prefix_caching=False,
        )
        req = UserRequest.create_mock(
            input_len=516, request_id="req", enable_thinking=False
        )
        task = Task(task_id=req.request_id, req=req)

        new_cache_ids = manager.prepare_metadata_before_prefill(task)

        assert len(new_cache_ids) == 2
        assert len(manager.task_to_cache_ids[task.task_id]) == 2
        assert len(manager.task_to_token_blocks[task.task_id]) == 2
        assert len(manager.task_to_token_blocks[task.task_id]) < ceil_div(516, 128)

    def test_realloc(self):
        """测试重新分配block数量"""
        cache_manager = PagedKVCacheManager(
            num_blocks=100,
            num_hot_req=100,
            max_seq_len=2048,
            dp_rank=0,
            block_size=512,
            enable_prefix_caching=False,
        )
        cache_manager.realloc(50)

        assert cache_manager.num_blocks == 50
        assert len(cache_manager.free_cache_ids) == 50

        # 测试最大分配的num_blocks不超过cache_manager.max_num_blocks (200)
        cache_manager.realloc(500)
        assert cache_manager.num_blocks == 400
        assert len(cache_manager.free_cache_ids) == 400

    def test_get_free_cache_idx(self):
        """测试get_free_cache_idx函数"""
        cache_manager = PagedKVCacheManager(
            num_blocks=100,
            num_hot_req=100,
            max_seq_len=2048,
            dp_rank=0,
            block_size=512,
            enable_prefix_caching=False,
        )

        # 从free_cache_ids中分配
        idx = cache_manager.get_free_cache_idx()

        assert isinstance(idx, int)
        assert 0 <= idx < 100
        assert idx not in cache_manager.free_cache_ids

        # 当free_cache_ids和cached_idle_blocks耗光时，raise No more blocks
        cache_manager.free_cache_ids = deque()
        with pytest.raises(Exception, match="No more free KVCache blocks"):
            cache_manager.get_free_cache_idx()

        # 当free_cache_ids耗光，cached_idle_blocks还未耗尽，从cached_idle_blocks中分配cache_idx
        cache_manager.free_cache_ids = deque()
        runtime = BlockRuntime(cache_idx=5, active_cnt=0)
        cache_manager.cached_idle_blocks[5] = runtime

        idx = cache_manager.get_free_cache_idx()
        assert idx == 5
        assert 5 not in cache_manager.cached_idle_blocks
        assert runtime.cache_idx is None

    def test_get_free_cache_idx_collects_evicted_blk_hash(self):
        """测试被动逐出时会记录evicted blk hash，并可按上限批量弹出"""
        cache_manager = PagedKVCacheManager(
            num_blocks=100,
            num_hot_req=100,
            max_seq_len=2048,
            dp_rank=0,
            block_size=4,
            enable_prefix_caching=False,
        )

        cache_manager.free_cache_ids = deque()
        identity = cache_manager.identity_builder.make_identity(
            token_chunk=[1, 2, 3, 4],
            pre_blk_hash=NONE_BLK_HASH,
            canonical_prefix_hashes=True,
        )

        block = TokenBlock(identity=identity, runtime=BlockRuntime(cache_idx=7))
        cache_manager.cached_idle_blocks[7] = block
        cache_manager.cache_idx_to_hash[7] = block.blk_hash

        idx = cache_manager.get_free_cache_idx()
        assert idx == 7

        popped = cache_manager.pop_evicted_blk_hashes(max_items=512)
        assert popped == [block.blk_hash]
        assert cache_manager.pop_evicted_blk_hashes(max_items=512) == []

    def test_identity_chain_builder(self):
        """测试BlockIdentityChainBuilder构建逻辑"""
        BLOCK_SIZE = 512

        builder = BlockIdentityChainBuilder.acquire("__test_kv_isolated", BLOCK_SIZE)

        # 测试pre_blk_hash为None
        tokens = list(range(512))
        identity = builder.make_identity(tokens, NONE_BLK_HASH)

        assert list(identity.tokens) == tokens
        assert identity.blk_size == BLOCK_SIZE
        assert identity.pre_blk_hash == NONE_BLK_HASH

        # 测试给定一个pre_hash值
        tokens = list(range(BLOCK_SIZE))
        pre_hash = "some_previous_hash"
        identity = builder.make_identity(tokens, pre_hash)
        assert identity.pre_blk_hash == pre_hash

        # 测试tokens长度超过block_size长度
        tokens = list(range(BLOCK_SIZE + 1))
        with pytest.raises(ValueError):
            builder.make_identity(tokens, NONE_BLK_HASH)

    def test_task_life_cycle_in_cache_manager(self):
        """测试任务在cache_manager中的生命周期，
        涉及函数:prepare_metadata_before_prefill/decode, finalize_metadata_all_decode
        """
        cache_manager = PagedKVCacheManager(
            num_blocks=100,
            num_hot_req=100,
            max_seq_len=2048,
            dp_rank=0,
            block_size=512,
            enable_prefix_caching=False,
        )

        Backend.cache_managers = [{"main": cache_manager}]

        # 在warmup或API server中完成的步骤
        req = UserRequest.create_mock(
            input_len=600, request_id=f"test_task", enable_thinking=False
        )
        task = Task(f"{req.request_id}", req)
        task.task_type == TaskType.Prefill

        # 在scheduler中完成步骤
        task.dp_rank = 0
        task_n_cached_blocks = cache_manager.num_cached_blocks(task)
        assert task_n_cached_blocks == 0, f"{task_n_cached_blocks} vs 0"
        assert task.cached_seq_len == 0, f"{task.cached_seq_len} vs 0"
        task.set_prefill_chunk_size_for_one_step(300)  # 假如prefill_chunk_size为300
        task.new_cache_ids = {
            "main": cache_manager.prepare_metadata_before_prefill(task)
        }

        assert task.consumed_req_tokens == 0
        assert task.new_cache_ids["main"] == [0]
        assert cache_manager.task_to_cache_ids[task.task_id] == {0}
        task_token_blocks = cache_manager.task_to_token_blocks[task.task_id]
        assert len(task_token_blocks) == 1, f"{len(task_token_blocks)} vs 1"
        assert cache_manager.active_blocks[0] is task_token_blocks[0].runtime
        assert task_token_blocks[0].active_cnt == 1

        # 在executor中完成的步骤
        task.consume_req_tokens()
        assert task.consumed_req_tokens == 300
        assert task.task_type == TaskType.Prefill

        # 第二次被prefill调度
        assert cache_manager.num_cached_blocks(task) == 1
        assert task.cached_seq_len == 300
        task.set_prefill_chunk_size_for_one_step(300)  # 假如prefill_chunk_size为300
        task.new_cache_ids = {
            "main": cache_manager.prepare_metadata_before_prefill(task)
        }
        assert task.new_cache_ids["main"] == [1]
        assert cache_manager.task_to_cache_ids[task.task_id] == {0, 1}
        assert len(task_token_blocks) == 2
        assert cache_manager.active_blocks[1] is task_token_blocks[1].runtime
        assert task_token_blocks[1].active_cnt == 1

        # 在executor中完成的步骤
        task.consume_req_tokens()
        task.next_tokens = [1]
        task.prefix_tokens.append(task.next_tokens[0])
        assert task.consumed_req_tokens == 600
        assert task.prefix_tokens_len == 601
        assert task.task_type == TaskType.Decode

        set_global_args(
            OmegaConf.create(
                {
                    "infer": {
                        "max_seq_len": 2048,
                        "max_batch_size": 10,
                        "op_impl": "torch",
                        "cache_type": "paged",
                        "schedule_overlap": True,
                        "prefill_chunk_size": None,
                        "mtp_size": 500,
                        "dp_size": 1,
                    }
                }
            ),
            need_ensure=False,
            need_preprocess=False,
        )  # task计算alloc_seq_len时会用到全局的mtp_size信息
        cache_manager.mtp_size = 500  # 设定mtp_size为500

        # 开始被decode调度
        assert cache_manager.num_cached_blocks(task) == 2
        aviable_blocks = cache_manager.num_blocks - cache_manager.num_active_blocks
        cur_blocks = cache_manager.num_cached_blocks(task)
        target_blocks = ceil_div(
            task.alloc_seq_len,
            cache_manager.block_size,
        )
        assert task.alloc_seq_len == 1100
        assert aviable_blocks == 98
        assert cur_blocks == 2
        assert target_blocks == 3
        task.new_cache_ids = {
            "main": cache_manager.prepare_metadata_before_decode(task)
        }

        assert task.new_cache_ids["main"] == [2]
        assert cache_manager.task_to_cache_ids[task.task_id] == {0, 1, 2}
        assert cache_manager.active_blocks[2] is task_token_blocks[2].runtime
        assert task_token_blocks[1].active_cnt == 1

        # 在executor中执行decode step
        task.next_tokens = [1]
        task.prefix_tokens.extend([1] * 400)  # 假设主模型只接受了400个token
        assert task.prefix_tokens_len == 1001

        # 再次被decode调度
        assert cache_manager.num_cached_blocks(task) == 3
        aviable_blocks = cache_manager.num_blocks - cache_manager.num_active_blocks
        cur_blocks = cache_manager.num_cached_blocks(task)
        target_blocks = ceil_div(
            task.alloc_seq_len,
            cache_manager.block_size,
        )
        assert task.alloc_seq_len == 1500
        assert aviable_blocks == 97
        assert cur_blocks == 3
        assert target_blocks == 3
        task.new_cache_ids = {
            "main": cache_manager.prepare_metadata_before_decode(task)
        }
        assert task.new_cache_ids["main"] == []
        assert cache_manager.task_to_cache_ids[task.task_id] == {0, 1, 2}


@pytest.fixture
def cache_manager_with_prefix_caching():
    """创建开启prefix_caching的PagedKVCacheManager实例"""
    return PagedKVCacheManager(
        num_blocks=100,
        num_hot_req=100,
        max_seq_len=2048,
        dp_rank=0,
        block_size=512,
        enable_prefix_caching=True,
    )


class TestPagedKVCacheManagerWithPrefixCaching:

    def test_initialization_with_prefix_caching(self):
        """测试开启prefix_caching"""
        cache_manager = PagedKVCacheManager(
            num_blocks=100,
            num_hot_req=100,
            max_seq_len=2048,
            dp_rank=0,
            block_size=512,
            enable_prefix_caching=True,
        )

        assert cache_manager.enable_prefix_caching is True
        assert len(cache_manager.task_to_cache_ids) == 0
        assert len(cache_manager.task_to_token_blocks) == 0

    def test_identity_chain_builder_with_prefix_pool(
        self,
    ):
        cache_manager = PagedKVCacheManager(
            num_blocks=100,
            num_hot_req=100,
            max_seq_len=2048,
            dp_rank=0,
            block_size=512,
            enable_prefix_caching=True,
        )

        builder = cache_manager.identity_builder

        # tokens长度为blk_size
        tokens = list(range(512))
        identity_1 = builder.make_identity(
            tokens,
            NONE_BLK_HASH,
            canonical_prefix_hashes=cache_manager.enable_prefix_caching,
        )
        block1 = TokenBlock(
            identity=identity_1, runtime=BlockRuntime(cache_idx=0, active_cnt=1)
        )
        cache_manager.upload_to_identity_runtime_pool(block1)
        assert block1.tokens == tuple(tokens)
        assert block1.blk_size == cache_manager.block_size
        assert block1.pre_blk_hash == NONE_BLK_HASH
        assert len(block1.blk_hash) == 64  # SHA-256 produces 64 hex characters
        assert builder.hashed_block_pool[block1.blk_hash] == block1.identity
        assert cache_manager.identity_runtime_pool[block1.blk_hash] is block1.runtime

        # tokens 长度小于blk_size
        tokens = [1, 2, 3, 4]
        identity_2 = builder.make_identity(
            tokens,
            NONE_BLK_HASH,
            canonical_prefix_hashes=cache_manager.enable_prefix_caching,
        )
        block2 = TokenBlock(
            identity=identity_2, runtime=BlockRuntime(cache_idx=1, active_cnt=1)
        )
        cache_manager.upload_to_identity_runtime_pool(block2)
        assert block2.blk_hash is None
        assert block2.blk_hash not in builder.hashed_block_pool
        assert block2.blk_hash not in cache_manager.identity_runtime_pool

        # 相同的tokens和pre_blk_hash，会产生相同的tokenblock
        tokens = list(range(512))
        identity_3 = builder.make_identity(
            tokens,
            NONE_BLK_HASH,
            canonical_prefix_hashes=cache_manager.enable_prefix_caching,
        )
        assert identity_3 is block1.identity
        assert (
            cache_manager.identity_runtime_pool[identity_3.blk_hash] is block1.runtime
        )

        # 若此时为identity_3分配新的runtime并试图同步到cache_manager.identity_runtime_pool,
        # 不会改变cache_manager.identity_runtime_pool中blk_hash对应的runtime信息(为保持chache_manager与各rank上的KVCache元数据一致)
        block3 = TokenBlock(
            identity=identity_3, runtime=BlockRuntime(cache_idx=2, active_cnt=1)
        )
        cache_manager.upload_to_identity_runtime_pool(block3)
        assert cache_manager.identity_runtime_pool[block3.blk_hash] is block1.runtime

    def test_task_life_cycle_in_cache_manager(self):
        main_manager = PagedKVCacheManager(
            num_blocks=100,
            num_hot_req=100,
            max_seq_len=2048,
            dp_rank=0,
            block_size=512,
            enable_prefix_caching=True,
        )

        Backend.cache_managers = [{"main": main_manager}]
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.cache_manager_dict = {"main": main_manager}

        # 在warmup或API server中完成的步骤
        req_0 = UserRequest.create_mock(
            input_len=1024, request_id=f"req_0", enable_thinking=False
        )
        task_0 = Task(f"{req_0.request_id}", req_0)
        task_0.task_type == TaskType.Prefill

        # 在scheduler中完成步骤
        task_0.dp_rank = 0
        assert main_manager.num_cached_blocks(task_0) == 0
        assert len(main_manager.identity_builder.tid_to_identities[task_0.task_id]) == 2
        assert task_0.cached_seq_len == 0
        task_0.set_prefill_chunk_size_for_one_step(1024)  # 假如prefill_chunk_size为1024
        scheduler._prepare_prefill_metadata(task_0, cached_len=0)

        assert task_0.consumed_req_tokens == 0
        assert task_0.new_cache_ids["main"] == [0, 1]
        assert main_manager.task_to_cache_ids[task_0.task_id] == {
            0,
            1,
        }
        task0_token_blocks = main_manager.task_to_token_blocks[task_0.task_id]
        assert len(task0_token_blocks) == 2
        assert (
            main_manager.identity_runtime_pool[task0_token_blocks[0].blk_hash]
            == task0_token_blocks[0].runtime
        )
        assert (
            main_manager.identity_runtime_pool[task0_token_blocks[1].blk_hash]
            == task0_token_blocks[1].runtime
        )
        assert len(main_manager.active_blocks) == 2
        assert len(main_manager.cached_idle_blocks) == 0

        # 在executor中完成的步骤
        task_0.consume_req_tokens()
        task_0.prefix_tokens.append(1)
        task_0.next_tokens = [1]
        assert task_0.consumed_req_tokens == 1024
        assert task_0.task_type == TaskType.Decode

        # 此时来了一条req_1请求, 测试req_1的token被部分击中
        req_1 = UserRequest.create_mock(
            input_len=600, request_id=f"req_1", enable_thinking=False
        )
        task_1 = Task(f"{req_1.request_id}", req_1)
        task_1.task_type == TaskType.Prefill
        task_1.dp_rank = 0
        assert main_manager.num_cached_blocks(task_1) == 1
        assert len(main_manager.identity_builder.tid_to_identities[task_1.task_id]) == 2
        assert main_manager.num_cached_blocks(task_1) * main_manager.block_size == 512

        # 暂不调度task_1，让task_0结束推理
        main_manager.finalize_metadata_all_decode(task_0)
        for block in task0_token_blocks:
            assert block.active_cnt == 0
        assert len(main_manager.active_blocks) == 0
        assert len(main_manager.cached_idle_blocks.keys()) == 2
        assert task_0.task_id not in main_manager.task_to_cache_ids
        assert task_0.task_id not in main_manager.task_to_token_blocks
        assert task_0.task_id not in main_manager.identity_builder.tid_to_identities

        # free_cache_ids: [2,3,4,...], active_blocks:{}, cached_idle_blocks: {0,1}
        # 调度task_1
        num_cached_tokens = (
            main_manager.num_cached_blocks(task_1) * main_manager.block_size
        )
        num_uncomputed_tokens = task_1.prefix_tokens_len - num_cached_tokens
        assert num_uncomputed_tokens == 88
        task_1.set_prefill_chunk_size_for_one_step(88)
        scheduler._prepare_prefill_metadata(task_1, cached_len=num_cached_tokens)
        assert task_1.consumed_req_tokens == 512
        assert task_1.prefill_chunk_size == 88
        assert task_1.new_cache_ids["main"] == [0, 2]
        assert main_manager.task_to_cache_ids[task_1.task_id] == {
            0,
            2,
        }
        assert len(main_manager.active_blocks) == 2  # [0,2]
        assert len(main_manager.cached_idle_blocks.keys()) == 1  # [1]

        # 测试req_2的token被全部击中
        req_2 = UserRequest.create_mock(
            input_len=1024, request_id=f"req_2", enable_thinking=False
        )
        task_2 = Task(f"{req_2.request_id}", req_2)
        task_2.task_type == TaskType.Prefill
        task_2.dp_rank = 0
        assert (
            main_manager.num_cached_blocks(task_2) * main_manager.block_size == 1024
        )  # 被prefix caching击中
        num_uncomputed_tokens = task_2.prefix_tokens_len - (
            main_manager.num_cached_blocks(task_2) * main_manager.block_size
        )
        assert num_uncomputed_tokens == 0
        task_2.set_prefill_chunk_size_for_one_step(1)
        scheduler._prepare_prefill_metadata(task_2, cached_len=1024)
        assert task_2.consumed_req_tokens == 1023
        assert task_2.prefill_chunk_size == 1
        assert task_2.new_cache_ids["main"] == [0, 1]
        assert main_manager.task_to_cache_ids[task_2.task_id] == {
            0,
            1,
        }
        assert len(main_manager.active_blocks) == 3  # [0,1,2]
        main_manager.finalize_metadata_all_decode(task_1)
        main_manager.finalize_metadata_all_decode(task_2)


def _build_singleton_manager(checkpoint_interval=None, mtp_size=1, num_blocks=256):
    return SingletonPagedKVCacheManager(
        num_blocks=num_blocks,
        num_hot_req=4,
        max_seq_len=2048,
        mtp_size=mtp_size,
        dp_rank=0,
        checkpoint_interval=checkpoint_interval,
        enable_prefix_caching=True,
        manager_name="linear",
    )


def _build_mock_task(name, prompt_len, chunk_size, consumed=0):
    req = UserRequest.create_mock(
        input_len=prompt_len, request_id=name, enable_thinking=False
    )
    task = Task(name, req)
    task.task_type = TaskType.Prefill
    task.dp_rank = 0
    # 前缀缓存判定看的是「已经算到哪」（consumed_req_tokens），正常由调度器维护
    task.consumed_req_tokens = consumed
    task.set_prefill_chunk_size_for_one_step(chunk_size)
    return task


class TestSingletonPagedKVCacheManager:
    """linear attn / MTP state 的 manager：in-place block 与 ckpt block 分开管。"""

    def test_without_checkpoint_interval_has_no_prefix_cache(self):
        """没有 checkpoint_interval 时请求只有独占的 in-place block，前缀缓存恒关闭。"""
        manager = _build_singleton_manager(checkpoint_interval=None, mtp_size=2)

        assert manager.block_size == 1
        assert manager.enable_prefix_caching is False
        assert manager.max_blocks_per_req == 2
        # 总占用 = 固定预留（spec block）+ 与 token 位置存在映射的块（C=None 时恒 0）
        assert manager.num_blocks_for_seq_len(0) == 0
        assert manager.num_blocks_for_seq_len(1) == 2
        assert manager.num_fixed_blocks_per_req() == 2
        assert manager.num_token_mapped_blocks_for_seq_len(1) == 0

        task = _build_mock_task("req", 64, 64)
        assert manager.num_cached_blocks(task) == 0
        assert manager.num_owned_blocks(task) == 0

        new_cache_ids = manager.prepare_metadata_before_prefill(task)
        assert len(new_cache_ids) == 2
        # spec block 每请求独占，从不进基类的 task_to_cache_ids（那里只放 ckpt block）
        assert manager.task_to_spec_cache_ids[task.task_id] == new_cache_ids
        assert manager.task_to_cache_ids.get(task.task_id, set()) == set()
        # num_owned_blocks：2个inplace_blocks(fix_block_ids)
        assert manager.num_owned_blocks(task) == 2
        # 每个请求的所有 block 是一次性分配的，后续 prefill step 不再新增
        assert manager.prepare_metadata_before_prefill(task) == []
        # decode 也复用同一批 block
        assert manager.prepare_metadata_before_decode(task) == []

        manager.finalize_metadata_all_decode(task)
        assert task.task_id not in manager.task_to_spec_cache_ids
        assert task.task_id not in manager.task_to_cache_ids
        # in-place block 全部还回空闲池
        assert manager.num_owned_blocks(task) == 0
        assert len(manager.free_cache_ids) == manager.num_blocks

    def test_decode_without_inplace_blocks_raises(self):
        manager = _build_singleton_manager(checkpoint_interval=None)
        task = _build_mock_task("ghost", 64, 64)
        with pytest.raises(RuntimeError, match="without in-place blocks"):
            manager.prepare_metadata_before_decode(task)

    def test_checkpoint_interval_sets_block_size(self):
        manager = _build_singleton_manager(checkpoint_interval=64, mtp_size=2)

        # 一个 ckpt block 覆盖 C 个 token
        assert manager.block_size == 64
        assert manager.checkpoint_interval == 64
        assert manager.enable_prefix_caching is True
        # mtp_size 个 in-place block + ceil_div(max_seq_len, C) 个 ckpt block
        assert manager.max_blocks_per_req == 2 + ceil_div(2048, 64)
        # 总占用 = 固定预留（mtp_size）+ 与 token 位置存在映射的块（ckpt block）
        assert manager.num_fixed_blocks_per_req() == 2
        assert manager.num_blocks_for_seq_len(0) == 0
        assert manager.num_blocks_for_seq_len(64) == 2 + 1
        assert manager.num_blocks_for_seq_len(65) == 2 + 2
        # 位置映射本身只数 ckpt block，且与 len(task_to_cache_ids) 契约一致
        assert manager.num_token_mapped_blocks_for_seq_len(64) == 1
        assert manager.num_token_mapped_blocks_for_seq_len(65) == 2

    def test_block_size_must_equal_checkpoint_interval(self):
        with pytest.raises(ValueError, match="must equal"):
            SingletonPagedKVCacheManager(
                num_blocks=256,
                num_hot_req=4,
                max_seq_len=2048,
                mtp_size=1,
                checkpoint_interval=64,
                enable_prefix_caching=True,
                block_size=128,
                manager_name="linear",
            )

    def test_prefill_returns_spec_then_ckpt_blocks(self):
        """new_cache_ids 的布局：[spec_0 .. spec_{K-1} | ckpt_0 ... ckpt_k]。"""
        manager = _build_singleton_manager(checkpoint_interval=64)
        task = _build_mock_task("req", 256, 128)

        new_cache_ids = manager.prepare_metadata_before_prefill(task)

        # 128 token = 2 个 ckpt block，加上 1 个 in-place spec block
        assert len(new_cache_ids) == 3
        spec_ids = manager.task_to_spec_cache_ids[task.task_id]
        assert new_cache_ids[: len(spec_ids)] == spec_ids
        # task_to_cache_ids 只放可复用的 ckpt block（调度器据此做前缀缓存记账）
        assert manager.task_to_cache_ids[task.task_id] == set(new_cache_ids[1:])

        # spec block 独占，不再重复分配；续算只追加 ckpt block
        task.consume_req_tokens()
        task.set_prefill_chunk_size_for_one_step(128)
        follow_up = manager.prepare_metadata_before_prefill(task)
        assert len(follow_up) == 2
        assert spec_ids[0] not in follow_up

        manager.finalize_metadata_all_decode(task)
        # 请求持有的块全部释放：in-place 块回到空闲池，已被 prefix cache 复用的
        # ckpt 块留在 cached_idle_blocks 里等下一个同前缀的请求
        assert not manager.active_blocks
        assert len(manager.free_cache_ids) + len(manager.cached_idle_blocks) == (
            manager.num_blocks
        )

    def test_full_hit_task_reuses_spec_and_cached_ckpt_blocks(self):
        """同前缀的第二个任务：spec block 自己新分配，ckpt block 命中共享。"""
        manager = _build_singleton_manager(checkpoint_interval=64)
        first = _build_mock_task("req_a", 256, 256)
        assert len(manager.prepare_metadata_before_prefill(first)) == 1 + 4
        first_ckpt_ids = sorted(manager.task_to_cache_ids[first.task_id])

        # 命中 4 个 ckpt block（4 * 64 = 256 token），只差最后一个 token 没算
        second = _build_mock_task("req_b", 256, 1, consumed=255)
        assert manager.num_cached_blocks(second) == 4

        # new_cache_ids = [自己的 spec block | 命中的 4 个 ckpt block]
        new_cache_ids = manager.prepare_metadata_before_prefill(second)
        assert new_cache_ids[0] in manager.task_to_spec_cache_ids[second.task_id]
        assert sorted(new_cache_ids[1:]) == first_ckpt_ids
        # 命中没有新分配任何 ckpt block
        assert manager.task_to_cache_ids[second.task_id] == set(first_ckpt_ids)

        manager.finalize_metadata_all_decode(second)
        # first 还持有这些 ckpt block，不会被回收
        assert sorted(manager.task_to_cache_ids[first.task_id]) == first_ckpt_ids
        manager.finalize_metadata_all_decode(first)
        assert not manager.active_blocks
        assert len(manager.free_cache_ids) + len(manager.cached_idle_blocks) == (
            manager.num_blocks
        )

    def test_admission_budget_equals_first_allocation(self):
        """准入的 need 必须等于首次 prefill 真正要拿的块数（固定预留 + 位置映射的块）。

        旧实现里 num_blocks_for_seq_len 只数 ckpt block，need 偏小：池子接近打满时
        调度器会放行，随后 prepare 里 get_free_cache_idx 抛 "No more free KVCache blocks"。
        """
        manager = _build_singleton_manager(checkpoint_interval=64, mtp_size=2)
        task = _build_mock_task("req", 256, 64)
        cached_len = task.consumed_req_tokens
        target_len = cached_len + task.next_req_tokens_len

        need = manager.num_blocks_to_reserve(
            task, cached_len=cached_len, target_len=target_len
        )
        assert need == 2 + 1  # mtp_size 个 in-place block + 1 个 ckpt block
        assert len(manager.prepare_metadata_before_prefill(task)) == need

        # 续算时同样成立：need 恰好是这一步新追加的 ckpt block 数
        task.consume_req_tokens()
        task.set_prefill_chunk_size_for_one_step(64)
        cached_len = task.consumed_req_tokens
        target_len = cached_len + task.next_req_tokens_len
        need = manager.num_blocks_to_reserve(
            task, cached_len=cached_len, target_len=target_len
        )
        assert need == 1
        assert len(manager.prepare_metadata_before_prefill(task)) == need

    def test_admission_budget_counts_fixed_blocks_without_ownership(self):
        """任务命中前缀但还没进 manager 时，固定预留不能被当成"已占用"。

        PD decode 路径就是这样：cached_len 来自 prefill 侧的命中长度，而本实例上一个块
        都还没分配。旧实现会把固定预留抵扣掉，need 算成 0。
        """
        manager = _build_singleton_manager(checkpoint_interval=None, mtp_size=2)
        task = _build_mock_task("req", 256, 64, consumed=255)

        need = manager.num_blocks_to_reserve(
            task,
            cached_len=task.consumed_req_tokens,
            target_len=task.consumed_req_tokens + task.next_req_tokens_len,
        )
        assert need == 2
        assert len(manager.prepare_metadata_before_prefill(task)) == need

    def test_realloc_clears_spec_bookkeeping(self):
        manager = _build_singleton_manager(checkpoint_interval=64)
        task = _build_mock_task("req", 128, 128)
        manager.prepare_metadata_before_prefill(task)

        manager.realloc(1024)
        assert manager.num_blocks == 1024
        assert not manager.task_to_spec_cache_ids


class TestSingletonAdmissionCapacity:
    """准入容量检查必须把固定预留算进 need，否则放行后会在分配时炸掉。"""

    class _Harness:
        """只填 Scheduler 容量检查需要的属性，复用它的真实逻辑。"""

        _check_prefill_capacity = Scheduler._check_prefill_capacity
        _inflight_prefill_reserved_blocks = Scheduler._inflight_prefill_reserved_blocks

        def __init__(self, manager):
            self.cache_manager_dict = {manager.manager_name: manager}
            self.kvcache_block_threshold = manager.num_blocks

    def _manager(self, num_blocks):
        return SingletonPagedKVCacheManager(
            num_blocks=num_blocks,
            num_hot_req=4,
            max_seq_len=2048,
            mtp_size=2,
            dp_rank=0,
            checkpoint_interval=64,
            enable_prefix_caching=True,
            manager_name="linear",
        )

    def test_pool_exactly_fit_is_admitted_and_allocatable(self):
        """可用块刚好等于 need（含 mtp_size 个 in-place block）时必须能分配成功。"""
        manager = self._manager(num_blocks=6)
        harness = self._Harness(manager)

        holder = _build_mock_task("holder", 256, 64)
        assert len(manager.prepare_metadata_before_prefill(holder)) == 3

        newbie = _build_mock_task("newbie", 256, 64)
        assert harness._check_prefill_capacity(newbie, 0) is KVCacheCapacityStatus.OK
        # 放行就必须真的能分配出来（need == 实际申请块数 == 3 == 剩余空闲块）
        assert len(manager.prepare_metadata_before_prefill(newbie)) == 3

    def test_pool_one_block_short_is_congested(self):
        """少一块就必须判为拥塞，而不是放行后抛 No more free KVCache blocks。"""
        manager = self._manager(num_blocks=5)
        harness = self._Harness(manager)

        holder = _build_mock_task("holder", 256, 64)
        manager.prepare_metadata_before_prefill(holder)

        newbie = _build_mock_task("newbie", 256, 64)
        assert (
            harness._check_prefill_capacity(newbie, 0)
            is KVCacheCapacityStatus.CONGESTED
        )


class TestBlockBuilderInterfaces:
    def test_identity_chain_builder_reuses_existing_identity(self):
        builder = BlockIdentityChainBuilder.acquire("__pytest_iface_collision", 4)
        first = builder.make_identity([1, 2, 3, 4], NONE_BLK_HASH)
        second = builder.make_identity([1, 2, 3, 4], NONE_BLK_HASH)
        assert first is second
        assert first.blk_hash in builder.hashed_block_pool

    def test_identity_chain_builder_chain(self):
        builder = BlockIdentityChainBuilder.acquire("__pytest_iface_chain", 4)
        identities = builder.make_identity_chain_from_tokens([1, 2, 3, 4, 5, 6, 7, 8])
        assert len(identities) == 2
        assert identities[0].blk_hash is not None
        assert identities[1].pre_blk_hash == identities[0].blk_hash
        assert identities[0].blk_hash in builder.hashed_block_pool
