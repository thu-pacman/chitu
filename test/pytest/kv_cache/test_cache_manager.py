import pytest
from chitu.kv_cache import (
    DeepSeekV4CompressedKVCacheManager,
    DeepSeekV4SlidingKVCacheManager,
    PagedKVCacheManager,
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
from chitu.scheduler import Scheduler


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
        assert (
            task.kv_cache_len_used_in_completed_steps == 0
        ), f"{task.kv_cache_len_used_in_completed_steps} vs 0"
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
        assert task.kv_cache_len_used_in_completed_steps == 300
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
        )  # task计算kv_cache_len_used_in_completed_steps_and_next_step时会用到全局的mtp_size信息
        cache_manager.mtp_size = 500  # 设定mtp_size为500

        # 开始被decode调度
        assert cache_manager.num_cached_blocks(task) == 2
        aviable_blocks = cache_manager.num_blocks - cache_manager.num_active_blocks
        cur_blocks = cache_manager.num_cached_blocks(task)
        target_blocks = ceil_div(
            task.kv_cache_len_used_in_completed_steps_and_next_step,
            cache_manager.block_size,
        )
        assert task.kv_cache_len_used_in_completed_steps_and_next_step == 1600
        assert aviable_blocks == 98
        assert cur_blocks == 2
        assert target_blocks == 4
        task.new_cache_ids = {
            "main": cache_manager.prepare_metadata_before_decode(task)
        }

        assert task.new_cache_ids["main"] == [2, 3]
        assert cache_manager.task_to_cache_ids[task.task_id] == {0, 1, 2, 3}
        assert cache_manager.active_blocks[2] is task_token_blocks[2].runtime
        assert cache_manager.active_blocks[3] is task_token_blocks[3].runtime
        assert task_token_blocks[1].active_cnt == 1

        # 在executor中执行decode step
        task.next_tokens = [1]
        task.prefix_tokens.extend([1] * 400)  # 假设主模型只接受了400个token
        assert task.prefix_tokens_len == 1001

        # 再次被decode调度
        assert cache_manager.num_cached_blocks(task) == 4
        aviable_blocks = cache_manager.num_blocks - cache_manager.num_active_blocks
        cur_blocks = cache_manager.num_cached_blocks(task)
        target_blocks = ceil_div(
            task.kv_cache_len_used_in_completed_steps_and_next_step,
            cache_manager.block_size,
        )
        assert task.kv_cache_len_used_in_completed_steps_and_next_step == 2000
        assert aviable_blocks == 96
        assert cur_blocks == 4
        assert target_blocks == 4
        task.new_cache_ids = {
            "main": cache_manager.prepare_metadata_before_decode(task)
        }
        assert task.new_cache_ids["main"] == []
        assert cache_manager.task_to_cache_ids[task.task_id] == {0, 1, 2, 3}


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
        assert task_0.kv_cache_len_used_in_completed_steps == 0
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
