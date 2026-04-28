import pytest
from chitu.kv_cache import (
    PagedKVCacheManager,
    TokenBlock,
    BlockIdentity,
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


@pytest.fixture(autouse=True)
def setup_global_args():
    """Set up global arguments for tests."""
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
                "dp_config": {
                    "enabled": True,
                    "router": {
                        "pd_disaggregation": {
                            "enabled": False,
                        }
                    },
                },
            }
        ),
        need_ensure=False,
    )


@pytest.fixture
def cache_manager():
    """创建不开启prefix_caching的PagedKVCacheManager实例"""
    return PagedKVCacheManager(
        num_blocks=100,
        num_hot_req=100,
        max_seq_len=2048,
        dp_rank=0,
        block_size=512,
        enable_prefix_caching=False,
    )


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


class TestPagedKVCacheManager:
    """测试PagedKVCacheManager类"""

    def test_initialization(self, cache_manager: PagedKVCacheManager):
        """测试类初始化"""
        assert cache_manager.num_blocks == 100
        assert cache_manager.block_size == 512
        assert cache_manager.max_num_blocks == 400
        assert cache_manager.dp_rank == 0
        assert len(cache_manager.free_cache_ids) == 100
        assert len(cache_manager.active_blocks) == 0
        assert len(cache_manager.cached_idle_blocks) == 0
        assert cache_manager.enable_prefix_caching is False

    def test_realloc(self, cache_manager: PagedKVCacheManager):
        """测试重新分配block数量"""
        cache_manager.realloc(50)

        assert cache_manager.num_blocks == 50
        assert len(cache_manager.free_cache_ids) == 50

        # 测试最大分配的num_blocks不超过cache_manager.max_num_blocks (200)
        cache_manager.realloc(500)
        assert cache_manager.num_blocks == 400
        assert len(cache_manager.free_cache_ids) == 400

    def test_get_free_cache_idx(self, cache_manager: PagedKVCacheManager):
        """测试get_free_cache_idx函数"""

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

    def test_get_free_cache_idx_collects_evicted_blk_hash(
        self, cache_manager: PagedKVCacheManager
    ):
        """测试被动逐出时会记录evicted blk hash，并可按上限批量弹出"""
        cache_manager.free_cache_ids = deque()
        block = TokenBlock(tokens=[1, 2, 3, 4], blk_size=4, pre_blk_hash=NONE_BLK_HASH)
        block.generate_blk_hash()
        block.cache_idx = 7
        cache_manager.cached_idle_blocks[7] = block
        cache_manager.cache_idx_to_hash[7] = block.blk_hash

        idx = cache_manager.get_free_cache_idx()
        assert idx == 7

        popped = cache_manager.pop_evicted_blk_hashes(max_items=512)
        assert popped == [block.blk_hash]
        assert cache_manager.pop_evicted_blk_hashes(max_items=512) == []

    def test_identity_chain_builder(self, cache_manager: PagedKVCacheManager):
        """测试BlockIdentityChainBuilder构建逻辑"""
        builder = BlockIdentityChainBuilder(block_size=cache_manager.block_size)

        # 测试pre_blk_hash为None
        tokens = list(range(512))
        identity = builder.make_identity(tokens, NONE_BLK_HASH)

        assert list(identity.tokens) == tokens
        assert identity.blk_size == cache_manager.block_size
        assert identity.pre_blk_hash == NONE_BLK_HASH

        # 测试给定一个pre_hash值
        tokens = list(range(cache_manager.block_size))
        pre_hash = "some_previous_hash"
        identity = builder.make_identity(tokens, pre_hash)
        assert identity.pre_blk_hash == pre_hash
        assert (
            identity.blk_hash not in cache_manager.hashed_block_pool
        )  # 未开启prefix caching，不会维护hashed_block_pool

        # 测试tokens长度超过block_size长度
        tokens = list(range(cache_manager.block_size + 1))
        with pytest.raises(ValueError):
            builder.make_identity(tokens, NONE_BLK_HASH)

    def test_task_life_cycle_in_cache_manager(self, cache_manager: PagedKVCacheManager):
        """测试任务在cache_manager中的生命周期，
        涉及函数:prepare_metadata_before_prefill/decode, finalize_metadata_all_decode
        """

        Backend.cache_managers = [{"main": cache_manager}]

        # 在warmup或API server中完成的步骤
        req = UserRequest.create_mock(
            input_len=600, request_id=f"test_task", enable_thinking=False
        )
        task = Task(f"{req.request_id}", req)
        task.task_type == TaskType.Prefill

        # 在scheduler中完成步骤
        task.dp_rank = 0
        task.prompt_to_token_block(task.dp_rank)
        assert len(task.token_blocks) == 2
        assert task.kv_cache_len_used_in_completed_steps == 0
        task.set_prefill_chunk_size_for_one_step(300)  # 假如prefill_chunk_size为300
        cache_manager.prepare_metadata_before_prefill(task)

        assert task.consumed_req_tokens == 0
        assert task.new_cache_ids == [0]
        assert cache_manager.task_to_cache_ids[task.task_id] == {0}
        assert cache_manager.active_blocks[0] is task.token_blocks[0].runtime
        assert cache_manager.tid_to_cached_len[task.task_id] == 300

        # 在executor中完成的步骤
        task.consume_req_tokens()
        assert task.consumed_req_tokens == 300
        assert task.task_type == TaskType.Prefill

        # 第二次被prefill调度
        assert task.num_cached_blocks == 1
        assert task.kv_cache_len_used_in_completed_steps == 300
        task.set_prefill_chunk_size_for_one_step(300)  # 假如prefill_chunk_size为300
        cache_manager.prepare_metadata_before_prefill(task)
        assert task.new_cache_ids == [1]
        assert cache_manager.task_to_cache_ids[task.task_id] == {0, 1}
        assert cache_manager.active_blocks[1] is task.token_blocks[1].runtime
        assert task.token_blocks[1].active_cnt == 1
        assert cache_manager.tid_to_cached_len[task.task_id] == 600

        # 在executor中完成的步骤
        task.consume_req_tokens()
        task.next_token = 1
        task.prefix_tokens.append(task.next_token)
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
        )
        cache_manager.mtp_size = 500  # 设定mtp_size为500

        # 开始被decode调度
        assert task.num_cached_blocks == 2
        aviable_blocks = cache_manager.num_blocks - cache_manager.num_active_blocks
        cur_blocks = task.num_cached_blocks
        target_blocks = ceil_div(
            task.kv_cache_len_used_in_completed_steps_and_next_step,
            cache_manager.block_size,
        )
        assert task.kv_cache_len_used_in_completed_steps_and_next_step == 1100
        assert aviable_blocks == 98
        assert cur_blocks == 2
        assert target_blocks == 3
        cache_manager.prepare_metadata_before_decode(task)
        assert task.new_cache_ids == [2]
        assert cache_manager.task_to_cache_ids[task.task_id] == {0, 1, 2}
        assert cache_manager.active_blocks[2] is task.token_blocks[2].runtime
        assert task.token_blocks[1].active_cnt == 1
        assert cache_manager.tid_to_cached_len[task.task_id] == 1100

        # 在executor中执行decode step
        task.next_token = 1
        task.prefix_tokens.extend([1] * 400)  # 假设主模型只接受了400个token
        assert task.prefix_tokens_len == 1001

        # 再次被decode调度
        assert task.num_cached_blocks == 3
        aviable_blocks = cache_manager.num_blocks - cache_manager.num_active_blocks
        cur_blocks = task.num_cached_blocks
        target_blocks = ceil_div(
            task.kv_cache_len_used_in_completed_steps_and_next_step,
            cache_manager.block_size,
        )
        assert task.kv_cache_len_used_in_completed_steps_and_next_step == 1500
        assert aviable_blocks == 97
        assert cur_blocks == 3
        assert target_blocks == 3
        cache_manager.prepare_metadata_before_decode(task)
        assert task.new_cache_ids == []
        assert cache_manager.task_to_cache_ids[task.task_id] == {0, 1, 2}
        assert cache_manager.tid_to_cached_len[task.task_id] == 1500


class TestPagedKVCacheManagerWithPrefixCaching:

    def test_initialization_with_prefix_caching(
        self, cache_manager_with_prefix_caching: PagedKVCacheManager
    ):
        """测试开启prefix_caching"""

        assert cache_manager_with_prefix_caching.enable_prefix_caching is True
        assert isinstance(
            cache_manager_with_prefix_caching.hashed_block_pool, WeakValueDictionary
        )
        assert len(cache_manager_with_prefix_caching.task_hashed_block_cnt) == 0

    def test_identity_chain_builder_with_prefix_pool(
        self, cache_manager_with_prefix_caching: PagedKVCacheManager
    ):
        builder = BlockIdentityChainBuilder(
            block_size=cache_manager_with_prefix_caching.block_size,
            existing_identities=cache_manager_with_prefix_caching.hashed_block_pool,
        )

        # tokens长度为blk_size
        tokens = list(range(512))
        identity_1 = builder.make_identity(tokens, NONE_BLK_HASH, auto_register=True)
        block1 = TokenBlock(identity=identity_1, runtime=BlockRuntime())
        cache_manager_with_prefix_caching.get_or_register_identity(block1)

        assert block1.tokens == tokens
        assert block1.blk_size == cache_manager_with_prefix_caching.block_size
        assert block1.pre_blk_hash == NONE_BLK_HASH
        assert len(block1.blk_hash) == 64  # SHA-256 produces 64 hex characters
        assert (
            cache_manager_with_prefix_caching.hashed_block_pool[block1.blk_hash]
            == block1.identity
        )

        # tokens 长度小于blk_size
        tokens = [1, 2, 3, 4]
        identity = builder.make_identity(tokens, NONE_BLK_HASH, auto_register=True)
        assert identity.blk_hash is None
        assert (
            identity.blk_hash not in cache_manager_with_prefix_caching.hashed_block_pool
        )

        # 相同的tokens和pre_blk_hash，会产生相同的tokenblock
        tokens = list(range(512))
        identity_2 = builder.make_identity(tokens, NONE_BLK_HASH, auto_register=True)
        block2 = TokenBlock(identity=identity_2, runtime=BlockRuntime())
        cache_manager_with_prefix_caching.get_or_register_identity(block2)
        assert block2.tokens == tokens
        assert block2.identity == block1.identity
        assert block2.runtime == block1.runtime
        assert (
            cache_manager_with_prefix_caching.hashed_block_pool[block2.blk_hash]
            == block2.identity
        )
        assert block2.blk_hash == block1.blk_hash
        assert block2.pre_blk_hash == block1.pre_blk_hash

    def test_task_life_cycle_in_cache_manager(
        self, cache_manager_with_prefix_caching: PagedKVCacheManager
    ):
        Backend.cache_managers = [{"main": cache_manager_with_prefix_caching}]

        # 在warmup或API server中完成的步骤
        req_0 = UserRequest.create_mock(
            input_len=1024, request_id=f"req_0", enable_thinking=False
        )
        task_0 = Task(f"{req_0.request_id}", req_0)
        task_0.task_type == TaskType.Prefill

        # 在scheduler中完成步骤
        task_0.dp_rank = 0
        task_0.prompt_to_token_block(task_0.dp_rank)
        assert len(task_0.token_blocks) == 2
        assert task_0.kv_cache_len_used_in_completed_steps == 0
        task_0.set_prefill_chunk_size_for_one_step(1024)  # 假如prefill_chunk_size为1024
        cache_manager_with_prefix_caching.prepare_metadata_before_prefill(task_0)

        assert task_0.consumed_req_tokens == 0
        assert task_0.new_cache_ids == [0, 1]
        assert cache_manager_with_prefix_caching.task_to_cache_ids[task_0.task_id] == {
            0,
            1,
        }
        assert len(cache_manager_with_prefix_caching.active_blocks) == 2
        assert len(cache_manager_with_prefix_caching.cached_idle_blocks) == 0
        assert (
            cache_manager_with_prefix_caching.tid_to_cached_len[task_0.task_id] == 1024
        )

        # 在executor中完成的步骤
        task_0.consume_req_tokens()
        task_0.prefix_tokens.append(1)
        task_0.next_token = 1
        assert task_0.consumed_req_tokens == 1024
        assert task_0.task_type == TaskType.Decode

        # task_0在做一次decode，才能把hashed + cached block存入到cache_manager_with_prefix_caching.hashed_block_pool中
        cache_manager_with_prefix_caching.prepare_metadata_before_decode(task_0)
        assert (
            cache_manager_with_prefix_caching.task_hashed_block_cnt[task_0.task_id] == 2
        )
        assert (
            task_0.token_blocks[0].blk_hash
            in cache_manager_with_prefix_caching.hashed_block_pool
        )
        assert task_0.token_blocks[1].pre_blk_hash == task_0.token_blocks[0].blk_hash

        # 测试req_1的token被部分击中时
        # 此时来了一条req_1请求
        req_1 = UserRequest.create_mock(
            input_len=600, request_id=f"req_1", enable_thinking=False
        )
        task_1 = Task(f"{req_1.request_id}", req_1)
        task_1.task_type == TaskType.Prefill
        task_1.dp_rank = 0
        task_1.prompt_to_token_block(task_1.dp_rank)
        assert len(task_1.token_blocks) == 2
        assert (
            task_1.kv_cache_len_used_in_completed_steps == 512
        )  # 被prefix caching击中

        # 暂不调度task_1，让task_0结束推理
        cache_manager_with_prefix_caching.finalize_metadata_all_decode(task_0)
        for block in task_0.token_blocks:
            assert block.active_cnt == 0
        assert len(cache_manager_with_prefix_caching.active_blocks) == 0
        assert len(cache_manager_with_prefix_caching.cached_idle_blocks.keys()) == 3
        assert task_0.task_id not in cache_manager_with_prefix_caching.tid_to_cached_len
        assert task_0.task_id not in cache_manager_with_prefix_caching.task_to_cache_ids
        assert (
            task_0.task_id
            not in cache_manager_with_prefix_caching.task_hashed_block_cnt
        )

        # free_cache_ids: [3,4,...], active_blocks:{}, cached_idle_blocks: {0,1,2}
        # 调度task_1
        num_uncomputed_tokens = (
            task_1.prefix_tokens_len - task_1.kv_cache_len_used_in_completed_steps
        )
        assert num_uncomputed_tokens == 88
        task_1.set_prefill_chunk_size_for_one_step(88)
        cache_manager_with_prefix_caching.prepare_metadata_before_prefill(task_1)
        assert task_1.consumed_req_tokens == 512
        assert task_1.prefill_chunk_size == 88
        assert task_1.new_cache_ids == [0, 3]
        assert cache_manager_with_prefix_caching.task_to_cache_ids[task_1.task_id] == {
            0,
            3,
        }
        assert len(cache_manager_with_prefix_caching.active_blocks) == 2  # [0,3]
        assert (
            len(cache_manager_with_prefix_caching.cached_idle_blocks.keys()) == 2
        )  # [1,2]
        assert (
            cache_manager_with_prefix_caching.tid_to_cached_len[task_1.task_id] == 600
        )

        # 测试req_2的token被全部击中时
        req_2 = UserRequest.create_mock(
            input_len=1024, request_id=f"req_2", enable_thinking=False
        )
        task_2 = Task(f"{req_2.request_id}", req_2)
        task_2.task_type == TaskType.Prefill
        task_2.dp_rank = 0
        task_2.prompt_to_token_block(task_2.dp_rank)
        assert len(task_2.token_blocks) == 2
        assert (
            task_2.kv_cache_len_used_in_completed_steps == 1024
        )  # 被prefix caching击中
        num_uncomputed_tokens = (
            task_2.prefix_tokens_len - task_2.kv_cache_len_used_in_completed_steps
        )
        assert num_uncomputed_tokens == 0
        task_2.set_prefill_chunk_size_for_one_step(1)
        cache_manager_with_prefix_caching.prepare_metadata_before_prefill(task_2)
        assert (
            task_2.consumed_req_tokens == 1023
        )  # 最后一个token需要作为输入，获取next_token
        assert task_2.prefill_chunk_size == 1
        assert task_2.new_cache_ids == [0, 1]
        assert cache_manager_with_prefix_caching.task_to_cache_ids[task_2.task_id] == {
            0,
            1,
        }
        assert len(cache_manager_with_prefix_caching.active_blocks) == 3
        assert (
            cache_manager_with_prefix_caching.tid_to_cached_len[task_2.task_id] == 1024
        )

        cache_manager_with_prefix_caching.finalize_metadata_all_decode(task_1)
        cache_manager_with_prefix_caching.finalize_metadata_all_decode(task_2)


class TestBlockBuilderInterfaces:
    def test_identity_chain_builder_reuses_existing_identity(self):
        existing: dict[str, BlockIdentity] = {}
        builder = BlockIdentityChainBuilder(block_size=4, existing_identities=existing)
        first = builder.make_identity([1, 2, 3, 4], NONE_BLK_HASH, auto_register=True)
        second = builder.make_identity([1, 2, 3, 4], NONE_BLK_HASH, auto_register=False)
        assert first is second
        assert first.blk_hash in existing

    def test_identity_chain_builder_chain(self):
        existing: dict[str, BlockIdentity] = {}
        builder = BlockIdentityChainBuilder(block_size=4, existing_identities=existing)
        identities = builder.build(tokens=[1, 2, 3, 4, 5, 6, 7, 8], auto_register=True)
        assert len(identities) == 2
        assert identities[0].blk_hash is not None
        assert identities[1].pre_blk_hash == identities[0].blk_hash
        assert identities[0].blk_hash in existing
