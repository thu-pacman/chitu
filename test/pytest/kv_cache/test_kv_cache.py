import pytest
from chitu.kv_cache import GlobalLocalMap, PagedKVCache
from chitu.global_vars import set_global_args
from omegaconf import OmegaConf
from chitu.task import PackedTasksBase
from chitu.task_type import TaskType
import torch


@pytest.fixture(autouse=True)
def setup_global_args():
    """Set up global arguments for tests."""
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 1024,
                    "max_batch_size": 10,
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "schedule_overlap": True,
                    "prefill_chunk_size": None,
                    "mtp_size": 1,
                    "dp_size": 1,
                    "use_cuda_graph": True,
                    "enable_prefix_caching": False,
                }
            }
        ),
        need_ensure=False,
    )


class TestPagedKVCache:
    """Tests for PagedKVCache class."""

    @pytest.fixture
    def kvcache(self):
        """Create a PagedKVCache instance for testing."""
        layer_id_map = GlobalLocalMap.from_range(0, 4)

        return PagedKVCache(
            layer_id_map,
            num_hot_req=100,
            max_seq_len=2048,
            num_blocks=100,
            n_local_kv_heads=8,
            head_dim=64,
            block_size=512,
            device="cpu",
        )

    def test_initialization(self, kvcache: PagedKVCache):
        """Test PagedKVCache initialization."""
        assert kvcache.num_blocks == 100
        assert kvcache.block_size == 512
        assert kvcache.num_layers == 4
        assert kvcache.num_hot_req == 100
        assert kvcache.max_seq_len == 2048
        assert kvcache.max_num_blocks == 400
        assert "k" in kvcache.paged_kv_cache
        assert "v" in kvcache.paged_kv_cache
        assert kvcache.paged_kv_cache["k"].shape == (4, 100, 512, 8, 64)
        assert kvcache.paged_kv_cache["v"].shape == (4, 100, 512, 8, 64)
        assert len(kvcache.block_table) == 0
        assert kvcache.get_gpu_block_table() is not None

    def test_realloc(self, kvcache: PagedKVCache):
        """测试重新分配block数量,应该与cache_manager的表现一致"""
        kvcache.realloc(50)

        assert kvcache.num_blocks == 50
        for _, tensor in kvcache.paged_kv_cache.items():
            assert tensor.shape == (4, 50, 512, 8, 64)

        # 测试最大分配的num_blocks不超过paged_kv_cache.max_num_blocks (200)
        kvcache.realloc(500)
        assert kvcache.num_blocks == 400
        for _, tensor in kvcache.paged_kv_cache.items():
            assert tensor.shape == (4, 400, 512, 8, 64)

    def test_tasks_life_cycle_in_kv_cache(self, kvcache: PagedKVCache):

        # prefill task
        tasks = PackedTasksBase(
            num_tasks=2,
            task_ids=["req_0", "req_1"],
            task_type=TaskType.Prefill,
            tokens=[[1] * 512, [1, 2, 3, 4, 5, 6, 7, 8, 9]],
            num_tokens=521,
            new_cache_ids_list=[
                {"main": [0]},
                {"main": [1]},
            ],
        )

        # test prepare_cache_prefill
        kvcache.prepare_cache_prefill(tasks)
        assert kvcache.curr_tids == tasks.task_ids
        assert kvcache.seq_len_delta.old.lens_list == [0, 0]
        assert kvcache.seq_len_delta.new.lens_list == [512, 9]
        assert kvcache.tid_to_cached_len[tasks.task_ids[0]] == 512
        assert kvcache.tid_to_cached_len[tasks.task_ids[1]] == 9
        assert kvcache.block_table[tasks.task_ids[0]] == [0]
        assert kvcache.block_table[tasks.task_ids[1]] == [1]

        # test get_accessor after prepare_cache_prefill
        accessor = kvcache.get_accessor(0, is_mtp=False)
        assert torch.all(
            accessor.block_table == torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0]])
        )
        for _, tensor in accessor.kv.items():
            assert tensor.shape == (100, 512, 8, 64)

        # decode task
        tasks = PackedTasksBase(
            num_tasks=2,
            task_ids=["req_0", "req_1"],
            task_type=TaskType.Decode,
            tokens=[[2], [2]],
            num_tokens=2,
            new_cache_ids_list=[
                {"main": [2]},
                {},
            ],
        )

        # test prepare_cache_decode
        kvcache.prepare_cache_decode(tasks)
        assert kvcache.seq_len_delta.old.lens_list == [512, 9]
        assert kvcache.seq_len_delta.new.lens_list == [513, 10]

        # test accessor after prepare_cache_decode
        accessor = kvcache.get_accessor(0, is_mtp=False)
        assert torch.all(
            accessor.block_table == torch.tensor([[0, 2, 0, 0], [1, 0, 0, 0]])
        )

        # test finalize_cache_all_decode
        tasks = PackedTasksBase(
            num_tasks=2,
            task_ids=["req_0", "req_1"],
            task_type=TaskType.Special,
        )
        kvcache.finalize_cache_all_decode(tasks)
        assert "req_0" not in kvcache.tid_to_cached_len
        assert "req_0" not in kvcache.block_table
        assert "req_1" not in kvcache.tid_to_cached_len
        assert "req_1" not in kvcache.block_table

        # create packedtasks hit by prefix caching
        tasks = PackedTasksBase(
            num_tasks=2,
            task_ids=["req_3", "req_4"],
            task_type=TaskType.Prefill,
            tokens=[[1, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9]],
            num_tokens=11,
            new_cache_ids_list=[
                {"main": [0, 1]},
                {"main": [2]},
            ],
            inc_hit_tokens_list=[512, 0],
        )
        # req_3: [1]*512 + [1,1]
        # req_4: prompt length小于block size, 无法被击中

        # test prepare_cache_prefill when enable prefix caching
        kvcache.prepare_cache_prefill(tasks)
        assert kvcache.curr_tids == tasks.task_ids
        assert kvcache.seq_len_delta.old.lens_list == [512, 0]
        assert kvcache.seq_len_delta.new.lens_list == [514, 9]

        accessor = kvcache.get_accessor(0, is_mtp=False)
        assert torch.all(
            accessor.block_table == torch.tensor([[0, 1, 0, 0], [2, 0, 0, 0]])
        )

        # 结束'req_3','req_4'
        tasks = PackedTasksBase(
            num_tasks=2,
            task_ids=["req_3", "req_4"],
            task_type=TaskType.Special,
        )
        kvcache.finalize_cache_all_decode(tasks)
        assert "req_3" not in kvcache.tid_to_cached_len
        assert "req_3" not in kvcache.block_table
        assert "req_4" not in kvcache.tid_to_cached_len
        assert "req_4" not in kvcache.block_table
