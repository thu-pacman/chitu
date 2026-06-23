import pytest
from chitu.kv_cache import (
    GlobalLocalMap,
    DeepSeekV4PagedKVCache,
    DeepSeekV4SlidingWindowPagedKVCache,
    PagedKVCache,
)
from chitu.global_vars import set_global_args
from chitu.ops.kv_cache import append_to_sliding_window_paged_kv_cache
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
        need_preprocess=False,
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

    def test_paged_cache_can_decouple_page_table_seq_len(self):
        layer_id_map = GlobalLocalMap.from_range(0, 1)
        cache = PagedKVCache(
            layer_id_map,
            num_hot_req=2,
            max_seq_len=4096,
            page_table_max_seq_len=1024,
            num_blocks=8,
            shape_per_token_dict={"compressed": torch.Size([3])},
            dtype_dict={"compressed": torch.bfloat16},
            block_size=256,
            device="cpu",
        )

        assert cache.max_seq_len == 4096
        assert cache.max_total_delta_len == 8192
        assert cache.page_table_max_seq_len == 1024
        assert cache.max_blocks_per_req == 4
        assert cache.page_table_max_num_blocks == 8
        assert cache.max_num_blocks == 8
        cache.realloc(100)
        assert cache.num_blocks == 8
        assert cache.paged_kv_cache["compressed"].shape == (1, 8, 256, 3)

    def test_deepseek_v4_paged_cache_uses_manager_name_for_block_table(self):
        layer_id_map = GlobalLocalMap.from_range(0, 1)
        common_kwargs = {
            "num_hot_req": 2,
            "max_seq_len": 16,
            "num_blocks": 8,
            "shape_per_token_dict": {"sliding_window": torch.Size([3])},
            "dtype_dict": {"sliding_window": torch.bfloat16},
            "block_size": 4,
            "device": "cpu",
        }
        main_cache = DeepSeekV4PagedKVCache(
            layer_id_map,
            manager_name="main",
            request_shape_dict={"pending_kv_state": (4, 3)},
            request_dtype_dict={"pending_kv_state": torch.float32},
            **common_kwargs,
        )
        compressed_cache = DeepSeekV4PagedKVCache(
            layer_id_map,
            manager_name="compressed_csa",
            **common_kwargs,
        )

        tasks = PackedTasksBase(
            num_tasks=2,
            task_ids=["req_0", "req_1"],
            task_type=TaskType.Prefill,
            tokens=[[1] * 5, [2] * 3],
            num_tokens=8,
            new_cache_ids_list=[
                {"main": [7, 1], "compressed_csa": [3]},
                {"main": [2], "compressed_csa": [5, 6]},
            ],
        )

        main_cache.prepare_cache_prefill(tasks)
        compressed_cache.prepare_cache_prefill(tasks)

        assert main_cache.block_table["req_0"] == [7, 1]
        assert main_cache.block_table["req_1"] == [2]
        assert compressed_cache.block_table["req_0"] == [3]
        assert compressed_cache.block_table["req_1"] == [5, 6]
        assert main_cache.get_deepseek_v4_cache_slots(main_cache.seq_len_delta) == [
            7,
            2,
        ]
        accessor = main_cache.get_accessor(0)
        assert accessor.kv["pending_kv_state"].shape == (8, 4, 3)
        assert main_cache.estimate_bytes_per_block() == 72

        main_cache.request_buffer["pending_kv_state"][:, 7].fill_(1)
        main_cache.finalize_cache_all_decode(tasks)
        assert "req_0" not in main_cache.block_table
        assert "req_1" not in main_cache.block_table
        assert torch.all(main_cache.request_buffer["pending_kv_state"][:, 7] == 0)

    def test_deepseek_v4_sliding_window_paged_cache_is_one_page_ring(self):
        layer_id_map = GlobalLocalMap.from_range(0, 1)
        cache = DeepSeekV4SlidingWindowPagedKVCache(
            layer_id_map,
            manager_name="main",
            num_hot_req=2,
            max_seq_len=16,
            window_size=4,
            num_blocks=2,
            shape_per_token_dict={"sliding_window": torch.Size([3])},
            dtype_dict={"sliding_window": torch.bfloat16},
            request_shape_dict={"pending_score_state": (4, 3)},
            request_dtype_dict={"pending_score_state": torch.float32},
            block_size=4,
            device="cpu",
        )

        tasks = PackedTasksBase(
            num_tasks=1,
            task_ids=["req_0"],
            task_type=TaskType.Prefill,
            tokens=[[1] * 6],
            num_tokens=6,
            new_cache_ids_list=[{"main": [1]}],
        )

        cache.prepare_cache_prefill(tasks)

        assert cache.block_size == 4
        assert cache.max_seq_len == 16
        assert cache.page_table_max_seq_len == 4
        assert cache.max_blocks_per_req == 1
        assert cache.max_num_blocks == 2
        assert cache.block_table["req_0"] == [1]
        assert cache.paged_kv_cache["sliding_window"].shape == (1, 2, 4, 3)
        assert cache.request_buffer["pending_score_state"].shape == (1, 2, 4, 3)
        assert cache.estimate_bytes_per_block() == 72
        assert cache.get_deepseek_v4_cache_slots(cache.seq_len_delta) == [1]
        assert cache.get_accessor(0).kv["pending_score_state"].shape == (2, 4, 3)
        assert cache.page_ids.tolist() == [1, 1, 1, 1, 1, 1]
        assert cache.offs_in_page.tolist() == [0, 1, 2, 3, 0, 1]

    def test_sliding_window_append_keeps_final_window_without_duplicate_offsets(self):
        kv_cache = torch.zeros((2, 4, 1), dtype=torch.float32)
        page_table = torch.tensor([[1]], dtype=torch.int32)
        values = torch.arange(6, dtype=torch.float32).view(6, 1)
        positions = torch.arange(6, dtype=torch.long)
        seq_ids = torch.zeros(6, dtype=torch.long)

        append_to_sliding_window_paged_kv_cache(
            kv_cache,
            page_table,
            values,
            positions,
            seq_ids,
            window_size=4,
            final_lens=torch.tensor([6], dtype=torch.long),
            impl="torch",
        )

        assert kv_cache[0].tolist() == [[0.0], [0.0], [0.0], [0.0]]
        assert kv_cache[1].tolist() == [[4.0], [5.0], [2.0], [3.0]]

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
