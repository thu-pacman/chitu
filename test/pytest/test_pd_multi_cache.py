"""
Tests for PD disaggregation multi-cache manager support.

Covers real PagedKVCacheManager + SingletonPagedKVCacheManager code paths:
  - insert_kv_cache_from_transfer / insert_linear_state_from_transfer
  - prepare_cache_decode across multiple cache managers (the Qwen3-Next crash site)
  - finalize_cache_all_decode lifecycle
  - KeyError when any auxiliary cache is not populated (regression for the original bug)
"""

import os

import pytest
import torch

from chitu.kv_cache import (
    GlobalLocalMap,
    PagedKVCache,
    SingletonPagedKVCache,
)
from chitu.task import PackedTasksBase
from chitu.task_type import TaskType


_PD_UNIT_JOB_NAME = "pd_unit_test_h20"
_JOB_NAME = os.environ.get("CI_JOB_NAME") or os.environ.get("JOB_NAME")
if _JOB_NAME and _JOB_NAME != _PD_UNIT_JOB_NAME:
    pytest.skip("skip PD unit tests outside pd_unit_test_h20", allow_module_level=True)


# Helpers functions


def _build_paged_cache(device="cuda", num_layers=2, num_blocks=64, block_size=16):
    layer_map = GlobalLocalMap.from_range(0, num_layers)

    return PagedKVCache(
        layer_map,
        num_hot_req=16,
        max_seq_len=512,
        num_blocks=num_blocks,
        shape_per_token_dict={"kv_cache": torch.Size([2, 8])},
        dtype_dict={"kv_cache": torch.float16},
        n_local_kv_heads=2,
        head_dim=8,
        device=device,
        block_size=block_size,
    )


def _build_singleton_cache(device="cuda", num_layers=2, num_hot_req=16):
    layer_map = GlobalLocalMap.from_range(0, num_layers)

    return SingletonPagedKVCache(
        layer_map,
        num_hot_req=num_hot_req,
        shape_per_token_dict={"linear_state": torch.Size([2, 8])},
        n_local_kv_heads=2,
        head_dim=8,
        device=device,
    )


# Prepare cache decode across multiple cache managers
@pytest.mark.pd_unit
class TestMultiCachePrepareDecode:
    """Verify prepare_cache_decode works when all managers have entries,
    and raises KeyError when any manager is missing."""

    def test_main_only(self, cuda_available, global_args, init_distributed):
        main = _build_paged_cache()
        main.insert_kv_cache_from_transfer("r1", [0, 1, 2], 48)

        # 由main cache_manager分配的new_cache_ids_list
        tasks = PackedTasksBase(
            num_tasks=1,
            task_ids=["r1"],
            task_type=TaskType.Decode,
            new_cache_ids_list=[[3]],
        )

        main.prepare_cache_decode(tasks)
        assert main.block_table["r1"] == [
            0,
            1,
            2,
            3,
        ], f"{main.block_table['r1']} vs [0,1,2,3]"
        assert (
            main.tid_to_cached_len["r1"] == 49
        ), f"{main.tid_to_cached_len['r1']} vs 49"

    def test_main_linear(self, cuda_available, global_args, init_distributed):
        main = _build_paged_cache()
        linear = _build_singleton_cache()

        main.insert_kv_cache_from_transfer("r1", [0, 1, 2], 48)
        linear.insert_linear_state_from_transfer("r1", 0, 48)

        # 由 cache_manager分配的new_cache_ids_list
        tasks = PackedTasksBase(
            num_tasks=1,
            task_ids=["r1"],
            task_type=TaskType.Decode,
            new_cache_ids_list=[[3]],
        )

        main.prepare_cache_decode(tasks)
        assert main.block_table["r1"] == [
            0,
            1,
            2,
            3,
        ], f"{main.block_table['r1']} vs [0,1,2,3]"
        assert (
            main.tid_to_cached_len["r1"] == 49
        ), f"{main.tid_to_cached_len['r1']} vs 49"

        linear.prepare_cache_decode(tasks)
        assert linear.block_table["r1"] == [0], f"{linear.block_table['r1']} vs [0]"
        assert (
            linear.tid_to_cached_len["r1"] == 49
        ), f"{linear.tid_to_cached_len['r1']} vs 49"

    def test_main_indexer(self, cuda_available, global_args, init_distributed):
        main = _build_paged_cache()
        indexer = _build_paged_cache(num_blocks=32)

        main.insert_kv_cache_from_transfer("r1", [0, 1], 32)
        indexer.insert_kv_cache_from_transfer("r1", [0, 1], 32)

        # 由 cache_manager分配的new_cache_ids_list
        tasks = PackedTasksBase(
            num_tasks=1,
            task_ids=["r1"],
            task_type=TaskType.Decode,
            new_cache_ids_list=[[2]],
        )

        for cache in [main, indexer]:
            cache.prepare_cache_decode(tasks)
            assert cache.block_table["r1"] == [
                0,
                1,
                2,
            ], f"{type(cache).__name__}: {cache.block_table['r1']} vs [0,1,2]"
            assert (
                cache.tid_to_cached_len["r1"] == 33
            ), f"{type(cache).__name__}: {cache.tid_to_cached_len['r1']} vs 33"

    def test_triple_cache(self, cuda_available, global_args, init_distributed):
        main = _build_paged_cache()
        linear = _build_singleton_cache()
        indexer = _build_paged_cache(num_blocks=32)

        # insert kv cache req a
        main.insert_kv_cache_from_transfer("a", [0, 1], 32)
        linear.insert_linear_state_from_transfer("a", 0, 32)
        indexer.insert_kv_cache_from_transfer("a", [0, 1], 32)

        # insert kv cache req b
        main.insert_kv_cache_from_transfer("b", [2, 3], 32)
        linear.insert_linear_state_from_transfer("b", 1, 32)
        indexer.insert_kv_cache_from_transfer("b", [2, 3], 32)

        # 由 cache_manager分配的new_cache_ids_list
        tasks = PackedTasksBase(
            num_tasks=2,
            task_ids=["a", "b"],
            task_type=TaskType.Decode,
            new_cache_ids_list=[[4], [5]],
        )

        for cache in [main, indexer]:
            cache.prepare_cache_decode(tasks)
            assert cache.block_table["a"] == [
                0,
                1,
                4,
            ], f"{type(cache).__name__}: {cache.block_table['a']} vs [0,1,4]"
            assert (
                cache.tid_to_cached_len["a"] == 33
            ), f"{type(cache).__name__}: {cache.tid_to_cached_len['a']} vs 33"
            assert cache.block_table["b"] == [
                2,
                3,
                5,
            ], f"{type(cache).__name__}: {cache.block_table['b']} vs [2,3,5]"
            assert (
                cache.tid_to_cached_len["b"] == 33
            ), f"{type(cache).__name__}: {cache.tid_to_cached_len['b']} vs 33"

        linear.prepare_cache_decode(tasks)
        assert linear.block_table["a"] == [0], f"{linear.block_table['a']} vs [0]"
        assert (
            linear.tid_to_cached_len["a"] == 33
        ), f"{linear.tid_to_cached_len['a']} vs 33"
        assert linear.block_table["b"] == [1], f"{linear.block_table['a']} vs [1]"
        assert (
            linear.tid_to_cached_len["b"] == 33
        ), f"{linear.tid_to_cached_len['b']} vs 33"

    def test_linear_missing_raises(self, cuda_available, global_args, init_distributed):
        """Qwen3-Next regression: linear cache not populated."""
        main = _build_paged_cache()
        linear = _build_singleton_cache()

        main.insert_kv_cache_from_transfer("r1", [0, 1, 2], 48)

        # 由 cache_manager分配的new_cache_ids_list
        tasks = PackedTasksBase(
            num_tasks=1,
            task_ids=["r1"],
            task_type=TaskType.Decode,
            new_cache_ids_list=[[3]],
        )

        main.prepare_cache_decode(tasks)
        with pytest.raises(KeyError):
            linear.prepare_cache_decode(tasks)

    def test_indexer_missing_raises(
        self, cuda_available, global_args, init_distributed
    ):
        main = _build_paged_cache()
        indexer = _build_paged_cache(num_blocks=32)

        main.insert_kv_cache_from_transfer("r1", [0, 1], 32)

        # 由 cache_manager分配的new_cache_ids_list
        tasks = PackedTasksBase(
            num_tasks=1,
            task_ids=["r1"],
            task_type=TaskType.Decode,
            new_cache_ids_list=[[2]],
        )

        main.prepare_cache_decode(tasks)
        with pytest.raises(KeyError):
            indexer.prepare_cache_decode(tasks)

    def test_multi_request_batch(self, cuda_available, global_args, init_distributed):
        main = _build_paged_cache(num_blocks=128)
        linear = _build_singleton_cache()

        req_ids = [f"req_{i:03d}" for i in range(8)]
        for i, rid in enumerate(req_ids):
            main.insert_kv_cache_from_transfer(rid, [i * 2, i * 2 + 1], 32)
            linear.insert_linear_state_from_transfer(rid, i, 32)

        # 由 cache_manager分配的new_cache_ids_list
        tasks = PackedTasksBase(
            num_tasks=len(req_ids),
            task_ids=req_ids,
            task_type=TaskType.Decode,
            new_cache_ids_list=[[16 + i] for i in range(len(req_ids))],
        )

        main.prepare_cache_decode(tasks)
        for idx, rid in enumerate(req_ids):
            assert main.block_table[rid] == [
                idx * 2,
                idx * 2 + 1,
                16 + idx,
            ], f"{main.block_table[rid]} vs {[idx * 2, idx * 2 + 1, 16+idx]}"
            assert (
                main.tid_to_cached_len[rid] == 33
            ), f"{main.tid_to_cached_len[rid]} vs 33"

        linear.prepare_cache_decode(tasks)
        for idx, rid in enumerate(req_ids):
            assert linear.block_table[rid] == [
                idx
            ], f"{linear.block_table[rid]} vs {[idx]}"
            assert (
                linear.tid_to_cached_len[rid] == 33
            ), f"{linear.tid_to_cached_len[rid]} vs 33"


# Insert → prepare → finalize


@pytest.mark.pd_unit
class TestMultiCacheLifecycle:
    """Full lifecycle across multiple cache managers."""

    def test_main_linear_lifecycle(self, cuda_available, global_args, init_distributed):
        main = _build_paged_cache()
        linear = _build_singleton_cache()

        main.insert_kv_cache_from_transfer("r1", [0, 1], 32)
        linear.insert_linear_state_from_transfer("r1", 0, 32)

        # 由 cache_manager分配的new_cache_ids_list
        tasks = PackedTasksBase(
            num_tasks=1,
            task_ids=["r1"],
            task_type=TaskType.Decode,
            new_cache_ids_list=[[2]],
        )

        for cache in [main, linear]:
            cache.prepare_cache_decode(tasks)
        for cache in [main, linear]:
            cache.finalize_cache_all_decode(tasks)

        assert "r1" not in main.tid_to_cached_len
        assert "r1" not in linear.tid_to_cached_len

    def test_two_batches_sequential(
        self, cuda_available, global_args, init_distributed
    ):
        """batch-1 decode + finalize, then batch-2 arrives."""
        main = _build_paged_cache()
        linear = _build_singleton_cache()
        caches = [main, linear]

        for rid in ["r1", "r2"]:
            main.insert_kv_cache_from_transfer(rid, [0], 16)
            linear.insert_linear_state_from_transfer(rid, 0, 16)

        # 由 cache_manager分配的new_cache_ids_list
        tasks = PackedTasksBase(
            num_tasks=2,
            task_ids=["r1", "r2"],
            task_type=TaskType.Decode,
            new_cache_ids_list=[[1], [2]],
        )

        for cache in caches:
            cache.prepare_cache_decode(tasks)

        for cache in caches:
            cache.finalize_cache_all_decode(tasks)

        for rid in ["r3", "r4"]:
            main.insert_kv_cache_from_transfer(rid, [1], 16)
            linear.insert_linear_state_from_transfer(rid, 1, 16)

        # 由 cache_manager分配的new_cache_ids_list
        tasks = PackedTasksBase(
            num_tasks=2,
            task_ids=["r3", "r4"],
            task_type=TaskType.Decode,
            new_cache_ids_list=[[2], [3]],
        )

        for cache in caches:
            cache.prepare_cache_decode(tasks)

        for cache in caches:
            cache.finalize_cache_all_decode(tasks)

        assert len(main.tid_to_cached_len) == 0
        assert len(linear.tid_to_cached_len) == 0

    def test_triple_cache_lifecycle(
        self, cuda_available, global_args, init_distributed
    ):
        main = _build_paged_cache()
        linear = _build_singleton_cache()
        indexer = _build_paged_cache(num_blocks=32)
        caches = [main, linear, indexer]

        req_ids = ["a", "b", "c"]
        for i, rid in enumerate(req_ids):
            main.insert_kv_cache_from_transfer(rid, [i], 16)
            linear.insert_linear_state_from_transfer(rid, i, 16)
            indexer.insert_kv_cache_from_transfer(rid, [i], 16)

        # 由 cache_manager分配的new_cache_ids_list
        tasks = PackedTasksBase(
            num_tasks=3,
            task_ids=["a", "b", "c"],
            task_type=TaskType.Decode,
            new_cache_ids_list=[[3], [4], [5]],
        )

        for cache in caches:
            cache.prepare_cache_decode(tasks)

        for cache in caches:
            cache.finalize_cache_all_decode(tasks)

        for cache in caches:
            assert len(cache.tid_to_cached_len) == 0
            assert len(cache.block_table) == 0

    def test_partial_finalize(self, cuda_available, global_args, init_distributed):
        """Finalize some requests while others stay active."""
        main = _build_paged_cache()
        linear = _build_singleton_cache()
        caches = [main, linear]

        for i, rid in enumerate(["r1", "r2", "r3"]):
            main.insert_kv_cache_from_transfer(rid, [i], 16)
            linear.insert_linear_state_from_transfer(rid, i, 16)

        # 由 cache_manager分配的new_cache_ids_list
        bach1_tasks = PackedTasksBase(
            num_tasks=1,
            task_ids=["r1"],
            task_type=TaskType.Decode,
            new_cache_ids_list=[[3]],
        )

        for cache in caches:
            cache.prepare_cache_decode(bach1_tasks)

        for cache in caches:
            cache.finalize_cache_all_decode(bach1_tasks)

        assert "r1" not in main.tid_to_cached_len
        assert "r2" in main.tid_to_cached_len

        # 由 cache_manager分配的new_cache_ids_list
        bach2_tasks = PackedTasksBase(
            num_tasks=1,
            task_ids=["r2", "r3"],
            task_type=TaskType.Decode,
            new_cache_ids_list=[[4], [5]],
        )

        for cache in caches:
            cache.prepare_cache_decode(bach2_tasks)
