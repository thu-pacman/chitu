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

from chitu.cache_manager import (
    GlobalLocalMap,
    PagedKVCacheManager,
    SingletonPagedKVCacheManager,
)


_PD_UNIT_JOB_NAME = "pd_unit_test_h20"
_JOB_NAME = os.environ.get("CI_JOB_NAME") or os.environ.get("JOB_NAME")
if _JOB_NAME and _JOB_NAME != _PD_UNIT_JOB_NAME:
    pytest.skip("skip PD unit tests outside pd_unit_test_h20", allow_module_level=True)


# Helpers functions


def _build_paged_cm(device="cuda", num_layers=2, num_blocks=64, block_size=16):
    layer_map = GlobalLocalMap.from_range(0, num_layers)
    return PagedKVCacheManager(
        layer_map,
        num_hot_req=16,
        max_seq_len=512,
        shape_per_token_dict={"kv_cache": torch.Size([2, 8])},
        dtype_dict={"kv_cache": torch.float16},
        n_local_kv_heads=2,
        head_dim=8,
        device=device,
        block_size=block_size,
        num_blocks=num_blocks,
    )


def _build_singleton_cm(device="cuda", num_layers=2, num_hot_req=16):
    layer_map = GlobalLocalMap.from_range(0, num_layers)
    return SingletonPagedKVCacheManager(
        layer_map,
        num_hot_req=num_hot_req,
        shape_per_token_dict={"linear_state": torch.Size([2, 8])},
        dtype_dict={"linear_state": torch.float16},
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
        main = _build_paged_cm()
        main.insert_kv_cache_from_transfer("r1", [0, 1, 2], 48)
        main.prepare_cache_decode(["r1"])

    def test_main_linear(self, cuda_available, global_args, init_distributed):
        main = _build_paged_cm()
        linear = _build_singleton_cm()

        main.insert_kv_cache_from_transfer("r1", [0, 1, 2], 48)
        linear.insert_linear_state_from_transfer("r1", 0)

        for mgr in [main, linear]:
            mgr.prepare_cache_decode(["r1"])

    def test_main_indexer(self, cuda_available, global_args, init_distributed):
        main = _build_paged_cm()
        indexer = _build_paged_cm(num_blocks=32)

        main.insert_kv_cache_from_transfer("r1", [0, 1], 32)
        indexer.insert_kv_cache_from_transfer("r1", [0, 1], 32)

        for mgr in [main, indexer]:
            mgr.prepare_cache_decode(["r1"])

    def test_triple_cache(self, cuda_available, global_args, init_distributed):
        main = _build_paged_cm()
        linear = _build_singleton_cm()
        indexer = _build_paged_cm(num_blocks=32)

        for rid in ["a", "b"]:
            main.insert_kv_cache_from_transfer(rid, [0, 1], 32)
            linear.insert_linear_state_from_transfer(rid, 0)
            indexer.insert_kv_cache_from_transfer(rid, [0, 1], 32)

        for mgr in [main, linear, indexer]:
            mgr.prepare_cache_decode(["a", "b"])

    def test_linear_missing_raises(self, cuda_available, global_args, init_distributed):
        """Qwen3-Next regression: linear cache not populated."""
        main = _build_paged_cm()
        linear = _build_singleton_cm()

        main.insert_kv_cache_from_transfer("r1", [0, 1, 2], 48)

        main.prepare_cache_decode(["r1"])
        with pytest.raises(KeyError):
            linear.prepare_cache_decode(["r1"])

    def test_indexer_missing_raises(
        self, cuda_available, global_args, init_distributed
    ):
        main = _build_paged_cm()
        indexer = _build_paged_cm(num_blocks=32)

        main.insert_kv_cache_from_transfer("r1", [0, 1], 32)

        main.prepare_cache_decode(["r1"])
        with pytest.raises(KeyError):
            indexer.prepare_cache_decode(["r1"])

    def test_multi_request_batch(self, cuda_available, global_args, init_distributed):
        main = _build_paged_cm(num_blocks=128)
        linear = _build_singleton_cm()

        req_ids = [f"req_{i:03d}" for i in range(8)]
        for i, rid in enumerate(req_ids):
            main.insert_kv_cache_from_transfer(rid, [i * 2, i * 2 + 1], 32)
            linear.insert_linear_state_from_transfer(rid, i)

        for mgr in [main, linear]:
            mgr.prepare_cache_decode(req_ids)


# Insert → prepare → finalize


@pytest.mark.pd_unit
class TestMultiCacheLifecycle:
    """Full lifecycle across multiple cache managers."""

    def test_main_linear_lifecycle(self, cuda_available, global_args, init_distributed):
        main = _build_paged_cm()
        linear = _build_singleton_cm()

        main.insert_kv_cache_from_transfer("r1", [0, 1], 32)
        linear.insert_linear_state_from_transfer("r1", 0)

        for mgr in [main, linear]:
            mgr.prepare_cache_decode(["r1"])
        for mgr in [main, linear]:
            mgr.finalize_cache_all_decode("r1")

        assert "r1" not in main.req_id_to_seq_len
        assert "r1" not in linear.req_id_to_seq_len

    def test_two_batches_sequential(
        self, cuda_available, global_args, init_distributed
    ):
        """batch-1 decode + finalize, then batch-2 arrives."""
        main = _build_paged_cm()
        linear = _build_singleton_cm()
        managers = [main, linear]

        for rid in ["r1", "r2"]:
            main.insert_kv_cache_from_transfer(rid, [0], 16)
            linear.insert_linear_state_from_transfer(rid, 0)
        for mgr in managers:
            mgr.prepare_cache_decode(["r1", "r2"])
        for rid in ["r1", "r2"]:
            for mgr in managers:
                mgr.finalize_cache_all_decode(rid)

        for rid in ["r3", "r4"]:
            main.insert_kv_cache_from_transfer(rid, [1], 16)
            linear.insert_linear_state_from_transfer(rid, 1)
        for mgr in managers:
            mgr.prepare_cache_decode(["r3", "r4"])
        for rid in ["r3", "r4"]:
            for mgr in managers:
                mgr.finalize_cache_all_decode(rid)

        assert len(main.req_id_to_seq_len) == 0
        assert len(linear.req_id_to_seq_len) == 0

    def test_triple_cache_lifecycle(
        self, cuda_available, global_args, init_distributed
    ):
        main = _build_paged_cm()
        linear = _build_singleton_cm()
        indexer = _build_paged_cm(num_blocks=32)
        managers = [main, linear, indexer]

        req_ids = ["a", "b", "c"]
        for i, rid in enumerate(req_ids):
            main.insert_kv_cache_from_transfer(rid, [i], 16)
            linear.insert_linear_state_from_transfer(rid, i)
            indexer.insert_kv_cache_from_transfer(rid, [i], 16)

        for mgr in managers:
            mgr.prepare_cache_decode(req_ids)
        for rid in req_ids:
            for mgr in managers:
                mgr.finalize_cache_all_decode(rid)

        for mgr in managers:
            assert len(mgr.req_id_to_seq_len) == 0

    def test_partial_finalize(self, cuda_available, global_args, init_distributed):
        """Finalize some requests while others stay active."""
        main = _build_paged_cm()
        linear = _build_singleton_cm()
        managers = [main, linear]

        for i, rid in enumerate(["r1", "r2", "r3"]):
            main.insert_kv_cache_from_transfer(rid, [i], 16)
            linear.insert_linear_state_from_transfer(rid, i)
        for mgr in managers:
            mgr.prepare_cache_decode(["r1", "r2", "r3"])

        for mgr in managers:
            mgr.finalize_cache_all_decode("r1")
        assert "r1" not in main.req_id_to_seq_len
        assert "r2" in main.req_id_to_seq_len

        for mgr in managers:
            mgr.prepare_cache_decode(["r2", "r3"])
