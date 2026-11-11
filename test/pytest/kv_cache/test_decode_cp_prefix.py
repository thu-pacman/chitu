"""PD full-prompt prefix sharing: ownership, lifetime, and transfer contracts."""

import pytest


from dataclasses import dataclass
from types import SimpleNamespace

from omegaconf import OmegaConf

from chitu.backend import Backend
from chitu.global_vars import set_global_args
from chitu.kv_cache import PagedKVCacheManager
from chitu.kv_cache.prefix_caching import BlockIdentityChainBuilder
from chitu.distributed.pd_disaggregation.pd_scheduler import (
    _collect_router_kv_cache_stats,
)
from chitu.scheduler import Scheduler
from chitu.task import TaskStatus
from chitu.task_type import TaskType


@pytest.fixture(autouse=True)
def _grouped_prefix_args(monkeypatch):
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_seq_len": 1024,
                    "max_batch_size": 8,
                    "prefill_chunk_size": None,
                    "mtp_size": 1,
                    "dp_size": 1,
                    "pcp_size": 1,
                    "use_cuda_graph": False,
                    "schedule_overlap": True,
                    "op_impl": "cpu",
                    "enable_prefix_caching": True,
                },
                "multi_inst": {"role": "decode", "n_insts": 1},
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    monkeypatch.setattr(BlockIdentityChainBuilder, "_registry", {})


@dataclass
class _PrefixTask:
    task_id: str
    prefix_tokens: list[int]
    consumed_req_tokens: int = 0
    cached_seq_len: int = 0
    task_type: TaskType = TaskType.Prefill

    @property
    def prefix_tokens_len(self):
        return len(self.prefix_tokens)

    @property
    def alloc_seq_len(self):
        return len(self.prefix_tokens) + 1


def _group_manager(size=4):
    return PagedKVCacheManager(
        16,
        num_hot_req=8,
        max_seq_len=1024,
        dp_rank=0,
        dcp_size=size,
        block_size=2,
        enable_prefix_caching=True,
    )


@pytest.mark.parametrize("size", [1, 4])
def test_router_prefix_hits(monkeypatch, size):
    cm = _group_manager(size=size)
    first = _PrefixTask("first", list(range(15)))
    cm.prepare_metadata_before_prefill(first, eager_prefix_cache_insert=False)
    cm.publish_prefix_cache_blocks(first, 15)
    replay = _PrefixTask("replay", list(range(15)))
    replay.status = TaskStatus.PDDecodeIncoming
    scheduler = SimpleNamespace(cache_manager_dict={"main": cm})
    expected_cached_tokens = {1: 14, 2: 12, 4: 8}[size]
    assert (
        Scheduler._num_prefill_cached_tokens(scheduler, replay)
        == expected_cached_tokens
    )

    monkeypatch.setattr(Backend, "cache_managers", [{"main": cm}])
    stats = _collect_router_kv_cache_stats()
    assert stats["block_size"] == 2 * size
    router_builder = BlockIdentityChainBuilder.acquire("router", stats["block_size"])
    router_identities = router_builder.make_identity_chain(replay)
    manager_identities = cm.identity_builder.make_identity_chain(replay)
    assert [identity.blk_hash for identity in router_identities] == [
        identity.blk_hash for identity in manager_identities
    ]
