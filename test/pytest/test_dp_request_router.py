import types
from collections import OrderedDict
from unittest.mock import patch

from chitu.dp_request_router import RequestRouter, SchedulerStats
from chitu.schemas.serve_config import InstAddressesConfig, RouterConfig
from chitu.kv_cache import BlockIdentity, NONE_BLK_HASH, BlockIdentityChainBuilder
from dataclasses import dataclass, field


@dataclass
class MonkReq:
    request_id: str
    prompt_tokens: list[int] = field(default_factory=list)


def _build_router(algorithm: str = "prefix_cache_aware") -> RequestRouter:
    cfg = RouterConfig(
        is_router=True,
        stats_port=29600,
        token_port=29700,
        max_inflight_per_instance=24,
        routing_algorithm=algorithm,
        router_cache_miss_fallback_algorithm="least_loaded",
        router_hit_weight=1.0,
        router_load_penalty_weight=0.02,
        inst_addresses=[
            InstAddressesConfig(host="127.0.0.1", port=30000),
            InstAddressesConfig(host="127.0.0.1", port=30001),
        ],
    )
    router = RequestRouter(cfg)
    return router


def test_hit_aware_select_scheduler_prefers_instance_with_more_prefix_hits():
    router = _build_router(algorithm="prefix_cache_aware")

    # mark two schedulers alive
    router.policy.update_stats(
        SchedulerStats(
            local_instance_id=0,
            running_requests=20,  # higher load on purpose
            waiting_requests=0,
            pending_tokens=0,
            throughput_tokens_per_sec=0.0,
            last_update_time=0.0,
            is_alive=True,
            num_blocks=8,
            block_size=4,
        )
    )
    router.policy.update_stats(
        SchedulerStats(
            local_instance_id=1,
            running_requests=0,
            waiting_requests=0,
            pending_tokens=0,
            throughput_tokens_per_sec=0.0,
            last_update_time=0.0,
            is_alive=True,
            num_blocks=8,
            block_size=4,
        )
    )

    # request has two full blocks: [1,2,3,4], [5,6,7,8]
    req = MonkReq("req-hit", [1, 2, 3, 4, 5, 6, 7, 8])
    req_blocks = router.policy.build_req_token_blocks(req, local_instance_id=0)
    assert len(req_blocks) == 2
    # instance 0 hits two blocks; instance 1 hits zero.
    router.policy.remember_request(req, local_instance_id=0)
    router.policy.insert_req_blocks(req.request_id)

    selected = router.policy.select_scheduler(req)
    assert selected == 0


def test_select_scheduler_falls_back_to_lb_when_all_hit_zero():
    router = _build_router(algorithm="prefix_cache_aware")
    router.policy.update_stats(
        SchedulerStats(
            local_instance_id=0,
            running_requests=3,
            waiting_requests=0,
            pending_tokens=200,
            throughput_tokens_per_sec=0.0,
            last_update_time=0.0,
            is_alive=True,
            num_blocks=8,
            block_size=4,
        )
    )
    router.policy.update_stats(
        SchedulerStats(
            local_instance_id=1,
            running_requests=1,
            waiting_requests=0,
            pending_tokens=0,
            throughput_tokens_per_sec=0.0,
            last_update_time=0.0,
            is_alive=True,
            num_blocks=8,
            block_size=4,
        )
    )

    req = MonkReq("req-no-hit", [11, 12, 13, 14])
    # no cached blocks on both instances => fallback to least_loaded (scheduler 1)
    selected = router.policy.select_scheduler(req)
    assert selected == 1


def test_remember_req_and_insert_req_blocks_and_forget_req():
    router = _build_router(algorithm="prefix_cache_aware")
    router.policy.instances_num_total_blocks[0] = 2
    router.policy.instances_block_size[0] = 4

    req = MonkReq("req-test", [1, 2, 3, 4, 5, 6, 7, 8])
    router.policy.remember_request(req, local_instance_id=0)
    assert req.request_id in router.policy.req_to_request
    assert router.policy.req_to_scheduler[req.request_id] == 0

    router.policy.insert_req_blocks(req.request_id)
    assert len(router.policy.cached_blocks[0]) == 2

    router.policy.forget_request(req.request_id)
    assert req.request_id not in router.policy.req_to_request
    assert req.request_id not in router.policy.req_to_scheduler


def test_local_evict_moves_hash_to_buffer():
    router = _build_router(algorithm="prefix_cache_aware")
    router.policy.instances_num_total_blocks[0] = 1
    router.policy.instances_block_size[0] = 4

    req = MonkReq("req-evict", [1, 2, 3, 4, 5, 6, 7, 8])
    req_blocks = router.policy.build_req_token_blocks(req, local_instance_id=0)
    router.policy.remember_request(req, local_instance_id=0)
    router.policy.insert_req_blocks(req.request_id)

    # first block should be evicted from cached_blocks and moved to recycle pool
    assert req_blocks[0].blk_hash not in router.policy.cached_blocks[0]
    assert req_blocks[0].blk_hash in router.policy.evict_buffer[0]


def test_apply_instance_evicts():
    router = _build_router(algorithm="prefix_cache_aware")

    block = BlockIdentityChainBuilder.acquire(
        manager_name="router", block_size=4
    ).make_identity([1, 2, 3, 4], pre_blk_hash=NONE_BLK_HASH)

    assert block.blk_hash is not None

    # remove from cached_blocks
    router.policy.cached_blocks[0] = OrderedDict()
    router.policy.cached_blocks[0][block.blk_hash] = block
    router.policy.apply_instance_evicts(0, [block.blk_hash])
    assert block.blk_hash not in router.policy.cached_blocks[0]

    # remove from evict_buffer
    router.policy.evict_buffer[0] = OrderedDict()
    router.policy.evict_buffer[0][block.blk_hash] = 1.0
    router.policy.apply_instance_evicts(0, [block.blk_hash])
    assert block.blk_hash not in router.policy.evict_buffer[0]


def test_evict_buffer_keeps_fixed_capacity():
    router = _build_router(algorithm="prefix_cache_aware")
    router.policy.evict_buffer_size = 2
    router.policy._push_evict_buffer(0, "h1")
    router.policy._push_evict_buffer(0, "h2")
    router.policy._push_evict_buffer(0, "h3")

    assert "h1" not in router.policy.evict_buffer[0]
    assert "h2" in router.policy.evict_buffer[0]
    assert "h3" in router.policy.evict_buffer[0]
