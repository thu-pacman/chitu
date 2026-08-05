import asyncio
import time
import types
from collections import OrderedDict
from unittest.mock import patch
from dataclasses import dataclass, field
from omegaconf import OmegaConf

from chitu.dp_request_router import RequestRouter, SchedulerStats
from chitu.task import UserRequest
from chitu.global_vars import set_global_args
from chitu.schemas.serve_config import RouterConfig
from chitu.kv_cache import BlockIdentity, NONE_BLK_HASH, BlockIdentityChainBuilder


@dataclass
class MonkReq:
    request_id: str
    prompt_tokens: list[int] = field(default_factory=list)


def _build_router(algorithm: str = "prefix_cache_aware") -> RequestRouter:
    set_global_args(
        OmegaConf.create({"infer": {"max_seq_len": 8192}}),
        need_ensure=False,
        need_preprocess=False,
    )
    cfg = RouterConfig(
        is_router=True,
        max_inflight_per_instance=24,
        routing_algorithm=algorithm,
        router_cache_miss_fallback_algorithm="least_loaded",
        router_hit_weight=1.0,
        router_load_penalty_weight=0.02,
        router_evict_buffer_size=64,
        router_local_reservation_timeout_s=600.0,
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

    router.policy.remember_request(
        MonkReq("busy", list(range(200))), local_instance_id=0
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


def test_unified_dp_keeps_running_load_until_request_finishes():
    router = _build_router(algorithm="prefix_cache_aware")
    req = MonkReq("req-unified", list(range(20)))

    router.policy.remember_request(req, local_instance_id=0)
    router.policy.forget_request(req.request_id)

    assert router.policy.get_router_load(0) == (1, 0)

    router.policy.remove_request(req.request_id)

    assert router.policy.get_router_load(0) == (0, 0)


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


def test_request_router_rejects_requests_expired_while_pending(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    router = _build_router(algorithm="least_loaded")
    router.finished_requests = []
    router.sent_requests = []

    def finish_request_before_recv_stop(request, finish_reason=None, error=None):
        router.finished_requests.append((request.request_id, finish_reason, error))
        request.finish_reason = finish_reason
        request.stop_stream(error=error)

    async def send_request(local_instance_id: int, request: UserRequest):
        router.sent_requests.append((local_instance_id, request.request_id))

    router.finish_request_before_recv_stop = finish_request_before_recv_stop
    router._send_request = send_request

    reqs = []
    for i in range(16):
        req = UserRequest.create_mock(
            input_len=4, request_id=f"req-ttft-{i}", enable_thinking=False
        )
        req.ttft_timeout_s = 0.01
        asyncio.run(router.submit_request(req))
        assert req.ttft_deadline_ts == now[0] + req.ttft_timeout_s
        reqs.append(req)

    now[0] += 1.0

    async def run_processor():
        task = asyncio.create_task(router._request_processor_task())
        try:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if all(req.finished for req in reqs):
                    return
                await asyncio.sleep(0.001)
            assert all(req.finished for req in reqs)
        finally:
            router._shutdown = True
            await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(run_processor())

    assert router.finished_requests == [
        (req.request_id, "error", "TTFT timeout") for req in reqs
    ]
    assert router.sent_requests == []
    assert all(req.async_stream.error_message == "TTFT timeout" for req in reqs)
    assert router.total_requests == len(reqs)
