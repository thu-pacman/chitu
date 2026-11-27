import asyncio
import time
import types
from collections import OrderedDict
from unittest.mock import patch
from dataclasses import dataclass, field
import pytest
from omegaconf import OmegaConf

from chitu.dp_request_router import RequestRouter, SchedulerStats
from chitu.finish_reason import FinishReason
from chitu.task import UserRequest
from chitu.global_vars import set_global_args
from chitu.schemas.serve_config import RouterConfig
from chitu.kv_cache import BlockIdentityChainBuilder


@dataclass
class MonkReq:
    request_id: str
    prompt_tokens: list[int] = field(default_factory=list)


def _build_router(
    algorithm: str = "prefix_cache_aware",
    *,
    routing_by_req_len: bool = False,
    max_seq_lens: list[int] | None = None,
) -> RequestRouter:
    if max_seq_lens is None:
        max_seq_lens = [8192]
    set_global_args(
        OmegaConf.create(
            {
                "infer": {"max_seq_len": 8192, "enable_prefix_caching": True},
                "multi_inst": {"n_insts": len(max_seq_lens)},
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    cfg = RouterConfig(
        is_router=True,
        max_inflight_per_instance=24,
        routing_algorithm=algorithm,
        routing_algorithm_for_decode="prefix_cache_aware",
        router_cache_threshold=0.5,
        router_balance_abs_threshold=32,
        router_balance_rel_threshold=1.1,
        router_evict_buffer_size=64,
        router_local_reservation_timeout_s=600.0,
        launch_timeout=3600.0,
        routing_by_req_len=routing_by_req_len,
        decode_routing_by_req_len=False,
    )
    # ``_parse_scheduler_configs`` reads per-instance ``infer.max_seq_len`` from
    # the full resolved config. Building that config here would require many
    # unrelated fields, so drive the per-instance length directly.
    with patch.object(
        RequestRouter,
        "_scheduler_max_seq_len",
        lambda self, inst_id: max_seq_lens[inst_id],
    ):
        router = RequestRouter(cfg)
    return router


def _mark_schedulers_alive(router: RequestRouter, count: int) -> None:
    for local_instance_id in range(count):
        router.policy.update_stats(
            SchedulerStats(
                local_instance_id=local_instance_id,
                running_requests=0,
                waiting_requests=0,
                pending_tokens=0,
                throughput_tokens_per_sec=0.0,
                last_update_time=0.0,
                is_alive=True,
            )
        )


def test_hit_aware_select_scheduler_prefers_instance_with_more_prefix_hits():
    router = _build_router(algorithm="prefix_cache_aware")

    # mark two schedulers alive
    for local_instance_id in (0, 1):
        router.policy.update_stats(
            SchedulerStats(
                local_instance_id=local_instance_id,
                running_requests=0,
                waiting_requests=0,
                pending_tokens=0,
                throughput_tokens_per_sec=0.0,
                last_update_time=0.0,
                is_alive=True,
                num_blocks=64,
                block_size=4,
            )
        )

    # 16 full blocks of 4 tokens. Both score terms are prompt tokens, so the
    # prompt has to be long enough for the cache hit to be worth more than the
    # load the routed request itself puts on instance 0.
    req = MonkReq("req-hit", list(range(64)))
    req_block_hashes = router.policy.build_req_block_hashes(req, local_instance_id=0)
    assert len(req_block_hashes) == 16
    # instance 0 hits every block; instance 1 hits zero.
    router.policy.remember_request(req, local_instance_id=0)
    router.policy.insert_req_blocks(req.request_id)
    # Load is the router's own reservation, not the instance-reported figure,
    # which workers hard-code to 0.
    assert router.policy.get_router_load(0) == (1, 64)

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
    # no useful prefix affinity => two-gate seeds on the least-loaded scheduler.
    selected = router.policy.select_scheduler(req)
    assert selected == 1


def test_prefix_routing_checks_first_block_at_each_cache_block_size():
    policy = _build_router().policy
    req = MonkReq("mixed-block-sizes", list(range(65)))
    policy.instances_block_size.update({0: 4, 1: 8, 2: 4})
    for sid in (0, 1, 2):
        policy.cached_blocks[sid] = OrderedDict.fromkeys(["unrelated"])
    hashes = policy.build_req_block_hashes(req, 1)
    policy.cached_blocks[1].update(OrderedDict.fromkeys(hashes[:-1]))
    with patch.object(
        policy, "build_req_block_hashes", wraps=policy.build_req_block_hashes
    ) as build:
        assert policy.select_scheduler(req, eligible_ids=[0, 1, 2]) == 1
        build.assert_called_once_with(req, 1)


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


def test_unified_length_buckets_follow_instance_limits():
    router = _build_router(routing_by_req_len=True, max_seq_lens=[32, 8, 16])
    _mark_schedulers_alive(router, 3)
    assert router.prefill_and_decode_schedulers == {
        0: {"global_instance_id": 0, "max_seq_len": 32},
        1: {"global_instance_id": 1, "max_seq_len": 8},
        2: {"global_instance_id": 2, "max_seq_len": 16},
    }

    def candidates(req_seq_len: int) -> list[int]:
        return router._length_routing_candidates(
            policy=router.policy,
            schedulers=router.prefill_and_decode_schedulers,
            req_seq_len=req_seq_len,
        )

    # Smallest compatible bucket wins; a request longer than every instance is
    # not admitted at all.
    assert candidates(8) == [1]
    assert candidates(9) == [2]
    assert candidates(17) == [0]
    assert candidates(33) == []


def test_unified_length_bucket_spills_over_when_full():
    router = _build_router(routing_by_req_len=True, max_seq_lens=[8, 16])
    _mark_schedulers_alive(router, 2)
    policy = router.policy
    policy.max_inflight_per_scheduler = 1
    policy.remember_request(MonkReq("busy"), local_instance_id=0)

    assert router._length_routing_candidates(
        policy=policy,
        schedulers=router.prefill_and_decode_schedulers,
        req_seq_len=8,
    ) == [1]


def test_unified_length_bucket_reaps_stale_reservation():
    router = _build_router(routing_by_req_len=True, max_seq_lens=[8])
    _mark_schedulers_alive(router, 1)
    policy = router.policy
    policy.max_inflight_per_scheduler = 1
    policy.remember_request(MonkReq("stale"), local_instance_id=0)
    policy.router_reservations["stale"].reserved_at = (
        time.monotonic() - policy.router_reservation_timeout_s - 1
    )

    # The only compatible bucket looks full until the expired reservation is
    # reaped; without the reap this path returns no candidates and the request
    # is requeued forever.
    assert router._length_routing_candidates(
        policy=policy,
        schedulers=router.prefill_and_decode_schedulers,
        req_seq_len=8,
    ) == [0]
    assert "stale" not in policy.router_reservations


def test_unified_router_dispatches_by_prompt_plus_max_new_tokens():
    router = _build_router(routing_by_req_len=True, max_seq_lens=[16, 8])
    _mark_schedulers_alive(router, 2)
    sent: list[tuple[int, str]] = []

    async def send_request(local_instance_id: int, request: UserRequest):
        sent.append((local_instance_id, request.request_id))

    router._send_request = send_request

    # Both prompts fit the 8-token bucket, only the output budgets differ, so
    # the second request must use the 16-token bucket.
    short = UserRequest.create_mock(input_len=7, request_id="len-8", max_new_tokens=1)
    long = UserRequest.create_mock(input_len=7, request_id="len-16", max_new_tokens=9)

    async def run():
        router._shutdown = False
        task = asyncio.create_task(router._request_processor_task())
        try:
            await router.submit_request(short)
            await router.submit_request(long)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and len(sent) < 2:
                await asyncio.sleep(0.001)
        finally:
            router._shutdown = True
            await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(run())

    assert sorted(sent) == [(0, "len-16"), (1, "len-8")]


@pytest.mark.parametrize(
    "max_new_tokens,rejected",
    [
        # prompt 7 + 1 == the only instance's limit: admitted.
        (1, False),
        # prompt 7 + 2 exceeds it: no bucket can ever accept the request.
        (2, True),
    ],
)
def test_unified_router_rejects_only_unroutable_lengths(max_new_tokens, rejected):
    router = _build_router(routing_by_req_len=True, max_seq_lens=[8])
    _mark_schedulers_alive(router, 1)
    sent: list[tuple[int, str]] = []
    finished: list[tuple[str, FinishReason]] = []

    async def send_request(local_instance_id: int, request: UserRequest):
        sent.append((local_instance_id, request.request_id))

    def finish_request_before_recv_stop(
        request: UserRequest, finish_reason: FinishReason, error=None
    ):
        finished.append((request.request_id, finish_reason))
        request.stop_stream(finish_reason, error=error)

    router._send_request = send_request
    router.finish_request_before_recv_stop = finish_request_before_recv_stop

    request = UserRequest.create_mock(
        input_len=7, request_id="len-boundary", max_new_tokens=max_new_tokens
    )

    async def run():
        router._shutdown = False
        task = asyncio.create_task(router._request_processor_task())
        try:
            await router.submit_request(request)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not sent and not finished:
                await asyncio.sleep(0.001)
        finally:
            router._shutdown = True
            await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(run())

    assert not router.pending_requests
    if rejected:
        assert finished == [("len-boundary", FinishReason.LENGTH)]
        assert sent == []
    else:
        assert finished == []
        assert sent == [(0, "len-boundary")]


def test_local_evict_moves_hash_to_buffer():
    router = _build_router(algorithm="prefix_cache_aware")
    router.policy.instances_num_total_blocks[0] = 1
    router.policy.instances_block_size[0] = 4

    req = MonkReq("req-evict", [1, 2, 3, 4, 5, 6, 7, 8])
    req_block_hashes = router.policy.build_req_block_hashes(req, local_instance_id=0)
    router.policy.remember_request(req, local_instance_id=0)
    router.policy.insert_req_blocks(req.request_id)

    assert req_block_hashes[0] not in router.policy.cached_blocks[0]
    assert req_block_hashes[0] in router.policy.evict_buffer[0]


def test_apply_instance_evicts():
    router = _build_router(algorithm="prefix_cache_aware")

    router.policy.instances_block_size[0] = 4
    (blk_hash,) = router.policy.build_req_block_hashes(
        MonkReq("req-evict", [1, 2, 3, 4]), 0
    )
    assert blk_hash is not None

    # remove from cached_blocks
    router.policy.cached_blocks[0] = OrderedDict()
    router.policy.cached_blocks[0][blk_hash] = None
    router.policy.apply_instance_evicts(0, [blk_hash])
    assert blk_hash not in router.policy.cached_blocks[0]

    # remove from evict_buffer
    router.policy.evict_buffer[0] = OrderedDict()
    router.policy.evict_buffer[0][blk_hash] = 1.0
    router.policy.apply_instance_evicts(0, [blk_hash])
    assert blk_hash not in router.policy.evict_buffer[0]


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

    def finish_request_before_recv_stop(
        request, finish_reason: FinishReason, error=None
    ):
        router.finished_requests.append((request.request_id, finish_reason, error))
        request.finish_reason = finish_reason
        request.stop_stream(finish_reason, error=error)

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
