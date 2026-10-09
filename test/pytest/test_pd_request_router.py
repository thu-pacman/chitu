# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PD prefill routing policies (same style as ``test_dp_request_router``).

Exercises ``PrefixCacheAwarePolicy`` / ``LoadBalancer`` via ``policy.update_stats`` and
selection helpers — no ZMQ or ``_pd_stats_collector_task`` integration.
"""

import asyncio
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from typing import Any, Optional

from chitu.distributed.pd_disaggregation.pd_request_router import PDRequestRouter
from chitu.dp_request_router import LoadBalancer, PrefixCacheAwarePolicy, SchedulerStats
from chitu.global_vars import get_global_args, set_global_args
from chitu.schemas.serve_config import (
    PDDisaggregationConfig,
    RouterConfig,
)
from chitu.distributed.pd_disaggregation.pd_types import (
    PDRequestStatus,
    PendingPDRequest,
    SchedulerType,
)


@dataclass
class MockReq:
    """Minimal stand-in for :class:`~chitu.task.UserRequest` in routing tests."""

    request_id: str
    prompt_tokens: list[int] = field(default_factory=list)
    max_new_tokens: int = 1
    ttft_deadline_ts: Optional[float] = None
    finish_reason: Optional[str] = None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)


class TestPDRequestRouter(PDRequestRouter):
    """A testing PD request router without communication"""

    def __init__(self, config: RouterConfig, pd_config: PDDisaggregationConfig):
        super().__init__(config, pd_config)
        self._test_prefill_msg_queue: dict[int, list[dict[str, Any]]] = {}
        self._test_decode_msg_queue: dict[int, list[dict[str, Any]]] = {}
        self._test_finished_reqs: set[str] = set()

    # override communications

    async def _send_to_prefill_scheduler(
        self, local_instance_id: int, request_data: dict
    ):
        prefill_data = request_data.copy()
        prefill_data["target_role"] = "prefill"
        prefill_data["local_instance_id"] = local_instance_id
        if self._test_prefill_msg_queue.get(local_instance_id) is None:
            self._test_prefill_msg_queue[local_instance_id] = []
        self._test_prefill_msg_queue[local_instance_id].append(prefill_data)

    async def _send_to_decode_scheduler(
        self, local_instance_id: int, request_data: dict, prefill_scheduler_id: int
    ):
        decode_data = request_data.copy()
        decode_data["target_role"] = "decode"
        decode_data["local_instance_id"] = local_instance_id
        decode_data["prefill_scheduler_id"] = prefill_scheduler_id
        if self._test_decode_msg_queue.get(local_instance_id) is None:
            self._test_decode_msg_queue[local_instance_id] = []
        self._test_decode_msg_queue[local_instance_id].append(decode_data)

    def finish_request_before_recv_stop(
        self, request, finish_reason: Optional[str] = None, error: Optional[str] = None
    ):
        self._test_finished_reqs.add(request.request_id)
        request.finish_reason = finish_reason

    # tests

    def _test_create_mock_pd_request(
        self, request_id: str, prefill_scheduler_id: int, decode_scheduler_id: int
    ):
        request = MockReq(request_id=request_id)
        pd_request = PendingPDRequest(
            request_id=request_id,
            original_request=request,
            prefill_scheduler_id=prefill_scheduler_id,
            decode_scheduler_id=decode_scheduler_id,
            status=PDRequestStatus.DISPATCHED,
        )
        self.pending_pd_requests[request_id] = pd_request

    def _test_check_prefill_msg(self, local_instance_id, target_type, request_ids):
        real_request_ids = []
        if self._test_prefill_msg_queue.get(local_instance_id) is not None:
            real_request_ids = [
                msg["request_id"]
                for msg in self._test_prefill_msg_queue[local_instance_id]
                if msg["type"] == target_type
            ]
        assert set(request_ids) == set(
            real_request_ids
        ), f"Prefill instance {local_instance_id} message type {target_type}, except {request_ids}, find {real_request_ids}"

    def _test_check_decode_msg(self, local_instance_id, target_type, request_ids):
        real_request_ids = []
        if self._test_decode_msg_queue.get(local_instance_id) is not None:
            real_request_ids = [
                msg["request_id"]
                for msg in self._test_decode_msg_queue[local_instance_id]
                if msg["type"] == target_type
            ]
        assert set(request_ids) == set(
            real_request_ids
        ), f"Decode instance {local_instance_id} message type {target_type}, except {request_ids}, find {real_request_ids}"


def set_default_global_args():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {"enable_prefix_caching": True},
                "multi_inst": {
                    "n_insts": 3,
                    "inst_id": None,
                    "role": "prefill_and_decode",
                    "pd_disaggregation": {
                        "prefill_scheduler": None,
                        "decode_scheduler": None,
                    },
                    "inst_overrides": {
                        0: {
                            "multi_inst": {
                                "role": "prefill",
                                "pd_disaggregation": {
                                    "prefill_scheduler": {
                                        "max_batch_size": 32,
                                        "max_total_tokens": 8192,
                                        "batching_strategy": "varlen",
                                    },
                                },
                            }
                        },
                        1: {
                            "multi_inst": {
                                "role": "prefill",
                                "pd_disaggregation": {
                                    "prefill_scheduler": {
                                        "max_batch_size": 32,
                                        "max_total_tokens": 8192,
                                        "batching_strategy": "varlen",
                                    },
                                },
                            }
                        },
                        2: {
                            "multi_inst": {
                                "role": "decode",
                                "pd_disaggregation": {
                                    "decode_scheduler": {
                                        "scheduling_strategy": "immediate",
                                    },
                                },
                            }
                        },
                    },
                    "router": {
                        "is_router": True,
                    },
                },
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )


def _pd_router_config(
    *,
    routing_algorithm: str,
    routing_algorithm_for_decode: str = "power_of_two_choices",
) -> RouterConfig:
    set_default_global_args()
    return RouterConfig(
        is_router=True,
        max_inflight_per_instance=24,
        routing_algorithm=routing_algorithm,
        routing_algorithm_for_decode=routing_algorithm_for_decode,
        router_cache_threshold=0.5,
        router_balance_abs_threshold=32,
        router_balance_rel_threshold=1.1,
        router_evict_buffer_size=64,
        router_local_reservation_timeout_s=600.0,
        launch_timeout=3600.0,
        routing_by_req_len=False,
        decode_routing_by_req_len=False,
    )


def test_pd_prefill_prefix_among_prefers_more_prefix_hits():
    cfg = _pd_router_config(routing_algorithm="prefix_cache_aware")
    policy = PrefixCacheAwarePolicy(cfg)

    for local_instance_id in (0, 1):
        policy.update_stats(
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
    req = MockReq("r1", list(range(64)))
    policy.remember_request(req, local_instance_id=0)
    policy.insert_req_blocks(req.request_id)
    assert policy.get_router_load(0) == (1, 64)

    chosen = policy.select_scheduler(req)
    assert chosen == 0


def test_pd_prefill_prefix_among_prefers_higher_hit_prefill_with_real_block_size():
    cfg = _pd_router_config(routing_algorithm="prefix_cache_aware")
    policy = PrefixCacheAwarePolicy(cfg)
    block_size = 256

    policy.update_stats(
        SchedulerStats(
            local_instance_id=0,
            running_requests=0,
            waiting_requests=0,
            pending_tokens=0,
            throughput_tokens_per_sec=0.0,
            last_update_time=0.0,
            is_alive=True,
            num_blocks=16,
            block_size=block_size,
        )
    )
    policy.update_stats(
        SchedulerStats(
            local_instance_id=1,
            running_requests=20,
            waiting_requests=0,
            pending_tokens=0,
            throughput_tokens_per_sec=0.0,
            last_update_time=0.0,
            is_alive=True,
            num_blocks=16,
            block_size=block_size,
        )
    )

    block0 = list(range(0, block_size))
    block1 = list(range(block_size, block_size * 2))
    block2 = list(range(block_size * 2, block_size * 3))
    block3 = list(range(block_size * 3, block_size * 4))

    p0_warm_req = MockReq("p0-warm", block0)
    policy.remember_request(p0_warm_req, local_instance_id=0)
    policy.insert_req_blocks(p0_warm_req.request_id)

    p1_warm_req = MockReq("p1-warm", block0 + block1 + block2)
    policy.remember_request(p1_warm_req, local_instance_id=1)
    policy.insert_req_blocks(p1_warm_req.request_id)

    req = MockReq("pick-best-prefix", block0 + block1 + block2 + block3)

    assert policy.num_hit_blocks(0, policy.build_req_token_blocks(req, 0)) == 1
    assert policy.num_hit_blocks(1, policy.build_req_token_blocks(req, 1)) == 3
    assert policy.select_scheduler(req) == 1


def test_pd_prefill_prefix_fallback_uses_router_local_load_when_no_hits():
    cfg = _pd_router_config(routing_algorithm="prefix_cache_aware")
    policy = PrefixCacheAwarePolicy(cfg)

    policy.update_stats(
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
    policy.update_stats(
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

    req = MockReq("r-worker-load", [11, 12, 13, 14])
    # Worker-reported running_requests are ignored; with equal router-local load,
    # the two-gate min-load tie-breaker picks the last eligible scheduler.
    assert policy.select_scheduler(req) == 1

    policy.remember_request(MockReq("busy", list(range(200))), local_instance_id=0)
    req = MockReq("r-router-load", [21, 22, 23, 24])
    assert policy.select_scheduler(req) == 1


def test_pd_prefill_prefix_releases_router_load_on_first_token():
    cfg = _pd_router_config(routing_algorithm="prefix_cache_aware")
    policy = PrefixCacheAwarePolicy(cfg)
    for local_instance_id in (0, 1):
        policy.update_stats(
            SchedulerStats(
                local_instance_id=local_instance_id,
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

    req = MockReq("r-local", list(range(20)))
    policy.remember_request(req, local_instance_id=0)

    assert policy.get_router_load(0) == (1, 20)
    assert policy.select_scheduler(MockReq("r-next", [100, 101, 102, 103])) == 1

    policy.forget_request(req.request_id)

    assert policy.get_router_load(0) == (0, 0)


def test_pd_load_balancer_least_loaded_with_eligible_subset():
    """Keep ``LoadBalancer`` explicit eligible subset behavior covered."""
    cfg = _pd_router_config(routing_algorithm="least_loaded")
    policy = LoadBalancer(cfg)
    policy.update_stats(
        SchedulerStats(
            local_instance_id=0,
            running_requests=10,
            waiting_requests=0,
            pending_tokens=0,
            throughput_tokens_per_sec=0.0,
            last_update_time=0.0,
            is_alive=True,
        )
    )
    policy.update_stats(
        SchedulerStats(
            local_instance_id=1,
            running_requests=1,
            waiting_requests=0,
            pending_tokens=0,
            throughput_tokens_per_sec=0.0,
            last_update_time=0.0,
            is_alive=True,
        )
    )

    policy.remember_request(MockReq("busy", list(range(20))), local_instance_id=0)

    req = MockReq("lb", [])
    assert policy.select_scheduler(req, eligible_ids=[0, 1]) == 1


def test_pd_router_round_robin_policies():
    cfg = _pd_router_config(
        routing_algorithm="round_robin", routing_algorithm_for_decode="round_robin"
    )
    router = PDRequestRouter(cfg, PDDisaggregationConfig())

    assert isinstance(router.prefill_policy, LoadBalancer)
    assert router.prefill_policy.algorithm == "round_robin"
    assert router.decode_policy.algorithm == "round_robin"

    assert router.prefill_policy.select_scheduler(MockReq("r1")) == 0
    assert router.prefill_policy.select_scheduler(MockReq("r2")) == 1


def test_pd_router_prefix_prefill_decode_uses_decode_policy():
    cfg = _pd_router_config(
        routing_algorithm="prefix_cache_aware",
        routing_algorithm_for_decode="round_robin",
    )
    router = PDRequestRouter(cfg, PDDisaggregationConfig())

    assert isinstance(router.prefill_policy, PrefixCacheAwarePolicy)
    assert router.decode_policy.algorithm == "round_robin"


def _length_test_router(
    prefill_limits, decode_limits, *, length_routing=True, decode_length_routing=True
):
    cfg = _pd_router_config(
        routing_algorithm="prefix_cache_aware",
        routing_algorithm_for_decode="prefix_cache_aware",
    )
    cfg.routing_by_req_len = length_routing
    cfg.decode_routing_by_req_len = decode_length_routing
    args = get_global_args()
    instances = [("prefill", limit) for limit in prefill_limits] + [
        ("decode", limit) for limit in decode_limits
    ]
    args.multi_inst.n_insts = len(instances)
    args.multi_inst.inst_overrides = {
        i: {"multi_inst": {"role": role}} for i, (role, _) in enumerate(instances)
    }
    set_global_args(args, need_ensure=False, need_preprocess=False)

    class LengthTestPDRequestRouter(TestPDRequestRouter):
        def _scheduler_max_seq_len(self, inst_id):
            return instances[inst_id][1]

        def _reject_ttft_timeout(self, request, stage="ttft_router"):
            return False

    router = LengthTestPDRequestRouter(cfg, PDDisaggregationConfig())
    router.pd_coordination_service = None
    return router


async def _run_pd_router_iteration(router):
    router._shutdown = False
    task = asyncio.create_task(router._pd_request_processor_task())
    await asyncio.sleep(0)
    router._shutdown = True
    await asyncio.wait_for(task, timeout=1)


def test_pd_router_length_admission_respects_routing_flags():
    cases = [
        ([8], [16], 9, 1, True, True, True),
        ([16], [8], 9, 1, True, True, True),
        ([16], [8], 7, 2, True, True, True),
        ([4, 8], [4, 8], 7, 2, True, True, True),
        ([8], [16], 9, 1, True, False, True),
        ([16], [8], 7, 2, False, True, True),
        ([8], [16], 9, 1, False, True, False),
        ([16], [8], 7, 2, True, False, False),
        ([8], [8], 9, 1, False, False, False),
    ]
    for (
        prefill_limits,
        decode_limits,
        prompt_len,
        max_new_tokens,
        length_routing,
        decode_length_routing,
        rejected,
    ) in cases:
        router = _length_test_router(
            prefill_limits,
            decode_limits,
            length_routing=length_routing,
            decode_length_routing=decode_length_routing,
        )
        assert ("max_seq_len" in router.prefill_schedulers[0]) == length_routing
        assert ("max_seq_len" in router.decode_schedulers[0]) == decode_length_routing
        request = MockReq("oversize", [1] * prompt_len, max_new_tokens)
        selected_requests = []

        async def select(request):
            selected_requests.append(request)
            return True

        router._add_pd_request = select
        asyncio.run(router.add_request(request))
        asyncio.run(_run_pd_router_iteration(router))

        if rejected:
            assert not selected_requests
            assert request.finish_reason == "length"
            assert router._test_finished_reqs == {request.request_id}
            assert router.total_requests == 1
        else:
            assert selected_requests == [request]
            assert request.finish_reason is None
            assert not router._test_finished_reqs
            assert router.total_requests == 0
        assert not router.waiting_for_pd_workers
        assert not router.pending_pd_requests
        assert not router.pending_requests


def test_pd_router_length_boundary_and_queue_progress():
    cases = [
        ([8], [16], 8, 8, 0),
        ([16], [8], 7, 1, 0),
        ([4, 8], [4, 8], 7, 1, 1),
    ]
    for (
        prefill_limits,
        decode_limits,
        prompt_len,
        max_new_tokens,
        expected_sid,
    ) in cases:
        router = _length_test_router(prefill_limits, decode_limits)
        oversize = MockReq("oversize", [1] * (max(prefill_limits) + 1))
        valid = MockReq("valid", [1] * prompt_len, max_new_tokens)
        dispatched_requests = []

        async def dispatched(pd_request):
            dispatched_requests.append(pd_request)
            router._shutdown = True

        router._process_pd_request = dispatched
        asyncio.run(router.add_request(oversize))
        asyncio.run(router.add_request(valid))
        asyncio.run(asyncio.wait_for(router._pd_request_processor_task(), timeout=1))

        assert router._test_finished_reqs == {oversize.request_id}
        assert router.total_requests == 1
        assert not router.waiting_for_pd_workers
        pair = router.pending_pd_requests[valid.request_id]
        assert pair.prefill_scheduler_id == expected_sid
        assert pair.decode_scheduler_id == expected_sid
        assert dispatched_requests == [pair]


def test_pd_router_waits_for_capable_worker_recovery():
    for unavailable_role in ("prefill", "decode"):
        router = _length_test_router([4, 8], [4, 8])
        policy = (
            router.prefill_policy
            if unavailable_role == "prefill"
            else router.decode_policy
        )
        policy.scheduler_stats[1].is_alive = False
        request = MockReq("waiting", [1] * 7, 1)

        asyncio.run(router.add_request(request))
        asyncio.run(_run_pd_router_iteration(router))

        assert not router._test_finished_reqs
        assert router.total_requests == 0
        assert list(router.waiting_for_pd_workers) == [request]
        assert not router.pending_pd_requests

        policy.scheduler_stats[1].is_alive = True
        asyncio.run(_run_pd_router_iteration(router))

        assert not router._test_finished_reqs
        assert not router.waiting_for_pd_workers
        pair = router.pending_pd_requests[request.request_id]
        assert pair.prefill_scheduler_id == 1
        assert pair.decode_scheduler_id == 1


def test_pd_router_liveness_heartbeat_refreshes_online_worker():
    cfg = _pd_router_config(routing_algorithm="round_robin")
    router = PDRequestRouter(cfg, PDDisaggregationConfig())
    stats = router.prefill_policy.scheduler_stats[0]
    stats.last_heartbeat_time = 0.0
    router.prefill_schedulers[0]["instance_uuid"] = "prefill-uuid"

    router._record_liveness_heartbeat(
        {
            "pd_mode": "prefill_only",
            "local_instance_id": 0,
            "instance_uuid": "prefill-uuid",
        }
    )

    assert stats.last_heartbeat_time > 0.0
    assert stats.is_alive


def test_pd_router_handle_instance_fail():
    cfg = _pd_router_config(routing_algorithm="round_robin")
    router = TestPDRequestRouter(cfg, PDDisaggregationConfig())

    req_list = [
        ("test0", 0, 0),
        ("test1", 1, 0),
        ("test2", 0, 1),
    ]
    for request_id, prefill_id, decode_id in req_list:
        router._test_create_mock_pd_request(request_id, prefill_id, decode_id)

    asyncio.run(router.handle_dead_instance_prefill(1))
    router._test_check_decode_msg(0, "pd_prefill_fail", ["test1"])
    # prefill fail should not finish request
    assert len(router._test_finished_reqs) == 0

    asyncio.run(router.handle_dead_instance_decode(0))
    router._test_check_prefill_msg(0, "pd_decode_fail", ["test0"])
    router._test_check_prefill_msg(1, "pd_decode_fail", [])
    assert router._test_finished_reqs == {"test0"}
