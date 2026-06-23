# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PD prefill routing policies (same style as ``test_dp_request_router``).

Exercises ``PrefixCacheAwarePolicy`` / ``LoadBalancer`` via ``policy.update_stats`` and
selection helpers — no ZMQ or ``_pd_stats_collector_task`` integration.
"""

from omegaconf import OmegaConf
from dataclasses import dataclass, field

from chitu.distributed.pd_disaggregation.pd_request_router import PDRequestRouter
from chitu.dp_request_router import LoadBalancer, PrefixCacheAwarePolicy, SchedulerStats
from chitu.global_vars import set_global_args
from chitu.schemas.serve_config import (
    PDDisaggregationConfig,
    RouterConfig,
)


@dataclass
class MockReq:
    """Minimal stand-in for :class:`~chitu.task.UserRequest` in routing tests."""

    request_id: str
    prompt_tokens: list[int] = field(default_factory=list)


def set_default_global_args():
    set_global_args(
        OmegaConf.create(
            {
                "infer": {},
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
    router_cache_miss_fallback_algorithm: str = "least_loaded",
) -> RouterConfig:
    set_default_global_args()
    return RouterConfig(
        is_router=True,
        max_inflight_per_instance=24,
        routing_algorithm=routing_algorithm,
        routing_algorithm_for_decode=routing_algorithm_for_decode,
        router_cache_miss_fallback_algorithm=router_cache_miss_fallback_algorithm,
        router_hit_weight=1.0,
        router_load_penalty_weight=0.02,
    )


def test_pd_prefill_prefix_among_prefers_more_prefix_hits():
    cfg = _pd_router_config(routing_algorithm="prefix_cache_aware")
    policy = PrefixCacheAwarePolicy(cfg)

    policy.update_stats(
        SchedulerStats(
            local_instance_id=0,
            running_requests=20,
            waiting_requests=0,
            pending_tokens=0,
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

    req = MockReq("r1", [1, 2, 3, 4, 5, 6, 7, 8])
    policy.remember_request(req, local_instance_id=0)
    policy.insert_req_blocks(req.request_id)

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


def test_pd_prefill_prefix_among_fallback_least_loaded_when_no_hits():
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

    req = MockReq("r2", [11, 12, 13, 14])
    assert policy.select_scheduler(req) == 1


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

    req = MockReq("lb", [])
    assert policy.select_scheduler(req, eligible_ids=[0, 1]) == 1


def test_pd_router_round_robin_policies():
    cfg = _pd_router_config(
        routing_algorithm="round_robin",
        routing_algorithm_for_decode="round_robin",
        router_cache_miss_fallback_algorithm="power_of_two_choices",
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
