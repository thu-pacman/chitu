# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""The DP debug endpoints must report the load the engines reported.

The PD client-disconnect CI case (``test/client_interrupt_pd_test.py``) checks
that an interrupted request leaves every instance with an empty ``TaskPool``, and
reads that back from ``/dp/debug``: the instances report their own ``TaskPool``
counts over the stats channel, and the Router republishes them here - per role
for PD, and from the routing policy for DP (non-PD). ``/dp/test`` reports the
same load, so it is covered here too.
"""

import asyncio
from types import SimpleNamespace

import chitu.serve.api_app as api_app
from chitu.dp_request_router import SchedulerStats


def _stats(local_instance_id, running, waiting, *, is_alive=True, kv_send_inflight=0):
    return SchedulerStats(
        local_instance_id=local_instance_id,
        running_requests=running,
        waiting_requests=waiting,
        pending_tokens=0,
        throughput_tokens_per_sec=0.0,
        last_update_time=123.5,
        last_heartbeat_time=123.5,
        is_alive=is_alive,
        kv_send_inflight=kv_send_inflight,
    )


def _patch_debug_sources(monkeypatch, request_router, *, active_requests=0):
    monkeypatch.setattr(
        api_app,
        "get_global_args",
        lambda: SimpleNamespace(multi_inst=SimpleNamespace(n_insts=2)),
    )
    monkeypatch.setattr(api_app, "get_request_router", lambda *a, **kw: request_router)
    monkeypatch.setattr(
        api_app,
        "get_token_router",
        lambda *a, **kw: SimpleNamespace(
            active_requests=dict.fromkeys(range(active_requests), None),
            total_tokens_received=0,
        ),
    )
    monkeypatch.setattr(api_app, "get_server_status", lambda: True)


def _debug_info() -> dict:
    return asyncio.run(api_app.get_dp_debug_info())


def test_dp_debug_reports_both_pd_roles(monkeypatch):
    request_router = SimpleNamespace(
        total_requests=7,
        pending_requests=["req"],
        pending_pd_requests={"req": object()},
        prefill_policy=SimpleNamespace(
            scheduler_stats={0: _stats(0, 1, 0, kv_send_inflight=2)}
        ),
        decode_policy=SimpleNamespace(
            scheduler_stats={1: _stats(1, 0, 2, is_alive=False)}
        ),
    )
    _patch_debug_sources(monkeypatch, request_router, active_requests=1)

    info = _debug_info()

    assert "router_error" not in info
    router = info["request_router"]
    assert router["pending_pd_requests"] == 1
    assert "waiting_for_pd_workers" not in router
    # Separate id spaces: a Prefill and a Decode instance can share an id. The
    # Prefill instance is in the middle of a KV transfer here (kv_send_inflight).
    assert router["prefill_instances"] == {
        "0": {
            "running_requests": 1,
            "waiting_requests": 0,
            "kv_send_inflight": 2,
            "is_alive": True,
            "last_update_time": 123.5,
        }
    }
    assert router["decode_instances"]["1"] == {
        "running_requests": 0,
        "waiting_requests": 2,
        "kv_send_inflight": 0,
        "is_alive": False,
        "last_update_time": 123.5,
    }
    assert info["token_router"]["active_requests"] == 1


def test_dp_debug_reports_scheduler_stats_without_pd(monkeypatch):
    # The load lives on the policy (RequestRouter.policy); nothing named
    # ``load_balancer`` exists on the real router.
    request_router = SimpleNamespace(
        total_requests=3,
        pending_requests=[],
        policy=SimpleNamespace(scheduler_stats={4: _stats(4, 2, 1)}),
    )
    _patch_debug_sources(monkeypatch, request_router)

    router = _debug_info()["request_router"]

    assert router["scheduler_stats"]["4"]["running_requests"] == 2
    assert "prefill_instances" not in router


def test_dp_debug_reads_a_policy_exposed_as_load_balancer(monkeypatch):
    # Fallback for routers that expose their policy under the older name.
    request_router = SimpleNamespace(
        total_requests=0,
        pending_requests=[],
        load_balancer=SimpleNamespace(scheduler_stats={5: _stats(5, 1, 0)}),
    )
    _patch_debug_sources(monkeypatch, request_router)

    assert _debug_info()["request_router"]["scheduler_stats"]["5"] == {
        "running_requests": 1,
        "waiting_requests": 0,
        "kv_send_inflight": 0,
        "is_alive": True,
        "last_update_time": 123.5,
    }


def test_dp_debug_survives_a_router_without_scheduler_stats(monkeypatch):
    # The endpoint used to raise AttributeError here (``load_balancer`` is not an
    # attribute of the request router), which reported every router as broken
    # instead of showing its load.
    request_router = SimpleNamespace(total_requests=0, pending_requests=[])
    _patch_debug_sources(monkeypatch, request_router)

    info = _debug_info()

    assert "router_error" not in info
    assert info["request_router"]["scheduler_stats"] == {}


def test_dp_test_reports_the_policy_load(monkeypatch):
    # ``/dp/test`` read the same non-existent attribute and always answered {}.
    request_router = SimpleNamespace(
        pending_requests=[],
        scheduler_addresses=["tcp://a:1", "tcp://b:2"],
        policy=SimpleNamespace(scheduler_stats={7: _stats(7, 3, 0)}),
    )
    _patch_debug_sources(monkeypatch, request_router)

    result = asyncio.run(api_app.test_dp_system())

    assert "router_error" not in result
    assert result["connected_schedulers"] == 2
    assert result["connection_test"] == "success"
    assert result["load_balancer_stats"][7].running_requests == 3


def test_dp_debug_needs_the_summary_but_not_the_role_snapshots(monkeypatch):
    # A DP (non-PD) router has no P/D pairing, and the summary fields are what a
    # caller reads first: they must be there even when no stats arrived yet.
    request_router = SimpleNamespace(
        total_requests=0, pending_requests=[], policy=SimpleNamespace()
    )
    _patch_debug_sources(monkeypatch, request_router)

    router = _debug_info()["request_router"]

    assert router["total_requests"] == 0
    assert router["pending_requests"] == 0
    assert router["scheduler_stats"] == {}
