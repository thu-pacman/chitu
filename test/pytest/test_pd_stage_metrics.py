# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from prometheus_client import REGISTRY

from chitu.distributed.pd_disaggregation import pd_scheduler
from chitu.distributed.pd_disaggregation.pd_scheduler import PDSchedulerInfo
from chitu.metrics.definitions import MetricContext
from chitu.metrics.prometheus_collector import ROUTER_STAGES, observe_router_stage
from chitu.metrics.registry import metrics_runtime_context


@pytest.fixture
def metrics_runtime():
    ctx = metrics_runtime_context(MetricContext(is_router=True))
    ctx.__enter__()
    try:
        yield ctx
    finally:
        ctx.close()


def _stage(role, stage):
    labels = {"role": role, "stage": stage}
    count = REGISTRY.get_sample_value("chitu_pd_stage_duration_seconds_count", labels)
    total = REGISTRY.get_sample_value("chitu_pd_stage_duration_seconds_sum", labels)
    return (count or 0.0, total or 0.0)


def test_transfer_info_wait_is_not_restarted(monkeypatch, metrics_runtime):
    now = [100.0]
    monkeypatch.setattr(pd_scheduler, "time", SimpleNamespace(monotonic=lambda: now[0]))
    info = PDSchedulerInfo()
    info.begin_observe_stage_once("transfer_info")
    now[0] = 101.0
    info.begin_observe_stage_once("transfer_info")
    now[0] = 103.0
    info.finish_observe_stage("prefill", "transfer_info")
    assert _stage("prefill", "transfer_info") == pytest.approx((1, 3))


def test_router_stages_record_into_the_pd_stage_histogram(metrics_runtime):
    for i, stage in enumerate(ROUTER_STAGES, start=1):
        observe_router_stage(stage, 0.01 * i)

    for i, stage in enumerate(ROUTER_STAGES, start=1):
        count, total = _stage("router", stage)
        assert count == 1, stage
        assert total == pytest.approx(0.01 * i), stage
