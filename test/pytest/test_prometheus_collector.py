# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from chitu.backend import Backend
from chitu.metrics import PrometheusMetricsCollector, metrics_runtime_context
from chitu.metrics.definitions import MetricContext
import re
from omegaconf import OmegaConf
from chitu.global_vars import set_global_args


def set_default_global_args():
    # global_args here is useless, but it must exists
    set_global_args(
        OmegaConf.create(
            {
                "infer": {"mtp_size": 1},
                "metrics": {"gpu_memory_refresh_interval_s": 1.0},
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )


RANK_LIST = [0, 1, 2, 3, 5, 6, 4, 7]
RANK_DP = [(rank, dp) for dp, rank in enumerate(RANK_LIST)]


class MockGroup:
    def __init__(self, global_rank, rank_in_group):
        self._rank_in_group = rank_in_group
        self.global_rank = global_rank
        self.rank_list = RANK_LIST

    @property
    def is_first_rank(self):
        return True

    @property
    def rank_in_group(self):
        return self._rank_in_group


class Monkcachemanager:
    def __init__(self, num_blocks, num_free_blocks):
        self.num_blocks = num_blocks
        self.num_free_blocks = num_free_blocks
        self.device = "cpu"

    @property
    def num_used_blocks(self):
        return self.num_blocks - self.num_free_blocks


@pytest.mark.parametrize("rank_dp", RANK_DP)
def test_PrometheusMetricsCollector(rank_dp, monkeypatch):
    rank, dp_id = rank_dp
    set_default_global_args()
    monkeypatch.setattr(
        "chitu.metrics.prometheus_collector.get_dp_group",
        lambda: MockGroup(rank, dp_id),
    )

    with metrics_runtime_context(
        MetricContext(rank=rank, rank_in_dp=0, dp_rank=dp_id, inst_id=0),
        start_http_server=False,
    ):
        # 测试 get_instance
        collector = PrometheusMetricsCollector.get_instance(is_create=True)
        pattern = r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3}):(\d+)$"
        match = re.match(pattern, collector.addr)
        assert match, f"Invalid address: {collector.addr}"
        assert collector.rank == rank
        assert collector.dp_id == dp_id

        # 第一次增加generated_tokens
        PrometheusMetricsCollector.inc_generated_tokens(1)
        samples = collector.total_generated_tokens.collect()[0].samples
        target_sample = None
        for sample in samples:
            if sample.name == "chitu_total_generated_tokens_total":
                target_sample = sample
                break
        assert target_sample is not None, "找不到正确的指标样本"
        assert target_sample.labels["rank"] == str(rank)
        assert target_sample.labels["dp_id"] == str(dp_id)
        assert target_sample.value == 1

        # 第二次增加generated_tokens
        PrometheusMetricsCollector.inc_generated_tokens(3)
        samples = collector.total_generated_tokens.collect()[0].samples
        target_sample = None
        for sample in samples:
            if sample.name == "chitu_total_generated_tokens_total":
                target_sample = sample
                break
        assert target_sample is not None
        assert target_sample.value == 4

        # test inc_prompt_tokens
        # 第一次增加prompt_tokens
        PrometheusMetricsCollector.inc_prompt_tokens(1)
        samples = collector.total_prompt_tokens.collect()[0].samples
        target_sample = None
        for sample in samples:
            if sample.name == "chitu_total_prompt_tokens_total":
                target_sample = sample
                break
        assert target_sample is not None
        assert target_sample.labels["rank"] == str(rank)
        assert target_sample.labels["dp_id"] == str(dp_id)
        assert target_sample.value == 1

        # 第二次增加prompt_tokens
        PrometheusMetricsCollector.inc_prompt_tokens(4)
        samples = collector.total_prompt_tokens.collect()[0].samples
        target_sample = None
        for sample in samples:
            if sample.name == "chitu_total_prompt_tokens_total":
                target_sample = sample
                break
        assert target_sample is not None
        assert target_sample.value == 5

        # test inc_task_eviction
        PrometheusMetricsCollector.inc_task_eviction()
        samples = collector.total_task_evictions.collect()[0].samples
        target_sample = None
        for sample in samples:
            if sample.name == "chitu_total_task_evictions_total":
                target_sample = sample
                break

        assert target_sample.value == 1

        # test update_kvcache_usage
        mock_cache_manager = Monkcachemanager(num_blocks=100, num_free_blocks=50)
        Backend.cache_dict = {"main": mock_cache_manager}
        PrometheusMetricsCollector.update_kvcache_usage()
        samples = collector.kv_cache_usage.collect()[0].samples
        target_sample = None
        for sample in samples:
            if sample.name == "chitu_kv_cache_usage_ratio":
                target_sample = sample
                assert target_sample.value == 0.5  # 50/100 = 0.5

            if sample.name == "chitu_used_blocks":
                target_sample = sample
                assert target_sample.value == 50

            if sample.name == "chitu_total_blocks":
                target_sample = sample
                assert target_sample.value == 100

    assert PrometheusMetricsCollector._instance is None


def test_gauge_refresh_is_throttled(monkeypatch):
    """A gauge rebuild is allowed once per interval, not once per call.

    The decode loop asks for one on every step that retires a request, and the rebuild
    itself is what the GPU waits for; the answer only has to be as fresh as a scrape.
    """
    from chitu.metrics import prometheus_collector as pc

    # Pin the interval, so the case does not depend on whatever global config a
    # previous test left behind -- with an interval of 0 the first assertion would flip.
    set_global_args(
        OmegaConf.create(
            {
                "infer": {"mtp_size": 1},
                "metrics": {"gpu_memory_refresh_interval_s": 1.0},
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    monkeypatch.setattr(pc, "_gauge_refreshed_at", {})
    assert pc._gauge_refresh_due("gpu_memory") is True
    assert pc._gauge_refresh_due("gpu_memory") is False
    assert pc._gauge_refresh_due("another_gauge") is True  # per gauge, not global
    assert pc._gauge_refresh_due("another_gauge") is False

    pc._gauge_refreshed_at["gpu_memory"] -= pc._gauge_refresh_interval_s()
    assert pc._gauge_refresh_due("gpu_memory") is True
    # Zero disables throttling, allowing every call to rebuild the gauge.
    set_global_args(
        OmegaConf.create(
            {
                "infer": {"mtp_size": 1},
                "metrics": {"gpu_memory_refresh_interval_s": 0.0},
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    assert pc._gauge_refresh_interval_s() == 0.0
    assert pc._gauge_refresh_due("gpu_memory") is True
    assert pc._gauge_refresh_due("gpu_memory") is True
    set_default_global_args()


def _gauge_value(gauge, rank: int, dp_id: int, instance_id: int) -> float:
    for sample in gauge.collect()[0].samples:
        labels = sample.labels
        if (
            labels.get("rank") == str(rank)
            and labels.get("dp_id") == str(dp_id)
            and labels.get("instance_id") == str(instance_id)
        ):
            return sample.value
    raise AssertionError(
        f"no sample for rank={rank} dp_id={dp_id} instance_id={instance_id}"
    )


def test_update_task_counts_is_not_throttled(monkeypatch):
    """Every request event must reach the task count gauges.

    `update_task_counts` is not periodic: callers run it when a request starts or
    ends, and nothing refreshes the gauges in between. If a call is skipped because
    a timer says "too soon", the count stays stale until the next request event --
    and an idle server never produces one, so `Running: N reqs` stays wrong.
    """
    from chitu.metrics import prometheus_collector as pc

    set_default_global_args()
    monkeypatch.setattr(
        "chitu.metrics.prometheus_collector.get_dp_group",
        lambda: MockGroup(0, 0),
    )
    # A very long interval makes the point sharp: even then, the second event must
    # land. The fix is that task counts do not consult the interval at all.
    monkeypatch.setattr(pc, "_gauge_refresh_interval_s", lambda: 3600.0)
    monkeypatch.setattr(pc, "_gauge_refreshed_at", {})
    counts = iter([(3, 2), (0, 0)])
    monkeypatch.setattr(pc, "count_tasks_for_dp_rank", lambda dp_id: next(counts))

    with metrics_runtime_context(
        MetricContext(rank=0, rank_in_dp=0, dp_rank=0, inst_id=0),
        start_http_server=False,
    ):
        collector = PrometheusMetricsCollector.get_instance(is_create=True)
        PrometheusMetricsCollector.update_task_counts()
        assert _gauge_value(collector.running_requests, 0, 0, 0) == 3
        assert _gauge_value(collector.waiting_requests, 0, 0, 0) == 2

        # No sleep: this call is inside the refresh interval, and it must still win.
        PrometheusMetricsCollector.update_task_counts()
        assert _gauge_value(collector.running_requests, 0, 0, 0) == 0
        assert _gauge_value(collector.waiting_requests, 0, 0, 0) == 0


def test_update_GPU_usage_keeps_kvcache_unthrottled(monkeypatch):
    """The GPU refresh interval must not also freeze the KV cache gauges.

    `update_GPU_usage` used to return as soon as the NVML throttle said "not yet",
    which skipped the KV cache gauges too. Only the NVML memory query is expensive
    enough to throttle; the KV counters are plain reads and must follow every
    request event, or the last update before an idle period is lost.
    """
    from chitu.metrics import prometheus_collector as pc

    set_default_global_args()
    monkeypatch.setattr(
        "chitu.metrics.prometheus_collector.get_dp_group",
        lambda: MockGroup(0, 0),
    )
    monkeypatch.setattr(pc, "_gauge_refresh_interval_s", lambda: 3600.0)
    monkeypatch.setattr(pc, "_gauge_refreshed_at", {})
    cache = Monkcachemanager(num_blocks=100, num_free_blocks=50)
    monkeypatch.setattr(Backend, "cache_dict", {"main": cache})

    with metrics_runtime_context(
        MetricContext(rank=0, rank_in_dp=0, dp_rank=0, inst_id=0),
        start_http_server=False,
    ):
        collector = PrometheusMetricsCollector.get_instance(is_create=True)
        PrometheusMetricsCollector.update_GPU_usage()
        assert _gauge_value(collector.kv_cache_usage, 0, 0, 0) == 0.5
        assert _gauge_value(collector.used_blocks, 0, 0, 0) == 50

        # The NVML gate is closed now (3600 s), but the KV counters changed.
        cache.num_free_blocks = 100
        PrometheusMetricsCollector.update_GPU_usage()
        assert _gauge_value(collector.kv_cache_usage, 0, 0, 0) == 0.0
        assert _gauge_value(collector.used_blocks, 0, 0, 0) == 0
