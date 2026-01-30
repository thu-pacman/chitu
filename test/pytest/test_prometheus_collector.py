# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from chitu.metrics import PrometheusMetricsCollector
import re


class Backend:
    cache_manager = None


class MockGroup:
    def __init__(self, global_rank, rank_in_group):
        self.rank_in_group = rank_in_group
        self.global_rank = global_rank


class Monkcachemanager:
    def __init__(self, num_blocks, num_free_blocks):
        self.num_blocks = num_blocks
        self.num_free_blocks = num_free_blocks
        self.device = "cpu"

    def get_num_blocks(self):
        return self.num_blocks

    @property
    def num_used_blocks(self):
        return self.num_blocks - self.num_free_blocks


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("dp_id", [0, 1, 2, 3])
def test_PrometheusMetricsCollector(rank, dp_id, monkeypatch):
    # 模拟依赖
    monkeypatch.setattr("chitu.metrics.prometheus_collector.Backend", Backend)
    monkeypatch.setattr(
        "chitu.metrics.prometheus_collector.get_dp_group",
        lambda: MockGroup(rank, dp_id),
    )

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
    Backend.cache_manager = mock_cache_manager
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

    PrometheusMetricsCollector.stop_instance()
    assert PrometheusMetricsCollector._instance is None
