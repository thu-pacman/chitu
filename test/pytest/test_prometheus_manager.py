# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import chitu.metrics
from chitu.metrics import PrometheusMetricsCollector
from chitu.metrics import PrometheusServerManager
import time
from omegaconf import OmegaConf
from chitu.global_vars import set_global_args, get_global_args
import multiprocessing as mp


class Backend:
    cache_manager = None


class MockGroup:
    def __init__(self, global_rank, rank_in_group):
        self.rank_in_group = rank_in_group
        self.global_rank = global_rank
        self.device = "cpu"


class Monkcachemanager:
    def __init__(self, num_blocks, num_free_blocks):
        self.num_blocks = num_blocks
        self.num_free_blocks = num_free_blocks

    def get_num_blocks(self):
        return self.num_blocks

    @property
    def num_used_blocks(self):
        return self.num_blocks - self.num_free_blocks


def run_PrometheusServerManager(rank, dp_id, result_queue, stop_event):
    chitu.metrics.prometheus_collector.Backend = Backend
    chitu.metrics.prometheus_collector.get_dp_group = lambda: MockGroup(rank, dp_id)

    collector = PrometheusMetricsCollector.get_instance(is_create=True)
    result_queue.put(collector.addr)

    if dp_id == 0:
        mock_cache_manager = Monkcachemanager(num_blocks=100, num_free_blocks=50)
    else:
        mock_cache_manager = Monkcachemanager(num_blocks=100, num_free_blocks=20)
    Backend.cache_manager = mock_cache_manager

    while not stop_event.is_set():
        if dp_id == 0:
            PrometheusMetricsCollector.inc_prompt_tokens(10)
            PrometheusMetricsCollector.inc_generated_tokens(100)
            PrometheusMetricsCollector.inc_task_eviction()
            PrometheusMetricsCollector.update_kvcache_usage()  # 5/10
        if dp_id == 1:
            PrometheusMetricsCollector.inc_prompt_tokens(20)
            PrometheusMetricsCollector.inc_generated_tokens(200)
            PrometheusMetricsCollector.inc_task_eviction()
            PrometheusMetricsCollector.inc_task_eviction()
            PrometheusMetricsCollector.update_kvcache_usage()  # 8/10
        time.sleep(1)


def test_PrometheusServerManager():
    set_global_args(
        OmegaConf.create(
            {
                "metrics": {
                    "prometheus_listening_port": 9090,
                    "prometheus_config_file": "prometheus.yml",
                    "prometheus_data_dir": "prometheus_data",
                    "prometheus_scrape_interval": 1,
                    "log_interval": 10,
                }
            }
        ),
        need_ensure=False,
    )

    log_interval = get_global_args().metrics.log_interval

    result_queue = mp.Queue()
    stop_event = mp.Event()
    processes = []
    dp_size = 4

    for rank in range(8):
        dp_id = rank // dp_size
        p = mp.Process(
            target=run_PrometheusServerManager,
            args=(rank, dp_id, result_queue, stop_event),
        )
        p.start()
        processes.append(p)

    time.sleep(10)

    try:
        collector_addrs = []  # list of collector.addr
        for _ in range(8):
            addr = result_queue.get(timeout=10)
            collector_addrs.append(addr)

        print(collector_addrs)

        manager = PrometheusServerManager.get_instance(collector_addrs)
        assert manager.is_running()

        time.sleep(30)  # 等待PrometheusServer收集足够的数据点

        prompt_tps_each_rank = manager.query_metric_rate_each_rank(
            "chitu_total_prompt_tokens_total", time_window=f"{log_interval}s"
        )
        for rank_str, dp_id_str in prompt_tps_each_rank:
            if dp_id_str == "0":
                assert (
                    abs(float(prompt_tps_each_rank[(rank_str, dp_id_str)]) - 10) < 1
                ), f"prompt_tps_each_rank[{(rank_str,dp_id_str)}]:{prompt_tps_each_rank[(rank_str,dp_id_str)]}, expect 10"
            if dp_id_str == "1":
                assert (
                    abs(float(prompt_tps_each_rank[(rank_str, dp_id_str)]) - 20) < 1
                ), f"prompt_tps_each_rank[{(rank_str,dp_id_str)}]:{prompt_tps_each_rank[(rank_str,dp_id_str)]}, expect 20"

        gen_tps_each_rank = manager.query_metric_rate_each_rank(
            "chitu_total_generated_tokens_total", time_window=f"{log_interval}s"
        )
        for rank_str, dp_id_str in gen_tps_each_rank:
            if dp_id_str == "0":
                assert (
                    abs(float(gen_tps_each_rank[(rank_str, dp_id_str)]) - 100) < 1
                ), f"gen_tps_each_rank[{dp_id_str}]:{gen_tps_each_rank[(rank_str,dp_id_str)]}, expect 100"
            if dp_id_str == "1":
                assert (
                    abs(float(gen_tps_each_rank[(rank_str, dp_id_str)]) - 200) < 1
                ), f"gen_tps_each_rank[{dp_id_str}]:{gen_tps_each_rank[(rank_str,dp_id_str)]}, expect 200"

        eviction_rate_each_rank = manager.query_metric_rate_each_rank(
            "chitu_total_task_evictions_total", time_window=f"{log_interval}s"
        )
        for rank_str, dp_id_str in eviction_rate_each_rank:
            if dp_id_str == "0":
                assert (
                    abs(float(eviction_rate_each_rank[(rank_str, dp_id_str)]) - 1) < 1
                ), f"eviction_rate_each_rank[{dp_id_str}]:{eviction_rate_each_rank[(rank_str,dp_id_str)]}, expect 1"
            if dp_id_str == "1":
                assert (
                    abs(float(eviction_rate_each_rank[(rank_str, dp_id_str)]) - 2) < 1
                ), f"eviction_rate_each_rank[{dp_id_str}]:{eviction_rate_each_rank[(rank_str,dp_id_str)]}, expect 2"

        kvcache_usage_each_rank = manager.query_metric_latest_value_each_rank(
            "chitu_kv_cache_usage_ratio"
        )
        for rank_str, dp_id_str in kvcache_usage_each_rank:
            if dp_id_str == "0":
                assert (
                    abs(float(kvcache_usage_each_rank[(rank_str, dp_id_str)]) - 0.5)
                    < 1e-5
                ), f"kvcache_usage_each_rank[rank{rank_str},dp{dp_id_str}]:{kvcache_usage_each_rank[(rank_str,dp_id_str)]}, expect 0.5"
            if dp_id_str == "1":
                assert (
                    abs(float(kvcache_usage_each_rank[(rank_str, dp_id_str)]) - 0.8)
                    < 1e-5
                ), f"kvcache_usage_each_rank[{dp_id_str}]:{kvcache_usage_each_rank[(rank_str,dp_id_str)]}, expect 0.8"
    except Exception:
        raise
    finally:
        PrometheusServerManager.cleanup()
        stop_event.set()
        for p in processes:
            p.join(timeout=2.0)
            if p.is_alive():
                p.kill()
