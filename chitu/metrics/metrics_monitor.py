# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from logging import getLogger
from typing import Optional
from chitu.metrics import PrometheusServerManager
from chitu.global_vars import get_global_args
from chitu.metrics.task_stats import count_tasks

logger = getLogger(__name__)


class MetricsMonitor:
    """
    Independent metrics monitoring thread that periodically queries
    Prometheus metrics and logs metrics statistics.
    """

    def __init__(
        self,
        manager: PrometheusServerManager,
        log_interval: float = 10.0,
    ):
        """
        Args:
            manager: instance of PrometheusServerManager
            log_interval: Logging interval in seconds (sleep time in monitor loop)
        """
        self.manager = manager
        self.log_interval = log_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False

    def start(self):
        """Start the monitoring thread."""
        if self._started:
            logger.info("Metrics monitor already started")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="MetricsMonitor"
        )
        self._thread.start()
        self._started = True
        logger.info(f"Metrics monitor started (log_interval={self.log_interval}s)")

    def stop(self):
        """Stop the monitoring thread."""
        if not self._started:
            return

        self._stop_event.set()
        if self._thread:
            self._thread.join()
        self._started = False
        logger.info("Metrics monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop that runs in the background thread."""
        while not self._stop_event.is_set():
            time.sleep(self.log_interval)
            try:
                log_interval = f"{int(self.log_interval)}s"
                # self.manager.list_all_metrics()
                prompt_tps = self.manager.query_metric_rate_each_rank(
                    "chitu_total_prompt_tokens_total", time_window=log_interval
                )
                gen_tps = self.manager.query_metric_rate_each_rank(
                    "chitu_total_generated_tokens_total", time_window=log_interval
                )
                eviction_rate = self.manager.query_metric_rate_each_rank(
                    "chitu_total_task_evictions_total", time_window=log_interval
                )
                kvcache_usage = self.manager.query_metric_latest_value_each_rank(
                    "chitu_kv_cache_usage_ratio"
                )
                used_blocks = self.manager.query_metric_latest_value_each_rank(
                    "chitu_used_blocks"
                )
                total_blocks = self.manager.query_metric_latest_value_each_rank(
                    "chitu_total_blocks"
                )
                total_bytes = self.manager.query_metric_latest_value_each_rank(
                    "chitu_total_bytes"
                )
                used_bytes = self.manager.query_metric_latest_value_each_rank(
                    "chitu_used_bytes"
                )
                torch_allocated_bytes = (
                    self.manager.query_metric_latest_value_each_rank(
                        "chitu_torch_allocated_bytes"
                    )
                )
                self._print_stats(
                    prompt_tps,
                    gen_tps,
                    eviction_rate,
                    kvcache_usage,
                    used_blocks,
                    total_blocks,
                    total_bytes,
                    used_bytes,
                    torch_allocated_bytes,
                )
            except Exception as e:
                logger.error(f"Metrics query failed: {e}")

    def _print_stats(
        self,
        prompt_tps: dict[tuple[str, str], str],
        gen_tps: dict[tuple[str, str], str],
        eviction_rate: dict[tuple[str, str], str],
        kvcache_usage: dict[tuple[str, str], str],
        used_blocks: dict[tuple[str, str], str],
        total_blocks: dict[tuple[str, str], str],
        total_bytes: dict[tuple[str, str], str],
        used_bytes: dict[tuple[str, str], str],
        torch_allocated_bytes: dict[tuple[str, str], str],
    ):
        all_metric_dict = [
            prompt_tps,
            gen_tps,
            eviction_rate,
            kvcache_usage,
            used_blocks,
            total_blocks,
            total_bytes,
            used_bytes,
            torch_allocated_bytes,
        ]
        all_rank_dp_pairs = {
            key for metric_dict in all_metric_dict for key in metric_dict
        }
        for rank_dp in all_rank_dp_pairs:
            dp_id = int(rank_dp[1])
            rank = int(rank_dp[0])
            running, waiting = count_tasks(dp_id=dp_id)

            log_msg = self._build_stats_message(
                prompt_tps=float(prompt_tps.get(rank_dp, "-1")),
                gen_tps=float(gen_tps.get(rank_dp, "-1")),
                running=running,
                waiting=waiting,
                kv_cache_usage=float(kvcache_usage.get(rank_dp, "-1")),
                eviction_rate=float(eviction_rate.get(rank_dp, "-1")),
                used_blocks=int(used_blocks.get(rank_dp, "-1")),
                total_blocks=int(total_blocks.get(rank_dp, "-1")),
                total_bytes=float(total_bytes.get(rank_dp, "-1")),
                used_bytes=float(used_bytes.get(rank_dp, "-1")),
                torch_allocated_bytes=float(torch_allocated_bytes.get(rank_dp, "-1")),
            )
            logger.info(f"[rank{rank}, DP{dp_id}]: {log_msg}")

    def _build_stats_message(
        self,
        prompt_tps,
        gen_tps,
        running,
        waiting,
        kv_cache_usage,
        eviction_rate,
        used_blocks,
        total_blocks,
        total_bytes,
        used_bytes,
        torch_allocated_bytes,
    ):
        """Build metrics statistics message."""
        parts = [
            f"Avg prompt throughput: {prompt_tps:.1f} tokens/s",
            f"Avg generation throughput: {gen_tps:.1f} tokens/s",
            f"Running: {running} reqs",
            f"Waiting: {waiting} reqs",
            f"KV cache usage: {kv_cache_usage*100:.1f}%({used_blocks}/{total_blocks})",
            f"Task evictions: {eviction_rate:.2f}/s",
            f"GPU mem usage: {used_bytes/(1024**3):.2f} GB / {total_bytes/(1024**3):.2f} GB (torch allocated {torch_allocated_bytes/(1024**3):.2f} GB)",
        ]
        return ", ".join(parts)


_global_monitor: Optional[MetricsMonitor] = None


def start_prometheus_server_and_metrics_monitor(
    collector_addrs: list,
):
    """
    Start the prometheus_server and metrics monitor.

    Args:
        collector_addrs: Prometheus Server pull metrics from these addresses.
    """
    global _global_monitor
    if _global_monitor is not None:
        logger.warning("Metrics monitor already exists")
        return

    manager = PrometheusServerManager.get_instance(collector_addrs)
    log_interval = get_global_args().metrics.log_interval
    if not manager.is_running():
        logger.warning(
            f"PrometheusServer is not running, MetricsMonitor will not start."
        )
        return
    _global_monitor = MetricsMonitor(manager, log_interval)
    _global_monitor.start()


def stop_metrics_monitor():
    """Stop the global metrics monitor."""
    global _global_monitor
    if _global_monitor is not None:
        _global_monitor.stop()
        _global_monitor = None
    PrometheusServerManager.cleanup()
