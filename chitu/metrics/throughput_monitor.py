# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from logging import getLogger
from typing import Optional
from chitu.metrics.prometheus_collector import PrometheusMetricsCollector

logger = getLogger(__name__)


class ThroughputMonitor:
    """
    Independent throughput monitoring thread that periodically queries
    Prometheus metrics and logs throughput statistics.
    """

    def __init__(
        self, collector, log_interval: float = 10.0, collect_interval: float = 1.0
    ):
        """
        Args:
            collector: PrometheusMetricsCollector instance
            log_interval: Logging interval in seconds (sleep time in monitor loop)
            collect_interval: Statistics collection interval in seconds (for get_rate)
        """
        self.collector = collector
        self.log_interval = log_interval
        self.collect_interval = collect_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False

    def start(self):
        """Start the monitoring thread."""
        if self._started:
            logger.warning("Throughput monitor already started")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="ThroughputMonitor"
        )
        self._thread.start()
        self._started = True
        logger.info(
            f"Throughput monitor started (log_interval={self.log_interval}s, collect_interval={self.collect_interval}s)"
        )

    def stop(self):
        """Stop the monitoring thread."""
        if not self._started:
            return

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._started = False
        logger.info("Throughput monitor stopped")

    def _get_metric_value(self, metric, default=0, value_type=float):
        """Safely get metric value with default fallback."""
        if not metric:
            return default
        try:
            return value_type(metric._value.get())
        except Exception:
            return default

    def _build_stats_message(
        self, prompt_tps, gen_tps, running, waiting, kv_cache_usage, eviction_rate
    ):
        """Build throughput statistics message."""
        parts = [
            f"Avg prompt throughput: {prompt_tps:.1f} tokens/s",
            f"Avg generation throughput: {gen_tps:.1f} tokens/s",
            f"Running: {running} reqs",
            f"Waiting: {waiting} reqs",
            f"GPU KV cache usage: {kv_cache_usage*100:.1f}%",
            f"Task evictions: {eviction_rate:.2f}/s",
        ]
        return ", ".join(parts)

    def _get_backend_executor(self):
        """Get Backend.executor with proper import."""
        try:
            from chitu.backend import Backend

            return Backend.executor
        except Exception:
            return None

    def _monitor_loop(self):
        """Main monitoring loop that runs in the background thread."""
        while not self._stop_event.is_set():
            time.sleep(self.log_interval)
            try:
                self.collector.update_metrics()
                prompt_tps, gen_tps, eviction_rate = self.collector.get_rate(
                    interval_sec=self.collect_interval
                )

                kv_cache_usage = self._get_metric_value(
                    self.collector.kv_cache_usage, 0.0, float
                )

                executor = self._get_backend_executor()
                if executor and executor.dp_size > 1:
                    # DP mode: print stats for each DP rank
                    self._print_all_dp_stats(
                        executor, prompt_tps, gen_tps, kv_cache_usage, eviction_rate
                    )
                else:
                    # Non-DP mode: print single aggregated stats
                    self._print_single_stats(
                        executor, prompt_tps, gen_tps, kv_cache_usage, eviction_rate
                    )
            except Exception as e:
                logger.debug(f"Throughput query failed: {e}")

    def _print_single_stats(
        self, executor, prompt_tps, gen_tps, kv_cache_usage, eviction_rate
    ):
        """Print aggregated stats for non-DP mode."""
        from chitu.metrics.task_stats import count_tasks

        running, waiting = count_tasks(dp_id=None)
        log_msg = self._build_stats_message(
            prompt_tps, gen_tps, running, waiting, kv_cache_usage, eviction_rate
        )

        # Add rank info prefix if available
        if executor:
            rank_prefix = f"Rank {executor.rank}"
            if executor.dp_dispatcher and hasattr(
                executor.dp_dispatcher, "rank_in_group"
            ):
                dp_id = executor.dp_dispatcher.rank_in_group
                rank_prefix = f"Rank {executor.rank}, DP {dp_id}"
            log_msg = f"[{rank_prefix}] {log_msg}"

        logger.info(log_msg)

    def _print_all_dp_stats(
        self, executor, prompt_tps, gen_tps, kv_cache_usage, eviction_rate
    ):
        """Print stats for each DP rank separately."""
        from chitu.metrics.task_stats import count_tasks

        try:
            rank = executor.rank
            for dp_id in range(executor.dp_size):
                running, waiting = count_tasks(dp_id=dp_id)
                log_msg = self._build_stats_message(
                    prompt_tps, gen_tps, running, waiting, kv_cache_usage, eviction_rate
                )
                logger.info(f"[Rank {rank}] DP {dp_id}: {log_msg}")
        except Exception as e:
            logger.debug(f"Failed to print DP stats: {e}")


_global_monitor: Optional[ThroughputMonitor] = None


def start_throughput_monitor(
    collector, log_interval: float = 10.0, collect_interval: float = 1.0
):
    """
    Start the global throughput monitor.

    Args:
        collector: PrometheusMetricsCollector instance
        log_interval: Logging interval in seconds (sleep time)
        collect_interval: Statistics collection interval in seconds (for get_rate)
    """
    global _global_monitor
    if _global_monitor is not None:
        logger.warning("Throughput monitor already exists")
        return

    _global_monitor = ThroughputMonitor(collector, log_interval, collect_interval)
    _global_monitor.start()


def stop_throughput_monitor():
    """Stop the global throughput monitor."""
    global _global_monitor
    if _global_monitor is not None:
        _global_monitor.stop()
        _global_monitor = None
