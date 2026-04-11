# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import threading
from logging import getLogger
from typing import Optional

from chitu.backend import Backend
from chitu.metrics import PrometheusServerManager
from chitu.metrics.grafana_manager import GrafanaManager
from chitu.global_vars import get_global_args
from chitu.metrics.task_stats import count_tasks
from chitu.metrics.cache_stats import paged_kvcache_stats
from chitu.utils import ceil_div

try:
    from chitu.distributed.pd_disaggregation.pd_scheduler import (
        get_pd_scheduler_instance,
    )
except Exception:  # pragma: no cover - optional PD dependency
    get_pd_scheduler_instance = None

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
            self._thread.join(timeout=float(self.log_interval) + 1.0)
            if self._thread.is_alive():
                logger.warning(
                    "Metrics monitor thread did not exit in time; continue shutdown."
                )
        self._started = False
        logger.info("Metrics monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop that runs in the background thread."""
        while not self._stop_event.is_set():
            self._stop_event.wait(self.log_interval)
            if self._stop_event.is_set():
                break
            try:
                if (
                    hasattr(self.manager, "is_running")
                    and not self.manager.is_running()
                ):
                    logger.warning(
                        "Prometheus server is not running; skip metrics query."
                    )
                    continue
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
                cuda_total_bytes = self.manager.query_metric_latest_value_each_rank(
                    "chitu_cuda_total_bytes"
                )
                cuda_used_bytes = self.manager.query_metric_latest_value_each_rank(
                    "chitu_cuda_used_bytes"
                )
                torch_allocated_bytes = (
                    self.manager.query_metric_latest_value_each_rank(
                        "chitu_torch_allocated_bytes"
                    )
                )
                torch_reserved_bytes = self.manager.query_metric_latest_value_each_rank(
                    "chitu_torch_reserved_bytes"
                )
                mtp_proposed_rate = self.manager.query_metric_rate_each_rank(
                    "chitu_mtp_proposed_tokens_total", time_window=log_interval
                )
                mtp_accepted_rate = self.manager.query_metric_rate_each_rank(
                    "chitu_mtp_accepted_tokens_total", time_window=log_interval
                )
                total_hit_tokens = self.manager.query_metric_latest_value_each_rank(
                    "chitu_total_hit_tokens_total"
                )
                total_prompt_tokens = self.manager.query_metric_latest_value_each_rank(
                    "chitu_total_prompt_tokens_total"
                )
                self._print_stats(
                    prompt_tps,
                    gen_tps,
                    eviction_rate,
                    kvcache_usage,
                    used_blocks,
                    total_blocks,
                    cuda_total_bytes,
                    cuda_used_bytes,
                    torch_allocated_bytes,
                    torch_reserved_bytes,
                    total_hit_tokens,
                    total_prompt_tokens,
                    mtp_proposed_rate,
                    mtp_accepted_rate,
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
        cuda_total_bytes: dict[tuple[str, str], str],
        cuda_used_bytes: dict[tuple[str, str], str],
        torch_allocated_bytes: dict[tuple[str, str], str],
        torch_reserved_bytes: dict[tuple[str, str], str],
        total_hit_tokens: dict[tuple[str, str], str],
        total_prompt_tokens: dict[tuple[str, str], str],
        mtp_proposed_rate: dict[tuple[str, str], str] = None,
        mtp_accepted_rate: dict[tuple[str, str], str] = None,
    ):
        all_metric_dict = [
            prompt_tps,
            gen_tps,
            eviction_rate,
            kvcache_usage,
            used_blocks,
            total_blocks,
            cuda_total_bytes,
            cuda_used_bytes,
            torch_allocated_bytes,
            torch_reserved_bytes,
            total_hit_tokens,
            total_prompt_tokens,
        ]
        all_rank_dp_pairs = {
            key for metric_dict in all_metric_dict for key in metric_dict
        }
        dp_ids = {int(rank_dp[1]) for rank_dp in all_rank_dp_pairs}
        dp_size = max(dp_ids) + 1 if dp_ids else 1
        prealloc_blocks_by_dp = self._get_prealloc_blocks_by_dp(dp_size)
        for rank_dp in all_rank_dp_pairs:
            dp_id = int(rank_dp[1])
            rank = int(rank_dp[0])
            running, waiting = count_tasks(dp_id=dp_id)
            prealloc_blocks = (
                prealloc_blocks_by_dp.get(dp_id) if prealloc_blocks_by_dp else None
            )

            hit_tokens = int(total_hit_tokens.get(rank_dp, "0"))
            prompt_tokens = int(total_prompt_tokens.get(rank_dp, "0"))
            hit_rate = hit_tokens / prompt_tokens if prompt_tokens != 0 else 0

            used_blocks_value = int(used_blocks.get(rank_dp, "0"))
            total_blocks_value = int(total_blocks.get(rank_dp, "0"))
            kv_cache_usage = float(kvcache_usage.get(rank_dp, "0"))

            if used_blocks_value == 0 or total_blocks_value == 0 or kv_cache_usage == 0:
                used_blocks_value, total_blocks_value, kv_cache_usage = (
                    paged_kvcache_stats(dp_id=dp_id)
                )

            mtp_hit_rate = None
            if mtp_proposed_rate and mtp_accepted_rate:
                proposed = float(mtp_proposed_rate.get(rank_dp, "0"))
                accepted = float(mtp_accepted_rate.get(rank_dp, "0"))
                if proposed > 0:
                    mtp_hit_rate = accepted / proposed

            log_msg = self._build_stats_message(
                prompt_tps=float(prompt_tps.get(rank_dp, "0")),
                gen_tps=float(gen_tps.get(rank_dp, "0")),
                running=running,
                waiting=waiting,
                kv_cache_usage=float(kvcache_usage.get(rank_dp, "0")),
                eviction_rate=float(eviction_rate.get(rank_dp, "0")),
                used_blocks=used_blocks_value,
                total_blocks=total_blocks_value,
                prealloc_blocks=prealloc_blocks,
                cuda_total_bytes=float(cuda_total_bytes.get(rank_dp, "0")),
                cuda_used_bytes=float(cuda_used_bytes.get(rank_dp, "0")),
                torch_allocated_bytes=float(torch_allocated_bytes.get(rank_dp, "0")),
                torch_reserved_bytes=float(torch_reserved_bytes.get(rank_dp, "0")),
                hit_len=hit_tokens,
                prompt_tokens=prompt_tokens,
                hit_rate=hit_rate,
                mtp_hit_rate=mtp_hit_rate,
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
        prealloc_blocks,
        cuda_total_bytes,
        cuda_used_bytes,
        torch_allocated_bytes,
        torch_reserved_bytes,
        hit_len,
        prompt_tokens,
        hit_rate,
        mtp_hit_rate=None,
    ):
        """Build metrics statistics message."""
        parts = [
            f"Avg prompt throughput: {prompt_tps:.1f} tokens/s",
            f"Avg generation throughput: {gen_tps:.1f} tokens/s",
            f"Running: {running} reqs",
            f"Waiting: {waiting} reqs",
            f"KV cache usage: {kv_cache_usage*100:.1f}%({used_blocks}/{total_blocks})",
            f"Hit rate: {hit_rate*100:.1f}%({hit_len}/{prompt_tokens})",
            f"Task evictions: {eviction_rate:.2f}/s",
        ]
        if mtp_hit_rate is not None:
            parts.append(f"MTP hit rate: {mtp_hit_rate*100:.1f}%")
        prealloc_msg = str(int(prealloc_blocks)) if prealloc_blocks is not None else "-"
        parts.append(f"KV blocks prealloc: {prealloc_msg}")
        if cuda_total_bytes > 0 and cuda_used_bytes >= 0:
            used_gib = cuda_used_bytes / 1024**3
            total_gib = cuda_total_bytes / 1024**3
            if torch_allocated_bytes >= 0:
                torch_allocated_gib = torch_allocated_bytes / 1024**3
                torch_reserved_gib = torch_reserved_bytes / 1024**3
                torch_unused_gib = torch_reserved_gib - torch_allocated_gib
                non_torch_gib = (
                    max(cuda_used_bytes - torch_reserved_bytes, 0.0) / 1024**3
                )
                parts.append(
                    "GPU mem: "
                    f"{used_gib:.2f}/{total_gib:.2f} GiB "
                    f"(torch-allocated: {torch_allocated_gib:.2f} GiB, torch-unused: {torch_unused_gib:.2f} "
                    f"GiB, non-torch: {non_torch_gib:.2f} GiB)"
                )
            else:
                parts.append(f"GPU mem: {used_gib:.2f}/{total_gib:.2f} GiB")
        return ", ".join(parts)

    def _get_prealloc_blocks_by_dp(self, dp_size: int) -> Optional[dict[int, int]]:
        """Get prealloc KV blocks per DP rank from PD scheduler."""
        if get_pd_scheduler_instance is None:
            return None
        scheduler = get_pd_scheduler_instance()
        if scheduler is None:
            return None
        if Backend.cache_dict["main"] is None:
            return None
        if not hasattr(Backend.cache_dict["main"], "block_size"):
            return None
        block_size = Backend.cache_dict["main"].block_size
        if block_size <= 0:
            return None
        tokens_by_dp = getattr(
            scheduler, "_decode_prealloc_tokens_inflight_by_dp", None
        )
        if not tokens_by_dp:
            return None
        prealloc_blocks: dict[int, int] = {}
        for dp_id in range(dp_size):
            tokens = 0
            if dp_id < len(tokens_by_dp):
                tokens = int(tokens_by_dp[dp_id])
            prealloc_blocks[dp_id] = (
                int(ceil_div(tokens, block_size)) if tokens > 0 else 0
            )
        return prealloc_blocks


_global_monitor: Optional[MetricsMonitor] = None


def start_prometheus_server_and_metrics_monitor(
    collector_addrs: list,
):
    """
    Start the prometheus_server, optionally Grafana, and the metrics monitor.

    Args:
        collector_addrs: Prometheus Server pull metrics from these addresses.
    """
    global _global_monitor
    if _global_monitor is not None:
        logger.warning("Metrics monitor already exists")
        return

    manager = PrometheusServerManager.get_instance(collector_addrs)
    metrics_cfg = get_global_args().metrics
    log_interval = metrics_cfg.log_interval
    if not manager.is_running():
        logger.warning(
            f"PrometheusServer is not running, MetricsMonitor will not start."
        )
        return

    if getattr(metrics_cfg, "grafana_enabled", False):
        try:
            prometheus_url = (
                f"http://{metrics_cfg.prometheus_listening_host}:{manager.server_port}"
            )
            GrafanaManager.get_instance(prometheus_url)
        except Exception as e:
            logger.warning(f"Failed to start Grafana: {e}")

    _global_monitor = MetricsMonitor(manager, log_interval)
    _global_monitor.start()


def stop_metrics_monitor():
    """Stop the global metrics monitor and managed servers."""
    global _global_monitor
    if _global_monitor is not None:
        _global_monitor.stop()
        _global_monitor = None
    GrafanaManager.cleanup()
    PrometheusServerManager.cleanup()
