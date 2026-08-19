# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import threading
from logging import getLogger
from typing import Optional

from chitu.backend import Backend
from chitu.metrics import PrometheusServerManager
from chitu.metrics.grafana_manager import GrafanaManager
from chitu.global_vars import (
    get_global_args,
    get_multi_inst_ids_by_role,
    is_classic_pd_disagg,
    is_independent_multi_inst,
)
from chitu.metrics.definitions import (
    MetricType,
    prometheus_query_name,
    stdout_derived_metrics,
    stdout_raw_metrics,
)

logger = getLogger(__name__)


class MetricsFormatter:
    """
    MetricsFormatter
    """

    def __init__(self):
        args = get_global_args()
        dp_enabled = (
            getattr(args, "multi_inst", None) is not None
            and args.multi_inst.n_insts > 1
        )

        self.use_instance_info: bool = dp_enabled
        self.use_dp_info: bool = dp_enabled or (
            getattr(args, "infer", None) is not None and args.infer.dp_size > 1
        )
        self.use_rank_info: bool = True

        if dp_enabled:
            if is_classic_pd_disagg():
                self.pd_enabled = True
            elif is_independent_multi_inst():
                self.pd_enabled = False
            else:
                raise NotImplementedError(
                    "Mixing prefill_and_decode with prefill/decode roles is not supported"
                )
        else:
            self.pd_enabled = False
        self.num_prefill_instances: int = -1
        if self.pd_enabled:
            self.num_prefill_instances = len(get_multi_inst_ids_by_role("prefill"))

        self.prefix_format = self._get_prefix_format_str()

    def _rank_format_str(self) -> Optional[str]:
        return "Rank{rank}" if self.use_rank_info else None

    def _dp_format_str(self) -> Optional[str]:
        return "DP{dp_id}" if self.use_dp_info else None

    def _instance_format_str(self) -> Optional[str]:
        return "{instance_id}" if self.use_instance_info else None

    def _get_prefix_format_str(self) -> str:
        prefix_formats: list[str] = [
            s
            for s in (
                self._instance_format_str(),
                self._dp_format_str(),
                self._rank_format_str(),
            )
            if s is not None
        ]
        return "[" + ", ".join(prefix_formats) + "]: "

    def _format_instance(self, instance_id) -> str:
        if not self.use_instance_info:
            return ""
        if self.pd_enabled:
            if instance_id < self.num_prefill_instances:
                return f"Prefill{instance_id}"
            else:
                return f"Decode{instance_id - self.num_prefill_instances}"
        return f"Instance{instance_id}"

    def __call__(self, metadata, stats) -> str:
        instance_id = int(metadata[0])
        dp_id = int(metadata[1])
        rank = int(metadata[2])
        return self.prefix_format.format(
            instance_id=self._format_instance(instance_id),
            dp_id=dp_id,
            rank=rank,
        ) + ", ".join(stats)


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
        self.metrics_formatter = MetricsFormatter()
        self.enable_multi_instance = get_global_args().multi_inst.n_insts > 1
        self.log_interval = log_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False
        self.stdout_raw_metrics = stdout_raw_metrics()
        self.stdout_derived_metrics = stdout_derived_metrics()
        self.stdout_raw_query_names = {
            metric.name: prometheus_query_name(metric)
            for metric in self.stdout_raw_metrics
        }
        self.stdout_derived_names = {
            metric.name for metric in self.stdout_derived_metrics
        }

    def _stdout_query_name(self, metric_name: str) -> str:
        query_name = self.stdout_raw_query_names.get(metric_name)
        if query_name is None:
            raise KeyError(f"Metric {metric_name} is not enabled for stdout monitoring")
        return query_name

    def _require_stdout_derived(self, metric_name: str) -> None:
        if metric_name not in self.stdout_derived_names:
            raise KeyError(
                f"Derived metric {metric_name} is not enabled for stdout monitoring"
            )

    def _query_stdout_raw_metrics(self, time_window: str) -> tuple[
        dict[str, dict[tuple[str, str, str], str]],
        dict[str, dict[tuple[str, str, str], str]],
    ]:
        latest_values: dict[str, dict[tuple[str, str, str], str]] = {}
        rates: dict[str, dict[tuple[str, str, str], str]] = {}
        for metric in self.stdout_raw_metrics:
            query_name = self._stdout_query_name(metric.name)
            latest_values[metric.name] = (
                self.manager.query_metric_latest_value_each_rank(query_name)
            )
            if metric.type == MetricType.COUNTER:
                rates[metric.name] = self.manager.query_metric_rate_each_rank(
                    query_name, time_window=time_window
                )
        return latest_values, rates

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
                for metric in self.stdout_derived_metrics:
                    self._require_stdout_derived(metric.name)
                latest_values, rates = self._query_stdout_raw_metrics(log_interval)
                prompt_tps = rates.get("chitu_total_prompt_tokens", {})
                gen_tps = rates.get("chitu_total_generated_tokens", {})
                eviction_rate = rates.get("chitu_total_task_evictions", {})
                kvcache_usage = latest_values.get("chitu_kv_cache_usage_ratio", {})
                used_blocks = latest_values.get("chitu_used_blocks", {})
                total_blocks = latest_values.get("chitu_total_blocks", {})
                cuda_total_bytes = latest_values.get("chitu_cuda_total_bytes", {})
                cuda_used_bytes = latest_values.get("chitu_cuda_used_bytes", {})
                torch_allocated_bytes = latest_values.get(
                    "chitu_torch_allocated_bytes", {}
                )
                torch_reserved_bytes = latest_values.get(
                    "chitu_torch_reserved_bytes", {}
                )
                mtp_proposed_rate = rates.get("chitu_mtp_proposed_tokens", {})
                mtp_accepted_rate = rates.get("chitu_mtp_accepted_tokens", {})
                total_hit_tokens = latest_values.get("chitu_total_hit_tokens", {})
                total_prompt_tokens = latest_values.get("chitu_total_prompt_tokens", {})
                running_requests = latest_values.get("chitu_running_requests", {})
                waiting_requests = latest_values.get("chitu_waiting_requests", {})
                prealloc_blocks = latest_values.get("chitu_prealloc_blocks", {})
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
                    running_requests,
                    waiting_requests,
                    prealloc_blocks,
                )
            except Exception as e:
                logger.error(f"Metrics query failed: {e}")

    def _print_stats(
        self,
        prompt_tps: dict[tuple[str, str, str], str],
        gen_tps: dict[tuple[str, str, str], str],
        eviction_rate: dict[tuple[str, str, str], str],
        kvcache_usage: dict[tuple[str, str, str], str],
        used_blocks: dict[tuple[str, str, str], str],
        total_blocks: dict[tuple[str, str, str], str],
        cuda_total_bytes: dict[tuple[str, str, str], str],
        cuda_used_bytes: dict[tuple[str, str, str], str],
        torch_allocated_bytes: dict[tuple[str, str, str], str],
        torch_reserved_bytes: dict[tuple[str, str, str], str],
        total_hit_tokens: dict[tuple[str, str, str], str],
        total_prompt_tokens: dict[tuple[str, str, str], str],
        mtp_proposed_rate: Optional[dict[tuple[str, str, str], str]] = None,
        mtp_accepted_rate: Optional[dict[tuple[str, str, str], str]] = None,
        running_requests: Optional[dict[tuple[str, str, str], str]] = None,
        waiting_requests: Optional[dict[tuple[str, str, str], str]] = None,
        prealloc_blocks: Optional[dict[tuple[str, str, str], str]] = None,
    ):
        mtp_proposed_rate = mtp_proposed_rate or {}
        mtp_accepted_rate = mtp_accepted_rate or {}
        running_requests = running_requests or {}
        waiting_requests = waiting_requests or {}
        prealloc_blocks = prealloc_blocks or {}

        stats_parts: dict[tuple[str, str, str], list[str]] = {}

        def append_part(rank_dp: tuple[str, str, str], part: str):
            stats_parts.setdefault(rank_dp, []).append(part)

        for source, value in prompt_tps.items():
            append_part(source, f"Avg prompt throughput: {float(value):.1f} tokens/s")

        for source, value in gen_tps.items():
            append_part(
                source, f"Avg generation throughput: {float(value):.1f} tokens/s"
            )

        for source, value in eviction_rate.items():
            append_part(source, f"Task evictions: {float(value):.2f}/s")

        for source in total_prompt_tokens:
            prompt_tokens = int(total_prompt_tokens[source])
            hit_tokens = int(total_hit_tokens.get(source, "0"))
            hit_rate = hit_tokens / prompt_tokens if prompt_tokens != 0 else 0
            append_part(
                source, f"Hit rate: {hit_rate*100:.1f}%({hit_tokens}/{prompt_tokens})"
            )

        if mtp_proposed_rate and mtp_accepted_rate:
            for source, proposed_value in mtp_proposed_rate.items():
                proposed = float(proposed_value)
                if proposed <= 0:
                    continue
                accepted = float(mtp_accepted_rate.get(source, "0"))
                append_part(source, f"MTP hit rate: {accepted/proposed*100:.1f}%")

        task_metric_dicts = [
            prompt_tps,
            gen_tps,
            eviction_rate,
            total_hit_tokens,
            total_prompt_tokens,
            running_requests,
            waiting_requests,
            prealloc_blocks,
        ]
        if mtp_proposed_rate:
            task_metric_dicts.append(mtp_proposed_rate)
        if mtp_accepted_rate:
            task_metric_dicts.append(mtp_accepted_rate)
        task_rank_dp_pairs = {
            key for metric_dict in task_metric_dicts for key in metric_dict
        }

        for rank_dp in sorted(task_rank_dp_pairs):
            running = int(running_requests.get(rank_dp, "0"))
            waiting = int(waiting_requests.get(rank_dp, "0"))
            prealloced = int(prealloc_blocks.get(rank_dp, "-1"))
            append_part(rank_dp, f"Running: {running} reqs")
            append_part(rank_dp, f"Waiting: {waiting} reqs")
            if prealloced >= 0:
                append_part(rank_dp, f"KV blocks prealloc: {prealloced}")

        kvcache_rank_dp_pairs = (
            set(kvcache_usage) | set(used_blocks) | set(total_blocks)
        )
        for rank_dp in sorted(kvcache_rank_dp_pairs):
            used_blocks_value = int(used_blocks.get(rank_dp, "0"))
            total_blocks_value = int(total_blocks.get(rank_dp, "0"))
            kv_cache_usage_value = float(kvcache_usage.get(rank_dp, "-1"))
            if total_blocks_value > 0:
                append_part(
                    rank_dp,
                    f"KV cache usage: {kv_cache_usage_value*100:.1f}%"
                    f"({used_blocks_value}/{total_blocks_value})",
                )

        gpu_rank_dp_pairs = set(cuda_total_bytes) | set(cuda_used_bytes)
        for source in sorted(gpu_rank_dp_pairs):
            cuda_total = float(cuda_total_bytes.get(source, "0"))
            cuda_used = float(cuda_used_bytes.get(source, "0"))
            if cuda_total <= 0 or cuda_used < 0:
                continue
            parts = stats_parts.setdefault(source, [])
            torch_allocated = float(torch_allocated_bytes.get(source, "-1"))
            torch_reserved = float(torch_reserved_bytes.get(source, "-1"))
            used_gib = cuda_used / 1024**3
            total_gib = cuda_total / 1024**3
            if torch_allocated >= 0 and torch_reserved >= 0:
                torch_allocated_gib = torch_allocated / 1024**3
                torch_reserved_gib = torch_reserved / 1024**3
                torch_unused_gib = torch_reserved_gib - torch_allocated_gib
                non_torch_gib = max(cuda_used - torch_reserved, 0.0) / 1024**3
                parts.append(
                    "GPU mem: "
                    f"{used_gib:.2f}/{total_gib:.2f} GiB "
                    f"(torch-allocated: {torch_allocated_gib:.2f} GiB, torch-unused: {torch_unused_gib:.2f} "
                    f"GiB, non-torch: {non_torch_gib:.2f} GiB)"
                )
            else:
                parts.append(f"GPU mem: {used_gib:.2f}/{total_gib:.2f} GiB")

        for source in sorted(stats_parts):
            logger.info(self.metrics_formatter(source, stats_parts[source]))


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
    else:
        logger.info(
            "Grafana auto-start is disabled. Set metrics.grafana_enabled=true to start Grafana."
        )

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
