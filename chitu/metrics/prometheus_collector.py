# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
import threading
from contextlib import contextmanager
from typing import Optional
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import atexit
import torch

from chitu.backend import Backend
from chitu.distributed.parallel_state import (
    get_dp_group,
    get_tp_group,
    get_pp_group,
    get_pcp_group,
    get_dp_size,
)
from chitu.boot.tcp_ip import get_local_ip, get_free_port
from chitu.global_vars import get_global_args
from chitu.metrics.cache_stats import kvcache_stats, get_prealloc_blocks
from chitu.metrics.task_stats import count_tasks_for_dp_rank, count_tasks_non_dp
from chitu.accelerator_monitor import get_accelerator_memory_bytes
from chitu.metrics.registry import (
    close_active_metrics_runtime,
    get_active_metrics_runtime,
)
from chitu.metrics.definitions import MetricContext, MetricType, raw_metrics_for_context

logger = logging.getLogger(__name__)

_COLLECTOR_ATTR_BY_METRIC_NAME = {
    "chitu_total_generated_tokens": "total_generated_tokens",
    "chitu_total_prompt_tokens": "total_prompt_tokens",
    "chitu_total_hit_tokens": "total_hit_tokens",
    "chitu_total_task_evictions": "total_task_evictions",
    "chitu_mtp_proposed_tokens": "mtp_proposed_tokens",
    "chitu_mtp_accepted_tokens": "mtp_accepted_tokens",
    "chitu_kv_cache_usage_ratio": "kv_cache_usage",
    "chitu_used_blocks": "used_blocks",
    "chitu_total_blocks": "total_blocks",
    "chitu_cuda_total_bytes": "cuda_total_bytes",
    "chitu_cuda_used_bytes": "cuda_used_bytes",
    "chitu_torch_allocated_bytes": "torch_allocated_bytes",
    "chitu_torch_reserved_bytes": "torch_reserved_bytes",
    "chitu_running_requests": "running_requests",
    "chitu_waiting_requests": "waiting_requests",
    "chitu_prealloc_blocks": "prealloc_blocks",
}

_COLLECTOR_ZERO_INIT_METRICS = {
    "chitu_total_generated_tokens",
    "chitu_total_prompt_tokens",
    "chitu_total_hit_tokens",
    "chitu_total_task_evictions",
    "chitu_mtp_proposed_tokens",
    "chitu_mtp_accepted_tokens",
    "chitu_cuda_total_bytes",
    "chitu_cuda_used_bytes",
    "chitu_torch_allocated_bytes",
    "chitu_torch_reserved_bytes",
    "chitu_running_requests",
    "chitu_waiting_requests",
}


# 部分参考自 sglang 的 metrics.py和 vllm
# ---------------------------------------------------------------------------
# PD Disaggregation Metrics
# ---------------------------------------------------------------------------

_ROLE_CONTEXT = lambda role: MetricContext(inst_role=role)
_SERVICE_CONTEXT = MetricContext(is_router=True)


def _current_instance_rank_labels() -> dict[str, str]:
    collector = PrometheusMetricsCollector.get_instance()
    if collector is None:
        raise RuntimeError("Prometheus metrics collector is not initialized")
    return {"instance_id": str(collector.instance_id), "rank": str(collector.rank)}


def _role_metric(name: str, role: str):
    runtime = get_active_metrics_runtime()
    if runtime is None:
        return None
    metric = runtime.registry.get_for_context(name, _ROLE_CONTEXT(role))
    if metric is None:
        raise RuntimeError(f"Metric {name} is not collected for role {role}")
    return metric


def _service_metric(name: str):
    runtime = get_active_metrics_runtime()
    if runtime is None:
        return None
    metric = runtime.registry.get_for_context(name, _SERVICE_CONTEXT)
    if metric is None:
        raise RuntimeError(f"Metric {name} is not collected by the service context")
    return metric


def _rank_metric(name: str):
    runtime = get_active_metrics_runtime()
    if runtime is None:
        return None
    collector = PrometheusMetricsCollector.get_instance()
    if collector is None:
        raise RuntimeError("Prometheus metrics collector is not initialized")
    metric = runtime.registry.get_for_context(name, collector.metric_context)
    if metric is None:
        raise RuntimeError(
            f"Metric {name} is not collected for rank={collector.rank} instance={collector.instance_id}"
        )
    return metric


# ---------------------------------------------------------------------------
# Lightweight helpers (zero-allocation hot path)
# ---------------------------------------------------------------------------


@contextmanager
def observe_pd_stage(role: str, stage: str):
    """Context-manager that records stage duration into the Histogram.

    Uses ``time.monotonic()`` to avoid wall-clock jumps.  The cost is a
    single ``Histogram.observe(float)`` call on exit -- no string formatting
    or memory allocation on the hot path.
    """
    start = time.monotonic()
    try:
        yield
    finally:
        duration = time.monotonic() - start
        metric = _role_metric("chitu_pd_stage_duration_seconds", role)
        if metric is not None:
            metric.labels(role=role, stage=stage).observe(duration)


def observe_stage_duration(role: str, stage: str, duration_s: float):
    """Record a pre-computed stage duration (seconds) into the Histogram."""
    if duration_s >= 0:
        metric = _role_metric("chitu_pd_stage_duration_seconds", role)
        if metric is not None:
            metric.labels(role=role, stage=stage).observe(duration_s)


def observe_e2e_duration(duration_s: float):
    """Record end-to-end request duration."""
    if duration_s >= 0:
        metric = _service_metric("chitu_e2e_request_duration_seconds")
        if metric is not None:
            metric.observe(duration_s)


def observe_ttft(duration_s: float):
    """Record time-to-first-token."""
    if duration_s >= 0:
        metric = _service_metric("chitu_time_to_first_token_seconds")
        if metric is not None:
            metric.observe(duration_s)


def observe_kv_transfer(size_bytes: float = 0, duration_s: float | None = None):
    """Record KV transfer size and duration."""
    if get_active_metrics_runtime() is None:
        return
    labels = _current_instance_rank_labels()
    if size_bytes > 0:
        metric = _rank_metric("chitu_kv_transfer_size_bytes")
        assert metric is not None
        metric.labels(**labels).observe(size_bytes)
    if duration_s is not None and duration_s >= 0:
        metric = _rank_metric("chitu_kv_transfer_duration_seconds")
        assert metric is not None
        metric.labels(**labels).observe(duration_s)


def set_queue_size(role: str, queue_name: str, size: int):
    """Set a queue depth gauge."""
    metric = _role_metric("chitu_pd_queue_size", role)
    if metric is not None:
        metric.labels(role=role, queue_name=queue_name).set(size)


def inc_completed_requests(role: str, count: int = 1):
    """Increment completed requests counter."""
    if get_active_metrics_runtime() is None:
        return
    metric = _rank_metric("chitu_completed_requests_total")
    assert metric is not None
    metric.labels(role=role, **_current_instance_rank_labels()).inc(count)


def inc_kv_transfer_failures(role: str, count: int = 1):
    """Increment KV transfer failure counter."""
    if get_active_metrics_runtime() is None:
        return
    metric = _rank_metric("chitu_kv_transfer_failures_total")
    assert metric is not None
    metric.labels(role=role, **_current_instance_rank_labels()).inc(count)


def inc_request_timeouts(stage: str, count: int = 1):
    """Increment request timeout counter."""
    metric = _service_metric("chitu_request_timeouts_total")
    if metric is not None:
        metric.labels(stage=stage).inc(count)


def set_router_pending_requests(size: int):
    """Set the service-level router pending-request gauge."""
    metric = _service_metric("chitu_router_pending_requests")
    if metric is not None:
        metric.set(size)


def set_active_requests(role: str, size: int):
    """Set the active-request gauge for a role."""
    metric = _role_metric("chitu_active_requests", role)
    if metric is not None:
        metric.labels(role=role).set(size)


class PrometheusMetricsCollector:
    """Singleton metrics collector for Prometheus"""

    _instance: Optional["PrometheusMetricsCollector"] = None
    _lock = threading.RLock()
    _shutting_down = False
    addrs: Optional[list[str]] = None

    @classmethod
    def metric_context_from_distributed(cls) -> MetricContext:
        dp_group = get_dp_group()
        tp_pcp_pp_group = (
            get_tp_group()
            .cartesian_product(get_pcp_group())
            .cartesian_product(get_pp_group())
        )
        multi_inst = getattr(get_global_args(), "multi_inst", None)
        inst_id = 0 if multi_inst is None else multi_inst.inst_id
        inst_role = None if multi_inst is None else getattr(multi_inst, "role", None)
        return MetricContext(
            rank=dp_group.global_rank,
            rank_in_dp=tp_pcp_pp_group.rank_in_group,
            dp_rank=dp_group.rank_in_group,
            inst_id=inst_id,
            inst_role=inst_role,
        )

    @classmethod
    def get_instance(
        cls, is_create: bool = False
    ) -> Optional["PrometheusMetricsCollector"]:
        if cls._shutting_down:
            return None

        if cls._instance is not None:
            return cls._instance

        if not is_create:
            return None

        with cls._lock:
            if cls._instance is None:
                runtime = get_active_metrics_runtime()
                if runtime is None:
                    raise RuntimeError(
                        "Prometheus metrics collector requires an active metrics runtime context"
                    )
                metric_context = runtime.metric_context
                cls._instance = cls(
                    rank=0 if metric_context.rank is None else metric_context.rank,
                    rank_in_dp=(
                        0
                        if metric_context.rank_in_dp is None
                        else metric_context.rank_in_dp
                    ),
                    dp_rank=(
                        0 if metric_context.dp_rank is None else metric_context.dp_rank
                    ),
                    inst_id=(
                        0 if metric_context.inst_id is None else metric_context.inst_id
                    ),
                    is_router=metric_context.is_router,
                    metric_context=metric_context,
                )
                if cls._instance.addr is not None:
                    cls.addrs = [cls._instance.addr]
                else:
                    cls.addrs = []
        return cls._instance

    @classmethod
    def get_router_instance(
        cls, is_create: bool = False
    ) -> Optional["PrometheusMetricsCollector"]:
        if cls._shutting_down:
            return None

        if cls._instance is not None:
            return cls._instance

        if not is_create:
            return None

        with cls._lock:
            if cls._instance is None:
                runtime = get_active_metrics_runtime()
                if runtime is None:
                    raise RuntimeError(
                        "Prometheus router metrics collector requires an active metrics runtime context"
                    )
                if not runtime.metric_context.is_router:
                    raise RuntimeError(
                        "Router metrics collector requires a router context"
                    )
                cls._instance = cls(
                    is_router=True,
                    metric_context=runtime.metric_context,
                )
                if cls._instance.addr is not None:
                    cls.addrs = [cls._instance.addr]
                else:
                    cls.addrs = []
        return cls._instance

    def __init__(
        self,
        rank: int = 0,
        rank_in_dp: int = 0,
        dp_rank: int = 0,
        inst_id: int = 0,
        is_router: bool = False,
        metric_context: MetricContext | None = None,
    ):
        self.rank: int = rank
        self.rank_in_dp: int = rank_in_dp
        self.dp_id: int = dp_rank
        self.instance_id: int = inst_id
        self.is_dp_metrics_rank = rank_in_dp == 0
        self.is_main_rank = rank == 0
        self.is_router = is_router
        self.metric_context: MetricContext
        self.dp_size = get_dp_size()

        self.total_generated_tokens: Optional[Counter] = None
        self.total_prompt_tokens: Optional[Counter] = None
        self.total_hit_tokens: Optional[Counter] = None
        self.total_task_evictions: Optional[Counter] = None
        self.mtp_proposed_tokens: Optional[Counter] = None
        self.mtp_accepted_tokens: Optional[Counter] = None
        self.kv_cache_usage: Optional[Gauge] = None
        self.used_blocks: Optional[Gauge] = None
        self.total_blocks: Optional[Gauge] = None
        self.cuda_total_bytes: Optional[Gauge] = None
        self.cuda_used_bytes: Optional[Gauge] = None
        self.torch_allocated_bytes: Optional[Gauge] = None
        self.torch_reserved_bytes: Optional[Gauge] = None
        self.running_requests: Optional[Gauge] = None
        self.waiting_requests: Optional[Gauge] = None
        self.prealloc_blocks: Optional[Gauge] = None
        self.collector_server = None
        self.collector_thread = None
        self.ip: Optional[str] = None
        self.port: Optional[int] = None

        try:
            if metric_context is None:
                metric_context = MetricContext(
                    rank=rank,
                    rank_in_dp=rank_in_dp,
                    dp_rank=dp_rank,
                    inst_id=inst_id,
                    is_router=is_router,
                )
            self.metric_context = metric_context
            runtime = get_active_metrics_runtime()
            if runtime is None:
                raise RuntimeError(
                    "Prometheus metrics collector requires an active metrics runtime context"
                )
            metric_definitions = raw_metrics_for_context(metric_context)
            metrics = runtime.registry.get_all_for_context(metric_context)

            for metric in metric_definitions:
                prometheus_metric = metrics[metric.name]
                attr_name = _COLLECTOR_ATTR_BY_METRIC_NAME.get(metric.name)
                if attr_name is not None:
                    setattr(self, attr_name, prometheus_metric)
                if metric.name not in _COLLECTOR_ZERO_INIT_METRICS:
                    continue
                labeled_metric = prometheus_metric.labels(
                    rank=rank, dp_id=dp_rank, instance_id=inst_id
                )
                if metric.type == MetricType.COUNTER:
                    labeled_metric.inc(0)
                elif metric.type == MetricType.GAUGE:
                    labeled_metric.set(0)
            self._initialize_common_labeled_metrics(metric_context)
            # init kv cache and prealloc blocks via update_kvcache_usage
            self._update_kvcache_usage()

            if runtime.start_http_server:
                try:
                    ip = get_local_ip()
                except Exception as e:
                    logger.warning(
                        f"Failed to get local IP: {e}, use 127.0.0.1 as default"
                    )
                    ip = "127.0.0.1"
                port = get_free_port()
                self.collector_server, self.collector_thread = start_http_server(
                    port
                )  # Prometheus serve pull data from this server
                self.ip = ip
                self.port = port
                logger.info(f"Prometheus metrics server started on addr {self.addr}")
            else:
                self.ip = "127.0.0.1"
                self.port = 0
            atexit.register(PrometheusMetricsCollector.stop_instance)

        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics collector: {e}")
            raise

    @property
    def addr(self) -> Optional[str]:
        if self.ip is None or self.port is None:
            return None
        return f"{self.ip}:{self.port}"

    def _initialize_common_labeled_metrics(self, metric_context: MetricContext) -> None:
        runtime = get_active_metrics_runtime()
        if runtime is None:
            raise RuntimeError("Prometheus metrics collector is not initialized")
        if metric_context.inst_role in {"prefill", "decode"}:
            role = metric_context.inst_role
            runtime.registry.get("chitu_kv_transfer_failures_total").labels(
                role=role,
                instance_id=str(metric_context.inst_id),
                rank=str(metric_context.rank),
            ).inc(0)
            runtime.registry.get("chitu_completed_requests_total").labels(
                role=role,
                instance_id=str(metric_context.inst_id),
                rank=str(metric_context.rank),
            ).inc(0)
        if metric_context.is_router:
            for stage in (
                "ttft_router",
                "ttft_pd_router",
                "ttft_scheduler",
                "ttft_pd_prefill",
                "ttft_pd_decode_incoming",
                "ttft_pd_decode_prealloc",
            ):
                _service_metric("chitu_request_timeouts_total").labels(stage=stage).inc(
                    0
                )
            _service_metric("chitu_router_pending_requests").set(0)

    @classmethod
    def _metric_attr_or_none(cls, metric_name: str, attr_name: str):
        if get_active_metrics_runtime() is None:
            return None, None
        collector = cls.get_instance()
        if not collector:
            raise RuntimeError("Prometheus metrics collector is not initialized")
        if metric_name not in {
            metric.name for metric in raw_metrics_for_context(collector.metric_context)
        }:
            return collector, None
        metric = getattr(collector, attr_name)
        if metric is None:
            raise RuntimeError(
                f"Metric {metric_name} is selected but not initialized for rank={collector.rank} instance={collector.instance_id}"
            )
        return collector, metric

    @classmethod
    def inc_generated_tokens(cls, count: int = 1):
        if count < 0:
            return

        collector, metric = cls._metric_attr_or_none(
            "chitu_total_generated_tokens", "total_generated_tokens"
        )
        if collector is None or metric is None:
            return
        try:
            metric.labels(
                rank=collector.rank,
                dp_id=collector.dp_id,
                instance_id=collector.instance_id,
            ).inc(count)
        except Exception as e:
            logger.error(f"inc_generated_tokens failed: {e}")

    @classmethod
    def inc_mtp_tokens(cls, proposed: int, accepted: int):
        if proposed <= 0:
            return

        collector, proposed_metric = cls._metric_attr_or_none(
            "chitu_mtp_proposed_tokens", "mtp_proposed_tokens"
        )
        _, accepted_metric = cls._metric_attr_or_none(
            "chitu_mtp_accepted_tokens", "mtp_accepted_tokens"
        )
        if collector is None or proposed_metric is None or accepted_metric is None:
            return
        try:
            proposed_metric.labels(
                rank=collector.rank,
                dp_id=collector.dp_id,
                instance_id=collector.instance_id,
            ).inc(proposed)
            accepted_metric.labels(
                rank=collector.rank,
                dp_id=collector.dp_id,
                instance_id=collector.instance_id,
            ).inc(accepted)
        except Exception as e:
            logger.error(f"inc_mtp_tokens failed: {e}")

    @classmethod
    def get_mtp_stats(cls) -> tuple[int, int]:
        """Return (total_proposed, total_accepted) across all DP ranks on this process."""
        collector = cls.get_instance()
        if not collector:
            return 0, 0
        total_proposed = 0
        total_accepted = 0
        if collector.mtp_proposed_tokens:
            for metric in collector.mtp_proposed_tokens.collect():
                for sample in metric.samples:
                    if sample.name.endswith("_total"):
                        total_proposed += int(sample.value)
        if collector.mtp_accepted_tokens:
            for metric in collector.mtp_accepted_tokens.collect():
                for sample in metric.samples:
                    if sample.name.endswith("_total"):
                        total_accepted += int(sample.value)
        return total_proposed, total_accepted

    @classmethod
    def inc_prompt_tokens(cls, count: int = 1):
        if count < 0:
            return

        collector, metric = cls._metric_attr_or_none(
            "chitu_total_prompt_tokens", "total_prompt_tokens"
        )
        if collector is None or metric is None:
            return
        try:
            metric.labels(
                rank=collector.rank,
                dp_id=collector.dp_id,
                instance_id=collector.instance_id,
            ).inc(count)
        except Exception as e:
            logger.error(f"inc_prompt_tokens failed: {e}")

    @classmethod
    def inc_hit_tokens(cls, count: int = 1):
        if count < 0:
            return

        collector, metric = cls._metric_attr_or_none(
            "chitu_total_hit_tokens", "total_hit_tokens"
        )
        if collector is None or metric is None:
            return

        try:
            metric.labels(
                rank=collector.rank,
                dp_id=collector.dp_id,
                instance_id=collector.instance_id,
            ).inc(count)
        except Exception as e:
            logger.error(f"inc_hit_tokens failed: {e}")

    @classmethod
    def update_task_counts(cls):
        """Update running/waiting request count metrics."""
        collector, running_metric = cls._metric_attr_or_none(
            "chitu_running_requests", "running_requests"
        )
        _, waiting_metric = cls._metric_attr_or_none(
            "chitu_waiting_requests", "waiting_requests"
        )
        if collector is None or running_metric is None or waiting_metric is None:
            return
        try:
            if collector.is_main_rank:
                running, waiting = count_tasks_for_dp_rank(collector.dp_id)
            else:
                running, waiting = count_tasks_non_dp()
            running_metric.labels(
                rank=collector.rank,
                dp_id=collector.dp_id,
                instance_id=collector.instance_id,
            ).set(running)
            waiting_metric.labels(
                rank=collector.rank,
                dp_id=collector.dp_id,
                instance_id=collector.instance_id,
            ).set(waiting)
        except Exception as e:
            logger.error(f"update_task_counts failed: {e}")

    def _update_kvcache_usage(self):
        if Backend.cache_dict is None or "main" not in Backend.cache_dict:
            return
        if not self.total_blocks or not self.used_blocks or not self.kv_cache_usage:
            raise RuntimeError(
                f"KV cache metrics are not collected for rank={self.rank} instance={self.instance_id}"
            )

        kvcache_stats_dict = kvcache_stats(self.is_main_rank, self.dp_id)
        for dp_id in kvcache_stats_dict:
            rank = get_dp_group().rank_list[dp_id]
            total_blocks, used_blocks, kvcache_usage = kvcache_stats_dict[dp_id]
            self.total_blocks.labels(
                rank=rank, dp_id=dp_id, instance_id=self.instance_id
            ).set(total_blocks)
            self.used_blocks.labels(
                rank=rank, dp_id=dp_id, instance_id=self.instance_id
            ).set(used_blocks)
            if kvcache_usage >= 0:
                self.kv_cache_usage.labels(
                    rank=rank, dp_id=dp_id, instance_id=self.instance_id
                ).set(kvcache_usage)
            else:
                logger.error(
                    f"Unexpected {type(Backend.cache_dict['main']).__name__}.num_blocks({total_blocks}), update_kvcache_usage failed. "
                )
        if self.prealloc_blocks is not None:
            prealloc_blocks_dict = get_prealloc_blocks(self.dp_size)
            if prealloc_blocks_dict:
                for dp_id in prealloc_blocks_dict:
                    rank = get_dp_group().rank_list[dp_id]
                    prealloc_blocks = prealloc_blocks_dict[dp_id]
                    self.prealloc_blocks.labels(
                        rank=rank, dp_id=dp_id, instance_id=self.instance_id
                    ).set(prealloc_blocks)

    @classmethod
    def update_kvcache_usage(cls):
        """Update KV cache usage metrics."""
        collector, metric = cls._metric_attr_or_none(
            "chitu_kv_cache_usage_ratio", "kv_cache_usage"
        )
        if collector is None or metric is None:
            return

        try:
            collector._update_kvcache_usage()
        except Exception as e:
            logger.error(f"update_kvcache_usage failed: {e}")

    @classmethod
    def update_GPU_usage(cls):
        if get_active_metrics_runtime() is None:
            return
        if Backend.cache_dict is None:
            return

        cls.update_kvcache_usage()

        collector = cls.get_instance()
        if not collector:
            raise RuntimeError("Prometheus metrics collector is not initialized")
        if (
            not collector.cuda_total_bytes
            or not collector.cuda_used_bytes
            or not collector.torch_allocated_bytes
            or not collector.torch_reserved_bytes
        ):
            raise RuntimeError(
                f"GPU memory metrics are not collected for rank={collector.rank} instance={collector.instance_id}"
            )

        try:
            device = Backend.cache_dict["main"].device
            if (not isinstance(device, torch.device)) or (
                isinstance(device, torch.device) and device.type != "cuda"
            ):
                return

            device_index = device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            mem_info = get_accelerator_memory_bytes(device_index, os.getpid())
            if mem_info is not None:
                cuda_used_bytes, cuda_total_bytes = mem_info
            else:
                free_bytes, cuda_total_bytes = torch.cuda.mem_get_info(device_index)
                cuda_used_bytes = cuda_total_bytes - free_bytes
            memory_stats = torch.cuda.memory_stats(device_index)
            torch_allocated = memory_stats["allocated_bytes.all.current"]
            torch_reserved = memory_stats["reserved_bytes.all.current"]
            collector.cuda_total_bytes.labels(
                rank=collector.rank,
                dp_id=collector.dp_id,
                instance_id=collector.instance_id,
            ).set(cuda_total_bytes)
            collector.cuda_used_bytes.labels(
                rank=collector.rank,
                dp_id=collector.dp_id,
                instance_id=collector.instance_id,
            ).set(cuda_used_bytes)
            collector.torch_allocated_bytes.labels(
                rank=collector.rank,
                dp_id=collector.dp_id,
                instance_id=collector.instance_id,
            ).set(torch_allocated)
            collector.torch_reserved_bytes.labels(
                rank=collector.rank,
                dp_id=collector.dp_id,
                instance_id=collector.instance_id,
            ).set(torch_reserved)

        except Exception as e:
            logger.error(f"update_GPU_usage failed: {e}")
            pass

    @classmethod
    def inc_task_eviction(cls):
        """Increment task eviction counter"""
        collector, metric = cls._metric_attr_or_none(
            "chitu_total_task_evictions", "total_task_evictions"
        )
        if collector is None or metric is None:
            return
        try:
            metric.labels(
                rank=collector.rank,
                dp_id=collector.dp_id,
                instance_id=collector.instance_id,
            ).inc()
        except Exception as e:
            logger.error(f"inc_task_eviction failed: {e}")
            pass

    @classmethod
    def _stop_instance_resources(cls):
        cls._shutting_down = True
        with cls._lock:
            instance = cls._instance
            if instance is None:
                cls._shutting_down = False
                return
            try:
                if instance.collector_server:
                    instance.collector_server.shutdown()
                    instance.collector_server.server_close()
                if instance.collector_thread:
                    instance.collector_thread.join(timeout=5.0)
                logger.info(
                    f"[rank {instance.rank}]: Prometheus metrics collector stopped"
                )
            except Exception as e:
                logger.error(f"Error during collector cleanup: {e}")
            finally:
                cls._instance = None
                cls.addrs = None
                cls._shutting_down = False

    @classmethod
    def stop_instance(cls):
        runtime = get_active_metrics_runtime()
        if runtime is not None:
            close_active_metrics_runtime()
            return
        cls._stop_instance_resources()
