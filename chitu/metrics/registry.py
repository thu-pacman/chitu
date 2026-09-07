# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime registry and process-wide context for Prometheus metric objects."""

from __future__ import annotations

from types import TracebackType
from typing import Any, TypeAlias
import threading

from prometheus_client import Counter, Gauge, Histogram, REGISTRY

from chitu.metrics.definitions import (
    MetricContext,
    MetricDefinition,
    MetricType,
    get_metric_definition,
    raw_metrics_for_context,
)

PrometheusMetric: TypeAlias = Counter | Gauge | Histogram


class MetricRegistry:
    """Create and cache Prometheus metrics by definition name."""

    def __init__(self):
        self._metrics: dict[str, PrometheusMetric] = {}
        self._lock = threading.RLock()

    def get(self, name: str) -> PrometheusMetric:
        with self._lock:
            metric = self._metrics.get(name)
            if metric is None:
                metric = self._create(get_metric_definition(name))
                self._metrics[name] = metric
            return metric

    def get_for_context(
        self, name: str, context: MetricContext
    ) -> PrometheusMetric | None:
        if name not in {metric.name for metric in raw_metrics_for_context(context)}:
            return None
        return self.get(name)

    def get_all_for_context(
        self, context: MetricContext
    ) -> dict[str, PrometheusMetric]:
        return {
            metric.name: self.get(metric.name)
            for metric in raw_metrics_for_context(context)
        }

    def items(self) -> tuple[tuple[str, PrometheusMetric], ...]:
        with self._lock:
            return tuple(self._metrics.items())

    def clear(self) -> None:
        with self._lock:
            self._metrics.clear()

    def _create(self, metric: MetricDefinition) -> PrometheusMetric:
        if metric.type == MetricType.COUNTER:
            if metric.labels:
                return Counter(metric.name, metric.help_en, list(metric.labels))
            return Counter(metric.name, metric.help_en)
        if metric.type == MetricType.GAUGE:
            if metric.labels:
                return Gauge(metric.name, metric.help_en, list(metric.labels))
            return Gauge(metric.name, metric.help_en)
        if metric.type == MetricType.HISTOGRAM:
            kwargs: dict[str, Any] = {}
            if metric.buckets is not None:
                kwargs["buckets"] = metric.buckets
            if metric.labels:
                return Histogram(
                    metric.name, metric.help_en, list(metric.labels), **kwargs
                )
            return Histogram(metric.name, metric.help_en, **kwargs)
        raise ValueError(f"Unsupported metric type: {metric.type}")


class MetricsRuntimeContext:
    """Own the process-wide Prometheus metric lifecycle for one metric context."""

    def __init__(
        self,
        metric_context: MetricContext,
        *,
        start_http_server: bool = True,
    ):
        self.metric_context = metric_context
        self.start_http_server = start_http_server
        self.registry = MetricRegistry()
        self._entered = False

    def __enter__(self) -> "MetricsRuntimeContext":
        global _active_metrics_runtime
        with _active_metrics_runtime_lock:
            if _active_metrics_runtime is not None:
                raise RuntimeError("Nested metrics runtime contexts are not supported")
            _active_metrics_runtime = self
            self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        global _active_metrics_runtime
        with _active_metrics_runtime_lock:
            if not self._entered:
                return
            if _active_metrics_runtime is not self:
                raise RuntimeError(
                    "Cannot close a metrics runtime context that is not active"
                )

            from chitu.metrics.prometheus_collector import PrometheusMetricsCollector

            PrometheusMetricsCollector._stop_instance_resources()
            for metric_name, metric in self.registry.items():
                try:
                    REGISTRY.unregister(metric)
                except Exception:
                    pass
            self.registry.clear()
            _active_metrics_runtime = None
            self._entered = False


_active_metrics_runtime: MetricsRuntimeContext | None = None
_active_metrics_runtime_lock = threading.RLock()


def metrics_runtime_context(
    metric_context: MetricContext,
    *,
    start_http_server: bool = True,
) -> MetricsRuntimeContext:
    return MetricsRuntimeContext(
        metric_context,
        start_http_server=start_http_server,
    )


def get_active_metrics_runtime() -> MetricsRuntimeContext | None:
    return _active_metrics_runtime


def close_active_metrics_runtime() -> None:
    runtime = get_active_metrics_runtime()
    if runtime is not None:
        runtime.close()
