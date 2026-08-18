# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Typed metric metadata used by runtime construction and documentation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


class MetricCategory(str, Enum):
    THROUGHPUT = "Throughput"
    REQUESTS = "Requests"
    KV_CACHE = "KV cache"
    MEMORY = "Memory"
    PD_DISAGGREGATION = "PD disaggregation"
    ERRORS = "Errors and timeouts"


class MetricScope(str, Enum):
    PER_RANK = "per_rank"
    PER_DP_RANK = "per_dp_rank"
    PER_INSTANCE = "per_instance"
    PER_ROLE = "per_role"
    WHOLE_SERVICE = "whole_service"


@dataclass(frozen=True)
class MetricContext:
    rank: int | None = None
    rank_in_dp: int | None = None
    dp_rank: int | None = None
    inst_id: int | None = None
    inst_role: str | None = None
    is_router: bool = False


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    type: MetricType
    help_en: str
    help_zh: str
    labels: tuple[str, ...] = ()
    unit: str | None = None
    category: MetricCategory | str = "Other"
    scope: MetricScope = MetricScope.WHOLE_SERVICE

    buckets: tuple[float, ...] | None = None
    stdout: bool = False
    grafana: bool = False
    notes_en: tuple[str, ...] = ()
    notes_zh: tuple[str, ...] = ()


@dataclass(frozen=True)
class DerivedMetricDefinition:
    name: str
    help_en: str
    help_zh: str
    expression: str
    category: MetricCategory | str
    source_metrics: tuple[str, ...]
    scope: MetricScope = MetricScope.WHOLE_SERVICE

    stdout: bool = False
    grafana: bool = False
    notes_en: tuple[str, ...] = ()
    notes_zh: tuple[str, ...] = ()


PD_STAGE_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
)

RAW_METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        name="chitu_pd_stage_duration_seconds",
        type=MetricType.HISTOGRAM,
        help_en="PD disaggregation per-stage request latency in seconds.",
        help_zh="PD 分离各阶段请求延迟（秒）。",
        labels=("role", "stage"),
        unit="seconds",
        category=MetricCategory.PD_DISAGGREGATION,
        scope=MetricScope.PER_ROLE,
        buckets=PD_STAGE_BUCKETS,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_e2e_request_duration_seconds",
        type=MetricType.HISTOGRAM,
        help_en="End-to-end request latency from router receive to decode completion.",
        help_zh="从 router 接收请求到 decode 完成的端到端请求延迟。",
        unit="seconds",
        category=MetricCategory.REQUESTS,
        scope=MetricScope.WHOLE_SERVICE,
        buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_time_to_first_token_seconds",
        type=MetricType.HISTOGRAM,
        help_en="Time from request arrival to first token generated.",
        help_zh="从请求到达到生成首个 token 的时间。",
        unit="seconds",
        category=MetricCategory.REQUESTS,
        scope=MetricScope.WHOLE_SERVICE,
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_kv_transfer_duration_seconds",
        type=MetricType.HISTOGRAM,
        help_en="KV cache transfer duration in seconds.",
        help_zh="KV cache 传输耗时（秒）。",
        labels=("instance_id", "rank"),
        unit="seconds",
        category=MetricCategory.PD_DISAGGREGATION,
        scope=MetricScope.PER_RANK,
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_kv_transfer_size_bytes",
        type=MetricType.HISTOGRAM,
        help_en="KV cache transfer size in bytes per request.",
        help_zh="每个请求的 KV cache 传输大小（字节）。",
        labels=("instance_id", "rank"),
        unit="bytes",
        category=MetricCategory.PD_DISAGGREGATION,
        scope=MetricScope.PER_RANK,
        buckets=(1e6, 1e7, 5e7, 1e8, 5e8, 1e9),
    ),
    MetricDefinition(
        name="chitu_pd_queue_size",
        type=MetricType.GAUGE,
        help_en="Current queue size for PD disaggregation stages.",
        help_zh="PD 分离各阶段当前队列长度。",
        labels=("role", "queue_name"),
        category=MetricCategory.PD_DISAGGREGATION,
        scope=MetricScope.PER_ROLE,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_router_pending_requests",
        type=MetricType.GAUGE,
        help_en="Number of pending requests in router.",
        help_zh="Router 中待处理请求数量。",
        category=MetricCategory.REQUESTS,
        scope=MetricScope.WHOLE_SERVICE,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_active_requests",
        type=MetricType.GAUGE,
        help_en="Number of active streaming requests.",
        help_zh="活跃流式请求数量。",
        labels=("role",),
        category=MetricCategory.REQUESTS,
        scope=MetricScope.PER_ROLE,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_kv_transfer_failures_total",
        type=MetricType.COUNTER,
        help_en="Total KV transfer failures.",
        help_zh="KV 传输失败总数。",
        labels=("role", "instance_id", "rank"),
        category=MetricCategory.ERRORS,
        scope=MetricScope.PER_RANK,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_request_timeouts_total",
        type=MetricType.COUNTER,
        help_en="Total request timeouts.",
        help_zh="总超时请求数。",
        labels=("stage",),
        category=MetricCategory.ERRORS,
        scope=MetricScope.WHOLE_SERVICE,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_completed_requests_total",
        type=MetricType.COUNTER,
        help_en="Total completed requests.",
        help_zh="完成请求总数。",
        labels=("role", "instance_id", "rank"),
        category=MetricCategory.REQUESTS,
        scope=MetricScope.PER_RANK,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_kv_cache_usage_ratio",
        type=MetricType.GAUGE,
        help_en="KV cache usage ratio (used_blocks / total_blocks).",
        help_zh="KV cache 使用率（used_blocks / total_blocks）。",
        labels=("rank", "dp_id", "instance_id"),
        unit="ratio",
        category=MetricCategory.KV_CACHE,
        scope=MetricScope.PER_RANK,
        stdout=True,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_used_blocks",
        type=MetricType.GAUGE,
        help_en="KV cache used blocks.",
        help_zh="KV cache 已使用 block 数。",
        labels=("rank", "dp_id", "instance_id"),
        category=MetricCategory.KV_CACHE,
        scope=MetricScope.PER_RANK,
        stdout=True,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_total_blocks",
        type=MetricType.GAUGE,
        help_en="KV cache total blocks.",
        help_zh="KV cache 总 block 数。",
        labels=("rank", "dp_id", "instance_id"),
        category=MetricCategory.KV_CACHE,
        scope=MetricScope.PER_RANK,
        stdout=True,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_cuda_total_bytes",
        type=MetricType.GAUGE,
        help_en="Accelerator total memory in bytes.",
        help_zh="加速器总显存（字节）。",
        labels=("rank", "dp_id", "instance_id"),
        unit="bytes",
        category=MetricCategory.MEMORY,
        scope=MetricScope.PER_RANK,
        stdout=True,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_cuda_used_bytes",
        type=MetricType.GAUGE,
        help_en="Accelerator used memory in bytes, including torch allocated memory, torch reserved but unused memory, and other accelerator memory.",
        help_zh="加速器已用显存（字节），包括 torch 已分配显存、torch 已预留未使用显存以及其他加速器显存。",
        labels=("rank", "dp_id", "instance_id"),
        unit="bytes",
        category=MetricCategory.MEMORY,
        scope=MetricScope.PER_RANK,
        stdout=True,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_torch_allocated_bytes",
        type=MetricType.GAUGE,
        help_en="Torch allocated GPU memory in bytes.",
        help_zh="Torch 已分配 GPU 显存（字节）。",
        labels=("rank", "dp_id", "instance_id"),
        unit="bytes",
        category=MetricCategory.MEMORY,
        scope=MetricScope.PER_RANK,
        stdout=True,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_torch_reserved_bytes",
        type=MetricType.GAUGE,
        help_en="Torch reserved GPU memory in bytes, including allocated memory and reserved but unused memory.",
        help_zh="Torch 已预留 GPU 显存（字节），包括已分配显存和已预留未使用显存。",
        labels=("rank", "dp_id", "instance_id"),
        unit="bytes",
        category=MetricCategory.MEMORY,
        scope=MetricScope.PER_RANK,
        stdout=True,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_prealloc_blocks",
        type=MetricType.GAUGE,
        help_en="Number of pre-allocated blocks for PD disaggregation.",
        help_zh="PD 分离预分配 block 数量。",
        labels=("rank", "dp_id", "instance_id"),
        category=MetricCategory.KV_CACHE,
        scope=MetricScope.PER_INSTANCE,
        stdout=True,
    ),
    MetricDefinition(
        name="chitu_total_generated_tokens",
        type=MetricType.COUNTER,
        help_en="Total tokens generated by executor.",
        help_zh="Executor 生成 token 总数。",
        labels=("rank", "dp_id", "instance_id"),
        category=MetricCategory.THROUGHPUT,
        scope=MetricScope.PER_DP_RANK,
        stdout=True,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_total_prompt_tokens",
        type=MetricType.COUNTER,
        help_en="Total prompt tokens processed by executor.",
        help_zh="Executor 处理 prompt token 总数。",
        labels=("rank", "dp_id", "instance_id"),
        category=MetricCategory.THROUGHPUT,
        scope=MetricScope.PER_DP_RANK,
        stdout=True,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_total_hit_tokens",
        type=MetricType.COUNTER,
        help_en="Total prompt tokens hit by prefix caching.",
        help_zh="Prefix cache 命中的 prompt token 总数。",
        labels=("rank", "dp_id", "instance_id"),
        category=MetricCategory.KV_CACHE,
        scope=MetricScope.PER_DP_RANK,
        stdout=True,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_total_task_evictions",
        type=MetricType.COUNTER,
        help_en="Total number of tasks evicted due to insufficient KV cache.",
        help_zh="由于 KV cache 不足而被驱逐的任务总数。",
        labels=("rank", "dp_id", "instance_id"),
        category=MetricCategory.KV_CACHE,
        scope=MetricScope.PER_DP_RANK,
        stdout=True,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_mtp_proposed_tokens",
        type=MetricType.COUNTER,
        help_en="Total MTP proposed tokens (mtp_size-1 per task per decode step).",
        help_zh="MTP 提议 token 总数（每个任务每个 decode step 为 mtp_size-1）。",
        labels=("rank", "dp_id", "instance_id"),
        category=MetricCategory.THROUGHPUT,
        scope=MetricScope.PER_DP_RANK,
        stdout=True,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_mtp_accepted_tokens",
        type=MetricType.COUNTER,
        help_en="Total MTP accepted tokens after verification.",
        help_zh="验证后接受的 MTP token 总数。",
        labels=("rank", "dp_id", "instance_id"),
        category=MetricCategory.THROUGHPUT,
        scope=MetricScope.PER_DP_RANK,
        stdout=True,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_running_requests",
        type=MetricType.GAUGE,
        help_en="Number of currently running requests.",
        help_zh="当前运行中的请求数量。",
        labels=("rank", "dp_id", "instance_id"),
        category=MetricCategory.REQUESTS,
        scope=MetricScope.PER_DP_RANK,
        stdout=True,
        grafana=True,
    ),
    MetricDefinition(
        name="chitu_waiting_requests",
        type=MetricType.GAUGE,
        help_en="Number of currently waiting requests.",
        help_zh="当前等待中的请求数量。",
        labels=("rank", "dp_id", "instance_id"),
        category=MetricCategory.REQUESTS,
        scope=MetricScope.PER_DP_RANK,
        stdout=True,
        grafana=True,
    ),
)

DERIVED_METRICS: tuple[DerivedMetricDefinition, ...] = (
    DerivedMetricDefinition(
        name="chitu_prompt_throughput_tokens_per_second",
        help_en="Prompt token throughput per DP rank.",
        help_zh="每个 DP rank 的 prompt token 吞吐。",
        expression="rate(chitu_total_prompt_tokens_total[interval])",
        category=MetricCategory.THROUGHPUT,
        source_metrics=("chitu_total_prompt_tokens",),
        scope=MetricScope.PER_DP_RANK,
        stdout=True,
        grafana=True,
    ),
    DerivedMetricDefinition(
        name="chitu_generation_throughput_tokens_per_second",
        help_en="Generated token throughput per DP rank.",
        help_zh="每个 DP rank 的生成 token 吞吐。",
        expression="rate(chitu_total_generated_tokens_total[interval])",
        category=MetricCategory.THROUGHPUT,
        source_metrics=("chitu_total_generated_tokens",),
        scope=MetricScope.PER_DP_RANK,
        stdout=True,
        grafana=True,
    ),
    DerivedMetricDefinition(
        name="chitu_task_eviction_rate_per_second",
        help_en="Task eviction rate per DP rank.",
        help_zh="每个 DP rank 的任务驱逐速率。",
        expression="rate(chitu_total_task_evictions_total[interval])",
        category=MetricCategory.KV_CACHE,
        source_metrics=("chitu_total_task_evictions",),
        scope=MetricScope.PER_DP_RANK,
        stdout=True,
        grafana=True,
    ),
    DerivedMetricDefinition(
        name="chitu_prefix_cache_hit_rate",
        help_en="Ratio of prefix-cache hit prompt tokens to total prompt tokens.",
        help_zh="Prefix cache 命中的 prompt token 占总 prompt token 的比例。",
        expression="chitu_total_hit_tokens_total / chitu_total_prompt_tokens_total",
        category=MetricCategory.KV_CACHE,
        source_metrics=("chitu_total_hit_tokens", "chitu_total_prompt_tokens"),
        scope=MetricScope.PER_DP_RANK,
        stdout=True,
        grafana=True,
    ),
    DerivedMetricDefinition(
        name="chitu_mtp_hit_rate",
        help_en="Ratio of accepted MTP tokens to proposed MTP tokens.",
        help_zh="MTP 接受 token 占提议 token 的比例。",
        expression="rate(chitu_mtp_accepted_tokens_total[interval]) / rate(chitu_mtp_proposed_tokens_total[interval])",
        category=MetricCategory.THROUGHPUT,
        source_metrics=("chitu_mtp_accepted_tokens", "chitu_mtp_proposed_tokens"),
        scope=MetricScope.PER_DP_RANK,
        stdout=True,
        grafana=True,
    ),
    DerivedMetricDefinition(
        name="chitu_pd_stage_latency_quantile",
        help_en="PD stage latency quantiles from stage duration histogram buckets.",
        help_zh="基于阶段延迟直方图 bucket 的 PD 阶段延迟分位数。",
        expression="histogram_quantile(q, sum(rate(chitu_pd_stage_duration_seconds_bucket[interval])) by (le, role, stage))",
        category=MetricCategory.PD_DISAGGREGATION,
        source_metrics=("chitu_pd_stage_duration_seconds",),
        scope=MetricScope.PER_ROLE,
        grafana=True,
    ),
    DerivedMetricDefinition(
        name="chitu_ttft_latency_quantile",
        help_en="Time-to-first-token latency quantiles.",
        help_zh="首 token 延迟分位数。",
        expression="histogram_quantile(q, sum(rate(chitu_time_to_first_token_seconds_bucket[interval])) by (le))",
        category=MetricCategory.REQUESTS,
        source_metrics=("chitu_time_to_first_token_seconds",),
        scope=MetricScope.WHOLE_SERVICE,
        grafana=True,
    ),
    DerivedMetricDefinition(
        name="chitu_e2e_request_latency_quantile",
        help_en="End-to-end request latency quantiles.",
        help_zh="端到端请求延迟分位数。",
        expression="histogram_quantile(q, sum(rate(chitu_e2e_request_duration_seconds_bucket[interval])) by (le))",
        category=MetricCategory.REQUESTS,
        source_metrics=("chitu_e2e_request_duration_seconds",),
        scope=MetricScope.WHOLE_SERVICE,
        grafana=True,
    ),
    DerivedMetricDefinition(
        name="chitu_kv_transfer_speed_bytes_per_second",
        help_en="KV transfer speed in bytes per second, derived from transferred bytes and transfer duration.",
        help_zh="KV 传输速度（字节/秒），由传输字节数和传输耗时计算得到。",
        expression="sum(rate(chitu_kv_transfer_size_bytes_sum[interval])) by (instance_id, rank) / sum(rate(chitu_kv_transfer_duration_seconds_sum[interval])) by (instance_id, rank)",
        category=MetricCategory.PD_DISAGGREGATION,
        source_metrics=(
            "chitu_kv_transfer_size_bytes",
            "chitu_kv_transfer_duration_seconds",
        ),
        scope=MetricScope.PER_RANK,
        grafana=True,
    ),
    DerivedMetricDefinition(
        name="chitu_completed_request_rate_per_second",
        help_en="Completed request rate by instance, rank, and role.",
        help_zh="按实例、rank 和角色统计的完成请求速率。",
        expression="rate(chitu_completed_requests_total[interval])",
        category=MetricCategory.REQUESTS,
        source_metrics=("chitu_completed_requests_total",),
        scope=MetricScope.PER_RANK,
        grafana=True,
    ),
    DerivedMetricDefinition(
        name="chitu_kv_transfer_failure_rate_per_second",
        help_en="KV transfer failure rate by instance, rank, and role.",
        help_zh="按实例、rank 和角色统计的 KV 传输失败速率。",
        expression="rate(chitu_kv_transfer_failures_total[interval])",
        category=MetricCategory.ERRORS,
        source_metrics=("chitu_kv_transfer_failures_total",),
        scope=MetricScope.PER_RANK,
        grafana=True,
    ),
    DerivedMetricDefinition(
        name="chitu_request_timeout_rate_per_second",
        help_en="Request timeout rate by stage.",
        help_zh="按阶段统计的请求超时速率。",
        expression="rate(chitu_request_timeouts_total[interval])",
        category=MetricCategory.ERRORS,
        source_metrics=("chitu_request_timeouts_total",),
        scope=MetricScope.WHOLE_SERVICE,
        grafana=True,
    ),
)

_METRIC_BY_NAME = {metric.name: metric for metric in RAW_METRICS}
_DERIVED_METRIC_BY_NAME = {metric.name: metric for metric in DERIVED_METRICS}
_LABEL_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def get_metric_definition(name: str) -> MetricDefinition:
    return _METRIC_BY_NAME[name]


def raw_metrics_by_name() -> dict[str, MetricDefinition]:
    return dict(_METRIC_BY_NAME)


def derived_metrics_by_name() -> dict[str, DerivedMetricDefinition]:
    return dict(_DERIVED_METRIC_BY_NAME)


def raw_metrics_for_scope(scope: MetricScope) -> tuple[MetricDefinition, ...]:
    return tuple(metric for metric in RAW_METRICS if metric.scope == scope)


def should_collect_metric(metric: MetricDefinition, context: MetricContext) -> bool:
    if metric.scope == MetricScope.PER_RANK:
        return context.rank is not None
    if metric.scope == MetricScope.PER_DP_RANK:
        return context.rank_in_dp == 0 and context.dp_rank is not None
    if metric.scope == MetricScope.PER_INSTANCE:
        return context.rank == 0 and context.inst_id is not None
    if metric.scope == MetricScope.PER_ROLE:
        return context.inst_role is not None
    if metric.scope == MetricScope.WHOLE_SERVICE:
        return context.is_router
    return False


def raw_metrics_for_context(context: MetricContext) -> tuple[MetricDefinition, ...]:
    return tuple(
        metric for metric in RAW_METRICS if should_collect_metric(metric, context)
    )


def stdout_raw_metrics() -> tuple[MetricDefinition, ...]:
    return tuple(metric for metric in RAW_METRICS if metric.stdout)


def stdout_derived_metrics() -> tuple[DerivedMetricDefinition, ...]:
    return tuple(metric for metric in DERIVED_METRICS if metric.stdout)


def grafana_raw_metrics() -> tuple[MetricDefinition, ...]:
    return tuple(metric for metric in RAW_METRICS if metric.grafana)


def grafana_derived_metrics() -> tuple[DerivedMetricDefinition, ...]:
    return tuple(metric for metric in DERIVED_METRICS if metric.grafana)


def prometheus_query_name(metric: MetricDefinition) -> str:
    if metric.type != MetricType.COUNTER:
        return metric.name
    return metric.name if metric.name.endswith("_total") else f"{metric.name}_total"


def histogram_series_names(metric: MetricDefinition) -> tuple[str, ...]:
    if metric.type != MetricType.HISTOGRAM:
        return ()
    return (f"{metric.name}_bucket", f"{metric.name}_count", f"{metric.name}_sum")


def validate_metric_definitions() -> list[str]:
    errors: list[str] = []
    names = [metric.name for metric in RAW_METRICS]
    if len(names) != len(set(names)):
        errors.append("raw metric names must be unique")
    derived_names = [metric.name for metric in DERIVED_METRICS]
    if len(derived_names) != len(set(derived_names)):
        errors.append("derived metric names must be unique")
    for metric in RAW_METRICS:
        if not metric.name.startswith("chitu_"):
            errors.append(f"{metric.name}: raw metric names must start with chitu_")
        if not metric.help_en or not metric.help_zh:
            errors.append(f"{metric.name}: help_en and help_zh are required")
        for label in metric.labels:
            if not _LABEL_RE.match(label):
                errors.append(f"{metric.name}: invalid label name {label!r}")
        if metric.type == MetricType.HISTOGRAM:
            if not metric.buckets:
                errors.append(f"{metric.name}: histogram metrics require buckets")
            elif tuple(sorted(metric.buckets)) != metric.buckets:
                errors.append(f"{metric.name}: histogram buckets must be sorted")
        elif metric.buckets is not None:
            errors.append(f"{metric.name}: only histograms may define buckets")
        if metric.unit == "bytes" and not metric.name.endswith("_bytes"):
            errors.append(f"{metric.name}: byte metrics should use the _bytes suffix")
        if metric.unit == "seconds" and not metric.name.endswith("_seconds"):
            errors.append(
                f"{metric.name}: second metrics should use the _seconds suffix"
            )
        if metric.unit == "ratio" and not metric.name.endswith("_ratio"):
            errors.append(f"{metric.name}: ratio metrics should use the _ratio suffix")
    if "chitu_kv_transfer_speed_gbps" in _METRIC_BY_NAME:
        errors.append("chitu_kv_transfer_speed_gbps must not be a raw metric")
    if "chitu_kv_transfer_speed_bytes_per_second" in _METRIC_BY_NAME:
        errors.append(
            "chitu_kv_transfer_speed_bytes_per_second must be a derived metric"
        )
    if "chitu_kv_transfer_speed_bytes_per_second" not in _DERIVED_METRIC_BY_NAME:
        errors.append(
            "chitu_kv_transfer_speed_bytes_per_second must be documented as a derived metric"
        )
    for derived_metric in DERIVED_METRICS:
        if not derived_metric.name.startswith("chitu_"):
            errors.append(
                f"{derived_metric.name}: derived metric names should start with chitu_"
            )
        if not derived_metric.help_en or not derived_metric.help_zh:
            errors.append(f"{derived_metric.name}: help_en and help_zh are required")
        for source_metric in derived_metric.source_metrics:
            if source_metric not in _METRIC_BY_NAME:
                errors.append(
                    f"{derived_metric.name}: unknown source metric {source_metric}"
                )
    return errors
