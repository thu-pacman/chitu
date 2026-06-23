# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
import threading
from contextlib import contextmanager
from typing import Optional
from prometheus_client import Counter, Gauge, Histogram, start_http_server, REGISTRY
import atexit
import torch

from chitu.backend import Backend
from chitu.distributed.parallel_state import (
    get_dp_group,
    get_tp_group,
    get_pp_group,
    get_dp_size,
)
from chitu.boot.tcp_ip import get_local_ip, get_free_port
from chitu.global_vars import get_global_args
from chitu.metrics.cache_stats import kvcache_stats, get_prealloc_blocks
from chitu.metrics.task_stats import count_tasks_for_dp_rank, count_tasks_non_dp

logger = logging.getLogger(__name__)


# 部分参考自 sglang 的 metrics.py和 vllm
# ---------------------------------------------------------------------------
# PD Disaggregation Metrics (module-level singletons, created on first import)
# These are safe to define at module scope; Prometheus client handles
# duplicate registration gracefully when the same metric name is reused.
# ---------------------------------------------------------------------------

# -- Histograms: per-stage latency ------------------------------------------
_PD_STAGE_BUCKETS = (
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

chitu_pd_stage_duration_seconds = Histogram(
    "chitu_pd_stage_duration_seconds",
    "PD disaggregation per-stage request latency in seconds",
    ["role", "stage"],
    buckets=_PD_STAGE_BUCKETS,
)

chitu_e2e_request_duration_seconds = Histogram(
    "chitu_e2e_request_duration_seconds",
    "End-to-end request latency from router recv to decode complete",
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
)

chitu_time_to_first_token_seconds = Histogram(
    "chitu_time_to_first_token_seconds",
    "Time from request arrival to first token generated",
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)

chitu_kv_transfer_duration_seconds = Histogram(
    "chitu_kv_transfer_duration_seconds",
    "KV cache transfer duration in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

chitu_kv_transfer_size_bytes = Histogram(
    "chitu_kv_transfer_size_bytes",
    "KV cache transfer size in bytes per request",
    buckets=(1e6, 1e7, 5e7, 1e8, 5e8, 1e9),
)

# -- Gauges: queue depths ---------------------------------------------------

chitu_pd_queue_size = Gauge(
    "chitu_pd_queue_size",
    "Current queue size for PD disaggregation stages",
    ["role", "queue_name"],
)

chitu_router_pending_requests = Gauge(
    "chitu_router_pending_requests",
    "Number of pending requests in router",
)

chitu_active_requests = Gauge(
    "chitu_active_requests",
    "Number of active streaming requests",
    ["role"],
)

chitu_kv_transfer_speed_gbps = Gauge(
    "chitu_kv_transfer_speed_gbps",
    "Latest KV transfer speed in GB/s",
)

# -- Counters: errors / completions -----------------------------------------

chitu_kv_transfer_failures_total = Counter(
    "chitu_kv_transfer_failures_total",
    "Total KV transfer failures",
    ["role"],
)

chitu_request_timeouts_total = Counter(
    "chitu_request_timeouts_total",
    "Total request timeouts",
    ["stage"],
)

chitu_completed_requests_total = Counter(
    "chitu_completed_requests_total",
    "Total completed requests",
    ["role"],
)


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
        chitu_pd_stage_duration_seconds.labels(role=role, stage=stage).observe(duration)


def observe_stage_duration(role: str, stage: str, duration_s: float):
    """Record a pre-computed stage duration (seconds) into the Histogram."""
    if duration_s >= 0:
        chitu_pd_stage_duration_seconds.labels(role=role, stage=stage).observe(
            duration_s
        )


def observe_e2e_duration(duration_s: float):
    """Record end-to-end request duration."""
    if duration_s >= 0:
        chitu_e2e_request_duration_seconds.observe(duration_s)


def observe_ttft(duration_s: float):
    """Record time-to-first-token."""
    if duration_s >= 0:
        chitu_time_to_first_token_seconds.observe(duration_s)


def observe_kv_transfer(
    duration_s: float, size_bytes: float = 0, speed_gbps: float = 0
):
    """Record KV transfer duration, size, and speed."""
    if duration_s >= 0:
        chitu_kv_transfer_duration_seconds.observe(duration_s)
    if size_bytes > 0:
        chitu_kv_transfer_size_bytes.observe(size_bytes)
    if speed_gbps > 0:
        chitu_kv_transfer_speed_gbps.set(speed_gbps)


def set_queue_size(role: str, queue_name: str, size: int):
    """Set a queue depth gauge."""
    chitu_pd_queue_size.labels(role=role, queue_name=queue_name).set(size)


def inc_completed_requests(role: str, count: int = 1):
    """Increment completed requests counter."""
    chitu_completed_requests_total.labels(role=role).inc(count)


def inc_kv_transfer_failures(role: str, count: int = 1):
    """Increment KV transfer failure counter."""
    chitu_kv_transfer_failures_total.labels(role=role).inc(count)


def inc_request_timeouts(stage: str, count: int = 1):
    """Increment request timeout counter."""
    chitu_request_timeouts_total.labels(stage=stage).inc(count)


_pynvml = None
_pynvml_failed = False


def _import_pynvml():
    global _pynvml, _pynvml_failed
    if _pynvml_failed:
        return None
    if _pynvml is not None:
        return _pynvml
    try:
        import pynvml  # type: ignore
    except Exception:
        _pynvml_failed = True
        return None
    _pynvml = pynvml
    return _pynvml


def _nvml_handle_for_device(pynvml, device_index: int):
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        device_tokens = [
            token.strip() for token in cuda_visible_devices.split(",") if token.strip()
        ]
        if device_index < len(device_tokens):
            token = device_tokens[device_index]
            if token.startswith(("GPU-", "MIG-")):
                return pynvml.nvmlDeviceGetHandleByUUID(token)
            try:
                return pynvml.nvmlDeviceGetHandleByIndex(int(token))
            except ValueError:
                pass
    return pynvml.nvmlDeviceGetHandleByIndex(device_index)


def _get_nvml_memory_bytes(device_index: int, pid: int):
    pynvml = _import_pynvml()
    if pynvml is None:
        return None
    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        handle = _nvml_handle_for_device(pynvml, device_index)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = int(mem_info.total)
        used = int(mem_info.used)
        try:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except Exception:
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses_v2(handle)
            except Exception:
                logger.warning(
                    "Failed to get compute running processes, use empty list as default"
                )
                processes = []
        for proc in processes:
            if proc.pid != pid:
                continue
            proc_used = getattr(proc, "usedGpuMemory", None)
            if proc_used is not None and proc_used > 0:
                used = int(proc_used)
            break
        return used, total
    except Exception as e:
        logger.warning(f"Failed to get NVML memory bytes: {e}")
        return None
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class PrometheusMetricsCollector:
    """Singleton metrics collector for Prometheus"""

    _instance: Optional["PrometheusMetricsCollector"] = None
    _lock = threading.RLock()
    _shutting_down = False  # 正在关闭为True，未关闭和关闭完成为False
    addrs: Optional[list[str]] = None

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
                dp_id = get_dp_group().rank_in_group
                multi_inst = getattr(get_global_args(), "multi_inst", None)
                instance_id = 0 if multi_inst is None else multi_inst.inst_id
                rank = get_dp_group().global_rank
                cls._instance = cls(rank, dp_id, instance_id)
                cls.addrs = [cls._instance.addr]
        return cls._instance

    def __init__(self, rank: int = 0, dp_id: int = 0, instance_id: int = 0):
        self.rank: int = rank
        self.dp_id: int = dp_id
        self.instance_id: int = instance_id

        tp_group = get_tp_group()
        dp_group = get_dp_group()
        pp_group = get_pp_group()
        self.is_dp_metrics_rank = (
            tp_group.rank_in_group == 0 and pp_group.rank_in_group == 0
        )
        self.is_main_rank = dp_group.is_first_rank
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
        self.addr = None

        try:
            # --- Per-rank metrics (kv cache, GPU memory)
            self.kv_cache_usage = Gauge(
                "chitu_kv_cache_usage_ratio",
                "KV cache usage ratio (used_blocks / total_blocks)",
                ["rank", "dp_id", "instance_id"],
            )
            self.used_blocks = Gauge(
                "chitu_used_blocks",
                "KV cache used blocks",
                ["rank", "dp_id", "instance_id"],
            )
            self.total_blocks = Gauge(
                "chitu_total_blocks",
                "KV cache total blocks",
                ["rank", "dp_id", "instance_id"],
            )
            self.cuda_total_bytes = Gauge(
                "chitu_cuda_total_bytes",
                "CUDA total memory (bytes)",
                ["rank", "dp_id", "instance_id"],
            )
            self.cuda_used_bytes = Gauge(
                "chitu_cuda_used_bytes",
                "CUDA used memory (bytes), including torch allocated memory, torch "
                "reserved but unused memory, and other CUDA memory",
                ["rank", "dp_id", "instance_id"],
            )
            self.torch_allocated_bytes = Gauge(
                "chitu_torch_allocated_bytes",
                "torch allocated GPU memory (bytes)",
                ["rank", "dp_id", "instance_id"],
            )
            self.torch_reserved_bytes = Gauge(
                "chitu_torch_reserved_bytes",
                "torch reserved GPU memory (bytes), including torch allocated memory, "
                "and torch reserved but unused memory",
                ["rank", "dp_id", "instance_id"],
            )

            self.cuda_total_bytes.labels(
                rank=rank, dp_id=dp_id, instance_id=instance_id
            ).set(0)
            self.cuda_used_bytes.labels(
                rank=rank, dp_id=dp_id, instance_id=instance_id
            ).set(0)
            self.torch_allocated_bytes.labels(
                rank=rank, dp_id=dp_id, instance_id=instance_id
            ).set(0)
            self.torch_reserved_bytes.labels(
                rank=rank, dp_id=dp_id, instance_id=instance_id
            ).set(0)

            # --- Per-dp metrics (throughput, task counts)
            if self.is_dp_metrics_rank:
                self.total_generated_tokens = Counter(
                    "chitu_total_generated_tokens",
                    "Total tokens generated by executor",
                    ["rank", "dp_id", "instance_id"],
                )
                self.total_prompt_tokens = Counter(
                    "chitu_total_prompt_tokens",
                    "Total prompt tokens processed by executor",
                    ["rank", "dp_id", "instance_id"],
                )
                self.total_hit_tokens = Counter(
                    "chitu_total_hit_tokens",
                    "total prompt tokens hit by prefix caching",
                    ["rank", "dp_id", "instance_id"],
                )
                self.total_task_evictions = Counter(
                    "chitu_total_task_evictions",
                    "Total number of tasks evicted due to insufficient KV cache",
                    ["rank", "dp_id", "instance_id"],
                )
                self.mtp_proposed_tokens = Counter(
                    "chitu_mtp_proposed_tokens",
                    "Total MTP proposed tokens (mtp_size-1 per task per decode step)",
                    ["rank", "dp_id", "instance_id"],
                )
                self.mtp_accepted_tokens = Counter(
                    "chitu_mtp_accepted_tokens",
                    "Total MTP accepted tokens after verification",
                    ["rank", "dp_id", "instance_id"],
                )
                self.running_requests = Gauge(
                    "chitu_running_requests",
                    "Number of currently running requests",
                    ["rank", "dp_id", "instance_id"],
                )
                self.waiting_requests = Gauge(
                    "chitu_waiting_requests",
                    "Number of currently waiting requests",
                    ["rank", "dp_id", "instance_id"],
                )
                self.total_generated_tokens.labels(
                    rank=rank, dp_id=dp_id, instance_id=instance_id
                ).inc(0)
                self.total_prompt_tokens.labels(
                    rank=rank, dp_id=dp_id, instance_id=instance_id
                ).inc(0)
                self.total_hit_tokens.labels(
                    rank=rank, dp_id=dp_id, instance_id=instance_id
                ).inc(0)
                self.total_task_evictions.labels(
                    rank=rank, dp_id=dp_id, instance_id=instance_id
                ).inc(0)
                self.mtp_proposed_tokens.labels(
                    rank=rank, dp_id=dp_id, instance_id=instance_id
                ).inc(0)
                self.mtp_accepted_tokens.labels(
                    rank=rank, dp_id=dp_id, instance_id=instance_id
                ).inc(0)
                self.running_requests.labels(
                    rank=rank, dp_id=dp_id, instance_id=instance_id
                ).set(0)
                self.waiting_requests.labels(
                    rank=rank, dp_id=dp_id, instance_id=instance_id
                ).set(0)
            if self.is_main_rank:
                self.prealloc_blocks = Gauge(
                    "chitu_prealloc_blocks",
                    "Number of pre-allocated blocks for PD disaggrigation",
                    ["rank", "dp_id", "instance_id"],
                )
            # init kv cache and prealloc blocks via update_kvcache_usage
            self.update_kvcache_usage()

            try:
                ip = get_local_ip()
            except Exception as e:
                logger.warning(f"Failed to get local IP: {e}, use 127.0.0.1 as default")
                ip = "127.0.0.1"
            port = get_free_port()
            self.collector_server, self.collector_thread = start_http_server(
                port
            )  # Prometheus serve pull data from this server
            self.addr = f"{ip}:{port}"
            logger.info(f"Prometheus metrics server started on addr {self.addr}")
            atexit.register(PrometheusMetricsCollector.stop_instance)

        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics collector: {e}")
            raise

    @classmethod
    def inc_generated_tokens(cls, count: int = 1):
        if count < 0:
            return

        collector = cls.get_instance()
        if not collector or not collector.total_generated_tokens:
            return
        try:
            collector.total_generated_tokens.labels(
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

        collector = cls.get_instance()
        if not collector or not collector.mtp_proposed_tokens:
            return
        try:
            collector.mtp_proposed_tokens.labels(
                rank=collector.rank,
                dp_id=collector.dp_id,
                instance_id=collector.instance_id,
            ).inc(proposed)
            collector.mtp_accepted_tokens.labels(
                rank=collector.rank,
                dp_id=collector.dp_id,
                instance_id=collector.instance_id,
            ).inc(accepted)
        except Exception as e:
            logger.error(f"inc_mtp_tokens failed: {e}")

    @classmethod
    def inc_prompt_tokens(cls, count: int = 1):
        if count < 0:
            return

        collector = cls.get_instance()
        if not collector or not collector.total_prompt_tokens:
            return
        try:
            collector.total_prompt_tokens.labels(
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

        collector = cls.get_instance()
        if not collector or not collector.total_hit_tokens:
            return

        try:
            collector.total_hit_tokens.labels(
                rank=collector.rank,
                dp_id=collector.dp_id,
                instance_id=collector.instance_id,
            ).inc(count)
        except Exception as e:
            logger.error(f"inc_hit_tokens failed: {e}")

    @classmethod
    def update_task_counts(cls):
        """Update running/waiting request count metrics."""
        collector = cls.get_instance()
        if not collector or not collector.running_requests:
            return
        try:
            if collector.is_main_rank:
                running, waiting = count_tasks_for_dp_rank(collector.dp_id)
            else:
                running, waiting = count_tasks_non_dp()
            collector.running_requests.labels(
                rank=collector.rank,
                dp_id=collector.dp_id,
                instance_id=collector.instance_id,
            ).set(running)
            collector.waiting_requests.labels(
                rank=collector.rank,
                dp_id=collector.dp_id,
                instance_id=collector.instance_id,
            ).set(waiting)
        except Exception as e:
            logger.error(f"update_task_counts failed: {e}")

    @classmethod
    def update_kvcache_usage(cls):
        """Update KV cache usage metrics."""
        collector = cls.get_instance()
        if not collector or Backend.cache_dict is None:
            return

        try:
            kvcache_stats_dict = kvcache_stats(collector.is_main_rank, collector.dp_id)
            for dp_id in kvcache_stats_dict:
                rank = get_dp_group().rank_list[dp_id]
                total_blocks, used_blocks, kvcache_usage = kvcache_stats_dict[dp_id]
                collector.total_blocks.labels(
                    rank=rank, dp_id=dp_id, instance_id=collector.instance_id
                ).set(total_blocks)
                collector.used_blocks.labels(
                    rank=rank, dp_id=dp_id, instance_id=collector.instance_id
                ).set(used_blocks)
                if kvcache_usage >= 0:
                    collector.kv_cache_usage.labels(
                        rank=rank, dp_id=dp_id, instance_id=collector.instance_id
                    ).set(kvcache_usage)
                else:
                    logger.error(
                        f"Unexpected {type(Backend.cache_dict['main']).__name__}.num_blocks({total_blocks}), update_kvcache_usage failed. "
                    )
            if collector.prealloc_blocks is not None:
                prealloc_blocks_dict = get_prealloc_blocks(collector.dp_size)
                if prealloc_blocks_dict:
                    for dp_id in prealloc_blocks_dict:
                        rank = get_dp_group().rank_list[dp_id]
                        prealloc_blocks = prealloc_blocks_dict[dp_id]
                        collector.prealloc_blocks.labels(
                            rank=rank, dp_id=dp_id, instance_id=collector.instance_id
                        ).set(prealloc_blocks)
        except Exception as e:
            logger.error(f"update_kvcache_usage failed: {e}")

    @classmethod
    def update_GPU_usage(cls):
        if Backend.cache_dict is None:
            return

        cls.update_kvcache_usage()

        collector = cls.get_instance()
        if not collector:
            return

        try:
            device = Backend.cache_dict["main"].device
            if (not isinstance(device, torch.device)) or (
                isinstance(device, torch.device) and device.type != "cuda"
            ):
                return

            device_index = device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            mem_info = _get_nvml_memory_bytes(device_index, os.getpid())
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
        collector = cls.get_instance()
        if not collector or not collector.total_task_evictions:
            return

        try:
            collector.total_task_evictions.labels(
                rank=collector.rank,
                dp_id=collector.dp_id,
                instance_id=collector.instance_id,
            ).inc()
        except Exception as e:
            logger.error(f"inc_task_eviction failed: {e}")
            pass

    @classmethod
    def stop_instance(cls):
        cls._shutting_down = True
        with cls._lock:
            if cls._instance is None:
                return

            instance = cls._instance
            try:
                # Stop HTTP server
                if instance.collector_server:
                    instance.collector_server.shutdown()
                    instance.collector_server.server_close()

                # Wait for thread to finish
                if instance.collector_thread:
                    instance.collector_thread.join(timeout=5.0)

                # Unregister metrics
                for attr_name in dir(instance):
                    attr = getattr(instance, attr_name, None)
                    if isinstance(attr, (Counter, Gauge, Histogram)):
                        try:
                            REGISTRY.unregister(attr)
                        except Exception as e:
                            logger.debug(
                                f"Metric {attr_name} already unregistered or failed: {e}"
                            )

                logger.info(
                    f"[rank {instance.rank}]: Prometheus metrics collector stopped and cleaned up"
                )
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            finally:
                cls._instance = None
                cls._shutting_down = False
