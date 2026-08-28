# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import time
import math
from enum import Enum, auto
from logging import getLogger
from typing import Optional, TYPE_CHECKING, Callable
from typing_extensions import override
from collections import deque

from chitu.task import TaskPool, TaskType, Task, TaskStatus
from chitu.global_vars import get_global_args, SlotHandle, is_pd_prefill_only
from chitu.hooks import TaskEvictHook, NoopTaskEvictHook
from chitu.utils import ceil_div
from chitu.backend import Backend
from chitu.distributed.partition import compute_local_batch_size_dist_in_dp
from chitu.metrics.prometheus_collector import (
    PrometheusMetricsCollector,
    inc_completed_requests,
    inc_request_timeouts,
)

if TYPE_CHECKING:
    from chitu.kv_cache import KVCacheManagerBase

logger = getLogger(__name__)


class KVCacheCapacityStatus(Enum):
    """Result of a KV cache capacity check for a candidate task.

    OK:
        The task can be scheduled — enough free blocks are available.
    CONGESTED:
        The task needs more blocks than currently free, but the total number
        of blocks it would need is within the cache's physical capacity.  The
        scheduler should skip this task for now and retry later.
    EXCEEDS_CAPACITY:
        The task's sequence length requires more blocks than the cache can
        physically hold, even if all blocks were free.  The task must be
        terminated (evicted).
    """

    OK = auto()
    CONGESTED = auto()
    EXCEEDS_CAPACITY = auto()


class SchedulerGroupList:
    """Ring-buffer of schedule groups for pipeline parallelism.

    In PP mode, schedule groups rotate each step.  Each group records the task
    IDs scheduled in its slot so they can be marked waiting and later released
    with ``unwait()`` after the corresponding pipeline delay.
    """

    def __init__(self, num_sgroup: int, type: str = "paged"):
        self.num_sgroup = num_sgroup
        self.type = type
        self._sgroup_list = [[] for i in range(num_sgroup)]
        self.is_skew = self.type == "skew"
        if self.is_skew:
            self._sgroup_tasks_list = [[] for i in range(num_sgroup)]

        self.current_sgroup_id = 0

    def __len__(self) -> int:
        return sum(len(sgroup) for sgroup in self._sgroup_list)

    def release_sgroup(self, sgroup_id: Optional[int] = None):
        """
        Release all the tasks in the given sgroup. For default, the first sgroup will be released.
        """
        if sgroup_id is None:
            sgroup_id = (self.current_sgroup_id + 1) % self.num_sgroup
        released_task_ids = self._sgroup_list[sgroup_id]
        for task_id in released_task_ids:
            task = TaskPool.pool.get(task_id)
            if task is not None:
                task.unwait()
        self._sgroup_list[sgroup_id] = []
        return released_task_ids

    def get_current_sgroup(self):
        return self.current_sgroup_id

    def switch_to_next_sgroup(self):
        self.current_sgroup_id = (self.current_sgroup_id + 1) % self.num_sgroup
        self.release_sgroup(self.current_sgroup_id)
        return self.current_sgroup_id

    def set_task_ids(self, task_ids: list[str], keep_unwait: Optional[set[str]] = None):
        """Write task_ids into the current sgroup slot and `wait()` each.

        Args:
            keep_unwait: intermediate prefill task_ids to keep schedulable on
                the next consecutive step. Such tasks are `unwait()`ed AND left out of the
                sgroup list, so a later `release_sgroup` of this slot does not prematurely
                re-unwait them — only the completing chunk (not in keep_unwait) stays listed
                and `Waiting`, and is released `pp_size` steps later, which is the decode
                gate (the first decode token returns `pp_size-1` steps after the completing
                chunk). Default None preserves the original round-robin behavior.
        """
        keep_unwait = keep_unwait or set()
        self._sgroup_list[self.current_sgroup_id] = [
            tid for tid in task_ids if tid not in keep_unwait
        ]
        for task_id in task_ids:
            TaskPool.pool[task_id].wait()
        for task_id in keep_unwait:
            TaskPool.pool[task_id].unwait()

    def get_sgroup_all_tasks(self, sgroup_id: int):
        # assert self.is_skew
        return self._sgroup_tasks_list[sgroup_id]

    def get_current_sgroup_all_tasks(self):
        return self.get_sgroup_all_tasks(self.current_sgroup_id)


class Scheduler:
    @staticmethod
    def build(args, infer_args, *, dp_rank: int):
        max_reqs_per_dp = compute_local_batch_size_dist_in_dp(
            infer_args.max_batch_size, infer_args.dp_size
        )[dp_rank]
        if infer_args.prefill_chunk_size is not None:
            # infer.prefill_chunk_size is the GLOBAL (total) prefill chunk size across
            # all DP and CP ranks. The scheduler for a given DP rank packs its share
            # of the global budget: config // dp_size tokens per step. These tokens
            # are then split across the pcp_size CP ranks, so each CP rank only
            # processes 1/pcp_size of them (i.e. config // pcp_size per rank).
            global_chunk_size = infer_args.prefill_chunk_size
            prefill_chunk_size_per_dp: Optional[int] = (
                global_chunk_size // infer_args.dp_size
                + int(dp_rank < global_chunk_size % infer_args.dp_size)
            )
        else:
            prefill_chunk_size_per_dp: Optional[int] = None

        if infer_args.cache_type == "skew":
            return SkewScheduler(
                max_reqs_per_dp,
                None,
                dp_rank=dp_rank,
                scheduler_type=args.type.lower(),
                prefill_chunk_size=prefill_chunk_size_per_dp,
            )

        if infer_args.pp_size > 1:
            if args.pp_config.pp_micro_batch_size_prefill == "max":
                prefill_num_tasks = ceil_div(max_reqs_per_dp, infer_args.pp_size)
            else:
                assert args.pp_config.pp_micro_batch_size_prefill.isdigit()
                prefill_num_tasks = int(args.pp_config.pp_micro_batch_size_prefill)
            if args.pp_config.pp_micro_batch_size_decode == "max":
                decode_num_tasks = ceil_div(max_reqs_per_dp, infer_args.pp_size)
            else:
                assert args.pp_config.pp_micro_batch_size_decode.isdigit()
                decode_num_tasks = int(args.pp_config.pp_micro_batch_size_decode)
        else:
            prefill_num_tasks = max_reqs_per_dp
            decode_num_tasks = max_reqs_per_dp

        cache_manager_dict = Backend.cache_managers[dp_rank]

        return Scheduler(
            max_reqs_per_dp,
            prefill_num_tasks,
            decode_num_tasks,
            args.type.lower(),
            cache_manager_dict,
            num_scheduler_groups=infer_args.pp_size,
            dp_rank=dp_rank,
            prefill_chunk_size=prefill_chunk_size_per_dp,
        )

    def __init__(
        self,
        max_running_tasks: int,
        prefill_num_tasks: int,
        decode_num_tasks: int,
        scheduler_type: str,
        cache_manager_dict: Optional[dict[str, "KVCacheManagerBase"]],
        *,
        num_scheduler_groups: int,
        dp_rank: int = 0,
        prefill_chunk_size: Optional[int] = None,
    ):
        """
        Initialize the scheduler.

        Supported scheduling algorithms:
            - "fcfs": First come, first service.
            - "fifo": Alias for "fcfs".
            - "request_preset": Prioritize tasks based on its preset priority.
            - "prefill_first": Prioritize prefill tasks over decode tasks.
            - "stride": Each task has a priority value P, and a score S (starts from 0), at scheduling point,
              update the scores: S += P * elapsed_time. Select the tasks with top scores and reset their
              scores back to 0.
            - "deadline": Each task has a deadline time `DDL = request_arrival_time + prefix_tokens_len * alpha +
              max_output_tokens * beta`. Select the tasks with nearest DDL. Alpha and beta are arbitary value,
              defaults to 1ms.
            - "prefix_align": Batch tasks with similar input lengths togather.

        Args:
            prefill_num_tasks (int): Max batch size for prefill stage
            decode_num_tasks (int): Max batch size for decode stage
            scheduler_type (str): The type of scheduling algorithm to use. Can be a single string, e.g,
                "prefill_first", or a comma-separated string of multiple types for multi-key priority, e.g.,
                "request_preset,prefill_first".
            dp_rank: In case of DP, one Scheduler is dedicated for each rank, and `dp_rank` is the rank id.
                Only tasks assigned to this DP rank or not assigned with a DP rank will be scheduled by this
                Scheduler instance.
        """

        super().__init__()
        assert prefill_num_tasks > 0, "prefill_num_tasks must be greater than 0"
        assert decode_num_tasks > 0, "decode_num_tasks must be greater than 0"
        self.max_running_tasks = max_running_tasks
        self.prefill_num_tasks = prefill_num_tasks
        self.decode_num_tasks = decode_num_tasks
        self.prefill_chunk_size = prefill_chunk_size
        self.cache_manager_dict = cache_manager_dict
        self.num_scheduler_groups = num_scheduler_groups
        self.sgroup_list = SchedulerGroupList(num_sgroup=self.num_scheduler_groups)
        self.dp_rank = dp_rank

        self.scorers = self.get_scheduler_method_scorers(
            scheduler_type, lambda: self.scheduling_ts
        )

        self.reset_kvcache_block_threshold()
        self.is_warmup_stage = False
        self.has_schedule_overlap = get_global_args().infer.schedule_overlap
        self._task_evict_hook: TaskEvictHook = NoopTaskEvictHook()
        self._pd_ready_exec_delays_ms: list[float] = []
        self._pd_ready_exec_last_log_ts = 0.0
        self._pd_ready_exec_log_interval_s = 5.0

    def reset_kvcache_block_threshold(self):
        if type(self) == SkewScheduler:
            return
        self.kvcache_block_threshold = self.cache_manager_dict["main"].num_blocks

    def start_warmup(self):
        self.is_warmup_stage = True

    def end_warmup(self):
        self.is_warmup_stage = False

    def set_task_evict_hook(self, hook: TaskEvictHook):
        self._task_evict_hook = hook

    @staticmethod
    def get_scheduler_method_scorers(
        scheduler_type: str, get_ts: Callable[[], int]
    ) -> list[Callable[[Task], int | float]]:
        # determine scoring method
        scorers = []
        scheduler_types = [st for st in scheduler_type.split(",") if st]
        if not scheduler_types:
            raise ValueError("scheduler.type must contain at least one scheduler type")
        for st in scheduler_types:
            if st == "request_preset":
                scorers.append(lambda task: task.priority)
            elif st == "prefill_first":
                scorers.append(
                    lambda task: 1 if task.task_type == TaskType.Prefill else 0
                )
            elif st == "fcfs" or st == "fifo":
                scorers.append(lambda task: -task.arrv_ts)
            elif st == "stride":
                scorers.append(
                    lambda task, get_ts=get_ts: task.priority
                    * (get_ts() - task.arrv_ts)
                )
            elif st == "deadline":
                scorers.append(lambda task: -task.sched_ddl)
            elif st == "prefix_align":
                scorers.append(lambda task: -task.prefix_tokens_len)
            else:
                raise NotImplementedError(f"Scheduler type {st} not implemented")

        return scorers

    def scorer(self, task: Task):
        if self.is_warmup_stage:
            fn = lambda task: (
                1 if task.task_type == TaskType.Prefill else 0
            )  # prefill first
            return (fn(task),)
        return tuple(fn(task) for fn in self.scorers)

    def _num_prefill_cached_tokens(self, task: Task) -> int:
        """获取任务在所有manager中已缓存最小长度
        Args:
            task: 要获取已缓存最小长度的任务
        Return:
            不开启prefix caching时，返回当前任务已计算长度
            开启prefix caching时，返回当前任务在各manager中的最小击中长度
        """
        completed_tokens = (
            0
            if task.status == TaskStatus.PDDecodeIncoming
            else task.kv_cache_len_used_in_completed_steps
        )
        if not self.cache_manager_dict["main"].enable_prefix_caching:
            return completed_tokens

        num_cached_tokens = task.prefix_tokens_len
        cache_manager_block_sizes: list[int] = []
        for manager in list(self.cache_manager_dict.values()):
            # Skip Cache that does not support prefix caching e.g. singleton
            if not manager.enable_prefix_caching:
                continue
            cache_manager_block_sizes.append(manager.block_size)
            num_cached_tokens = min(
                num_cached_tokens, manager.num_cached_blocks(task) * manager.block_size
            )
            if num_cached_tokens <= completed_tokens:
                return completed_tokens

        if cache_manager_block_sizes:
            block_lcm = math.lcm(*cache_manager_block_sizes)
            num_cached_tokens = (num_cached_tokens // block_lcm) * block_lcm

        return max(completed_tokens, num_cached_tokens)

    def _inflight_prefill_reserved_blocks(
        self, cache_manager, exclude_task_id: str
    ) -> int:
        """Count blocks that in-flight prefill tasks still need to reach their full prompt length.

        When admitting a new prefill task, the scheduler must reserve enough
        blocks for every already-running prefill to finish.  Without this
        reservation, multiple partial prefill tasks could each hold blocks while
        waiting for more, creating a deadlock where none can reach decode.

        Args:
            cache_manager: The cache manager to count against.
            exclude_task_id: Exclude this task from the count (typically the
                candidate task being evaluated for admission).
        """
        reserved = 0
        for tid, cache_ids in cache_manager.task_to_cache_ids.items():
            if not cache_ids or tid == exclude_task_id:
                continue
            task = TaskPool.pool.get(tid)
            if task is None or task.task_type != TaskType.Prefill:
                continue
            need = cache_manager.num_blocks_for_seq_len(task.prefix_tokens_len) - len(
                cache_ids
            )
            reserved += max(0, need)
        return reserved

    def _check_prefill_capacity(
        self,
        task,
        cached_len: int,
        *,
        need_reserve_capacity: bool = False,
        pd_prealloc_tokens: int = -1,
    ) -> KVCacheCapacityStatus:
        """检查是否所有cache_manager容量都足以容纳当前Prefill任务(task已缓存token长度为cached_len)
        Args:
            task: 当前正在检查kv cache容量的Prefill任务
            cached_len: 当前任务已缓存的长度
            need_reserve_capacity: 是否需要为任务预留容量。新任务准入时需为所有在途
                prefill任务预留其完成所需容量，以避免互相占用导致的调度死锁。
            pd_prealloc_tokens: pd decode阶段的预留prefill容量token数，只在pd decode调用时生效。
        """
        is_pd_decode_check = pd_prealloc_tokens != -1
        target_len = (
            pd_prealloc_tokens
            if is_pd_decode_check
            else cached_len + task.next_req_tokens_len
        )

        for name, cache_manager in self.cache_manager_dict.items():
            target_blocks = cache_manager.num_blocks_for_seq_len(target_len)
            if target_blocks > cache_manager.num_blocks:
                return KVCacheCapacityStatus.EXCEEDS_CAPACITY

            cur_blocks = cache_manager.num_blocks_for_seq_len(cached_len)
            block_threshold = (
                self.kvcache_block_threshold
                if (name == "main" and not is_pd_decode_check)
                else cache_manager.num_blocks
            )
            available_blocks = (
                block_threshold
                - cache_manager.num_active_blocks
                - cache_manager.num_cached_idle_blocks(
                    task, max_cached_token_len=cached_len
                )
            )
            # 准入「新」prefill任务时，必须为所有在途prefill任务的完成预留容量，
            # 否则会出现两个部分完成的prefill互相占块、谁都无法完成的死锁。
            if need_reserve_capacity:
                available_blocks -= self._inflight_prefill_reserved_blocks(
                    cache_manager, exclude_task_id=task.task_id
                )
            if target_blocks - cur_blocks > available_blocks:
                return KVCacheCapacityStatus.CONGESTED
        return KVCacheCapacityStatus.OK

    def _check_decode_capacity(self, task) -> KVCacheCapacityStatus:
        """检查是否所有cache_manager容量都可容纳decode任务
        Args:
            task: 当前正在检查kv cache容量的Decode任务
        """
        for _, cache_manager in self.cache_manager_dict.items():
            target_blocks = cache_manager.num_blocks_for_seq_len(
                task.kv_cache_len_used_in_completed_steps_and_next_step
            )
            if target_blocks > cache_manager.num_blocks:
                return KVCacheCapacityStatus.EXCEEDS_CAPACITY

            available_blocks = (
                cache_manager.num_blocks - cache_manager.num_active_blocks
            )
            cur_blocks = len(cache_manager.task_to_cache_ids[task.task_id])
            if target_blocks - cur_blocks > available_blocks:
                return KVCacheCapacityStatus.CONGESTED
        return KVCacheCapacityStatus.OK

    def _drop_ttft_expired_candidates(self, task_ids: list[str]) -> list[str]:
        now = time.time()
        kept: list[str] = []
        for task_id in task_ids:
            task = TaskPool.pool.get(task_id)
            req = task.req if task is not None else None
            should_reject = (
                req is not None
                and req.ttft_expired(now)
                and task.task_type == TaskType.Prefill
                and req.num_output_tokens == 0
                and task.consumed_req_tokens == 0
                and not task.new_cache_ids
            )
            if not should_reject:
                if task is not None:
                    kept.append(task_id)
                continue

            task.set_stopped()
            req.finish_reason = "error"
            sender = getattr(req.async_stream, "token_sender", None)
            if sender is not None:
                sender.send_error(req.request_id, "TTFT timeout")
                req.async_stream.stop_signal = True
                req.completion_time = time.monotonic()
            else:
                req.stop_stream(error="TTFT timeout")
            TaskPool.remove(task_id)
            inc_request_timeouts("ttft_scheduler")
            logger.warning(
                "[TTFT_TIMEOUT][scheduler] req_id=%s overdue_s=%.3f",
                task_id,
                max(0.0, now - req.ttft_deadline_ts),
            )
        return kept

    def _terminate_task_exceeds_capacity(self, task_id: str) -> None:
        """Stop a task whose sequence exceeds KV cache physical capacity."""
        task = TaskPool.pool.get(task_id)
        if task is None:
            return

        task.set_stopped()
        if task.req is not None:
            task.req.finish_reason = "length"
            if not task.req.finished:
                task.req.stop_stream()

        has_kv_cache = any(
            task_id in cache_manager.task_to_cache_ids
            for cache_manager in self.cache_manager_dict.values()
        )
        if has_kv_cache:
            for cache_manager in self.cache_manager_dict.values():
                cache_manager.finalize_metadata_all_decode(task)
            Backend.executor.special_step([task_id], type="EndTask")

        TaskPool.remove(task_id)
        inc_completed_requests("worker", 1)
        logger.warning(
            f"Task {task_id} exceeds KV cache capacity and was terminated",
            extra={
                "task_id": task_id,
                "event": "scheduler_task_exceeds_capacity",
                "finish_reason": "length: exceeds_capacity",
            },
        )

    def _prepare_prefill_metadata(
        self, task, cached_len: int, *, eager_prefix_cache_insert: bool = True
    ) -> None:
        """task进行prefill前的元数据准备: 维护本次新增击中长度、本次结束后已消费的token，
        调用prepare_metadata_before_prefill维护所有cache_manager中的元信息，
        维护本次cache_manager为任务新新分配的kv block ids
        """
        assert (
            task.consumed_req_tokens <= cached_len <= task.prefix_tokens_len
        ), f"{task.consumed_req_tokens} vs {cached_len} vs {task.prefix_tokens_len}"
        if cached_len == task.prefix_tokens_len:
            cached_len = task.prefix_tokens_len - 1

        task.set_inc_hit_tokens(cached_len - task.consumed_req_tokens)
        task.consumed_req_tokens = cached_len

        for name, cache_manager in self.cache_manager_dict.items():
            task.new_cache_ids[name] = cache_manager.prepare_metadata_before_prefill(
                task, eager_prefix_cache_insert=eager_prefix_cache_insert
            )

    def _prepare_decode_metadata(self, task) -> None:
        """task进行Decode前的元数据准备: 调用prepare_metadata_before_decode维护所有cache_manager中的元信息，
        维护本次cache_manager为任务新新分配的kv block ids
        """
        for name, cache_manager in self.cache_manager_dict.items():
            task.new_cache_ids[name] = cache_manager.prepare_metadata_before_decode(
                task
            )

    def prepare_for_schedule(self) -> None:
        """为本次调度准备sgroup"""
        self.sgroup_list.switch_to_next_sgroup()

    def schedule(
        self,
        strict_allowed_task_type: set[TaskType] = {TaskType.Prefill, TaskType.Decode},
        ready_task_ids: Optional[list[str]] = None,
    ) -> list[str]:
        sgroup_id = self.sgroup_list.get_current_sgroup()
        if TaskPool.is_empty():
            logger.debug("TaskPool is empty, returning empty task list.")
            return []

        self.scheduling_ts = time.perf_counter_ns()

        # collect ready task ids
        # task_to_cache_ids is a defaultdict; only tasks with allocated cache
        # blocks should count against the per-DP running-request limit.
        n_running = sum(
            1
            for cache_ids in self.cache_manager_dict["main"].task_to_cache_ids.values()
            if cache_ids
        )

        task_ids: list[str] = []
        source_task_ids = (
            ready_task_ids if ready_task_ids is not None else TaskPool.id_list
        )
        for task_id in source_task_ids:
            task = TaskPool.pool.get(task_id)
            if task is None or task.task_type not in strict_allowed_task_type:
                continue
            if (
                task.dp_rank is None
                and task.preferred_dp_rank is not None
                and task.preferred_dp_rank != self.dp_rank
            ):
                continue
            if (
                task.dp_rank is None or self.dp_rank == task.dp_rank
            ) and task.can_schedule():
                if n_running >= self.max_running_tasks and task.dp_rank is None:
                    continue
                if task.dp_rank is None:
                    n_running += 1
                task_ids.append(task_id)

        if len(task_ids) == 0:
            # No avaliable tasks, returning empty task list.
            # This is in a busy loop waiting for tasks, so don't print logs here.
            return []

        task_ids = self._drop_ttft_expired_candidates(task_ids)
        if len(task_ids) == 0:
            return []

        task_ids.sort(
            key=lambda x: self.scorer(TaskPool.pool[x]),
            reverse=True,  # Largest first
        )  # list.sort is a stable sort

        filter_task_type = TaskPool.pool[
            task_ids[0]
        ].task_type  # make the highest priority task's type as the filter_task_type

        # Unexpected tasks
        if filter_task_type not in {TaskType.Prefill, TaskType.Decode}:
            raise NotImplementedError(f"Unexpected task type: {filter_task_type}")

        # scheduling prefill tasks
        if filter_task_type == TaskType.Prefill:
            prefill_task_ids = self._schedule_prefill_tasks(task_ids)
            if prefill_task_ids:
                task_ids = prefill_task_ids
            else:
                # No available prefill tasks, we can schedule decode at this condition
                filter_task_type = TaskType.Decode

        # scheduling decode tasks
        if filter_task_type == TaskType.Decode:
            task_ids = self._schedule_decode_tasks(task_ids)[: self.decode_num_tasks]

        # Allocate sgroup for for task_ids
        keep_unwait: Optional[set[str]] = None
        if self.num_scheduler_groups > 1:
            keep_unwait = {
                task_id
                for task_id in task_ids
                if TaskPool.pool[task_id].task_type == TaskType.Prefill
                and not TaskPool.pool[task_id].has_output()
            }
        self.sgroup_list.set_task_ids(task_ids, keep_unwait=keep_unwait)

        # postprocess
        for task_id in task_ids:
            TaskPool.pool[task_id].sched_ts = self.scheduling_ts
            TaskPool.pool[task_id].sched_group_id = sgroup_id
            TaskPool.pool[task_id].dp_rank = self.dp_rank

        self._record_pd_ready_exec_latency(task_ids)
        self._log_pd_ready_exec_latency()

        logger.debug(f"Selected task_ids:")
        for task_id in task_ids:
            task = TaskPool.pool[task_id]
            if task.task_type == TaskType.Prefill:
                if task.prefill_chunk_size is None:
                    logger.debug(
                        f"- {task_id}: Prefill token {task.consumed_req_tokens} to end"
                    )
                else:
                    logger.debug(
                        f"- {task_id}: Prefill token {task.consumed_req_tokens} to {task.consumed_req_tokens + task.prefill_chunk_size}"
                    )
            else:
                logger.debug(f"- {task_id}: Decode")

        return task_ids

    def _record_pd_ready_exec_latency(self, task_ids: list[str]) -> None:
        now = time.perf_counter()
        for tid in task_ids:
            task = TaskPool.pool.get(tid)
            if task is None or task.task_type != TaskType.Decode:
                continue
            ready_ts = getattr(task, "pd_ready_ts", None)
            if ready_ts is None:
                continue
            if hasattr(task, "pd_exec_ts"):
                continue
            task.pd_exec_ts = now
            delay_ms = max(0.0, (now - float(ready_ts)) * 1000.0)
            self._pd_ready_exec_delays_ms.append(delay_ms)

    def _log_pd_ready_exec_latency(self) -> None:
        now = time.perf_counter()
        if (now - self._pd_ready_exec_last_log_ts) < self._pd_ready_exec_log_interval_s:
            return
        self._pd_ready_exec_last_log_ts = now
        if not self._pd_ready_exec_delays_ms:
            return
        delays = sorted(self._pd_ready_exec_delays_ms)
        self._pd_ready_exec_delays_ms = []
        n = len(delays)
        avg = sum(delays) / n

        def _pct(p: float) -> float:
            idx = int(math.ceil(p / 100.0 * n) - 1)
            idx = max(0, min(n - 1, idx))
            return delays[idx]

        logger.debug(
            "[PD_STATS][decode.ready_to_exec] "
            f"count={n} avg_ms={avg:.2f} "
            f"p50_ms={_pct(50):.2f} p90_ms={_pct(90):.2f} "
            f"p95_ms={_pct(95):.2f} p99_ms={_pct(99):.2f}"
        )

    def _prepare_pd_decode_kvcache(
        self, task_id: str, pd_prealloc_tokens: int
    ) -> tuple[KVCacheCapacityStatus, int, dict[str, int]]:
        task = TaskPool.pool[task_id]
        num_cached_tokens = self._num_prefill_cached_tokens(task)
        cache_manager_hit_block_counts = {
            name: (
                manager.num_blocks_for_seq_len(num_cached_tokens)
                if getattr(manager, "enable_prefix_caching", False)
                else 0
            )
            for name, manager in self.cache_manager_dict.items()
        }
        num_uncomputed_tokens = task.prefix_tokens_len - num_cached_tokens
        capacity_status = self._check_prefill_capacity(
            task,
            num_cached_tokens,
            need_reserve_capacity=False,
            pd_prealloc_tokens=pd_prealloc_tokens,
        )
        if capacity_status is not KVCacheCapacityStatus.OK:
            return capacity_status, num_cached_tokens, cache_manager_hit_block_counts
        task.set_prefill_chunk_size_for_one_step(
            1 if num_uncomputed_tokens == 0 else num_uncomputed_tokens
        )
        self._prepare_prefill_metadata(
            task, num_cached_tokens, eager_prefix_cache_insert=False
        )
        task.consume_req_tokens()
        return capacity_status, num_cached_tokens, cache_manager_hit_block_counts

    def _schedule_prefill_tasks(self, task_ids: list[str]) -> list[str]:
        """Prefill tasks scheduling with congestion control
        Args:
            task_ids: list of unwait task ids
        Return:
            sched_out_task_ids: list of unwait prefill task ids
        """
        sched_out_task_ids = []
        prefill_tokens = 0

        idx = 0
        while idx < len(task_ids):
            task_id = task_ids[idx]
            idx += 1
            if task_id not in TaskPool.pool:
                continue
            if len(sched_out_task_ids) >= self.prefill_num_tasks:
                break

            task = TaskPool.pool[task_id]
            if task.task_type != TaskType.Prefill:
                # filter out non-prefill tasks
                continue

            # prefill chunk size check
            if (self.prefill_chunk_size is not None) and (
                self.prefill_chunk_size - prefill_tokens <= 0
            ):
                break
            task_prefill_chunk_size = (
                self.prefill_chunk_size - prefill_tokens
                if self.prefill_chunk_size is not None
                else task.prefix_tokens_len
            )

            # check task's remain tokens

            # task.prefix_tokens_len: in prefill stage, it's prompt length
            # num_cached_tokens: number of tokens that are hit by cached_idle_blocks or active_blocks
            num_cached_tokens = self._num_prefill_cached_tokens(task)
            num_uncomputed_tokens = task.prefix_tokens_len - num_cached_tokens
            if num_uncomputed_tokens == 0:
                pd_prefill_only = is_pd_prefill_only()
                if (
                    self.is_warmup_stage
                    or get_global_args().infer.mtp_size > 1
                    or pd_prefill_only
                ):
                    # Fall back to legacy 1-token prefill:
                    # - warmup: decode graph not captured yet, full prefill needed to
                    #   estimate memory;
                    # - mtp>1: bootstrap decode only supports mtp==1; the MTP draft
                    #   state machine (mtp cache, accept_index, is_classic_decoding)
                    #   depends on prefill_step initialization, so skipping prefill
                    #   would break it
                    # - PD prefill-only: decode waits for PrefillDone/first_token from
                    #   the prefill hook, so a full-hit request must still execute one
                    #   prefill step.
                    prefill_tokens += 1
                    task.set_prefill_chunk_size_for_one_step(1)
                    sched_out_task_ids.append(task_id)
                    self._prepare_prefill_metadata(task, num_cached_tokens)
                    continue
                # Fully cached (mtp==1): skip prefill here and let _schedule_decode_tasks
                # convert this task to a bootstrap Decode (graphed decode for the
                # first token, see _schedule_decode_tasks).
                continue

            task_prefill_chunk_size = min(
                task_prefill_chunk_size, num_uncomputed_tokens
            )
            task_origin_prefill_chunk_size = task.prefill_chunk_size
            task.set_prefill_chunk_size_for_one_step(task_prefill_chunk_size)

            need_reserve_capacity = task.consumed_req_tokens == 0
            capacity_status = self._check_prefill_capacity(
                task, num_cached_tokens, need_reserve_capacity=need_reserve_capacity
            )
            if capacity_status is KVCacheCapacityStatus.EXCEEDS_CAPACITY:
                task.prefill_chunk_size = task_origin_prefill_chunk_size
                self._terminate_task_exceeds_capacity(task_id)
                continue
            if capacity_status is KVCacheCapacityStatus.CONGESTED:
                task.prefill_chunk_size = task_origin_prefill_chunk_size
                if need_reserve_capacity:
                    # 全新任务被拥塞（含为在途prefill预留容量导致）：跳过它继续检查
                    # 排序靠后的、已在途的部分完成任务，让后者有机会推进而不被挡住。
                    continue
                # 已在途的部分任务被拥塞：此时已无容量再准入任何新prefill，僵局只能靠
                # 逐出更低优先级的在途prefill来打破。
                # 逐出后原地重试当前任务，避免空调度、避免再等一轮。
                scheduled_set = set(sched_out_task_ids)
                if self._evict_lowest_priority_inflight_prefill(
                    keep_task_id=task_id, scheduled_set=scheduled_set
                ):
                    idx -= 1  # 重试当前任务（每次逐出一个更低优先级任务，逐步腾出容量）
                    continue
                # 没有可逐出的更低优先级任务：当前KV容量确实不足，停止本轮prefill调度。
                break

            prefill_tokens += task_prefill_chunk_size
            sched_out_task_ids.append(task_id)
            self._prepare_prefill_metadata(task, num_cached_tokens)

        return sched_out_task_ids

    def _evict_lowest_priority_inflight_prefill(
        self, keep_task_id: str, scheduled_set: set[str]
    ) -> bool:
        """逐出优先级最低的、持有块的在途prefill任务以释放容量，用于
        打破多个正在prefill的任务互相占块，导致谁都无法跑完的僵局。

        Args:
            keep_task_id: 当前正试图推进的任务，不可被逐出（它优先级更高）。
            scheduled_set: 本轮已调度出去、即将运行的任务，不可被逐出。
        Return:
            是否成功逐出了一个任务。
        """
        main = self.cache_manager_dict["main"]
        candidates = [
            TaskPool.pool[tid]
            for tid, cache_ids in main.task_to_cache_ids.items()
            if cache_ids
            and tid != keep_task_id
            and tid not in scheduled_set
            and tid in TaskPool.pool
            and TaskPool.pool[tid].task_type == TaskType.Prefill
        ]
        if not candidates:
            return False
        keep_score = self.scorer(TaskPool.pool[keep_task_id])
        # scorer越大越优先；只逐出优先级严格低于当前任务的，避免逐出更该跑的任务。
        victim = min(candidates, key=lambda t: self.scorer(t))
        if self.scorer(victim) >= keep_score:
            return False
        logger.warning(
            f"Prefill congestion: evicting lower-priority in-flight prefill "
            f"{victim.task_id} to let {keep_task_id} make progress and avoid deadlock.",
            extra={
                "event": "scheduler_prefill_congestion_evict",
                "evicted_task_id": victim.task_id,
                "keep_task_id": keep_task_id,
            },
        )
        # 主动让位式逐出：不施加拥塞控制(阈值折半)，否则刚释放的容量会被重新挤占
        self.evict_task(victim.task_id, congestion_control=False)
        return True

    def _schedule_decode_tasks(self, task_ids: list[str]) -> list[str]:
        """Decode tasks scheduling, evicting the last prioriety decode task when these is no more block
        Args:
            task_ids: list of unwait task ids
        Return:
            decode_task_ids: list of unwait decode task ids
        """

        decode_task_ids = deque()
        cached_prefill_task_ids = (
            self._task_evict_hook.get_prefill_task_ids()
        )  # 已开始prefill但未完成的任务

        for tid in task_ids:
            if tid not in TaskPool.pool:
                continue
            task = TaskPool.pool[tid]
            if task.task_type == TaskType.Decode:
                decode_task_ids.append((tid, None))
            elif task.task_type == TaskType.Prefill:
                num_cached_tokens = self._num_prefill_cached_tokens(task)
                if (
                    get_global_args().infer.mtp_size == 1
                    and task.prefix_tokens_len >= 2
                    and self.cache_manager_dict["main"].enable_prefix_caching
                    and task.prefix_tokens_len - num_cached_tokens == 0
                ):
                    decode_task_ids.append((tid, num_cached_tokens))
                elif task.task_id in self.cache_manager_dict["main"].task_to_cache_ids:
                    cached_prefill_task_ids.append(tid)

        if not decode_task_ids:
            return []

        sched_out_task_ids = []
        evict_tasks = []

        while decode_task_ids and len(sched_out_task_ids) < self.decode_num_tasks:
            candidate_task_id, num_cached_tokens = decode_task_ids.popleft()
            candidate_task = TaskPool.pool[candidate_task_id]
            is_bootstrap_candidate = num_cached_tokens is not None
            if is_bootstrap_candidate:
                # Confirmed scheduling this candidate this step: convert to a
                # Decode task and reactivate the KV prefix blocks. Via
                # _prepare_prefill_metadata -> prepare_metadata_before_prefill, the
                # prefix blocks are reactivated; without this they could be evicted
                # or block_table left empty, and decode would read no KV.
                self._prepare_prefill_metadata(candidate_task, num_cached_tokens)
                candidate_task.task_type = TaskType.Decode
                candidate_task.prefill_chunk_size = None
                candidate_task.consumed_req_tokens = candidate_task.prefix_tokens_len
                candidate_task.next_token = candidate_task.prefix_tokens[-1]
                candidate_task.has_unsync_new_token = False

            capacity_status = self._check_decode_capacity(candidate_task)
            if capacity_status is KVCacheCapacityStatus.OK:
                sched_out_task_ids.append(candidate_task_id)
                if not is_bootstrap_candidate:
                    # A bootstrap candidate already had _prepare_prefill_metadata
                    # reactivate the prefix blocks and write the reactivated
                    # cache_idx into task.new_cache_ids (serialized to each stage
                    # to fill block_table). It must NOT call _prepare_decode_metadata:
                    # that would clear new_cache_ids.
                    self._prepare_decode_metadata(candidate_task)
                continue
            if capacity_status is KVCacheCapacityStatus.EXCEEDS_CAPACITY:
                self._terminate_task_exceeds_capacity(candidate_task_id)
                continue

            if not decode_task_ids and not cached_prefill_task_ids:
                # 容量不足，无任务可逐出
                break

            # 容量不足，可逐出低优先级任务（优先逐出已开始prefill但未完成的任务）
            decode_task_ids.appendleft((candidate_task_id, None))
            if cached_prefill_task_ids:
                evict_task_id = cached_prefill_task_ids.pop()
            else:
                evict_tid, _ = decode_task_ids.pop()
                evict_task_id = evict_tid
            evict_tasks.append(evict_task_id)
            self.evict_task(evict_task_id)

        if len(evict_tasks) > 0:
            logger.warning(
                f"KV cache capacity reached limit, forcing eviction of {len(evict_tasks)} decode tasks, this may impact throughput and latency. To prevent performance degradation, consider decreasing max_batch_size or increasing or num_blocks."
            )

        return sched_out_task_ids

    def evict_task(self, task_id: str, congestion_control: bool = True):
        """Evicting kv cache in kv_cache manager of the given task_id, restore task state to its pre-prefilling state
        Args:
            task_id: the task_id that need to be evicted
            congestion_control: 是否施加拥塞控制(将kvcache_block_threshold折半)。
                decode因容量不足被动逐出时为True；为打破prefill僵局而主动逐出低优先级
                在途任务以让位给更高优先级任务时应为False——此时折半阈值会重新挤占刚释放
                的容量，反而令被让位的任务无法推进，把临时拥塞变成永久死锁。

        Evicting Rules
        - For PP=1, raise error when current tasks list is empty
        - For PP>1, raise error when current DP rank has no tasks
        """
        if not self._task_evict_hook.check_evict(task_id):
            # evict task not in TaskPool, skip evicting
            return
        task = TaskPool.pool[task_id]

        # Remove kvcache of this task
        if task.has_unsync_new_token:
            task.evicting_with_new_token = True
        else:
            task.next_token = -1
        for cache_manager in self.cache_manager_dict.values():
            cache_manager.finalize_metadata_all_decode(
                task
            )  # 清除KVCacheManager中的元数据
        Backend.executor.special_step(
            [task.task_id], type="EndTask"
        )  # 清除KVCache中的元数据

        logger.warning(
            f"Evicted task {task_id} due to insufficient KV cache",
            extra={
                "task_id": task_id,
                "event": "scheduler_task_evicted",
                "kvcache_block_threshold": self.kvcache_block_threshold,
                "total_blocks": self.cache_manager_dict["main"].num_blocks,
            },
        )

        # Update metrics: record task eviction
        PrometheusMetricsCollector.inc_task_eviction()

        # For congestion control
        if congestion_control:
            self.kvcache_block_threshold = max(1, self.kvcache_block_threshold // 2)
        self._task_evict_hook.on_evict_done(task)

    def reorder_tasks_for_batching(self, task_ids):
        pass

    def update(self, cur_task_ids: list[str]) -> list[str]:
        self.sgroup_list.release_sgroup()
        removed_task_ids = []
        task_ids = cur_task_ids
        task_ids = list(set(task_ids))
        self.reorder_tasks_for_batching(task_ids)
        for task_id in task_ids:
            task = TaskPool.pool[task_id]
            if task.need_remove():
                removed_task_ids.append(task_id)
                if type(self) != SkewScheduler:
                    for cache_manager in self.cache_manager_dict.values():
                        cache_manager.finalize_metadata_all_decode(task)
                    self.kvcache_block_threshold = self.cache_manager_dict[
                        "main"
                    ].num_blocks
                    logger.debug(
                        f"Task({task_id}) finished decoding, increasing kvcache_block_threshold to {self.kvcache_block_threshold}, while the number of total blocks is {self.cache_manager_dict['main'].num_blocks}"
                    )
                TaskPool.remove(task_id)
                self._task_evict_hook.on_task_remove(task)

        if removed_task_ids:
            inc_completed_requests("worker", len(removed_task_ids))
            logger.debug(f"[scheduler.update] removed_tasks={removed_task_ids}")
            logger.info(
                f"Completed {len(removed_task_ids)} tasks",
                extra={
                    "event": "scheduler_task_completed",
                    "completed_tasks": len(removed_task_ids),
                },
            )

        return removed_task_ids

    def is_done(self):
        return len(TaskPool.pool) == 0


class SkewScheduler(Scheduler):

    def __init__(
        self,
        max_batch_size: int,
        cache_manager_dict: Optional[dict],
        *,
        dp_rank: int = 0,
        scheduler_type: str,
        prefill_chunk_size: Optional[int] = None,
    ):
        args = get_global_args()
        self.slot_handle = SlotHandle(max_batch_size, args.infer.pp_size)
        super().__init__(
            max_batch_size,
            ceil_div(max_batch_size, self.slot_handle.num_slots),  # prefill_num_tasks
            ceil_div(max_batch_size, self.slot_handle.num_slots),  # decode_num_tasks
            scheduler_type,
            cache_manager_dict,
            dp_rank=dp_rank,
            num_scheduler_groups=self.slot_handle.num_slots,
            prefill_chunk_size=prefill_chunk_size,
        )
        self.sgroup_list = SchedulerGroupList(
            num_sgroup=self.num_scheduler_groups, type="skew"
        )

    @override
    def schedule(
        self,
        strict_allowed_task_type: set[TaskType] = {TaskType.Prefill, TaskType.Decode},
        ready_task_ids: Optional[list[str]] = None,
    ) -> list[str]:
        sgroup_id = self.sgroup_list.get_current_sgroup()
        if TaskPool.is_empty():
            logger.debug("TaskPool is empty, returning empty task list.")
            return []

        # collect ready task ids
        source_task_ids = (
            ready_task_ids if ready_task_ids is not None else TaskPool.id_list
        )
        has_correct_dp_rank = lambda t: t.dp_rank is None or self.dp_rank == t.dp_rank
        task_ids = [
            tid
            for tid in source_task_ids
            if tid in TaskPool.pool
            and has_correct_dp_rank(TaskPool.pool[tid])
            and TaskPool.pool[tid].can_schedule()
            and TaskPool.pool[tid].task_type in strict_allowed_task_type
        ]
        if not task_ids:
            return []

        task_ids.sort(key=lambda x: self.scorer(TaskPool.pool[x]), reverse=True)

        # Prepare to schedule the earlist released free slot group
        sgroup_capacity = self.slot_handle.get_slot_size(sgroup_id)
        sgroup = self.sgroup_list.get_current_sgroup_all_tasks()

        # determine target task type (use highest-priority task's type)
        target_task_type = None
        for tid in task_ids:
            task = TaskPool.pool[tid]
            if task.sched_group_id is not None and task.sched_group_id != sgroup_id:
                continue
            target_task_type = task.task_type
            break

        if target_task_type is None:
            return []

        # When slot_group's lenght smaller than it's capacity, fill new tasks into it.
        if len(sgroup) < sgroup_capacity:
            # fill sgroup: Decode first then Prefill (if target is Decode), else only target type
            if target_task_type == TaskType.Decode:
                decode_fillable = [
                    tid
                    for tid in task_ids
                    if TaskPool.pool[tid].sched_group_id is None
                    and TaskPool.pool[tid].task_type == TaskType.Decode
                ]
                # capped by decode limit/remaining capacity
                num_decode = min(self.decode_num_tasks, sgroup_capacity - len(sgroup))
                sgroup.extend(decode_fillable[:num_decode])
                for tid in decode_fillable[:num_decode]:
                    TaskPool.pool[tid].sched_group_id = sgroup_id

                remaining_cap = sgroup_capacity - len(sgroup)
                if remaining_cap > 0:
                    # filter unbound Prefill tasks for remaining capacity
                    prefill_fillable = [
                        tid
                        for tid in task_ids
                        if TaskPool.pool[tid].sched_group_id is None
                        and TaskPool.pool[tid].task_type == TaskType.Prefill
                    ]
                    num_prefill = min(self.prefill_num_tasks, remaining_cap)
                    sgroup.extend(prefill_fillable[:num_prefill])
                    for tid in prefill_fillable[:num_prefill]:
                        TaskPool.pool[tid].sched_group_id = sgroup_id
            else:
                fillable = [
                    tid
                    for tid in task_ids
                    if TaskPool.pool[tid].sched_group_id is None
                    and TaskPool.pool[tid].task_type == target_task_type
                ]
                num_to_fill = min(
                    (
                        self.prefill_num_tasks
                        if target_task_type == TaskType.Prefill
                        else self.decode_num_tasks
                    ),
                    sgroup_capacity - len(sgroup),
                )
                to_add = fillable[:num_to_fill]
                sgroup.extend(to_add)
                for tid in to_add:
                    TaskPool.pool[tid].sched_group_id = sgroup_id

        curr_split = self.find_prefill_task_start_pos_sgroup(sgroup)

        if target_task_type == TaskType.Decode:
            ret_task_ids = sgroup[:curr_split]
        else:
            ret_task_ids = sgroup[curr_split:]
            if self.prefill_chunk_size is not None and ret_task_ids:
                limit = self._chunk_prefill_tasks_count(ret_task_ids)
                ret_task_ids = ret_task_ids[:limit]

        #  ensure only target type tasks are returned
        final_task_ids = [
            tid
            for tid in ret_task_ids
            if TaskPool.pool[tid].task_type == target_task_type
        ]

        if not final_task_ids:
            return []

        self.sgroup_list.set_task_ids(final_task_ids)

        for tid in sgroup:
            TaskPool.pool[tid].dp_rank = self.dp_rank

        return final_task_ids

    def _chunk_prefill_tasks_count(self, prefill_task_ids: list[str]) -> int:
        remaining_prefill_tokens = self.prefill_chunk_size
        for i in range(len(prefill_task_ids)):
            task = TaskPool.pool[prefill_task_ids[i]]
            task_remaining_tokens = task.prefix_tokens_len - task.consumed_req_tokens
            task_prefill_chunk_size = min(
                task_remaining_tokens, remaining_prefill_tokens
            )
            task.set_prefill_chunk_size_for_one_step(task_prefill_chunk_size)
            remaining_prefill_tokens -= task_prefill_chunk_size
            if remaining_prefill_tokens <= 0:
                break
        return i + 1

    def find_prefill_task_start_pos_sgroup(self, sgroup):
        n = len(sgroup)
        left, right = 0, n  # [left,right)
        while left < right:
            mid = (left + right) // 2
            task = TaskPool.pool[sgroup[mid]]
            if task.task_type == TaskType.Prefill:
                right = mid
            else:
                left = mid + 1
        return right

    @override
    def reorder_tasks_for_batching(self, task_ids):
        for task_id in task_ids:
            if (
                TaskPool.pool[task_id].need_remove()
                and TaskPool.pool[task_id].sched_group_id is not None
            ):
                sgroup_id = TaskPool.pool[task_id].sched_group_id
                sgroup = self.sgroup_list.get_sgroup_all_tasks(sgroup_id)
                index = sgroup.index(task_id)
                sgroup[index] = sgroup[-1]
                sgroup.pop()
