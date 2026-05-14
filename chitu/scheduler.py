# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import time
import math
from logging import getLogger
from typing import Optional
from typing_extensions import override
from collections import deque, defaultdict

from chitu.task import TaskPool, TaskType, Task
from chitu.global_vars import get_global_args, SlotHandle
from chitu.utils import ceil_div
from chitu.backend import Backend
from chitu.distributed.partition import compute_local_batch_size_dist_in_dp
from chitu.kv_cache import KVCacheManagerBase, PagedKVCacheManager
from chitu.metrics.prometheus_collector import (
    PrometheusMetricsCollector,
    inc_completed_requests,
)
from chitu.distributed.pd_disaggregation.pd_log_utils import pd_verbose_enabled

logger = getLogger(__name__)


class SchedulerGroupList:
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

    def set_task_ids(self, task_ids: list[str]):
        self._sgroup_list[self.current_sgroup_id] = task_ids
        for task_id in task_ids:
            TaskPool.pool[task_id].wait()

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
            prefill_chunk_size_per_dp: Optional[int] = (
                infer_args.prefill_chunk_size // infer_args.dp_size
                + int(dp_rank < infer_args.prefill_chunk_size % infer_args.dp_size)
            )
        else:
            prefill_chunk_size_per_dp: Optional[int] = None

        if infer_args.cache_type == "skew":
            return SkewScheduler(
                max_reqs_per_dp,
                None,
                dp_rank=dp_rank,
                original_scheduler_type=args.type.lower(),
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
        assert (
            type(cache_manager_dict["main"]) == PagedKVCacheManager
        ), f"Scheduler only support PagedKVCacheManager, found {type(cache_manager_dict['main'])}"

        return Scheduler(
            max_reqs_per_dp,
            prefill_num_tasks,
            decode_num_tasks,
            Scheduler._normalize_scheduler_type(args.type.lower()),
            cache_manager_dict,
            num_scheduler_groups=infer_args.pp_size,
            dp_rank=dp_rank,
            original_scheduler_type=args.type.lower(),
            prefill_chunk_size=prefill_chunk_size_per_dp,
        )

    @staticmethod
    def _normalize_scheduler_type(scheduler_type: str) -> str:
        """Map aliases and PD-specific scheduler types to base types.

        - "prefill_only" -> "prefill_first"
        - "decode_only"  -> "fcfs"
        """
        parts = [p.strip().lower() for p in scheduler_type.split(",") if p.strip()]
        normalized_parts = []
        for part in parts:
            if part == "prefill_only":
                normalized_parts.append("prefill_first")
            elif part == "decode_only":
                normalized_parts.append("fcfs")
            else:
                normalized_parts.append(part)

        if not normalized_parts:
            normalized_parts = ["fcfs"]
        return ",".join(normalized_parts)

    def __init__(
        self,
        max_runing_tasks: int,
        prefill_num_tasks: int,
        decode_num_tasks: int,
        scheduler_type: str,
        cache_manager_dict: Optional[dict[str, KVCacheManagerBase]],
        *,
        num_scheduler_groups: int,
        dp_rank: int = 0,
        original_scheduler_type: str = None,
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
        self.max_runing_tasks = max_runing_tasks
        self.prefill_num_tasks = prefill_num_tasks
        self.decode_num_tasks = decode_num_tasks
        self.prefill_chunk_size = prefill_chunk_size
        self.cache_manager_dict = cache_manager_dict
        self.num_scheduler_groups = num_scheduler_groups
        self.sgroup_list = SchedulerGroupList(num_sgroup=self.num_scheduler_groups)
        self.dp_rank = dp_rank

        # strict-only gating derived from original type string
        self.strict_allowed_task_type: set[TaskType] = self._extract_strict_task_type(
            original_scheduler_type
            if original_scheduler_type is not None
            else scheduler_type
        )

        # determine scoring method
        self.scorers = []
        scheduler_type = Scheduler._normalize_scheduler_type(scheduler_type)
        for st in scheduler_type.split(","):
            if st == "request_preset":
                self.scorers.append(lambda task: task.priority)
            elif st == "prefill_first":
                self.scorers.append(
                    lambda task: 1 if task.task_type == TaskType.Prefill else 0
                )
            elif st == "fcfs" or st == "fifo":
                self.scorers.append(lambda task: -task.arrv_ts)
            elif st == "stride":
                self.scorers.append(
                    lambda task: task.priority * (self.scheduling_ts - task.arrv_ts)
                )
            elif st == "deadline":
                self.scorers.append(lambda task: -task.sched_ddl)
            elif st == "prefix_align":
                self.scorers.append(lambda task: -task.prefix_tokens_len)
            else:
                raise NotImplementedError(f"Scheduler type {st} not implemented")

        self.reset_kvcache_block_threshold()
        self.is_warmup_stage = False
        self.has_schedule_overlap = get_global_args().infer.schedule_overlap
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

    def scorer(self, task: Task):
        if self.is_warmup_stage:
            fn = lambda task: (
                1 if task.task_type == TaskType.Prefill else 0
            )  # prefill first
            return (fn(task),)
        return tuple(fn(task) for fn in self.scorers)

    def _num_prefill_cached_tokens(self, task: Task) -> int:
        completed_tokens = task.kv_cache_len_used_in_completed_steps
        if not self.cache_manager_dict["main"].enable_prefix_caching:
            return completed_tokens

        num_cached_tokens = task.prefix_tokens_len
        for manager in list(self.cache_manager_dict.values()):
            num_cached_tokens = min(
                num_cached_tokens, manager.num_cached_blocks(task) * manager.block_size
            )
            if num_cached_tokens <= completed_tokens:
                return completed_tokens

        return num_cached_tokens

    def _check_prefill_capacity(self, task, cached_len: int) -> bool:
        for name, cache_manager in self.cache_manager_dict.items():
            cur_blocks = ceil_div(cached_len, cache_manager.block_size)
            target_blocks = ceil_div(
                cached_len + task.next_req_tokens_len,
                cache_manager.block_size,
            )
            block_threshold = (
                self.kvcache_block_threshold
                if name == "main"
                else cache_manager.num_blocks
            )
            available_blocks = (
                block_threshold
                - cache_manager.num_active_blocks
                - cache_manager.num_cached_idle_blocks(
                    task, max_cached_token_len=cached_len
                )
            )
            if target_blocks - cur_blocks > available_blocks:
                return False
        return True

    def _check_decode_capacity(self, task) -> bool:
        for _, cache_manager in self.cache_manager_dict.items():
            available_blocks = (
                cache_manager.num_blocks - cache_manager.num_active_blocks
            )
            cur_blocks = len(cache_manager.task_to_cache_ids[task.task_id])
            target_blocks = ceil_div(
                task.kv_cache_len_used_in_completed_steps_and_next_step,
                cache_manager.block_size,
            )
            if target_blocks - cur_blocks > available_blocks:
                return False
        return True

    def _prepare_prefill_metadata(self, task, cached_len: int) -> None:
        assert (
            task.consumed_req_tokens <= cached_len <= task.prefix_tokens_len
        ), f"{task.consumed_req_tokens} vs {cached_len} vs {task.prefix_tokens_len}"
        if cached_len == task.prefix_tokens_len:
            cached_len = task.prefix_tokens_len - 1

        for name, cache_manager in self.cache_manager_dict.items():
            task.new_cache_ids[name] = cache_manager.prepare_metadata_before_prefill(
                task, max_cached_token_len=cached_len
            )

        task.inc_hit_tokens = cached_len - task.consumed_req_tokens
        task.consumed_req_tokens = cached_len

    def _prepare_decode_metadata(self, task) -> None:
        for name, cache_manager in self.cache_manager_dict.items():
            task.new_cache_ids[name] = cache_manager.prepare_metadata_before_decode(
                task
            )

    def prepare_for_schedule(self) -> None:
        self.sgroup_list.switch_to_next_sgroup()

    def schedule(
        self,
        strict_allowed_task_type: set[TaskType] = {TaskType.Prefill, TaskType.Decode},
    ) -> list[str]:
        sgroup_id = self.sgroup_list.get_current_sgroup()
        if TaskPool.is_empty():
            logger.debug("TaskPool is empty, returning empty task list.")
            return []

        self.scheduling_ts = time.perf_counter_ns()

        # collect ready task ids
        n_running = len(self.cache_manager_dict["main"].tid_to_cached_len)

        task_ids: list[str] = []
        for task_id in TaskPool.id_list:
            task = TaskPool.pool[task_id]
            if (
                task.dp_rank is None
                and task.preferred_dp_rank is not None
                and task.preferred_dp_rank != self.dp_rank
            ):
                continue
            if (
                task.dp_rank is None
                or self.dp_rank == task.dp_rank
                and task.can_schedule()
            ):
                if n_running == self.max_runing_tasks and task.dp_rank is None:
                    continue
                if task.dp_rank is None:
                    n_running += 1
                task_ids.append(task_id)

        # enforce strict-only gating if enabled
        strict_allowed_task_type = strict_allowed_task_type.intersection(
            self.strict_allowed_task_type
        )
        if len(strict_allowed_task_type) == 0:
            # chitu_main 里面先用 strict_allowed_task_type=Prefill调一次 schedule()，如果拿不到任务，再用 Decode 再调一次
            # 在PD分离只有decode_only，就会走到这里
            return []
        task_ids = [
            tid
            for tid in task_ids
            if TaskPool.pool[tid].task_type in strict_allowed_task_type
        ]
        if len(task_ids) == 0:
            # No avaliable tasks, returning empty task list.
            # This is in a busy loop waiting for tasks, so don't print logs here.
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
        self.sgroup_list.set_task_ids(task_ids)

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

        if pd_verbose_enabled():
            logger.info(
                "[PD_STATS][decode.ready_to_exec] "
                f"count={n} avg_ms={avg:.2f} "
                f"p50_ms={_pct(50):.2f} p90_ms={_pct(90):.2f} "
                f"p95_ms={_pct(95):.2f} p99_ms={_pct(99):.2f}"
            )

    def _schedule_prefill_tasks(self, task_ids: list[str]) -> list[str]:
        """Prefill tasks scheduling with congestion control
        Args:
            task_ids: list of unwait task ids
        Return:
            sched_out_task_ids: list of unwait prefill task ids
        """
        sched_out_task_ids = []
        prefill_tokens = 0
        kv_cache_manager = self.cache_manager_dict["main"]
        block_size = kv_cache_manager.block_size

        for task_id in task_ids:
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

            for _, cache_manager in self.cache_manager_dict.items():
                cache_manager.ensure_task_token_blocks(task)

            # check task's remain tokens

            # task.prefix_tokens_len: in prefill stage, it's prompt length
            # num_cached_tokens: number of tokens that are hit by cached_idle_blocks or active_blocks
            num_cached_tokens = self._num_prefill_cached_tokens(task)
            num_uncomputed_tokens = task.prefix_tokens_len - num_cached_tokens
            if num_uncomputed_tokens == 0:
                # prompt_len == num_cached_tokens
                # All prompt tokens are hit, but an additional step is required to compute the logits.
                # This task in the step consumes 1 of the chunk_prefill_size budget, but doesn't need additional KV cache capacity.
                # Only the last token of the prompt requires an extra prefill step.
                prefill_tokens += 1
                task.set_prefill_chunk_size_for_one_step(1)
                sched_out_task_ids.append(task_id)
                self._prepare_prefill_metadata(task, num_cached_tokens)
                continue

            task_prefill_chunk_size = min(
                task_prefill_chunk_size, num_uncomputed_tokens
            )
            task_origin_prefill_chunk_size = task.prefill_chunk_size
            task.set_prefill_chunk_size_for_one_step(task_prefill_chunk_size)

            if not self._check_prefill_capacity(task, num_cached_tokens):
                task.prefill_chunk_size = task_origin_prefill_chunk_size
                break

            prefill_tokens += task_prefill_chunk_size
            sched_out_task_ids.append(task_id)
            self._prepare_prefill_metadata(task, num_cached_tokens)

        if (
            len(sched_out_task_ids) == 0
            and self.kvcache_block_threshold == kv_cache_manager.num_blocks
            and kv_cache_manager.num_active_blocks == 0
        ):
            raise RuntimeError(
                "KV cache capacity is insufficient to support prefilling.\n"
                f"  - Block size: {block_size}\n"
                f"  - available blocks: {self.kvcache_block_threshold - kv_cache_manager.num_active_blocks}\n"
                f"  - Prefill chunk size: {self.prefill_chunk_size if self.prefill_chunk_size is not None else 'inf'}\n"
                "However, all prefill prompts are too long:\n"
                f"{[TaskPool.pool[idx].prefix_tokens_len for idx in task_ids if TaskPool.pool[idx].task_type == TaskType.Prefill]}"
            )
        return sched_out_task_ids

    def _schedule_decode_tasks(self, task_ids: list[str]) -> list[str]:
        """Decode tasks scheduling, evicting the last prioriety decode task when these is no more block
        Args:
            task_ids: list of unwait task ids
        Return:
            decode_task_ids: list of unwait decode task ids
        """

        decode_task_ids = deque()
        cached_prefill_task_ids = []  # 已开始prefill但未完成的任务

        for tid in task_ids:
            task = TaskPool.pool[tid]
            if task.task_type == TaskType.Decode:
                decode_task_ids.append(tid)
            elif (
                task.task_type == TaskType.Prefill
                and task.task_id in self.cache_manager_dict["main"].tid_to_cached_len
            ):
                cached_prefill_task_ids.append(tid)

        if not decode_task_ids:
            return []

        sched_out_task_ids = []
        evict_tasks = []

        while decode_task_ids and len(sched_out_task_ids) < self.decode_num_tasks:
            candidate_task_id = decode_task_ids.popleft()
            candidate_task = TaskPool.pool[candidate_task_id]

            if self._check_decode_capacity(candidate_task):
                sched_out_task_ids.append(candidate_task_id)
                self._prepare_decode_metadata(candidate_task)
                continue

            if not decode_task_ids and not cached_prefill_task_ids:
                # 容量不足，无任务可逐出
                break

            # 容量不足，可逐出低优先级任务（优先逐出已开始prefill但未完成的任务）
            decode_task_ids.appendleft(candidate_task_id)
            if cached_prefill_task_ids:
                evict_task_id = cached_prefill_task_ids.pop()
            else:
                evict_task_id = decode_task_ids.pop()
            evict_tasks.append(evict_task_id)
            self.evict_task(evict_task_id)

        if not sched_out_task_ids and len(self.sgroup_list) == 0:
            if len(decode_task_ids) > 0:
                prefix_len = TaskPool.pool[decode_task_ids[0]].prefix_tokens_len
            else:
                prefix_len = -1
            raise Exception(
                f"KV_cache capacity is insufficient to support decoding completion (batch_size=1, prefix_len={prefix_len})."
            )

        if len(evict_tasks) > 0:
            logger.warning(
                f"KV cache capacity reached limit, forcing eviction of {len(evict_tasks)} decode tasks, this may impact throughput and latency. To prevent performance degradation, consider decreasing max_batch_size or increasing or num_blocks."
            )

        return sched_out_task_ids

    def evict_task(self, task_id: str):
        """Evicting kv cache in kv_cache manager of the given task_id, restore task state to its pre-prefilling state
        Args:
            task_id: the task_id that need to be evicted

        Evicting Rules
        - For PP=1, raise error when current tasks list is empty
        - For PP>1, raise error when current DP rank has no tasks
        """
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

        # Restore the task's status to before prefill
        task.task_type = TaskType.Prefill
        task.prefill_chunk_size = None
        task.consumed_req_tokens = 0
        task.sched_group_id = None
        task.dp_rank = None

        # For congestion control
        self.kvcache_block_threshold = max(1, self.kvcache_block_threshold // 2)

    @staticmethod
    def _extract_strict_task_type(scheduler_type: str) -> set[TaskType]:
        """Return TaskType when strict-only is requested, otherwise None.

        Recognized tokens:
        - "prefill_only" => TaskType.Prefill
        - "decode_only"  => TaskType.Decode
        If both appear, no strict gating will be applied.
        """
        if not scheduler_type:
            return {TaskType.Prefill, TaskType.Decode}
        parts = [p.strip().lower() for p in scheduler_type.split(",") if p.strip()]
        has_prefill_only = any(p == "prefill_only" for p in parts)
        has_decode_only = any(p == "decode_only" for p in parts)
        if has_prefill_only and not has_decode_only:
            return {TaskType.Prefill}
        if has_decode_only and not has_prefill_only:
            return {TaskType.Decode}
        return {TaskType.Prefill, TaskType.Decode}

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
        original_scheduler_type: str,
        prefill_chunk_size: Optional[int] = None,
    ):
        args = get_global_args()
        self.slot_handle = SlotHandle(max_batch_size, args.infer.pp_size)
        super().__init__(
            max_batch_size,
            ceil_div(max_batch_size, self.slot_handle.num_slots),  # prefill_num_tasks
            ceil_div(max_batch_size, self.slot_handle.num_slots),  # decode_num_tasks
            Scheduler._normalize_scheduler_type(original_scheduler_type),
            cache_manager_dict,
            dp_rank=dp_rank,
            num_scheduler_groups=self.slot_handle.num_slots,
            original_scheduler_type=original_scheduler_type,
            prefill_chunk_size=prefill_chunk_size,
        )
        self.sgroup_list = SchedulerGroupList(
            num_sgroup=self.num_scheduler_groups, type="skew"
        )

    @override
    def schedule(
        self,
        strict_allowed_task_type: set[TaskType] = {TaskType.Prefill, TaskType.Decode},
    ) -> list[str]:
        sgroup_id = self.sgroup_list.get_current_sgroup()
        if TaskPool.is_empty():
            logger.debug("TaskPool is empty, returning empty task list.")
            return []

        # enforce strict-only gating if enabled
        strict_allowed_task_type = strict_allowed_task_type.intersection(
            self.strict_allowed_task_type
        )

        # collect ready task ids
        has_correct_dp_rank = lambda t: t.dp_rank is None or self.dp_rank == t.dp_rank
        task_ids = [
            tid
            for tid in TaskPool.id_list
            if has_correct_dp_rank(TaskPool.pool[tid])
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
