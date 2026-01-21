# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import time
from logging import getLogger
from typing import Optional
from typing_extensions import override
from collections import deque, defaultdict

from chitu.task import (
    PackedTasks,
    TaskPool,
    TaskType,
    PackedTasksBase,
    SerializedPackedTasksPayloadType,
    PPTaskCollector,
)
from chitu.global_vars import get_global_args, SlotHandle
from chitu.utils import ceil_div
from chitu.backend import Backend
from chitu.distributed.partition import compute_local_batch_size_dist_in_dp
from chitu.metrics.prometheus_collector import PrometheusMetricsCollector

logger = getLogger(__name__)


class Scheduler:
    @staticmethod
    def build(args, infer_args, *, dp_rank: int):
        max_reqs_per_dp = compute_local_batch_size_dist_in_dp(
            infer_args.max_reqs, infer_args.dp_size
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

        return Scheduler(
            max_reqs_per_dp,
            prefill_num_tasks,
            decode_num_tasks,
            Scheduler._normalize_scheduler_type(args.type.lower()),
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
        self.num_scheduler_groups = num_scheduler_groups
        self.free_sgroups = deque(
            range(num_scheduler_groups)
        )  # scheduler group doesn't have any waiting task.
        self.used_sgroups = set()  # scheduler group has waiting tasks.
        self.sgroup_waiting_tasks = defaultdict(set)  # {sched_group_id: waiting_tasks}.
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

        self.kvcache_block_threshold = Backend.cache_manager.get_num_blocks()
        self.is_warmup_stage = False
        self.has_schedule_overlap = get_global_args().infer.schedule_overlap

    def reset_kvcache_block_threshold(self):
        self.kvcache_block_threshold = Backend.cache_manager.get_num_blocks()

    def start_warmup(self):
        self.is_warmup_stage = True

    def end_warmup(self):
        self.is_warmup_stage = False

    def scorer(self, task):
        if self.is_warmup_stage:
            fn = lambda task: (
                1 if task.task_type == TaskType.Prefill else 0
            )  # prefill first
            return (fn(task),)
        return tuple(fn(task) for fn in self.scorers)

    def schedule(
        self,
        strict_allowed_task_type: set[TaskType] = {TaskType.Prefill, TaskType.Decode},
    ) -> list[str]:
        if TaskPool.is_empty():
            logger.debug("TaskPool is empty, returning empty task list.")
            return []

        if not self.free_sgroups:
            logger.debug("No available scheduler group, returning empty task list.")
            return []

        self.scheduling_ts = time.perf_counter_ns()

        # collect ready task ids
        n_running = sum(
            1
            for task_id in TaskPool.id_list
            if TaskPool.pool[task_id].dp_rank == self.dp_rank
        )
        task_ids: list[str] = []
        for task_id in TaskPool.id_list:
            task = TaskPool.pool[task_id]
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
            raise RuntimeError("No task type is allowed for this scheduling")
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
                task_ids = prefill_task_ids[: self.prefill_num_tasks]
            else:
                # No available prefill tasks, we can schedule decode at this condition
                filter_task_type = TaskType.Decode

        # scheduling decode tasks
        if filter_task_type == TaskType.Decode:
            task_ids = self._schedule_decode_tasks(task_ids)[: self.decode_num_tasks]

        # Allocate sgroup for for task_ids
        sgroup_id = self.free_sgroups.popleft()
        self.used_sgroups.add(sgroup_id)
        self.sgroup_waiting_tasks[sgroup_id] = set(task_ids)

        # postprocess
        for task_id in task_ids:
            TaskPool.pool[task_id].sched_ts = self.scheduling_ts
            TaskPool.pool[task_id].sched_group_id = sgroup_id
            TaskPool.pool[task_id].dp_rank = self.dp_rank

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

    def _schedule_prefill_tasks(self, task_ids: list[str]) -> list[str]:
        """Prefill tasks scheduling with congestion control
        Args:
            task_ids: list of unwait task ids
        Return:
            prefill_task_ids: list of unwait prefill task ids
        """
        prefill_task_ids = list(
            filter(
                lambda task_id: TaskPool.pool[task_id].task_type == TaskType.Prefill,
                task_ids,
            )
        )

        # Apply chunk prefill
        if self.prefill_chunk_size is not None:
            num_chunk_prefill_tasks = self._chunk_prefill_tasks_count(prefill_task_ids)
            prefill_task_ids = prefill_task_ids[:num_chunk_prefill_tasks]

        # Check KVCacheManager's capacity
        #
        # NOTE: Please directly compute number of blocks here instead of getting from
        # `Backend.cache_manager`, because here we may be scheduling for another rank.
        num_used_blocks = 0
        for task_id in TaskPool.pool.keys():
            task = TaskPool.pool[task_id]
            if task.dp_rank == self.dp_rank:
                cur_blocks = ceil_div(
                    task.kv_cache_len_used_in_completed_steps,
                    Backend.cache_manager.get_block_size(),
                )
                num_used_blocks += cur_blocks
        num_need_blocks = 0
        num_tasks = 0
        for task_id in prefill_task_ids:
            task = TaskPool.pool[task_id]
            target_seq_len = task.consumed_req_tokens + len(task.next_req_tokens())
            assert (
                task.kv_cache_len_used_in_completed_steps
                <= task.kv_cache_len_used_in_completed_steps_and_next_step
            )
            cur_blocks = ceil_div(
                task.kv_cache_len_used_in_completed_steps,
                Backend.cache_manager.get_block_size(),
            )
            target_blocks = ceil_div(
                task.kv_cache_len_used_in_completed_steps_and_next_step,
                Backend.cache_manager.get_block_size(),
            )
            num_need_blocks += target_blocks - cur_blocks
            if num_used_blocks + num_need_blocks > self.kvcache_block_threshold:
                break
            num_tasks += 1

        if (
            num_tasks == 0
            and self.kvcache_block_threshold == Backend.cache_manager.get_num_blocks()
            and num_used_blocks == 0
        ):
            prefix_len = TaskPool.pool[prefill_task_ids[0]].prefix_tokens_len
            raise RuntimeError(
                f"KV_cache capacity is insufficient to support prefilling (batch_size=1, prefix_len={prefix_len})"
            )

        return prefill_task_ids[:num_tasks]

    def _chunk_prefill_tasks_count(self, prefill_task_ids: list[str]) -> int:
        prefill_tokens = 0
        total_prefill_tokens_len_to_schedule = 0

        for i in range(len(prefill_task_ids)):
            total_prefill_tokens_len_to_schedule += (
                TaskPool.pool[prefill_task_ids[i]].prefix_tokens_len
                - TaskPool.pool[prefill_task_ids[i]].consumed_req_tokens
            )

        for i in range(len(prefill_task_ids)):
            task = TaskPool.pool[prefill_task_ids[i]]
            task_remaining_tokens = task.prefix_tokens_len - task.consumed_req_tokens
            if (
                self.num_scheduler_groups == 1
                or task_remaining_tokens > self.prefill_chunk_size
                or total_prefill_tokens_len_to_schedule
                >= self.num_scheduler_groups * self.prefill_chunk_size
            ):
                task_prefill_chunk_size = min(
                    task_remaining_tokens, self.prefill_chunk_size - prefill_tokens
                )
                task.set_prefill_chunk_size_for_one_step(task_prefill_chunk_size)
                prefill_tokens += task_prefill_chunk_size
                if prefill_tokens >= self.prefill_chunk_size:
                    return i + 1
            else:
                if prefill_tokens + task_remaining_tokens > self.prefill_chunk_size:
                    return i
                task_prefill_chunk_size = task_remaining_tokens
                task.set_prefill_chunk_size_for_one_step(task_prefill_chunk_size)
                prefill_tokens += task_prefill_chunk_size
        return i + 1

    def _schedule_decode_tasks(self, task_ids: list[str]) -> list[str]:
        """Decode tasks scheduling, evicting the last prioriety decode task when these is no more block
        Args:
            task_ids: list of unwait task ids
        Return:
            decode_task_ids: list of unwait decode task ids
        """
        decode_task_ids = list(
            filter(
                lambda task_id: TaskPool.pool[task_id].task_type == TaskType.Decode,
                task_ids,
            )
        )

        def has_enough_block():
            # NOTE: Please directly compute number of blocks here instead of getting from
            # `Backend.cache_manager`, because here we may be scheduling for another rank.
            num_need_blocks = 0
            for task_id in TaskPool.pool.keys():
                task = TaskPool.pool[task_id]
                if task.finished_decode:
                    continue
                assert (
                    task.kv_cache_len_used_in_completed_steps
                    <= task.kv_cache_len_used_in_completed_steps_and_next_step
                )
                if task_id in decode_task_ids:
                    seq_len = task.kv_cache_len_used_in_completed_steps_and_next_step
                elif task.dp_rank == self.dp_rank:
                    seq_len = task.kv_cache_len_used_in_completed_steps
                else:
                    seq_len = 0
                num_need_blocks += ceil_div(
                    seq_len, Backend.cache_manager.get_block_size()
                )
            if num_need_blocks <= Backend.cache_manager.get_num_blocks():
                return True
            logger.debug(
                f"Cache manager has no more free blocks to support current decoding tasks: "
                f"need {num_need_blocks} blocks in total, but only {Backend.cache_manager.get_num_blocks()} blocks in total."
            )
            return False

        evicted_tasks = []
        while not has_enough_block():
            if len(decode_task_ids) == 1:
                prefix_len = TaskPool.pool[decode_task_ids[0]].prefix_tokens_len
                raise Exception(
                    f"KV_cache capacity is insufficient to support decoding completion (batch_size=1, prefix_len={prefix_len})."
                )
            need_evict_task_id = decode_task_ids.pop()
            self.evict_decode_task(need_evict_task_id)
            evicted_tasks.append(need_evict_task_id)

        if len(evicted_tasks) > 0:
            logger.warning(
                f"KV cache capacity reached limit, forcing eviction of {len(evicted_tasks)} decode tasks, this may impact throughput and latency. To prevent performance degradation, consider decreasing max_reqs or increasing or num_blocks."
            )

        return decode_task_ids

    def evict_decode_task(self, task_id: str):
        """Evicting kv cache in kv_cache manager of the given task_id, restore task state to its pre-prefilling state
        Args:
            task_id: the task_id that need to be evicted
        """
        task = TaskPool.pool[task_id]
        if task.finished_decode:
            return

        # Remove kvcache of this task
        task.next_token = -1
        task.evicting = True
        task.handle = None
        tasks = PackedTasksBase(
            num_tasks=1,
            task_ids=[task_id],
            req_ids=[task.req.request_id],
            task_type=TaskType.Special,
            payload_type=SerializedPackedTasksPayloadType.EndTask,
        )
        Backend.executor.step(tasks)
        PPTaskCollector.update_ongoing(PackedTasks([task_id]))
        logger.warning(
            f"Evicted task {task_id} due to insufficient KV cache",
            extra={
                "task_id": task_id,
                "event": "scheduler_task_evicted",
                "kvcache_block_threshold": self.kvcache_block_threshold,
                "total_blocks": Backend.cache_manager.get_num_blocks(),
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

    def update(self, cur_task_ids: list[str], unwait_task_ids: list[str] = []):
        removed_task_ids = []
        removed_kvcache_task_ids = []
        task_ids = cur_task_ids + unwait_task_ids
        task_ids = list(set(task_ids))
        self.reorder_tasks_for_batching(task_ids)
        for task_id in task_ids:
            # Update Task's sched_group_id and sgroup_waiting_cnt
            if (
                TaskPool.pool[task_id].finish_last_step()
                and TaskPool.pool[task_id].sched_group_id is not None
            ):
                sgroup_id = TaskPool.pool[task_id].sched_group_id
                if task_id in self.sgroup_waiting_tasks[sgroup_id]:
                    self.sgroup_waiting_tasks[sgroup_id].remove(task_id)
                if not isinstance(self, SkewScheduler):
                    TaskPool.pool[task_id].sched_group_id = None
            if isinstance(self, SkewScheduler) and not TaskPool.pool[task_id].running():
                TaskPool.pool[task_id].sched_group_id = None
        for task_id in task_ids:
            task = TaskPool.pool[task_id]
            if (
                not task.finished_decode
                and task.task_type == TaskType.Decode
                and (
                    (not self.has_schedule_overlap and task.need_remove())
                    or (self.has_schedule_overlap and not task.has_model_run())
                )
            ):
                task.finished_decode = True
                removed_kvcache_task_ids.append(task_id)
                num_total_blocks = Backend.cache_manager.get_num_blocks()
                self.kvcache_block_threshold = num_total_blocks
                logger.debug(
                    f"Task({task_id}) finished decoding, increasing kvcache_block_threshold to {self.kvcache_block_threshold}, while the number of total blocks is {num_total_blocks}"
                )
            if task.need_remove():
                removed_task_ids.append(task_id)
                TaskPool.remove(task_id)

        if removed_task_ids:
            logger.debug(f"[scheduler.update] removed_decode_tasks={removed_task_ids}")

        # Update used_sgroups and free_sgroups according to sgroup status
        for sgroup_id in list(self.used_sgroups):
            if len(self.sgroup_waiting_tasks[sgroup_id]) == 0:
                self.used_sgroups.remove(sgroup_id)
                self.free_sgroups.append(sgroup_id)

        if removed_task_ids:
            logger.info(
                f"Completed {len(removed_task_ids)} tasks",
                extra={
                    "event": "scheduler_task_completed",
                    "completed_tasks": len(removed_task_ids),
                },
            )

        return removed_task_ids, removed_kvcache_task_ids

    def is_done(self):
        return len(TaskPool.pool) == 0


class SkewScheduler(Scheduler):

    def __init__(
        self,
        max_reqs: int,
        *,
        dp_rank: int = 0,
        original_scheduler_type: str,
        prefill_chunk_size: Optional[int] = None,
    ):
        args = get_global_args()
        self.slot_handle = SlotHandle(max_reqs, args.infer.pp_size)
        super().__init__(
            max_reqs,
            ceil_div(max_reqs, self.slot_handle.num_slots),  # prefill_num_tasks
            ceil_div(max_reqs, self.slot_handle.num_slots),  # decode_num_tasks
            Scheduler._normalize_scheduler_type(original_scheduler_type),
            dp_rank=dp_rank,
            num_scheduler_groups=self.slot_handle.num_slots,
            original_scheduler_type=original_scheduler_type,
            prefill_chunk_size=prefill_chunk_size,
        )
        self.sgroup_list = [[] for _ in range(self.slot_handle.num_slots)]
        self.sgroup_waiting_tasks = defaultdict(set)  # {slot_group_id: waiting_tasks}
        self.free_sgroups = deque(
            range(self.slot_handle.num_slots)
        )  # free slot_group doesn't have any waiting tasks
        self.used_sgroups = set()  # used slot_group has one or more waiting task.

    @override
    def schedule(
        self,
        strict_allowed_task_type: set[TaskType] = {TaskType.Prefill, TaskType.Decode},
    ) -> list[str]:

        # no available slot group or empty task pool
        if not self.free_sgroups or TaskPool.is_empty():
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
        sgroup_id = self.free_sgroups.popleft()
        sgroup = self.sgroup_list[sgroup_id]
        sgroup_capacity = self.slot_handle.get_slot_size(sgroup_id)

        # determine target task type (use highest-priority task's type)
        target_task_type = None
        for tid in task_ids:
            task = TaskPool.pool[tid]
            if task.sched_group_id is not None and task.sched_group_id != sgroup_id:
                continue
            target_task_type = task.task_type
            break

        if target_task_type is None:
            self.free_sgroups.append(sgroup_id)
            return []

        self.used_sgroups.add(sgroup_id)

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
            self.used_sgroups.remove(sgroup_id)
            self.free_sgroups.append(sgroup_id)
            return []

        self.sgroup_waiting_tasks[sgroup_id] = set(final_task_ids)
        for tid in sgroup:
            TaskPool.pool[tid].dp_rank = self.dp_rank

        return final_task_ids

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
                not TaskPool.pool[task_id].running()
                and TaskPool.pool[task_id].sched_group_id is not None
            ):
                sgroup_id = TaskPool.pool[task_id].sched_group_id
                if task_id in self.sgroup_waiting_tasks[sgroup_id]:
                    self.sgroup_waiting_tasks[sgroup_id].remove(task_id)
                sgroup = self.sgroup_list[sgroup_id]
                index = sgroup.index(task_id)
                sgroup[index] = sgroup[-1]
                sgroup.pop()
