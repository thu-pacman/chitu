# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import time
from logging import getLogger
from typing import Optional
from typing_extensions import override
from collections import deque, defaultdict

from chitu.task import TaskPool, TaskType, DPTaskCollector
from chitu.global_vars import get_slot_handle, get_global_args
from chitu.utils import ceil_div
from chitu.distributed.parallel_state import get_dp_group, get_pp_group
from chitu.backend import Backend
from chitu.task import (
    PackedTasksBase,
    SerializedPackedTasksPayloadType,
)
from chitu.metrics.prometheus_collector import PrometheusMetricsCollector

logger = getLogger(__name__)


class Scheduler:
    @staticmethod
    def build(args, infer_args):
        if get_dp_group().group_size > 1:
            num_tasks_per_rank = ceil_div(
                infer_args.max_reqs, get_dp_group().group_size
            )
            return DPFifoScheduler(num_tasks_per_rank)

        if get_slot_handle():
            return SkewScheduler(
                infer_args.max_reqs,
                args.type.lower(),
                infer_args.prefill_chunk_size,
            )

        if infer_args.pp_size > 1:
            if args.pp_config.prefill_num_tasks_divided_by_pp:
                prefill_num_tasks = ceil_div(infer_args.max_reqs, infer_args.pp_size)
            else:
                prefill_num_tasks = args.pp_config.prefill_num_tasks
            if args.pp_config.enforce_decode_num_tasks_max:
                decode_num_tasks = ceil_div(infer_args.max_reqs, infer_args.pp_size)
            else:
                decode_num_tasks = args.pp_config.decode_num_tasks
        else:
            prefill_num_tasks = infer_args.max_reqs
            decode_num_tasks = infer_args.max_reqs

        return Scheduler(
            prefill_num_tasks,
            decode_num_tasks,
            Scheduler._normalize_scheduler_type(args.type.lower()),
            num_scheduler_groups=infer_args.pp_size,
            original_scheduler_type=args.type.lower(),
            prefill_chunk_size=infer_args.prefill_chunk_size,
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
        prefill_num_tasks: int,
        decode_num_tasks: int,
        scheduler_type: str,
        num_scheduler_groups: int,
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
        """

        super().__init__()
        assert prefill_num_tasks > 0, "prefill_num_tasks must be greater than 0"
        assert decode_num_tasks > 0, "decode_num_tasks must be greater than 0"
        self.prefill_num_tasks = prefill_num_tasks
        self.decode_num_tasks = decode_num_tasks
        self.prefill_chunk_size = prefill_chunk_size
        self.num_scheduler_groups = num_scheduler_groups
        self.free_sgroups = deque(
            range(num_scheduler_groups)
        )  # scheduler group doesn't have any waiting task.
        self.used_sgroups = set()  # scheduler group has waiting tasks.
        self.sgroup_waiting_tasks = defaultdict(set)  # {sched_group_id: waiting_tasks}.

        # strict-only gating derived from original type string
        self.strict_allowed_task_type = self._extract_strict_task_type(
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

    def schedule(self) -> list[str]:
        if TaskPool.is_empty():
            logger.debug("TaskPool is empty, returning empty task list.")
            return []

        if not self.free_sgroups:
            logger.debug("No available scheduler group, returning empty task list.")
            return []

        self.scheduling_ts = time.perf_counter_ns()
        # collect ready task ids
        task_ids = list(
            filter(
                lambda x: not TaskPool.pool[x].waiting
                and not TaskPool.pool[x].need_remove(),
                TaskPool.id_list,
            )
        )

        # enforce strict-only gating if enabled
        if getattr(self, "strict_allowed_task_type", None) is not None:
            task_ids = [
                tid
                for tid in task_ids
                if TaskPool.pool[tid].task_type == self.strict_allowed_task_type
            ]
            if len(task_ids) == 0:
                logger.debug(
                    "Strict-only gating active and no allowed tasks available."
                )
                return []
        if len(task_ids) == 0:
            logger.debug("All tasks are waiting, returning empty task list.")
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
        num_need_blocks = 0
        num_additional_blocks_task_need = (
            Backend.cache_manager.num_additional_blocks_req_need
        )
        num_used_blocks = Backend.cache_manager.num_used_blocks
        num_tasks = 0
        for task_id in prefill_task_ids:
            task = TaskPool.pool[task_id]
            target_seq_len = task.consumed_req_tokens + len(task.next_req_tokens())
            num_need_blocks += num_additional_blocks_task_need(
                task.req.request_id, target_seq_len
            )
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
        for i in range(len(prefill_task_ids)):
            task = TaskPool.pool[prefill_task_ids[i]]
            task_remaining_tokens = task.prefix_tokens_len - task.consumed_req_tokens
            task_prefill_chunk_size = min(
                task_remaining_tokens, self.prefill_chunk_size - prefill_tokens
            )
            task.set_prefill_chunk_size_for_one_step(task_prefill_chunk_size)
            prefill_tokens += task_prefill_chunk_size
            if prefill_tokens >= self.prefill_chunk_size:
                return i + 1
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

        # Check KVCacheManager's capacity
        num_additional_blocks_task_need = (
            Backend.cache_manager.num_additional_blocks_req_need
        )

        def has_enough_block():
            num_need_blocks = 0
            for task_id in decode_task_ids:
                task = TaskPool.pool[task_id]
                if task.has_model_run():
                    num_need_blocks += num_additional_blocks_task_need(
                        task.req.request_id, task.prefix_tokens_len
                    )
            num_free_blocks = Backend.cache_manager.num_free_blocks
            if num_free_blocks >= num_need_blocks:
                return True
            logger.debug(
                f"Cache manager has no more free blocks to support current decoding tasks: need {num_need_blocks} free blocks, cache manager has {num_free_blocks} free blocks"
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
                f"KV cache capacity reached limit, forcing eviction of {len(evicted_tasks)} decode tasks, this may impact throughput and latency. To prevent performance degradation, consider increasing max_reqs or num_blocks."
            )

        return decode_task_ids

    def evict_decode_task(self, task_id: str):
        """Evicting kv cache in kv_cache manager of the given task_id, restore task state to its pre-prefilling state
        Args:
            task_id: the task_id that need to be evicted
        """
        task = TaskPool.pool[task_id]

        # Remove kvcache of this task
        task.next_token = -1
        task.waiting = False
        task.handle = None
        tasks = PackedTasksBase(
            num_tasks=1,
            task_ids=[task_id],
            req_ids=[task.req.request_id],
            task_type=TaskType.Special,
            payload_type=SerializedPackedTasksPayloadType.EndTask,
        )
        Backend.executor.step(tasks)
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

        # For congestion control
        self.kvcache_block_threshold = max(1, self.kvcache_block_threshold // 2)

    @staticmethod
    def _extract_strict_task_type(scheduler_type: str):
        """Return TaskType when strict-only is requested, otherwise None.

        Recognized tokens:
        - "prefill_only" => TaskType.Prefill
        - "decode_only"  => TaskType.Decode
        If both appear, no strict gating will be applied.
        """
        if not scheduler_type:
            return None
        parts = [p.strip().lower() for p in scheduler_type.split(",") if p.strip()]
        has_prefill_only = any(p == "prefill_only" for p in parts)
        has_decode_only = any(p == "decode_only" for p in parts)
        if has_prefill_only and not has_decode_only:
            return TaskType.Prefill
        if has_decode_only and not has_prefill_only:
            return TaskType.Decode
        return None

    def reorder_tasks_for_batching(self, task_ids):
        args = get_global_args()
        if args.infer.cache_type == "skew":
            for task_id in task_ids:
                if TaskPool.pool[task_id].need_remove():
                    if TaskPool.pool[task_id].task_type == TaskType.Decode:
                        remove_index = TaskPool.id_list.index(task_id)
                        for decode_id in reversed(TaskPool.id_list):
                            if (
                                TaskPool.pool[decode_id].task_type == TaskType.Decode
                                and decode_id != task_id
                            ):
                                decode_index = TaskPool.id_list.index(decode_id)
                                (
                                    TaskPool.id_list[remove_index],
                                    TaskPool.id_list[decode_index],
                                ) = (
                                    TaskPool.id_list[decode_index],
                                    TaskPool.id_list[remove_index],
                                )
                                break

    def update(self, cur_task_ids: list[str], unwait_task_ids: list[str] = []):
        removed_task_ids = []
        task_ids = cur_task_ids + unwait_task_ids
        task_ids = list(set(task_ids))
        self.reorder_tasks_for_batching(task_ids)
        if not isinstance(self, DPFifoScheduler):
            for task_id in task_ids:
                # Update Task's sched_group_id and  sgroup_waiting_cnt
                if (
                    not TaskPool.pool[task_id].waiting
                    and TaskPool.pool[task_id].sched_group_id is not None
                ):
                    sgroup_id = TaskPool.pool[task_id].sched_group_id
                    self.sgroup_waiting_tasks[sgroup_id].remove(task_id)
                    # TaskPool.pool[task_id].sched_group_id = None
        for task_id in task_ids:
            task = TaskPool.pool[task_id]
            if task.need_remove():
                if task.task_type == TaskType.Decode:
                    removed_task_ids.append(task_id)
                    num_total_blocks = Backend.cache_manager.get_num_blocks()
                    self.kvcache_block_threshold = num_total_blocks
                    logger.debug(
                        f"Task({task_id}) finished decoding, increasing kvcache_block_threshold to {self.kvcache_block_threshold}, while the number of total blocks is {num_total_blocks}"
                    )
                TaskPool.remove(task_id)

        if removed_task_ids:
            logger.debug(f"[scheduler.update] removed_decode_tasks={removed_task_ids}")

        # Update used_sgroups and free_sgroups according to sgroup status
        if not isinstance(self, DPFifoScheduler):
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

        return removed_task_ids

    def is_done(self):
        return len(TaskPool.pool) == 0


class SkewScheduler(Scheduler):

    def __init__(
        self,
        max_reqs: int,
        original_scheduler_type: str,
        prefill_chunk_size: Optional[int] = None,
    ):
        super().__init__(
            ceil_div(max_reqs, get_slot_handle().num_slots),  # prefill_num_tasks
            ceil_div(max_reqs, get_slot_handle().num_slots),  # decode_num_tasks
            Scheduler._normalize_scheduler_type(original_scheduler_type),
            num_scheduler_groups=get_slot_handle().num_slots,
            original_scheduler_type=original_scheduler_type,
            prefill_chunk_size=prefill_chunk_size,
        )
        self.slot_handle = get_slot_handle()
        self.sgroup_list = [[] for _ in range(self.slot_handle.num_slots)]
        self.sgroup_waiting_tasks = defaultdict(set)  # {slot_group_id: waiting_tasks}
        self.free_sgroups = deque(
            range(self.slot_handle.num_slots)
        )  # free slot_group doesn't have any waiting tasks
        self.used_sgroups = set()  # used slot_group has one or more waiting task.

    @override
    def schedule(self) -> list[str]:

        # no available slot group
        if not self.free_sgroups:
            logger.debug("No available slot group, returning empty task list.")
            return []

        if TaskPool.is_empty():
            logger.debug("TaskPool is empty, returning empty task list.")
            return []

        # collect ready task ids
        task_ids = list(
            filter(
                lambda x: not TaskPool.pool[x].waiting
                and not TaskPool.pool[x].need_remove(),
                TaskPool.id_list,
            )
        )

        # enforce strict-only gating if enabled
        if getattr(self, "strict_allowed_task_type", None) is not None:
            task_ids = [
                tid
                for tid in task_ids
                if TaskPool.pool[tid].task_type == self.strict_allowed_task_type
            ]
            if len(task_ids) == 0:
                logger.debug(
                    "Strict-only gating active and no allowed tasks available."
                )
                return []
        if len(task_ids) == 0:
            logger.debug("All tasks are waiting, returning empty task list.")
            return []

        # Prepare to schedule the earlist released free slot group
        sgroup_id = self.free_sgroups.popleft()
        sgroup_capacity = self.slot_handle.get_slot_size(sgroup_id)
        sgroup = self.sgroup_list[sgroup_id]
        self.used_sgroups.add(sgroup_id)
        self.slot_handle.set_slot_idx(
            sgroup_id
        )  # To inform kvcache the current dealing sgroup_id

        # When slot_group's lenght smaller than it's capacity, fill new tasks into it.
        if len(sgroup) < sgroup_capacity:

            # Only sort tasks outside the slot_group
            task_ids = list(
                filter(
                    lambda x: TaskPool.pool[x].sched_group_id is None,
                    task_ids,
                )
            )
            task_ids.sort(
                key=lambda x: self.scorer(TaskPool.pool[x]),
                reverse=True,  # Largest first
            )  # list.sort is a stable sort

            if sgroup:
                # Tasks already exists in slot_group
                filter_task_type = TaskPool.pool[sgroup[-1]].task_type
            elif task_ids:
                # There are no tasks in slot_group, but are tasks outside the slot_group
                filter_task_type = TaskPool.pool[task_ids[0]].task_type
            else:
                logger.debug(
                    "All tasks are allocated into other slot groups, return empty task list."
                )
                return []

            task_ids = list(
                filter(
                    lambda task_id: TaskPool.pool[task_id].task_type
                    == filter_task_type,
                    task_ids,
                )
            )
            sgroup_remaining_capacity = sgroup_capacity - len(sgroup)
            sgroup.extend(task_ids[:sgroup_remaining_capacity])

        for task_id in sgroup:
            TaskPool.pool[task_id].sched_group_id = sgroup_id

        # When chunk_prefill enable, task type in a slot group may be like: [Decode, ... Decode, Prefill, ... Prefill]
        start_pos = 0
        if (
            self.prefill_chunk_size is not None
            and TaskPool.pool[sgroup[-1]].task_type == TaskType.Prefill
        ):
            start_pos = self.find_prefill_task_start_pos_sgroup(sgroup)
        ret_task_ids = sgroup[start_pos:]

        # Apply chunk prefill
        if (
            TaskPool.pool[sgroup[-1]].task_type == TaskType.Prefill
            and self.prefill_chunk_size is not None
        ):
            num_tasks = self._chunk_prefill_tasks_count(ret_task_ids)
            ret_task_ids = ret_task_ids[:num_tasks]

        self.sgroup_waiting_tasks[sgroup_id] = set(ret_task_ids)
        return ret_task_ids

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
        args = get_global_args()
        if args.infer.cache_type == "skew":
            for task_id in task_ids:
                if TaskPool.pool[task_id].need_remove():
                    if TaskPool.pool[task_id].task_type == TaskType.Decode:
                        sgroup = self.sgroup_list[TaskPool.pool[task_id].sched_group_id]
                        index = sgroup.index(task_id)
                        sgroup[index] = sgroup[-1]
                        sgroup.pop()


class DPFifoScheduler(Scheduler):  # used for expert_data_parallel
    def __init__(
        self,
        max_num_tasks: int,
    ):
        # max num tasks per dp instance
        self.max_num_tasks_per_dp = max_num_tasks
        self.dp_size = get_dp_group().group_size
        self.pp_size = get_pp_group().group_size
        self.have_task = None
        self.kvcache_block_threshold = 0
        self.is_warmup_stage = False
        self._rr_owner_cursor = 0

    def schedule(self) -> list[list[str]]:
        self.have_task = False
        # entry diagnostics
        num_prefill = sum(
            1
            for _id in TaskPool.pool
            if TaskPool.pool[_id].task_type == TaskType.Prefill
            and not TaskPool.pool[_id].waiting
        )
        num_decode = sum(
            1
            for _id in TaskPool.pool
            if TaskPool.pool[_id].task_type == TaskType.Decode
            and not TaskPool.pool[_id].waiting
            and not TaskPool.pool[_id].need_remove()
        )
        logger.debug(
            f"[dpfifo.enter] prefill_ready={num_prefill} decode_ready={num_decode} pool_size={len(TaskPool.pool)}"
        )
        prefill_task_ids = filter(
            lambda x: TaskPool.pool[x].task_type == TaskType.Prefill
            and not TaskPool.pool[x].waiting,
            TaskPool.pool.keys(),
        )
        prefill_task_ids = sorted(
            prefill_task_ids,
            key=lambda x: TaskPool.pool[x].req.start_time,
            reverse=False,
        )
        prefill_task_ids = list(prefill_task_ids)

        # During warmup prefill, select each task at most once per warmup epoch to match expected scheduling count
        # However, if a task was selected but didn't complete prefill (due to budget constraints),
        # it must be allowed to be scheduled again
        if self.is_warmup_stage:
            limited_ids = []
            for tid in prefill_task_ids:
                task = TaskPool.pool[tid]
                # Allow task if: 1) never seen, OR 2) seen but not finished prefill
                is_seen = getattr(task, "_warmup_prefill_seen", False)
                has_remaining = task.consumed_req_tokens < task.prefix_tokens_len

                if not is_seen or has_remaining:
                    limited_ids.append(tid)
                    if not is_seen:
                        task._warmup_prefill_seen = True
                if len(limited_ids) >= self.max_num_tasks_per_dp * self.dp_size:
                    break
            prefill_task_ids = limited_ids
        else:
            prefill_task_ids = prefill_task_ids[
                : self.max_num_tasks_per_dp * self.dp_size
            ]

        if len(prefill_task_ids) > 0:
            # 计算各 DP rank 当前占用（持有 KV 槽的请求），并据此得到剩余可分配槽位
            occupied = [0 for _ in range(self.dp_size)]
            for tid, t in TaskPool.pool.items():
                owner = getattr(t, "cache_owner", None)
                if (
                    owner is not None
                    and 0 <= owner < self.dp_size
                    and not t.need_remove()
                ):
                    occupied[owner] += 1
            free_slots = [
                max(0, self.max_num_tasks_per_dp - occupied[r])
                for r in range(self.dp_size)
            ]

            def next_rank_with_free_slot():
                # 选择仍有空闲槽位的 rank；若无可用，返回 None
                for _ in range(self.dp_size):
                    r = self._rr_owner_cursor
                    self._rr_owner_cursor = (self._rr_owner_cursor + 1) % self.dp_size
                    if free_slots[r] > 0:
                        return r
                return None

            task_lists = [[] for _ in range(self.dp_size)]
            for task_id in prefill_task_ids:
                task = TaskPool.pool[task_id]
                owner = getattr(task, "cache_owner", None)
                if owner is None:
                    # 仅当该 rank 仍有空闲槽位时才为“首次出现”的请求分配 owner
                    r = next_rank_with_free_slot()
                    if r is None:
                        # 所有 rank 均已满，跳过本轮对该新请求的调度，等待后续有槽释放
                        continue
                    task.cache_owner = r
                    free_slots[r] -= 1
                    task_lists[r].append(task_id)
                else:
                    # 已有 owner 的请求不会新增占用，允许继续在原 rank 调度
                    if 0 <= owner < self.dp_size:
                        task_lists[owner].append(task_id)

            # per-rank chunk prefill slicing if enabled
            prefill_chunk_size = get_global_args().infer.prefill_chunk_size
            if prefill_chunk_size is not None and prefill_chunk_size > 0:
                base = prefill_chunk_size // self.dp_size
                rem = prefill_chunk_size % self.dp_size
                logger.debug(
                    f"[dpfifo.prefill] dp_size={self.dp_size} chunk={prefill_chunk_size} base={base} rem={rem} prefill_candidates={[len(x) for x in task_lists]}"
                )
                for r in range(self.dp_size):
                    budget = base + (rem if r == 0 else 0)
                    if budget <= 0:
                        task_lists[r] = []
                        continue
                    assigned = 0
                    new_list = []
                    for tid in task_lists[r]:
                        task = TaskPool.pool[tid]
                        remaining = task.prefix_tokens_len - task.consumed_req_tokens
                        if remaining <= 0:
                            continue
                        take = min(remaining, max(0, budget - assigned))
                        if take <= 0:
                            break
                        task.set_prefill_chunk_size_for_one_step(take)
                        new_list.append(tid)
                        assigned += take
                        if assigned >= budget:
                            break

                    logger.debug(
                        f"[dpfifo.prefill] rank={r} budget={budget} assigned={assigned} selected={len(new_list)} ids={new_list}"
                    )
                    task_lists[r] = new_list

            # make sure tasks do not exceed max_num_tasks_per_dp
            for i in range(self.dp_size):
                if len(task_lists[i]) > self.max_num_tasks_per_dp:
                    task_lists[i] = task_lists[i][: self.max_num_tasks_per_dp]
            self.have_task = any(len(lst) > 0 for lst in task_lists)
            logger.debug(
                f"[dpfifo.prefill] selected per-rank sizes={[len(x) for x in task_lists]} have_task={self.have_task}"
            )

            # Fallback: prefill candidates存在，但本轮chunk切完后为空，尝试直接调度 decode 任务，避免空步导致卡住
            # But during warmup stage, we should NOT schedule decode during prefill phase
            if not self.have_task and not self.is_warmup_stage:
                decode_task_ids = [
                    tid
                    for tid in TaskPool.pool.keys()
                    if TaskPool.pool[tid].task_type == TaskType.Decode
                    and not TaskPool.pool[tid].waiting
                    and not TaskPool.pool[tid].need_remove()
                ]
                task_lists = [[] for _ in range(self.dp_size)]
                for tid in decode_task_ids:
                    task = TaskPool.pool[tid]
                    owner = getattr(task, "cache_owner", 0) % self.dp_size
                    task_lists[owner].append(tid)

                # 截断到每 rank 上限
                for i in range(self.dp_size):
                    if len(task_lists[i]) > self.max_num_tasks_per_dp:
                        task_lists[i] = task_lists[i][: self.max_num_tasks_per_dp]

                self.have_task = any(len(lst) > 0 for lst in task_lists)

                logger.debug(
                    f"[dpfifo.fallback_decode] decode_per_rank={[len(x) for x in task_lists]} have_task={self.have_task}"
                )
        else:
            # no prefill tasks - schedule decode tasks
            # During warmup stage, skip decode scheduling during prefill phase
            if self.is_warmup_stage:
                logger.debug(f"[dpfifo.decode] skip decode during warmup prefill phase")
                task_lists = [[] for _ in range(self.dp_size)]
            else:
                decode_task_ids = filter(
                    lambda x: TaskPool.pool[x].task_type == TaskType.Decode
                    and not TaskPool.pool[x].waiting
                    and not TaskPool.pool[x].need_remove(),
                    TaskPool.pool.keys(),
                )

                decode_task_ids = list(decode_task_ids)
                logger.debug(
                    f"[dpfifo.decode] decode_candidates={len(decode_task_ids)}"
                )

                # For decode tasks, we need to make sure they are sent to their cache owner
                task_lists = [[] for _ in range(self.dp_size)]
                if len(decode_task_ids) > 0:
                    self.have_task = True

                for task_id in decode_task_ids:
                    task = TaskPool.pool[task_id]
                    task_lists[task.cache_owner].append(task_id)

                # make sure tasks do not exceed max_num_tasks_per_dp
                for i in range(self.dp_size):
                    if len(task_lists[i]) > self.max_num_tasks_per_dp:
                        task_lists[i] = task_lists[i][: self.max_num_tasks_per_dp]

        if self.have_task:
            DPTaskCollector.prepare_dp_tasks(task_lists)
            logger.debug(
                f"[dpfifo.return] have_task=True per-rank={[len(x) for x in task_lists]}"
            )
            return task_lists[0]
        else:
            logger.debug("[dpfifo.return] have_task=False")
            return []
