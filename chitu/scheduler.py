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
from chitu.distributed.parallel_state import get_dp_group
from chitu.backend import Backend
from chitu.task import (
    PackedTasksBase,
    SerializedPackedTasksPayloadType,
)

logger = getLogger(__name__)


class Scheduler:
    @staticmethod
    def build(args, infer_args):
        if get_dp_group().group_size > 1:
            return DPFifoScheduler(infer_args.max_reqs)

        if get_slot_handle():
            return SkewPipelineScheduler(infer_args.max_reqs)

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
            - "deadline": Each task has a deadline time `DDL = request_arrival_time + prefix_length * alpha +
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
        self.sgroup_waiting_cnt = defaultdict(
            int
        )  # {sched_group_id: waiting_tasks_count}.

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
                self.scorers.append(lambda task: -task.prefix_length)
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
            filter(lambda x: not TaskPool.pool[x].waiting, TaskPool.id_list)
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

        sgroup_id = self.free_sgroups.popleft()
        self.used_sgroups.add(sgroup_id)
        self.sgroup_waiting_cnt[sgroup_id] = len(task_ids)

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
        num_tasks = 0
        block_size = Backend.cache_manager.get_block_size()
        num_used_block = Backend.cache_manager.num_used_blocks
        num_needed_block = 0
        num_total_tokens = 0

        def has_enough_block(num_tasks):
            nonlocal num_needed_block, num_total_tokens
            if num_tasks >= len(prefill_task_ids):
                return False
            task = TaskPool.pool[prefill_task_ids[num_tasks]]
            if task.consumed_req_tokens == 0:
                """Only tasks that are not allocated kv blocks before need new blocks"""
                prefix_token_len = task.prefix_tokens_len
                num_total_tokens += prefix_token_len
                num_needed_block += ceil_div(prefix_token_len, block_size)

            if (
                num_total_tokens
                > get_global_args().infer.max_seq_len * self.prefill_num_tasks
            ):
                return False
            return num_needed_block + num_used_block <= self.kvcache_block_threshold

        while has_enough_block(num_tasks):
            num_tasks += 1

        if (
            num_tasks == 0
            and self.kvcache_block_threshold == Backend.cache_manager.get_num_blocks()
            and num_used_block == 0
        ):
            prefix_len = TaskPool.pool[prefill_task_ids[0]].prefix_tokens_len
            raise RuntimeError(
                f"KV_cache capacity is insufficient to support prefilling (batch_size=1, prefix_len={prefix_len})"
            )

        if self.prefill_chunk_size is not None:
            prefill_tokens = 0
            for i in range(len(prefill_task_ids)):
                task = TaskPool.pool[prefill_task_ids[i]]
                task_remaining_tokens = (
                    task.prefix_tokens_len - task.consumed_req_tokens
                )
                task_prefill_chunk_size = min(
                    task_remaining_tokens, self.prefill_chunk_size - prefill_tokens
                )
                task.set_prefill_chunk_size_for_one_step(task_prefill_chunk_size)
                prefill_tokens += task_prefill_chunk_size
                if prefill_tokens >= self.prefill_chunk_size:
                    prefill_task_ids = prefill_task_ids[: i + 1]
                    break

        return prefill_task_ids[:num_tasks]

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
        task_needs_new_block = Backend.cache_manager.is_block_full_for_req

        def has_enough_block():
            num_need_blocks = sum(
                1
                for task_id in decode_task_ids
                if task_needs_new_block(TaskPool.pool[task_id].req.request_id)
            )
            if num_need_blocks > self.decode_num_tasks:
                return False
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
        task.next_token = -1
        task.waiting = False
        task.handle = None
        tasks = PackedTasksBase(
            num_tasks=1,
            task_ids=[task_id],
            req_ids=[task.req.request_id],
            task_type=TaskType.Decode,
            payload_type=SerializedPackedTasksPayloadType.EndTask,
        )
        Backend.executor.step(tasks)
        task.task_type = TaskType.Prefill
        self.kvcache_block_threshold = max(1, self.kvcache_block_threshold // 2)
        logger.debug(
            f"Temporarily evicting task({task_id}), reducing kvcache_block_threshold to {self.kvcache_block_threshold}, while the number of total blocks is {Backend.cache_manager.get_num_blocks()}"
        )

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
        for task_id in task_ids:

            # Update scheduler group status.
            if (
                not TaskPool.pool[task_id].waiting
                and TaskPool.pool[task_id].sched_group_id is not None
            ):
                sgroup_id = TaskPool.pool[task_id].sched_group_id
                self.sgroup_waiting_cnt[sgroup_id] -= 1
                TaskPool.pool[task_id].sched_group_id = None
                if self.sgroup_waiting_cnt[sgroup_id] == 0:
                    self.used_sgroups.remove(sgroup_id)
                    self.free_sgroups.append(sgroup_id)

            if TaskPool.pool[task_id].need_remove():
                if TaskPool.pool[task_id].task_type == TaskType.Decode:
                    removed_task_ids.append(task_id)
                    num_total_blocks = Backend.cache_manager.get_num_blocks()
                    self.kvcache_block_threshold = min(
                        num_total_blocks, self.kvcache_block_threshold * 2
                    )
                    logger.debug(
                        f"Task({task_id}) finished decoding, increasing kvcache_block_threshold to {self.kvcache_block_threshold}, while the number of total blocks is {num_total_blocks}"
                    )
                TaskPool.remove(task_id)

        return removed_task_ids

    def is_done(self):
        return len(TaskPool.pool) == 0


class SkewPipelineScheduler(Scheduler):

    def __init__(self, max_reqs: int):
        super().__init__(
            max_reqs, max_reqs, "prefill_first", get_global_args().infer.pp_size
        )
        self.max_reqs = max_reqs
        self.slot_handle = get_slot_handle()
        self.decode_slots = [[] for _ in range(self.slot_handle.num_slots)]
        self.slot_id = 0

    @override
    def schedule(self) -> list[str]:
        # search unwaiting prefill tasks
        prefill_task_ids = list(
            filter(
                lambda x: TaskPool.pool[x].task_type == TaskType.Prefill
                and not TaskPool.pool[x].waiting,
                TaskPool.id_list,
            )
        )

        # find slot_group with one or more empty slots
        local_idx = -1
        num_tasks = 0
        for idx, slots in enumerate(self.decode_slots):
            slot_group_capacity = self.slot_handle.get_slot_size(idx)
            if len(slots) < slot_group_capacity:
                local_idx = idx
                num_tasks = slot_group_capacity - len(slots)
                break

        ret_task_ids = []
        if local_idx != -1 and num_tasks > 0 and prefill_task_ids:
            # add prefill tasks into selected slot_group: local_idx
            ret_task_ids = prefill_task_ids[:num_tasks]
            self.decode_slots[local_idx].extend(ret_task_ids)
            self.slot_handle.set_slot_idx(local_idx)

        # totally separate prefilling and decoding stages
        if not ret_task_ids:
            for idx, slot_ids in enumerate(self.decode_slots):
                if slot_ids and all(
                    not TaskPool.pool[slot_id].waiting for slot_id in slot_ids
                ):
                    decode_task_ids = slot_ids[: self.max_reqs]
                    ret_task_ids.extend(decode_task_ids)
                    self.slot_handle.set_slot_idx(idx)
                    break

        return ret_task_ids

    @override
    def reorder_tasks_for_batching(self, task_ids):
        args = get_global_args()
        if args.infer.cache_type == "skew":
            for task_id in task_ids:
                if TaskPool.pool[task_id].need_remove():
                    if TaskPool.pool[task_id].task_type == TaskType.Decode:
                        for lst in self.decode_slots:
                            if task_id in lst:
                                index = lst.index(task_id)
                                lst[index] = lst[-1]
                                lst.pop()
                                break


class DPFifoScheduler(Scheduler):  # used for expert_data_parallel
    def __init__(
        self,
        max_num_tasks: int,
    ):
        # max num tasks per dp instance
        self.max_num_tasks_per_dp = max_num_tasks
        self.dp_size = get_dp_group().group_size
        self.have_task = None
        self.kvcache_block_threshold = 0

    def schedule(self) -> list[list[str]]:
        self.have_task = False

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
        prefill_task_ids = list(prefill_task_ids)[
            : self.max_num_tasks_per_dp * self.dp_size
        ]

        if len(prefill_task_ids) > 0:
            task_lists = [[] for _ in range(self.dp_size)]
            for i, task_id in enumerate(prefill_task_ids):
                task = TaskPool.pool[task_id]
                task.cache_owner = i % self.dp_size
                task_lists[i % self.dp_size].append(task_id)

            # make sure tasks do not exceed max_num_tasks_per_dp
            for i in range(self.dp_size):
                if len(task_lists[i]) > self.max_num_tasks_per_dp:
                    task_lists[i] = task_lists[i][: self.max_num_tasks_per_dp]
            self.have_task = True
        else:
            # no prefill tasks
            decode_task_ids = filter(
                lambda x: TaskPool.pool[x].task_type == TaskType.Decode
                and not TaskPool.pool[x].waiting,
                TaskPool.pool.keys(),
            )

            decode_task_ids = list(decode_task_ids)

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
            return task_lists
        else:
            return []
