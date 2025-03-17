import time
from logging import getLogger
from typing import List  # Please keep Python 3.8 compatible

import torch

from chitu.global_vars import get_global_args, get_slot_handle
from chitu.task import TaskPool, TaskType

logger = getLogger(__name__)


class Scheduler:
    @staticmethod
    def build(args):
        if args.type.lower() == "fifo" or args.type.lower() == "fcfs":
            return FcfsScheduler(args.fcfs.num_tasks, args.fcfs.enable_hybrid)
        if args.type.lower() == "prefill_first":
            slot_handle = get_slot_handle()
            if slot_handle:
                return SkewPipelineScheduler(
                    args.prefill_first.num_tasks, args.prefill_first.enable_hybrid
                )
            else:
                return PrefillFirstScheduler(
                    args.prefill_first.num_tasks, args.prefill_first.enable_hybrid
                )
        if args.type.lower() == "stride":
            return StrideScheduler(args.stride.num_tasks, args.stride.enable_hybrid)
        if args.type.lower() == "deadline":
            return DdlScheduler(args.deadline.num_tasks, args.deadline.enable_hybrid)
        if args.type.lower() == "prefix_align":
            return PrefixAlignScheduler(
                args.prefix_align.num_tasks, args.prefix_align.enable_hybrid
            )
        if args.type.lower() == "balance":
            return BalanceScheduler(args.balance.num_tasks, args.balance.enable_hybrid)
        else:
            raise NotImplementedError(f"Scheduler {args.type} not implemented")

    def schedule(self) -> List[str]:
        raise NotImplementedError()

    def update(self, cur_task_ids, unwait_task_ids=[]):
        removed_task_ids = []
        task_ids = cur_task_ids + unwait_task_ids
        task_ids = list(set(task_ids))
        for task_id in task_ids:
            if TaskPool.pool[task_id].need_remove():
                if TaskPool.pool[task_id].task_type == TaskType.Decode:
                    removed_task_ids.append(task_id)
                assert TaskPool.remove(
                    task_id
                ), f"Task {task_id} not found in pool {TaskPool.pool.keys()}"
            # assert False
        # for task_id in unwait_task_ids:
        #     if task_id in TaskPool.id_list and TaskPool.pool[task_id].need_remove():
        #         assert TaskPool.remove(task_id), "Task not found in pool"
        return removed_task_ids

    def is_done(self):
        return len(TaskPool.pool) == 0


class FcfsScheduler(Scheduler):
    """
    first come, first service
    note that no arrival_time record, implicitly ordered by TaskPool.add -> list.append
    """

    def __init__(self, num_tasks: int, enable_hybrid: bool):
        super().__init__()
        assert num_tasks > 0, "num_tasks must be greater than 0"
        self.num_tasks = num_tasks
        self.enable_hybrid = enable_hybrid

    def schedule(self) -> List[str]:
        if self.enable_hybrid:
            ret_task_ids = TaskPool.id_list[: self.num_tasks]
        else:
            filter_task_type = (
                TaskPool.pool[TaskPool.id_list[0]].task_type
                if len(TaskPool.id_list) > 0
                else TaskType.Prefill
            )
            filtered_task_ids = filter(
                lambda x: TaskPool.pool[x].task_type == filter_task_type,
                TaskPool.id_list,
            )
            ret_task_ids = list(filtered_task_ids)[: self.num_tasks]
        if filter_task_type == TaskType.Prefill:
            ret_task_ids = ret_task_ids[:1]
        logger.debug(f"Selected task_ids: {self.ret_task_ids}")
        return ret_task_ids


class PrefillFirstScheduler(Scheduler):
    """
    always select prefill tasks, in a fifo manner, if any.
    decode tasks will be selected only if no prefill task
    """

    def __init__(self, num_tasks: int, enable_hybrid: bool):
        super().__init__()
        assert num_tasks > 0, "num_tasks must be greater than 0"
        self.num_tasks = num_tasks
        self.enable_hybrid = enable_hybrid

    def schedule(self) -> List[str]:
        prefill_task_ids = filter(
            lambda x: TaskPool.pool[x].task_type == TaskType.Prefill
            and not TaskPool.pool[x].waiting,
            TaskPool.id_list,
        )
        # select at most num_tasks prefill tasks
        ret_task_ids = list(prefill_task_ids)[: self.num_tasks]
        # if no prefill tasks or enable hybrid, select decode tasks if there is room left
        if len(ret_task_ids) == 0 or (
            self.enable_hybrid and len(ret_task_ids) < self.num_tasks
        ):
            decode_task_ids = filter(
                lambda x: TaskPool.pool[x].task_type == TaskType.Decode
                and not TaskPool.pool[x].waiting,
                TaskPool.id_list,
            )
            ret_task_ids.extend(
                list(decode_task_ids)[: self.num_tasks - len(ret_task_ids)]
            )
        logger.debug(f"Selected task_ids: {ret_task_ids}")

        # if (
        #     len(ret_task_ids) > 0
        #     and TaskPool.pool[ret_task_ids[0]].task_type == TaskType.Prefill
        # ):
        #     ret_task_ids = ret_task_ids[:1]
        return ret_task_ids


class SkewPipelineScheduler(Scheduler):

    def __init__(self, num_tasks: int, enable_hybrid: bool):
        super().__init__()
        assert num_tasks > 0, "num_tasks must be greater than 0"
        self.num_tasks = num_tasks
        self.enable_hybrid = enable_hybrid
        self.slot_handle = get_slot_handle()
        self.decode_slots = [[] for _ in range(self.slot_handle.num_slots)]
        self.slot_id = 0

    def schedule(self) -> List[str]:
        #  search for an empty slot to prefill
        prefill_task_ids = filter(
            lambda x: TaskPool.pool[x].task_type == TaskType.Prefill
            and not TaskPool.pool[x].waiting,
            TaskPool.id_list,
        )

        local_idx = -1
        for idx, slots in enumerate(self.decode_slots):
            if len(slots) == 0:
                local_idx = idx
                break

        num_tasks = 0 if local_idx == -1 else self.slot_handle.get_slot_size(local_idx)
        ret_task_ids = list(prefill_task_ids)[:num_tasks]

        if num_tasks:
            self.decode_slots[local_idx].extend(ret_task_ids)

        decode_task_ids = []
        if len(ret_task_ids) == 0 or (
            self.enable_hybrid and len(ret_task_ids) < self.num_tasks
        ):
            for idx, slot_ids in enumerate(self.decode_slots):
                if len(slot_ids) > 0 and not TaskPool.pool[slot_ids[0]].waiting:
                    local_idx = idx
                    decode_task_ids = slot_ids
                    break
            ret_task_ids.extend(
                list(decode_task_ids)[: self.num_tasks - len(ret_task_ids)]
            )

        if len(ret_task_ids):
            self.slot_handle.set_slot_idx(local_idx)

        return ret_task_ids


class StrideScheduler(Scheduler):
    """
    each task has a priority value P, and a score S (starts from 0),
    at scheduling point, update the scores:
    S += P * elapsed_time
    select the tasks with top scores and reset their scores back to 0.
    """

    def __init__(self, num_tasks: int, enable_hybrid: bool):
        super().__init__()
        assert num_tasks > 0, "num_tasks must be greater than 0"
        self.num_tasks = num_tasks
        self.enable_hybrid = enable_hybrid

    def schedule(self) -> List[str]:
        # update sched_score
        for task_id in TaskPool.id_list:
            task = TaskPool.pool[task_id]
            task.sched_score += task.priority * (time.perf_counter_ns() - task.sched_ts)
            task.sched_ts = time.perf_counter_ns()
        # sort by sched_score and select top tasks
        if self.enable_hybrid:
            ret_task_ids = sorted(
                TaskPool.id_list,
                key=lambda x: TaskPool.pool[x].sched_score,
                reverse=True,
            )[: self.num_tasks]
        else:
            filter_task_type = (
                TaskPool.pool[TaskPool.id_list[0]].task_type
                if len(TaskPool.id_list) > 0
                else TaskType.Prefill
            )
            filtered_task_ids = filter(
                lambda x: TaskPool.pool[x].task_type == filter_task_type,
                TaskPool.id_list,
            )
            ret_task_ids = sorted(
                list(filtered_task_ids),
                key=lambda x: TaskPool.pool[x].sched_score,
                reverse=True,
            )[: self.num_tasks]
        # reset sched_score of selected tasks
        for task_id in ret_task_ids:
            TaskPool.pool[task_id].sched_score = 0
        logger.debug(f"Selected task_ids: {ret_task_ids}")
        return ret_task_ids


class DdlScheduler(Scheduler):
    """
    each task has a deadline time DDL
    DDL = request_arrival_time + prefix_length*alpha + max_output_tokens*beta
    select the tasks with nearest DDL.
    alpha and beta are arbitary value, defaults to 1ms.
    """

    def __init__(self, num_tasks: int, enable_hybrid: bool):
        super().__init__()
        assert num_tasks > 0, "num_tasks must be greater than 0"
        self.num_tasks = num_tasks
        self.enable_hybrid = enable_hybrid

    def schedule(self) -> List[str]:
        # sort by ddl and select top tasks
        if self.enable_hybrid:
            ret_task_ids = sorted(
                TaskPool.id_list, key=lambda x: TaskPool.pool[x].sched_ddl
            )[: self.num_tasks]
        else:
            filter_task_type = (
                TaskPool.pool[TaskPool.id_list[0]].task_type
                if len(TaskPool.id_list) > 0
                else TaskType.Prefill
            )
            filtered_task_ids = filter(
                lambda x: TaskPool.pool[x].task_type == filter_task_type,
                TaskPool.id_list,
            )
            ret_task_ids = sorted(
                list(filtered_task_ids), key=lambda x: TaskPool.pool[x].sched_ddl
            )[: self.num_tasks]
        logger.debug(f"Selected task_ids: {ret_task_ids}")
        return ret_task_ids


class PrefixAlignScheduler(Scheduler):
    """
    each task has a prefix length, for a prefill task, prefix = tokenized prompt,
    for a decode task, prefix = prefill's prefix + generated tokens.
    at scheduling point, try to select as many tasks with close prefix_length as possible.
    E.g., 4 tasks with prefix_length 100 and 2 tasks with prefix_length 300, select the former 4 tasks.
    """

    def __init__(self, num_tasks: int, enable_hybrid: bool):
        super().__init__()
        assert num_tasks > 0, "num_tasks must be greater than 0"
        self.num_tasks = num_tasks
        self.enable_hybrid = enable_hybrid

    def schedule(self) -> List[str]:
        # TODO: no definition on 'close prefix length', sort by prefix length and select the longest
        if self.enable_hybrid:
            ret_task_ids = sorted(
                TaskPool.id_list, key=lambda x: TaskPool.pool[x].prefix_length
            )[: self.num_tasks]
        else:
            filter_task_type = (
                TaskPool.pool[TaskPool.id_list[0]].task_type
                if len(TaskPool.id_list) > 0
                else TaskType.Prefill
            )
            filtered_task_ids = filter(
                lambda x: TaskPool.pool[x].task_type == filter_task_type,
                TaskPool.id_list,
            )
            ret_task_ids = sorted(
                list(filtered_task_ids), key=lambda x: TaskPool.pool[x].prefix_length
            )[: self.num_tasks]
        logger.debug(f"Selected task_ids: {ret_task_ids}")
        return ret_task_ids


class BalanceScheduler(Scheduler):
    """
    an advanced scheduler that pack appropriate prefill tasks and decode tasks to
    acheive the best hardware utilization. since prefill tasks are computation-intensive
    while decode tasks are memory-intensive, there must be a best hybrid solution.
    however, searching for the optimal solution is expensive,
    so a somehow heuristic algorithm will be applied.
    """

    def __init__(self, num_tasks: int, enable_hybrid: bool):
        super().__init__()
        assert num_tasks > 0, "num_tasks must be greater than 0"
        self.num_tasks = num_tasks
        self.enable_hybrid = enable_hybrid

    def schedule(self) -> List[str]:
        prefill_task_ids = list(
            filter(
                lambda x: TaskPool.pool[x].task_type == TaskType.Prefill,
                TaskPool.id_list,
            )
        )
        decode_task_ids = list(
            filter(
                lambda x: TaskPool.pool[x].task_type == TaskType.Decode,
                TaskPool.id_list,
            )
        )
        if self.enable_hybrid:  # TODO: currently half and half
            prefill_count = (
                self.num_tasks // 2
                if len(decode_task_ids) >= self.num_tasks // 2
                else self.num_tasks - len(decode_task_ids)
            )
            decode_count = self.num_tasks - prefill_count
            ret_task_ids = prefill_task_ids[:prefill_count]
            ret_task_ids.extend(decode_task_ids[:decode_count])
        else:  # fall back to prefill_first
            ret_task_ids = (
                prefill_task_ids[: self.num_tasks]
                if len(prefill_task_ids) > 0
                else decode_task_ids[: self.num_tasks]
            )
        logger.debug(f"Selected task_ids: {ret_task_ids}")
        return ret_task_ids
