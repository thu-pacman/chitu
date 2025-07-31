import time
from logging import getLogger
from typing import Iterable, List  # Please keep Python 3.8 compatible
from typing_extensions import override

from chitu.task import TaskPool, TaskType
from chitu.global_vars import get_slot_handle, get_global_args
from chitu.utils import ceil_div
from chitu.distributed.parallel_state import get_dp_group

from chitu.backend import Backend

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
                decode_num_tasks = infer_args.max_reqs
            else:
                decode_num_tasks = args.pp_config.decode_num_tasks
        else:
            prefill_num_tasks = infer_args.max_reqs
            decode_num_tasks = infer_args.max_reqs

        return Scheduler(prefill_num_tasks, decode_num_tasks, args.type.lower())

    def __init__(
        self,
        prefill_num_tasks: int,
        decode_num_tasks: int,
        scheduler_type: str,
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

        # determine scoring method
        self.scorers = []
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

    def scorer(self, task):
        return tuple(fn(task) for fn in self.scorers)

    def schedule(self) -> List[str]:
        if TaskPool.is_empty():
            logger.debug("TaskPool is empty, returning empty task list.")
            return []

        self.scheduling_ts = time.perf_counter_ns()
        task_ids = list(
            filter(lambda x: not TaskPool.pool[x].waiting, TaskPool.id_list)
        )
        if len(task_ids) == 0:
            logger.debug("All tasks are waiting, returning empty task list.")
            return []

        task_ids.sort(
            key=lambda x: self.scorer(TaskPool.pool[x]),
            reverse=True,  # Largest first
        )  # list.sort is a stable sort
        filter_task_type = TaskPool.pool[task_ids[0]].task_type
        task_ids = list(
            filter(
                lambda task_id: TaskPool.pool[task_id].task_type == filter_task_type,
                task_ids,
            )
        )
        if filter_task_type == TaskType.Prefill:
            task_ids = task_ids[: self.prefill_num_tasks]
        elif filter_task_type == TaskType.Decode:
            task_ids = task_ids[: self.decode_num_tasks]
        else:
            raise NotImplementedError(f"Unexpected task type: {filter_task_type}")

        # postprocess
        for task_id in task_ids:
            TaskPool.pool[task_id].sched_ts = self.scheduling_ts

        logger.debug(f"Selected task_ids: {task_ids}")
        return task_ids

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

    def update(self, cur_task_ids: List[str], unwait_task_ids: List[str] = []):
        removed_task_ids = []
        task_ids = cur_task_ids + unwait_task_ids
        task_ids = list(set(task_ids))
        self.reorder_tasks_for_batching(task_ids)
        for task_id in task_ids:
            if TaskPool.pool[task_id].need_remove():
                if TaskPool.pool[task_id].task_type == TaskType.Decode:
                    removed_task_ids.append(task_id)
                TaskPool.remove(task_id)
        return removed_task_ids

    def is_done(self):
        return len(TaskPool.pool) == 0


class SkewPipelineScheduler(Scheduler):

    def __init__(self, max_reqs: int):
        super().__init__(max_reqs, max_reqs, "prefill_first")
        self.max_reqs = max_reqs
        self.slot_handle = get_slot_handle()
        self.decode_slots = [[] for _ in range(self.slot_handle.num_slots)]
        self.slot_id = 0

    @override
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
        assert num_tasks <= self.max_reqs
        ret_task_ids = list(prefill_task_ids)[:num_tasks]

        if num_tasks:
            self.decode_slots[local_idx].extend(ret_task_ids)

        decode_task_ids = []
        if len(ret_task_ids) == 0:
            for idx, slot_ids in enumerate(self.decode_slots):
                if len(slot_ids) > 0 and not TaskPool.pool[slot_ids[0]].waiting:
                    local_idx = idx
                    decode_task_ids = slot_ids
                    break
            ret_task_ids.extend(
                list(decode_task_ids)[: self.max_reqs - len(ret_task_ids)]
            )

        if len(ret_task_ids):
            self.slot_handle.set_slot_idx(local_idx)

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

    def schedule(self) -> List[List[str]]:
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
            Backend.task_id_list = task_lists
            Backend.all_task_ids = [
                task_id for task_ids in task_lists for task_id in task_ids
            ]
            return task_lists
        else:
            return []
