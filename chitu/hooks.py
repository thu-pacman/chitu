# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0


from typing import Protocol, Optional
import logging
import time

from chitu.global_vars import get_global_args
from chitu.task import (
    Task,
    TaskType,
    TaskPool,
    PackedTasks,
    PackedTasksBase,
    DPTaskCollector,
    TaskCollector,
)
from chitu.serve.event_loop import get_server_event_loop
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chitu.task import PackedTasksBase
from chitu.distributed.pd_disaggregation.pd_log_utils import pd_trace_enabled
from chitu.distributed.pd_disaggregation.pd_scheduler import get_pd_scheduler_instance
from chitu.distributed.parallel_state import get_dp_group
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chitu.distributed.pd_disaggregation.kv_transfer import (
        KVManagerPrefill,
        KVManagerDecode,
    )

logger = logging.getLogger(__name__)


class TokenSink(Protocol):
    """Sink for streaming tokens out of backend.

    Default LocalTokenSink writes into the request object. PD deployments may
    override to push tokens to a distributed token router.
    """

    def emit_batch(
        self,
        task_list: list[Task],
        token_list: list[list[int]],
        logprobs_list: Optional[list[list[float]]] = None,
        token_idxs_list: Optional[list[list[int]]] = None,
    ) -> None:
        pass


class LocalTokenSink:
    def emit_batch(
        self,
        task_list: list[Task],
        token_list: list[list[int]],
        logprobs_list: Optional[list[list[float]]] = None,
        token_idxs_list: Optional[list[list[int]]] = None,
    ) -> None:
        if logprobs_list is None or token_idxs_list is None:
            for task, token in zip(task_list, token_list):
                task.req.add_data(token, notify_server=False)
        else:
            for task, token, logprobs, token_idxs in zip(
                task_list, token_list, logprobs_list, token_idxs_list
            ):
                task.req.add_data(token, logprobs, token_idxs, notify_server=False)
        for task in task_list:
            if task.user_request_finished():
                task.req.stop_stream()

        def notify_all_response_in_batch():
            for task in task_list:
                task.req.notify_server_data_added_from_server_thread()

        if (loop := get_server_event_loop()) is not None:
            # No need to notify if there is no server (e.g. offline inference)
            loop.call_soon_threadsafe(notify_all_response_in_batch)


class KVTransferHook(Protocol):
    """Hook for KV transfer across prefill/decode engines in PD mode.

    Implementations may send KV + first-token metadata after prefill, and/or
    receive KV before decode. Default (NoopKVTransferHook) does nothing.
    """

    def on_prefill_done(self, tasks: PackedTasksBase):
        pass

    def before_decode_step(
        self,
        req_ids: list[str],
    ):
        pass


class NoopKVTransferHook:
    def on_prefill_done(self, tasks: PackedTasksBase):
        return

    def before_decode_step(
        self,
        req_ids: list[str],
    ):
        return


class MooncakeKVTransferHook:
    """KV transfer hook backed by Mooncake KVManager.

    disaggregation_mode:
      - "prefill": send KV after prefill
      - "decode" : receive KV before decode
    """

    def __init__(
        self, kv_manager: "KVManagerPrefill|KVManagerDecode", disaggregation_mode: str
    ):
        self.kv_manager = kv_manager
        self.mode = disaggregation_mode

    def on_prefill_done(self, tasks: PackedTasksBase):
        if self.kv_manager is None:
            return
        if self.mode != "prefill":
            return
        if tasks.num_tasks == 0 and not DPTaskCollector.available():
            return

        first_tokens = None
        if isinstance(tasks, PackedTasks) and tasks.generated_result is not None:
            first_tokens = tasks.generated_result.tokens.flatten()

        # Send KV cache and first-token metadata to decode side.
        if tasks.num_tasks > 0:
            req_ids_output = tasks.output_task_ids
            request_cached_tokens = {}
            for t in getattr(tasks, "output_tasks", []):
                if t is None or getattr(t, "req", None) is None:
                    continue
                request_cached_tokens[str(t.req.request_id)] = int(t.req.num_hit_tokens)
            num_hit_tokens = [
                request_cached_tokens.get(rid, 0) for rid in req_ids_output
            ]
            if first_tokens is None:
                # 看到该日志表示：该 rank 只传输 KV Cache（不包含首 token）
                logger.debug(f"[KVHook] sending KV-only for requests: {req_ids_output}")
            else:
                # 看到该日志表示：该 rank 传输 KV Cache + first-token
                logger.debug(
                    f"[KVHook] sending KV+token for requests: {req_ids_output}"
                )

            _kv_send_start = time.monotonic()
            for rid in req_ids_output:
                logger.debug(f"[PD_STAGE][prefill.kv_send.start] req_id={rid}")

            if pd_trace_enabled():
                logger.debug(
                    f"[PD_TRACE][prefill.kv_send] req_ids={req_ids_output} batch={len(req_ids_output)} "
                    f"cache_type={get_global_args().infer.cache_type}"
                )

            self.kv_manager.send_kv_cache(
                first_tokens=first_tokens,
                request_ids=req_ids_output,
                num_hit_tokens=num_hit_tokens,
            )

            # Record KV send duration (covers enqueue; actual RDMA transfer is async)
            _kv_send_dur = time.monotonic() - _kv_send_start
            from chitu.metrics.prometheus_collector import observe_stage_duration

            observe_stage_duration("prefill", "kv_send", _kv_send_dur)

        from chitu.backend import Backend  # local import to avoid cycles

        if DPTaskCollector.available():
            tasks = DPTaskCollector.get_total_packedtasks()
        if Backend.executor._pd_prefill_only:
            output_tasks = getattr(tasks, "output_tasks", None)
            if output_tasks is None:
                output_tasks = [
                    TaskPool.pool.get(task_id) for task_id in tasks.output_task_ids
                ]
            stopped_task_ids = []
            for task in output_tasks:
                if task is None:
                    continue
                if task.req is not None and not task.req.finish_reason:
                    task.req.finish_reason = "prefill_only"
                # Stop task so it won't start decode, and enqueue it for the
                # normal Scheduler.update() cleanup path. Async PP can detach
                # result readiness from the current collector slot, so relying
                # only on executor-side ready_tasks can leave stopped prefill
                # tasks in TaskPool and block graceful termination.
                task.set_stopped()
                stopped_task_ids.append(task.task_id)
            if stopped_task_ids:
                TaskCollector.add_update_task_ids(stopped_task_ids)

    def before_decode_step(
        self,
        req_ids: list[str],
    ):
        if self.kv_manager is None:
            return
        if self.mode != "decode":
            return
        # Receive KV cache from prefill side and insert into local engine.
        from chitu.backend import Backend  # local import to avoid cycles

        if len(req_ids) == 0:
            return
        # Short-circuit if KV already present for all requests.
        pending: list[str] = []
        for rid in req_ids:
            if self.kv_manager._info(rid, create=False) is not None:
                pending.append(rid)
        if not pending:
            return

        for req_id in pending:
            first_token, num_hit_tokens = self.kv_manager.recv_kv_cache_and_insert(
                req_id
            )

            task = TaskPool.pool.get(req_id)
            if task is None:
                continue
            task.update_response_sync([first_token])

        if get_dp_group().group_id == 0:
            from chitu.metrics.prometheus_collector import PrometheusMetricsCollector

            PrometheusMetricsCollector.inc_generated_tokens(len(pending))


class TaskEvictHook(Protocol):
    """Hook for handling task evicting event in Scheduler

    Default hook does nothing
    """

    def get_prefill_task_ids(self) -> list[str]:
        return []

    def check_evict(self, task_id: str) -> bool:
        return True

    def on_evict_done(self, task: Task):
        pass


class NoopTaskEvictHook:
    def get_prefill_task_ids(self) -> list[str]:
        return []

    def check_evict(self, task_id: str) -> bool:
        return TaskPool.pool.get(task_id) is not None

    def on_evict_done(self, task: Task):
        # Restore the task's status to before prefill
        task.task_type = TaskType.Prefill
        task.prefill_chunk_size = None
        task.consumed_req_tokens = 0
        task.sched_group_id = None
        task.dp_rank = None


class PDTaskEvictHook:
    def get_prefill_task_ids(self) -> list[str]:
        # TODO: return pd decode preparing tasks
        return []

    def check_evict(self, task_id: str) -> bool:
        if (
            TaskPool.pool.get(task_id) is not None
            and not TaskPool.pool.get(task_id).is_pd_status()
        ):
            return True
        # evict prefilling tasks on decode rank
        pd_scheduler = get_pd_scheduler_instance()
        pd_scheduler.stop_request(task_id, force_stop=True)
        logger.warning(f"Evicted task {task_id} due to insufficient KV cache")
        return False

    def on_evict_done(self, task: Task):
        # Mark stopped so need_remove() returns True and update() drops it
        task.set_stopped()
        TaskCollector.add_update_task_ids([task.task_id])
        if getattr(task, "req", None) is not None and not task.req.finished:
            task.req.finish_reason = "evicted"
            pd_scheduler = get_pd_scheduler_instance()
            if pd_scheduler is None:
                return
            pd_scheduler.kv_manager.remove_request_all_rank(task.req.request_id)
            token_manager = pd_scheduler.token_manager
            if token_manager is not None:
                token_manager.token_sender.send_evict(
                    task.req.request_id, task.req.num_hit_tokens
                )
