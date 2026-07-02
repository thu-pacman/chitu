# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0


from typing import Protocol, Optional
import logging
import time

from chitu.global_vars import get_global_args
from chitu.task import (
    Task,
    TaskPool,
    PackedTasks,
    PackedTasksBase,
    DPTaskCollector,
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


class DPTokenSink:
    """No-op sink for PD decode flow.

    Tokens are already streamed by DP Token Manager via task wrapper during
    postprocess_sync_part. Emitting again here would duplicate outputs.
    """

    def emit_batch(
        self,
        task_list: list[Task],
        token_list: list[list[int]],
        logprobs_list: Optional[list[list[float]]] = None,
        token_idxs_list: Optional[list[list[int]]] = None,
    ) -> None:
        return


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
            first_tokens = tasks.generated_result.tokens.flatten().tolist()

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
                    f"first_tokens={first_tokens} "
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
        if Backend.executor._pd_prefill_only and isinstance(tasks, PackedTasks):
            for t in tasks.output_tasks:
                if t is None:
                    continue
                if t.req is not None and not t.req.finish_reason:
                    t.req.finish_reason = "prefill_only"
                # Also clear `waiting` to ensure need_remove() becomes True immediately.
                t.unwait()
                t.set_stopped()

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
            # Worker rank 的 task 没有 req；先把值挂在 task 上，后续由主 rank 汇聚回传。
            task._pd_cached_hit_tokens_for_dp_emit = num_hit_tokens
            if not task.has_next_token():
                task.update_response_sync([first_token])
                # DP worker rank 的 task 没有被 DPTaskWrapper 替换 update_response_sync，
                # 上面的 update_response_sync 只更新了本地状态，不会把 token 发给 Router。
                # 在 task 上标记这个 token，后续 collect_token 会把它带回 rank 0 补发。
                if Backend.executor.rank != 0:
                    task._pd_first_token_for_dp_emit = first_token
            if task.req is not None and not task._pd_first_token_applied:
                task.req.num_hit_tokens = max(task.req.num_hit_tokens, num_hit_tokens)
                task._pd_first_token_applied = True
                max_seq_len = get_global_args().infer.max_seq_len
                remaining = max(0, int(max_seq_len) - int(task.prefix_tokens_len))
                task.req.max_new_tokens = min(int(task.req.max_new_tokens), remaining)

        if get_dp_group().group_id == 0:
            from chitu.metrics.prometheus_collector import PrometheusMetricsCollector

            PrometheusMetricsCollector.inc_generated_tokens(len(pending))


class TaskEvictHook(Protocol):
    """Hook for handling task evicting event in Scheduler

    Default hook does nothing
    """

    def before_evict(self, task: Task):
        pass

    def on_evict_done(self, task: Task):
        pass


class NoopTaskEvictHook:
    def before_evict(self, task: Task):
        pass

    def on_evict_done(self, task: Task):
        pass


class PDTaskEvictHook:
    def before_evict(self, task: Task):
        pass

    def on_evict_done(self, task: Task):
        # Mark stopped so need_remove() returns True and update() drops it
        task.set_stopped()
        if getattr(task, "req", None) is not None and not task.req.finished:
            task.req.finish_reason = "evicted"
            token_manager = get_pd_scheduler_instance().token_manager
            if token_manager is not None:
                token_manager.token_sender.send_finish(
                    task.req.request_id, finish_reason="evicted"
                )
