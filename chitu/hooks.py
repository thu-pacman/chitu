# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0


from typing import Protocol, Optional
import logging
import asyncio
import torch
import time

from chitu.global_vars import get_global_args
from chitu.task import (
    Task,
    TaskPool,
    PackedTasks,
    PackedTasksBase,
)
from chitu.serve.event_loop import get_server_event_loop
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chitu.task import PackedTasksBase
from chitu.distributed.parallel_state import get_pp_group
from chitu.distributed.pd_disaggregation.pd_log_utils import (
    pd_trace_enabled,
    pd_verbose_enabled,
)

logger = logging.getLogger(__name__)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chitu.distributed.pd_disaggregation.kv_transfer.kv_manager import KVManager


class KVTransferHook(Protocol):
    """Hook for KV transfer across prefill/decode engines in PD mode.

    Implementations may send KV + first-token metadata after prefill, and/or
    receive KV before decode. Default (NoopKVTransferHook) does nothing.
    """

    def on_prefill_done(
        self, req_ids_output: list[str], logits, tasks: Optional[PackedTasksBase] = None
    ):
        pass

    def before_decode_step(self, req_ids: list[str]):
        pass


class NoopKVTransferHook:
    def on_prefill_done(
        self, req_ids_output: list[str], logits, tasks: Optional[PackedTasksBase] = None
    ):
        return

    def before_decode_step(self, req_ids: list[str]):
        return


class TokenSink(Protocol):
    """Sink for streaming tokens out of backend.

    Default LocalTokenSink writes into the request object. PD deployments may
    override to push tokens to a distributed token router.
    """

    def emit_batch(
        self,
        task_list: list[Task],
        token_list: list[int],
        logprobs_list: Optional[list[list[float]]] = None,
        token_idxs_list: Optional[list[list[int]]] = None,
    ) -> None:
        pass


class LocalTokenSink:
    def emit_batch(
        self,
        task_list: list[Task],
        token_list: list[int],
        logprobs_list: Optional[list[list[float]]] = None,
        token_idxs_list: Optional[list[list[int]]] = None,
        mtp_token_list: Optional[list[list[int]]] = None,
    ) -> None:
        if logprobs_list is None or token_idxs_list is None:
            if mtp_token_list is None:
                for task, token in zip(task_list, token_list):
                    task.req.add_data(token, notify_server=False)
            else:
                for task, token, value_list in zip(
                    task_list, token_list, mtp_token_list
                ):
                    task.req.add_data(value_list + [token], notify_server=False)
        else:
            for task, token, logprobs, token_idxs in zip(
                task_list, token_list, logprobs_list, token_idxs_list
            ):
                task.req.add_data(token, logprobs, token_idxs, notify_server=False)

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
        token_list: list[int],
        logprobs_list: Optional[list[list[float]]] = None,
        token_idxs_list: Optional[list[list[int]]] = None,
    ) -> None:
        return


class MooncakeKVTransferHook:
    """KV transfer hook backed by Mooncake KVManager.

    disaggregation_mode:
      - "prefill": send KV after prefill
      - "decode" : receive KV before decode
    """

    def __init__(self, kv_manager: "KVManager", disaggregation_mode: str):
        self.kv_manager = kv_manager
        self.mode = disaggregation_mode

    def on_prefill_done(
        self, req_ids_output: list[str], logits, tasks: Optional[PackedTasksBase] = None
    ):
        if self.kv_manager is None:
            return
        if self.mode != "prefill":
            return
        if not req_ids_output:
            return
        # Send KV cache and first-token metadata to decode side.
        #
        # In PP>1:
        # - Non-last PP stages do not own the final LM head, so their `logits` is actually
        #   intermediate hidden states (not usable by Decode). They should send KV-only.
        # - Last PP stage sends KV and first-token metadata.
        kv_cache = self.kv_manager.kv_cache
        send_tokens = None
        pp_size = get_global_args().infer.pp_size
        if pp_size > 1:
            if not get_pp_group().is_last_rank:
                send_tokens = None

        should_send_tokens = isinstance(logits, torch.Tensor) and (
            pp_size <= 1 or get_pp_group().is_last_rank
        )
        if pd_verbose_enabled():
            if not should_send_tokens:
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

        if should_send_tokens:
            from chitu.backend import Backend  # local import to avoid cycles

            task_list: Optional[list[Task]] = None
            if isinstance(tasks, PackedTasks) and getattr(tasks, "tasks", None):
                tasks_by_id = {t.task_id: t for t in tasks.tasks}
                if all(rid in tasks_by_id for rid in req_ids_output):
                    task_list = [tasks_by_id[rid] for rid in req_ids_output]
            elif tasks is not None:
                if all(rid in TaskPool.pool for rid in req_ids_output):
                    task_list = [TaskPool.pool[rid] for rid in req_ids_output]

            if task_list:
                packed = PackedTasks([t.task_id for t in task_list], tasks=task_list)
                with torch.inference_mode():
                    send_tokens = Backend.executor.sample(logits, packed).to(
                        dtype=torch.int32
                    )
            else:
                logger.warning(
                    "[KVHook] missing task metadata for sampling; fallback to argmax."
                )
                send_tokens = torch.argmax(logits, dim=-1).to(dtype=torch.int32)
        if pd_trace_enabled():
            logger.debug(
                f"[PD_TRACE][prefill.kv_send] req_ids={req_ids_output} batch={len(req_ids_output)} "
                f"token_shape={list(send_tokens.shape) if isinstance(send_tokens, torch.Tensor) else None} "
                f"cache_type={get_global_args().infer.cache_type}"
            )

        self.kv_manager.send_kv_cache(
            first_tokens=send_tokens,
            request_ids=req_ids_output,
            kv_cache=kv_cache,
        )

        # Record KV send duration (covers enqueue; actual RDMA transfer is async)
        _kv_send_dur = time.monotonic() - _kv_send_start
        from chitu.metrics.prometheus_collector import observe_stage_duration

        observe_stage_duration("prefill", "kv_send", _kv_send_dur)

        from chitu.backend import Backend  # local import to avoid cycles

        if Backend.executor._pd_prefill_only:
            for rid in req_ids_output:
                t = TaskPool.pool.get(rid)
                if t is None:
                    continue
                if t.req is not None and not t.req.finish_reason:
                    t.req.finish_reason = "prefill_only"
                # `decode_status` is a read-only property; update the internal state directly.
                # Also clear `waiting` to ensure need_remove() becomes True immediately.
                t.waiting = False
                t.stopped = True

    def before_decode_step(self, req_ids: list[str]):
        if self.kv_manager is None:
            return
        if self.mode != "decode":
            return
        # Receive KV cache from prefill side and insert into local engine.
        from chitu.backend import Backend  # local import to avoid cycles

        # FIXME: Managers other than "main"
        kv_cache = self.kv_manager.kv_cache
        if len(req_ids) == 0:
            return
        # Short-circuit if KV already present for all requests.
        # NOTE: Must check kv_cache, not just block_table!
        # reserve_blocks_for_transfer() allocates blocks but doesn't set seq_len.
        # Only insert_kv_cache_from_transfer() sets kv_cache after KV data transfer.
        pending: list[str] = []
        for rid in req_ids:
            has_kv = False
            # Check if seq_len is set (indicates KV was fully transferred and inserted)
            if hasattr(kv_cache, "tid_to_cached_len"):
                has_kv = rid in kv_cache.tid_to_cached_len
            elif hasattr(kv_cache, "block_table"):
                has_kv = rid in kv_cache.block_table
            if not has_kv:
                pending.append(rid)
        if not pending:
            return

        # 看到该日志表示：Decode 即将阻塞等待 KV pull succ
        if pd_verbose_enabled():
            logger.debug(f"[KVHook] receiving KV for requests: {pending}")
        if pd_trace_enabled():
            logger.debug(
                f"[PD_TRACE][decode.kv_pull_start] pending={pending} total_req_ids={len(req_ids)} "
                f"rank={Backend.executor.rank} tp={Backend.executor.tp_size} "
                f"dp={Backend.executor.dp_size} ep={Backend.executor.ep_size}"
            )
        prefix_lens = []
        for rid in pending:
            t = TaskPool.pool.get(rid)
            prefix_lens.append(int(t.prefix_tokens_len) if t is not None else 0)
        if pd_trace_enabled():
            logger.debug(
                f"[PD_TRACE][decode.kv_pull_prefix] pending={pending} prefix_lens={prefix_lens}"
            )

        # Ensure the local decode rank knows which prefill engine_rank to talk to.
        # In PD decode-only, the scheduler runs only on dp main rank and sets the binding there.
        # Worker ranks must recover this binding from Task bootstrap metadata.
        for rid in pending:
            task = TaskPool.pool.get(rid)
            if task is None or not hasattr(task, "pd_prefill_engine_rank"):
                continue
            prefill_rank = task.pd_prefill_engine_rank
            if prefill_rank is None:
                continue
            room = self.kv_manager._to_uuid(rid)
            if self.kv_manager.prefill_target_rank_by_room.get(room) is None:
                self.kv_manager.set_prefill_target_engine_rank(rid, prefill_rank)

        first_tokens = self.kv_manager.recv_kv_cache_and_insert(
            request_ids=pending, kv_cache=kv_cache, prefix_lens=prefix_lens
        )
        if pd_trace_enabled():
            logger.debug(
                f"[PD_TRACE][decode.kv_pull_done] pending={pending} token_shape={list(first_tokens.shape)}"
            )
        if len(pending) == 0:
            return
        tokens_cpu = first_tokens.to(dtype=torch.int64, device="cpu").tolist()
        for rid, token in zip(pending, tokens_cpu):
            task = TaskPool.pool.get(rid)
            if task is None:
                continue
            if task.next_token < 0:
                task.update_response_sync(int(token))
                # DP worker rank 的 task 没有被 DPTaskWrapper 替换 update_response_no_sync，
                # 上面的 update_response_sync 只更新了本地状态，不会把 token 发给 Router。
                # 在 task 上标记这个 token，后续 collect_token 会把它带回 rank 0 补发。
                task._pd_first_token_for_dp_emit = int(token)
            if getattr(task, "req", None) is not None and not getattr(
                task, "_pd_first_token_applied", False
            ):
                task._pd_first_token_applied = True
                max_seq_len = get_global_args().infer.max_seq_len
                remaining = max(0, int(max_seq_len) - int(task.prefix_tokens_len))
                task.req.max_new_tokens = min(int(task.req.max_new_tokens), remaining)
        from chitu.metrics.prometheus_collector import PrometheusMetricsCollector

        PrometheusMetricsCollector.inc_generated_tokens(len(tokens_cpu))
