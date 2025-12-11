# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0


from typing import Protocol, Optional
import logging
import asyncio

from chitu.task import Task
from chitu.serve.event_loop import get_server_event_loop


class KVTransferHook(Protocol):
    """Hook for KV transfer across prefill/decode engines in PD mode.

    Implementations may send KV + first-token logits after prefill, and/or
    receive KV before decode. Default (NoopKVTransferHook) does nothing.
    """

    def on_prefill_done(self, req_ids_output: list[str], logits):
        pass

    def before_decode_step(self, req_ids: list[str]):
        pass


class NoopKVTransferHook:
    def on_prefill_done(self, req_ids_output: list[str], logits):
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

    def __init__(self, kv_manager, disaggregation_mode: str):
        self.kv_manager = kv_manager
        self.mode = disaggregation_mode

    def on_prefill_done(self, req_ids_output: list[str], logits):
        if self.kv_manager is None:
            return
        if self.mode != "prefill":
            return
        # Send KV cache and first-token logits to decode side.
        cache_manager = self.kv_manager.cache_manager
        if logits is None or len(req_ids_output) == 0:
            return
        logging.getLogger(__name__).info(
            f"[KVHook] sending KV+logits for requests: {req_ids_output}"
        )
        self.kv_manager.send_kv_cache(
            logits=logits, request_ids=req_ids_output, cache_manager=cache_manager
        )

    def before_decode_step(self, req_ids: list[str]):
        if self.kv_manager is None:
            return
        if self.mode != "decode":
            return
        # Receive KV cache from prefill side and insert into local engine.
        from chitu.backend import Backend  # local import to avoid cycles

        cache_manager = self.kv_manager.cache_manager or Backend.cache_manager
        if len(req_ids) == 0:
            return
        # Short-circuit if KV already present for all requests
        pending: list[str] = []
        for rid in req_ids:
            has_kv = False
            if hasattr(cache_manager, "get_page_indices"):
                indices = cache_manager.get_page_indices(rid)
                has_kv = bool(indices)
            elif hasattr(cache_manager, "block_table"):
                bt = getattr(cache_manager, "block_table", {})
                has_kv = bool(bt.get(rid, []))
            if not has_kv:
                pending.append(rid)
        if not pending:
            return

        logging.getLogger(__name__).info(
            f"[KVHook] receiving KV for requests: {pending}"
        )
        _ = self.kv_manager.recv_kv_cache_and_insert(
            request_ids=pending, cache_manager=cache_manager
        )
