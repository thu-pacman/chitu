# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
PD disaggregation Scheduler
Extends the base Scheduler to support Prefill-only and Decode-only modes
"""

import asyncio
import time
from enum import Enum
from logging import getLogger
from typing import Optional, Any
import os
import torch

from chitu.scheduler import Scheduler
from chitu.task import TaskType, Task
from chitu.global_vars import get_global_args
from chitu.distributed.pd_disaggregation.kv_transfer.kv_manager import (
    KVManager,
    DisaggregationMode,
)
from chitu.distributed.pd_disaggregation.kv_transfer.mooncake.metadata import (
    MetadataBuffers,
)
from chitu.distributed.pd_disaggregation.pd_types import PDRequestStatus

logger = getLogger(__name__)


class PDSchedulerMode(Enum):
    """PD Scheduler mode"""

    PREFILL_ONLY = "prefill_only"
    DECODE_ONLY = "decode_only"
    UNIFIED = "unified"  # traditional mode


class PDScheduler(Scheduler):
    """
    PD disaggregation Scheduler
    Supports Prefill-only, Decode-only, and traditional unified modes
    """

    def __init__(
        self,
        prefill_num_tasks: int,
        decode_num_tasks: int,
        scheduler_type: str,
        pd_mode: PDSchedulerMode = PDSchedulerMode.UNIFIED,
        scheduler_id: int = 0,
    ):
        # Filter out PD-specific scheduler types before passing to parent
        filtered_scheduler_type = self._filter_scheduler_type(scheduler_type)
        super().__init__(
            prefill_num_tasks,
            decode_num_tasks,
            filtered_scheduler_type,
            get_global_args().infer.pp_size,
        )

        self.pd_mode = pd_mode
        self.scheduler_id = scheduler_id
        self.original_scheduler_type = scheduler_type

        # PD disaggregation related state
        self.pending_decode_requests: dict[str, dict] = {}  # request_id -> request_info
        self.kv_manager: Optional[KVManager] = None
        self.metadata_buffers: Optional[MetadataBuffers] = None
        self.token_manager = None  # DP token manager for streaming back to Router

        # Initialize PD components
        self._init_pd_components()

        logger.info(f"initialized pd scheduler in {pd_mode.value} mode")

    def _filter_scheduler_type(self, scheduler_type: str) -> str:
        """Filter out PD-specific scheduler types"""
        # Remove PD-specific types and map to valid base scheduler types
        parts = scheduler_type.split(",")
        filtered_parts = []

        for part in parts:
            part = part.strip().lower()
            if part in ["prefill_only", "decode_only"]:
                # Replace with a valid base scheduler type
                if part == "prefill_only":
                    filtered_parts.append("prefill_first")
                elif part == "decode_only":
                    filtered_parts.append("fcfs")
            else:
                filtered_parts.append(part)

        # Ensure we have at least one valid scheduler type
        if not filtered_parts:
            filtered_parts = ["fcfs"]

        return ",".join(filtered_parts)

    def _init_pd_components(self):
        """Initialize PD disaggregation components"""
        if self.pd_mode == PDSchedulerMode.UNIFIED:
            logger.info("scheduler running in unified mode, skipping pd components")
            return

        try:
            # Create metadata buffers
            buffer_size = max(self.prefill_num_tasks, self.decode_num_tasks) * 2
            self.metadata_buffers = MetadataBuffers(buffer_size)

            # Determine disaggregation mode
            if self.pd_mode == PDSchedulerMode.PREFILL_ONLY:
                disaggregation_mode = DisaggregationMode.PREFILL
            elif self.pd_mode == PDSchedulerMode.DECODE_ONLY:
                disaggregation_mode = DisaggregationMode.DECODE
            else:
                raise ValueError(f"unsupported pd mode: {self.pd_mode}")

            # Create KV manager
            # Note: cache_manager will be set later in the initialization process
            self.kv_manager = KVManager(
                cache_manager=None,  # Will be set later
                metadata_buffers=self.metadata_buffers,
                disaggregation_mode=disaggregation_mode,
            )

            logger.info(f"initialized pd components for {self.pd_mode.value} mode")

        except Exception as e:
            logger.error(f"failed to initialize pd components: {e}")
            raise

    def set_cache_manager(self, cache_manager):
        """Set cache manager after initialization"""
        if self.kv_manager is not None:
            self.kv_manager.cache_manager = cache_manager
            # Re-register buffers with the actual cache manager (now safe)
            try:
                self.kv_manager.register_buffer_to_engine()
            except Exception as e:
                logger.error(f"failed to register buffers to transfer engine: {e}")
            logger.info("cache manager set for kv manager")

    def set_token_manager(self, token_manager):
        """Attach DP token manager so we can stream tokens back to Router."""
        self.token_manager = token_manager
        logger.info("token manager set for pd scheduler")

    async def process_request(self, request_data: dict[str, Any]):
        """Process incoming request"""
        request_id = request_data.get("request_id")
        request_type = request_data.get("type", "regular")
        scheduler_type = request_data.get("scheduler_type")

        logger.debug(f"processing request {request_id} of type {request_type}")

        if request_type == "pd_request":
            if (
                scheduler_type == "prefill"
                and self.pd_mode == PDSchedulerMode.PREFILL_ONLY
            ):
                await self._process_prefill_request(request_data)
            elif (
                scheduler_type == "decode"
                and self.pd_mode == PDSchedulerMode.DECODE_ONLY
            ):
                await self._process_decode_request(request_data)
            else:
                logger.warning(
                    f"scheduler type mismatch: got {scheduler_type}, mode is {self.pd_mode}"
                )
        else:
            # Regular request - use traditional processing
            await self._process_regular_request(request_data)

    async def _process_prefill_request(self, request_data: dict[str, Any]):
        """Process Prefill-only request"""
        request_id = request_data["request_id"]
        original_request = request_data["request"]

        logger.info(f"processing prefill request: {request_id}")

        try:
            # Create task from request
            task = self._create_task_from_request(original_request, TaskType.Prefill)

            # Execute prefill (simplified). KV + logits sending is handled by KV hook.
            _ = await self._execute_prefill(task)
            logger.info(
                f"prefill completed for request: {request_id}; kv transfer will be handled by hook"
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"failed to process prefill request {request_id}: {e}")

    async def _process_decode_request(self, request_data: dict[str, Any]):
        """Process Decode-only request"""
        request_id = request_data["request_id"]
        original_request = request_data["request"]
        prefill_scheduler_id = request_data.get("prefill_scheduler_id")

        logger.info(
            f"processing decode request: {request_id} from prefill scheduler {prefill_scheduler_id}"
        )

        try:
            # Store decode request info
            decode_info = {
                "request_id": request_id,
                "original_request": original_request,
                "prefill_scheduler_id": prefill_scheduler_id,
                "status": PDRequestStatus.PENDING,
                "created_time": time.time(),
            }
            self.pending_decode_requests[request_id] = decode_info

            # 告知 KVManager 该请求应当绑定到的 Prefill engine_rank（与 Router 的 prefill_scheduler_id 对应）
            try:
                if (
                    self.kv_manager is not None
                    and hasattr(self.kv_manager, "set_prefill_target_engine_rank")
                    and prefill_scheduler_id is not None
                ):
                    self.kv_manager.set_prefill_target_engine_rank(
                        request_id, int(prefill_scheduler_id)
                    )
            except Exception as e:
                logger.warning(
                    f"failed to set prefill target engine rank for {request_id}: {e}"
                )

            # Start waiting for KV cache
            asyncio.create_task(self._wait_for_kv_cache(request_id))

        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.error(f"failed to process decode request {request_id}: {e}")

    async def _wait_for_kv_cache(self, request_id: str):
        """Wait for KV cache and start decode"""
        decode_info = self.pending_decode_requests.get(request_id)
        if not decode_info:
            logger.error(f"decode info not found for request: {request_id}")
            return

        try:
            logger.info(f"waiting for kv cache for request: {request_id}")

            # Wait for KV cache through KV manager (fake prefill)
            if self.kv_manager:
                first_token_logits = self.kv_manager.recv_kv_cache_and_insert(
                    request_ids=[request_id],
                    cache_manager=self.kv_manager.cache_manager,
                )

                logger.info(f"received kv cache for request: {request_id}")

                # Update status
                decode_info["status"] = PDRequestStatus.DECODE_RUNNING
                decode_info["decode_start_time"] = time.time()

                # Create task for decode
                task = self._create_task_from_request(
                    decode_info["original_request"], TaskType.Decode
                )

                # Execute decode with received first token
                await self._execute_decode(task, first_token_logits)

                # Update status
                decode_info["status"] = PDRequestStatus.COMPLETED
                decode_info["decode_complete_time"] = time.time()

                logger.info(f"completed decode for request: {request_id}")

        except Exception as e:
            logger.error(f"failed to wait for kv cache for request {request_id}: {e}")
            if decode_info:
                decode_info["status"] = PDRequestStatus.FAILED
                decode_info["error_message"] = str(e)

    async def _process_regular_request(self, request_data: dict[str, Any]):
        """Process regular request (traditional mode)"""
        # Use parent class logic for regular requests
        logger.info("processing regular request in traditional mode")
        # This would integrate with the existing Scheduler logic

    def _create_task_from_request(self, request, task_type: TaskType) -> Task:
        """Create Task object from request using existing UserRequest/Task semantics"""
        from chitu.task import UserRequest

        # Support dict payload from Router
        if isinstance(request, dict):
            request_id = (
                request.get("conversation_id")
                or request.get("request_id")
                or str(time.time())
            )
            messages = request.get("messages") or request.get("message") or []
            max_new_tokens = (
                request.get("max_new_tokens") or request.get("max_tokens") or 50
            )
            temperature = request.get("temperature", 1.0)
            top_p = request.get("top_p", 0.9)
            top_k = request.get("top_k", 50)
            frequency_penalty = request.get("frequency_penalty", 0.0)
            chat_template_kwargs = request.get("chat_template_kwargs", {})
        else:
            request_id = getattr(
                request,
                "conversation_id",
                getattr(request, "request_id", str(time.time())),
            )
            messages = getattr(request, "messages", getattr(request, "message", []))
            max_new_tokens = getattr(
                request, "max_new_tokens", getattr(request, "max_tokens", 50)
            )
            temperature = getattr(request, "temperature", 1.0)
            top_p = getattr(request, "top_p", 0.9)
            top_k = getattr(request, "top_k", 50)
            frequency_penalty = getattr(request, "frequency_penalty", 0.0)
            chat_template_kwargs = getattr(request, "chat_template_kwargs", {})

        # Build a UserRequest compatible with engine
        user_req = UserRequest(
            message=messages,
            request_id=request_id,
            max_new_tokens=int(max_new_tokens),
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            chat_template_kwargs=(
                chat_template_kwargs if isinstance(chat_template_kwargs, dict) else {}
            ),
        )

        # Respect stop_with_eos/ignore_eos semantics from request (if exists)
        stop_with_eos = True

        # From dict payload
        if isinstance(request, dict):
            if request.get("ignore_eos") is True:
                stop_with_eos = False
            elif request.get("stop_with_eos") is False:
                stop_with_eos = False
        else:
            # From object payloads
            if getattr(request, "ignore_eos", False) is True:
                stop_with_eos = False
            elif getattr(request, "stop_with_eos", None) is False:
                stop_with_eos = False

        task = Task(
            task_id=user_req.request_id,
            req=user_req,
            priority=getattr(request, "priority", 1),
            stop_with_eos=stop_with_eos,
        )
        # Wrap task to enable streaming tokens to Router if token_manager is available
        try:
            if self.token_manager is not None:
                from chitu.dp_token_sender import DPTaskWrapper  # type: ignore

                task = self.token_manager.wrap_task(task)
        except Exception as e:
            logger.warning(f"failed to wrap task with token manager: {e}")
        if task_type == TaskType.Decode:
            task.consume_req_tokens()
        return task

    def _tokenize_messages(self, messages):
        """Tokenize messages (placeholder implementation)"""
        # This should integrate with the actual tokenizer
        # For now, return a dummy token sequence
        return [1, 2, 3, 4, 5]  # Placeholder

    async def _execute_prefill(self, task: Task) -> torch.Tensor:
        """Execute prefill via unified Executor path (PP/TP compatible)."""
        from chitu.backend import Backend
        from chitu.task import (
            PackedTasksBase,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        logger.info(f"executing prefill for task: {task.task_id}")

        # Ensure PackedTasksBase configured
        try:
            from chitu.task import PackedTasks as _PT
            from chitu.task import PackedTasksBase as _PTB

            if not _PTB.configured:
                args = get_global_args()
                from chitu.task import PackedTasks as __PT

                __PT.configure(max_num_tasks=args.infer.max_reqs)
        except Exception:
            pass

        # Build a PackedTasksBase with one task (avoid TaskPool dependency)
        tokens = task.prefix_tokens
        tasks = PackedTasksBase(
            num_tasks=1,
            task_ids=[task.task_id],
            req_ids=[task.req.request_id],
            task_type=TaskType.Prefill,
            tokens=[tokens],
            num_tokens=len(tokens),
            has_outputs=[1],
            payload_type=SerializedPackedTasksPayloadType.Prefill,
        )

        # Run through executor TP-only path (PP not supported in PD mode for now)
        logits = Backend.executor.prefill_step_tp_only(tasks)

        logger.info(f"prefill completed for task: {task.task_id}")
        return logits

    async def _execute_decode(self, task: Task, first_token_logits: torch.Tensor):
        """Execute real decode from first-token logits"""
        from chitu.backend import Backend
        import torch

        logger.info(f"executing decode for task: {task.task_id}")

        tokens = task.prefix_tokens
        req_id = task.req.request_id
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        try:
            Backend.cache_manager.prepare_cache_prefill([req_id], [len(tokens)])
            payload_prefill = torch.tensor(
                tokens, device=torch.device(local_rank), dtype=torch.int64
            )
            output_token_offsets = torch.tensor(
                [payload_prefill.size(0) - 1],
                dtype=torch.int32,
                device=payload_prefill.device,
            )
            prefill_logits_local = Backend.model.prefill(
                payload_prefill, output_token_offsets
            )
            Backend.cache_manager.finalize_cache_all_prefill()
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.warning(f"fake prefill on decode failed or skipped: {e}")
            prefill_logits_local = None

        if first_token_logits is not None and first_token_logits.numel() > 0:
            next_token = torch.argmax(first_token_logits.view(-1)).item()
        elif prefill_logits_local is not None and prefill_logits_local.numel() > 0:
            next_token = torch.argmax(prefill_logits_local.view(-1)).item()
        else:
            raise ValueError("no logits available for decode")
        assert isinstance(next_token, int)

        # Use Task API so DP wrapper can stream token back to Router
        try:
            task.update_response_sync(next_token)
        except Exception:
            # Fallback to direct add if wrapper not present
            task.req.add_data(next_token)

        # Early stop on EOS right after first token if needed
        if task.stop_with_eos and task.next_token in Backend.tokenizer.stop_tokens:
            task.req.finish_reason = "stop"
            logger.info(f"decode completed for task: {task.task_id}")
            return

        max_new = task.req.max_new_tokens
        req_id = task.req.request_id

        for _ in range(max_new - 1):
            # Decode one step via TP-only executor path
            step_logits = Backend.executor.decode_step_tp_only([req_id], [next_token])
            next_token = torch.argmax(step_logits.view(-1)).item()
            assert isinstance(next_token, int)
            try:
                task.update_response_sync(next_token)
            except Exception:
                task.req.add_data(next_token)

            # Cache finalize is handled inside decode_step_tp_only
            if task.stop_with_eos and task.next_token in Backend.tokenizer.stop_tokens:
                task.req.finish_reason = "stop"
                break

        logger.info(f"decode completed for task: {task.task_id}")

    def get_pd_stats(self) -> dict:
        """Get PD disaggregation statistics"""
        stats = {
            "pd_mode": self.pd_mode.value,
            "scheduler_id": self.scheduler_id,
            "pending_decode_requests": len(self.pending_decode_requests),
        }

        if self.pd_mode == PDSchedulerMode.DECODE_ONLY:
            # Add decode-specific stats
            status_counts = {}
            for decode_info in self.pending_decode_requests.values():
                status = (
                    decode_info["status"].value
                    if hasattr(decode_info["status"], "value")
                    else str(decode_info["status"])
                )
                status_counts[status] = status_counts.get(status, 0) + 1

            stats["decode_status_counts"] = status_counts

        return stats


class PrefillOnlyScheduler(PDScheduler):
    """Prefill-only Scheduler"""

    def __init__(
        self,
        prefill_num_tasks: int,
        scheduler_type: str = "prefill_first",
        scheduler_id: int = 0,
    ):
        super().__init__(
            prefill_num_tasks=prefill_num_tasks,
            decode_num_tasks=1,  # Not used in prefill-only mode
            scheduler_type=scheduler_type,
            pd_mode=PDSchedulerMode.PREFILL_ONLY,
            scheduler_id=scheduler_id,
        )


class DecodeOnlyScheduler(PDScheduler):
    """Decode-only Scheduler"""

    def __init__(
        self, decode_num_tasks: int, scheduler_type: str = "fcfs", scheduler_id: int = 0
    ):
        super().__init__(
            prefill_num_tasks=1,  # Not used in decode-only mode
            decode_num_tasks=decode_num_tasks,
            scheduler_type=scheduler_type,
            pd_mode=PDSchedulerMode.DECODE_ONLY,
            scheduler_id=scheduler_id,
        )
