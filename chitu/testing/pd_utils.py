# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""PD-disaggregation end-to-end test: inject mock requests and verify they
complete through the full prefill→decode→finish pipeline."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Optional

from chitu.global_vars import get_global_args
from chitu.metrics import stop_metrics_monitor
from chitu.task import UserRequest
from chitu.dp_router import get_request_router, get_token_router

if TYPE_CHECKING:
    from chitu.distributed.pd_disaggregation.pd_request_router import PDRequestRouter

logger = logging.getLogger(__name__)

# Shared system prompt long enough to fill ≥1 full KV block (block_size=64 tokens).
_SHARED_SYSTEM_PROMPT = (
    "你是一位知识渊博、经验丰富的人工智能助手，擅长回答各种类型的问题，"
    "包括但不限于烹饪食谱、编程技术、科学知识、生活建议、历史文化、数学计算、"
    "语言翻译、创意写作等领域。你的回答应该详细、专业、条理清晰，确保内容"
    "准确、全面且易于理解。在回答之前，请先仔细分析用户的问题意图和需求，"
    "然后组织你的思考过程，最后给出结构化的高质量回答。"
)


class PDTestRunner:
    """Orchestrates PD-disaggregation smoke-test requests.

    Called from ``PDRequestRouter._wait_for_pd_instances()`` when
    ``pd_test.enable`` is non-zero.  The runner dispatches ``req_num`` mock
    requests, monitors them via the Token Router, prints per-request
    results, and finally shuts the router down.
    """

    def __init__(self, router: "PDRequestRouter") -> None:
        test_cfg = get_global_args().pd_test

        self.req_timeout = test_cfg.req_timeout
        self.num_requests = test_cfg.req_num
        self.output_len = test_cfg.output_len

        self.num_completed = 0
        self.num_failed = 0
        self.start_time = 0.0

        self.req_pool: dict[str, UserRequest] = {}
        # enable == 2 → inject a shared system prompt so that later
        # requests can hit the prefix cache.
        self._inject_system_prompt = test_cfg.enable == 2

        logger.info(
            f"[PD_TEST] test mode enabled: num_requests={self.num_requests} "
            f"timeout={self.req_timeout:.1f}s output_len={self.output_len} "
            f"inject_system_prompt={self._inject_system_prompt}"
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        try:
            request_ids = await self._create_requests()
            if not request_ids:
                logger.warning("[PD_TEST] no test requests were created, skipping")
                return

            await self._monitor(request_ids)
            await self._shutdown()
        except Exception as e:
            logger.exception(
                f"[PD_TEST] An error occured in PDTestRunner, PD test failed! {type(e).__name__}: {str(e)}"
            )
            os._exit(1)

    # ------------------------------------------------------------------
    # Create requests
    # ------------------------------------------------------------------

    _TEST_MESSAGES: list = [
        [{"role": "user", "content": "宫保鸡丁怎么做?"}],
        [{"role": "user", "content": "what is the recipe of Kung Pao chicken?"}],
        [{"role": "user", "content": "怎么写程序?"}],
        [{"role": "user", "content": "飞机在对流层还是平流层飞?"}],
        [{"role": "user", "content": "怎么避免加班?"}],
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
    ]

    def _build_message(self, i: int) -> list[dict[str, str]]:
        """Build one test message.

        When ``pd_test.enable == 2``, every message is prefixed with a
        shared system prompt so that later requests hit the prefix cache.
        """
        msg = self._TEST_MESSAGES[i % len(self._TEST_MESSAGES)]
        if self._inject_system_prompt:
            return [{"role": "system", "content": _SHARED_SYSTEM_PROMPT}] + msg
        return msg

    async def _create_requests(self) -> list[str]:
        logger.info(f"[PD_TEST] creating {self.num_requests} test requests")
        request_ids: list[str] = []
        request_router: "PDRequestRouter" = get_request_router()
        token_router = get_token_router()

        for i in range(self.num_requests):
            msg = self._build_message(i)
            req = UserRequest.create(
                msg,
                request_id=f"pd_test_{i:06d}",
                max_new_tokens=self.output_len,
                frequency_penalty=0.0,
                temperature=0,
            )
            request_ids.append(req.request_id)
            logger.debug(
                f"[PD_TEST] request {req.request_id} prompt={msg[0]['content'][:40]}..."
            )
            await request_router.add_request(req)
            await token_router.register_request(req)
            self.req_pool[req.request_id] = req

        self.start_time = time.time()
        logger.info(
            f"[PD_TEST] all {len(request_ids)} test requests dispatched, "
            f"start_time={self.start_time:.3f}"
        )
        return request_ids

    # ------------------------------------------------------------------
    # Monitor
    # ------------------------------------------------------------------

    async def _monitor(
        self,
        request_ids: list[str],
        poll_interval_s: float = 0.5,
        allow_request_error: bool = True,
    ) -> None:
        logger.info(
            f"[PD_TEST] monitoring {len(request_ids)} requests, "
            f"timeout={self.req_timeout:.1f}s"
        )

        request_router: "PDRequestRouter" = get_request_router()
        token_router = get_token_router()
        request_set = frozenset(request_ids)
        running: list[str] = request_ids
        completed: set[str] = set()
        failed: set[str] = set()

        while running:
            now = time.time()
            still_running: list[str] = []
            for request_id in running:
                if self.req_pool[request_id].finished:
                    if (
                        not allow_request_error
                        and self.req_pool[request_id].async_stream.error_message
                        is not None
                    ):
                        failed.add(request_id)
                        self.num_failed = len(failed)
                        self.num_completed = len(completed)
                        return
                    completed.add(request_id)
                else:
                    pd_req = request_router.pending_pd_requests.get(request_id, None)
                    if (
                        pd_req is not None
                        and now - pd_req.created_time > self.req_timeout
                    ):
                        logger.error(
                            f"[PD_TEST] request {request_id} exceeded timeout "
                            f"({now - pd_req.created_time:.1f}s > "
                            f"{self.req_timeout:.1f}s), "
                            f"status={pd_req.status.value} "
                            f"error={pd_req.error_message or 'none'}"
                        )
                        failed.add(request_id)
                        self.num_failed = len(failed)
                        self.num_completed = len(completed)
                        return
                    still_running.append(request_id)

            running = still_running
            await asyncio.sleep(poll_interval_s)

        total_elapsed = time.time() - self.start_time
        self.num_completed = len(completed)
        self.num_failed = len(failed)
        logger.info(
            f"[PD_TEST] all requests finished: completed={self.num_completed} "
            f"failed={self.num_failed} total_elapsed={total_elapsed:.3f}s"
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def _shutdown(self) -> None:
        logger.info(
            f"[PD_TEST] test finished: completed={self.num_completed} "
            f"failed={self.num_failed}, shutting down router"
        )

        if self.num_failed > 0:
            logger.error(
                f"[PD_TEST] {self.num_failed}/{self.num_requests} "
                f"requests failed — check logs above for details"
            )
        else:
            logger.info(
                f"[PD_TEST] all {self.num_completed} requests "
                f"completed successfully"
            )

        request_router: "PDRequestRouter" = get_request_router()
        token_router = get_token_router()
        for rid, req in self.req_pool.items():
            if not rid.startswith("pd_test_"):
                continue
            completed_flag = rid not in token_router.active_requests
            logger.warning(
                f"[PD_TEST][result] rid={rid} "
                f"input_len={req.prompt_len} max_new_tokens={req.max_new_tokens} "
                f"output_tokens={req.num_output_tokens} "
                f"cached_tokens={req.num_hit_tokens} "
                f"output={req.output}"
            )

        # When prefix caching is enabled and test mode is 2 (shared system
        # prompt), verify that at least one request hit the prefix cache.
        if get_global_args().infer.enable_prefix_caching and self._inject_system_prompt:
            num_cache_hit = sum(
                1
                for rid, req in self.req_pool.items()
                if rid.startswith("pd_test_") and req.num_hit_tokens > 0
            )
            if num_cache_hit == 0:
                logger.error(
                    "[PD_TEST] prefix caching check FAILED: no request has "
                    "num_hit_tokens > 0 despite enable_prefix_caching=True "
                    "and pd_test.enable=2"
                )
                self.num_failed += 1
            else:
                logger.info(
                    f"[PD_TEST] prefix caching check PASSED: "
                    f"{num_cache_hit} request(s) hit the prefix cache"
                )

        await asyncio.sleep(1.0)

        token_router = get_token_router(check_exist=False)
        if token_router is not None:
            await token_router.begin_termination()

        await request_router.terminate_instances()
        drained = await request_router.wait_for_instances_terminated()
        if not drained:
            logger.error(
                "[PD_TEST] graceful shutdown timed out — instances did not "
                "terminate within the drain window"
            )
            self.num_failed += 1

        if token_router is not None:
            await token_router.shutdown()
        await request_router.shutdown()
        stop_metrics_monitor()

        logger.info("[PD_TEST] exiting process")
        os._exit(0 if self.num_failed == 0 else 1)
