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
from typing import TYPE_CHECKING

from chitu.dp_token_router import get_token_router
from chitu.global_vars import get_global_args
from chitu.task import UserRequest

if TYPE_CHECKING:
    from chitu.distributed.pd_disaggregation.pd_request_router import PDRequestRouter

logger = logging.getLogger(__name__)


class PDTestRunner:
    """Orchestrates PD-disaggregation smoke-test requests.

    Called from ``PDRequestRouter._wait_for_pd_instances()`` when
    ``pd_test.enable=True``.  The runner dispatches ``req_num`` mock
    requests, monitors them via the Token Router, prints per-request
    results, and finally shuts the router down.
    """

    def __init__(self, router: PDRequestRouter) -> None:
        self._router = router

        test_cfg = get_global_args().pd_test

        self.req_timeout = test_cfg.req_timeout
        self.num_requests = test_cfg.req_num
        self.output_len = test_cfg.output_len

        self.num_completed = 0
        self.num_failed = 0
        self.start_time = 0.0

        logger.info(
            f"[PD_TEST] test mode enabled: num_requests={self.num_requests} "
            f"timeout={self.req_timeout:.1f}s output_len={self.output_len}"
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        request_ids = await self._create_requests()
        if not request_ids:
            logger.warning("[PD_TEST] no test requests were created, skipping")
            return

        await self._monitor(request_ids)
        await self._shutdown()

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

    async def _create_requests(self) -> list[str]:
        logger.info(f"[PD_TEST] creating {self.num_requests} test requests")
        request_ids: list[str] = []
        token_router = get_token_router()

        for i in range(self.num_requests):
            msg = self._TEST_MESSAGES[i % len(self._TEST_MESSAGES)]
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
            await self._router.add_request(req)
            await token_router.register_request(req)

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
        self, request_ids: list[str], poll_interval_s: float = 0.5
    ) -> None:
        logger.info(
            f"[PD_TEST] monitoring {len(request_ids)} requests, "
            f"timeout={self.req_timeout:.1f}s"
        )

        token_router = get_token_router()
        request_set = frozenset(request_ids)
        completed: set[str] = set()
        failed: set[str] = set()

        while len(completed) + len(failed) < len(request_set):
            now = time.time()

            for pd_req in list(self._router.pending_pd_requests.values()):
                rid = pd_req.request_id
                if rid not in request_set or rid in completed or rid in failed:
                    continue
                if now - pd_req.created_time > self.req_timeout:
                    logger.error(
                        f"[PD_TEST] request {rid} exceeded timeout "
                        f"({now - pd_req.created_time:.1f}s > "
                        f"{self.req_timeout:.1f}s), "
                        f"status={pd_req.status.value} "
                        f"error={pd_req.error_message or 'none'}"
                    )
                    failed.add(rid)
                    self.num_failed = len(failed)
                    self.num_completed = len(completed)
                    return

            # Completed when token router removed it from active_requests.
            for rid in request_ids:
                if rid in completed or rid in failed:
                    continue
                if (
                    rid not in token_router.active_requests
                    and rid in self._router.pending_pd_requests
                ):
                    completed.add(rid)

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

        token_router = get_token_router()
        for rid, pd_req in self._router.pending_pd_requests.items():
            if not rid.startswith("pd_test_"):
                continue
            req = pd_req.original_request
            completed_flag = rid not in token_router.active_requests
            status_text = "COMPLETED" if completed_flag else pd_req.status.value.upper()
            logger.warning(
                f"[PD_TEST][result] rid={rid} status={status_text} "
                f"input_len={req.prompt_len} max_new_tokens={req.max_new_tokens} "
                f"output_tokens={req.num_output_tokens} output={req.output}"
            )

        await asyncio.sleep(1.0)
        await self._router.shutdown()

        logger.info("[PD_TEST] exiting process")
        os._exit(0 if self.num_failed == 0 else 1)
