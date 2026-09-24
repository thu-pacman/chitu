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
from chitu.testing.phonebook import (
    SHARED_SEED,
    check_phonebook_test_results,
    check_prefix_cache_hit,
    gen_phonebook_prompt,
)

if TYPE_CHECKING:
    from chitu.distributed.pd_disaggregation.pd_request_router import PDRequestRouter

logger = logging.getLogger(__name__)

# 共享 system prompt 要够长，至少要能填满几个完整的 KV block——比一个 block 短的前缀
# 永远不会被复用，pd_test 末尾的 "prefix caching check" 就会失败。
# block size 随模型/后端变化（chitu/kv_cache/registry.py: default_paged_block_size_policy）：
# MLA absorb 的模型（如 GLM-4.7-Flash / DeepSeek）是 64，而线性注意力模型
# （Qwen3-Next / Qwen3.5 / GLM-5.3-Flash 在 H20 上）是 256，所以这里按最大的 256 来准备。
# 8 段（约 1.4k 字）能装满 3 个 256 的 block。
_BASE_SYSTEM_PROMPT = (
    "你是一位知识渊博、经验丰富的人工智能助手，擅长回答各种类型的问题，"
    "包括但不限于烹饪食谱、编程技术、科学知识、生活建议、历史文化、数学计算、"
    "语言翻译、创意写作等领域。你的回答应该详细、专业、条理清晰，确保内容"
    "准确、全面且易于理解。在回答之前，请先仔细分析用户的问题意图和需求，"
    "然后组织你的思考过程，最后给出结构化的高质量回答。"
)

_SHARED_SYSTEM_PROMPT = "\n\n".join(
    f"[背景说明 {i:02d}] {_BASE_SYSTEM_PROMPT}" for i in range(1, 9)
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
        # phonebook → long shared phone-book body per request, with the answer
        # checked at the end. Supersedes the shared system prompt above: the
        # phone-book body is already long enough to fill several KV blocks.
        self._phonebook = bool(test_cfg.phonebook)
        self._phonebook_num_entries = int(test_cfg.phonebook_num_entries)
        self._shared_prefix = self._inject_system_prompt or self._phonebook
        # request_id → expected phone number (phonebook mode only)
        self._expected: dict[str, str] = {}

        logger.info(
            f"[PD_TEST] test mode enabled: num_requests={self.num_requests} "
            f"timeout={self.req_timeout:.1f}s output_len={self.output_len} "
            f"inject_system_prompt={self._inject_system_prompt} "
            f"phonebook={self._phonebook} "
            f"phonebook_num_entries={self._phonebook_num_entries}"
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

    def _build_message(self, i: int) -> tuple[list[dict[str, str]], Optional[str]]:
        """Build one test message, plus its expected answer when applicable.

        When ``pd_test.enable == 2``, every message is prefixed with a
        shared system prompt so that later requests hit the prefix cache.
        When ``pd_test.phonebook`` is set, the message is instead a long
        phone-book prompt: every request shares the same body (fixed seed) and
        asks about a different entry, and the returned expected number is what
        the reply must contain — that is what turns a prefix-cache hit into a
        correctness check rather than a mere hit counter.
        """
        if self._phonebook:
            exp_idx = (i * 7 + 3) % self._phonebook_num_entries
            prompt, expected = gen_phonebook_prompt(
                self._phonebook_num_entries, exp_idx, seed=SHARED_SEED
            )
            return [{"role": "user", "content": prompt}], expected
        msg = self._TEST_MESSAGES[i % len(self._TEST_MESSAGES)]
        if self._inject_system_prompt:
            return [{"role": "system", "content": _SHARED_SYSTEM_PROMPT}] + msg, None
        return msg, None

    async def _create_requests(self) -> list[str]:
        logger.info(f"[PD_TEST] creating {self.num_requests} test requests")
        request_ids: list[str] = []
        request_router: "PDRequestRouter" = get_request_router()
        token_router = get_token_router()

        for i in range(self.num_requests):
            msg, expected = self._build_message(i)
            req = UserRequest.create(
                msg,
                request_id=f"pd_test_{i:06d}",
                max_new_tokens=self.output_len,
                frequency_penalty=0.0,
                temperature=0,
            )
            request_ids.append(req.request_id)
            if expected is not None:
                self._expected[req.request_id] = expected
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
        all_reqs: list[tuple[UserRequest, Optional[str]]] = []
        answer_pairs: list[tuple[UserRequest, str]] = []
        for rid, req in self.req_pool.items():
            if not rid.startswith("pd_test_"):
                continue
            completed_flag = rid not in token_router.active_requests
            expected = self._expected.get(rid)
            logger.warning(
                f"[PD_TEST][result] rid={rid} "
                f"input_len={req.prompt_len} max_new_tokens={req.max_new_tokens} "
                f"output_tokens={req.num_output_tokens} "
                f"cached_tokens={req.num_hit_tokens} "
                f"expected={expected} "
                f"output={req.output}"
            )
            all_reqs.append((req, expected))
            if expected is not None:
                answer_pairs.append((req, expected))

        # In phone-book mode the reply must still contain the right number: a
        # prefix-cache hit that restores the wrong KV / linear-attention state
        # would otherwise go unnoticed.
        if answer_pairs:
            try:
                check_phonebook_test_results(answer_pairs, context="PD_TEST")
            except AssertionError as e:
                logger.error(f"[PD_TEST] phone-book check FAILED: {e}")
                self.num_failed += 1

        # When prefix caching is enabled and the requests share a long prompt
        # (test mode 2 or phone-book mode), verify that at least one request hit
        # the prefix cache.
        if (
            get_global_args().infer.enable_prefix_caching
            and self._shared_prefix
            and all_reqs
        ):
            try:
                check_prefix_cache_hit(all_reqs, required=True, context="PD_TEST")
            except AssertionError as e:
                logger.error(f"[PD_TEST] prefix caching check FAILED: {e}")
                self.num_failed += 1

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
