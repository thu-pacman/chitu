#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Repeated OpenAI-compatible chat requests that share a long fixed prefix.

Use this to manually verify PD Router prefix-aware prefill routing: after the first
completion returns (first token triggers router shadow-cache updates), later requests
should show increasing prefix block hits in Router logs::

    grep '\\[PD_ROUTER\\]\\[prefill_scores\\]' log/router.*.log

Requires only stdlib (no httpx). Typical invocation::

    python test/pd_router_shared_prefix_requests.py \\
        --base-url http://172.31.0.38:21006 \\
        --model Qwen3-30B-A3B \\
        --num-requests 12 \\
        --max-tokens 32

Use ``--stream false`` (default) so each request finishes and first-token path runs
before the next request unless you intentionally overlap traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# Long shared prefix (natural language, tokenizer-dependent block boundaries).
# Keep stable across requests; only the final user line changes. The router's
# prefix-cache score counts full KV blocks only (256 tokens in the current PD
# setup), so repeat the paragraph enough times to guarantee reusable blocks.
_SHARED_PREFIX_PARAGRAPH = """以下是背景说明（请在后续对话中始终视为上下文的一部分，不要重复全文）：

机器学习推理服务往往采用数据并行与张量并行组合部署。前缀缓存可以在多台设备之间复用已计算的 KV，
从而减少重复 prefill 开销。路由层若感知前缀命中，可以把请求调度到更可能命中缓存的实例上。

为了验证路由与缓存行为，我们使用固定段落作为公共前缀，仅在最后追加一条不同的用户问题。
"""

DEFAULT_SHARED_PREFIX = "\n\n".join(
    f"[固定前缀段落 {i:02d}]\n{_SHARED_PREFIX_PARAGRAPH}" for i in range(1, 17)
)


def build_messages(run_index: int, shared_prefix: str) -> list[dict]:
    suffix = f"第 {run_index} 次提问：请只回答一个数字 {run_index}，不要输出其它文字。"
    content = shared_prefix + "\n\n" + suffix
    return [{"role": "user", "content": content}]


def post_chat_completion(
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    stream: bool,
    timeout_s: float,
) -> tuple[int, float, str]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0.0,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            elapsed = time.perf_counter() - t0
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, elapsed, raw
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - t0
        raw = e.read().decode("utf-8", errors="replace")
        return e.code, elapsed, raw


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Send many chat requests sharing the same long prefix (PD router test)."
    )
    ap.add_argument(
        "--base-url",
        default="http://127.0.0.1:21006",
        help="Router HTTP base URL (no trailing path).",
    )
    ap.add_argument(
        "--model",
        default="Qwen3-30B-A3B",
        help="model field in JSON body (must match server config).",
    )
    ap.add_argument(
        "--num-requests", type=int, default=10, help="How many POSTs to send."
    )
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument(
        "--delay-s",
        type=float,
        default=0.2,
        help="Sleep between successful requests (serial pressure).",
    )
    ap.add_argument(
        "--timeout-s",
        type=float,
        default=600.0,
        help="Per-request HTTP timeout.",
    )
    ap.add_argument(
        "--stream",
        action="store_true",
        help="Use stream=true (default off for simpler response parsing).",
    )
    ap.add_argument(
        "--prefix-file",
        default=None,
        help="If set, read shared prefix text from this file instead of built-in paragraph.",
    )
    args = ap.parse_args()

    if args.prefix_file:
        shared_prefix = open(args.prefix_file, encoding="utf-8").read()
    else:
        shared_prefix = DEFAULT_SHARED_PREFIX

    print(
        f"base_url={args.base_url} model={args.model} num_requests={args.num_requests} "
        f"max_tokens={args.max_tokens} stream={args.stream}",
        file=sys.stderr,
    )
    print(
        "Tip: watch Router log for [PD_ROUTER][prefill_scores] and [PD_ROUTER][prefill_select].",
        file=sys.stderr,
    )

    failures = 0
    for i in range(args.num_requests):
        messages = build_messages(i + 1, shared_prefix)
        status, elapsed, raw = post_chat_completion(
            args.base_url,
            args.model,
            messages,
            args.max_tokens,
            args.stream,
            args.timeout_s,
        )
        preview = raw[:200].replace("\n", " ") if raw else ""
        print(
            f"[{i+1}/{args.num_requests}] http={status} time={elapsed:.2f}s body_preview={preview!r}"
        )
        if status != 200:
            failures += 1
            print(f"  full_body={raw}", file=sys.stderr)
        elif args.delay_s > 0:
            time.sleep(args.delay_s)

    if failures:
        print(f"Done with {failures} non-200 responses.", file=sys.stderr)
        return 1
    print("All requests returned HTTP 200.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
