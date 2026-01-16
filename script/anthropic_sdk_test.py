#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026
# SPDX-License-Identifier: Apache-2.0
"""
Minimal Anthropic SDK integration test against a compatible /v1/messages endpoint.

Env (overridable by flags): CHITU_BASE_URL, ANTHROPIC_API_KEY, ANTHROPIC_MODEL, MAX_TOKENS
"""

import argparse
import os
import sys
import time
from typing import Any, Optional

from anthropic import Anthropic  # type: ignore

TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "non_stream_basic",
        "stream": False,
        "kwargs": {
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "宫保鸡丁怎么做?"}],
        },
    },
    {
        "name": "stream_basic",
        "stream": True,
        "kwargs": {
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [{"role": "user", "content": "宫保鸡丁怎么做?"}],
        },
    },
]


def _require_env_or_arg(value: Optional[str], env_name: str, flag: str) -> str:
    if value:
        return value
    v = os.getenv(env_name)
    if v:
        return v
    raise SystemExit(f"Missing {flag} (or set {env_name})")


def _optional_api_key(value: Optional[str]) -> str:
    v = value or os.getenv("ANTHROPIC_API_KEY")
    return v if v is not None else "no-api-key"


def _extract_blocks(content: Any) -> tuple[str, str]:
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    for block in content or []:
        t = getattr(block, "type", None)
        if t == "text":
            text_parts.append(getattr(block, "text", "") or "")
        elif t == "thinking":
            thinking_parts.append(getattr(block, "thinking", "") or "")
    return "".join(text_parts).strip(), "".join(thinking_parts).strip()


def _split_think_block(s: str) -> tuple[str, str]:
    if not s:
        return "", ""
    start = s.find("<think>")
    if start < 0:
        return s, ""
    head = s[:start]
    tail = s[start + len("<think>") :]
    end = tail.find("</think>")
    if end >= 0:
        thinking = tail[:end]
    else:
        thinking = tail
    return head.strip(), thinking.strip()


def _split_think_tags(s: str) -> tuple[str, str, bool]:
    if not s:
        return "", "", False
    i = s.find("<think>")
    if i < 0:
        return s, "", False
    head = s[:i]
    tail = s[i + len("<think>") :]
    return head, tail, True


def _run_case(
    client: Any, *, model: str, max_tokens: int, case: dict[str, Any]
) -> tuple[bool, str]:
    t0 = time.perf_counter()
    name = case["name"]
    stream = bool(case["stream"])
    kwargs = dict(case["kwargs"])

    saw_answer = False
    saw_thinking = False
    in_thinking = False

    try:
        if stream:
            with client.messages.stream(
                model=model, max_tokens=max_tokens, **kwargs
            ) as s:
                for event in s:
                    if getattr(event, "type", None) != "content_block_delta":
                        continue

                    delta = getattr(event, "delta", None)
                    dtype = getattr(delta, "type", None)

                    if dtype == "thinking_delta":
                        if not saw_thinking:
                            if saw_answer:
                                sys.stdout.write("\n\n")
                            sys.stdout.write("[THINKING]\n")
                            saw_thinking = True
                        sys.stdout.write(getattr(delta, "thinking", "") or "")
                        sys.stdout.flush()
                        in_thinking = True
                        continue

                    if dtype == "text_delta":
                        chunk = getattr(delta, "text", "") or ""
                        if not chunk:
                            continue

                        if not in_thinking:
                            head, tail, switched = _split_think_tags(chunk)
                            if head:
                                sys.stdout.write(head)
                                sys.stdout.flush()
                                saw_answer = True
                            if switched:
                                if not saw_thinking:
                                    if saw_answer:
                                        sys.stdout.write("\n\n")
                                    sys.stdout.write("[THINKING]\n")
                                    saw_thinking = True
                                if tail:
                                    sys.stdout.write(tail)
                                    sys.stdout.flush()
                                in_thinking = True
                        else:
                            sys.stdout.write(chunk)
                            sys.stdout.flush()
                        continue

                msg = s.get_final_message()
        else:
            msg = client.messages.create(model=model, max_tokens=max_tokens, **kwargs)

        content = getattr(msg, "content", None)
        final_text, final_thinking = _extract_blocks(content)
        if not final_thinking:
            head, think = _split_think_block(final_text)
            if think:
                final_text = head
                final_thinking = think

        if not stream:
            if final_thinking:
                sys.stdout.write("[THINKING]\n" + final_thinking + "\n")
                sys.stdout.flush()
                saw_thinking = True
                sys.stdout.write("\n")
            if final_text:
                sys.stdout.write(final_text + "\n")
                sys.stdout.flush()
                saw_answer = True
        else:
            if (not saw_answer) and final_text:
                sys.stdout.write(final_text + "\n")
                sys.stdout.flush()
                saw_answer = True
            if (not saw_thinking) and final_thinking:
                if saw_answer:
                    sys.stdout.write("\n\n")
                sys.stdout.write("[THINKING]\n" + final_thinking)
                sys.stdout.flush()
                saw_thinking = True
            if saw_thinking:
                sys.stdout.write("\n")
                sys.stdout.flush()

        assert getattr(msg, "id", None), "missing message.id"
        assert getattr(msg, "model", None), "missing message.model"
        assert saw_answer or final_text, "empty answer"

        dt = time.perf_counter() - t0
        return True, f"---------- {name}: PASS ({dt:.3f}s) ----------"
    except Exception as e:
        dt = time.perf_counter() - t0
        return False, f"---------- {name}: FAIL ({dt:.3f}s): {e} ----------"


def main() -> int:
    ap = argparse.ArgumentParser(description="Anthropic SDK integration smoke test")
    ap.add_argument(
        "--base-url",
        default=os.getenv("CHITU_BASE_URL"),
        help="Compatible API base URL",
    )
    ap.add_argument(
        "--model", default=os.getenv("ANTHROPIC_MODEL", ""), help="Model name / alias"
    )
    ap.add_argument(
        "--api-key", default=os.getenv("ANTHROPIC_API_KEY"), help="API key (optional)"
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("MAX_TOKENS", "1024")),
        help="max_tokens",
    )
    args = ap.parse_args()

    try:
        base_url = _require_env_or_arg(args.base_url, "CHITU_BASE_URL", "--base-url")
        api_key = _optional_api_key(args.api_key)
        model = _require_env_or_arg(args.model, "ANTHROPIC_MODEL", "--model")
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2

    client = Anthropic(api_key=api_key, base_url=base_url)

    ok = True
    for case in TEST_CASES:
        passed, line = _run_case(
            client, model=model, max_tokens=args.max_tokens, case=case
        )
        print(line)
        ok = ok and passed

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
