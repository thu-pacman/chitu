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
from threading import Thread
from typing import Any, Optional
import urllib.request
from urllib.error import HTTPError

import hydra
import torch
from anthropic import Anthropic  # type: ignore

from chitu.chitu_main import chitu_init, warmup_engine
from chitu.schemas import ServeConfig
from chitu.serve.api_server import start_uvicorn
from chitu.serve.common import start_worker
from chitu.utils import get_config_dir_path, get_chitu_env

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


def _run_sdk_tests(*, base_url: str, model: str, api_key: str, max_tokens: int) -> int:
    client = Anthropic(api_key=api_key, base_url=base_url)
    ok = True
    for case in TEST_CASES:
        passed, line = _run_case(client, model=model, max_tokens=max_tokens, case=case)
        print(line)
        ok = ok and passed
    return 0 if ok else 1


def _wait_http_ready(host: str, port: int, timeout: float) -> None:
    url = f"http://{host}:{port}/ping"
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, method="POST")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return
        except HTTPError as e:
            last_err = e
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(f"HTTP server not ready: {last_err}")


@hydra.main(
    version_base=None,
    config_path=get_chitu_env(
        "CHITU_CONFIG_PATH", get_config_dir_path(), legacy_names=["CONFIG_PATH"]
    ),
    config_name=get_chitu_env(
        "CHITU_CONFIG_NAME", "serve_config", legacy_names=["CONFIG_NAME"]
    ),
)
def hydra_main(args: ServeConfig):
    chitu_init(args)
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    warmup_engine(args)

    import chitu.serve.api_server as api_server

    api_server.server_status = True
    worker_thread = Thread(target=start_worker, daemon=True)
    worker_thread.start()

    rank = torch.distributed.get_rank()
    if rank == 0:
        uvicorn_thread = Thread(target=start_uvicorn, args=(args,), daemon=True)
        uvicorn_thread.start()

        host = os.getenv("CHITU_BASE_HOST", "127.0.0.1")
        timeout = float(os.getenv("CHITU_ANTHROPIC_TIMEOUT", "60"))
        _wait_http_ready(host, args.serve.port, timeout)

        base_url = os.getenv("CHITU_BASE_URL", f"http://{host}:{args.serve.port}")
        api_key = os.getenv("ANTHROPIC_API_KEY", "test-api-key")
        model = os.getenv("ANTHROPIC_MODEL", args.models.name)
        max_tokens = int(os.getenv("MAX_TOKENS", "1024"))
        exit_code = _run_sdk_tests(
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])


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

    return _run_sdk_tests(
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    if os.getenv("CHITU_START_SERVER", "false").lower() == "true":
        hydra_main()
    else:
        raise SystemExit(main())
