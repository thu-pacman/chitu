# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise a PD service with Decode CP, PP, MTP, and prefix caching.

Run with ``python3 test/dcp_pp_test.py PORT [--host HOST]``.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
import urllib.request


def complete(base, name, prompt, expected=None):
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(
            {
                "model": "GLM-5.2-W8A8",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 160,
                "temperature": 0,
                "enable_thinking": False,
                "stream": False,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    print(
        json.dumps({"case": name, "response": result}, ensure_ascii=False), flush=True
    )
    choice = result["choices"][0]
    content = choice["message"]["content"]
    assert content and "�" not in content, result
    assert choice["finish_reason"] == "stop", result
    if expected is not None:
        assert expected in content, result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ready-timeout", type=float, default=1800)
    args = parser.parse_args()
    base = f"http://{args.host}:{args.port}"
    deadline = time.monotonic() + args.ready_timeout
    while True:
        try:
            with urllib.request.urlopen(base + "/server_status", timeout=5) as response:
                if json.load(response)["initialized"]:
                    break
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("PD service did not initialize")
        time.sleep(2)

    complete(base, "generation", "请用三句话解释为什么夏天适合早晨或傍晚散步。")
    complete(base, "arithmetic", "12减5再加3等于多少？只回复数字。", "10")
    prompt = (
        "记录的暗号是蓝色风铃。\n"
        + "这是用于验证长上下文的背景资料，内容涉及日常天气与城市散步。\n" * 1100
        + "请回忆第一行记录的暗号，只回复暗号。"
    )
    first = complete(base, "long_context", prompt, "蓝色风铃")
    assert first["usage"]["prompt_tokens"] > 8192, first
    repeated = complete(base, "prefix_repeat", prompt, "蓝色风铃")
    assert repeated["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, repeated

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                complete,
                base,
                f"concurrent_{i}",
                f"请只回复：请求{i}成功。",
                f"请求{i}成功",
            )
            for i in range(4)
        ]
        for future in futures:
            future.result()
    print("DECODE_CP_PP_PASSED", flush=True)


if __name__ == "__main__":
    main()
