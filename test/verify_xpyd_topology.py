#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import re
import sys
import time

import requests


def _format_messages(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content", "")
            if content:
                parts.append(f"[{role}] {content}")
    return "\n".join(parts)


def send_request(
    base_url: str, model_path: str, messages: list[dict], max_tokens: int, stream: bool
) -> tuple[bool, str]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    body = {
        "model": model_path,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0.7,
    }
    try:
        input_text = _format_messages(messages)
        if stream:
            with requests.post(url, json=body, stream=True, timeout=60) as resp:
                if resp.status_code != 200:
                    return False, f"HTTP {resp.status_code}"
                # Accumulate streamed tokens, print once per request
                generated_chunks: list[str] = []
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw
                    if line.startswith(b"data: "):
                        line = line[len(b"data: ") :]
                    if line == b"[DONE]":
                        break
                    try:
                        payload = json.loads(line)
                        delta = payload.get("choices", [{}])[0].get("delta", {})
                        piece = delta.get("content") or delta.get("reasoning_content")
                        if piece:
                            generated_chunks.append(piece)
                    except Exception:
                        continue
                output_text = "".join(generated_chunks)
                print("Input:\n" + input_text)
                print("Output:\n" + output_text + "\n", flush=True)
                return True, "ok"
        else:
            resp = requests.post(url, json=body, timeout=60)
            if resp.status_code != 200:
                return False, f"HTTP {resp.status_code}"
            try:
                data = resp.json()
                output_text = (
                    data.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                print("Input:\n" + input_text)
                print("Output:\n" + (output_text or "") + "\n", flush=True)
            except Exception:
                pass
            return True, "ok"
    except Exception as e:
        return False, f"EXC {e}"


def query_router_status(base_url: str) -> dict:
    try:
        url = f"{base_url.rstrip('/')}/dp/status"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def _discover_log_files(log_dir: str) -> tuple[dict[int, str], dict[int, str], str]:
    """Recursively discover prefill*.log / decode*.log / router.log under log_dir."""
    prefill_map: dict[int, str] = {}
    decode_map: dict[int, str] = {}
    router_log = os.path.join(log_dir, "router.log")

    if not os.path.isdir(log_dir):
        return prefill_map, decode_map, router_log

    # Prefer top-level router.log if exists; otherwise take the first found
    found_router = os.path.exists(router_log)

    for root, _dirs, files in os.walk(log_dir):
        for fn in files:
            m = re.match(r"prefill(\d+)\.log$", fn)
            if m:
                idx = int(m.group(1)) - 1
                prefill_map[idx] = os.path.join(root, fn)
                continue
            m = re.match(r"decode(\d+)\.log$", fn)
            if m:
                idx = int(m.group(1)) - 1
                decode_map[idx] = os.path.join(root, fn)
                continue
            if (not found_router) and fn == "router.log":
                router_log = os.path.join(root, fn)
                found_router = True

    return prefill_map, decode_map, router_log


def parse_logs(log_dir: str) -> dict:
    """Parse logs and summarize routing and hits for arbitrary P×D.
    Recursively discovers prefill/decode logs to avoid missing files under subdirectories.
    """
    prefill_map, decode_map, router_log = _discover_log_files(log_dir)

    # Initialize counters
    router_rr: dict[str, int] = {}
    prefill_ids: dict[str, set] = {f"P{i}": set() for i in sorted(prefill_map.keys())}
    decode_ids: dict[str, set] = {f"D{j}": set() for j in sorted(decode_map.keys())}

    # Router routing decisions
    if os.path.exists(router_log):
        with open(router_log, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.search(r"created pd request: (\w+) -> (P\d+)-(D\d+)", line)
                if m:
                    key = f"{m.group(2)}-{m.group(3)}"
                    router_rr[key] = router_rr.get(key, 0) + 1

    # Prefill logs（按 request_id 去重）
    for i, fn in prefill_map.items():
        tag = f"P{i}"
        if os.path.exists(fn):
            with open(fn, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = re.search(r"processing prefill request: (\w+)", line)
                    if m:
                        prefill_ids[tag].add(m.group(1))

    # Decode logs（按 request_id 去重）
    for j, fn in decode_map.items():
        tag = f"D{j}"
        if os.path.exists(fn):
            with open(fn, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = re.search(r"processing decode request: (\w+)", line)
                    if m:
                        decode_ids[tag].add(m.group(1))
                        continue
                    m2 = re.search(r"received kv cache for request: (\w+)", line)
                    if m2:
                        decode_ids[tag].add(m2.group(1))

    # Build summaries
    prefill_hits = {k: len(v) for k, v in prefill_ids.items()}
    decode_hits = {k: len(v) for k, v in decode_ids.items()}
    return {
        "router_routing": router_rr,
        "prefill_hits": prefill_hits,
        "decode_hits": decode_hits,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify 2P/ND topology by sending concurrent requests and parsing logs"
    )
    parser.add_argument(
        "--router",
        default="http://localhost:21003",
        help="Router base URL, e.g., http://<ip>:21003",
    )
    parser.add_argument(
        "--model", default="/data/nfs/Qwen3-32B", help="Model path/name"
    )
    parser.add_argument("--log_dir", default="logs", help="Logs directory")
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Number of concurrent requests"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=64, help="Max tokens per request"
    )
    parser.add_argument("--stream", action="store_true", help="Use streaming responses")
    parser.add_argument(
        "--wait_seconds",
        type=float,
        default=8.0,
        help="Max seconds to wait for decode logs to flush",
    )
    args = parser.parse_args()

    messages_bank = [
        [{"role": "user", "content": "你是谁?"}],
        [{"role": "user", "content": "讲个笑话"}],
        [{"role": "user", "content": "简单介绍一下中国历史"}],
        [{"role": "user", "content": "宫保鸡丁怎么做?"}],
        [{"role": "user", "content": "写一首五言绝句"}],
        [{"role": "user", "content": "1+1等于几?"}],
    ]

    print("Sending requests...")
    ok = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = []
        for i in range(args.concurrency):
            msgs = messages_bank[i % len(messages_bank)]
            futs.append(
                ex.submit(
                    send_request,
                    args.router,
                    args.model,
                    msgs,
                    args.max_tokens,
                    args.stream,
                )
            )
        for fu in concurrent.futures.as_completed(futs):
            success, info = fu.result()
            ok += 1 if success else 0
            if not success:
                print("Request failed:", info)

    # Give Router a moment to print routing logs
    time.sleep(1.0)

    status = query_router_status(args.router)
    if status:
        print("Router /dp/status:")
        try:
            print(json.dumps(status, ensure_ascii=False, indent=2))
        except Exception:
            print(status)

    # Poll logs until decode hits appear or timeout
    deadline = time.time() + max(0.0, args.wait_seconds)
    res = parse_logs(args.log_dir)
    expected = ok  # number of successful HTTP responses
    while time.time() < deadline:
        total_dec_hits = sum(res.get("decode_hits", {}).values())
        if expected == 0:
            break
        # stop early if enough decode hits observed
        if total_dec_hits >= min(
            expected, total_dec_hits if total_dec_hits > 0 else expected
        ):
            break
        time.sleep(0.5)
        res = parse_logs(args.log_dir)
    print(
        "\nRouting summary (from logs):", json.dumps(res, ensure_ascii=False, indent=2)
    )

    # Validation for generic P×D: all discovered Prefill/Decode should handle >=1 request,
    # and Router routing should include at least one route for every discovered P and D.
    prefill_all = sorted(res["prefill_hits"].keys())
    decode_all = sorted(res["decode_hits"].keys())
    p_ok = all(res["prefill_hits"][k] > 0 for k in prefill_all) if prefill_all else True
    d_ok = all(res["decode_hits"][k] > 0 for k in decode_all) if decode_all else True
    # Router seen Ps/Ds
    seen_ps = {
        re.match(r"(P\d+)-D\d+", k).group(1)
        for k in res["router_routing"].keys()
        if re.match(r"(P\d+)-D\d+", k)
    }
    seen_ds = {
        re.match(r"P\d+-(D\d+)", k).group(1)
        for k in res["router_routing"].keys()
        if re.match(r"P\d+-(D\d+)", k)
    }
    rr_ok = (set(prefill_all).issubset(seen_ps)) and (set(decode_all).issubset(seen_ds))

    if p_ok and d_ok and rr_ok:
        print(
            "\n[PASS] Topology validated: all Prefill/Decode instances handled requests and Router routed across them."
        )
        sys.exit(0)
    else:
        print("\n[WARN] 2P2D validation incomplete. Details:")
        print(f"  Prefill hits: {res['prefill_hits']}")
        print(f"  Decode  hits: {res['decode_hits']}")
        print(f"  Router   rr: {res['router_routing']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
