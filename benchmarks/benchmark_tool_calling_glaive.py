# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark tool-calling accuracy using NousResearch/func-calling-eval-glaive.

Usage:
    python benchmark_tool_calling_glaive.py --host localhost --port 8000 --api openai
"""

import os
import json
import argparse
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_API_KEY = os.getenv("API_KEY", "example_key")
DEFAULT_CACHE_ROOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "datasets"
    / "NousResearch___func-calling-eval-glaive"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run NousResearch/func-calling-eval-glaive through OpenAI and/or "
            "Anthropic compatible tool-calling APIs."
        )
    )
    parser.add_argument(
        "--api",
        choices=["openai", "anthropic", "both"],
        default="both",
        help="Which API surface to test.",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Server IP or hostname.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port.",
    )
    parser.add_argument(
        "--tool-choice",
        choices=["auto", "required"],
        default="auto",
        help="Tool choice policy for tool-calling requests.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API key used for both clients.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL"),
        help="Model name. If omitted, it is discovered from the OpenAI models API.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in the dataset.",
    )
    parser.add_argument(
        "--count",
        type=int,
        help="How many samples to run. Default is all rows from --start onward.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="max_tokens/max_new_tokens sent to the API.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSONL output path.",
    )
    return parser.parse_args()


def get_openai_base_url(args: argparse.Namespace) -> str:
    return f"http://{args.host}:{args.port}/v1"


def get_anthropic_base_url(args: argparse.Namespace) -> str:
    return f"http://{args.host}:{args.port}"


def load_glaive_rows() -> list[dict[str, Any]]:
    try:
        dataset = load_dataset("NousResearch/func-calling-eval-glaive")["train"]
        return [dataset[i] for i in range(len(dataset))]
    except Exception:
        import pyarrow.ipc as ipc

        arrow_files = sorted(
            DEFAULT_CACHE_ROOT.rglob("func-calling-eval-glaive-train.arrow")
        )
        if not arrow_files:
            raise
        with ipc.open_stream(str(arrow_files[-1])) as reader:
            return reader.read_all().to_pylist()


def resolve_model_name(model: str | None, base_url: str, api_key: str) -> str:
    if model:
        return model
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    models = client.models.list().data
    if not models:
        raise RuntimeError("No model available from OpenAI models API")
    else:
        print(f"Get model name: {models}")
    return models[0].id


def parse_expected_call(
    row: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tools = json.loads(row["tools"])
    completion = json.loads(row["completion"])
    return tools, completion


def openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anthropic_tools = []
    for tool in tools:
        function = tool.get("function", {})
        anthropic_tools.append(
            {
                "name": function["name"],
                "description": function.get("description", ""),
                "input_schema": function.get("parameters", {"type": "object"}),
            }
        )
    return anthropic_tools


def extract_openai_result(message: Any) -> dict[str, Any]:
    tool_calls = []
    for tool_call in message.tool_calls or []:
        tool_calls.append(
            {
                "id": tool_call.id,
                "name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments),
            }
        )
    return {
        "text": message.content or "",
        "tool_calls": tool_calls,
    }


def extract_anthropic_result(message: Any) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in message.content or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(getattr(block, "text", "") or "")
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": getattr(block, "id", None),
                    "name": getattr(block, "name", ""),
                    "arguments": getattr(block, "input", None) or {},
                }
            )
    return {
        "text": "".join(text_parts).strip(),
        "tool_calls": tool_calls,
    }


def run_openai_case(
    client: Any,
    *,
    model: str,
    row: dict[str, Any],
    tool_choice: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    tools, expected = parse_expected_call(row)
    response = client.chat.completions.create(
        model=model,
        messages=row["prompt"],
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=False,
        max_tokens=max_tokens,
        temperature=temperature,
        # extra_body={"enable_thinking": False},
    )
    result = extract_openai_result(response.choices[0].message)
    return build_case_result("openai", row, expected, result)


def run_anthropic_case(
    client: Any,
    *,
    model: str,
    row: dict[str, Any],
    tool_choice: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    tools, expected = parse_expected_call(row)
    anthropic_tools = openai_tools_to_anthropic(tools)
    response = client.messages.create(
        model=model,
        messages=row["prompt"],
        tools=anthropic_tools,
        tool_choice={"type": "auto" if tool_choice == "auto" else "any"},
        max_tokens=max_tokens,
        temperature=temperature,
    )
    result = extract_anthropic_result(response)
    return build_case_result("anthropic", row, expected, result)


def build_case_result(
    api: str,
    row: dict[str, Any],
    expected: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    first_call = result["tool_calls"][0] if result["tool_calls"] else None
    matched = bool(
        first_call
        and len(result["tool_calls"]) == 1
        and first_call["name"] == expected["name"]
        and first_call["arguments"] == expected["arguments"]
    )
    return {
        "api": api,
        "category": row["category"],
        "task": row["task"],
        "source": row["source"],
        "prompt": row["prompt"],
        "expected": expected,
        "actual": {
            "text": result["text"],
            "tool_calls": result["tool_calls"],
        },
        "passed": matched,
    }


def print_case_result(index: int, result: dict[str, Any]) -> None:
    actual_calls = result["actual"]["tool_calls"]
    actual_name = actual_calls[0]["name"] if actual_calls else "<none>"
    status = "PASS" if result["passed"] else "FAIL"
    print(
        f"[{result['api']}] #{index} {status} "
        f"expected={result['expected']['name']} actual={actual_name} "
        f"task={result['task']}"
    )
    if result["passed"]:
        return
    print(f"  prompt={json.dumps(result['prompt'], ensure_ascii=False)}")
    print(f"  expected={json.dumps(result['expected'], ensure_ascii=False)}")
    print(f"  actual={json.dumps(result['actual'], ensure_ascii=False)}")


def maybe_write_result(path: Path | None, result: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def run_api(
    api: str,
    *,
    rows: list[tuple[int, dict[str, Any]]],
    model: str,
    args: argparse.Namespace,
) -> tuple[int, int]:
    if api == "openai":
        from openai import OpenAI

        client = OpenAI(base_url=get_openai_base_url(args), api_key=args.api_key)
        runner = run_openai_case
    else:
        from anthropic import Anthropic

        client = Anthropic(base_url=get_anthropic_base_url(args), api_key=args.api_key)
        runner = run_anthropic_case

    passed = 0
    total = 0
    for index, row in rows:
        try:
            result = runner(
                client,
                model=model,
                row=row,
                tool_choice=args.tool_choice,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        except Exception as exc:
            result = {
                "api": api,
                "category": row["category"],
                "task": row["task"],
                "source": row["source"],
                "prompt": row["prompt"],
                "expected": json.loads(row["completion"]),
                "actual": {"text": "", "tool_calls": [], "error": str(exc)},
                "passed": False,
            }
        print_case_result(index, result)
        maybe_write_result(args.output, result)
        passed += int(result["passed"])
        total += 1

    print(f"[{api}] summary: {passed}/{total} passed")
    return passed, total


def main() -> None:
    args = parse_args()
    rows = load_glaive_rows()
    total_rows = len(rows)
    end = total_rows if args.count is None else min(total_rows, args.start + args.count)
    selected_rows = list(enumerate(rows))[args.start : end]
    if not selected_rows:
        raise SystemExit("No dataset rows selected")
    print(
        f"dataset size={total_rows}, selected range=[{args.start}, {end}), "
        f"selected count={len(selected_rows)}"
    )

    model = resolve_model_name(args.model, get_openai_base_url(args), args.api_key)
    apis = ["openai", "anthropic"] if args.api == "both" else [args.api]

    total_passed = 0
    total_count = 0
    for api in apis:
        passed, count = run_api(api, rows=selected_rows, model=model, args=args)
        total_passed += passed
        total_count += count

    print(f"[overall] summary: {total_passed}/{total_count} passed")


if __name__ == "__main__":
    main()
