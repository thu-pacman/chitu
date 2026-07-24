# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_one_record(data):
    """将单条 benchmark 结果字典解析为绘图所需输入。"""
    start_ts = data["start_timestamps"]
    response_ts = data["response_timestamp"]
    batch_size = data["batch_size"]
    model_name = data["model"]["name"]
    total_input_tokens = data["total_input_tokens"]
    total_output_tokens = data["total_output_tokens"]

    assert len(start_ts) == batch_size
    assert len(response_ts) == batch_size

    return (
        start_ts,
        response_ts,
        model_name,
        batch_size,
        total_input_tokens,
        total_output_tokens,
    )


def load_data_jsonl(file_path):
    """
    加载 JSONL 文件（每行一个 JSON，如 benchmark_serving --append-result 输出）。
    返回 (start_ts, response_ts, model_name, batch_size, total_input_tokens, total_output_tokens) 的列表。
    """
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(parse_one_record(data))
    return records


def prepare_plot_data(start_ts, response_ts):
    start_x, start_y = [], []
    token_x, token_y = [], []
    first_x, first_y = [], []
    t0 = 0

    for req_id, (s_time, token_times) in enumerate(zip(start_ts, response_ts)):
        if req_id == 0:
            t0 = s_time
        start_x.append(s_time - t0)
        start_y.append(req_id)

        if isinstance(token_times, list) and len(token_times) > 0:
            # 第一个响应时间单独收集，用不同颜色表示
            first_x.append(token_times[0] - t0)
            first_y.append(req_id)
            for t_time in token_times[1:]:
                token_x.append(t_time - t0)
                token_y.append(req_id)

    return start_x, start_y, token_x, token_y, first_x, first_y


def plot_timestamps(
    start_x,
    start_y,
    token_x,
    token_y,
    first_x,
    first_y,
    model_name,
    batch_size,
    total_input_tokens,
    total_output_tokens,
    save_path=None,
):
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.scatter(
        token_x, token_y, c="#1f77b4", s=2, alpha=0.6, label="Token response time"
    )

    ax.scatter(
        first_x, first_y, c="#1fb426", s=2, alpha=0.6, label="First token response time"
    )

    ax.scatter(
        start_x, start_y, c="#d62728", s=2, alpha=0.6, label="Request start time"
    )

    title = (
        f"Response time of all tokens in each request\n"
        f"Model: {model_name} | Batch size: {batch_size} | "
        f"Total input tokens: {total_input_tokens} | Total output tokens: {total_output_tokens}"
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Request ID", fontsize=12, fontweight="bold")

    y_min, y_max = min(start_y), max(start_y)
    ax.set_yticks(np.arange(y_min, y_max + 1, step=max(1, (y_max - y_min) // 20)))
    ax.set_ylim(y_min - 1, y_max + 1)

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Scatter saved: {save_path}")
    else:
        plt.show()


def load_records(file_path):
    """
    加载 benchmark 结果：自动识别单条 JSON 与 JSONL（多行）。
    单条 JSON：整个文件为一个对象（如 benchmark_serving 未加 --append-result 时的输出）。
    JSONL：每行一个 JSON 对象（来自 benchmark_serving --append-result）。
    返回 (start_ts, response_ts, model_name, batch_size, total_input_tokens, total_output_tokens) 的列表。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # 先尝试按单条 JSON 解析（支持多行格式化）
    try:
        data = json.loads(raw)
        return [parse_one_record(data)]
    except json.JSONDecodeError:
        pass

    # 按 JSONL 处理：每行一个 JSON
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        records.append(parse_one_record(data))
    if not records:
        raise ValueError(f"No valid JSON/JSONL content in {file_path}")
    return records


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--benchmark-results", required=True)
    parser.add_argument("--save-path", required=True, help="图片保存目录")
    args = parser.parse_args()

    records = load_records(args.benchmark_results)
    output_dir = args.save_path
    os.makedirs(output_dir, exist_ok=True)

    for record in records:
        (
            start_timestamps,
            response_timestamps,
            model_name,
            batch_size,
            total_input_tokens,
            total_output_tokens,
        ) = record

        save_path = os.path.join(output_dir, f"{model_name}_{batch_size}.jpg")

        start_x, start_y, token_x, token_y, first_x, first_y = prepare_plot_data(
            start_timestamps, response_timestamps
        )

        plot_timestamps(
            start_x,
            start_y,
            token_x,
            token_y,
            first_x,
            first_y,
            model_name=model_name,
            batch_size=batch_size,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            save_path=save_path,
        )
