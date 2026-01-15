# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
import json
import matplotlib.pyplot as plt
import numpy as np


def load_data(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

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


def prepare_plot_data(start_ts, response_ts):
    start_x, start_y = [], []
    token_x, token_y = [], []
    t0 = 0

    for req_id, (s_time, token_times) in enumerate(zip(start_ts, response_ts)):
        if req_id == 0:
            t0 = s_time
        start_x.append(s_time - t0)
        start_y.append(req_id)

        if isinstance(token_times, list) and len(token_times) > 0:
            for t_time in token_times:
                token_x.append(t_time - t0)
                token_y.append(req_id)

    return start_x, start_y, token_x, token_y


def plot_timestamps(
    start_x,
    start_y,
    token_x,
    token_y,
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--benchmark-results", required=True)
    parser.add_argument("--save-path", required=True)
    args = parser.parse_args()

    (
        start_timestamps,
        response_timestamps,
        model_name,
        batch_size,
        total_input_tokens,
        total_output_tokens,
    ) = load_data(args.benchmark_results)

    start_x, start_y, token_x, token_y = prepare_plot_data(
        start_timestamps, response_timestamps
    )

    plot_timestamps(
        start_x,
        start_y,
        token_x,
        token_y,
        model_name=model_name,
        batch_size=batch_size,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        save_path=args.save_path,
    )
