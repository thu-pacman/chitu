# SPDX-FileCopyrightText: 2025 vLLM Team
# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0
#
# The serving benchmark logic is partially adapted from vLLM's benchmark_serving
#  (https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py),
#  licensed under Apache 2.0. This adaption aims to follow widely-used
#  benchmarking practices for LLM inference throughput and latency.


"""
Benchmark runner for Chitu performance testing.
"""


import requests
import os
import sys
import time
import json
import random
import argparse
import traceback
import numpy as np
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict

import aiohttp
import asyncio

try:
    from transformers import AutoTokenizer
except:
    print("Failed to import AutoTokenizer, local tokenizer is disabled.")


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
RESULT_FILE = "benchmark_results.jsonl"
CI_PIPELINE_SOURCE = os.environ.get("CI_PIPELINE_SOURCE", "web")
TIMEOUT = 300 if CI_PIPELINE_SOURCE == "schedule" else 10000


@dataclass
class BenchmarkConfig:
    model_name: str
    batch_size: int
    input_length: int
    output_length: int
    num_iterations: int
    warmup_iterations: int
    dataset: str
    request_interval: float
    force_max_min_bs: bool
    tokenizer_path: str | None = None
    dataset_path: str | None = None


@dataclass
class BenchmarkResult:
    tps: int
    latency_ms: float
    throughput: float
    config: BenchmarkConfig


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""
    start_timestamp: float = 0.0
    response_timestamp: list[float] = field(default_factory=list)


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]


class ShareGPTDataset:
    """
    Implements the ShareGPT dataset.  Loads data from a JSON file and generates
    sample requests based on conversation turns.
    """

    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path
        self.load_data()
        self.curr_idx = 0

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = json.load(f)
        # Filter entries with at least two conversation turns.
        self.data = [
            conversation["value"]
            for entry in self.data
            if "conversations" in entry
            for conversation in entry["conversations"]
        ]

    def sample_single_content(
        self,
        tokenize_func,
        detokenize_func,
        content_len: int,
    ):
        token_ids = []
        while len(token_ids) < content_len:
            curr_token_ids = tokenize_func(self.data[self.curr_idx])
            token_ids += curr_token_ids
            self.curr_idx = (self.curr_idx + 1) % len(self.data)
        token_ids = token_ids[:content_len]
        return detokenize_func(token_ids)


class BenchmarkServing:
    """Runs performance benchmarks for Chitu models."""

    def __init__(self, config: BenchmarkConfig, base_url: str):
        """Initialize the benchmark runner.

        Args:
            config: Benchmark configuration
            base_url: URL of the Chitu server endpoint
        """
        self.config = config
        self.base_url = base_url
        self.results: list[BenchmarkResult] = []
        self.current_run_metrics = {}

        self.tokenizer = None
        if self.config.tokenizer_path:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.tokenizer_path,
                    trust_remote_code=True,
                )
                self.tokenize = partial(self.tokenizer.encode, add_special_tokens=False)
                self.detokenize = partial(
                    self.tokenizer.decode, skip_special_tokens=True
                )
            except:
                print(
                    "Local tokenizer is not loaded, trying tokenizer interface from server if available."
                )

        if not self.tokenizer:
            try:
                test_prompt = "tokenizer test message"
                test_token_ids = self.remote_tokenize(test_prompt)
                self.remote_detokenize(test_token_ids)
            except:
                raise Exception(
                    "Tokenizer interface unavailable, please install transformers and set local tokenizer using --tokenizer-path."
                )

            self.tokenize = self.remote_tokenize
            self.detokenize = self.remote_detokenize

        if self.config.dataset == "sharegpt":
            assert (
                self.config.dataset_path
            ), "args.dataset_path should be set to use sharegpt dataset."
            try:
                self.dataset = ShareGPTDataset(self.config.dataset_path)
            except:
                raise Exception(
                    "Dataset is not loaded, sharegpt dataset is not supported!"
                )

        # Validate configuration
        if config.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        if config.num_iterations < 1:
            raise ValueError("Number of iterations must be at least 1")

        # Print configuration
        print(f"Benchmark Configuration:")
        print(f"  Model: {config.model_name}")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Input Length: {config.input_length}")
        print(f"  Output Length: {config.output_length}")
        print(f"  Iterations: {config.num_iterations}")
        print(f"  Warmup Iterations: {config.warmup_iterations}")
        print(f"  Base URL: {base_url}")

    def remote_tokenize(self, prompt: str):
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "add_special_tokens": False,
        }

        response = requests.post(
            f"{self.base_url}/tokenize",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=TIMEOUT,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Tokenize request failed with status {response.status_code}: {response.text}"
            )

        data = response.json()
        tokens = data.get("tokens", [])

        return tokens

    def remote_detokenize(self, tokens: list[int]):
        payload = {
            "model": self.config.model_name,
            "tokens": tokens,
        }

        response = requests.post(
            f"{self.base_url}/detokenize",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=TIMEOUT,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Detokenize request failed with status {response.status_code}: {response.text}"
            )

        data = response.json()
        prompt = data.get("prompt", "")

        return prompt

    async def _async_run_inference(
        self, session, payload: Dict[str, Any]
    ) -> RequestFuncOutput:
        """Run an async inference pass with proper measurement."""
        output = RequestFuncOutput()
        generated_text = ""

        # Send request and measure time
        st = time.perf_counter()
        output.start_timestamp = st

        try:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=TIMEOUT,
            ) as response:
                if response.status == 200:
                    first_chunk_received = False

                    async for chunk in response.content:
                        chunk = chunk.strip()
                        if not chunk:
                            continue
                        chunk = chunk.decode("utf-8").removeprefix("data: ")
                        if chunk != "[DONE]":
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if usage := data.get("usage"):
                                output.output_tokens = int(
                                    usage.get("completion_tokens")
                                )
                                output.prompt_len = int(usage.get("prompt_tokens"))
                                most_recent_timestamp = time.perf_counter()
                            elif choices := data.get("choices"):
                                # Note that text could be empty here
                                # e.g. for special tokens
                                text = choices[0].get("delta").get("content")
                                if not text:
                                    text = (
                                        choices[0].get("delta").get("reasoning_content")
                                    )
                                timestamp = time.perf_counter()
                                output.response_timestamp.append(timestamp)
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT."
                            "This response will be marked as failed!"
                        )
                    output.generated_text = generated_text
                    output.latency = most_recent_timestamp - st
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        return output

    # TODO: support hf dataset
    def _get_test_messages(self):
        if self.config.dataset == "random":
            """
            Get test messages based on configuration.
            Generate message based on input length,
            ensure decoded-then-encoded prompt length matches the target token length.
            For example, for GPT2Tokenizer:
            [6880, 6881] -> ['Ġcalls', 'here'] ->
            [1650, 939, 486] -> ['Ġcall', 'sh', 'ere']
            """
            test_content = self.detokenize(
                self.tokenize(
                    self.detokenize(
                        [
                            random.randint(100, 10000)
                            for _ in range(self.config.input_length)
                        ]
                    )
                )[: self.config.input_length]
            )
            return [{"role": "user", "content": test_content.strip()}]
        elif self.config.dataset == "sharegpt":
            content = self.dataset.sample_single_content(
                self.tokenize, self.detokenize, self.config.input_length
            )
            return [{"role": "user", "content": content}]
        else:
            raise Exception("args.dataset only supports random or sharegpt")

    async def run_async(self):
        tasks: list[asyncio.Task] = []
        interval = self.config.request_interval
        messages_list = []

        for _ in range(self.config.batch_size):
            messages_list.append(self._get_test_messages())

        connector = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(
            connector=connector, trust_env=True, timeout=AIOHTTP_TIMEOUT
        ) as session:

            for i in range(self.config.batch_size):

                # Prepare request payload
                payload = {
                    "model": self.config.model_name,
                    "messages": messages_list[i],
                    "max_tokens": self.config.output_length,
                    "stream": True,
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "top_k": 50,
                    "min_batch_size": (
                        self.config.batch_size if self.config.force_max_min_bs else 1
                    ),
                    "stop_with_eos": False,
                }

                tasks.append(
                    asyncio.create_task(self._async_run_inference(session, payload))
                )
                await asyncio.sleep(interval)
            outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)

        return outputs

    def benchmark(self):
        # Warmup
        print(f"Warming up with {self.config.warmup_iterations} iterations...")
        for _ in range(self.config.warmup_iterations):
            asyncio.run(self.run_async())

        print(f"Running {self.config.num_iterations} benchmark iterations...")

        # Measure
        outputs: list[RequestFuncOutput] = []

        start_time = time.perf_counter()

        for _ in range(self.config.num_iterations):
            outputs.extend(asyncio.run(self.run_async()))

        total_time = time.perf_counter() - start_time

        return outputs, total_time


def calculate_metrics(
    outputs: list[RequestFuncOutput],
    dur_s: float,
    selected_percentiles: list[float],
    config: BenchmarkConfig | None = None,
) -> tuple[BenchmarkMetrics, list[int]]:
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []

    need_tokenizer_recalc = False
    for out in outputs:
        if out.success and out.output_tokens <= 0:
            need_tokenizer_recalc = True
            break

    if need_tokenizer_recalc and config:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name, trust_remote_code=True
            )

            success_indices: list[int] = []
            texts: list[str] = []
            for i, out in enumerate[RequestFuncOutput](outputs):
                if out.success:
                    success_indices.append(i)
                    texts.append(out.generated_text)

            if texts:
                encoded = tokenizer(
                    texts,
                    add_special_tokens=False,
                    return_length=True,
                )
                lengths = encoded.get("length")
                if lengths is None:
                    input_ids_batch = encoded["input_ids"]
                    lengths = [len(ids) for ids in input_ids_batch]

                actual_output_lens = [0] * len(outputs)
                for idx, out_len in zip(success_indices, lengths):
                    actual_output_lens[idx] = int(out_len)

                for i, out in enumerate(outputs):
                    if not out.success:
                        continue
                    output_len = actual_output_lens[i]
                    total_input += out.prompt_len
                    tpot = 0.0
                    if output_len > 1:
                        latency_minus_ttft = out.latency - out.ttft
                        tpot = latency_minus_ttft / (output_len - 1)
                        tpots.append(tpot)
                    all_tpots.append(tpot)
                    itls += out.itl
                    ttfts.append(out.ttft)
                    e2els.append(out.latency)
                    completed += 1
        except Exception:
            need_tokenizer_recalc = False

    if not need_tokenizer_recalc:
        actual_output_lens = []
        for i in range(len(outputs)):
            if outputs[i].success:
                output_len = outputs[i].output_tokens
                actual_output_lens.append(output_len)
                total_input += outputs[i].prompt_len
                tpot = 0
                if output_len > 1:
                    latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                    tpot = latency_minus_ttft / (output_len - 1)
                    tpots.append(tpot)
                all_tpots.append(tpot)
                itls += outputs[i].itl
                ttfts.append(outputs[i].ttft)
                e2els.append(outputs[i].latency)
                completed += 1
            else:
                actual_output_lens.append(0)

    # if completed == 0:
    #     warnings.warn(
    #         "All requests failed. This is likely due to a misconfiguration "
    #         "on the benchmark arguments.",
    #         stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[
            (p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[
            (p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[
            (p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
        ],
    )

    return metrics, actual_output_lens


def process_one_metric(
    # E.g., "ttft"
    metric_attribute_name: str,
    # E.g., "TTFT"
    metric_name: str,
    # E.g., "Time to First Token"
    metric_header: str,
    result,
    metrics,
    selected_percentile_metrics,
):
    # This function prints and adds statistics of the specified
    # metric.
    if metric_attribute_name not in selected_percentile_metrics:
        return
    print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
    print(
        "{:<40} {:<10.2f}".format(
            f"Mean {metric_name} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms"),
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            f"Median {metric_name} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms"),
        )
    )
    result[f"mean_{metric_attribute_name}_ms"] = getattr(
        metrics, f"mean_{metric_attribute_name}_ms"
    )
    result[f"median_{metric_attribute_name}_ms"] = getattr(
        metrics, f"median_{metric_attribute_name}_ms"
    )
    result[f"std_{metric_attribute_name}_ms"] = getattr(
        metrics, f"std_{metric_attribute_name}_ms"
    )
    for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
        p_word = str(int(p)) if int(p) == p else str(p)
        print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
        result[f"p{p_word}_{metric_attribute_name}_ms"] = value


def save_dict_result(result: dict, output_dir: str, append: bool = False):
    """Save dict benchmark results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, RESULT_FILE)
    if append:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    else:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Run Chitu performance benchmarks")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-dir", help="Output dir of benchmark json file")
    parser.add_argument("--base-url", help="URL of the Chitu server endpoint")
    parser.add_argument("--metric-percentiles", type=str, default="99")
    parser.add_argument("--percentile-metrics", type=str, default="ttft,tpot,itl")
    parser.add_argument("--append-result", action="store_true")
    parser.add_argument("--dataset", default="random", choices=["random", "sharegpt"])
    parser.add_argument("--dataset-path")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--request-interval", type=float, default=0.0)
    parser.add_argument("--force-max-min-bs", action="store_true")

    args = parser.parse_args()

    config = BenchmarkConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        input_length=args.input_len,
        output_length=args.output_len,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
        dataset=args.dataset,
        tokenizer_path=args.tokenizer_path,
        dataset_path=args.dataset_path,
        request_interval=args.request_interval,
        force_max_min_bs=args.force_max_min_bs,
    )

    runner = BenchmarkServing(config, base_url=args.base_url)

    # Run benchmark
    print("\nStarting benchmark...")

    outputs, total_time = runner.benchmark()

    metrics, actual_output_lens = calculate_metrics(
        outputs=outputs,
        dur_s=total_time,
        selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
    )

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", total_time))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total Token throughput (tok/s):", metrics.total_token_throughput
        )
    )

    model = {"name": config.model_name}

    result = {
        "model": model,
        "batch_size": config.batch_size,
        "duration": total_time,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "start_timestamps": [output.start_timestamp for output in outputs],
        "response_timestamp": [output.response_timestamp for output in outputs],
    }

    selected_percentile_metrics = args.percentile_metrics.split(",")

    process_one_metric(
        "ttft",
        "TTFT",
        "Time to First Token",
        result,
        metrics,
        selected_percentile_metrics,
    )
    process_one_metric(
        "tpot",
        "TPOT",
        "Time per Output Token (excl. 1st token)",
        result,
        metrics,
        selected_percentile_metrics,
    )
    process_one_metric(
        "itl",
        "ITL",
        "Inter-token Latency",
        result,
        metrics,
        selected_percentile_metrics,
    )
    process_one_metric(
        "e2el",
        "E2EL",
        "End-to-end Latency",
        result,
        metrics,
        selected_percentile_metrics,
    )

    print("=" * 50)

    if args.output_dir:
        save_dict_result(result, args.output_dir, append=args.append_result)
        print(f"\nDetailed results saved to {args.output_dir}/{RESULT_FILE}")


if __name__ == "__main__":
    main()
