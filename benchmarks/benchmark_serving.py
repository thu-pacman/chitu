"""
Benchmark runner for Chitu performance testing.

The serving benchmark logic is partially adapted from vLLM's benchmark_serving
  (https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py),
  licensed under Apache 2.0. This adaption aims to follow widely-used
  benchmarking practices for LLM inference throughput and latency.
"""

import requests
import os
import sys
import time
import json
import argparse
import traceback
import numpy as np
import torch
from typing import List
from dataclasses import dataclass, field

import aiohttp
import asyncio

import pkg_resources

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class BenchmarkConfig:
    model_name: str
    batch_size: int
    sequence_length: int
    num_iterations: int
    warmup_iterations: int
    stream: bool
    # max_concurrency: int


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
        self.results: List[BenchmarkResult] = []
        self.current_run_metrics = {}

        # Validate configuration
        if config.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        if config.num_iterations < 1:
            raise ValueError("Number of iterations must be at least 1")

        # Print configuration
        print(f"Benchmark Configuration:")
        print(f"  Model: {config.model_name}")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Sequence Length: {config.sequence_length}")
        print(f"  Iterations: {config.num_iterations}")
        print(f"  Warmup Iterations: {config.warmup_iterations}")
        print(f"  Base URL: {base_url}")

    def _run_inference(self, min_batch_size: int = 1):
        """Run a single inference pass with proper measurement."""
        # Create a request based on configuration
        messages = self._get_test_messages()

        # Prepare request payload
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": self.config.sequence_length,
            "stream": False,  # Non-streaming for consistent measurement
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": 50,
            "min_batch_size": min_batch_size,
            "stop_with_eos": False,
        }

        # Send request and measure time
        start_time = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        end_time = time.perf_counter()

        # Process response
        if response.status_code != 200:
            raise RuntimeError(
                f"Request failed with status {response.status_code}: {response.text}"
            )

        response_data = response.json()

        # Calculate metrics
        latency = (end_time - start_time) * 1000  # ms

        # Store detailed metrics for this run
        self.current_run_metrics = {
            "latency_ms": latency,
            "prompt_tokens": int(
                response_data.get("usage", {}).get("prompt_tokens", 0)
            ),
            "completion_tokens": int(
                response_data.get("usage", {}).get("completion_tokens", 0)
            ),
            "total_tokens": int(response_data.get("usage", {}).get("total_tokens", 0)),
        }

        tps = self.current_run_metrics.get("completion_tokens") * 1000 / latency
        return latency, tps

    async def _async_run_inference(self, min_batch_size: int = 1) -> RequestFuncOutput:
        """Run an async inference pass with proper measurement."""
        # Create a request based on configuration
        messages = self._get_test_messages()

        # Prepare request payload
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": self.config.sequence_length,
            "stream": True,  # Non-streaming for consistent measurement
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": 50,
            "min_batch_size": min_batch_size,
            "stop_with_eos": False,
        }

        async with aiohttp.ClientSession(
            trust_env=True, timeout=AIOHTTP_TIMEOUT
        ) as session:
            output = RequestFuncOutput()
            generated_text = ""

            # Send request and measure time
            st = time.perf_counter()

            try:
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=60,
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
                                elif choices := data.get("choices"):
                                    # Note that text could be empty here
                                    # e.g. for special tokens
                                    text = choices[0].get("delta").get("content")
                                    timestamp = time.perf_counter()
                                    # First token
                                    if not first_chunk_received:
                                        first_chunk_received = True
                                        ttft = time.perf_counter() - st
                                        output.ttft = ttft

                                    # Decoding phase
                                    else:
                                        output.itl.append(
                                            timestamp - most_recent_timestamp
                                        )

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

    def save_results(self, output_dir: str):
        """Save benchmark results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)

        results = []
        for r in self.results:
            results.append(
                {
                    "latency_ms": r.latency_ms,
                    "throughput": r.throughput,
                    "config": {
                        "model_name": r.config.model_name,
                        "batch_size": r.config.batch_size,
                        "sequence_length": r.config.sequence_length,
                        # "device": r.config.device,
                    },
                }
            )

        output_file = os.path.join(output_dir, "benchmark_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    # TODO: support hf dataset
    def _get_test_messages(self):
        """Get test messages based on configuration."""
        # Define a set of standard test messages with varying complexity
        test_messages = [
            # Short query
            [{"role": "user", "content": "What is machine learning?"}],
            # Medium query
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Explain the difference between supervised and unsupervised learning.",
                },
            ],
            # Long conversation
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What are the main attractions in Paris?"},
                {
                    "role": "assistant",
                    "content": "Paris has many famous attractions including the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.",
                },
                {"role": "user", "content": "Tell me more about the Louvre Museum."},
            ],
        ]

        # Select message complexity based on sequence length
        if self.config.sequence_length < 128:
            return test_messages[0]
        elif self.config.sequence_length < 512:
            return test_messages[1]
        else:
            return test_messages[2]

    def run_batch_benchmark(self) -> List[BenchmarkResult]:
        """Run benchmark with concurrent requests to simulate batch processing."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(
            f"Running batch benchmark with {self.config.batch_size} concurrent requests..."
        )

        results = []

        with ThreadPoolExecutor(max_workers=self.config.batch_size) as executor:
            # Submit all requests concurrently
            futures = []
            for _ in range(self.config.batch_size):
                futures.append(
                    executor.submit(
                        self._run_inference, min_batch_size=self.config.batch_size
                    )
                )

            # Collect results as they complete
            for future in as_completed(futures):
                latency, tps = future.result()
                result = BenchmarkResult(
                    latency_ms=latency,
                    tps=tps,
                    throughput=(1000 / latency),  # req/s
                    config=self.config,
                )
                results.append(result)

                # Print progress
                print(
                    f"Batch request completed: "
                    f"TPS={result.tps:.2f} tok/s, "
                    f"Latency={result.latency_ms:.2f}ms, "
                    f"Throughput={result.throughput:.2f} req/s"
                )

        return results

    def run(self) -> BenchmarkResult:
        """Run the benchmark and return aggregated results."""
        if self.config.batch_size > 1:
            results = self.run_batch_benchmark()
        else:
            # Warmup
            print(f"Warming up with {self.config.warmup_iterations} iterations...")
            for _ in range(self.config.warmup_iterations):
                self._run_inference()

            # Reset GPU memory stats after warmup
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            print(f"Running {self.config.num_iterations} benchmark iterations...")

            # Measure
            results = []
            for i in range(self.config.num_iterations):
                latency, tps = self._run_inference()

                result = BenchmarkResult(
                    latency_ms=latency,
                    tps=tps,
                    throughput=(1000 / latency),  # req/s
                    config=self.config,
                )

                results.append(result)

                # Print progress
                print(
                    f"Iteration {i+1}/{self.config.num_iterations}: "
                    f"TPS={result.tps:.2f} req/s, "
                    f"Latency={result.latency_ms:.2f}ms, "
                    f"Throughput={result.throughput:.2f} req/s, "
                )

        # Aggregate results
        avg_tps = sum(r.tps for r in results) / len(results)
        avg_latency = sum(r.latency_ms for r in results) / len(results)
        avg_throughput = sum(r.throughput for r in results) / len(results)

        final_result = BenchmarkResult(
            tps=avg_tps,
            latency_ms=avg_latency,
            throughput=avg_throughput,
            config=self.config,
        )

        self.results.append(final_result)
        return final_result

    async def run_async(self):
        tasks: list[asyncio.Task] = []
        for _ in range(self.config.batch_size):
            tasks.append(
                asyncio.create_task(
                    self._async_run_inference(min_batch_size=self.config.batch_size)
                )
            )
        outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)

        # outputs = asyncio.run(self._async_run_inference())

        return outputs


def calculate_metrics(
    outputs: list[RequestFuncOutput],
    dur_s: float,
    selected_percentiles: list[float],
) -> tuple[BenchmarkMetrics, list[int]]:
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
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
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
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


def save_dict_result(result: dict, output_dir: str):
    """Save dict benchmark results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "benchmark_results.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run Chitu performance benchmarks")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output-dir")
    parser.add_argument("--base-url", help="URL of the Chitu server endpoint")
    parser.add_argument("--metric-percentiles", type=str, default="99")
    parser.add_argument("--percentile-metrics", type=str, default="ttft,tpot,itl")

    args = parser.parse_args()

    config = BenchmarkConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
        stream=args.stream,
        # max_concurrency=args.max_concurrency,
    )

    runner = BenchmarkServing(config, base_url=args.base_url)

    # Run benchmark
    print("\nStarting benchmark...")
    start_time = time.perf_counter()
    if not config.stream:
        result = runner.run()

        total_time = time.perf_counter() - start_time

        # Print results
        print(f"\nBenchmark completed in {total_time:.2f} seconds")
        print(f"\nBenchmark Results:")
        print(f"Average TPS: {result.tps:.2f}")
        print(f"Average Latency: {result.latency_ms:.2f} ms")
        print(f"Throughput: {result.throughput:.2f} samples/sec")
        if args.output_dir:
            runner.save_results(args.output_dir)
            print(
                f"\nDetailed results saved to {args.output_dir}/benchmark_results.json"
            )
    else:
        outputs = asyncio.run(runner.run_async())

        total_time = time.perf_counter() - start_time

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

        codebase = {
            "name": "chitu",
            "version": pkg_resources.get_distribution("chitu").version,
        }

        model = {"name": config.model_name}

        result = {
            "env": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
            "codebase": codebase,
            "model": model,
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
            save_dict_result(result, args.output_dir)
            print(
                f"\nDetailed results saved to {args.output_dir}/benchmark_results.json"
            )


if __name__ == "__main__":
    main()
