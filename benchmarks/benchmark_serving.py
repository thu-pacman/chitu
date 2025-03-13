"""
Benchmark runner for Chitu performance testing.
"""

import requests
import os
import time
import json
import argparse
import torch
from typing import List
from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    model_name: str
    batch_size: int
    sequence_length: int
    num_iterations: int
    warmup_iterations: int


@dataclass
class BenchmarkResult:
    tps: int
    latency_ms: float
    throughput: float
    config: BenchmarkConfig


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

    def _run_inference(self):
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
                        "device": r.config.device,
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

    # TODO: Implement batch benchmark
    def run_batch_benchmark(self) -> List[BenchmarkResult]:
        """Run benchmark with concurrent requests to simulate batch processing."""
        return []

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


def main():
    parser = argparse.ArgumentParser(description="Run Chitu performance benchmarks")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output-dir")
    parser.add_argument("--base-url", help="URL of the Chitu server endpoint")

    args = parser.parse_args()

    config = BenchmarkConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
    )

    runner = BenchmarkServing(config, base_url=args.base_url)

    # Run benchmark
    print("\nStarting benchmark...")
    start_time = time.perf_counter()
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
        print(f"\nDetailed results saved to {args.output_dir}/benchmark_results.json")


if __name__ == "__main__":
    main()
