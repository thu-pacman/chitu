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
from typing import Any, AsyncGenerator, Dict, Optional
from typing_extensions import override

import aiohttp
import asyncio

try:
    from transformers import AutoTokenizer
except Exception:
    print("Failed to import AutoTokenizer, local tokenizer is disabled.")


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
RESULT_FILE = "benchmark_results.jsonl"
CI_PIPELINE_SOURCE = os.environ.get("CI_PIPELINE_SOURCE", "web")
TIMEOUT = 300 if CI_PIPELINE_SOURCE == "schedule" else 10000


@dataclass
class BenchmarkConfig:
    model_name: str
    max_concurrency: Optional[int]
    num_requests: int
    num_iterations: int
    warmup_requests: int
    warmup_iterations: int
    input_length: Optional[int]
    output_length: int
    temperature: float
    top_p: float
    top_k: int
    dataset: str
    request_rate: float
    request_interval: float
    min_batch_size: Optional[int]
    stop_with_eos: bool
    print_generated: bool
    tokenizer_path: Optional[str] = None
    dataset_path: Optional[str] = None


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
    max_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]
    # Average number of successful requests in flight during the benchmark.
    concurrency: float
    max_concurrent_requests: int


class Dataset:
    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path
        self.data = self.load_data()
        self.curr_idx = 0

    def load_data(self) -> list[list[dict]]:
        raise NotImplementedError()

    def next_single_content(self) -> list[dict]:
        """
        Get the next content from the dataset
        """

        ret = self.data[self.curr_idx]
        self.curr_idx = (self.curr_idx + 1) % len(self.data)

        if ret[-1]["role"] == "assistant":
            # Pop the last turn if the turn is from the assistant, so the model can
            # answer the question
            return ret[:-1]
        else:
            return ret

    def sample_single_content(
        self, tokenize_func, detokenize_func, content_len: int
    ) -> list[dict]:
        """
        Sample the next >=1 contents and form a single content of the specified length
        """

        token_ids = []
        while len(token_ids) < content_len:
            cat_str = self._cat_all_turns(self.data[self.curr_idx])
            curr_token_ids = tokenize_func(cat_str)
            token_ids += curr_token_ids
            self.curr_idx = (self.curr_idx + 1) % len(cat_str)
        token_ids = token_ids[:content_len]
        return [{"role": "user", "content": detokenize_func(token_ids)}]

    def _cat_all_turns(self, turns: list[dict]) -> str:
        return "\n".join([turn["content"] for turn in turns])


class ShareGPTDataset(Dataset):
    """
    ShareGPT dataset.
    """

    @override
    def load_data(self) -> list[list[dict]]:
        with open(self.dataset_path, encoding="utf-8") as f:
            raw_data = json.load(f)
        data: list[list[dict]] = []
        for raw_session in raw_data:
            session: list[dict] = []
            if len(raw_session["conversations"]) == 0:
                # There are empty conversations in the dataset. Skip them.
                continue
            for turn in raw_session["conversations"]:
                if turn["from"] in {"human", "user"}:
                    session.append({"role": "user", "content": turn["value"]})
                elif turn["from"] in {"gpt", "bing", "chatgpt", "bard"}:
                    session.append({"role": "assistant", "content": turn["value"]})
                elif turn["from"] == "system":
                    session.append({"role": "system", "content": turn["value"]})
                else:
                    raise RuntimeError(f"Unexpected \"from\" field: {turn['from']}")
            data.append(session)
        return data


class ChituTraceDataset(Dataset):
    """
    Trace of chitu generated by setting `debug.save_trace_dir` to chitu.
    """

    @override
    def load_data(self) -> list[list[dict]]:
        data: list[list[dict]] = []
        with open(self.dataset_path, encoding="utf-8") as f:
            for line in f:
                raw_session = json.loads(line.strip())
                session: list[dict] = []
                for turn in raw_session["message"]:
                    session.append({"role": turn["role"], "content": turn["content"]})
                data.append(session)
        return data


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
            except Exception:
                print(
                    "Local tokenizer is not loaded, trying tokenizer interface from server if available."
                )

        if not self.tokenizer:
            try:
                test_prompt = "tokenizer test message"
                test_token_ids = self.remote_tokenize(test_prompt)
                self.remote_detokenize(test_token_ids)
            except Exception:
                raise Exception(
                    "Tokenizer interface unavailable, please install transformers and set local tokenizer using --tokenizer-path."
                )

            self.tokenize = self.remote_tokenize
            self.detokenize = self.remote_detokenize

        if self.config.dataset != "random":
            try:
                if self.config.dataset_path is None:
                    raise ValueError(
                        '--dataset-path must be set for using --dataset other than "random"'
                    )
                if self.config.dataset == "sharegpt":
                    self.dataset: Dataset = ShareGPTDataset(self.config.dataset_path)
                elif self.config.dataset == "chitu-trace":
                    self.dataset: Dataset = ChituTraceDataset(self.config.dataset_path)
                else:
                    raise ValueError(f"Unknown dataset {self.config.dataset}")
            except Exception as e:
                raise RuntimeError(
                    f"Unable to load dataset {self.config.dataset} from {self.config.dataset_path}"
                ) from e

        # Validate configuration
        if config.max_concurrency is not None and config.max_concurrency < 1:
            raise ValueError("Max concurrency must be at least 1")
        if config.num_requests < 1:
            raise ValueError("Number of requests must be at least 1")
        if config.warmup_requests < 0:
            raise ValueError("Number of warmup requests cannot be negative")
        if config.warmup_iterations < 0:
            raise ValueError("Number of warmup iterations cannot be negative")
        if config.request_rate <= 0 and config.request_rate != float("inf"):
            raise ValueError("Request rate must be positive or inf")
        if config.request_interval < 0:
            raise ValueError("Request interval cannot be negative")
        if config.min_batch_size is not None and config.min_batch_size < 1:
            raise ValueError("Forced min_batch_size must be at least 1")
        if config.min_batch_size is not None and config.max_concurrency is not None:
            assert (
                config.min_batch_size <= config.max_concurrency
            ), "--force-min-bs must not exceed --max-concurrency"
        if config.num_iterations < 1:
            raise ValueError("Number of iterations must be at least 1")
        # Print configuration
        print(f"Benchmark Configuration:")
        print(f"  Model: {config.model_name}")
        print(f"  Max Concurrency: {config.max_concurrency}")
        print(
            "  Forced min_batch_size: "
            f"{config.min_batch_size if config.min_batch_size is not None else 'disabled'}"
        )
        print(f"  Requests per Iteration: {config.num_requests}")
        print(f"  Iterations: {config.num_iterations}")
        print(f"  Warmup Requests: {config.warmup_requests}")
        print(f"  Warmup Iterations: {config.warmup_iterations}")
        print(f"  Input Length: {config.input_length}")
        print(f"  Output Length: {config.output_length}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Top P: {config.top_p}")
        print(f"  Top K: {config.top_k}")
        print(f"  Request Rate: {config.request_rate} req/s")
        print(f"  Request Interval: {config.request_interval} s")
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
        prompt = data.get("prompt") or data.get("text", "")

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
        most_recent_timestamp = st

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
                else:
                    output.success = False
                    response_text = await response.text()
                    output.error = (
                        f"HTTP {response.status} from "
                        f"{self.base_url}/v1/chat/completions: {response_text}"
                    )
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
            if self.config.input_length is None:
                raise ValueError(
                    "--input-length <int> is required when using --dataset random"
                )
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
        elif self.config.dataset in {"sharegpt", "chitu-trace"}:
            if self.config.input_length is None:
                return self.dataset.next_single_content()
            else:
                return self.dataset.sample_single_content(
                    self.tokenize, self.detokenize, self.config.input_length
                )
        else:
            raise Exception("args.dataset only supports random or sharegpt")

    def _build_payload(self, messages: list[dict]) -> Dict[str, Any]:
        """Build one OpenAI-compatible streaming request payload."""
        return {
            "model": self.config.model_name,
            "messages": messages,
            "max_completion_tokens": self.config.output_length,
            "max_tokens": self.config.output_length,
            "stream": True,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "min_batch_size": self.config.min_batch_size or 1,
            "stop_with_eos": self.config.stop_with_eos,
            "ignore_eos": not self.config.stop_with_eos,
            "stream_options": {"include_usage": True},
        }

    async def _get_requests(
        self,
        payloads: list[Dict[str, Any]],
        request_rate: float,
    ) -> AsyncGenerator[tuple[int, Dict[str, Any]], None]:
        """Yield requests at the specified rate.

        ``inf`` submits all requests as quickly as possible. A finite rate uses
        a Poisson process to sample arrival intervals.
        """
        for request_id, payload in enumerate(payloads):
            yield request_id, payload
            if request_rate == float("inf"):
                continue
            await asyncio.sleep(np.random.exponential(1.0 / request_rate))

    async def run_async(
        self,
        payloads: list[Dict[str, Any]],
        request_rate: Optional[float] = None,
    ) -> list[RequestFuncOutput]:
        """Run requests with specified rate and concurrency control.

        ``request_rate`` controls request arrival. ``max_concurrency`` limits
        in-flight requests.
        """
        if not payloads:
            return []

        rate = self.config.request_rate if request_rate is None else request_rate
        semaphore = (
            asyncio.Semaphore(self.config.max_concurrency)
            if self.config.max_concurrency is not None
            else None
        )

        connector = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(
            connector=connector,
            trust_env=True,
            timeout=AIOHTTP_TIMEOUT,
        ) as session:

            async def limited_request(payload: Dict[str, Any]) -> RequestFuncOutput:
                if semaphore is None:
                    return await self._async_run_inference(session, payload)
                async with semaphore:
                    return await self._async_run_inference(session, payload)

            tasks: list[asyncio.Task[RequestFuncOutput]] = []
            async for _, payload in self._get_requests(payloads, rate):
                tasks.append(asyncio.create_task(limited_request(payload)))

            return await asyncio.gather(*tasks)

    def warmup_requests(self) -> None:
        """Run request-level warmup without including it in benchmark results."""
        if self.config.warmup_requests > 0:
            print(f"Warming up with {self.config.warmup_requests} requests...")
            warmup_payloads = [
                self._build_payload(self._get_test_messages())
                for _ in range(self.config.warmup_requests)
            ]
            warmup_outputs = asyncio.run(
                self.run_async(warmup_payloads, request_rate=float("inf"))
            )
            if not any(output.success for output in warmup_outputs):
                raise RuntimeError(
                    "Warmup failed. Check the benchmark arguments and server. "
                    f"First error: {warmup_outputs[0].error}"
                )
        else:
            print("Skipping warmup.")

    def warmup_iteration(self, iteration: int) -> None:
        """Run one full, unreported benchmark iteration as warmup."""
        print(
            f"Starting warmup iteration "
            f"{iteration}/{self.config.warmup_iterations}..."
        )
        outputs, _ = self.benchmark()
        if not any(output.success for output in outputs):
            raise RuntimeError(
                "Warmup iteration failed. Check the benchmark arguments and "
                f"server. First error: {outputs[0].error}"
            )

    def benchmark(self):
        print(
            f"Running {self.config.num_requests} requests at "
            f"request_rate={self.config.request_rate}, "
            f"max_concurrency={self.config.max_concurrency}..."
        )
        benchmark_payloads = [
            self._build_payload(self._get_test_messages())
            for _ in range(self.config.num_requests)
        ]

        start_time = time.perf_counter()
        outputs = asyncio.run(self.run_async(benchmark_payloads))
        total_time = time.perf_counter() - start_time

        return outputs, total_time


def calculate_metrics(
    outputs: list[RequestFuncOutput],
    dur_s: float,
    selected_percentiles: list[float],
    config: Optional[BenchmarkConfig] = None,
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
    successful_outputs = [out for out in outputs if out.success]
    events: list[tuple[float, int]] = []
    for out in successful_outputs:
        events.append((out.start_timestamp, 1))
        events.append((out.start_timestamp + out.latency, -1))
    # stamp  delta  active    max
    # 0.0     +1      1        1
    # 0.5     +1      2        2
    # 1.0     -1      1        2
    active_requests = 0
    max_concurrent_requests = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active_requests += delta
        max_concurrent_requests = max(max_concurrent_requests, active_requests)

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
        max_itl_ms=np.max(itls or 0) * 1000,
        percentiles_itl_ms=[
            (p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
        ],
        concurrency=sum(e2els) / dur_s,
        max_concurrent_requests=max_concurrent_requests,
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
    include_max: bool = False,
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
    if include_max:
        max_value = getattr(metrics, f"max_{metric_attribute_name}_ms")
        print("{:<40} {:<10.2f}".format(f"Max {metric_name} (ms):", max_value))
        result[f"max_{metric_attribute_name}_ms"] = max_value


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
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help=(
            "Maximum number of concurrent requests. Together with "
            "--request-rate, this controls fixed-concurrency serving tests."
        ),
    )
    parser.add_argument(
        "--num-requests",
        dest="num_requests",
        type=int,
        default=None,
        help=(
            "Number of requests to process in each iteration. Defaults to "
            "max_concurrency."
        ),
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=None,
        help=(
            "Total warmup requests before all iterations. Defaults to "
            "max_concurrency, or 1 when max concurrency is not set."
        ),
    )
    parser.add_argument(
        "--warmup-iteration",
        type=int,
        default=0,
        help=(
            "Number of complete benchmark iterations to run before collecting "
            "results."
        ),
    )
    parser.add_argument("--input-len", type=int)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
    )
    parser.add_argument("--output-dir", help="Output dir of benchmark json file")
    parser.add_argument("--base-url", help="URL of the Chitu server endpoint")
    parser.add_argument("--metric-percentiles", type=str, default="90,95,99")
    parser.add_argument("--percentile-metrics", type=str, default="ttft,tpot,itl")
    parser.add_argument("--append-result", action="store_true")
    parser.add_argument(
        "--dataset", default="random", choices=["random", "sharegpt", "chitu-trace"]
    )
    parser.add_argument("--dataset-path")
    parser.add_argument("--tokenizer-path")
    request_rate_group = parser.add_mutually_exclusive_group()
    request_rate_group.add_argument(
        "--request-rate",
        type=float,
        default=None,
        help=(
            "Number of requests per second. If inf, requests are submitted as "
            "quickly as possible; otherwise, inter-arrival times follow a "
            "Poisson process."
        ),
    )
    request_rate_group.add_argument(
        "--request-interval",
        type=float,
        default=None,
        help=(
            "Mean interval in seconds between requests. This is the reciprocal "
            "of --request-rate; 0 submits requests as quickly as possible."
        ),
    )
    parser.add_argument(
        "--force-min-bs",
        type=int,
        metavar="BATCH_SIZE",
        default=None,
        help="Force the server request field min_batch_size to this value.",
    )
    parser.add_argument("--stop-with-eos", action="store_true", default=False)
    parser.add_argument("--print-generated", action="store_true")

    args = parser.parse_args()

    max_concurrency = args.max_concurrency
    num_requests = (
        args.num_requests if args.num_requests is not None else (max_concurrency or 1)
    )
    warmup_requests = (
        args.warmup_requests
        if args.warmup_requests is not None
        else (max_concurrency or 1)
    )
    if args.request_rate is not None:
        if args.request_rate <= 0 and args.request_rate != float("inf"):
            parser.error("--request-rate must be positive or inf")
        request_rate = args.request_rate
    elif args.request_interval is not None:
        if args.request_interval < 0:
            parser.error("--request-interval cannot be negative")
        request_rate = (
            float("inf") if args.request_interval == 0 else 1 / args.request_interval
        )
    else:
        request_rate = float("inf")
    request_interval = 0.0 if request_rate == float("inf") else 1 / request_rate

    config = BenchmarkConfig(
        model_name=args.model,
        max_concurrency=max_concurrency,
        num_requests=num_requests,
        num_iterations=args.iterations,
        warmup_requests=warmup_requests,
        warmup_iterations=args.warmup_iteration,
        input_length=args.input_len,
        output_length=args.output_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        dataset=args.dataset,
        tokenizer_path=args.tokenizer_path,
        dataset_path=args.dataset_path,
        request_rate=request_rate,
        request_interval=request_interval,
        min_batch_size=args.force_min_bs,
        stop_with_eos=args.stop_with_eos,
        print_generated=args.print_generated,
    )

    runner = BenchmarkServing(config, base_url=args.base_url)

    runner.warmup_requests()

    for iteration in range(1, config.warmup_iterations + 1):
        runner.warmup_iteration(iteration)

    for iteration in range(1, config.num_iterations + 1):
        print(
            f"\nStarting benchmark iteration " f"{iteration}/{config.num_iterations}..."
        )
        outputs, total_time = runner.benchmark()
        metrics, actual_output_lens = calculate_metrics(
            outputs=outputs,
            dur_s=total_time,
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            config=config,
        )
        report_benchmark_result(
            outputs=outputs,
            total_time=total_time,
            metrics=metrics,
            actual_output_lens=actual_output_lens,
            config=config,
            args=args,
            iteration=iteration,
        )


def report_benchmark_result(
    outputs: list[RequestFuncOutput],
    total_time: float,
    metrics: BenchmarkMetrics,
    actual_output_lens: list[int],
    config: BenchmarkConfig,
    args: argparse.Namespace,
    iteration: int,
) -> None:
    print(
        "{s:{c}^{n}}".format(
            s=f" Serving Benchmark Result ({iteration}/{config.num_iterations}) ",
            n=50,
            c="=",
        )
    )
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
    print("{:<40} {:<10.2f}".format("Concurrency:", metrics.concurrency))
    print(
        "{:<40} {:<10}".format(
            "Max concurrent requests:", metrics.max_concurrent_requests
        )
    )

    model = {"name": config.model_name}

    result = {
        "model": model,
        "max_concurrency": config.max_concurrency,
        "min_batch_size": config.min_batch_size,
        "num_requests": config.num_requests,
        "iteration": iteration,
        "num_iterations": config.num_iterations,
        "warmup_requests": config.warmup_requests,
        "warmup_iterations": config.warmup_iterations,
        "request_rate": config.request_rate,
        "request_interval": config.request_interval,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "duration": total_time,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "concurrency": metrics.concurrency,
        "max_concurrent_requests": metrics.max_concurrent_requests,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "start_timestamps": [output.start_timestamp for output in outputs],
        "response_timestamp": [output.response_timestamp for output in outputs],
    }

    if config.print_generated:
        print("{s:{c}^{n}}".format(s=" Generated Texts ", n=50, c="="))
        for i, output in enumerate(outputs):
            print(f"\n--- Request {i} (success={output.success}) ---")
            print(output.generated_text)
        print("=" * 50)

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
        include_max=True,
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
