# SPDX-FileCopyrightText: 2025 vLLM Team
# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0
#
# The benchmark logic is partially adapted from vLLM's benchmark_dataset.py,
#  licensed under Apache 2.0. This adaption aims to follow widely-used
#  benchmarking practices for LLM inference throughput and latency.

import os
import time
import random
import logging
import json
import hydra
import torch
from datetime import datetime

from logging import getLogger
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from chitu.task import UserRequest, TaskPool, Task
from chitu.chitu_main import (
    chitu_init,
    chitu_run,
    chitu_start,
    chitu_terminate,
    chitu_is_terminated,
    warmup_engine,
)
from chitu.global_vars import get_timers
from chitu.schemas import ServeConfig
from chitu.utils import get_config_dir_path, try_get_profiler

logger = getLogger(__name__)

counter = 1


def gen_sequential_id(len=8):
    global counter
    req_id = f"{counter:0{len}x}"
    counter += 1
    return req_id


class ShareGPTDataset:
    """
    Implements the ShareGPT dataset.  Loads data from a JSON file and generates
    sample requests based on conversation turns.
    """

    def __init__(self, dataset_path: str, random_seed: int = 42) -> None:
        self.dataset_path = dataset_path
        self.random_seed = random_seed
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = json.load(f)
        # Filter entries with at least two conversation turns.
        self.data = [
            entry
            for entry in self.data
            if "conversations" in entry and len(entry["conversations"]) >= 2
        ]
        random.seed(self.random_seed)
        random.shuffle(self.data)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        input_len: int,
        max_new_tokens: int,
    ) -> list:
        reqs: list = []
        n = len(self.data)
        if n == 0:
            raise ValueError("dataset is empty after filtering.")

        idx = 0
        passes = 0
        collected_at_last_pass = (
            -1
        )  # progress guard for the case of zero eligible entries

        # Allow wrap-around (duplicate sampling) until reaching num_requests,
        # with a guard: if a full pass yields no additional eligible samples, stop early.
        while len(reqs) < num_requests:
            entry = self.data[idx]
            prompt = entry["conversations"][0]["value"]

            prompt_ids = tokenizer(
                prompt, max_length=input_len, truncation=True, add_special_tokens=False
            ).input_ids

            if len(prompt_ids) >= input_len:
                reqs.append(
                    UserRequest(
                        message=None,
                        request_id=gen_sequential_id(),
                        tokens=prompt_ids,
                        max_new_tokens=max_new_tokens,
                    )
                )

            idx += 1
            if idx == n:
                # Completed one pass; if no progress, we cannot satisfy the request count.
                passes += 1
                if len(reqs) == collected_at_last_pass:
                    logger.warning(
                        f"ShareGPTDataset: only sampled {len(reqs)} requests (requested {num_requests}); "
                        f"not enough entries meeting input_len={input_len}. Consider using dataset='random' "
                        f"or reducing input_len."
                    )
                    break
                collected_at_last_pass = len(reqs)
                # Deterministic reshuffle per pass to provide variety while keeping reproducibility
                random.seed(self.random_seed + passes)
                random.shuffle(self.data)
                idx = 0

        return reqs


def random_requests(args, num_reqs):
    vocab_size = args.models.vocab_size
    prompt_len = args.benchmark.input_len
    max_new_tokens = args.benchmark.output_len

    reqs = []
    for _ in range(num_reqs):
        req = UserRequest(
            message=None,
            request_id=gen_sequential_id(),
            tokens=[random.randint(0, vocab_size - 1) for _ in range(prompt_len)],
            max_new_tokens=max_new_tokens,
        )
        reqs.append(req)
    return reqs


def get_requests(args, num_reqs) -> list[UserRequest]:
    if args.benchmark.dataset == "random":
        return random_requests(args, num_reqs)
    else:
        dataset = ShareGPTDataset(args.benchmark.dataset_path)
        return dataset.sample(
            AutoTokenizer.from_pretrained(
                args.models.tokenizer_path, trust_remote_code=True
            ),  # [TODO] support tokenizer other than hf
            num_requests=num_reqs,
            input_len=args.benchmark.input_len,
            max_new_tokens=args.benchmark.output_len,
        )


def count_num_tokens(reqs: list[UserRequest]):
    output_num_tokens = sum(req.async_stream.tokens_len for req in reqs)
    total_num_tokens = sum(req.prompt_len for req in reqs) + output_num_tokens
    return output_num_tokens, total_num_tokens


def run_benchmark(args, timers, is_main_rank, rank):
    warmup_engine(args)
    iters = args.benchmark.iters
    num_reqs_list = args.benchmark.num_reqs_list
    stop_with_eos = args.benchmark.stop_with_eos
    debug_print = args.benchmark.debug_print
    profiler_dir = args.benchmark.profile_dir
    with_stack = args.benchmark.profiler_with_stack

    profiler = None
    if profiler_dir is not None:
        time_str = datetime.now().strftime("%m%d_%H%M")
        profiler_dir = os.path.join(profiler_dir, f"profiler_{time_str}")
        os.makedirs(profiler_dir, exist_ok=True)
        profiler = try_get_profiler(
            profiler_dir, warmup=1, active=1, repeat=1, with_stack=with_stack
        )
        profiler.start()

    for num_reqs in num_reqs_list:
        for i in range(iters):
            chitu_start()
            if is_main_rank:
                reqs = get_requests(args, num_reqs)
                for req in reqs:
                    TaskPool.add(Task(req.request_id, req, stop_with_eos=stop_with_eos))
            t_start = time.perf_counter()
            while not chitu_is_terminated():
                chitu_run()
                if is_main_rank and TaskPool.all_finished():
                    break
            t_end = time.perf_counter()

            if is_main_rank:
                total_time = t_end - t_start
                output_num_tokens, total_num_tokens = count_num_tokens(reqs)
                logger.info(
                    f"------ NumReqs {num_reqs} Iter {i + 1} ------\n"
                    f"Time cost: {total_time:.4f} sec\n"
                    f"Total token: {total_num_tokens}\n"
                    f"Total output token: {output_num_tokens}\n"
                    f"Total throughput: {total_num_tokens / total_time:.2f} tps\n"
                    f"Total output throughput: {output_num_tokens / total_time:.2f} tps\n"
                )
                if debug_print:
                    logger.info(f"First request output: {reqs[0].output}\n")

            if profiler:
                profiler.step()

            chitu_terminate()  # terminate in iters loop

        torch.cuda.empty_cache()

    if profiler:
        profiler.stop()


def adjust_benchmark_args(args):
    if args.benchmark.profile_dir is not None:
        args.benchmark.output_len = 5  # only decode 5 tokens for profiling
        num_reqs_list = args.benchmark.num_reqs_list
        if len(num_reqs_list) > 1:
            args.benchmark.num_reqs_list = num_reqs_list[-1:]
            logger.info(
                f"Adjust num_reqs_list to {args.benchmark.num_reqs_list} for profiling"
            )


@hydra.main(
    version_base=None,
    config_path=os.getenv("CONFIG_PATH", get_config_dir_path()),
    config_name=os.getenv("CONFIG_NAME", "serve_config"),
)
def main(args: ServeConfig):
    logger.setLevel(logging.DEBUG)
    logger.info(f"Run with args: {args}")

    adjust_benchmark_args(args)

    chitu_init(args)
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    timers = get_timers()
    rank = torch.distributed.get_rank()
    is_main_rank = rank == 0
    run_benchmark(args, timers, is_main_rank, rank)
    logger.info("Waiting for all ranks to finish...")
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])


if __name__ == "__main__":
    main()

    # Sometimes torch.distributed will hang during destruction if CUDA graph is enabled.
    # As a workaround, we `exec` a dummy process to kill the current process, without
    # returning an error.
    # Don't exec bash because it loads startup scripts
    os.execl("/usr/bin/true", "true")  # /usr/bin/true does nothing but exits
