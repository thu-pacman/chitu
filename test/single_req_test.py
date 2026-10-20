import hydra
import torch
import time
import os
import sys
import logging
from logging import getLogger
from pathlib import Path

from chitu.task import UserRequest, TaskPool, Task
from chitu.metrics.prometheus_collector import PrometheusMetricsCollector
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
from chitu.testing.phonebook import (
    SHARED_SEED,
    check_phonebook_test_results,
    check_prefix_cache_hit,
    expected_pairs,
    gen_phonebook_prompt,
    phonebook_mode_active,
    phonebook_num_entries,
    phonebook_shared_enabled,
    phonebook_test_enabled,
)
from chitu.utils import get_config_dir_path, gen_req_id, get_chitu_env

logger = getLogger(__name__)

msgs = [
    [{"role": "user", "content": "宫保鸡丁怎么做?"}],
    [{"role": "user", "content": "what is the recipe of Kung Pao chicken?"}],
    [{"role": "user", "content": "怎么写程序?"}],
    [{"role": "user", "content": "飞机在对流层还是平流层飞?"}],
    [{"role": "user", "content": "怎么避免加班?"}],
    [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
]
# long-context test
msgs_long = [
    [
        {
            "role": "user",
            "content": (Path(__file__).parent / "test_texts/test_text.txt").read_text(
                encoding="utf-8"
            ),
        }
    ],
]
msgs_vl = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请描述这张图片的内容"},
                {"type": "image", "image": "test/test_images/test.jpg"},
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请描述这张图片的内容"},
                {"type": "image", "image": "test/test_images/test2.jpg"},
            ],
        }
    ],
]


def use_long_context_msgs(model_type: str) -> bool:
    return model_type in {"deepseek-v3", "deepseek-v4", "glm-5-2"}


# --- “电话本”测试 ---------------------------------
# The prompt builders and result checkers live in chitu/testing/phonebook.py so
# that the PD disaggregation harness (chitu/testing/pd_utils.py) can reuse them.
# Set CHITU_TEST_PHONEBOOK=true for the needle test (one phone book per request),
# or CHITU_TEST_PHONEBOOK_SHARED=true to give every request the same long body so
# the prefix cache must hit (see chitu/testing/phonebook.py for the details).

USE_TOOLS = False
# Exercise text generation for multimodal models without changing model loading.
USE_TEXT_PROMPTS = os.environ.get("CHITU_TEST_TEXT_PROMPTS", "false") == "true"
msg_tools = [
    {
        "messages": [
            {
                "role": "user",
                "content": f"hows the weather in {loc}? use the tool to get it",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ],
        "tool_choice": "required",
        "max_tokens": 1024,
    }
    for loc in ["beijing", "shanghai", "guangzhou", "shenzhen"]
]


counter = 1


def gen_debug_req_id(len=8):
    global counter
    req_id = f"{counter:0{len}x}"
    counter += 1
    return req_id


def gen_reqs_fake(num_reqs, prompt_len, max_new_tokens, frequency_penalty):
    from chitu.backend import Backend

    reqs: list[UserRequest] = []
    for i in range(num_reqs):
        req = UserRequest.create_mock(
            input_len=prompt_len,
            request_id=f"{gen_req_id()}",
            max_new_tokens=max_new_tokens,
            frequency_penalty=frequency_penalty,
            random_tokens=True,
        )
        req.messages = Backend.tokenizer.decode(req.prompt_tokens)
        reqs.append(req)
    return reqs


def gen_reqs_real(num_reqs, max_new_tokens, frequency_penalty, is_vl=False):
    reqs: list[UserRequest] = []
    temperature = float(os.environ.get("CHITU_TEST_TEMPERATURE", "1"))
    for i in range(num_reqs):
        if USE_TOOLS:
            req_data = msg_tools[i % len(msg_tools)]
            msg = req_data["messages"]
            req = UserRequest.create(
                msg,
                f"{gen_req_id()}",
                max_new_tokens=max_new_tokens,
                frequency_penalty=frequency_penalty,
                temperature=temperature,
                tools=req_data["tools"],
            )
        elif is_vl:
            msg = msgs_vl[i % len(msgs_vl)]
            req = UserRequest.create(
                msg,
                f"{gen_req_id()}",
                max_new_tokens=max_new_tokens,
                frequency_penalty=frequency_penalty,
                temperature=temperature,
            )
        else:
            msg = msgs[i % len(msgs)]
            req = UserRequest.create(
                msg,
                f"{gen_req_id()}",
                max_new_tokens=max_new_tokens,
                frequency_penalty=frequency_penalty,
                temperature=temperature,
            )
        req.messages = msg
        reqs.append(req)
    return reqs


def _make_phonebook_req(prompt: str, expected: str, max_new_tokens: int) -> UserRequest:
    msg = [{"role": "user", "content": prompt}]
    req = UserRequest.create(
        msg,
        f"{gen_req_id()}",
        max_new_tokens=max_new_tokens,
        frequency_penalty=0.0,
        temperature=0,
    )
    req.messages = msg
    req._needle_expected = expected
    return req


def gen_reqs_needle(num_reqs, max_new_tokens):
    """Generate phone-book test requests.

    Uses greedy decoding (temperature 0) and a large phone book
    so the tokenized prompt exceeds `index_topk` and the
    indexer's top-k actually prunes. Each request gets its own phone book
    (`seed=1234+i`), so nothing here is reusable from the prefix cache.
    """
    num_entries = phonebook_num_entries()
    reqs: list[UserRequest] = []
    for i in range(num_reqs):
        exp_idx = (i * 7 + 3) % num_entries
        prompt, expected = gen_phonebook_prompt(
            num_entries, exp_idx, seed=SHARED_SEED + i
        )
        reqs.append(_make_phonebook_req(prompt, expected, max_new_tokens))
    return reqs


#: Advances across batches so the second batch asks about other entries than the
#: first one while keeping the phone-book body byte-identical.
_shared_exp_counter = 0


def gen_reqs_needle_shared(num_reqs, max_new_tokens):
    """Generate phone-book requests that all share one long prompt body.

    Every request uses the same seed, hence the same phone book; only the final
    question (which entry to look up) differs. The shared body is what the
    prefix cache has to hit, and the per-request expected number is what proves
    the restored KV / linear-attention state is the right one.
    """
    global _shared_exp_counter
    num_entries = phonebook_num_entries()
    reqs: list[UserRequest] = []
    for _ in range(num_reqs):
        exp_idx = (_shared_exp_counter * 7 + 3) % num_entries
        _shared_exp_counter += 1
        prompt, expected = gen_phonebook_prompt(num_entries, exp_idx, seed=SHARED_SEED)
        reqs.append(_make_phonebook_req(prompt, expected, max_new_tokens))
    return reqs


def gen_reqs(num_reqs, max_new_tokens, frequency_penalty, is_vl=False):
    global local_args, msgs
    if phonebook_shared_enabled():
        return gen_reqs_needle_shared(num_reqs, max_new_tokens)
    if phonebook_test_enabled():
        return gen_reqs_needle(num_reqs, max_new_tokens)

    if (
        use_long_context_msgs(local_args.models.type)
        and local_args.infer.max_seq_len >= 4096
        and msgs[: len(msgs_long)] != msgs_long
    ):
        msgs = msgs_long + msgs

    if local_args.request.prompt_tokens_len > 0:
        return gen_reqs_fake(
            num_reqs,
            local_args.request.prompt_tokens_len,
            max_new_tokens,
            frequency_penalty,
        )
    else:
        return gen_reqs_real(num_reqs, max_new_tokens, frequency_penalty, is_vl)


def run_pipe_or_tensor_parallelism(args, timers):
    rank = torch.distributed.get_rank()
    warmup_engine(args)

    needle_reqs: list[tuple[UserRequest, str]] = []
    for i in range(2):
        chitu_start()
        if rank == 0:
            reqs = gen_reqs(
                num_reqs=args.infer.max_batch_size,
                max_new_tokens=args.request.max_new_tokens,
                frequency_penalty=args.request.frequency_penalty,
                is_vl=hasattr(args.models, "vision_config")
                and not args.infer.language_model_only
                and not USE_TEXT_PROMPTS,
            )
            for req in reqs:
                TaskPool.add(Task(req.request_id, req, stop_with_eos=True))
            logger.info(f"------ batch {i} ------")
            t_start = time.perf_counter()
            timers("overall").start()

        steps = 0
        while not chitu_is_terminated():
            steps += 1
            chitu_run()
            if rank == 0 and TaskPool.all_finished():
                break  # Rank 0 can temporarily leave to do other things

        if rank == 0:
            timers("overall").stop()
            t_end = time.perf_counter()
            logger.info(f"Total steps : {steps}")
            logger.info(f"Time cost {t_end - t_start}")
            logger.info(
                f"max GPU memory used: {torch.cuda.max_memory_allocated() / 1024**3} GB"
            )
            # MTP acceptance: from Prometheus counters (same source as metrics_monitor)
            n_drafts = args.infer.mtp_size - 1 if args.infer.mtp_size > 1 else 0
            if n_drafts > 0 and steps > 0:
                proposed, accepted = PrometheusMetricsCollector.get_mtp_stats()
                if proposed > 0:
                    logger.info(
                        f"MTP acceptance: accepted={accepted}/{proposed} "
                        f"({accepted/proposed:.1%} | {accepted/len(reqs):.2f}/req)"
                    )

            for i, req in enumerate(reqs):
                if sys.stdout.isatty():
                    GRAY = "\033[1;30m"
                    RESET = "\033[0m"
                else:
                    GRAY = ""
                    RESET = ""
                logger.info(
                    f"Response in rank {rank}: reqs[{i}].output={req.output}, "
                    f"reqs[{i}].output_len={req.num_output_tokens}, "
                    f"reqs[{i}].finish_reason={req.finish_reason}, "
                    f"{GRAY}reqs[{i}].input={req.messages}{RESET}"
                )

            if phonebook_mode_active():
                needle_reqs.extend(expected_pairs(reqs))
            timers.log()
        chitu_terminate()

    if rank == 0 and phonebook_mode_active():
        check_phonebook_test_results(needle_reqs, context="needle")
        check_prefix_cache_hit(
            needle_reqs,
            required=bool(
                phonebook_shared_enabled() and local_args.infer.enable_prefix_caching
            ),
            context="needle",
        )


def run_normal(args, timers):
    rank = torch.distributed.get_rank()
    warmup_engine(args)

    needle_reqs: list[tuple[UserRequest, str]] = []
    for i in range(2):
        reqs = gen_reqs(
            num_reqs=args.infer.max_batch_size,
            max_new_tokens=args.request.max_new_tokens,
            frequency_penalty=args.request.frequency_penalty,
            is_vl=hasattr(args.models, "vision_config")
            and not args.infer.language_model_only
            and not USE_TEXT_PROMPTS,
        )
        for req in reqs:
            TaskPool.add(Task(req.request_id, req, stop_with_eos=True))
        logger.info(f"------ batch {i} ------")
        t_start = time.time()
        timers("overall").start()
        steps = 0
        while len(TaskPool.pool) > 0:
            steps += 1
            chitu_run()
        while not TaskPool.all_finished():
            # no new token generated, but chitu is not terminated
            chitu_run()

        print("GPU memory used : ", torch.cuda.memory_allocated())
        timers("overall").stop()
        t_end = time.time()
        logger.info(f"Total steps : {steps}")
        logger.info(f"Time cost {t_end - t_start}")
        n_drafts = args.infer.mtp_size - 1 if args.infer.mtp_size > 1 else 0
        if n_drafts > 0 and steps > 0:
            proposed, accepted = PrometheusMetricsCollector.get_mtp_stats()
            if proposed > 0:
                logger.info(
                    f"MTP acceptance: accepted={accepted}/{proposed} "
                    f"({accepted/proposed:.1%} | {accepted/len(reqs):.2f}/req)"
                )

        for i, req in enumerate(reqs):
            logger.info(
                f"Response in rank {rank}: reqs[{i}].output={req.output}, "
                f"reqs[{i}].output_len={req.num_output_tokens}, "
                f"reqs[{i}].finish_reason={req.finish_reason}"
            )

        if phonebook_mode_active():
            needle_reqs.extend(expected_pairs(reqs))
        timers.log()

    if phonebook_mode_active():
        check_phonebook_test_results(needle_reqs, context="needle")
        check_prefix_cache_hit(
            needle_reqs,
            required=bool(
                phonebook_shared_enabled() and args.infer.enable_prefix_caching
            ),
            context="needle",
        )


@hydra.main(
    version_base=None,
    config_path=get_chitu_env(
        "CHITU_CONFIG_PATH", get_config_dir_path(), legacy_names=["CONFIG_PATH"]
    ),
    config_name=get_chitu_env(
        "CHITU_CONFIG_NAME", "serve_config", legacy_names=["CONFIG_NAME"]
    ),
)
def main(args: ServeConfig):
    global local_args
    local_args = args
    logger.setLevel(logging.DEBUG)
    logger.info(f"Run with args: {args}")

    args = chitu_init(args)
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    timers = get_timers()
    logger.debug("finish init")
    if (
        args.infer.pp_size > 1
        or args.infer.tp_size > 1
        or args.infer.dp_size > 1
        or args.infer.pcp_size > 1
    ):
        run_pipe_or_tensor_parallelism(args, timers)
    else:
        run_normal(args, timers)


if __name__ == "__main__":
    main()

    # Sometimes torch.distributed will hang during destruction if CUDA graph is enabled.
    # As a workaround, we `exec` a dummy process to kill the current process, without
    # returning an error.
    logger.info("Waiting for all ranks to finish...")
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    # Don't exec bash because it loads startup scripts
    os.execl("/usr/bin/true", "true")  # /usr/bin/true does nothing but exits
