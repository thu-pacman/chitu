import hydra
import torch
import time
import os
import sys
import random
import logging
from logging import getLogger
from pathlib import Path

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


def use_long_context_msgs(model_name: str) -> bool:
    return "DeepSeek-V3" in model_name or "DeepSeek-V4" in model_name


# --- “电话本”测试 ---------------------------------
# This builds a phone book far longer than index_topk and asks for one
# specific entry's number. The correct answer can only be produced if the
# indexer selects the right tokens, so a wrong answer flags an indexer
# regression. Enabled by setting the ``CHITU_TEST_NEEDLE`` env var.
_BASE_NAMES = [
    "张三",
    "李四",
    "王五",
    "赤兔",
    "八卦炉",
    "刘备",
    "关羽",
    "张飞",
    "赵云",
    "曹操",
    "曹丕",
    "曹植",
    "吕布",
    "貂蝉",
    "孙尚香",
    "孙权",
]


def gen_phonebook_prompt(num_entries: int, exp_idx: int, seed: int = 1234):
    """Build a phone-book prompt and return ``(prompt, expected_number)``.

    ``num_entries`` should be large enough that the tokenized prompt exceeds
    ``index_topk`` (2048 for DeepSeek-V3.2), forcing the indexer's top-k to
    actually prune. ``exp_idx`` picks which entry is asked about.
    """
    rng = random.Random(seed)
    entries = []
    numbers = []
    used = set()
    for i in range(num_entries):
        name = f"{_BASE_NAMES[i % len(_BASE_NAMES)]}{i:04d}"
        # 11-digit phone number, unique per entry.
        while True:
            number = "1" + "".join(str(rng.randint(0, 9)) for _ in range(10))
            if number not in used:
                used.add(number)
                break
        numbers.append(number)
        entries.append(f"{name}\t{number}")

    needle_name = f"{_BASE_NAMES[exp_idx % len(_BASE_NAMES)]}{exp_idx:04d}"
    expected_number = numbers[exp_idx]

    body = "\n".join(entries)
    prompt = (
        "下面是一份电话簿，每行是一个联系人的姓名和电话号码，用制表符分隔。\n"
        "请仔细阅读，然后回答末尾的问题。\n\n"
        f"{body}\n\n"
        f"问题：{needle_name} 的电话号码是多少？请只输出这一串数字，不要输出其它内容。"
    )
    return prompt, expected_number


def phonebook_test_enabled() -> bool:
    """Whether to run the phone-book needle request instead of the usual msgs."""
    return os.environ.get("CHITU_TEST_PHONEBOOK", "false") == "true"


def check_phonebook_test_results(reqs):
    """Assert every request's output contains its expected number.

    Raises AssertionError on a miss so the failure propagates to a nonzero
    exit code and fails CI.
    """
    misses = []
    for i, req in enumerate(reqs):
        expected = getattr(req, "_needle_expected", None)
        if expected is None:
            continue
        output = req.output or ""
        hit = expected in output
        logger.info(
            f"[needle] req[{i}] expected={expected} hit={hit} output={output!r}"
        )
        if not hit:
            misses.append((i, expected, output))
    if misses:
        raise AssertionError(
            "phone-book needle test failed; the indexer likely selected the "
            f"wrong tokens. misses={misses}"
        )


USE_TOOLS = False
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
    for i in range(num_reqs):
        if USE_TOOLS:
            req_data = msg_tools[i % len(msg_tools)]
            msg = req_data["messages"]
            req = UserRequest.create(
                msg,
                f"{gen_req_id()}",
                max_new_tokens=max_new_tokens,
                frequency_penalty=frequency_penalty,
                temperature=1,
                tools=req_data["tools"],
            )
        elif is_vl:
            msg = msgs_vl[i % len(msgs_vl)]
            req = UserRequest.create(
                msg,
                f"{gen_req_id()}",
                max_new_tokens=max_new_tokens,
                frequency_penalty=frequency_penalty,
                temperature=1,
            )
        else:
            msg = msgs[i % len(msgs)]
            req = UserRequest.create(
                msg,
                f"{gen_req_id()}",
                max_new_tokens=max_new_tokens,
                frequency_penalty=frequency_penalty,
                temperature=1,
            )
        req.messages = msg
        reqs.append(req)
    return reqs


def gen_reqs_needle(num_reqs, max_new_tokens):
    """Generate phone-book test requests.

    Uses greedy decoding (temperature 0) and a large phone book
    so the tokenized prompt exceeds `index_topk` and the
    indexer's top-k actually prunes.
    """
    num_entries = 500
    reqs: list[UserRequest] = []
    for i in range(num_reqs):
        exp_idx = (i * 7 + 3) % num_entries
        prompt, expected = gen_phonebook_prompt(num_entries, exp_idx, seed=1234 + i)
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
        reqs.append(req)
    return reqs


def gen_reqs(num_reqs, max_new_tokens, frequency_penalty, is_vl=False):
    global local_args, msgs
    if phonebook_test_enabled():
        return gen_reqs_needle(num_reqs, max_new_tokens)

    if (
        use_long_context_msgs(local_args.models.name)
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

    needle_reqs = []
    for i in range(2):
        chitu_start()
        if rank == 0:
            reqs = gen_reqs(
                num_reqs=args.infer.max_batch_size,
                max_new_tokens=args.request.max_new_tokens,
                frequency_penalty=args.request.frequency_penalty,
                is_vl=hasattr(args.models, "vision_config")
                and not args.infer.language_model_only,
            )
            for req in reqs:
                TaskPool.add(Task(req.request_id, req, stop_with_eos=True))
            logger.info(f"------ batch {i} ------")
            t_start = time.perf_counter()
            timers("overall").start()

        tokens = 0
        while not chitu_is_terminated():
            tokens += 1
            chitu_run()
            if rank == 0 and TaskPool.all_finished():
                break  # Rank 0 can temporarily leave to do other things

        if rank == 0:
            timers("overall").stop()
            t_end = time.perf_counter()
            logger.info(f"Tokens generate : {tokens}")
            logger.info(f"Time cost {t_end - t_start}")
            logger.info(
                f"max GPU memory used: {torch.cuda.max_memory_allocated() / 1024**3} GB"
            )

            for i, req in enumerate(reqs):
                if sys.stdout.isatty():
                    GRAY = "\033[1;30m"
                    RESET = "\033[0m"
                else:
                    GRAY = ""
                    RESET = ""
                logger.info(
                    f"Response in rank {rank}: reqs[{i}].output={req.output}, {GRAY}reqs[{i}].input={req.messages}{RESET}"
                )

            if phonebook_test_enabled():
                needle_reqs.extend(reqs)
            timers.log()
        chitu_terminate()

    if rank == 0 and phonebook_test_enabled():
        check_phonebook_test_results(needle_reqs)


def run_normal(args, timers):
    rank = torch.distributed.get_rank()
    warmup_engine(args)

    needle_reqs = []
    for i in range(2):
        reqs = gen_reqs(
            num_reqs=args.infer.max_batch_size,
            max_new_tokens=args.request.max_new_tokens,
            frequency_penalty=args.request.frequency_penalty,
            is_vl=hasattr(args.models, "vision_config")
            and not args.infer.language_model_only,
        )
        for req in reqs:
            TaskPool.add(Task(req.request_id, req, stop_with_eos=True))
        logger.info(f"------ batch {i} ------")
        t_start = time.time()
        timers("overall").start()
        tokens = 0
        while len(TaskPool.pool) > 0:
            tokens += 1
            chitu_run()
        while not TaskPool.all_finished():
            # no new token generated, but chitu is not terminated
            chitu_run()

        print("GPU memory used : ", torch.cuda.memory_allocated())
        timers("overall").stop()
        t_end = time.time()
        logger.info(f"Tokens generate : {tokens}")
        logger.info(f"Time cost {t_end - t_start}")

        for i, req in enumerate(reqs):
            logger.info(f"Response in rank {rank}: reqs[{i}].output={req.output}")

        if phonebook_test_enabled():
            needle_reqs.extend(reqs)
        timers.log()

    if phonebook_test_enabled():
        check_phonebook_test_results(needle_reqs)


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
