import hydra
from omegaconf import DictConfig
import torch
import time
import os
import random
import logging
from logging import getLogger
from faker import Faker

from chitu.task import UserRequest, TaskPool, Task
from chitu.chitu_main import (
    chitu_init,
    chitu_run,
    chitu_terminate,
    chitu_is_terminated,
)
from chitu.global_vars import get_timers
from chitu.utils import get_config_dir_path

# -----------utils part begin--------------
import json
import math
import numpy as np


def save_result(data_list, filename="data.json"):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data_list, file, ensure_ascii=False, indent=4)


def load_result(filename="data.json"):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            data_list = json.load(file)
        return data_list
    except FileNotFoundError:
        print(f"文件 {filename} 不存在，无法读取数据。")
        return []


def check_result(result_0_lst, result_1_lst, name_0, name_1, err=[-1, -1]):
    for result_it in range(min(len(result_0_lst), len(result_1_lst))):
        result_0 = result_0_lst[result_it]
        result_1 = result_1_lst[result_it]
        assert (
            result_0["prompt"] == result_1["prompt"]
        ), f"prompt difference in result {result_it}:{result_0['prompt']} vs {result_1['prompt']}"
        for logit_it in range(min(len(result_0["logits"]), len(result_1["logits"]))):
            logit0 = result_0["logits"][logit_it]
            logit1 = result_1["logits"][logit_it]
            for it in range(min(len(logit0), len(logit1))):
                if logit1[it] == 0.0:
                    err[0] = max(err[0], abs(logit0[it] - logit1[it]))
                else:
                    if 0.2 < max(
                        0.0, abs(logit0[it] - logit1[it]) - 5e-2 * abs(logit1[it])
                    ):
                        print(logit0[it], " vs ", logit1[it])
                    # err[1] = max(
                    #    err[1],
                    #    max(0.0, abs(logit0[it] - logit1[it]) - 0.5) / abs(logit1[it]),
                    # )
                    err[0] = max(
                        err[0],
                        max(0.0, abs(logit0[it] - logit1[it]) - 5e-2 * abs(logit1[it])),
                    )
            assert np.allclose(
                logit0, logit1, atol=0.5, rtol=5e-2
            ), f"logits difference in result {result_it}:: logit {logit_it}; aerr:{err[0]}, rerr:{err[1]}"


# -----------utils part end--------------

logger = getLogger(__name__)

msgs = [
    [
        {
            "role": "user",
            "content": "宫保鸡丁怎么做?",
        }
    ],
    [{"role": "user", "content": "what is the recipe of Kung Pao chicken?"}],
    [{"role": "user", "content": "怎么写程序?"}],
    [{"role": "user", "content": "飞机在对流层还是平流层飞?"}],
    [{"role": "user", "content": "怎么避免加班?"}],
    [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
    # [
    #     {"role": "user", "content": "I am going to Paris, what should I see?"},
    #     {
    #         "role": "assistant",
    #         "content": """\
    #     Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:
    #     1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
    #     2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
    #     3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.
    #     These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
    #     },
    #     {"role": "user", "content": "What is so great about #1?"},
    # ],
]

counter = 1


def gen_debug_req_id(len=8):
    global counter
    req_id = f"{counter:0{len}x}"
    counter += 1
    return req_id


def gen_req_id(len=8):
    random_number = random.getrandbits(len * 4)
    hex_string = f"{random_number:0{len}x}"
    # logger.warning(f"generating req {hex_string}")
    return hex_string


def gen_reqs_fake(num_reqs, prompt_len, max_new_tokens):
    from chitu.backend import Backend

    def generate_prompt(token_length, tkn):
        while True:
            tokens = [random.randint(100, 1000) for _ in range(token_length)]
            if len(tkn.encode(tkn.decode(tokens), bos=False, eos=True)) == token_length:
                return tkn.decode(tokens)

    reqs = []
    for i in range(num_reqs):
        msg = generate_prompt(prompt_len - 1, Backend.tokenizer)
        req = UserRequest(msg, f"{gen_req_id()}", max_new_tokens=max_new_tokens)
        reqs.append(req)
    return reqs


def gen_reqs_real(num_reqs, max_new_tokens):
    reqs = []
    for i in range(num_reqs):
        req = UserRequest(
            msgs[i % len(msgs)],
            f"{gen_req_id()}",
            max_new_tokens=max_new_tokens,
            temperature=1,
        )
        reqs.append(req)
    return reqs


def gen_reqs(num_reqs, max_new_tokens):
    global local_args
    if local_args.request.prompt_tokens_len > 0:
        return gen_reqs_fake(
            num_reqs, local_args.request.prompt_tokens_len, max_new_tokens
        )
    else:
        return gen_reqs_real(num_reqs, max_new_tokens)


def run_pipe_or_tensor_parallelism(args, timers, history_result):
    result = []
    result_prompt = []
    result_logits = []
    result_tokens = []
    history_it = 0
    rank = torch.distributed.get_rank()
    for i in range(1):
        if rank == 0:
            reqs = gen_reqs(
                num_reqs=args.infer.max_reqs,
                max_new_tokens=args.request.max_new_tokens,
            )
            for req in reqs:
                req._test_flag = True
                if not history_result == None:
                    req._test_standard_tokens = history_result[history_it]["tokens"]
                    history_it = history_it + 1
                TaskPool.add(Task(f"{req.request_id}", req, req.message))
                result_prompt.append(req.message[0]["content"])
        t_start = time.time()
        timers("overall").start()
        while not chitu_is_terminated():
            chitu_run()
            if rank == 0 and len(TaskPool.pool) == 0:
                break  # Rank 0 can temperarily leave to do other things
        timers("overall").stop()
        t_end = time.time()
        logger.warning(f"Time cost {t_end - t_start}")

        if rank == 0:
            for req in reqs:
                logger.warning(f"Response in rank {rank}: {req.output}")
            result_logits.extend([req._test_logits for req in reqs])
            result_tokens.extend([req._test_tokens for req in reqs])

        timers.log()

    chitu_terminate()

    if rank == 0:
        for it in range(len(result_prompt)):
            prompt = result_prompt[it]
            logits = result_logits[it]
            tokens = result_tokens[it]
            result.append({"prompt": prompt, "logits": logits, "tokens": tokens})
    return result


def run_normal(args, timers, history_result):
    result = []
    result_prompt = []
    result_logits = []
    result_tokens = []
    history_it = 0
    rank = torch.distributed.get_rank()
    for i in range(1):
        reqs = gen_reqs(
            num_reqs=args.infer.max_reqs, max_new_tokens=args.request.max_new_tokens
        )
        for req in reqs:
            req._test_flag = True
            if not history_result == None:
                req._test_standard_tokens = history_result[history_it]["tokens"]
                history_it = history_it + 1
            TaskPool.add(Task(f"{req.request_id}", req, req.message))
            result_prompt.append(req.message[0]["content"])
        t_start = time.time()
        timers("overall").start()
        while len(TaskPool.pool) > 0:
            chitu_run()

        print("GPU memory used : ", torch.cuda.memory_allocated())
        timers("overall").stop()
        t_end = time.time()
        logger.warning(f"Time cost {t_end - t_start}")

        for req in reqs:
            logger.warning(f"Response in rank {rank}: {req.output}")
        result_logits.extend([req._test_logits for req in reqs])
        result_tokens.extend([req._test_tokens for req in reqs])

        timers.log()

    for it in range(len(result_prompt)):
        prompt = result_prompt[it]
        logits = result_logits[it]
        tokens = result_tokens[it]
        result.append({"prompt": prompt, "logits": logits, "tokens": tokens})
    return result


@hydra.main(
    version_base=None,
    config_path=os.getenv("CONFIG_PATH", get_config_dir_path()),
    config_name=os.getenv("CONFIG_NAME", "serve_config"),
)
def main(args: DictConfig):
    global local_args
    local_args = args
    logger.setLevel(logging.DEBUG)
    logger.info(f"Run with args: {args}")

    chitu_init(args, logging_level=logging.INFO)
    timers = get_timers()
    logger.debug(f"finish init")

    rank = torch.distributed.get_rank()

    history_path = os.getenv("HISTORY_PATH", "./example/history/history.txt")
    history_result = None
    if rank == 0:
        if os.path.exists(history_path):
            history_result = load_result(history_path)

    now_result = None
    if args.infer.pp_size > 1 or args.infer.tp_size > 1:
        now_result = run_pipe_or_tensor_parallelism(args, timers, history_result)
    else:
        now_result = run_normal(args, timers, history_result)

    if rank == 0:
        if history_result is not None:
            err = [0, 0]
            check_result(now_result, history_result, "now", "history", err)
            print("!!!!!!!!!aerr_max: ", err[0])
            print("!!!!!!!!!rerr_max: ", err[1])
        else:
            logger.warning(
                "No history result to compare. This is OK for a newly added test case. "
                "Merge this commit to `regression_test_reference` branch to update the "
                "reference result."
            )


if __name__ == "__main__":
    main()

    # Sometimes torch.distributed will hang during destruction if CUDA graph is enabled.
    # As a workaround, we `exec` a dummy process to kill the current process, without
    # returning an error.
    logger.info("Waiting for all ranks to finish...")
    torch.distributed.barrier()
    # Don't exec bash because it loads startup scripts
    os.execl("/usr/bin/echo", "Exiting")  # os.execl rejects "", so print something
