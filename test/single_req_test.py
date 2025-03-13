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
    return hex_string


def gen_reqs_fake(num_reqs, prompt_len, max_new_tokens):
    from chitu.backend import Backend

    def generate_prompt(token_length, tkn):
        while True:
            tokens = [random.randint(100, 1000) for _ in range(token_length)]
            if (
                len(tkn.encode(tkn.decode(tokens), bos=False, eos=False))
                == token_length
            ):
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


def run_pipe_or_tensor_parallelism(args, timers):
    rank = torch.distributed.get_rank()
    for i in range(2):
        if rank == 0:
            reqs = gen_reqs(
                num_reqs=args.infer.max_reqs,
                max_new_tokens=args.request.max_new_tokens,
            )
            for req in reqs:
                TaskPool.add(Task(f"{req.request_id}", req, req.message))
        t_start = time.time()
        timers("overall").start()
        while not chitu_is_terminated():
            chitu_run()
            if rank == 0 and len(TaskPool.pool) == 0:
                break  # Rank 0 can temperarily leave to do other things
        timers("overall").stop()
        t_end = time.time()
        logger.info(f"Time cost {t_end - t_start}")

        if rank == 0:
            for req in reqs:
                logger.info(f"Response in rank {rank}: {req.output}")

        timers.log()

    chitu_terminate()


def run_normal(args, timers):
    rank = torch.distributed.get_rank()
    for i in range(3):
        reqs = gen_reqs(
            num_reqs=args.infer.max_reqs, max_new_tokens=args.request.max_new_tokens
        )
        for req in reqs:
            TaskPool.add(Task(f"{req.request_id}", req, req.message))
        t_start = time.time()
        timers("overall").start()
        while len(TaskPool.pool) > 0:
            chitu_run()

        print("GPU memory used : ", torch.cuda.memory_allocated())
        timers("overall").stop()
        t_end = time.time()
        logger.info(f"Time cost {t_end - t_start}")

        for req in reqs:
            logger.info(f"Response in rank {rank}: {req.output}")

        timers.log()


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
    if args.infer.pp_size > 1 or args.infer.tp_size > 1:
        run_pipe_or_tensor_parallelism(args, timers)
    else:
        run_normal(args, timers)


if __name__ == "__main__":
    main()
