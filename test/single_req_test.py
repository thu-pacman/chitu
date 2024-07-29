from cinfer.task import UserRequest, TaskPool, PrefillTask

# from cinfer.backend import Backend
from cinfer.cinfer_main import cinfer_init, cinfer_run
from cinfer.global_vars import set_global_variables, get_timers
from faker import Faker
import hydra
from omegaconf import DictConfig
import torch
import time

from logging import getLogger
import logging

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


def gen_req_id(len=8):
    random_number = random.getrandbits(len * 4)
    hex_string = f"{random_number:0{len}x}"
    # logger.warning(f"generating req {hex_string}")
    return hex_string


def gen_reqs(num_reqs, prompt_len, max_new_tokens):
    fake = Faker()
    reqs = []
    for i in range(num_reqs):
        msg = ""
        for j in range(prompt_len):
            msg += fake.word() + " "
        req = UserRequest(msg, f"{gen_req_id()}", max_new_tokens=max_new_tokens)
        reqs.append(req)
    return reqs


import random


def gen_reqs_real(num_reqs, max_new_tokens):
    reqs = []
    for i in range(num_reqs):
        req = UserRequest(
            msgs[i % len(msgs)], f"{gen_req_id()}", max_new_tokens=max_new_tokens
        )
        reqs.append(req)
    return reqs


def run_pipe(args, timers):
    rank = torch.distributed.get_rank()
    for i in range(2):
        if rank == 0:
            # reqs = gen_reqs(
            #     num_reqs=args.infer.max_reqs,
            #     prompt_len=512,
            #     max_new_tokens=args.request.max_new_tokens,
            # )
            reqs = gen_reqs_real(
                num_reqs=args.infer.max_reqs, max_new_tokens=args.request.max_new_tokens
            )
            for req in reqs:
                TaskPool.add(PrefillTask(f"prefill_{req.request_id}", req, req.message))
        t_start = time.time()
        timers("overall").start()
        while len(TaskPool.pool) > 0 or rank != 0:
            cinfer_run()
        timers("overall").stop()
        t_end = time.time()
        logger.warning(f"Time cost {t_end - t_start}")

        for req in reqs:
            logger.warning(f"Response in rank {rank}: {req.output}")

        timers.log()


def run_normal(args, timers):
    rank = torch.distributed.get_rank()
    for i in range(3):
        # reqs = gen_reqs(
        #     num_reqs=args.infer.max_reqs,
        #     prompt_len=512,
        #     max_new_tokens=args.request.max_new_tokens,
        # )
        reqs = gen_reqs_real(
            num_reqs=args.infer.max_reqs, max_new_tokens=args.request.max_new_tokens
        )
        for req in reqs:
            TaskPool.add(PrefillTask(f"prefill_{req.request_id}", req, req.message))
        t_start = time.time()
        timers("overall").start()
        while len(TaskPool.pool) > 0:
            cinfer_run()

        print("CPU memory used : ", torch.cuda.memory_allocated())
        timers("overall").stop()
        t_end = time.time()
        logger.warning(f"Time cost {t_end - t_start}")

        for req in reqs:
            logger.warning(f"Response in rank {rank}: {req.output}")

        timers.log()


@hydra.main(
    version_base=None, config_path="../example/configs", config_name="serve_config"
)
def main(args: DictConfig):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    print(args)
    set_global_variables()
    timers = get_timers()

    cinfer_init(args)
    logger.warning(f"finish init")
    if args.infer.parallel_type == "pipe":
        run_pipe(args, timers)
    else:
        run_normal(args, timers)


if __name__ == "__main__":
    main()
