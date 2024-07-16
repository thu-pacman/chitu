from cinfer.task import UserRequest, TaskPool, PrefillTask
from cinfer.model import Backend
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
    [{"role": "user", "content": "what is the recipe of Kung Pao chicken?"}],
    # [{"role": "user", "content": "what is the recipe of Kung Pao chicken?"}],
    [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
    [
        {"role": "user", "content": "I am going to Paris, what should I see?"},
        {
            "role": "assistant",
            "content": """\
        Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:
        1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
        2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
        3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.
        These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
        },
        {"role": "user", "content": "What is so great about #1?"},
    ],
]


def gen_req_id(len=8):
    random_number = random.getrandbits(len * 4)
    hex_string = f"{random_number:0{len}x}"
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


@hydra.main(
    version_base=None, config_path="../example/configs", config_name="serve_config"
)
def main(args: DictConfig):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    set_global_variables()
    timers = get_timers()

    Backend.build(args.model)
    cinfer_init(args)
    logger.warning("Backend built")

    # reqs = gen_reqs(
    #     num_reqs=16, prompt_len=512, max_new_tokens=args.request.max_new_tokens
    # )
    rank = torch.distributed.get_rank()
    for i in range(2):
        if rank == 0:
            reqs = gen_reqs_real(num_reqs=1, max_new_tokens=args.request.max_new_tokens)
            for req in reqs:
                TaskPool.add(PrefillTask(f"prefill_{req.request_id}", req, req.message))
        timers("overall").start()
        while len(TaskPool.pool) > 0 or rank != 0:
            cinfer_run()
        timers("overall").stop()

        for req in reqs:
            logger.warning(f"Response in rank {rank}: {req.output}")

        timers.log()


if __name__ == "__main__":
    main()
