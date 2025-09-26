import hydra
import torch
import time
import os
import random
import logging
from logging import getLogger

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
from chitu.utils import get_config_dir_path, gen_req_id

logger = getLogger(__name__)

msgs = [
    [{"role": "user", "content": "宫保鸡丁怎么做?"}],
    [{"role": "user", "content": "what is the recipe of Kung Pao chicken?"}],
    [{"role": "user", "content": "怎么写程序?"}],
    [{"role": "user", "content": "飞机在对流层还是平流层飞?"}],
    [{"role": "user", "content": "怎么避免加班?"}],
    [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
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
    ]
]
counter = 1


def gen_debug_req_id(len=8):
    global counter
    req_id = f"{counter:0{len}x}"
    counter += 1
    return req_id


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

    reqs: list[UserRequest] = []
    for i in range(num_reqs):
        msg = generate_prompt(prompt_len - 1, Backend.tokenizer)
        req = UserRequest(msg, f"{gen_req_id()}", max_new_tokens=max_new_tokens)
        reqs.append(req)
    return reqs


def gen_reqs_real(num_reqs, max_new_tokens, is_vl=False):
    reqs: list[UserRequest] = []
    for i in range(num_reqs):
        if is_vl:
            req = UserRequest(
                msgs_vl[i % len(msgs_vl)],
                f"{gen_req_id()}",
                max_new_tokens=max_new_tokens,
                temperature=1,
            )
        else:
            req = UserRequest(
                msgs[i % len(msgs)],
                f"{gen_req_id()}",
                max_new_tokens=max_new_tokens,
                temperature=1,
            )
        reqs.append(req)
    return reqs


def gen_reqs(num_reqs, max_new_tokens, is_vl=False):
    global local_args
    if local_args.request.prompt_tokens_len > 0:
        return gen_reqs_fake(
            num_reqs, local_args.request.prompt_tokens_len, max_new_tokens
        )
    else:
        return gen_reqs_real(num_reqs, max_new_tokens, is_vl)


def run_pipe_or_tensor_parallelism(args, timers):
    rank = torch.distributed.get_rank()
    warmup_engine(args)

    for i in range(2):
        chitu_start()
        if rank == 0:
            reqs = gen_reqs(
                num_reqs=args.infer.max_reqs,
                max_new_tokens=args.request.max_new_tokens,
                is_vl=hasattr(args.models, "vision_config"),
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
            if rank == 0 and len(TaskPool.pool) == 0:
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
                logger.info(f"Response in rank {rank}: reqs[{i}].output={req.output}")

            timers.log()
        chitu_terminate()


def run_normal(args, timers):
    rank = torch.distributed.get_rank()
    warmup_engine(args)

    for i in range(2):
        reqs = gen_reqs(
            num_reqs=args.infer.max_reqs,
            max_new_tokens=args.request.max_new_tokens,
            is_vl=hasattr(args.models, "vision_config"),
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

        print("GPU memory used : ", torch.cuda.memory_allocated())
        timers("overall").stop()
        t_end = time.time()
        logger.info(f"Tokens generate : {tokens}")
        logger.info(f"Time cost {t_end - t_start}")

        for i, req in enumerate(reqs):
            logger.info(f"Response in rank {rank}: reqs[{i}].output={req.output}")

        timers.log()


@hydra.main(
    version_base=None,
    config_path=os.getenv("CONFIG_PATH", get_config_dir_path()),
    config_name=os.getenv("CONFIG_NAME", "serve_config"),
)
def main(args: ServeConfig):
    global local_args
    local_args = args
    logger.setLevel(logging.DEBUG)
    logger.info(f"Run with args: {args}")

    chitu_init(args, logging_level=logging.INFO)
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    timers = get_timers()
    logger.debug("finish init")
    if args.infer.pp_size > 1 or args.infer.tp_size > 1 or args.infer.dp_size > 1:
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
