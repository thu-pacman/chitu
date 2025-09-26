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
    chitu_terminate,
    chitu_is_terminated,
    warmup_engine,
)
from chitu.global_vars import get_timers
from chitu.schemas import ServeConfig
from chitu.utils import get_config_dir_path, gen_req_id

# -----------utils part begin--------------
import json


def save_result(data_list, filename="data.json"):
    torch.save(data_list, filename)


def load_result(filename="data.json"):
    try:
        data_list = torch.load(filename)
        return data_list
    except FileNotFoundError:
        print(f"文件 {filename} 不存在，无法读取数据。")
        return []


def check_result(
    result_0_lst, result_1_lst, threshold=0.99
) -> tuple[float, float, int, int]:
    tot_cos_sim = 0.0
    logit_cnt = 0
    min_cos_sim = float("inf")
    min_cos_sim_result_it = -1
    min_cos_sim_logit_it = -1

    for result_it in range(min(len(result_0_lst), len(result_1_lst))):
        result_0 = result_0_lst[result_it]
        result_1 = result_1_lst[result_it]
        assert (
            result_0["prompt"] == result_1["prompt"]
        ), f"prompt difference in result {result_it}:{result_0['prompt']} vs {result_1['prompt']}"
        assert (
            len(result_1["logits"]) > 0
        ), f"history result {result_it} has empty logits"
        assert (
            len(result_0["logits"]) > 0
        ), f"current result {result_it} has empty logits"

        num_logits = min(len(result_0["logits"]), len(result_1["logits"]))
        if num_logits == 0:
            continue
        logit_cnt += num_logits

        logit0 = result_0["logits"][:num_logits]
        logit1 = result_1["logits"][:num_logits]

        dp = torch.sum(logit0.float() * logit1.float(), dim=-1)
        norm = torch.norm(logit0.float(), p=2, dim=-1) * torch.norm(
            logit1.float(), p=2, dim=-1
        )
        mask = norm != 0
        cos_sim = torch.where(mask, dp / norm, 0.0)

        min_val, min_idx = list(map(lambda x: x.item(), torch.min(cos_sim, dim=0)))
        if min_val < min_cos_sim:
            min_cos_sim = min_val
            min_cos_sim_logit_it = min_idx
            min_cos_sim_result_it = result_it

        assert (
            min_val >= threshold
        ), f"cosine similarity difference in result {result_it}:: logit {min_idx}, min_cos_sim: {min_val}, threshold: {threshold}"

        tot_cos_sim += torch.sum(cos_sim).item()

    if min_cos_sim == float("inf"):
        min_cos_sim = 0.0

    avg_cos_sim = tot_cos_sim / logit_cnt if logit_cnt > 0 else 0.0

    return (avg_cos_sim, min_cos_sim, min_cos_sim_result_it, min_cos_sim_logit_it)


# -----------utils part end--------------

logger = getLogger(__name__)

msgs = [
    [{"role": "user", "content": "宫保鸡丁怎么做?"}],
    [{"role": "user", "content": "what is the recipe of Kung Pao chicken?"}],
    [{"role": "user", "content": "怎么写程序?"}],
    [{"role": "user", "content": "飞机在对流层还是平流层飞?"}],
    [{"role": "user", "content": "怎么避免加班?"}],
    [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
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
            if len(tkn.encode(tkn.decode(tokens), bos=False, eos=True)) == token_length:
                return tkn.decode(tokens)

    reqs: list[UserRequest] = []
    for i in range(num_reqs):
        msg = generate_prompt(prompt_len - 1, Backend.tokenizer)
        req = UserRequest(msg, f"{gen_req_id()}", max_new_tokens=max_new_tokens)
        reqs.append(req)
    return reqs


def gen_reqs_real(num_reqs, max_new_tokens):
    reqs: list[UserRequest] = []
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
                TaskPool.add(Task(req.request_id, req))
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
            logits = torch.tensor(result_logits[it])
            tokens = torch.tensor(result_tokens[it])
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
            TaskPool.add(Task(req.request_id, req))
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
        logits = torch.tensor(result_logits[it])
        tokens = torch.tensor(result_tokens[it])
        result.append({"prompt": prompt, "logits": logits, "tokens": tokens})
    return result


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

    rank = torch.distributed.get_rank()
    warmup_engine(args)

    update_history = os.getenv("UPDATE_HISTORY", "false").lower() == "true"
    history_path = os.getenv("HISTORY_PATH", "./example/history/history.txt")
    history_result = None
    if rank == 0 and not update_history:
        if os.path.exists(history_path):
            history_result = load_result(history_path)

    now_result = None
    if args.infer.pp_size > 1 or args.infer.tp_size > 1:
        now_result = run_pipe_or_tensor_parallelism(args, timers, history_result)
    else:
        now_result = run_normal(args, timers, history_result)

    if rank == 0:
        if update_history:
            logger.info(
                "UPDATE_HISTORY is set to true, saving current result as history."
            )
            save_result(now_result, history_path)
        else:
            if history_result is not None:
                threshold = 0.99
                (
                    avg_cos_sim,
                    min_cos_sim,
                    min_cos_sim_result_it,
                    min_cos_sim_logit_it,
                ) = check_result(now_result, history_result, threshold=threshold)
                logger.info(f"!!!!!!!!! Average cosine similarity: {avg_cos_sim}")
                logger.info(f"!!!!!!!!! Minimum cosine similarity: {min_cos_sim}")
                logger.info(
                    f"!!!!!!!!! Minimum cosine similarity found in result {min_cos_sim_result_it}, logit {min_cos_sim_logit_it}"
                )
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
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    # Don't exec bash because it loads startup scripts
    os.execl("/usr/bin/true", "true")  # /usr/bin/true does nothing but exits
