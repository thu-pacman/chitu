import hydra
import torch
import torch.nn.functional as F
import time
import os
import random
import logging
from logging import getLogger
from pathlib import Path

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
from chitu.utils import get_config_dir_path, gen_req_id, get_chitu_env

# -----------utils part begin--------------


def save_results(data_list, filename="data.json"):
    torch.save(data_list, filename)


def load_results(filename="data.json"):
    try:
        filename = Path(filename)
        data_list = torch.load(filename)
        return data_list
    except FileNotFoundError:
        print(f"文件 {filename} 不存在，无法读取数据。")
        return []


def analyze_topk_similarity(
    topk_logits1: torch.Tensor,
    topk_tokens1: torch.Tensor,
    topk_logits2: torch.Tensor,
    topk_tokens2: torch.Tensor,
):
    """
    计算两组 Top-K 预测结果的相似度，包含语义对齐的余弦相似度、概率占比、数量占比。
    """
    B, K = topk_tokens1.shape
    device = topk_tokens1.device

    # 0. 预生成索引序列
    indices = torch.arange(K, device=device).float()

    # 1. 计算公共元素数量及Mask (前num_shared个位置为公共区域)
    is_shared1 = (topk_tokens1.unsqueeze(2) == topk_tokens2.unsqueeze(1)).any(dim=2)
    is_shared2 = (topk_tokens2.unsqueeze(2) == topk_tokens1.unsqueeze(1)).any(dim=2)
    share_num = is_shared1.sum(dim=1)
    share_mask = torch.arange(K, device=device).unsqueeze(0) < share_num.unsqueeze(1)

    # 2. T1排序：公共元素优先，内部保持原序
    sort_idx1 = torch.argsort(is_shared1.float() * K + (K - indices), descending=True)
    logits1 = torch.gather(topk_logits1, 1, sort_idx1)

    # 3. T2排序：查找T2元素在T1中的索引作为对齐依据，公共元素优先
    match_idx = (
        (topk_tokens2.unsqueeze(2) == topk_tokens1.unsqueeze(1)).float().argmax(dim=2)
    )
    sort_key2 = torch.where(
        is_shared2, K - match_idx.float(), torch.tensor(-1.0, device=device)
    )
    sort_idx2 = torch.argsort(sort_key2, descending=True)
    logits2 = torch.gather(topk_logits2, 1, sort_idx2)

    # 4. 仅对公共部分计算余弦相似度
    mask_float = share_mask.float()
    cosine_sim = F.cosine_similarity(
        logits1 * mask_float, logits2 * mask_float, dim=1, eps=1e-8
    )

    # 5. 计算公共部分的概率占比 (取两者较小值)
    ratio1 = F.softmax(logits1, dim=1).mul(mask_float).sum(dim=1)
    ratio2 = F.softmax(logits2, dim=1).mul(mask_float).sum(dim=1)

    share_prob = torch.min(ratio1, ratio2)
    share_token = share_num / K
    return cosine_sim, share_prob, share_token


def check_results(results_now, results_ref) -> tuple[float, float, int, int]:
    logger.info("checking cosine simulariy, share prob, share token")
    logger.info("cs_avg cs_min sp_avg sp_min st_avg st_min")
    tolerance = torch.tensor([0.994, 0.96, 0.995, 0.7, 0.65, 0.05], dtype=torch.float64)

    def check_metric(metric: torch.Tensor, mark="❌✅"):
        all_ok = metric >= tolerance
        s = [f"{m:.6f}{mark[ok]}" for m, ok in zip(metric.tolist(), all_ok.tolist())]
        logger.info(" ".join(s))
        return torch.all(all_ok).item()

    min_metric = torch.ones(len(tolerance), dtype=torch.float64)
    fails = []
    for i, (result_now, result_ref) in enumerate(zip(results_now, results_ref)):
        assert (
            result_now["prompt"] == result_ref["prompt"]
        ), f"prompt difference in result {i}:{result_now['prompt']} vs {result_ref['prompt']}"
        assert (
            len(result_ref["topk_logits"]) > 0
        ), f"history result {i} has empty logits"
        assert (
            len(result_now["topk_logits"]) > 0
        ), f"current result {i} has empty logits"

        num_logits = min(len(result_now["topk_logits"]), len(result_ref["topk_logits"]))
        if num_logits == 0:
            continue

        cos_sim, share_prob, share_token = analyze_topk_similarity(
            result_now["topk_logits"][:num_logits],
            result_now["topk_tokens"][:num_logits],
            result_ref["topk_logits"][:num_logits],
            result_ref["topk_tokens"][:num_logits],
        )
        metric = torch.stack(
            [
                cos_sim.mean(),
                cos_sim.min(),
                share_prob.mean(),
                share_prob.min(),
                share_token.mean(),
                share_token.min(),
            ]
        )
        if not check_metric(metric):
            fails.append(i)
        min_metric = torch.min(min_metric, metric)

    logger.info("worst:")
    assert check_metric(min_metric) == (len(fails) == 0)
    logger.info("tolerance:")
    assert check_metric(tolerance, mark="🚩🚩")

    if fails:
        logger.error(f"failed req: {fails}")
        raise UserWarning(f"failed req: {fails}")
    else:
        logger.info("All asserts passed")


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
# long-context test
msgs_long = [
    [
        {
            "role": "user",
            "content": Path("test/test_texts/test_text.txt").read_text(
                encoding="utf-8"
            ),
        }
    ],
]

counter = 1


def gen_debug_req_id(len=8):
    global counter
    req_id = f"{counter:0{len}x}"
    counter += 1
    return req_id


def gen_reqs_fake(num_reqs, prompt_len, max_new_tokens, frequency_penalty):
    from chitu.backend import Backend

    def generate_prompt(token_length, tkn):
        while True:
            tokens = [random.randint(100, 1000) for _ in range(token_length)]
            if len(tkn.encode(tkn.decode(tokens), bos=False, eos=True)) == token_length:
                return tkn.decode(tokens)

    reqs: list[UserRequest] = []
    for i in range(num_reqs):
        msg = generate_prompt(prompt_len - 1, Backend.tokenizer)
        req = UserRequest.create(
            msg,
            f"{gen_req_id()}",
            max_new_tokens=max_new_tokens,
            frequency_penalty=frequency_penalty,
        )
        req.messages = msg
        reqs.append(req)
    return reqs


def gen_reqs_real(num_reqs, max_new_tokens, frequency_penalty):
    reqs: list[UserRequest] = []
    for i in range(num_reqs):
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


def gen_reqs(num_reqs, max_new_tokens, frequency_penalty):
    global local_args, msgs
    if "DeepSeek-V3.2" in local_args.models.name:
        msgs = msgs_long + msgs

    if local_args.request.prompt_tokens_len > 0:
        return gen_reqs_fake(
            num_reqs,
            local_args.request.prompt_tokens_len,
            max_new_tokens,
            frequency_penalty,
        )
    else:
        return gen_reqs_real(num_reqs, max_new_tokens, frequency_penalty)


def run(args: ServeConfig, results_ref):
    logger.info(f"Run with args: {args}")
    chitu_init(args)
    logger.info("finish init")
    timers = get_timers()
    warmup_engine(args)
    results = []
    rank = torch.distributed.get_rank()
    for i in range(1):
        if rank == 0:
            reqs = gen_reqs(
                num_reqs=args.infer.max_batch_size,
                max_new_tokens=args.request.max_new_tokens,
                frequency_penalty=args.request.frequency_penalty,
            )
            for j, req in enumerate(reqs):
                req._test_flag = True
                if not results_ref == None:
                    result_it = (i * len(reqs) + j) % len(results_ref)
                    req._test_standard_tokens = results_ref[result_it]["tokens"]
                TaskPool.add(Task(req.request_id, req))
        t_start = time.time()
        timers("overall").start()
        while not chitu_is_terminated():
            chitu_run()
            if rank == 0 and TaskPool.all_finished():
                break  # Rank 0 can temperarily leave to do other things
        timers("overall").stop()
        t_end = time.time()
        logger.warning(f"Time cost {t_end - t_start}")

        if rank == 0:
            for req in reqs:
                logger.info(f"Response {len(req._test_tokens)} tokens: {req.output}")
                result = {
                    "prompt": req.messages[0]["content"],
                    "topk_logits": torch.stack(req._test_topk_logits),
                    "topk_tokens": torch.stack(req._test_topk_tokens),
                    "tokens": torch.tensor(req._test_tokens),
                }
                results.append(result)

        timers.log()

    chitu_terminate()

    return results


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
    rank = int(os.getenv("RANK"))
    update_history = os.environ.get("UPDATE_HISTORY", "").lower() == "true"
    history_path = os.getenv("HISTORY_PATH", "")
    history2_path = os.getenv("HISTORY2_PATH", "")

    assert history_path
    with_history = Path(history_path).exists()
    if update_history:
        assert not with_history, "history exists, forget to change HISTORY_VERSION ?"
    if with_history:
        assert Path(history_path).is_file()

    results_ref = None
    if rank == 0 and with_history:
        results_ref = load_results(history_path)

    if history2_path:
        logger.info("skip run")
        if rank == 0:
            results_now = load_results(history2_path)
            logger.info(f"load now result from {history2_path}")
    else:
        logger.info("start run")
        results_now = run(args, results_ref)

    if rank != 0:
        return

    if with_history:
        logger.info("checking result...")
        check_results(results_now, results_ref)
    else:
        logger.warning(f"history file {history_path} not found, do automatic save")
        save_results(results_now, history_path)
        if not update_history:
            raise UserWarning(f"History saved at {history_path}")


if __name__ == "__main__":
    successful = True
    try:
        main()
    except:
        logger.exception("exception")
        successful = False
    finally:
        if torch.distributed.is_initialized():
            logger.info("Waiting for all ranks to finish...")
            torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK"))])
            logger.info("All ranks finished")
        # Sometimes torch.distributed will hang during destruction if CUDA graph is enabled.
        # As a workaround, we `exec` a dummy process to kill the current process, without
        # returning an error.
        # Don't exec bash because it loads startup scripts
        if successful:
            # /usr/bin/true does nothing but exits
            os.execl("/usr/bin/true", "true")
        else:
            os.execl("/usr/bin/false", "false")
