import os

import pytest
import torch
from omegaconf import OmegaConf

from chitu.backend import Backend
from chitu.kv_cache import PagedKVCacheManager
from chitu.distributed.parallel_state import initialize_parallel_groups
from chitu.distributed.partition import compute_local_batch_size_dist_in_dp
import chitu.global_vars as global_vars
from chitu.global_vars import set_global_args
from chitu.scheduler import Scheduler
from chitu.task import Task, TaskPool, TaskType, UserRequest

_PD_UNIT_JOB_NAME = "pd_unit_test_h20"
_JOB_NAME = os.environ.get("CI_JOB_NAME") or os.environ.get("JOB_NAME")
if _JOB_NAME and _JOB_NAME != _PD_UNIT_JOB_NAME:
    pytest.skip("skip PD unit tests outside pd_unit_test_h20", allow_module_level=True)


def _maybe_init_dist():
    if torch.distributed.is_initialized():
        return
    if os.environ.get("RANK") is None or os.environ.get("WORLD_SIZE") is None:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend, init_method="env://")


@pytest.mark.pd_dist
def test_pd_dp_shard_round_robin():
    """
    This test is intended to run under torchrun with WORLD_SIZE>=2.
    Example:
      torchrun --nproc_per_node=2 pytest -m pd_dist -k test_pd_dp_shard_round_robin
    """
    _maybe_init_dist()
    if not torch.distributed.is_initialized():
        pytest.skip("run with torchrun --nproc_per_node>=2")

    world_size = torch.distributed.get_world_size()
    if world_size < 2:
        pytest.skip("need WORLD_SIZE>=2")

    cfg = OmegaConf.create(
        {
            "models": {
                "name": "unit-test",
                "vocab_size": 128,
                "n_kv_heads": 2,
                "head_dim": 8,
            },
            "infer": {
                "tp_size": 1,
                "pp_size": 1,
                "dp_size": world_size,
                "ep_size": 1,
                "mtp_size": 1,
                "op_impl": "cpu",
                "prefill_chunk_size": None,
                "schedule_overlap": False,
                "max_seq_len": 128,
                "max_batch_size": 8,
                "use_cuda_graph": False,
            },
            "dp_config": {
                "dp_id": 0,
                "router": {
                    "host": "127.0.0.1",
                    "pd_disaggregation": {
                        "enabled": True,
                        "metadata_sync_port": 0,
                        "bootstrap_port": 0,
                    },
                },
            },
            "scheduler": {
                "type": "decode_only",
                "pp_config": {
                    "prefill_num_tasks_divided_by_pp": True,
                    "prefill_num_tasks": None,
                    "enforce_decode_num_tasks_max": True,
                    "decode_num_tasks": None,
                },
            },
        }
    )
    set_global_args(cfg, need_ensure=False)
    if global_vars._GLOBAL_TIMERS is None:
        global_vars._set_timers()
    Backend.args = global_vars.get_global_args()

    initialize_parallel_groups(
        tp_size=1, pp_size=1, dp_size=world_size, ep_size=1, etp_size=1
    )

    Backend.cache_managers = [
        {
            "main": PagedKVCacheManager(
                num_blocks=128,
                num_hot_req=cfg.infer.max_batch_size,
                max_seq_len=cfg.infer.max_seq_len,
                dp_rank=i,
                block_size=16,
                mtp_size=1,
                enable_prefix_caching=False,
            )
        }
        for i in range(2)
    ]

    TaskPool.reset()

    # Create tasks assigned to dp_rank 0/1
    tasks = []
    for i in range(4):
        req = UserRequest(
            message="",
            tokens=[1],
            request_id=f"r0-{i}",
            max_new_tokens=1,
            enable_reasoning=False,
        )
        t = Task(task_id=req.request_id, req=req)
        t.task_type = TaskType.Decode
        t.dp_rank = 0
        tasks.append(t)
    for i in range(4):
        req = UserRequest(
            message="",
            tokens=[1],
            request_id=f"r1-{i}",
            max_new_tokens=1,
            enable_reasoning=False,
        )
        t = Task(task_id=req.request_id, req=req)
        t.task_type = TaskType.Decode
        t.dp_rank = 1
        tasks.append(t)

    for t in tasks:
        TaskPool.add(t)

    max_reqs_per_dp = compute_local_batch_size_dist_in_dp(
        cfg.infer.max_batch_size, cfg.infer.dp_size
    )
    sched0 = Scheduler(
        max_reqs_per_dp[0],
        prefill_num_tasks=1,
        decode_num_tasks=4,
        scheduler_type="fcfs",
        cache_manager_dict=Backend.cache_dict[0],
        num_scheduler_groups=1,
        dp_rank=0,
    )
    sched1 = Scheduler(
        max_reqs_per_dp[1],
        prefill_num_tasks=1,
        decode_num_tasks=4,
        scheduler_type="fcfs",
        cache_manager_dict=Backend.cache_dict[0],
        num_scheduler_groups=1,
        dp_rank=1,
    )

    batch0 = sched0.schedule(strict_allowed_task_type={TaskType.Decode})
    batch1 = sched1.schedule(strict_allowed_task_type={TaskType.Decode})

    assert all(TaskPool.pool[tid].dp_rank == 0 for tid in batch0)
    assert all(TaskPool.pool[tid].dp_rank == 1 for tid in batch1)
