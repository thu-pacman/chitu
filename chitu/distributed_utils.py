import torch
import torch.distributed as dist
from logging import getLogger

logger = getLogger(__name__)
from chitu.backend import Backend
from chitu.executor import TASK_TENSOR_TAG
from chitu.distributed.parallel_state import (
    get_tp_group,
    get_cpu_tp_group,
    get_pp_pair_group,
)


def propagate_tensor_to_all_devices(tensor):
    if Backend.args.infer.pp_size > 1:
        pg = get_pp_pair_group(0, Backend.args.infer.tp_size)
        dist.send(
            tensor=tensor,
            dst=Backend.args.infer.tp_size,
            tag=TASK_TENSOR_TAG,
            group=Backend.group_gloo if Backend.use_gloo else pg,
        )
        if Backend.args.infer.tp_size > 1:
            _broadcast_within_pp_stage(tensor)
    elif Backend.args.infer.tp_size > 1:
        src_rank = torch.distributed.get_rank()
        _broadcast_to_tp_group(tensor, src_rank=src_rank)


def _broadcast_within_pp_stage(tensor):
    if not Backend.use_gloo:
        dist.broadcast(tensor=tensor, src=Backend.pp_main_rank, group=get_tp_group())
    elif Backend.args.infer.pp_size == 1:
        dist.broadcast(
            tensor=tensor, src=Backend.pp_main_rank, group=Backend.group_gloo
        )
    else:
        dist.broadcast(
            tensor=tensor, src=Backend.pp_main_rank, group=get_cpu_tp_group()
        )


def _broadcast_to_tp_group(tensor, src_rank):
    if not Backend.use_gloo:
        dist.broadcast(tensor=tensor, src=src_rank)
    elif Backend.args.infer.pp_size == 1:
        dist.broadcast(tensor=tensor, src=src_rank, group=get_cpu_tp_group())
    else:
        dist.broadcast(tensor=tensor, src=src_rank, group=get_cpu_tp_group())


def is_local_master_rank() -> bool:
    """
    Determine if the current process is the local master rank.

    In DP (Data Parallel) mode:
    - Use local rank within DP group to determine if it's the master process
    - For example: Global rank 0 in DP Group 0 and Global rank 2 in DP Group 1 are both master processes

    In non-DP mode:
    - Use global rank to determine if it's the master process, only global rank 0 is the master process

    Returns:
        bool: If the current process is the local master rank, return True, otherwise return False
    """
    import os
    import torch

    # Check if distributed environment is initialized
    if not torch.distributed.is_initialized():
        return True  # If not initialized, default to master process

    global_rank = torch.distributed.get_rank()

    # Check if DP mode is enabled
    try:
        from chitu.distributed.parallel_state import get_dp_group

        dp_group = get_dp_group()

        # Check if it's a contiguous DP group mode
        if (
            hasattr(dp_group, "contiguous_mode")
            and dp_group.contiguous_mode
            and hasattr(dp_group, "local_rank_in_group")
        ):

            # Contiguous DP group mode: use local rank within group
            local_rank_in_group = getattr(dp_group, "local_rank_in_group", 0)
            return local_rank_in_group == 0

    except (ImportError, AttributeError, AssertionError):
        # If failed to get DP group info, fallback to traditional mode
        pass

    # Non-DP mode or fallback: use global rank
    return global_rank == 0
