import torch.distributed as dist

from chitu.backend import Backend
from chitu.executor import TASK_TENSOR_TAG
from chitu.tensor_parallel import get_tp_group, get_pp_group, get_cpu_tp_group


def propagate_tensor_to_all_devices(tensor):
    if Backend.args.infer.pp_size > 1:
        pg = get_pp_group(0, Backend.args.infer.tp_size)
        dist.send(
            tensor=tensor,
            dst=Backend.args.infer.tp_size,
            tag=TASK_TENSOR_TAG,
            group=Backend.group_gloo if Backend.use_gloo else pg,
        )
        if Backend.args.infer.tp_size > 1:
            _broadcast_within_pp_stage(tensor)
    elif Backend.args.infer.tp_size > 1:
        _broadcast_to_tp_group(tensor, src_rank=0)


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
        dist.broadcast(tensor=tensor, src=src_rank, group=Backend.group_gloo)
    else:
        dist.broadcast(tensor=tensor, src=src_rank, group=get_cpu_tp_group())
