import functools
import logging
import operator
import os
from logging import getLogger

import torch
import torch.distributed

from chitu.backend import Backend, BackendState
from chitu.device_type import is_nvidia
from chitu.distributed_utils import propagate_tensor_to_all_devices
from chitu.executor import Executor
from chitu.global_vars import get_global_args, set_global_variables, set_quant_variables
from chitu.scheduler import Scheduler
from chitu.task import (
    PackedTasks,
    PackedTasksBase,
    SerializedPackedTasksPayloadType,
    Task,
    TaskPool,
    TaskType,
    UserRequest,
)
from chitu.utils import gen_req_id

logger = getLogger(__name__)


def init_logger(logging_level=logging.INFO):
    base_name = __name__.split(".")[0]
    base_logger = getLogger(base_name)
    base_logger.setLevel(logging_level)

    def add_rank_to_msg(record):
        if torch.distributed.is_initialized():
            record.msg = f"[Rank {torch.distributed.get_rank()}] {record.msg}"
        return True

    def add_filter_to_all_parent_handlers(cur_logger):
        for handler in cur_logger.handlers:
            handler.addFilter(add_rank_to_msg)
        if cur_logger.parent:
            add_filter_to_all_parent_handlers(cur_logger.parent)

    # If there is no handlers, create a new handler to hold the filter. If not (very likely
    # because we launch from Hydra, and Hydra setup the root logger), a new handler will only
    # duplicate the logs. In this case, we should add the filter to all the existing handlers.
    if base_logger.hasHandlers():  # Including handlers from the parents
        add_filter_to_all_parent_handlers(base_logger)
    else:
        handler = logging.StreamHandler()
        handler.addFilter(add_rank_to_msg)
        base_logger.addHandler(handler)


def should_calculate_blocks(args):
    return (
        args.infer.cache_type == "paged"
        and args.infer.num_blocks == -1
        and torch.distributed.get_rank() == 0
    )


def init_cache_static():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)


def get_additional_block_num(total_gpu_memory, cache_manager):
    """Calculate additional block numbers based on available memory"""

    def tuple_product(t):
        return functools.reduce(operator.mul, t, 1)

    peak_memory = torch.cuda.memory_stats(0)["allocated_bytes.all.peak"]
    torch.cuda.empty_cache()
    torch_allocated_bytes = torch.cuda.memory_stats(0)["allocated_bytes.all.current"]
    total_allocated_bytes = (
        torch.cuda.mem_get_info(0)[1] - torch.cuda.mem_get_info(0)[0]
    )
    non_torch_allocations = total_allocated_bytes - torch_allocated_bytes
    if non_torch_allocations > 0:
        peak_memory += non_torch_allocations
    additional_kv_cache_memory = total_gpu_memory * 0.98 - peak_memory
    block_mem = (
        2
        * cache_manager.block_size
        * tuple_product(cache_manager.k_shape_per_sample)
        * cache_manager.num_layers
    )
    if cache_manager.v_shape_per_sample is not None:
        block_mem *= 2
    num_blocks = int(additional_kv_cache_memory) // block_mem
    if num_blocks < 0:
        num_blocks = 0
    return num_blocks


def warmup_engine(args):
    logger.warning("Starting inference system warmup...")
    init_cache_static()
    num_warmup_reqs = args.infer.max_reqs
    warmup_msg = [{"role": "user", "content": "Hi"}]

    for i in range(num_warmup_reqs):
        req = UserRequest(
            warmup_msg,
            f"{gen_req_id()}",
            max_new_tokens=10,
            temperature=0.7,
            top_k=1,
        )
        task = Task(
            f"{req.request_id}",
            req,
            req.message,
            max_seq_len=args.infer.max_seq_len,
        )
        TaskPool.add(task)

    logger.warning(f"Added {num_warmup_reqs} warmup requests to TaskPool")

    while len(TaskPool.pool) > 0:
        chitu_run()
    if should_calculate_blocks(args):
        _, total_gpu_memory = torch.cuda.mem_get_info(0)
        get_global_args().infer.num_blocks = (
            get_additional_block_num(total_gpu_memory, Backend.cache_manager)
            + args.infer.max_reqs
        )

    logger.warning("Inference system warmup completed")


def chitu_init(args, logging_level=logging.INFO):
    if (
        is_nvidia()
        and torch.distributed.is_nccl_available()
        and torch.cuda.nccl.version() <= (2, 21, 5)
    ):
        os.environ["NCCL_NVLS_NCHANNELS"] = "32"

    init_logger(logging_level)

    # Deal with legacy arguments
    if hasattr(args.infer, "soft_fp8") and args.infer.soft_fp8:
        logger.warning(
            "Argument `infer.soft_fp8=True` is deprecated. Use `infer.raise_lower_bit_float_to=bfloat16` instead."
        )
        args.infer.raise_lower_bit_float_to = "bfloat16"
    if hasattr(args, "dtype") and args.dtype is not None:
        logger.warning(
            "Argument `dtype` is deprecated. Use `float_16bit_variant` instead."
        )
        args.float_16bit_variant = args.dtype

    if args.infer.attn_type == "npu":
        try:
            import torch_npu
            from torch_npu.contrib import transfer_to_npu

            torch.cuda.CUDAGraph = torch.npu.NPUGraph
        except ImportError:
            raise ImportError("torch_npu is not installed")

    set_quant_variables(args)
    set_global_variables(args)

    Backend.build(args)
    rank = torch.distributed.get_rank()
    if rank == 0:
        scheduler = Scheduler.build(args.scheduler, args.infer)
        Backend.scheduler = scheduler
    executor = Executor.build(args)
    Backend.executor = executor
    PackedTasks.configure(max_num_tasks=args.infer.max_reqs)


def remove_task_other_device(remove_task_ids):
    # TODO handle in executor?
    if len(remove_task_ids) == 0:
        return
    # Since we are removing, any task type is fine
    task_tensor = PackedTasksBase(
        len(remove_task_ids), remove_task_ids, remove_task_ids, TaskType.Decode
    ).serialize(
        payload_type=SerializedPackedTasksPayloadType.EndTask,
        device="cpu" if Backend.use_gloo else 0,
    )
    propagate_tensor_to_all_devices(task_tensor)


def update_ongoing_tasks():
    to_remove = []
    unwait_tasks = []
    logits_list = []
    for ogr in Backend.ongoing_reqs:
        if ogr.handle.is_completed():
            to_remove.append(ogr)
            unwait_tasks.append(ogr.waiting_task)
            logits_list.append(ogr.logits.view(-1, ogr.logits.shape[-1]))
            for task in ogr.waiting_task.tasks:
                task.unwait()
    for tr in to_remove:
        Backend.ongoing_reqs.remove(tr)
    return unwait_tasks, logits_list


def chitu_update(task_ids, rank, world_size):
    if rank == 0:
        TaskPool.display()
    if world_size == 1:
        Backend.scheduler.update(task_ids)
    else:
        unwait_tasks, logits = update_ongoing_tasks()
        for idx, task in enumerate(unwait_tasks):
            Backend.executor.update_response(task, logits[idx])
        unwait_task_ids = [t.task_id for task in unwait_tasks for t in task.tasks]
        removed_decode_task_ids = Backend.scheduler.update(task_ids, unwait_task_ids)
        remove_task_other_device(removed_decode_task_ids)


@torch.inference_mode()
def chitu_run():
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if rank == 0:
        task_ids = Backend.scheduler.schedule()
        if len(task_ids) == 0:  # no tasks to do, but some tasks are waiting
            chitu_update(task_ids, rank, world_size)
            return
        if rank == 0:
            logger.debug(f"Processing {task_ids}")
        tasks = PackedTasks(task_ids, rank)
    else:
        tasks = None
    Backend.executor.step(tasks)

    if Backend.args.infer.pp_size > 1 and rank == 0:
        chitu_update(task_ids, rank, world_size)
    elif rank == 0:
        TaskPool.display()
        removed_decode_task_ids = Backend.scheduler.update(task_ids)
        if world_size != 1:
            remove_task_other_device(removed_decode_task_ids)


def chitu_terminate():
    if torch.distributed.get_rank() == 0:
        Backend.state = BackendState.Terminating
        Backend.executor.step(None)


def chitu_is_terminated():
    return Backend.state == BackendState.Terminated
