import functools
import logging
import operator
import os
from logging import getLogger
from typing import List

import torch
import torch.distributed

from chitu.backend import Backend, BackendState
from chitu.cache_manager import PagedKVCacheManager
from chitu.device_type import is_nvidia
from chitu.distributed_utils import propagate_tensor_to_all_devices
from chitu.executor import Executor
from chitu.global_vars import (
    get_global_args,
    set_global_variables,
    set_quant_variables,
    set_backend_variables,
)
from chitu.scheduler import Scheduler
from chitu.task import (
    PackedTasks,
    PackedTasksBase,
    SerializedPackedTasksPayloadType,
    Task,
    TaskPool,
    TaskType,
    UserRequest,
    MockFixedLengthedUserRequest,
)
from chitu.utils import gen_req_id, try_import_opt_dep

numa, has_numa = try_import_opt_dep("numa", "cpu")
cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")

logger = getLogger(__name__)

EDP_SKIP_FLAG = True


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


def get_additional_block_num(
    total_gpu_memory, cache_manager, gpu_memory_utilization=0.98
):
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
    additional_kv_cache_memory = total_gpu_memory * gpu_memory_utilization - peak_memory
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
    if args.infer.dp_size > 1:  # not suppoort non_expert_data_parallel
        return

    logger.warning("Starting inference system warmup...")
    init_cache_static()
    num_warmup_reqs = args.infer.max_reqs

    for i in range(num_warmup_reqs):
        # TODO: After we implement chunked prefill, use the chunk size here for warmup_seq_len
        warmup_seq_len = 1
        warmup_max_new_tokens = 2
        req = MockFixedLengthedUserRequest(
            warmup_seq_len,
            f"{gen_req_id()}",
            max_new_tokens=warmup_max_new_tokens,
            temperature=0.7,
            top_k=1,
        )
        task = Task(f"{req.request_id}", req)
        TaskPool.add(task)

    logger.warning(f"Added {num_warmup_reqs} warmup requests to TaskPool")

    while len(TaskPool.pool) > 0:
        chitu_run()
    if should_calculate_blocks(args):
        assert isinstance(Backend.cache_manager, PagedKVCacheManager)
        _, total_gpu_memory = torch.cuda.mem_get_info(0)
        gpu_memory_utilization = args.infer.gpu_memory_utilization
        get_global_args().infer.num_blocks = (
            get_additional_block_num(
                total_gpu_memory, Backend.cache_manager, gpu_memory_utilization
            )
            + Backend.cache_manager.num_blocks
        )

    logger.warning("Inference system warmup completed")


def check_checkpoint_path(args):
    if args.models.ckpt_dir is None:
        raise ValueError(
            f"No checkpoint path provided. You can set it in command line by adding `models.ckpt_dir=<path>`. The model {args.models.name} can be downloaded from {args.models.source}"
        )
    if args.models.tokenizer_path is None:
        logger.info(
            f"Using {args.models.ckpt_dir} as the path to tokenizer. If the tokenizer has a different path, please set in command line by adding `models.tokenizer_path=<path>`"
        )
        args.models.tokenizer_path = args.models.ckpt_dir


def chitu_init(args, logging_level=None):
    debug = os.getenv("CHITU_DEBUG", "0") == "1"
    global EDP_SKIP_FLAG

    if (
        is_nvidia()
        and torch.distributed.is_nccl_available()
        and torch.cuda.nccl.version() <= (2, 21, 5)
    ):
        os.environ["NCCL_NVLS_NCHANNELS"] = "32"

    if logging_level is None:
        logging_level = logging.DEBUG if debug else logging.INFO
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
        # Set environ for ascend
        from chitu.utils import get_ascend_custom_opp_path

        site_packages_path = get_ascend_custom_opp_path()
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = site_packages_path

        # Bind process to CPU NUMA
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        if args.infer.bind_process_to_cpu == "auto":
            if not has_cpuinfer and not has_numa:
                args.infer.bind_process_to_cpu = "none"
            elif not has_numa:
                logger.warning(
                    "'cpuinfer' is found but 'numa' is mising. Disabling NUMA binding. "
                    "For better CPU inference performance, please refer to README.md and "
                    "install the full '[cpu]' optional dependency."
                )
                args.infer.bind_process_to_cpu = "none"
            elif not numa.available():
                logger.warning(
                    "NUMA is not support on this OS or hardware platform. Disabling NUMA binding."
                )
                args.infer.bind_process_to_cpu = "none"
            elif numa.get_max_node() + 1 < local_world_size:
                logger.info("Disable NUMA binding due to insufficient NUMA nodes.")
                args.infer.bind_process_to_cpu = "none"
            else:
                args.infer.bind_process_to_cpu = "numa"
        if args.infer.bind_process_to_cpu == "numa":
            numa.bind({local_rank})
        elif args.infer.bind_process_to_cpu == "none":
            pass
        else:
            raise ValueError(
                f"Unsupported infer.bind_process_to_cpu={args.infer.bind_process_to_cpu}"
            )

    if args.infer.use_cuda_graph:
        if args.infer.attn_type == "flash_infer":
            args.infer.use_cuda_graph = False
            args.infer.cuda_graph_backend = "flash_infer"

    # Check checkpoint exists
    check_checkpoint_path(args)

    set_quant_variables(args)
    set_backend_variables(args)
    set_global_variables(args, debug=debug)

    args = get_global_args()
    Backend.build(args)
    rank = torch.distributed.get_rank()
    if rank == 0 or (
        args.infer.dp_size > 1
    ):  # [HACK] temporary workaround to support dp+ep
        scheduler = Scheduler.build(args.scheduler, args.infer)
        Backend.scheduler = scheduler
        EDP_SKIP_FLAG = False
    executor = Executor.build(args)
    Backend.executor = executor
    PackedTasks.configure(max_num_tasks=args.infer.max_reqs)


def remove_kvcache_all_device(remove_task_ids):
    if len(remove_task_ids) == 0:
        return

    # Remove KV cache on this device
    for task_id in remove_task_ids:
        Backend.cache_manager.finalize_cache_all_decode(task_id)

    # Propagate metadata to remove KV cache on other devices
    if torch.distributed.get_world_size() > 1:
        task_tensor = PackedTasksBase(
            num_tasks=len(remove_task_ids),
            task_ids=remove_task_ids,
            req_ids=remove_task_ids,
            task_type=TaskType.Decode,  # Since we are removing, any task type is fine
        ).serialize(
            payload_type=SerializedPackedTasksPayloadType.EndTask,
            device="cpu" if Backend.use_gloo else 0,
        )
        propagate_tensor_to_all_devices(task_tensor)


@torch.inference_mode()
def chitu_run_normal():
    task_ids = Backend.scheduler.schedule()

    if task_ids:
        # compute
        logger.debug(f"Processing {task_ids}")
        tasks = PackedTasks(task_ids)
        logits = Backend.executor.step(tasks)

        # postprocess
        if len(Backend.last_batch_results) > 0:
            Backend.executor.postprocess_async_part(
                Backend.last_batch_results.popleft()
            )
        curr_batch_result = Backend.executor.postprocess_sync_part(tasks, logits)
        Backend.last_batch_results.append(curr_batch_result)
        removed_decode_task_ids = Backend.scheduler.update(task_ids)
        remove_kvcache_all_device(removed_decode_task_ids)
    elif len(Backend.last_batch_results) > 0:
        # ensure the last batch result is processed
        Backend.executor.postprocess_async_part(Backend.last_batch_results.popleft())


def _update_ongoing_tasks():
    unwait_tasks: List[PackedTasks] = []
    logits_list: List[torch.Tensor] = []
    for ogr in Backend.ongoing_reqs:
        if ogr.handle.is_completed():
            Backend.ongoing_reqs.remove(ogr)
            unwait_tasks.append(ogr.waiting_task)
            logits_list.append(ogr.logits.view(-1, ogr.logits.shape[-1]))
            for task in ogr.waiting_task.tasks:
                task.unwait()
    return unwait_tasks, logits_list


@torch.inference_mode()
def chitu_run_pp():
    task_ids = Backend.scheduler.schedule()

    if task_ids:
        # compute
        logger.debug(f"Processing {task_ids}")
        tasks = PackedTasks(task_ids)
        Backend.executor.step(tasks)

        # postprocess async part
        if len(Backend.last_batch_results) > 0:
            Backend.executor.postprocess_async_part(
                Backend.last_batch_results.popleft()
            )
    elif len(Backend.last_batch_results) > 0:
        # ensure the last batch result is processed
        Backend.executor.postprocess_async_part(Backend.last_batch_results.popleft())

    # postprocess sync part
    unwait_batches, logits = _update_ongoing_tasks()
    for idx, batch in enumerate(unwait_batches):
        Backend.last_batch_results.append(
            Backend.executor.postprocess_sync_part(batch, logits[idx])
        )
    unwait_task_ids = [t.task_id for batch in unwait_batches for t in batch.tasks]
    removed_decode_task_ids = Backend.scheduler.update(task_ids, unwait_task_ids)
    remove_kvcache_all_device(removed_decode_task_ids)


@torch.inference_mode()
def chitu_run():
    global EDP_SKIP_FLAG
    rank = torch.distributed.get_rank()
    if rank != 0 and EDP_SKIP_FLAG:  # [HACK] temporary workaround to support dp+ep
        Backend.executor.step(None)
        return

    if Backend.args.infer.pp_size > 1:
        chitu_run_pp()
    else:
        chitu_run_normal()


def chitu_terminate():
    if torch.distributed.get_rank() == 0:
        Backend.state = BackendState.Terminated
        terminated_task_tensor = PackedTasksBase.serialize_special(
            SerializedPackedTasksPayloadType.TerminateBackend,
            device="cpu" if Backend.use_gloo else 0,
        )
        propagate_tensor_to_all_devices(terminated_task_tensor)


def chitu_is_terminated():
    return Backend.state == BackendState.Terminated
