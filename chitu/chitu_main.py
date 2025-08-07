# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

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
    block_mem *= (1 if cache_manager.k_shape_per_sample is not None else 0) + (
        1 if cache_manager.v_shape_per_sample is not None else 0
    )
    num_blocks = int(additional_kv_cache_memory) // block_mem
    return max(0, num_blocks)


def warmup_engine(args):
    if args.infer.pp_size > 1 and args.infer.cache_type == "paged":
        assert isinstance(Backend.cache_manager, PagedKVCacheManager)
        logger.warning("Warming-up is not supported when PP is enabled. Skipping")
        if args.infer.num_blocks == -1:
            logger.warning(
                "Auto infer.num_blocks (infer.num_blocks=-1) relies on warming-up to calculate the number of "
                "blocks, but this is not supported when PP is enabled. A safe but inefficient value is used."
            )
            new_num_block = (
                args.infer.max_reqs
                * args.infer.max_seq_len
                // Backend.cache_manager.block_size
            )
            get_global_args().infer.num_blocks = new_num_block
            Backend.cache_manager.realloc(new_num_block)
        return

    rank = torch.distributed.get_rank()

    logger.warning("Starting inference system warmup...")

    init_cache_static()
    num_warmup_reqs = args.infer.max_reqs
    warmup_seq_len = 1
    warmup_max_new_tokens = 2
    if rank == 0:
        for i in range(num_warmup_reqs):
            # TODO: After we implement chunked prefill, use the chunk size here for warmup_seq_len
            req = MockFixedLengthedUserRequest(
                warmup_seq_len,
                f"{gen_req_id()}",
                max_new_tokens=warmup_max_new_tokens,
                temperature=0.7,
                top_k=1,
            )
            task = Task(f"{req.request_id}", req, stop_with_eos=False)
            TaskPool.add(task)
            logger.warning(f"Added {num_warmup_reqs} warmup requests to TaskPool")

    if rank > 0:
        chitu_run()  # An extra run is needed because our implementation is asymmetric
    for _ in range(warmup_max_new_tokens):
        chitu_run()

    if rank == 0:
        assert len(TaskPool.pool) == 0, "TaskPool should be empty after warmup"

    logger.warning("Inference system warmup completed")

    if args.infer.cache_type == "paged" and args.infer.num_blocks == -1:
        assert isinstance(Backend.cache_manager, PagedKVCacheManager)
        _, total_gpu_memory = torch.cuda.mem_get_info(0)
        gpu_memory_utilization = args.infer.gpu_memory_utilization
        new_num_block = (
            get_additional_block_num(
                total_gpu_memory, Backend.cache_manager, gpu_memory_utilization
            )
            + Backend.cache_manager.num_blocks
        )

        if torch.distributed.get_world_size() > 1:
            new_num_block_tensor = torch.tensor(new_num_block).cuda()
            torch.distributed.all_reduce(
                new_num_block_tensor, torch.distributed.ReduceOp.RedOpType.MIN
            )
            new_num_block = new_num_block_tensor.item()

        get_global_args().infer.num_blocks = new_num_block
        Backend.cache_manager.realloc(new_num_block)


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
    if rank == 0:
        scheduler = Scheduler.build(args.scheduler, args.infer)
        Backend.scheduler = scheduler
    executor = Executor.build(args)
    Backend.executor = executor
    PackedTasks.configure(max_num_tasks=args.infer.max_reqs)
    logger.warning(f"[CHITU_INIT] [Rank {rank}] Chitu initialized")


def remove_kvcache_all_device(remove_task_ids):
    if len(remove_task_ids) == 0:
        return
    # Since we are removing, any task type is fine
    tasks = PackedTasksBase(
        num_tasks=len(remove_task_ids),
        task_ids=remove_task_ids,
        req_ids=remove_task_ids,
        task_type=TaskType.Decode,
        payload_type=SerializedPackedTasksPayloadType.EndTask,
    )
    Backend.executor.step(tasks)


@torch.inference_mode()
def chitu_run_normal():
    task_ids = Backend.scheduler.schedule()

    if task_ids:
        # compute
        logger.debug(f"Processing {task_ids}")
        tasks = (
            PackedTasks(task_ids[0])
            if type(task_ids[0]) == list
            else PackedTasks(task_ids)
        )

        logits = Backend.executor.step(tasks)

        if Backend.task_id_list is not None:
            tasks = Backend.all_tasks
            logits = Backend.cat_logits
            task_ids = Backend.all_task_ids
            Backend.task_id_list = None

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
    rank = torch.distributed.get_rank()
    if rank != 0:
        Backend.executor.step(None)
        return

    if Backend.args.infer.pp_size > 1:
        chitu_run_pp()
    else:
        chitu_run_normal()


async def start_enhanced_scheduler_service(rank: int, dp_config: dict, args):
    # only main rank of dp group start enhanced scheduler service
    if rank != 0:
        logger.warning(
            f"[Enhanced Scheduler {rank}] only main rank of dp group start Enhanced Scheduler service"
        )
        return

    """Start Enhanced Scheduler service, listen to ZMQ requests"""
    import zmq
    import zmq.asyncio
    import msgpack
    import time

    logger.warning(f"[Enhanced Scheduler {rank}] Starting...")

    # Initialize ZMQ
    context = zmq.asyncio.Context()

    # Receive request socket
    request_socket = context.socket(zmq.PULL)
    request_port = dp_config.scheduler_base_port
    request_address = f"tcp://{dp_config.scheduler_base_host}:{request_port}"
    request_socket.bind(request_address)
    logger.warning(
        f"[Enhanced Scheduler {rank}] Listening to requests: {request_address}"
    )

    # Send statistics socket
    stats_socket = context.socket(zmq.PUSH)
    stats_address = f"tcp://{dp_config.router.host}:{dp_config.router.stats_port}"  # Router stats port
    stats_socket.connect(stats_address)
    logger.warning(
        f"[Enhanced Scheduler {rank}] connected to stats service: {stats_address}"
    )

    # Start DP Token Manager
    try:
        from chitu.dp_token_sender import start_dp_token_manager

        dp_group_id = rank  # Use rank as DP group ID
        router_token_address = f"tcp://{dp_config.router.host}:{dp_config.router.token_port}"  # Token Router listen address

        logger.warning(
            f"[Enhanced Scheduler {rank}] Starting DP Token Manager, group ID={dp_group_id}"
        )
        token_manager = await start_dp_token_manager(dp_group_id, router_token_address)
        logger.warning(
            f"[Enhanced Scheduler {rank}] DP Token Manager started successfully"
        )
    except Exception as e:
        logger.error(
            f"[Enhanced Scheduler {rank}] DP Token Manager failed to start: {e}"
        )
        # print stack trace
        import traceback

        logger.error(
            f"[Enhanced Scheduler {rank}] DP Token Manager failed to start: {traceback.format_exc()}"
        )
        return

    # Performance statistics
    processed_requests = 0
    start_time = time.time()

    logger.warning(
        f"[Enhanced Scheduler {rank}] Starting to process requests, scheduler listening to requests on {request_address}"
    )
    try:
        while True:
            # Check if there are requests
            if await request_socket.poll(timeout=100):  # 100ms timeout
                try:
                    # Receive request
                    data = await request_socket.recv()
                    request_data = msgpack.unpackb(data, raw=False)

                    logger.debug(
                        f"[Enhanced Scheduler {rank}] Received request: {request_data.get('request_id', 'unknown')}"
                    )

                    # Process request
                    await process_scheduler_request(rank, request_data)
                    processed_requests += 1

                except Exception as e:
                    logger.error(
                        f"[Enhanced Scheduler {rank}] Failed to process request: {e}"
                    )

            # Send statistics periodically
            current_time = time.time()
            # Send statistics every second
            if (current_time - start_time) >= 1.0:
                elapsed = current_time - start_time
                throughput = processed_requests / elapsed

                stats = {
                    "scheduler_id": rank,
                    "dp_group_id": int(
                        dp_config.dp_id
                    ),  # 这里利用传入的dp_id作为stat所属dp group的唯一标识，因为rank值全是0，无法作为参考
                    "running_requests": (
                        len(Backend.ongoing_reqs)
                        if hasattr(Backend, "ongoing_reqs")
                        else 0
                    ),
                    "waiting_requests": (
                        len(getattr(Backend.scheduler, "waiting_queue", []))
                        if hasattr(Backend, "scheduler") and Backend.scheduler
                        else 0
                    ),
                    "pending_tokens": 0,  # TODO: calculate pending tokens
                    "throughput_tokens_per_sec": throughput,
                    "last_update_time": current_time,
                    "heartbeat": True,
                }

                try:
                    stats_data = msgpack.packb(stats)
                    await stats_socket.send(stats_data)
                    logger.debug(
                        f"[Enhanced Scheduler {rank}] throughput: {throughput:.2f}"
                    )
                except Exception as e:
                    logger.error(
                        f"[Enhanced Scheduler {rank}] throughput send failed: {e}"
                    )

                # Reset counter
                processed_requests = 0
                start_time = current_time

    except KeyboardInterrupt:
        logger.warning(f"[Enhanced Scheduler {rank}] Received interrupt signal")
    except Exception as e:
        logger.error(f"[Enhanced Scheduler {rank}] Service exception: {e}")
    finally:
        # Clean up resources
        request_socket.close()
        stats_socket.close()
        context.term()
        logger.warning(f"[Enhanced Scheduler {rank}] Service stopped")


async def process_scheduler_request(rank: int, request_data: dict):
    """处理来自Router的调度请求"""
    try:
        from chitu.task import UserRequest, Task, TaskPool
        from chitu.utils import gen_req_id

        # 构造UserRequest对象
        request_id = request_data.get("request_id", gen_req_id())
        message = request_data.get("message", [])
        max_new_tokens = request_data.get("max_new_tokens", 50)
        temperature = request_data.get("temperature", 1.0)
        top_p = request_data.get("top_p", 1.0)
        top_k = request_data.get("top_k", 50)
        logprobs = request_data.get("logprobs", False)
        top_logprobs = request_data.get("top_logprobs", None)

        # 创建UserRequest
        user_request = UserRequest(
            message=message,
            request_id=request_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )

        # 创建Task
        task = Task(task_id=request_id, req=user_request)

        try:
            from chitu.dp_token_sender import get_dp_token_manager

            dp_group_id = rank  # 使用rank作为DP组ID
            token_manager = get_dp_token_manager(dp_group_id)

            if token_manager is not None:
                # 包装Task以启用Token发送功能
                wrapped_task = token_manager.wrap_task(task)
                TaskPool.add(wrapped_task)
            else:
                # 如果Token Manager未初始化，直接添加原始Task
                TaskPool.add(task)

        except Exception as e:
            # 如果DP Token Manager获取失败，回退到原始Task
            logger.error(f"[Enhanced Scheduler {rank}] DP Token Manager获取失败: {e}")
            TaskPool.add(task)
            logger.warning(f"[Enhanced Scheduler {rank}] 回退到原始任务: {request_id}")

        logger.debug(f"[Enhanced Scheduler {rank}] 任务处理完成: {request_id}")

    except Exception as e:
        logger.error(f"[Enhanced Scheduler {rank}] 处理请求失败: {e}")
        import traceback

        logger.error(f"[Enhanced Scheduler {rank}] 错误详情: {traceback.format_exc()}")


def chitu_terminate():
    if torch.distributed.get_rank() == 0:
        Backend.state = BackendState.Terminated
        terminated_task = PackedTasksBase(
            num_tasks=0,
            payload_type=SerializedPackedTasksPayloadType.TerminateBackend,
        )
        Backend.executor.step(terminated_task)


def chitu_is_terminated():
    return Backend.state == BackendState.Terminated
