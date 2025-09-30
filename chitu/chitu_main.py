# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import operator
import os
from logging import getLogger
import psutil
import importlib.util

import torch
import torch.distributed

from chitu.backend import Backend, BackendState
from chitu.cache_manager import PagedKVCacheManager
from chitu.device_type import is_nvidia
from chitu.executor import Executor, BatchResult
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
    DPTaskCollector,
)
from chitu.utils import gen_req_id, try_import_opt_dep, try_import_and_setup_torch_npu
from chitu.schemas.utils import ModelConfigResolver
from chitu.utils import ceil_div

numa, has_numa = try_import_opt_dep("numa", "cpu")
cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)


def get_additional_block_num(cache_manager, memory_utilization=0.98):
    """
    Calculate additional block numbers based on available memory.
    Works on both CPU and GPU machines.

    Args:
        cache_manager: The cache manager object.
        memory_utilization: Fraction of GPU/CPU memory to use (default: 0.98).

    Returns:
        Number of additional blocks that can be allocated.
    """

    def tuple_product(t):
        return functools.reduce(operator.mul, t, 1)

    block_mem = (
        2
        * cache_manager.block_size
        * tuple_product(cache_manager.k_shape_per_sample)
        * cache_manager.num_layers
    )
    block_mem *= (1 if cache_manager.k_shape_per_sample is not None else 0) + (
        1 if cache_manager.v_shape_per_sample is not None else 0
    )

    if get_global_args().infer.op_impl == "cpu":
        process = psutil.Process(os.getpid())
        current_process_mem = process.memory_info().vms
        additional_memory = (
            psutil.virtual_memory().total * memory_utilization - current_process_mem
        )
        num_blocks = int(additional_memory) // block_mem
        return max(0, num_blocks)

    torch.cuda.synchronize()  # Wait for all kernels to finish before we can get peak memory usage
    _, total_memory = torch.cuda.mem_get_info(0)
    peak_memory = torch.cuda.memory_stats(0)["allocated_bytes.all.peak"]
    torch.cuda.empty_cache()
    torch_allocated_bytes = torch.cuda.memory_stats(0)["allocated_bytes.all.current"]
    total_allocated_bytes = (
        torch.cuda.mem_get_info(0)[1] - torch.cuda.mem_get_info(0)[0]
    )
    non_torch_allocations = total_allocated_bytes - torch_allocated_bytes
    if non_torch_allocations > 0:
        peak_memory += non_torch_allocations
    additional_kv_cache_memory = total_memory * memory_utilization - peak_memory

    num_blocks = int(additional_kv_cache_memory) // block_mem
    return max(0, num_blocks)


def _auto_set_num_blocks_after_warmup(args):
    if args.infer.cache_type == "paged" and args.infer.num_blocks == -1:
        assert isinstance(Backend.cache_manager, PagedKVCacheManager)
        additional_blocks = get_additional_block_num(
            Backend.cache_manager, args.infer.memory_utilization
        )
        new_num_block = Backend.cache_manager.num_blocks + additional_blocks

        if torch.distributed.get_world_size() > 1:
            new_num_block_tensor = torch.tensor(new_num_block).cuda()
            torch.distributed.all_reduce(
                new_num_block_tensor, torch.distributed.ReduceOp.RedOpType.MIN
            )
            new_num_block = new_num_block_tensor.item()

        get_global_args().infer.num_blocks = new_num_block
        Backend.cache_manager.realloc(new_num_block)
        if torch.distributed.get_rank() == 0:
            Backend.scheduler.reset_kvcache_block_threshold()


def _warmup_via_taskpool(args):
    if args.infer.pp_size > 1:
        logger.warning("Warming-up is not supported when PP is enabled. Skipping")
        if args.infer.cache_type == "paged":
            assert isinstance(Backend.cache_manager, PagedKVCacheManager)
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
                if torch.distributed.get_rank() == 0:
                    Backend.scheduler.reset_kvcache_block_threshold()
        return

    rank = torch.distributed.get_rank()

    logger.info("Starting inference system warmup...")

    init_cache_static()
    num_warmup_reqs = args.infer.max_reqs
    prefill_chunk_size = args.infer.prefill_chunk_size
    if prefill_chunk_size is not None:
        warmup_seq_len = max(
            min(
                prefill_chunk_size // num_warmup_reqs,
                args.infer.max_seq_len - 1,
            ),
            1,
        )
    else:
        logger.warning(
            "infer.prefill_chunk_size is not set, GPU memory usage estimation may be incorrect (may cause OOM)"
        )
        warmup_seq_len = 1
        prefill_chunk_size = args.infer.max_seq_len * args.infer.max_reqs
    if rank == 0:
        for i in range(num_warmup_reqs):
            req = MockFixedLengthedUserRequest(
                warmup_seq_len,
                f"{gen_req_id()}",
                max_new_tokens=2,  # 1 prefill + 1 decode
                temperature=0.7,
                top_k=1,
            )
            task = Task(f"{req.request_id}", req, stop_with_eos=False)
            TaskPool.add(task)
        logger.info(f"Added {num_warmup_reqs} warmup requests to TaskPool")
        Backend.scheduler.start_warmup()

    num_required_prefill_schedules = ceil_div(
        ceil_div(warmup_seq_len * num_warmup_reqs, prefill_chunk_size),
        args.infer.dp_size,
    )  # The number of times the prefill tasks needs to be scheduled to be completed

    # prefill
    for _ in range(num_required_prefill_schedules):
        chitu_run()

    if rank == 0:
        assert (
            len(TaskPool.pool) == num_warmup_reqs
        ), "Tasks should still be there after prefill"

    # decode
    chitu_run()

    if rank == 0:
        assert len(TaskPool.pool) == 0, "TaskPool should be empty after warmup"
        Backend.scheduler.end_warmup()

    # An extra run is needed to receive "finalize KV cache" message from rank 0
    if rank > 0:
        chitu_run()

    logger.info("Inference system warmup completed")


def _warmup_backend_direct(args, decode_steps: int = 2):
    logger.info("Starting local backend warmup (direct)...")
    init_cache_static()
    # Minimal request
    req_id = "__warmup__"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    tokens = torch.tensor([1], device=torch.device(local_rank), dtype=torch.int64)
    # Prefill
    from chitu.batched_seq_len import BatchedSeqLen

    Backend.cache_manager.prepare_cache_prefill([req_id], [1])
    # output_token_offsets 需要指向每个序列的最后一个 token 下标
    output_token_offsets = torch.tensor(
        [tokens.size(0) - 1], dtype=torch.int32, device=tokens.device
    )
    _ = Backend.model.prefill(tokens, output_token_offsets)
    Backend.cache_manager.finalize_cache_all_prefill()
    # Decode steps
    for _ in range(max(1, decode_steps)):
        Backend.cache_manager.prepare_cache_decode([req_id])
        # decode 期望 tokens 为 1D [batch], 否则 embedding 输出为 3D 导致后续形状不匹配
        step_token = torch.tensor(
            [0], device=torch.device(local_rank), dtype=torch.int64
        )
        batch_size = step_token.size(0)  # = 1
        _ = Backend.model.decode(step_token, batch_size)
        Backend.cache_manager.finalize_cache_single_decode([req_id])
    # Clean KV for this request
    Backend.cache_manager.finalize_cache_all_decode(req_id)
    logger.info("Local backend warmup (direct) completed")


def warmup_engine(args):
    # Router 进程不做 warmup
    try:
        if getattr(args.dp_config.router, "is_router", False):
            return
    except Exception:
        pass

    # PP>1 + paged：保持原跳过与兜底策略
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
            if torch.distributed.get_rank() == 0:
                Backend.scheduler.reset_kvcache_block_threshold()
        return

    # PD→direct，非PD→taskpool
    pd_enabled = False
    try:
        pd_enabled = (
            hasattr(args.dp_config.router, "pd_disaggregation")
            and args.dp_config.router.pd_disaggregation.enabled
        )
    except Exception:
        pd_enabled = False

    runner = "direct" if pd_enabled else "taskpool"
    if runner == "taskpool":
        _warmup_via_taskpool(args)
    else:
        _warmup_backend_direct(args, decode_steps=2)
    _auto_set_num_blocks_after_warmup(args)


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
    if hasattr(args.models, "processor_path") and args.models.processor_path is None:
        logger.info(
            f"Using {args.models.ckpt_dir} as the path to processor. If the processor has a different path, please set in command line by adding `models.processor_path=<path>`"
        )
        args.models.processor_path = args.models.ckpt_dir


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

    if (
        args.infer.prefill_chunk_size is not None
        and args.infer.prefill_chunk_size > args.infer.max_reqs * args.infer.max_seq_len
    ):
        logger.warning(
            f"infer.prefill_chunk_size ({args.infer.prefill_chunk_size}) is larger than "
            f"infer.max_reqs ({args.infer.max_reqs}) * infer.max_seq_len "
            f"({args.infer.max_seq_len}), which has no effect. Reducing it to infer.max_reqs "
            f" * infer.max_seq_len."
        )
        args.infer.prefill_chunk_size = args.infer.max_reqs * args.infer.max_seq_len

    if args.infer.prefill_chunk_size is not None:
        if args.infer.dp_size > 1:
            logger.warning(
                "Disabling infer.prefill_chunk_size because it is not compatible with DP yet"
            )
            args.infer.prefill_chunk_size = None
        if args.infer.pp_size > 1 and args.infer.cache_type == "skew":
            logger.warning(
                "Disabling infer.prefill_chunk_size because it is not compatible with PP+skew yet"
            )
            args.infer.prefill_chunk_size = None

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

    if args.infer.use_cuda_graph == "auto":
        spec = importlib.util.find_spec("deep_ep")
        if args.models.name in [
            "Mixtral-8x7B-Instruct-v0.1",
            "Qwen3-30B-A3B-mix-fp4-fp8",
            "Qwen3-Next-80B-A3B-Instruct",
        ]:
            args.infer.use_cuda_graph = False
        elif args.infer.dp_size > 1 and spec is None:
            args.infer.use_cuda_graph = False
        elif args.infer.attn_type == "ref":
            args.infer.use_cuda_graph = False
        elif args.infer.op_impl is not None and args.infer.op_impl == "cpu":
            args.infer.use_cuda_graph = False
        elif (
            args.models is not None
            and str(args.models).find("'backend': 'cpuinfer'") != -1
        ):
            args.infer.use_cuda_graph = False
        elif args.models is not None and str(args.models).find("'type': 'mixq'") != -1:
            args.infer.use_cuda_graph = False
        elif (
            args.infer.attn_type == "npu"
            and args.infer.cache_type == "paged"
            and (args.models.type is not None and args.models.type == "deepseek-v3")
        ):
            args.infer.use_cuda_graph = False
        else:
            args.infer.use_cuda_graph = True

    # Check checkpoint exists
    check_checkpoint_path(args)

    # Parse model configuration, supporting dynamic reading from config.json files
    # Uses $(config.json:field_name) syntax, e.g., n_heads: "$(config.json:head_dim)"
    model_resolver = ModelConfigResolver()
    args.models = model_resolver.process_config_dict(args.models, args.models.ckpt_dir)

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
    logger.info("Chitu has been initialized")


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

        tokens = Backend.executor.step(tasks)

        # postprocess
        if len(Backend.last_batch_results) > 0 and Backend.args.infer.dp_size <= 1:
            Backend.executor.postprocess_async_part(
                Backend.last_batch_results.popleft()
            )

        if DPTaskCollector.has_available_tasks():
            tasks = DPTaskCollector.get_total_packedtasks()
            task_ids = tasks.task_ids
            curr_batch_result = BatchResult(
                num_tasks=tasks.num_tasks,
                tasks=tasks.output_tasks,
                next_tokens=tokens,
                return_logprobs=tasks.return_logprobs,
                logprobs=tasks.logprobs.cpu() if tasks.return_logprobs else None,
                token_idxs=tasks.token_idxs.cpu() if tasks.return_logprobs else None,
            )
            DPTaskCollector.clear()
        else:
            curr_batch_result = Backend.executor.postprocess_sync_part(tasks, tokens)

        Backend.last_batch_results.append(curr_batch_result)
        removed_decode_task_ids = Backend.scheduler.update(task_ids)
        remove_kvcache_all_device(removed_decode_task_ids)
    elif len(Backend.last_batch_results) > 0:
        # ensure the last batch result is processed
        Backend.executor.postprocess_async_part(Backend.last_batch_results.popleft())


def _update_ongoing_tasks():
    unwait_tasks: list[PackedTasks] = []
    logits_list: list[torch.Tensor] = []
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
    dp_id = args.dp_config.dp_id
    if rank != 0:
        logger.warning(
            f"[Enhanced Scheduler {dp_id}] only main rank of dp group start Enhanced Scheduler service"
        )
        return

    """Start Enhanced Scheduler service, listen to ZMQ requests"""
    import zmq
    import zmq.asyncio
    import msgpack
    import time

    logger.warning(f"[Enhanced Scheduler {dp_id}] Starting...")

    # Initialize ZMQ
    context = zmq.asyncio.Context()

    # Receive request socket
    request_socket = context.socket(zmq.PULL)
    request_port = dp_config.scheduler_base_port
    request_address = f"tcp://{dp_config.scheduler_base_host}:{request_port}"
    request_socket.bind(request_address)
    logger.warning(
        f"[Enhanced Scheduler {dp_id}] Listening to requests: {request_address}"
    )

    # Send statistics socket
    stats_socket = context.socket(zmq.PUSH)
    stats_address = f"tcp://{dp_config.router.host}:{dp_config.router.stats_port}"  # Router stats port
    stats_socket.connect(stats_address)
    logger.warning(
        f"[Enhanced Scheduler {dp_id}] connected to stats service: {stats_address}"
    )

    # Start DP Token Manager
    try:
        from chitu.dp_token_sender import start_dp_token_manager

        dp_id = get_global_args().dp_config.dp_id
        router_token_address = f"tcp://{dp_config.router.host}:{dp_config.router.token_port}"  # Token Router listen address

        logger.warning(
            f"[Enhanced Scheduler {dp_id}] Starting DP Token Manager, group ID={dp_id}"
        )
        await start_dp_token_manager(dp_id, router_token_address)
        logger.warning(
            f"[Enhanced Scheduler {dp_id}] DP Token Manager started successfully"
        )
    except Exception as e:
        logger.error(
            f"[Enhanced Scheduler {dp_id}] DP Token Manager failed to start: {e}"
        )
        # print stack trace
        import traceback

        logger.error(
            f"[Enhanced Scheduler {dp_id}] DP Token Manager failed to start: {traceback.format_exc()}"
        )
        return

    # Performance statistics
    processed_requests = 0
    start_time = time.time()

    logger.warning(
        f"[Enhanced Scheduler {dp_id}] Starting to process requests, scheduler listening to requests on {request_address}"
    )
    try:
        while True:
            # Check if there are requests
            if await request_socket.poll(timeout=100):  # 100ms timeout
                try:
                    # Receive request
                    data = await request_socket.recv()
                    request_data = msgpack.unpackb(data, raw=False)

                    logger.info(
                        f"[Enhanced Scheduler {dp_id}] Received request: {request_data.get('request_id', 'unknown')}"
                    )

                    # Process request
                    await process_scheduler_request(rank, request_data)
                    processed_requests += 1

                except Exception as e:
                    logger.error(
                        f"[Enhanced Scheduler {dp_id}] Failed to process request: {e}"
                    )

            # Send statistics periodically
            current_time = time.time()
            # Send statistics every second
            if (current_time - start_time) >= 1.0:
                elapsed = current_time - start_time
                throughput = processed_requests / elapsed

                stats = {
                    "scheduler_id": dp_config.dp_id,
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
                        f"[Enhanced Scheduler {dp_id}] throughput: {throughput:.2f}"
                    )
                except Exception as e:
                    logger.error(
                        f"[Enhanced Scheduler {dp_id}] throughput send failed: {e}"
                    )

                # Reset counter
                processed_requests = 0
                start_time = current_time

    except KeyboardInterrupt:
        logger.warning(f"[Enhanced Scheduler {dp_id}] Received interrupt signal")
    except Exception as e:
        logger.error(f"[Enhanced Scheduler {dp_id}] Service exception: {e}")
    finally:
        # Clean up resources
        request_socket.close()
        stats_socket.close()
        context.term()
        logger.warning(f"[Enhanced Scheduler {dp_id}] Service stopped")


async def process_scheduler_request(rank: int, request_data: dict):
    """Handle scheduling requests from Router"""
    try:
        # Build UserRequest object
        request_id = request_data.get("request_id", gen_req_id())
        message = request_data.get("message", [])
        max_new_tokens = request_data.get("max_new_tokens", 50)
        temperature = request_data.get("temperature", 1.0)
        top_p = request_data.get("top_p", 1.0)
        top_k = request_data.get("top_k", 50)
        logprobs = request_data.get("logprobs", False)
        top_logprobs = request_data.get("top_logprobs", None)

        # Create UserRequest
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

        # Create Task, honoring stop/ignore_eos semantics from request_data
        stop_with_eos = True
        if request_data.get("ignore_eos"):
            stop_with_eos = False
        elif not request_data.get("stop_with_eos"):
            stop_with_eos = False

        task = Task(task_id=request_id, req=user_request, stop_with_eos=stop_with_eos)

        try:
            from chitu.dp_token_sender import get_dp_token_manager

            dp_id = get_global_args().dp_config.dp_id
            token_manager = get_dp_token_manager(dp_id)
            # ensure token manager started
            await token_manager.start()
            if token_manager is not None:
                # Wrap Task to enable token sending
                wrapped_task = token_manager.wrap_task(task)
                TaskPool.add(wrapped_task)
            else:
                # If Token Manager not initialized, add the original Task directly
                TaskPool.add(task)

        except Exception as e:
            # If DP Token Manager acquisition fails, fall back to original Task
            logger.error(
                f"[Enhanced Scheduler {dp_id}] Failed to get DP Token Manager: {e}"
            )
            TaskPool.add(task)
            logger.warning(
                f"[Enhanced Scheduler {dp_id}] Fallback to original task: {request_id}"
            )

        logger.debug(f"[Enhanced Scheduler {dp_id}] Request handled: {request_id}")

    except Exception as e:
        logger.error(f"[Enhanced Scheduler {dp_id}] Failed to process request: {e}")
        import traceback

        logger.error(
            f"[Enhanced Scheduler {dp_id}] Error details: {traceback.format_exc()}"
        )


def chitu_start():
    Backend.state = BackendState.Running


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
