# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
from logging import WARNING, INFO, getLogger
import os
import re
import sys
from pathlib import Path
import random
from typing import Optional, Sequence, Any, TypeVar, get_type_hints
from dataclasses import is_dataclass, fields
from types import UnionType
import importlib
import importlib.resources
import threading
import weakref
from collections import deque
from types import UnionType
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import thread as _cf_thread

import numpy as np
import torch
import torch.distributed as dist

from chitu.global_vars import get_global_args
from chitu.import_utils import (
    try_import_and_setup_torch_npu,
    try_import_platform_dep,
    try_import_opt_dep,
)
from chitu.device_type import is_ascend
from chitu.serve.request_id import gen_req_id

logger = getLogger(__name__)


class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor whose worker threads are daemon.

    A worker stuck in a C++ call (e.g. a mooncake RDMA transfer waiting on a
    peer that is shutting down) would otherwise keep the Python interpreter from
    exiting on terminate — hanging the job after the service has drained.
    Daemon threads are not joined at interpreter exit, so a stuck background
    task cannot block process exit.
    """

    def _adjust_thread_count(self):
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            t = threading.Thread(
                name=thread_name,
                target=_cf_thread._worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                ),
                daemon=True,
            )
            t.start()
            self._threads.add(t)
            _cf_thread._threads_queues[t] = self._work_queue


def get_chitu_env(
    name: str, default: Optional[str] = None, *, legacy_names: Sequence[str] = []
) -> Optional[str]:
    assert name.startswith(
        "CHITU_"
    ), f"To chitu developers: Please always use CHITU_ prefix for chitu-specific environment variables."

    if name in os.environ:
        return os.environ[name]

    for legacy_name in legacy_names:
        if legacy_name in os.environ:
            logger.warning(
                f"Environment variable {legacy_name} is recognized but deprecated. Please Use {name} instead."
            )
            return os.environ[legacy_name]

    return default


def get_chitu_bool_env(
    name: str, default: Optional[bool] = None, *, legacy_names: Sequence[str] = []
) -> Optional[bool]:
    if default is not None:
        default_str = "1" if default else "0"
    else:
        default_str = None
    str_val = get_chitu_env(name, default_str, legacy_names=legacy_names)
    if str_val is not None and str_val.lower() in {"1", "true", "yes", "on"}:
        return True
    elif str_val is not None and str_val.lower() in {"0", "false", "no", "off"}:
        return False
    else:
        return None


def should_pretty_log(args=None) -> bool:
    """Whether pretty (interactive) terminal output should be printed.

    Follows the `pretty_log` serve config:
      - "auto": only when stdout and stderr are attached to a terminal.
      - "true": always, even when redirected (e.g. `2>&1 | tee`).
      - "false": never.
    When `args` is omitted or does not define `pretty_log` (e.g. global args
    are not initialized yet during logging setup), fall back to terminal
    detection.
    """
    pretty_log = str(getattr(args, "pretty_log", "auto")).lower() if args else "auto"
    if pretty_log in ("true", "1"):
        return True
    if pretty_log in ("false", "0"):
        return False
    return sys.stdout.isatty() and sys.stderr.isatty()


_regex_special_chars = set(".^$*+?{}[]|()")


def is_layer(layer_name: str, full_name: str) -> bool:
    if any(ch in _regex_special_chars for ch in layer_name):
        return re.search(layer_name, full_name) is not None
    else:
        return (
            f".{layer_name}." in full_name
            or full_name.startswith(layer_name + ".")
            or full_name.endswith("." + layer_name)
        )


def get_config_dir_path():
    return str(importlib.resources.files("chitu") / "config")


def parse_dtype(
    name: str,
) -> torch.dtype:
    if name == "float32":
        return torch.float32
    elif name == "float16":
        return torch.float16
    elif name == "bfloat16":
        return torch.bfloat16
    elif name == "float8_e4m3fn":
        return torch.float8_e4m3fn
    elif name == "float4_e2m1":
        return torch.uint8
    else:
        assert False


def ceil_div(a, b):
    return (a + b - 1) // b


def is_power_of_two(n: int) -> bool:
    return (n != 0) and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    return 1 if n == 0 else 2 ** (n - 1).bit_length()


def proportion_split(
    tensor: torch.Tensor, proportion: list[int], dim: int = 0
) -> tuple[torch.Tensor, ...]:
    """
    Similar to torch.split, but do not require a split-size list summing up
    exactly to the size of the tensor. Instead it splits the tensor proportionally.

    E.g. 1, splitting a dimension of size 10 with proprotion [1, 4] will result in
    two tensors of the dimension in size 2 and 8.

    E.g. 2, splitting a dimension of size 3 with proprotion [10, 20] will result in
    two tensors of the dimension in size 1 and 2.
    """

    tot = sum(proportion)
    if tot == tensor.shape[dim]:
        return torch.split(tensor, proportion, dim=dim)
    elif tot > tensor.shape[dim]:
        if tot % tensor.shape[dim] != 0:
            raise ValueError(
                f"Proportions {proportion} sum up to {tot}, which must be a multiple "
                f"or a factor of the dimension size {tensor.shape[dim]}"
            )
        ratio = tot // tensor.shape[dim]
        return torch.split(tensor, [p // ratio for p in proportion], dim=dim)
    else:
        if tensor.shape[dim] % tot != 0:
            raise ValueError(
                f"Proportions {proportion} sum up to {tot}, which must be a multiple "
                f"or a factor of the dimension size {tensor.shape[dim]}"
            )
        ratio = tensor.shape[dim] // tot
        return torch.split(tensor, [p * ratio for p in proportion], dim=dim)


def pad_tensor(x, target_size, dim=0, value=0):
    current_size = x.size(dim)
    assert current_size <= target_size

    if current_size == target_size:
        return x

    pad_size = target_size - current_size
    pad_pattern = [0] * (x.dim() * 2)
    pad_idx = (x.dim() - dim - 1) * 2 + 1
    pad_pattern[pad_idx] = pad_size

    padded_x = torch.nn.functional.pad(x, pad_pattern, mode="constant", value=value)

    return padded_x


class DataSaver:
    """数据保存装饰器类"""

    def __init__(
        self,
        max_files: int = 5,
        save_prob: float = 0.1,
        save_dir: str = "test_data",
        save_tensors: list[str] = [],
        save_attrs: list[str] = [],
        save_locals: list[str] = [],
        save_return: bool = True,
    ):
        self.max_files = max_files
        self.save_prob = save_prob
        self.save_dir = Path(save_dir + "/")
        self.saved_files: list[str] = []  # 存储所有保存的文件名
        self.replaceable_files: list[str] = []  # 存储可替换的文件名
        self.call_count = 0
        self.random = random.Random(42)  # 使用固定种子确保可重复性
        self.save_return = save_return  # 是否默认保存函数返回值

        # 获取当前机器编号和卡号
        self.machine_id = int(os.environ.get("RANK", 0)) // 8  # 假设每台机器8张卡
        self.card_id = int(os.environ.get("RANK", 0)) % 8

        # 创建保存目录
        self.save_dir.mkdir(exist_ok=True)

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数的参数名和值
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # 获取参数名和值的映射
            param_dict = bound_args.arguments
            param_names = list(param_dict.keys())
            args_names = list(param_dict.values())

            # 在函数执行前收集要保存的输入数据
            input_data = {}

            # 保存指定的输入张量
            for i, name in enumerate(param_names):
                if name in self.save_tensors and i < len(args_names):
                    # 对于需要保存的张量，创建副本
                    logger.info(f"clone input param: {name}, its val: {args_names[i]}")
                    if isinstance(args_names[i], torch.Tensor):
                        input_data[name] = args_names[i].clone()
                    else:
                        input_data[name] = args_names[i]

            # 保存指定的关键字参数张量
            for name in self.save_tensors:
                if name in kwargs:
                    # logger.info(f"clone input kwarg: {name}, its val:{kwargs[name]}")
                    if isinstance(kwargs[name], torch.Tensor):
                        input_data[name] = kwargs[name].clone()
                    else:
                        input_data[name] = kwargs[name]

            # 保存指定的类成员变量
            for name in self.save_attrs:
                # logger.info(f"clone input attr: {name}, its val:{getattr(args[0], name)}")
                if hasattr(args[0], name):  # args[0] 是 self
                    attr = getattr(args[0], name)
                    if isinstance(attr, torch.Tensor):
                        input_data[name] = attr.clone()
                    else:
                        input_data[name] = attr

            # 执行原始函数
            try:
                # 适配多个返回值的情况
                if isinstance(func(*args, **kwargs), tuple):
                    result = func(*args, **kwargs)
                else:
                    result = (func(*args, **kwargs),)
            except Exception as e:
                logger.error(f"保存数据失败: {e}, 只保存输入文件")
                save_data = input_data.copy()
                torch.save(
                    save_data,
                    self.save_dir
                    / f"{func.__name__}_m{self.machine_id}_c{self.card_id}_exec_error.pt",
                )
                raise e

            # 决定是否保存数据
            self.call_count += 1
            if (
                len(self.saved_files) < self.max_files
                or self.random.random() < self.save_prob
            ):
                # 合并输入数据和输出数据
                save_data = input_data.copy()

                # 默认保存函数返回结果
                if self.save_return:
                    # logger.info(f"clone return result: {result}, its val:{result}")
                    save_data["func_return"] = result

                # 获取 layer_index 和 decode_step
                layer_index = (
                    args[0].layer_id if args and hasattr(args[0], "layer_id") else 0
                )
                decode_step = 0
                if args and hasattr(args[0], "cache"):
                    kv_cache = args[0].cache
                    if hasattr(kv_cache, "curr_tids") and kv_cache.curr_tids:
                        req_id = kv_cache.curr_tids[0]
                        decode_step = kv_cache.tid_to_cached_len.get(req_id, 0)

                # 获取模型名称和数据类型
                try:
                    args = get_global_args()
                    model_name = args.models.type
                    model_path = args.models.ckpt_dir
                except Exception:
                    model_name = "unknown"
                    model_path = "unknown"
                model_dtype = (
                    "fp4" if get_global_args().infer.npu_fusion_fp4 else "bf16"
                )
                # 添加推理步骤信息
                save_data["inference_info"] = {
                    "machine_id": self.machine_id,
                    "card_id": self.card_id,
                    "call_count": self.call_count,
                    "function_name": func.__name__,
                    "batch_size": (
                        args_names[0].shape[0]
                        if isinstance(args_names[0], torch.Tensor)
                        else None
                    ),
                    "decode_step": decode_step,
                    "layer_index": layer_index,
                    "model_name": model_name,
                    "model_dtype": model_dtype,
                    "model_path": model_path,
                }

                # 生成文件名
                filename = f"{func.__name__}_{model_name}_{model_dtype}_m{self.machine_id}_c{self.card_id}_l{layer_index}_d{decode_step}_{self.call_count}.pt"

                if self.call_count == 1:
                    # 第一次调用，永久保存
                    logger.info(f"首次调用，永久保存数据到 {self.save_dir}/{filename}")
                    torch.save(save_data, self.save_dir / filename)
                    self.saved_files.append(filename)
                elif (
                    len(self.replaceable_files) < self.max_files - 1
                ):  # 减1是因为要保留第一次的文件
                    # 如果还没达到最大可替换文件数量，直接保存
                    logger.info(f"保存数据到 {self.save_dir}/{filename}")
                    torch.save(save_data, self.save_dir / filename)
                    self.saved_files.append(filename)
                    self.replaceable_files.append(filename)
                else:
                    # 如果已经达到最大可替换文件数量，随机替换一个可替换的文件
                    replace_idx = self.random.randint(
                        0, len(self.replaceable_files) - 1
                    )
                    old_filename = self.replaceable_files[replace_idx]
                    # 删除旧文件
                    (self.save_dir / old_filename).unlink(missing_ok=True)
                    # 保存新文件
                    torch.save(save_data, self.save_dir / filename)
                    self.saved_files[self.saved_files.index(old_filename)] = filename
                    self.replaceable_files[replace_idx] = filename
                    logger.info(
                        f"替换文件 {self.save_dir}/{old_filename} 为 {self.save_dir}/{filename}"
                    )

            return result

        return wrapper


def log_with_rank(msg, rank=0, prefix="", level=WARNING, logger=logger):
    """
    根据指定的 rank 输出日志，默认只输出 rank 0 的日志

    Args:
        msg: 日志消息
        rank: 指定要输出日志的 rank，默认为 0
        prefix: 日志前缀
        level: 日志级别，默认为 logging.INFO
        logger: logger 实例，默认为当前模块的 logger
    """
    import torch.distributed as dist

    current_rank = dist.get_rank() if dist.is_initialized() else 0

    if current_rank == rank:
        if prefix:
            msg = f"[Rank {current_rank}] {prefix}{msg}"
        else:
            msg = f"[Rank {current_rank}] {msg}"

        if level == INFO:
            logger.info(msg)
        elif level == logging.WARNING:
            logger.warning(msg)
        elif level == logging.ERROR:
            logger.error(msg)
        elif level == logging.DEBUG:
            logger.debug(msg)
        else:
            logger.log(level, msg)


torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


def invalidate_cached_property(obj, name):
    """
    Suppose `obj` has a `functools.cached_property` named `name`, this function invalidate the cache

    `@cached_property` properties can be invalidated by just deleting them. See
    https://docs.python.org/3/library/functools.html#functools.cached_property

    However, we shall NOT do the following:
    ```
    if hasattr(obj, name):
        delattr(obj, name)
    ```

    because `hasattr` evaluates the property first, which is redundant.

    Therefore, we shall try and catch
    """

    try:
        delattr(obj, name)
    except AttributeError:
        pass


def try_get_profiler(
    profiler_dir: str,
    wait: int = 0,
    warmup: int = 0,
    active: int = 1000,
    repeat: int = 1,
    with_stack: bool = False,
):
    if has_torch_npu:
        from chitu.npu_utils import try_get_npu_profiler

        return try_get_npu_profiler(
            profiler_dir, wait, warmup, active, repeat, with_stack
        )
    else:
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=repeat
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                dir_name=profiler_dir,
                worker_name=f"rank_{torch.distributed.get_rank()}",
                use_gzip=True,
            ),
            record_shapes=False,
            profile_memory=False,
            with_stack=with_stack,
            with_modules=False,
            with_flops=False,
        )


def gather_str_to_dst_rank(strings: str, dst: int, group=None) -> Optional[list]:
    """
    将所有rank的string字符串收集到dst rank

    :param strings: 当前rank的字符串
    :type strings: str
    :param group: 通信组
    :type dst: int
    :param dst: 目标rank
    :type dst: int
    :return: 仅在rank0返回所有rank的字符串列表，其他rank返回空列表
    :rtype: list
    """
    group_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    gather_list = []

    # Use broadcast instead of gather to avoid NCCL issues
    # 使用dist.gather在5090卡上会报错
    for src_rank in range(group_size):
        if rank == src_rank:
            data = [strings]
        else:
            data = [None]

        dist.broadcast_object_list(data, src=src_rank, group=group)

        if rank == dst:
            gather_list.append(data[0])

    return gather_list if rank == dst else None


def create_tensor(data, device, dtype=None, sync_free=True):
    if sync_free and not is_ascend():
        pin_memory = not isinstance(data, np.ndarray)
        return torch.tensor(data, dtype=dtype, pin_memory=pin_memory).to(
            device=device, non_blocking=True
        )
    return torch.tensor(data, dtype=dtype, device=device)


def dataclass_to_dict(obj: Any) -> dict[str, Any] | Any:
    """
    将dataclass转为字典，支持field为 dataclass或Union[dataclass, ...] 的嵌套转换
    """
    if not is_dataclass(obj):
        return obj
    return {
        field.name: dataclass_to_dict(getattr(obj, field.name)) for field in fields(obj)
    }


T = TypeVar("T")


def dataclass_from_dict(data: Any, cls: type[T]) -> T:
    """
    将字典转为dataclass，支持field为 dataclass或Union[dataclass, ...] 的嵌套转换
    """
    if not is_dataclass(cls) or not isinstance(data, dict):
        return data
    kwargs = {}
    type_hints = get_type_hints(cls)
    for field in fields(cls):
        fcls = type_hints[field.name]
        if isinstance(fcls, UnionType):
            for fcls in fcls.__args__:
                if is_dataclass(fcls):
                    break
        kwargs[field.name] = dataclass_from_dict(data[field.name], fcls)
    return cls(**kwargs)


def prefetch_state_dict(state_dict: dict[str, torch.Tensor], max_workers: int = 16):
    """
    多线程将state_dict预取到内存，不阻塞主线程
    """
    state_dict = state_dict.copy()  # 避免state_dict之后被修改

    def _prefetch(key: str):
        # 假设绝大部分tensor都是磁盘mmap到内存再view，此时只需做一次clone即可完成预取
        state_dict[key].clone()

    executor = DaemonThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="prefetch"
    )
    executor.map(_prefetch, state_dict)


def fetch_state_dict_to_device(
    state_dict: dict[str, torch.Tensor], max_workers: int = 16
):
    """
    多线程将 state_dict non-blocking 搬运到 GPU，inplace 替换 CPU tensor 为
    GPU tensor。不做 clone，缺页触发 + pageable→pinned→DMA 一步完成。
    """
    device = torch.cuda.current_device()
    main_stream = torch.cuda.current_stream()
    work = deque(state_dict.keys())
    if not work:
        return

    events: list[torch.cuda.Event] = []

    def _worker():
        torch.cuda.set_device(device)  # Required, or we will init allocator on GPU 0
        while True:
            try:
                key = work.popleft()
            except IndexError:
                break
            try:
                state_dict[key] = state_dict[key].to(device=device, non_blocking=True)
            except Exception as e:
                # A failed .to() leaves partial weights on GPU — do not let other
                # workers keep moving weights for a doomed process. This is startup
                # (no reporter yet): crash immediately from this thread. Local
                # import avoids a module cycle.
                logger.exception("fetch_to_gpu: .to() failed for %s", key)
                from chitu.serve.crash import report_and_exit

                report_and_exit(
                    f"fetch_state_dict_to_device failed: {e!r}", immediate=True
                )
                return  # unreachable in practice; defensive
        event = torch.cuda.Event()
        event.record()
        # list.append is atomic under the GIL; the main thread reads events only
        # after joining all workers (happens-before), so no lock is needed here.
        events.append(event)

    n_workers = min(max_workers, len(work))
    threads = [
        threading.Thread(target=_worker, name="fetch_to_gpu") for _ in range(n_workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for event in events:
        main_stream.wait_event(event)
