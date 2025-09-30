# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0


import functools
import logging
from logging import WARNING, INFO, getLogger
import os
import re
from pathlib import Path
import random
from typing import Any
import socket
import site

import torch
import importlib
import importlib.resources
from chitu.device_type import is_ascend

from chitu.global_vars import get_global_args

logger = getLogger(__name__)


def try_import_opt_dep(pkg_name: str, opt_dep_name: str) -> tuple[Any, bool]:
    """
    Import an optional dependency.

    The package name and optional dependency name should be consistent with the listing
    in `setup.py`. For example, you can list a Python package `my_quant_wxax` in the
    `quant` extra of `setup.py`, then you can use this function like `try_import_opt_dep('my_quant_wxax', 'quant')`,
    and the user may install the optional dependency like `pip install chitu[quant]`.

    DO NOT use this function to import platform-specific dependencies that users are unable
    to install at their will. Use `try_import_platform_dep` instead.

    Args:
        pkg_name (str): The name of the Python package to import.
        opt_dep_name (str): The name of the optional dependency category in `setup.py`.

    Returns:
        [0]: The imported module if successful, or a dummy object that raises an ImportError.
        [1]: A boolean indicating whether the import was successful.
    """

    # Keep this sync with get_requires.py
    opt_deps = {
        "quant",
        "muxi_layout_kernels",
        "muxi_w8a8_kernels",
        "ascend_kernels",
        "flash_attn",
        "flashinfer",
        "flash_mla",
        "deep_gemm",
        "deep_ep",  # [TODO] add installation support
        "cpu",
    }
    assert (
        opt_dep_name in opt_deps
    ), f"To chitu developers: Please don't use {opt_dep_name} as an optional dependency name, it is not listed in get_requires.py."

    try:
        return importlib.import_module(pkg_name), True
    except ImportError as e:

        class ReportErrorWhenUsed:
            def __init__(self, e):
                self.root_cause = e

            def __getattr__(self, item):
                raise ImportError(
                    f"Optional dependency '{opt_dep_name}' is not installed. "
                    f"Please refer to README.md for installation instructions."
                ) from self.root_cause

        return ReportErrorWhenUsed(e), False


def try_import_platform_dep(pkg_name: str) -> tuple[Any, bool]:
    """
    Import a dependency that may not be available on all platforms.

    DO NOT use this functions to import optional dependencies that users can pick. Use `try_import_opt_dep` instead.

    Args:
        pkg_name (str): The name of the Python package to import.

    Returns:
        [0]: The imported module if successful, or a dummy object that raises an ImportError.
        [1]: A boolean indicating whether the import was successful.
    """

    try:
        return importlib.import_module(pkg_name), True
    except ImportError as e:

        class ReportErrorWhenUsed:
            def __init__(self, e):
                self.root_cause = e

            def __getattr__(self, item):
                raise ImportError(
                    f"Chitu does not support this case because '{pkg_name}' is not present on this platform. "
                    f"This is likely a bug of Chitu."
                ) from self.root_cause

        return ReportErrorWhenUsed(e), False


_torch_npu_has_set_up = False


def try_import_and_setup_torch_npu():
    """
    Try importing `torch_npu`. If successful, also do some setup.
    """

    global _torch_npu_has_set_up

    torch_npu, has_torch_npu = try_import_platform_dep("torch_npu")

    if has_torch_npu and not _torch_npu_has_set_up:
        # Make "torch.cuda" point to NPU devices
        from torch_npu.contrib import transfer_to_npu

        torch.cuda.CUDAGraph = torch.npu.NPUGraph

        # Allow using NpuFractalNzTensor and NpuFractalZnTensor
        torch_npu.npu.config.allow_internal_format = True

        # Setup paths to op libraries
        site_packages_path = get_ascend_custom_opp_path()
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = site_packages_path

        _torch_npu_has_set_up = True

    return torch_npu, has_torch_npu


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


def compute_layer_dist_in_pipe(num_layers, world_size):
    args = get_global_args()
    if args.infer.pp_layer_partition is not None:
        assert (
            len(args.infer.pp_layer_partition) == world_size
            and sum(args.infer.pp_layer_partition) == args.models.n_layers
        ), f"pp_layer_partition must be a list of length {world_size} and sum up to {args.models.n_layers}"
        num_layers_of_each_rank = args.infer.pp_layer_partition
    else:
        num_layers_of_each_rank = [
            num_layers // world_size + (1 if i < num_layers % world_size else 0)
            for i in range(world_size)
        ]
        # If non-divisible, make the fisrst and the last rank to have fewer layers, because they have pre-layers and post-layers
        if world_size > 2 and num_layers_of_each_rank[0] > num_layers_of_each_rank[-2]:
            num_layers_of_each_rank[0] -= 1
            num_layers_of_each_rank[-2] += 1
    return num_layers_of_each_rank


def get_config_dir_path():
    return str(importlib.resources.files("chitu") / "config")


def get_ascend_custom_opp_path():
    site_packages_path = os.path.join(site.getsitepackages()[0], "vendors", "customize")
    return site_packages_path


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
                    cache_manager = args[0].cache
                    if (
                        hasattr(cache_manager, "curr_req_ids")
                        and cache_manager.curr_req_ids
                    ):
                        req_id = cache_manager.curr_req_ids[0]
                        decode_step = cache_manager.req_id_to_seq_len.get(req_id, 0)

                # 获取模型名称和数据类型
                try:
                    args = get_global_args()
                    model_name = args.models.type
                    model_path = args.models.ckpt_dir
                except:
                    model_name = "unknown"
                    model_path = "unknown"
                model_dtype = (
                    "fp4" if get_global_args().infer.npu_fusion_fp4 else "bf16"
                )
                # 添加推理步骤信息
                # print(f"args_names: {args_names}")
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


def gen_req_id(len=8):
    random_number = random.getrandbits(len * 4)
    hex_string = f"{random_number:0{len}x}"
    return hex_string


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


# For disaggregation mode
def get_free_port():
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def get_local_ip() -> str:
    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and ip != "127.0.0.1" and ip != "0.0.0.0":
            return ip
    except Exception:
        pass


torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


def top_k_top_p_min_p_sampling_from_logits(
    logits: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    # TODO: Support min_ps
):
    """A top-k, top-p and min-p sampling implementation."""
    from chitu.ops import multinomial

    if is_ascend() and has_torch_npu:
        assert logits.dim() == 2
        assert (
            top_ps.shape[0] == logits.shape[0]
        ), f"top_ps.shape[0]={top_ps.shape[0]} didn't match logits.shape[0]={logits.shape[0]}"
        assert (
            top_ks.shape[0] == logits.shape[0]
        ), f"top_ks.shape[0]={top_ks.shape[0]} didn't match logits.shape[0]={logits.shape[0]}"
        top_ps = top_ps.to(torch.float)
        top_ks = top_ks.to(torch.int32)
        logits = torch_npu.npu_top_k_top_p(logits, top_ps, top_ks)
        sampled_index = multinomial(logits, num_samples=1, impl="sync-free").view(-1)
        return sampled_index

    # SPDX-SnippetBegin
    # SPDX-License-Identifier: Apache-2.0
    # SPDX-SnippetCopyrightText: 2025 SGLang Team
    # SPDX—SnippetName: top_k_top_p_min_p_sampling_from_logits_torch
    #
    # This sampling implementation is originally from SGLang
    # (https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/sampler.py),
    # licensed under Apache 2.0.
    probs = torch.softmax(logits, dim=-1)
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # TODO: Support min_ps like: min_p_thresholds = probs_sort[:, 0] * min_ps

    top_p_mask = (probs_sum - probs_sort) > top_ps.view(-1, 1)
    top_k_mask = torch.arange(0, probs.shape[-1], device=probs.device).view(
        1, -1
    ) >= top_ks.view(-1, 1)
    if is_ascend():
        probs_sort *= ~(top_p_mask | top_k_mask)
    else:
        probs_sort[top_p_mask | top_k_mask] = 0.0
    # TODO: Support min_ps like:  probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.max(dim=-1, keepdim=True)[0])
    sampled_index = multinomial(probs_sort, num_samples=1, impl="sync-free")
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids
    # SPDX-SnippetEnd


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
    repeat: int = 0,
    with_stack: bool = False,
):
    if has_torch_npu:
        from chitu.npu_utils import try_get_npu_profiler

        return try_get_npu_profiler(
            profiler_dir, wait, warmup, active, repeat, with_stack
        )
    else:  # TODO add nvidia profiler
        raise NotImplementedError("Not supported yet")
