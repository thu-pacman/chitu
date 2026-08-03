# SPDX-FileCopyrightText: 2022 NVIDIA CORPORATION
# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Megatron-LM
#
# This file has adaption of open-source code from the following sources:
# - https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/global_vars.py

import operator
import os
import time
import functools
from logging import getLogger
from typing import Any, Optional

import torch
from omegaconf import OmegaConf
import re

from chitu.boot.arg_utils import resolve_default_args
from chitu.device_type import has_native_fp8
from chitu.import_utils import try_import_platform_dep, try_import_opt_dep
from chitu.schemas.serve_config import ServeConfig, StaticConfig
from chitu.schemas.utils import ModelConfigResolver

logger = getLogger(__name__)

triton, has_triton = try_import_platform_dep("triton")
numa, has_numa = try_import_opt_dep("numa", "cpu")
deep_ep, has_deep_ep = try_import_opt_dep("deep_ep", "deep_ep")

_RAW_GLOBAL_ARGS: Optional[ServeConfig] = None
_GLOBAL_ARGS = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_TIMERS = None
_GLOBAL_MEMORY_BUFFER = None
_GLOBAL_SLOT_HANDLE = None
_GLOBAL_DEBUG: bool = False
_GLOBAL_INSTANCE_ID: int = -1
_GLOBAL_RANK_ID: int = 0


def get_global_memory_buffer():
    _ensure_var_is_initialized(_GLOBAL_MEMORY_BUFFER, "global memory buffer")
    return _GLOBAL_MEMORY_BUFFER


def get_slot_handle():
    # _ensure_var_is_initialized(_GLOBAL_SLOT_HANDLE, "slot_handle")
    return _GLOBAL_SLOT_HANDLE


def set_global_variables(global_args=None, debug=False):
    _set_debug(debug)
    set_global_args(global_args)
    _set_timers()
    multi_inst_args = getattr(get_global_args(), "multi_inst")
    if multi_inst_args is not None and multi_inst_args.inst_id is not None:
        _set_instance_id(multi_inst_args.inst_id)


def expand_layers(spec):
    layers = set()
    for item in spec:
        if isinstance(item, int):
            layers.add(item)
        elif isinstance(item, str) and "-" in item:
            lo, hi = item.split("-", 1)
            lo, hi = int(lo), int(hi)
            layers.update(range(lo, hi + 1))
        else:
            raise ValueError(f"Invalid layer spec: {item!r}")
    return sorted(layers)


def set_quant_variables(global_args=None):
    if global_args is None:
        return

    models = global_args.get("models", {})
    model_name = models.get("name")
    assert isinstance(model_name, str)
    model_name = model_name.lower()

    def _ensure_quant_config_struct():
        """Ensure models.quant_config exists and has a kv_cache rules list."""
        if models.get("quant_config", None) is None:
            OmegaConf.set_struct(models, False)
            models["quant_config"] = {
                "rules": [],
                "type": None,
                "kv_cache": {"rules": []},
            }
            OmegaConf.set_struct(models, True)
            return True

        if models.quant_config.get("kv_cache", None) is None:
            OmegaConf.set_struct(models.quant_config, False)
            models.quant_config["kv_cache"] = {"rules": []}
            OmegaConf.set_struct(models.quant_config, True)

        return False

    def _get_kv_cache_existing_rules():
        """Read existing kv_cache rules (for fallback)."""
        old_kv = models.quant_config.get("kv_cache", None)
        if old_kv is None:
            return []
        return (
            old_kv.get("rules", [])
            if isinstance(old_kv, dict)
            else getattr(old_kv, "rules", [])
        )

    def _normalize_rules(rules, default_type):
        """
        Fill rule.type, expand rule.layers if present.
        Return (normalized_rules, resolved_default_type).
        """
        out = []
        if rules and not default_type:
            default_type = rules[0].get("type", None)

        for rule in rules:
            rule_type = rule.get("type", None) or default_type
            OmegaConf.set_struct(rule, False)
            rule.type = rule_type
            if "layers" in rule:
                rule.layers = expand_layers(rule.get("layers", []))
            OmegaConf.set_struct(rule, True)
            out.append(rule)

        return out, default_type

    def _normalize_kv_cache_rules(rules):
        """Normalize KV cache rules. Each regex rule must explicitly declare its type."""
        out = []
        for rule in rules:
            if rule.get("type", None) is None:
                raise ValueError(
                    f"kv_cache quant rule {rule} must explicitly set its type"
                )
            OmegaConf.set_struct(rule, False)
            if "layers" in rule:
                rule.layers = expand_layers(rule.get("layers", []))
            OmegaConf.set_struct(rule, True)
            out.append(rule)
        return out

    def _match_entry(quant_list, key):
        """
        Find first config entry matching model_name for given key.
        key: "model" or "kv_cache"
        """
        for cfg in quant_list:
            pattern = cfg.get(key, "")
            if pattern and re.match(pattern, model_name):
                return cfg
        return None

    # Ensure quant_config exists + has kv_cache
    created_default = _ensure_quant_config_struct()
    if created_default:
        return

    # Prepare output quant_config skeleton (keep old kv_cache rules as fallback)
    old_kv_rules = _get_kv_cache_existing_rules()
    quant_config = {
        "rules": [],
        "type": models.quant_config.get("type", None),
        "kv_cache": {"rules": []},
    }

    quant_list = models.quant_config.get("quant", [])

    # Resolve model rules
    model_cfg = _match_entry(quant_list, "model")
    if model_cfg is not None:
        rules = model_cfg.get("rules", [])
        quant_config["rules"], quant_config["type"] = _normalize_rules(
            rules, quant_config["type"]
        )

    # Resolve kv_cache rules
    kv_cfg = _match_entry(quant_list, "kv_cache")
    if kv_cfg is not None:
        rules = kv_cfg.get("rules", [])
        quant_config["kv_cache"]["rules"] = _normalize_kv_cache_rules(rules)
    else:
        # fallback: keep existing kv_cache rules if present
        if old_kv_rules:
            quant_config["kv_cache"]["rules"] = old_kv_rules

    models.quant_config = quant_config


def set_backend_variables(global_args=None):
    if global_args is None:
        return

    models = global_args.get("models", {})
    model_name = models.get("name")
    assert isinstance(model_name, str)

    model_name = model_name.lower()
    if models.get("backend_config", None) is None:
        OmegaConf.set_struct(models, False)
        models["backend_config"] = {"rules": []}
        OmegaConf.set_struct(models, True)
        return

    backend_config = {"rules": []}
    backend_list = models.backend_config.get("backend", [])

    for config in backend_list:
        pattern = config.get("model", "")
        if pattern != "":
            if re.match(pattern, model_name):
                rules = config.get("rules", [])
                backend_config["rules"] = []
                for index, rule in enumerate(rules):
                    backend = rule.get("backend", "default")
                    layers = expand_layers(rule.get("layers", []))
                    OmegaConf.set_struct(rule, False)
                    rule.backend = backend
                    rule.layers = layers
                    OmegaConf.set_struct(rule, True)
                    backend_config["rules"].append(rule)
                models.backend_config = backend_config
                return

    models.backend_config = backend_config


def resolve_full_default_args(args):
    """
    This function provides default args resolving in addition to `esolve_default_args`. The
    additional part can't be done in boot phase.
    """

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    args = resolve_default_args(args)

    model_resolver = ModelConfigResolver()
    args.models = StaticConfig(
        model_resolver.process_config_dict(args.models, args.models.ckpt_dir)
    )

    if args.infer.bind_process_to_cpu != "none":
        if not has_numa:
            logger.warning(
                "Optional dependency '[numa]' is mising. Disabling NUMA binding."
            )
            args.infer.bind_process_to_cpu = "none"
        elif not numa.available():
            logger.warning(
                "NUMA is not support on this OS or hardware platform. Disabling NUMA binding."
            )
            args.infer.bind_process_to_cpu = "none"
    if (
        args.infer.bind_process_to_cpu == "one_numa_per_rank"
        and numa.get_max_node() + 1 < local_world_size
    ):
        logger.warning(
            "Disable NUMA binding due to insufficient NUMA nodes. Is is an inefficient setting of CPU inference."
        )
        args.infer.bind_process_to_cpu = "none"

    if args.infer.use_cuda_graph == "auto":
        if args.models.name in {
            "Mixtral-8x7B-Instruct-v0.1",
            "Qwen3-30B-A3B-mix-fp4-fp8",
        }:
            args.infer.use_cuda_graph = False
        elif (
            args.infer.ep_size > 1
            and args.infer.dp_size > 1
            and (args.infer.tp_size > 1 or not has_deep_ep)
        ):
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
        else:
            args.infer.use_cuda_graph = True

    if (args.models.type in {"deepseek-v3", "glm-5-2"}) and args.models.get(
        "index_topk", None
    ) is not None:
        from chitu.dsa_indexer import (
            HYGON_INDEXER_MAX_MTP_SIZE,
            support_indexer_deepgemm,
            support_indexer_hygon,
            use_fp8_dsa_indexer_kv,
            validate_indexer_config,
        )

        if args.infer.indexer_type == "auto":
            if use_fp8_dsa_indexer_kv(args):
                if (
                    support_indexer_deepgemm
                    and args.infer.cache_type == "paged"
                    and args.infer.mtp_size < 3
                ):
                    args.infer.indexer_type = "deepgemm"
                elif has_native_fp8() and has_triton:
                    args.infer.indexer_type = "triton"
                else:
                    raise NotImplementedError(
                        "No available FP8 infer.indexer_type found"
                    )
            elif (
                support_indexer_hygon
                and args.infer.cache_type == "paged"
                and args.infer.mtp_size <= HYGON_INDEXER_MAX_MTP_SIZE
                and int(args.models.index_head_dim) == 128
                and int(args.models.index_n_heads) in (32, 64)
            ):
                args.infer.indexer_type = "hygon"
            elif args.infer.cache_type == "paged":
                args.infer.indexer_type = "torch_bf16"
            else:
                raise NotImplementedError("No available BF16 infer.indexer_type found")
        validate_indexer_config(args, args.infer.indexer_type)

    logger.debug(f"Auto setting configs done. Full configs are: {args}")
    return args


def _set_instance_id(instance_id: int):
    global _GLOBAL_INSTANCE_ID
    _GLOBAL_INSTANCE_ID = instance_id


def get_instance_id():
    return _GLOBAL_INSTANCE_ID


def set_rank(rank: int):
    global _GLOBAL_RANK_ID
    _GLOBAL_RANK_ID = rank


def get_rank():
    return _GLOBAL_RANK_ID


def _set_debug(debug: bool):
    global _GLOBAL_DEBUG
    _GLOBAL_DEBUG = debug


def get_debug():
    return _GLOBAL_DEBUG


def set_slot_handle(max_batch_size, pp_size):
    global _GLOBAL_SLOT_HANDLE
    # _ensure_var_is_not_initialized(_GLOBAL_SLOT_HANDLE, "slot_handle")
    _GLOBAL_SLOT_HANDLE = SlotHandle(max_batch_size, pp_size)


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER, "tensorboard writer")

    if (
        hasattr(args, "tensorboard_dir")
        and args.tensorboard_dir
        and args.rank == (args.world_size - 1)
    ):
        try:
            from torch.utils.tensorboard import SummaryWriter

            logger.info("> setting tensorboard ...")
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=args.tensorboard_dir, max_queue=args.tensorboard_queue_size
            )
        except ModuleNotFoundError:
            logger.warning(
                "TensorBoard writing requested but is not "
                "available (are you using PyTorch 1.1.0 or later?), "
                "no TensorBoard logs will be written.",
                flush=True,
            )


def _apply_multi_inst_override(cfg: Any, override_inst_id: Optional[int] = None) -> Any:
    """Apply a multi-instance override to a config object."""
    multi_inst = getattr(cfg, "multi_inst", None)
    if multi_inst is None:
        return cfg

    normalized_overrides = {
        int(key): value
        for key, value in (getattr(multi_inst, "inst_overrides", None) or {}).items()
    }

    inst_id_value = getattr(multi_inst, "inst_id", None)
    assert (
        override_inst_id is not None or inst_id_value is not None
    ), "multi_inst.inst_id must be set when applying an instance config"

    inst_id = int(inst_id_value if override_inst_id is None else override_inst_id)
    if inst_id not in normalized_overrides:
        return cfg

    return OmegaConf.merge(cfg, normalized_overrides[inst_id])


def get_multi_inst_config(inst_id: int) -> Any:
    """Return the effective config for one multi-instance."""
    _ensure_var_is_initialized(_RAW_GLOBAL_ARGS, "global args")
    return _apply_multi_inst_override(_RAW_GLOBAL_ARGS, override_inst_id=inst_id)


def get_world_size_from_config(cfg: Any) -> int:
    """Return the world size from a config and validate parallelism consistency."""
    if cfg.infer.etp_size is None:
        if (
            cfg.infer.tp_size
            * cfg.infer.dp_size
            * cfg.infer.pcp_size
            % cfg.infer.ep_size
            != 0
        ):
            raise ValueError(
                f"Inconsistent parallelism: "
                f"tensor_parallel_size({cfg.infer.tp_size}) "
                f"* prefill_context_parallel_size({cfg.infer.pcp_size}) "
                f"* non_expert_data_parallel_size({cfg.infer.dp_size}) "
                f"should be divisible by expert_parallel_size({cfg.infer.ep_size}) "
                f"when expert_tensor_parallel_size is not set"
            )
        etp_size = (
            cfg.infer.tp_size * cfg.infer.pcp_size * cfg.infer.dp_size
        ) // cfg.infer.ep_size
    else:
        etp_size = cfg.infer.etp_size

    world_size = (
        cfg.infer.tp_size * cfg.infer.pcp_size * cfg.infer.dp_size * cfg.infer.pp_size
    )

    if world_size != etp_size * cfg.infer.ep_size * cfg.infer.pp_size:
        raise ValueError(
            f"Inconsistent parallelism: world_size({world_size}) should be equal to "
            f"expert_tensor_parallel_size({etp_size}) "
            f"* expert_parallel_size({cfg.infer.ep_size}) "
            f"* pipeline_parallel_size({cfg.infer.pp_size}) "
        )

    return world_size


def get_multi_inst_world_size(inst_id: int) -> int:
    """Return the effective torch world size for one multi-instance."""
    return get_world_size_from_config(get_multi_inst_config(inst_id))


@functools.cache
def get_multi_inst_ids_by_role(role: str) -> list[int]:
    """Return global instance IDs whose effective config has the given role."""
    _ensure_var_is_initialized(_RAW_GLOBAL_ARGS, "global args")
    base_cfg = _RAW_GLOBAL_ARGS
    base_multi_inst = getattr(base_cfg, "multi_inst", None)
    role_ids = []
    for inst_id in range(int(getattr(base_multi_inst, "n_insts", 1))):
        cfg = _apply_multi_inst_override(base_cfg, override_inst_id=inst_id)
        multi_inst = getattr(cfg, "multi_inst", None)
        if getattr(multi_inst, "role", "prefill_and_decode") == role:
            role_ids.append(inst_id)
    return role_ids


@functools.cache
def _get_effective_multi_inst_roles() -> tuple[str, ...]:
    """Return effective roles for all configured instances."""
    _ensure_var_is_initialized(_RAW_GLOBAL_ARGS, "global args")
    base_cfg = _RAW_GLOBAL_ARGS
    base_multi_inst = getattr(base_cfg, "multi_inst", None)
    return tuple(
        getattr(
            getattr(
                _apply_multi_inst_override(base_cfg, override_inst_id=inst_id),
                "multi_inst",
                None,
            ),
            "role",
            "prefill_and_decode",
        )
        for inst_id in range(int(getattr(base_multi_inst, "n_insts", 1)))
    )


def is_independent_multi_inst() -> bool:
    """Return True when every instance independently handles prefill and decode."""
    return all(
        role == "prefill_and_decode" for role in _get_effective_multi_inst_roles()
    )


def is_classic_pd_disagg() -> bool:
    """Return True when all instances are split into prefill-only or decode-only roles."""
    roles = _get_effective_multi_inst_roles()
    return all(role in ("prefill", "decode") for role in roles)


def is_pd_prefill_only() -> bool:
    return bool(
        is_classic_pd_disagg() and get_global_args().multi_inst.role == "prefill"
    )


def get_global_args(need_ensure=True):
    if need_ensure:
        _ensure_var_is_initialized(_GLOBAL_ARGS, "global args")
    return _GLOBAL_ARGS


def get_kv_transfer_args():
    """Return the KvTransferConfig for PD disaggregation KV transfer.

    No fallback — crashes if config is missing (expected in PD mode).
    """
    return get_global_args().multi_inst.pd_disaggregation.kv_transfer


def set_global_args(raw_args, need_ensure=True, need_preprocess=True):
    global _RAW_GLOBAL_ARGS
    global _GLOBAL_ARGS
    if need_ensure == True:
        _ensure_var_is_not_initialized(_RAW_GLOBAL_ARGS, "raw global args")
        _ensure_var_is_not_initialized(_GLOBAL_ARGS, "global args")
    _RAW_GLOBAL_ARGS = raw_args
    get_multi_inst_ids_by_role.cache_clear()
    _get_effective_multi_inst_roles.cache_clear()

    args = raw_args
    if need_preprocess:
        if (multi_inst := getattr(args, "multi_inst", None)) and getattr(
            multi_inst, "inst_id", None
        ) is not None:
            args = _apply_multi_inst_override(args)
        args = resolve_full_default_args(StaticConfig(args))
    _GLOBAL_ARGS = args


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, "timers")
    return _GLOBAL_TIMERS


def _set_timers():
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, "timers")
    _GLOBAL_TIMERS = Timers()


def _set_global_memory_buffer():
    """Initialize global buffer"""
    global _GLOBAL_MEMORY_BUFFER
    _ensure_var_is_not_initialized(_GLOBAL_MEMORY_BUFFER, "global memory buffer")
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, "{} is not initialized.".format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, "{} is already initialized.".format(name)


class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()
        self.cnt = 0

    def start(self):
        """Start the timer."""
        if not get_debug():
            return
        assert not self.started_, "timer has already been started"
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        if not get_debug():
            return
        assert self.started_, "timer is not started"
        torch.cuda.synchronize()
        self.elapsed_ += time.time() - self.start_time
        self.started_ = False
        self.cnt += 1

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False
        self.cnt = 0

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar(name + "-time", value, iteration)

    def log(self, names=[], normalizer=1.0, reset=True):
        """Log a group of timers."""
        if len(names) == 0:
            names = self.timers.keys()
        assert normalizer > 0.0
        string = "time (ms)"
        for name in names:
            cnt = self.timers[name].cnt
            if cnt == 0:
                continue
            elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
            string += " | {}: {:.2f} {} {:.2f}".format(
                name, elapsed_time, cnt, elapsed_time / cnt
            )
        logger.info(string)


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        required_len = functools.reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            self.buffer[(name, dtype)] = torch.empty(
                required_len,
                dtype=dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)


class SlotHandle:
    """
    split max_batch_size to micro_batch size
    max_req = 10, pp_size = 3, self.slots_size = [4, 3, 3]
    """

    def __init__(self, max_batch_size, pp_size):
        self.slots_size = self.split_slots(max_batch_size, pp_size)
        self.num_slots = len(self.slots_size)
        self.slot_idx = 0
        self.slot_start_idx = []
        self.slot_end_idx = []

        res = [0]
        for value in self.slots_size:
            res.append(res[-1] + value)
        self.slot_start_idx = res[:-1]
        self.slot_end_idx = res[1:]

    def split_slots(self, total, parts):
        result = [0] * parts
        for i in range(total):
            result[i % parts] += 1
        return result

    def get_slot_size(self, idx):
        return self.slots_size[idx]

    def set_slot_idx(self, idx):
        self.slot_idx = idx

    def get_slot_idx(self):
        return self.slot_idx

    def get_slot_start_end_idx(self, idx):
        return self.slot_start_idx[idx], self.slot_end_idx[idx]

    def get_current_slot_start_end_idx(self):
        return self.get_slot_start_end_idx(self.slot_idx)


def set_cuda_device():
    """Set the CUDA device for the current process/thread.

    Uses ``infer.device_ids[global_rank]`` to determine the device
    Must be called in every new thread entry point that interacts with CUDA.
    """
    rank = int(os.environ.get("RANK", 0))
    args = get_global_args()
    torch.cuda.set_device(args.infer.device_ids[rank])
