# SPDX-FileCopyrightText: 2022 NVIDIA CORPORATION
# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Megatron-LM
#
# This file has adaption of open-source code from the following sources:
# - https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/global_vars.py

import operator
import time
from functools import lru_cache, reduce
from logging import getLogger
from typing import Optional

import torch
from omegaconf import OmegaConf
import re

from chitu.schemas.serve_config import ServeConfig, StaticConfig

logger = getLogger(__name__)


_GLOBAL_ARGS: Optional[ServeConfig] = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_TIMERS = None
_GLOBAL_MEMORY_BUFFER = None
_GLOBAL_SLOT_HANDLE = None
_GLOBAL_DEBUG: bool = False


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
        """Ensure models.quant_config exists and has kv_cache sub-struct."""
        if models.get("quant_config", None) is None:
            OmegaConf.set_struct(models, False)
            models["quant_config"] = {
                "rules": [],
                "type": None,
                "kv_cache": {"rules": [], "type": None},
            }
            OmegaConf.set_struct(models, True)
            return True

        if models.quant_config.get("kv_cache", None) is None:
            OmegaConf.set_struct(models.quant_config, False)
            models.quant_config["kv_cache"] = {"rules": [], "type": None}
            OmegaConf.set_struct(models.quant_config, True)

        return False

    def _get_kv_cache_existing():
        """Read existing kv_cache config (for fallback)."""
        old_kv = models.quant_config.get("kv_cache", None)
        if old_kv is None:
            return None, []
        kv_type = (
            old_kv.get("type", None)
            if isinstance(old_kv, dict)
            else getattr(old_kv, "type", None)
        )
        kv_rules = (
            old_kv.get("rules", [])
            if isinstance(old_kv, dict)
            else getattr(old_kv, "rules", [])
        )
        return kv_type, kv_rules

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

    # Prepare output quant_config skeleton (keep old kv_cache as fallback)
    old_kv_type, old_kv_rules = _get_kv_cache_existing()
    quant_config = {
        "rules": [],
        "type": models.quant_config.get("type", None),
        "kv_cache": {"rules": [], "type": old_kv_type},
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
        quant_config["kv_cache"]["rules"], quant_config["kv_cache"]["type"] = (
            _normalize_rules(rules, quant_config["kv_cache"]["type"])
        )
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


@lru_cache(maxsize=1)
def get_global_args():
    _ensure_var_is_initialized(_GLOBAL_ARGS, "global args")
    cfg: ServeConfig = OmegaConf.to_object(_GLOBAL_ARGS)

    if isinstance(cfg, dict) and "models" in cfg and isinstance(cfg["models"], dict):
        cfg["models"] = StaticConfig(cfg["models"])
    elif hasattr(cfg, "models") and isinstance(cfg.models, dict):
        cfg.models = StaticConfig(cfg.models)
    if isinstance(cfg, dict):
        return StaticConfig(cfg)
    return cfg


def set_global_args(args, need_ensure=True):
    global _GLOBAL_ARGS
    if need_ensure == True:
        _ensure_var_is_not_initialized(_GLOBAL_ARGS, "global args")
    _GLOBAL_ARGS = args
    get_global_args.cache_clear()


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
        required_len = reduce(operator.mul, tensor_shape, 1)
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
        result = [0] * min(total, parts)
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
