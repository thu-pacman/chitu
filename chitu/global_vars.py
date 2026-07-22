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


def _check_checkpoint_path(args):
    if args.models.ckpt_dir is None:
        if not getattr(args.models, "is_pro", False):
            raise ValueError(
                f"No checkpoint path provided. You can set it in command line by adding "
                f"`models.ckpt_dir=<path>`. The model {args.models.name} can be downloaded "
                f"from {args.models.source}"
            )
        else:
            raise ValueError(
                f"No checkpoint path provided. You can set it in command line by adding "
                f"`models.ckpt_dir=<path>`. The model {args.models.name} is part of "
                f"chitu-pro, which may be obtained by concatting solution@chitu.ai"
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


def _has_cpu_layer(args) -> bool:
    if (backend_config := args.models.get("backend_config")) is not None:
        for config in backend_config.get("backend", []):
            if (pattern := config.get("model")) is not None:
                if re.match(pattern, args.models.name.lower()):
                    for rule in config.rules:
                        if rule.get("backend") == "cpuinfer":
                            return True
    return False


def resolve_default_args(args):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    ###################################################################
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
    if hasattr(args.infer, "do_load") and not args.infer.do_load:
        logger.warning(
            "Argument `infer.do_load=False` is deprecated. Use `debug.skip_model_load=True` instead."
        )
        args.debug.skip_model_load = True
    if hasattr(args.infer, "max_reqs") and args.infer.max_reqs is not None:
        args.infer.max_batch_size = args.infer.max_reqs
        logger.warning(
            f"Argument `infer.max_reqs={args.infer.max_reqs}` is deprecated. Use `infer.max_batch_size={args.infer.max_batch_size}` instead."
        )
    if getattr(args.infer, "max_concurrent_requests", None) is None:
        args.infer.max_concurrent_requests = args.infer.max_batch_size * 2
        logger.info(
            f"infer.max_concurrent_requests not set, defaulting to max_batch_size * 2 ({args.infer.max_concurrent_requests})"
        )

    if (
        hasattr(args.scheduler.pp_config, "prefill_num_tasks_divided_by_pp")
        and not args.scheduler.pp_config.prefill_num_tasks_divided_by_pp
    ):
        logger.warning(
            "Argument `scheduler.pp_config.prefill_num_tasks_divided_by_pp=False` is deprecated. Use `scheduler.pp_config.pp_micro_batch_size_prefill=<num>` instead."
        )
        assert (
            hasattr(args.scheduler.pp_config, "prefill_num_tasks")
            and args.scheduler.pp_config.prefill_num_tasks
        )
        args.scheduler.pp_config.pp_micro_batch_size_prefill = (
            args.scheduler.pp_config.prefill_num_tasks
        )
    if (
        hasattr(args.scheduler.pp_config, "enforce_decode_num_tasks_max")
        and not args.scheduler.pp_config.enforce_decode_num_tasks_max
    ):
        logger.warning(
            "Argument `scheduler.pp_config.enforce_decode_num_tasks_max=False` is deprecated. Use `scheduler.pp_config.pp_micro_batch_size_decode=<num>` instead."
        )
        assert (
            hasattr(args.scheduler.pp_config, "decode_num_tasks")
            and args.scheduler.pp_config.decode_num_tasks
        )
        args.scheduler.pp_config.pp_micro_batch_size_decode = (
            args.scheduler.pp_config.decode_num_tasks
        )

    ###################################################################
    # Deal with automatic arguments

    if args.infer.device_ids is None:
        args.infer.device_ids = [i % local_world_size for i in range(world_size)]
    if len(args.infer.device_ids) != world_size:
        raise ValueError(
            f"len(infer.device_ids) ({len(args.infer.device_ids)}) must be equalt to world_size ({world_size})"
        )

    if args.infer.prefill_chunk_size == "auto":
        # prefill_chunk_size is the GLOBAL budget across all DP and CP ranks.
        # Aim for ~4096 tokens processed per rank, so scale by both dp_size
        # and pcp_size.
        args.infer.prefill_chunk_size = 4096 * args.infer.dp_size * args.infer.pcp_size

    if (
        args.infer.prefill_chunk_size is not None
        and args.infer.prefill_chunk_size
        > args.infer.max_batch_size * args.infer.max_seq_len
    ):
        logger.warning(
            f"infer.prefill_chunk_size ({args.infer.prefill_chunk_size}) is larger than "
            f"infer.max_batch_size ({args.infer.max_batch_size}) * infer.max_seq_len "
            f"({args.infer.max_seq_len}), which has no effect. Reducing it to "
            f"infer.max_batch_size * infer.max_seq_len."
        )
        args.infer.prefill_chunk_size = (
            args.infer.max_batch_size * args.infer.max_seq_len
        )

    if args.infer.prefill_chunk_size is not None:
        if args.infer.pp_size > 1 and args.infer.cache_type == "skew":
            logger.warning(
                "Disabling infer.prefill_chunk_size because it is not compatible with PP+skew yet"
            )
            args.infer.prefill_chunk_size = None

    if args.infer.bind_process_to_cpu == "auto":
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
        elif _has_cpu_layer(args):
            if numa.get_max_node() + 1 < local_world_size:
                logger.warning(
                    "Disable NUMA binding due to insufficient NUMA nodes. Is is an inefficient setting of CPU inference."
                )
                args.infer.bind_process_to_cpu = "none"
            else:
                args.infer.bind_process_to_cpu = "one_numa_per_rank"
        else:
            args.infer.bind_process_to_cpu = "numa_near_device"

    if args.infer.use_cuda_graph == "auto":
        if args.models.name in {
            "Mixtral-8x7B-Instruct-v0.1",
            "Qwen3-30B-A3B-mix-fp4-fp8",
        }:
            args.infer.use_cuda_graph = False
        elif args.models.type in {"deepseek-v4"}:
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

    if args.infer.mtp_size <= 0:
        args.infer.mtp_size = 1

    if args.infer.full_warmup == "auto":
        # Suppose you are benchmarking Chitu with a fixed context length, if will end up
        # always running the decode stage with full batch size, then we set
        # `infer.full_warmup=False`. Otherwise, we set `infer.full_warmup=True`.
        if args.infer.pp_size > 1:
            # The actual batch size is affected by micro-batching
            args.infer.full_warmup = True
        elif args.infer.mtp_size > 1:
            # Some requests will end sooner depending on the success rate of MTP, so some
            # final batches will be unfull.
            args.infer.full_warmup = True
        else:
            args.infer.full_warmup = False

    if args.scheduler.pp_config.pp_micro_batch_size_prefill == "auto":
        args.scheduler.pp_config.pp_micro_batch_size_prefill = "max"

    if args.scheduler.pp_config.pp_micro_batch_size_decode == "auto":
        args.scheduler.pp_config.pp_micro_batch_size_decode = "max"

    if args.infer.embed_tokens_lm_head_tp_size == "auto":
        args.infer.embed_tokens_lm_head_tp_size = args.infer.tp_size
    else:
        assert (
            args.infer.embed_tokens_lm_head_tp_size.isdigit()
        ), "embed_tokens_lm_head_tp_size must be auto or an integer"

    from chitu.models.registry import ModelType

    if args.infer.mla_absorb == "auto":
        if args.models.type in {
            ModelType.DEEPSEEK_V3,
            ModelType.KIMI_K2_5,
            ModelType.GLM_5_2,
        }:
            args.infer.mla_absorb = "absorb-without-precomp"
        else:
            args.infer.mla_absorb = "none"

    if args.infer.dp_size > args.infer.max_batch_size:
        raise ValueError(
            f"infer.dp_size ({args.infer.dp_size}) cannot be greater than infer.max_batch_size ({args.infer.max_batch_size})"
        )

    _check_checkpoint_path(args)

    model_resolver = ModelConfigResolver()
    args.models = StaticConfig(
        model_resolver.process_config_dict(args.models, args.models.ckpt_dir)
    )

    if (
        args.models.type == ModelType.DEEPSEEK_V3
        or args.models.type == ModelType.GLM_5_2
    ) and args.models.get("index_topk", None) is not None:
        from chitu.dsa_indexer import (
            HYGON_INDEXER_MAX_MTP_SIZE,
            support_indexer_deepgemm,
            support_indexer_hygon,
            validate_indexer_config,
        )

        if args.infer.indexer_type == "auto":
            if (
                support_indexer_deepgemm
                and args.infer.cache_type == "paged"
                and args.infer.mtp_size < 3
            ):
                args.infer.indexer_type = "deepgemm"
            elif has_native_fp8() and has_triton:
                args.infer.indexer_type = "triton"
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
                raise NotImplementedError("No available infer.indexer_type found")
        validate_indexer_config(args, args.infer.indexer_type)

    if args.infer.process_group_timeout_seconds == "auto":
        if args.infer.full_warmup:
            # DeepGEMM JIT warmup compiles many kernels during the first forward pass
            # (each (n,k) shape sweeps m=[1, DG_WARMUP_MAX_M] building 15-29 distinct
            # kernels), which can take well over the NCCL default 600s watchdog timeout.
            # Use a generous timeout that covers the warmup compilation window when
            # enable full_warmup
            args.infer.process_group_timeout_seconds = 3600
        else:
            args.infer.process_group_timeout_seconds = None

    logger.debug(f"Auto setting configs done. Full configs are: {args}")
    return args


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
        args = resolve_default_args(StaticConfig(args))
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
