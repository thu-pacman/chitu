# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional, Sequence, List
from logging import getLogger

from omegaconf import OmegaConf

import sys
import os
import re

logger = getLogger(__name__)


@dataclass(frozen=True)
class ParallelismSizes:
    tp_size: int
    pp_size: int
    dp_size: int
    ep_size: int
    etp_size: int
    pcp_size: int
    embed_tokens_lm_head_tp_size: int
    world_size: int


def apply_multi_inst_override(cfg: Any, override_inst_id: Optional[int] = None) -> Any:
    """Apply the selected multi-instance override to a config object."""
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


def calculate_parallelism_sizes(cfg: Any) -> ParallelismSizes:
    """Return the derived parallelism sizes and validate their consistency."""
    tp_size = int(cfg.infer.tp_size)
    pp_size = int(cfg.infer.pp_size)
    dp_size = int(cfg.infer.dp_size)
    ep_size = int(cfg.infer.ep_size)
    pcp_size = int(cfg.infer.pcp_size)

    if pcp_size > 1 and dp_size > 1:
        raise ValueError(
            "infer.pcp_size > 1 cannot be used with infer.dp_size > 1 yet. "
            "Prefill CP currently uses the MoE allgather dispatcher group slot, "
            "so it cannot also express attention DP allgather."
        )

    raw_etp_size = cfg.infer.etp_size
    if raw_etp_size is None:
        if tp_size * pcp_size * dp_size % ep_size != 0:
            raise ValueError(
                f"Inconsistent parallelism: "
                f"tensor_parallel_size({tp_size}) "
                f"* prefill_context_parallel_size({pcp_size}) "
                f"* non_expert_data_parallel_size({dp_size}) "
                f"should be divisible by expert_parallel_size({ep_size}) "
                f"when expert_tensor_parallel_size is not set"
            )
        etp_size = tp_size * pcp_size * dp_size // ep_size
    else:
        etp_size = int(raw_etp_size)

    embed_tokens_lm_head_tp_size = int(cfg.infer.embed_tokens_lm_head_tp_size)
    if tp_size > 1:
        assert (
            embed_tokens_lm_head_tp_size == tp_size
        ), "embed_tokens_lm_head_tp_size must be equal to tensor_parallel_size when tensor_parallel_size > 1"
    elif dp_size > 1:
        assert (
            dp_size % embed_tokens_lm_head_tp_size == 0
        ), "non_expert_data_parallel_size must be divisible by embed_tokens_lm_head_tp_size when non_expert_data_parallel_size > 1"
    else:
        assert (
            embed_tokens_lm_head_tp_size == 1
        ), "embed_tokens_lm_head_tp_size must be 1 when tensor_parallel_size == 1 and non_expert_data_parallel_size == 1"

    world_size = tp_size * pcp_size * dp_size * pp_size

    if world_size != etp_size * ep_size * pp_size:
        raise ValueError(
            f"Inconsistent parallelism: world_size({world_size}) should be equal to "
            f"expert_tensor_parallel_size({etp_size}) "
            f"* expert_parallel_size({ep_size}) "
            f"* pipeline_parallel_size({pp_size}) "
        )

    return ParallelismSizes(
        tp_size=tp_size,
        pp_size=pp_size,
        dp_size=dp_size,
        ep_size=ep_size,
        etp_size=etp_size,
        pcp_size=pcp_size,
        embed_tokens_lm_head_tp_size=embed_tokens_lm_head_tp_size,
        world_size=world_size,
    )


def args_as_list(args) -> List[str]:
    if isinstance(args, str):
        return args.split()
    elif isinstance(args, Sequence):
        return [str(arg) for arg in args]
    else:
        raise ValueError(f"Unsupported argument type: {type(args)}")


def container_setup_cmd_wrapper_args(container_setup_cmd: Optional[str]) -> List[str]:
    if container_setup_cmd is None or container_setup_cmd == "":
        return []
    if not isinstance(container_setup_cmd, str):
        raise ValueError(
            f"boot.container_setup_cmd must be a string or null, "
            f"got {type(container_setup_cmd)}"
        )

    script = "\n".join([container_setup_cmd, 'exec "$@"'])
    return ["/bin/bash", "-c", script, "chitu-container-setup-cmd"]


def suffixed_name(name: Optional[str], suffix: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    if suffix is None or suffix == "":
        return name
    return f"{name}-{suffix}"


def _has_cpu_layer(args) -> bool:
    if (backend_config := args.models.get("backend_config")) is not None:
        for config in backend_config.get("backend", []):
            if (pattern := config.get("model")) is not None:
                if re.match(pattern, args.models.name.lower()):
                    for rule in config.rules:
                        if rule.get("backend") == "cpuinfer":
                            return True
    return False


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


def resolve_default_args(args):
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

    if args.boot.interactive_node_0 == "auto":
        args.boot.interactive_node_0 = sys.stdout.isatty() and args.boot.n_nodes == 1

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
        if _has_cpu_layer(args):
            args.infer.bind_process_to_cpu = "one_numa_per_rank"
        else:
            args.infer.bind_process_to_cpu = "numa_near_device"

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

    if args.infer.mla_absorb == "auto":
        if args.models.type in {"deepseek-v3", "kimi-k2-5", "glm-5-2"}:
            args.infer.mla_absorb = "absorb-without-precomp"
        else:
            args.infer.mla_absorb = "none"

    if args.infer.dp_size > args.infer.max_batch_size:
        raise ValueError(
            f"infer.dp_size ({args.infer.dp_size}) cannot be greater than infer.max_batch_size ({args.infer.max_batch_size})"
        )

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

    if args.infer.schedule_overlap == "auto":
        args.infer.schedule_overlap = True

    _check_checkpoint_path(args)

    return args
