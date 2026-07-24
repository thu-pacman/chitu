# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Sequence, List
from logging import getLogger
import sys
import os
import re

logger = getLogger(__name__)


def args_as_list(args) -> List[str]:
    if isinstance(args, str):
        return args.split()
    elif isinstance(args, Sequence):
        return [str(arg) for arg in args]
    else:
        raise ValueError(f"Unsupported argument type: {type(args)}")


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
    if "WORLD_SIZE" in os.environ and "LOCAL_WORLD_SIZE" in os.environ:
        # Inside torchrun. May or may not launched by chitu.boot
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    else:
        # Launching with chitu.boot
        world_size = args.boot.n_nodes * args.boot.n_gpus_per_node
        local_world_size = args.boot.n_gpus_per_node

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
