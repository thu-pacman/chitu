# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from hydra.experimental.callback import Callback
from omegaconf import DictConfig
import sys
from logging import getLogger

logger = getLogger(__name__)


class ServeConfigRules(Callback):
    def __init__(self) -> None:
        super().__init__()

    def _exit_with_error(self, message):
        """Fatal error, exit method"""
        logger.error(f"Config Error: {message}")
        sys.exit(1)

    def on_job_start(self, config: DictConfig, **kwargs) -> None:
        port = config.serve.port
        if not (1024 <= port <= 65535):
            self._exit_with_error(f"Port must be between 1024 and 65535, got {port}")

        num_blocks = config.infer.num_blocks
        if num_blocks < 0 and num_blocks != -1:
            self._exit_with_error(
                f"num_blocks must be positive or -1 (got {num_blocks})"
            )

        attn_type = config.infer.attn_type
        if attn_type == "npu":
            try:
                import torch_npu
            except ImportError:
                self._exit_with_error(
                    f"torch-npu required for attn_type=npu (got {attn_type})"
                )
        if attn_type not in {
            "auto",
            "flash_attn",
            "flash_mla",
            "flash_infer",
            "dllm",
            "hunyuan_attn",
            "triton",
            "npu",
            "hopper_mixed",
            "ref",
        }:
            self._exit_with_error(
                f"attn_type must be one of [auto, flash_attn, flash_mla, flash_infer, hunyuan_attn, triton, npu, hopper_mixed, ref], got {attn_type}"
            )

        model_name = config.models.name
        model_type = config.models.type
        if attn_type == "flash_infer":
            if config.models.n_heads % config.models.n_kv_heads != 0:
                self._exit_with_error(
                    f"model {model_name} is not compatible with flash_infer: "
                    f"n_heads ({config.models.n_heads}) must be divisible by "
                    f"n_kv_heads ({config.models.n_kv_heads})"
                )
        elif attn_type == "flash_mla":
            if model_type not in ["deepseek-v3", "deepseek-v4", "kimi-k2-5", "glm-5-2"]:
                self._exit_with_error(
                    f"model {model_name} is not compatible with flash_mla"
                )

        if model_type == "hf-gpt-oss" and attn_type != "ref":
            self._exit_with_error(f"model {model_name} is only compatible with ref")

        tokenizer_type = config.models.tokenizer_type
        if tokenizer_type not in {"hf", "tiktoken"}:
            self._exit_with_error(
                f"tokenizer_type must be one of [hf, tiktoken], got {tokenizer_type}"
            )

        op_impl = config.infer.op_impl
        if op_impl not in {"torch", "muxi_custom_kernel", "cpu"}:
            self._exit_with_error(
                f"op_impl must be one of [torch, muxi_custom_kernel, cpu], got {op_impl}"
            )

        bind_process_to_cpu = config.infer.bind_process_to_cpu
        if bind_process_to_cpu not in {
            "auto",
            "none",
            "one_numa_per_rank",
            "numa_near_device",
        }:
            self._exit_with_error(
                f"bind_process_to_cpu must be one of [auto, none, one_numa_per_rank numa_near_device], got {bind_process_to_cpu}"
            )

        bind_thread_to_cpu = config.infer.bind_thread_to_cpu
        if bind_thread_to_cpu not in {"physical_core", "logical_core"}:
            self._exit_with_error(
                f"bind_thread_to_cpu must be one of [physical_core, logical_core], got {bind_thread_to_cpu}"
            )

        multi_inst = config.multi_inst
        if multi_inst.router.is_router and multi_inst.n_insts <= 1:
            self._exit_with_error(
                f"multi_inst.n_insts must be greater than 1 when multi_inst.router.is_router is true, got {multi_inst.n_insts}"
            )
