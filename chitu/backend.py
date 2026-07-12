# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import gc
import functools
import os
import time
import re
from collections import deque
from datetime import timedelta
from enum import Enum
from glob import glob
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from safetensors.torch import safe_open
from tqdm import tqdm

from chitu.attn_backend import (
    FlashAttnBackend,
    FlashInferBackend,
    FlashMLABackend,
    HopperMixedBackend,
    HybridAttnBackend,
    DLLMAttnBackend,
    NpuAttnBackend,
    RefAttnBackend,
    TritonAttnBackend,
    NpuAttnBackend,
    HybridAttnBackend,
    DLLMAttnBackend,
    HunyuanAttnBackend,
)

from chitu.kv_cache.registry import (
    should_use_hopper_mixed_backend,
    can_use_hunyuan_attn,
)
from chitu.kv_cache import KVCacheManagerBase, PagedKVCache, KVCacheBase
from chitu.custom_gguf import *
from chitu.device_type import is_ascend, is_muxi
from chitu.distributed.parallel_state import (
    get_ep_group,
    get_dp_group,
    initialize_parallel_groups,
)
from chitu.distributed.coordinator import init_coordinator
from chitu.distributed.partition import compute_local_batch_size_dist_in_dp
from chitu.distributed.infiniband import auto_set_ib_envs
from chitu.hybrid_device import CPUParameter
from chitu.models.registry import ModelType, get_model_class
from chitu.native_layout import init_native_layout
from chitu.native_layout.base import TensorWithNativeLayout
from chitu.native_layout.npu import NpuFractalNzTensor, NpuFractalZnTensor
from chitu.quantization import (
    QuantizationRegistry,
    QuantizedMoeExpertsBase,
    get_quant_from_checkpoint_prefix,
    utils,
)
from chitu.tokenizer import (
    ChatFormat,
    ChatFormatHF,
    ChatFormatHF_dsv32,
    ChatFormatHF_dsv4,
    ChatFormatLLaDA,
    Tokenizer,
    TokenizerHF,
    Processor,
)
from chitu.tool_call import patch_chat_template
from chitu.utils import parse_dtype
from chitu.import_utils import try_import_opt_dep
from chitu.moe import init_moe_impl
from chitu.global_vars import set_slot_handle, set_cuda_device
from chitu.numa_utils import bind_process_to_numa
from chitu.kv_cache.providers import register_all_providers
from chitu.kv_cache.builders import build_cache_managers

if TYPE_CHECKING:
    from chitu.executor import Executor
    from chitu.scheduler import Scheduler

cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")


logger = getLogger(__name__)


class BackendState(Enum):
    Running = 1
    Terminating = 2  # All tasks done, but rank 0 should tell others to terminate
    Terminated = 3


class Backend:
    # init once
    model = None
    tokenizer = None
    cache_dict: dict[str, KVCacheBase] = {}
    formatter = None
    processor = None
    args = None
    curr_tids = None
    cache_type = ""
    # ---
    use_gloo = True
    group_gloo = None
    pp_end_stage = None
    pp_main_rank = None

    # components
    schedulers: Optional[list["Scheduler"]] = None  # One per each DP rank
    cache_managers: Optional[list[dict[str, "KVCacheManagerBase"]]] = (
        None  # One per each DP rank
    )
    executor: Optional["Executor"] = None
    kv_manager = None  # KVManager instance for PD disaggregation

    # mutable
    state = BackendState.Running

    # ---- MoE load balancer optional weight accessor ----
    # Provide a place to install a weight accessor from model/engine.
    moe_weight_accessor = None  # set via set_moe_weight_accessor()
    _moe_experts_by_layer: dict[int, object] = {}

    @staticmethod
    def get_moe_weight_accessor():
        """Return the registered MoE ExpertParamAccessor (if any)."""
        return Backend.moe_weight_accessor

    @staticmethod
    def set_moe_weight_accessor(accessor) -> None:
        """Install a MoE weight accessor and immediately register with the planner.

        Call this after the model is built and expert parameters are accessible.
        Safe to call multiple times; later calls overwrite the previous accessor.
        """
        Backend.moe_weight_accessor = accessor
        if accessor is None:
            return
        try:
            # Local import to avoid circular deps at module import time
            from chitu.moe.load_balancer import register_moe_weight_accessor

            register_moe_weight_accessor(accessor, get_ep_group())
        except Exception as e:
            logger.warning(f"Backend: failed to register MoE weight accessor: {e}")

    @staticmethod
    def register_moe_layer_experts(
        layer_id: int, experts_module: QuantizedMoeExpertsBase
    ) -> None:
        """Register the experts module for a specific layer and auto-wire accessor."""
        Backend._moe_experts_by_layer[int(layer_id)] = experts_module
        # Build and register accessor lazily on first registration
        if Backend.moe_weight_accessor is None:
            try:
                # Lazy import to avoid circular deps at import time
                from chitu.moe.load_balancer import ExpertParamAccessor

                class _ModelExpertsAccessor(ExpertParamAccessor):  # type: ignore

                    def _copy_in(
                        self, dst_tensor: torch.Tensor, src: torch.Tensor, slot: int
                    ):
                        dst_tensor.data[slot].copy_(
                            src.to(dtype=dst_tensor.dtype, device=dst_tensor.device)
                        )

                    def get_params(self, layer_id: int, slot: int):
                        mod = Backend._moe_experts_by_layer.get(int(layer_id))
                        if mod is None:
                            raise RuntimeError(
                                f"No experts module registered for layer {layer_id}"
                            )
                        out = {}
                        for name, param in mod.named_parameters():
                            if not isinstance(param, torch.Tensor):
                                continue
                            if param.dim() == 0:
                                continue
                            try:
                                value = param[slot]
                            except Exception:
                                continue
                            key = name
                            out[key] = value

                        if not out:
                            raise NotImplementedError(
                                "Cannot infer expert param tensors from named_parameters; "
                                "please implement get_expert_params/set_expert_params on experts module"
                            )
                        return out

                    def set_params(self, layer_id: int, slot: int, params):
                        mod = Backend._moe_experts_by_layer.get(int(layer_id))
                        if mod is None:
                            raise RuntimeError(
                                f"No experts module registered for layer {layer_id}"
                            )

                        # Generic write-back: iterate all module parameters and match by name
                        for name, param in mod.named_parameters():
                            if name not in params:
                                continue
                            src = params[name]
                            if not isinstance(param, torch.Tensor):
                                continue
                            if param.dim() == 0:
                                continue
                            try:
                                # write into the target expert slot in-place
                                self._copy_in(param, src, slot)
                            except Exception:
                                # Shape mismatch or non-expert tensor; skip
                                continue

                accessor = _ModelExpertsAccessor()
                Backend.set_moe_weight_accessor(accessor)
            except Exception as e:
                logger.warning(
                    f"Backend: failed to build/register ModelExpertsAccessor: {e}"
                )

    @staticmethod
    def build_model(args, cache_managers, *extra_args, **extra_kwargs):
        try:
            model_type = ModelType(args.type)
        except ValueError:
            raise ValueError(
                f"Model type '{args.type}' is not supported. "
                f"Available types: {[t.value for t in ModelType]}"
            )

        model_cls = get_model_class(model_type)
        if args.name.startswith("glm"):
            extra_kwargs["rotary_type"] = "interleaved-half"
        return model_cls(args, cache_managers, *extra_args, **extra_kwargs)

    # FIXME: When cache type is "skew", gloo backend cannot be used.
    @staticmethod
    def _init_distributed(args):
        """
        Initialize distributed training environment with tensor and pipeline parallelism.

        Arguments:
            args: Configuration object with distributed parameters
        """

        auto_set_ib_envs()

        is_router_process = os.environ.get("CHITU_ROUTER_PROCESS", "0") == "1"
        if is_router_process:
            # Router process: as independent subprocess, skip CUDA device binding
            logger.info(f"[Router] Router subprocess skip CUDA device binding")
            return

        # Get rank from environment variable because we have not initialize torch.distributed yet

        # Bind process to GPU. Please put it before init_process_group
        if args.infer.op_impl != "cpu":
            set_cuda_device()

        if not torch.distributed.is_initialized():
            if args.infer.op_impl == "cpu":
                torch.distributed.init_process_group("gloo")
            else:
                torch.distributed.init_process_group("nccl")

        if Backend.use_gloo:
            Backend.group_gloo = torch.distributed.new_group(backend="gloo")

        bind_process_to_numa(args.infer.bind_process_to_cpu)

        tensor_parallel_size = args.infer.tp_size
        pipeline_parallel_size = args.infer.pp_size
        non_expert_data_parallel_size = args.infer.dp_size
        expert_parallel_size = args.infer.ep_size
        expert_tensor_parallel_size = args.infer.etp_size
        prefill_context_parallel_size = args.infer.pcp_size
        global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        if prefill_context_parallel_size > 1 and non_expert_data_parallel_size > 1:
            raise ValueError(
                "infer.pcp_size > 1 cannot be used with infer.dp_size > 1 yet. "
                "Prefill CP currently uses the MoE allgather dispatcher group slot, "
                "so it cannot also express attention DP allgather."
            )

        if expert_tensor_parallel_size is None:
            assert (
                tensor_parallel_size
                * non_expert_data_parallel_size
                * prefill_context_parallel_size
                % expert_parallel_size
                == 0
            )
            expert_tensor_parallel_size = (
                tensor_parallel_size
                * prefill_context_parallel_size
                * non_expert_data_parallel_size
                // expert_parallel_size
            )
        embed_tokens_lm_head_tp_size = int(args.infer.embed_tokens_lm_head_tp_size)
        if tensor_parallel_size > 1:
            assert (
                embed_tokens_lm_head_tp_size == tensor_parallel_size
            ), "embed_tokens_lm_head_tp_size must be equal to tensor_parallel_size when tensor_parallel_size > 1"
        elif non_expert_data_parallel_size > 1:
            assert (
                non_expert_data_parallel_size % embed_tokens_lm_head_tp_size == 0
            ), "non_expert_data_parallel_size must be divisible by embed_tokens_lm_head_tp_size when non_expert_data_parallel_size > 1"
        else:
            assert (
                embed_tokens_lm_head_tp_size == 1
            ), "embed_tokens_lm_head_tp_size must be 1 when tensor_parallel_size == 1 and non_expert_data_parallel_size == 1"

        if (
            world_size
            != tensor_parallel_size
            * prefill_context_parallel_size
            * non_expert_data_parallel_size
            * pipeline_parallel_size
        ):
            raise ValueError(
                f"Inconsistent parallelism: world_size({world_size}) should be equal to "
                f"tensor_parallel_size({tensor_parallel_size}) "
                f"* prefill_context_parallel_size({prefill_context_parallel_size}) "
                f"* non_expert_data_parallel_size({non_expert_data_parallel_size}) "
                f"* pipeline_parallel_size({pipeline_parallel_size}) "
            )
        if (
            world_size
            != expert_tensor_parallel_size
            * expert_parallel_size
            * pipeline_parallel_size
        ):
            raise ValueError(
                f"Inconsistent parallelism: world_size({world_size}) should be equal to "
                f"expert_tensor_parallel_size({expert_tensor_parallel_size}) "
                f"* expert_parallel_size({expert_parallel_size}) "
                f"* pipeline_parallel_size({pipeline_parallel_size}) "
            )

        initialize_parallel_groups(
            tp_size=tensor_parallel_size,
            dp_size=non_expert_data_parallel_size,
            etp_size=expert_tensor_parallel_size,
            ep_size=expert_parallel_size,
            pp_size=pipeline_parallel_size,
            pcp_size=prefill_context_parallel_size,
            embed_tokens_lm_head_tp_size=embed_tokens_lm_head_tp_size,
        )
        if args.multi_inst.n_insts > 1:
            if args.coordinator.host is None:
                raise ValueError(
                    "coordinator.host is required when multi_inst.n_insts > 1"
                )
            if args.coordinator.port is None:
                raise ValueError(
                    "coordinator.port is required when multi_inst.n_insts > 1"
                )
            init_coordinator(
                args.coordinator.host, args.coordinator.port, is_coordinator_host=False
            )
        elif args.coordinator.host is None or args.coordinator.port is None:
            init_coordinator(
                None,
                None,
                is_coordinator_host=global_rank == 0,
                reuse_from_torchrun=True,
            )
        else:
            init_coordinator(
                args.coordinator.host,
                args.coordinator.port,
                is_coordinator_host=global_rank == 0,
            )

    @staticmethod
    def _setup_environment(args):
        """
        Set up random seed, default dtype, and check prerequisites.

        Arguments:
            args: Configuration with seed and dtype settings
        """
        torch.manual_seed(args.infer.seed)

        # Set default_dtype
        if args.float_16bit_variant == "float16":
            torch.set_default_dtype(torch.float16)
        elif args.float_16bit_variant == "bfloat16":
            torch.set_default_dtype(torch.bfloat16)
        else:
            raise NotImplementedError(f"Unsupported float_16bit_variant {args.dtype}")

    @staticmethod
    def _init_tokenizer(args):
        """
        Initialize the appropriate tokenizer based on model type.

        Arguments:
            args: Configuration with tokenizer settings

        Returns:
            Initialized tokenizer
        """
        model_name_lower = args.models.name.lower()
        trust_remote_code = (
            model_name_lower.startswith("glm-4")
            or model_name_lower.startswith("glm-5")
            or model_name_lower.startswith("kimi")
        )
        force_full_seq_decode = (
            args.models.tokenizer_force_full_seq_decode
            if hasattr(args.models, "tokenizer_force_full_seq_decode")
            else False
        )
        skip_special_tokens = (
            args.models.skip_special_tokens
            if hasattr(args.models, "skip_special_tokens")
            else True
        )

        if args.models.tokenizer_type == "hf":
            tokenizer = TokenizerHF(
                path=args.models.tokenizer_path,
                trust_remote_code=trust_remote_code,
                force_full_seq_decode=force_full_seq_decode,
                skip_special_tokens=skip_special_tokens,
            )
        else:
            tokenizer = Tokenizer(
                model_path=args.models.tokenizer_path,
                force_full_seq_decode=force_full_seq_decode,
            )
            assert (
                args.models.vocab_size == tokenizer.n_words
            ), f"{args.models.vocab_size} vs. {tokenizer.n_words}"

        patch_chat_template(tokenizer.model)
        return tokenizer

    @staticmethod
    def _init_processor(args):
        """
        Initialize the multimodal processor for vision-language models.

        Arguments:
            args: Configuration with model settings

        Returns:
            Initialized processor or None if not a multimodal model
        """

        if not hasattr(args.models, "vision_config") or (
            args.models.type == ModelType.HF_QWEN3_5 and args.infer.language_model_only
        ):
            return None

        processor = Processor(path=args.models.processor_path, trust_remote_code=True)

        logger.info(f"Initialized multimodal processor for {args.models.name}")
        return processor

    @staticmethod
    def _init_formatter(args):
        """
        Initialize the chat formatter based on model type.

        Arguments:
            args: Configuration with model settings

        Returns:
            Appropriate chat formatter instance
        """
        tokenizer_type = args.models.tokenizer_type
        chatformat_type = getattr(args.models, "chatformat_type", tokenizer_type)
        if chatformat_type == "dsv32":
            return ChatFormatHF_dsv32(Backend.tokenizer, Backend.processor)
        elif chatformat_type == "dsv4":
            return ChatFormatHF_dsv4(Backend.tokenizer, Backend.processor)
        elif chatformat_type == "llada":
            return ChatFormatLLaDA(Backend.tokenizer, Backend.processor)
        elif chatformat_type == "hf":
            return ChatFormatHF(Backend.tokenizer, Backend.processor)
        else:
            return ChatFormat(Backend.tokenizer)

    @staticmethod
    def _get_attention_backend_type(args):
        if args.infer.attn_type == "auto":
            if is_ascend():
                return NpuAttnBackend
            elif args.infer.op_impl == "cpu":
                return RefAttnBackend
            elif should_use_hopper_mixed_backend(args):
                return HopperMixedBackend
            elif args.models.type in [
                ModelType.DEEPSEEK_V3,
                ModelType.KIMI_K2_5,
                ModelType.GLM_5_2,
            ]:
                return FlashMLABackend
            else:
                return HybridAttnBackend
        elif args.infer.attn_type == "cpu":
            return RefAttnBackend
        elif args.infer.attn_type == "flash_attn":
            return FlashAttnBackend
        elif args.infer.attn_type == "flash_mla":
            return FlashMLABackend
        elif args.infer.attn_type == "flash_infer":
            return FlashInferBackend
        elif args.infer.attn_type == "dllm":
            return DLLMAttnBackend
        elif args.infer.attn_type == "triton":
            return TritonAttnBackend
        elif args.infer.attn_type == "npu":
            return NpuAttnBackend
        elif args.infer.attn_type == "ref":
            return RefAttnBackend
        elif args.infer.attn_type == "hopper_mixed":
            return HopperMixedBackend
        elif args.infer.attn_type == "hunyuan_attn":
            if can_use_hunyuan_attn(args):
                return HunyuanAttnBackend
            else:
                raise ValueError(
                    "HunyuanAttnBackend is not compatible with the current model/configuration"
                )
        else:
            raise ValueError(f"Unknown attn type {args.infer.attn_type}")

    @staticmethod
    def _init_attention_backend(attn_backend_type, args):
        # Yes, use `type` instead of `isinstance` here, because `AttnBackend`s inherit each other
        if attn_backend_type is FlashInferBackend:
            max_num_blocks = 0
            for cache in Backend.cache_dict.values():
                if not isinstance(cache, PagedKVCache):
                    raise NotImplementedError(
                        "`infer.attn_type=flash_infer` is only compatible with `infer.cache_type=paged`"
                    )
                max_num_blocks = max(max_num_blocks, cache.max_num_blocks)
            return attn_backend_type(max_num_blocks)
        if attn_backend_type is HunyuanAttnBackend:
            HunyuanAttnBackend.validate_model_config(args)
            for cache in Backend.cache_dict.values():
                if not isinstance(cache, PagedKVCache):
                    raise NotImplementedError(
                        "`infer.attn_type=hunyuan_attn` is only compatible with `infer.cache_type=paged`"
                    )
            head_dim = getattr(args, "head_dim", None)
            if head_dim is None:
                head_dim = args.dim // args.n_heads
            return attn_backend_type(
                head_dim=head_dim,
                n_heads=args.n_heads,
                n_kv_heads=getattr(args, "n_kv_heads", None),
            )
        else:
            return attn_backend_type()

    @staticmethod
    def _move_one_module_to_device(
        m: torch.nn.Module, non_blocking: bool = True, ignore_not_loaded: bool = False
    ):
        if Backend.args.infer.op_impl == "cpu":
            return

        # NOTE: m._parameters contains parameters in this module (non-recursive),
        # while m.parameters() returns all parameters in this module and its submodules
        # (recursive).
        for key in m._parameters:
            param = m._parameters[key]
            if param is not None:
                if not isinstance(param, CPUParameter):
                    if param.device == torch.device("meta"):
                        if not ignore_not_loaded:
                            assert False, f"Unexpected unloaded parameter {m}.{key}"
                        else:
                            continue
                    if is_muxi():
                        # Work around a muxi bug that convert from NHWC to NCHW for whatever
                        # 4-D tensor even its not a convolution weight.
                        param.data = param.data.cuda(
                            non_blocking=non_blocking
                        ).contiguous()
                    else:
                        param.data = param.data.cuda(non_blocking=non_blocking)
                        if (
                            Backend.args.models.type
                            in {
                                ModelType.HF_QWEN3_VL,
                                ModelType.HF_QWEN3_VL_MOE,
                                ModelType.HF_QWEN3_5,
                            }
                            and not param.data.is_contiguous()
                        ):
                            param.data = param.data.contiguous()
        for key in m._buffers:
            buffer = m._buffers[key]
            if buffer is not None:
                if buffer.device == torch.device("meta"):
                    # Buffers are expected possibly not to be loaded, so buffer.device may be "meta"
                    m._buffers[key] = torch.empty(
                        buffer.shape, dtype=buffer.dtype, device="cuda"
                    )
                elif is_muxi():
                    # Work around a muxi bug that convert from NHWC to NCHW for whatever
                    # 4-D tensor even its not a convolution weight.
                    m._buffers[key] = buffer.cuda(
                        non_blocking=non_blocking
                    ).contiguous()
                else:
                    m._buffers[key] = buffer.cuda(non_blocking=non_blocking)
                    if Backend.args.models.type in {
                        ModelType.HF_QWEN3_VL,
                        ModelType.HF_QWEN3_VL_MOE,
                        ModelType.HF_QWEN3_5,
                    } and (
                        (buf_cuda := m._buffers[key]) is not None
                        and not buf_cuda.is_contiguous()
                    ):
                        m._buffers[key] = m._buffers[key].contiguous()

    @staticmethod
    def _create_empty_model(model: torch.nn.Module, skip_preprocess: bool):
        if skip_preprocess:
            model.to_empty(device=torch.cuda.current_device())
            for p in model.parameters():
                # NPU format (FRACTAL_NZ / FRACTAL_ZN) cannot exist on
                # meta or CPU tensors — convert_from on meta only
                # produces a stub (ND format).  After to_empty we have
                # real NPU memory, so re-run convert_from to apply the
                # hardware format cast.
                if NpuFractalNzTensor.check_tensor(p):
                    p.data = NpuFractalNzTensor.convert_from(p.data).layout_tensor
                elif NpuFractalZnTensor.check_tensor(p):
                    p.data = NpuFractalZnTensor.convert_from(p.data).layout_tensor
        else:
            state_dict = {}
            for name, param in model.named_parameters():
                if isinstance(param, TensorWithNativeLayout):
                    t = param.native_layout
                    state_dict[name] = torch.empty(
                        t.state_dict_shape, dtype=t.state_dict_dtype
                    )
                else:
                    state_dict[name] = torch.empty(param.shape, dtype=param.dtype)
            model.load_state_dict(state_dict, assign=True)

    @staticmethod
    def _build_and_setup_model(args, attn_backend):
        """
        Build model architecture, load checkpoints, and apply quantization.

        Arguments:
            args: Configuration with model settings
            attn_backend: The initialized attention backend

        Returns:
            Fully set up model
        """
        Backend.args = args

        with torch.device("meta"):
            model = Backend._build_model_architecture(args, attn_backend)
            init_native_layout(model)

        if args.debug.skip_model_load:
            Backend._create_empty_model(model, args.skip_preprocess)
        else:
            Backend._load_checkpoint(model, args)

        # Move model to appropriate device
        model.apply(Backend._move_one_module_to_device)

        if torch.distributed.get_rank() == 0:
            logger.debug(f"Model structure: \n{model}")

        Backend.model = model

        # Try to auto-register a MoE weight accessor provided by the model/experts
        try:
            Backend._auto_register_moe_weight_accessor_if_available()
        except Exception:
            pass

        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _auto_register_moe_weight_accessor_if_available() -> None:
        """Auto-detect a MoE ExpertParamAccessor from the model and register it.

        Search order:
        1) model.get_moe_weight_accessor()
        2) model.moe_weight_accessor
        3) first layer's mlp.experts exposing get_moe_weight_accessor()/moe_weight_accessor
        If found, call Backend.set_moe_weight_accessor(accessor).
        """
        # Only meaningful when EP is enabled
        try:
            ep_size = int(getattr(Backend.args.infer, "ep_size", 1))
        except Exception:
            ep_size = 1
        if ep_size <= 1:
            return
        model = Backend.model
        if model is None:
            return
        accessor = None
        # model-level hook
        if hasattr(model, "get_moe_weight_accessor") and callable(
            getattr(model, "get_moe_weight_accessor")
        ):
            accessor = model.get_moe_weight_accessor()
        elif hasattr(model, "moe_weight_accessor"):
            accessor = getattr(model, "moe_weight_accessor")
        # layer-level hook (common: model.layers[i].mlp.experts)
        if (
            accessor is None
            and hasattr(model, "layers")
            and len(getattr(model, "layers")) > 0
        ):
            layer0 = model.layers[0]
            mlp = getattr(layer0, "mlp", None)
            experts = getattr(mlp, "experts", None) if mlp is not None else None
            if experts is not None:
                if hasattr(experts, "get_moe_weight_accessor") and callable(
                    getattr(experts, "get_moe_weight_accessor")
                ):
                    accessor = experts.get_moe_weight_accessor()
                elif hasattr(experts, "moe_weight_accessor"):
                    accessor = getattr(experts, "moe_weight_accessor")
        if accessor is not None:
            Backend.set_moe_weight_accessor(accessor)

    @staticmethod
    def _build_model_architecture(args, attn_backend):
        """
        Build the model architecture based on configuration.

        Arguments:
            args: Configuration with model settings
            attn_backend: The initialized attention backend

        Returns:
            Initialized model architecture
        """
        if args.models.type in [
            ModelType.DEEPSEEK_V3,
            ModelType.KIMI_K2_5,
            ModelType.HF_QWEN_3_MOE,
        ]:
            QuantizationRegistry._allowed_quant_for_merge_gate_up.append("blockfp4")

        if (
            args.infer.mla_absorb == "absorb-kv-only"
            and args.multi_inst.role == "decode"
        ):
            raise ValueError(
                "infer.mla_absorb=absorb-kv-only is only valid for Prefill instances, "
                f"but multi_inst.role={args.multi_inst.role}."
            )

        if args.infer.mla_absorb == "absorb-kv-only":
            logger.warning(
                "infer.mla_absorb=absorb-kv-only keeps the KV cache latent-only "
                "and reconstructs full K/V (kv_b_proj over kv_lora + k_pe broadcast) "
                "every prefill step. Per-layer cost: one extra kv_b_proj GEMM over "
                "(current chunk + history) tokens. Cache layout stays latent-only "
                "and has no PD impact. Decode-only instances should use "
                "absorb-without-precomp instead."
            )

        model_kwargs = dict(
            max_position_embeddings=args.infer.max_seq_len
            + (args.infer.mtp_size if args.infer.mtp_size > 1 else 0),
            attn_backend=attn_backend,
            op_impl=args.infer.op_impl,
            mla_absorb=args.infer.mla_absorb,
        )
        if args.models.type == ModelType.DEEPSEEK_V4:
            model_kwargs.update(
                pipeline_parallel_size=args.infer.pp_size,
                tensor_parallel_size=args.infer.tp_size,
            )

        return Backend.build_model(
            args.models,
            Backend.cache_dict,
            **model_kwargs,
        )

    @staticmethod
    def _handle_quantized_weights_casting(checkpoint, args):
        # For the FP8 variants of Qwen (Qwen3-30B-A3B-fp8 and Qwen3-235B-A22B-fp8), some checkpoint parameters
        # are stored in full precision (FP32) by default, but at runtime they’re also cast to BF16
        if args.models.name in ["Qwen3-30B-A3B-fp8", "Qwen3-235B-A22B-fp8"]:
            for k in checkpoint.keys():
                if (
                    checkpoint[k].dtype == torch.float32
                    and "scale" not in k
                    and "layernorm" not in k
                    and "norm" not in k
                ):
                    checkpoint[k] = checkpoint[k].to(torch.get_default_dtype())
        # Some platforms do not support float8, but we can run them with `infer.raise_lower_bit_float_to=bfloat16`.
        # However, we need to treat float8 items as uint8 first, to avoid the missing ops on these platforms.
        for k in checkpoint.keys():
            quant = get_quant_from_checkpoint_prefix(k, args.models.quant_config.rules)
            if parse_dtype(args.infer.raise_lower_bit_float_to).itemsize > 1:
                if quant == "blockfp8" and checkpoint[k].element_size() == 1:
                    checkpoint[k] = checkpoint[k].view(dtype=torch.uint8)
            if (
                quant in ("blockfp4", "blockfp4_merged")
                and checkpoint[k].element_size() == 1
            ):
                checkpoint[k] = checkpoint[k].view(dtype=torch.uint8)

        return checkpoint

    @staticmethod
    def _support_layerwise_loading():
        if is_ascend():
            return False
        else:
            return True

    @staticmethod
    def _load_checkpoint(model, args):
        """
        Load model parameters from checkpoint files.

        Arguments:
            model: The model to load parameters into
            args: Configuration with checkpoint settings
        """
        start_time = time.time()

        if (
            args.models.type == ModelType.DEEPSEEK_V3
            and args.models.quant_config.type
            in [
                "gguf",
                "q4km",
            ]
        ):
            logger.info(f"loading gguf file : {args.models.ckpt_dir}")
            ds_gguf_loader = GGUFLoader(args.models.ckpt_dir)
            load_gguf_deepseek_v3_gguf(model, ds_gguf_loader, args)
        else:
            quant_config = getattr(args.models, "quant_config", None)
            quant_name = getattr(quant_config, "name", None)
            quant_type = getattr(quant_config, "type", None)
            if args.models.type == ModelType.LLAMA:
                checkpoints = sorted(Path(args.models.ckpt_dir).glob("*.pth"))
                assert (
                    len(checkpoints) > 0
                ), f"no checkpoint files found in {args.models.ckpt_dir}"
                ckpt_path = checkpoints[0]
                checkpoint = torch.load(ckpt_path, map_location="cpu")
            elif quant_type in [
                "w4a8_per_token_per_group_asymm",
                "w4a8_per_token_per_channel_asymm",
                "w4_g128_symm_a8",
            ]:
                checkpoint = torch.load(
                    os.path.join(args.models.ckpt_dir, "pytorch_model.bin"),
                    map_location="cpu",
                )
                checkpoint = Backend._remove_prefix(checkpoint, "model.")
            elif quant_name in ["gguf", "q4km"]:
                checkpoint = load_state_dict_llama_gguf_mlp_layers(
                    GGUFLoader(args.models.ckpt_dir), len(model.layers)
                )
            elif args.models.type in {
                ModelType.HF_LLAMA,
                ModelType.HF_QWEN_3_MOE,
                ModelType.HF_QWEN3_VL,
                ModelType.HF_QWEN3_VL_MOE,
                ModelType.HF_GLM_Z1,
                ModelType.HF_GLM_4_MOE,
                ModelType.HF_GPT_OSS,
                ModelType.HF_MIXTRAL,
                ModelType.DEEPSEEK_V3,
                ModelType.KIMI_K2_5,
                ModelType.GLM_5_2,
                ModelType.HF_QWEN2_VL,
                ModelType.HF_QWEN3_NEXT,
                ModelType.HF_QWEN3_5,
                ModelType.LLADA2,
                ModelType.DEEPSEEK_V4,
            }:
                if Backend._support_layerwise_loading():
                    checkpoint = Backend._load_hf_checkpoint_layerwise(model, args)
                    logger.info(
                        f"Checkpoint loaded in {time.time() - start_time:.2f} seconds"
                    )
                    return
                else:
                    checkpoint = Backend._load_hf_checkpoint(model, args)
            else:
                raise NotImplementedError(f"Unsupported model type {args.models.type}")

            checkpoint = Backend._handle_quantized_weights_casting(checkpoint, args)

            model.load_state_dict_parallel(
                checkpoint,
                strict=True,
                assign=True,  # Replacing "meta" tensors in the model with tensors from the checkpoint
                skip_preprocess=args.skip_preprocess,
            )
        for layer in model.layers:
            mlp_component = getattr(layer, "mlp", None)
            if mlp_component is not None:
                experts = getattr(mlp_component, "experts", None)
                if (
                    experts is not None
                    and hasattr(experts, "warm_up")
                    and callable(experts.warm_up)
                ):
                    experts.warm_up()

        logger.info(f"Checkpoint loaded in {time.time() - start_time:.2f} seconds")

    @staticmethod
    def _remove_prefix(state_dict, prefix):
        return {
            k[len(prefix) :] if k.startswith(prefix) else k: v
            for k, v in state_dict.items()
        }

    @staticmethod
    def _build_hf_key_filter(args):

        def key_filter(k: str) -> bool:
            if (
                is_ascend()
                and args.models.type == ModelType.DEEPSEEK_V3
                and k.endswith(".weight_offset")
            ):
                return False
            if args.infer.mtp_size == 1:
                if (
                    args.models.type
                    in [
                        ModelType.DEEPSEEK_V3,
                        ModelType.HF_GLM_4_MOE,
                        ModelType.GLM_5_2,
                    ]
                    and f"model.layers.{args.models.n_layers}" in k
                ):
                    return False
                if (
                    args.models.type in [ModelType.HF_QWEN3_NEXT, ModelType.HF_QWEN3_5]
                    and "mtp." in k
                ):
                    return False
                if args.models.type == ModelType.DEEPSEEK_V4 and k.startswith("mtp."):
                    return False
            if args.infer.language_model_only and k.startswith("model.visual"):
                return False
            quant_config = getattr(args.models, "quant_config", None)
            if getattr(quant_config, "type", None) == "blockfp4" and (
                k.endswith(".k_scale") or k.endswith(".v_scale")
            ):
                return False
            if args.models.name in [
                "Qwen3-8B-ascend-int8",
                "Qwen3-14B-ascend-int8",
                "Qwen3-32B-ascend-int8",
                "Qwen2.5-72B-Instruct-ascend-int8",
                "Qwen2.5-VL-32B-Instruct-ascend-int8",
            ] and (k.endswith(".weight_scale") or k.endswith(".weight_offset")):
                return False
            if getattr(args.models, "tie_word_embeddings", False) and "lm_head." in k:
                return False
            return True

        return key_filter

    @staticmethod
    def _load_hf_checkpoint_layerwise(model, args):

        key_filter = Backend._build_hf_key_filter(args)

        def _map_key(
            k: str,
            checkpoint_prefix: str,
            model_prefix: str,
        ) -> str:
            if checkpoint_prefix != model_prefix and k.startswith(checkpoint_prefix):
                k = f"{model_prefix}{k[len(checkpoint_prefix):]}"
            return k

        def _load_and_apply(
            checkpoint_prefix: str,
            model_prefix: str,
            local_layer_prefix: str | None = None,
            extra_prefix_dict: dict[str, str] = None,
        ):
            """
            Example layer prefixes:
            checkpoint_prefix:      model.layer.{global_id}
            model_prefix:           layer.{global_id}
            local_layer_prefix:     layer.{local_id}
            """
            if args.skip_preprocess:
                checkpoint_prefix = local_layer_prefix or model_prefix

            try:
                state_dict = load_state_dict(
                    args.models.ckpt_dir,
                    skip_preprocess=args.skip_preprocess,
                    key_filter=key_filter,
                    prefix=checkpoint_prefix,
                    prefix_list=(
                        list(extra_prefix_dict.keys())
                        if extra_prefix_dict is not None
                        else None
                    ),
                )
            except Exception as e:
                raise RuntimeError(
                    f"Error loading checkpoint part from files by prefix {checkpoint_prefix}"
                ) from e
            assert state_dict, f"No state dict found for prefix {checkpoint_prefix}"

            if not args.skip_preprocess:
                mapped = {}
                for k, v in state_dict.items():
                    mapped[
                        _map_key(
                            k,
                            checkpoint_prefix,
                            model_prefix,
                        )
                    ] = v
                state_dict = mapped

                if extra_prefix_dict is not None:
                    for k in list(state_dict.keys()):
                        v = state_dict.pop(k)
                        new_k = k
                        for prefix_, replacement_ in extra_prefix_dict.items():
                            new_k = new_k.replace(prefix_, replacement_)
                        state_dict[new_k] = v

            target_prefix = local_layer_prefix or model_prefix
            try:
                state_dict = Backend._handle_quantized_weights_casting(state_dict, args)
                module = model.load_state_dict_by_prefix(
                    state_dict, target_prefix, args.skip_preprocess
                )
                module.apply(Backend._move_one_module_to_device)
            except Exception as e:
                raise RuntimeError(
                    f"Error loading tensors into model part by prefix {target_prefix}"
                ) from e

        # Load non-layer weights
        for checkpoint_prefix, model_prefix in model._get_non_layer_prefix_mappings():
            _load_and_apply(checkpoint_prefix, model_prefix)

        # Load transformer layers
        is_print_rank = int(os.environ.get("LOCAL_RANK", 0)) == 0
        has_separate_mtp_layer = (
            args.infer.mtp_size > 1
            and args.models.type in {ModelType.HF_QWEN3_5, ModelType.DEEPSEEK_V4}
            and model.local_begin_layer_id
            <= args.models.n_layers
            < model.local_end_layer_id
        )
        local_main_end_layer_id = (
            model.local_end_layer_id - 1
            if has_separate_mtp_layer
            else model.local_end_layer_id
        )
        for global_layer_id in tqdm(
            range(model.local_begin_layer_id, local_main_end_layer_id),
            disable=not is_print_rank,
            desc="Model loading",
            unit="layer",
            leave=False,
        ):
            checkpoint_prefix, model_prefix = model._get_layer_i_prefix_mapping(
                global_layer_id
            )
            local_layer_id = global_layer_id - model.local_begin_layer_id

            local_layer_prefix = f"layers.{local_layer_id}."
            _load_and_apply(
                checkpoint_prefix,
                model_prefix,
                local_layer_prefix=local_layer_prefix,
            )

        if has_separate_mtp_layer:
            for global_layer_id in tqdm(
                range(local_main_end_layer_id, model.local_end_layer_id),
                disable=not is_print_rank,
                desc="Model loading",
                unit="layer",
                leave=False,
            ):
                checkpoint_prefix, model_prefix, extra_prefix_dict = (
                    model._get_layer_mtp_prefix_mapping(global_layer_id)
                )
                local_layer_id = global_layer_id - model.local_begin_layer_id

                local_layer_prefix = f"layers.{local_layer_id}."
                _load_and_apply(
                    checkpoint_prefix,
                    model_prefix,
                    local_layer_prefix=local_layer_prefix,
                    extra_prefix_dict=extra_prefix_dict,
                )
        torch.cuda.empty_cache()

    @staticmethod
    def _load_hf_checkpoint(model, args):
        """
        Load checkpoint for Hugging Face model types.

        Arguments:
            args: Configuration with checkpoint settings

        Returns:
            Loaded checkpoint dictionary
        """

        key_filter = Backend._build_hf_key_filter(args)

        params = load_state_dict(
            args.models.ckpt_dir,
            skip_preprocess=args.skip_preprocess,
            key_filter=key_filter,
        )
        params = Backend._remove_prefix(params, "model.")
        return Backend._remove_prefix(params, "language_model.")

    @staticmethod
    def build(args):
        """
        Build and initialize the model, tokenizer, cache manager, and other components required for inference.

        Arguments:
            args: Configuration object containing model and training related configurations.
        """
        # Initialize distributed environment
        Backend._init_distributed(args)
        register_all_providers()

        # Dense KVCache and PP related
        if args.infer.cache_type == "skew":
            max_reqs_per_dp = compute_local_batch_size_dist_in_dp(
                args.infer.max_batch_size, args.infer.dp_size
            )[get_dp_group().rank_in_group]
            set_slot_handle(max_reqs_per_dp, args.infer.pp_size)

        if hasattr(args.models, "n_routed_experts") or hasattr(
            args.models, "num_experts"
        ):
            init_moe_impl(args)

        # Setup environment and basic configuration
        Backend._setup_environment(args)

        # Initialize tokenizer and formatter
        Backend.tokenizer = Backend._init_tokenizer(args)
        Backend.processor = Backend._init_processor(args)
        Backend.formatter = Backend._init_formatter(args)

        attn_backend_type = Backend._get_attention_backend_type(args)
        logger.info(f"attn_backend_type={attn_backend_type.__name__}")

        # Initialize cache managers
        bundle = build_cache_managers(args, attn_backend_type)
        Backend.cache_type = bundle.cache_type
        Backend.cache_dict = bundle.cache_dict
        Backend.cache_managers = bundle.cache_managers

        # Initialize attention backend
        attn_backend = Backend._init_attention_backend(attn_backend_type, args.models)

        Backend._build_and_setup_model(args, attn_backend)

        # After model/expert modules are fully built and registered, optionally run MoE P2P self-test
        try:
            from chitu.moe.load_balancer import warmup_for_moe_schema

            warmup_for_moe_schema()
        except Exception as _e:
            logger.debug(f"Skip/failed running warmup for MoE schema: {_e}")

        logger.info(
            f"Backend initialized with CUDA mem at {torch.cuda.memory_allocated()/1024**3:.2f} GB"
        )
        logger.info(
            f"Using {len(c10d._pg_map)} communication groups. If this number is too high, there may be too much memory reserved for underlying communication libraries."
        )
        return Backend

    @staticmethod
    def stop():
        setattr(Backend, "model", None)
        Backend.cache_dict.clear()
        setattr(Backend, "cache_managers", None)
        gc.collect()
        torch.cuda.empty_cache()


def load_state_dict(
    hf_ckpt_path,
    *,
    skip_preprocess=False,
    key_filter: Callable[[str], bool] = None,
    prefix: str = "",
    prefix_list: list[str] = None,
):
    if not skip_preprocess:
        path = os.path.join(hf_ckpt_path, "*.safetensors")
    else:
        rank = torch.distributed.get_rank()
        path = os.path.join(hf_ckpt_path, f"model.rank{rank}.safetensors")

    state_dict = {}
    ignored_params = []
    for file_path in glob(path):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if prefix and not name.startswith(prefix):
                    continue
                if key_filter is None or key_filter(name):
                    param: torch.Tensor = f.get_tensor(name)
                    state_dict[name] = param
                else:
                    ignored_params.append(name)

            if prefix_list is not None:
                for name in f.keys():
                    if name.startswith(tuple(prefix_list)):
                        param: torch.Tensor = f.get_tensor(name)
                        state_dict[name] = param

    if ignored_params:
        logger.info(
            f"Ignored some parameters because related model features are disabled: {ignored_params}"
        )

    return state_dict


def memory_used():
    logger.debug(
        f"gpu memory usage: {torch.cuda.memory_allocated()/(1024**3)} GB"
    )  # torch.cuda.max_memory_allocated()/(1024**3)) #, torch.cuda.memory_reserved()/(1024**3))
    import resource

    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logger.debug(f"cpu memory usage: {memory_usage / 1024} MB")


def load_gguf_deepseek_v3_gguf(model, ds_gguf_loader: GGUFLoader, args=None):
    logger.debug(f"loading layer : from 0 to 3")
    checkpoint0 = load_state_dict_deepseek_v3_gguf_mlp_layer(
        ds_gguf_loader, main_weight_dtype=args.models.main_weight_dtype
    )
    for _, model_prefix in model._get_non_layer_prefix_mappings():
        model.load_state_dict_by_prefix(
            {k: v for k, v in checkpoint0.items() if k.startswith(model_prefix)},
            model_prefix,
            replace=False,
            skip_preprocess=args.skip_preprocess,
        )
    for layer_id in range(3):
        _, model_prefix = model._get_layer_i_prefix_mapping(layer_id)
        model.load_state_dict_by_prefix(
            {k: v for k, v in checkpoint0.items() if k.startswith(model_prefix)},
            model_prefix,
            replace=False,
            skip_preprocess=args.skip_preprocess,
        )
    model.apply(
        functools.partial(
            Backend._move_one_module_to_device,
            non_blocking=False,  # Wait for done, and the memory is free'd
            ignore_not_loaded=True,
        )
    )
    del checkpoint0
    gc.collect()
    torch.cuda.empty_cache()
    cpu_layers = utils.collect_layers_by_type(
        ["q4km", "gguf"], args.models.quant_config.rules
    )
    for layer_id in range(3, 61):
        checkpoint = load_state_dict_deepseek_v3_gguf_moe_layer(
            ds_gguf_loader,
            cpu_layers,
            layer_id,
            layer_id + 1,
            parallel_moe_load=True,
            main_weight_dtype=args.models.main_weight_dtype,
        )
        _, model_prefix = model._get_layer_i_prefix_mapping(layer_id)
        model.load_state_dict_by_prefix(
            checkpoint,
            model_prefix,
            replace=False,
            skip_preprocess=args.skip_preprocess,
        )
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("initing cpu tensors!")


def load_state_dict_llama_gguf_mlp_layers(llama_gguf_loader: GGUFLoader, layer_num=64):
    state_dict = {}

    state_dict["embed_tokens.weight"] = llama_gguf_loader.load_gguf_tensor(
        name="token_embd.weight", target_dtype=torch.bfloat16
    )
    state_dict["lm_head.weight"] = llama_gguf_loader.load_gguf_tensor(
        name="output.weight", target_dtype=torch.bfloat16
    )
    state_dict["norm.weight"] = llama_gguf_loader.load_gguf_tensor(
        name="output_norm.weight", target_dtype=torch.bfloat16
    )

    translation_llama = {
        ".input_layernorm.weight": ".attn_norm.weight",
        ".self_attn.q_proj.weight": ".attn_q.weight",
        ".self_attn.q_proj.bias": ".attn_q.bias",
        ".self_attn.k_proj.weight": ".attn_k.weight",
        ".self_attn.k_proj.bias": ".attn_k.bias",
        ".self_attn.v_proj.weight": ".attn_v.weight",
        ".self_attn.v_proj.bias": ".attn_v.bias",
        ".self_attn.o_proj.weight": ".attn_output.weight",
        ".mlp.down_proj.weight": ".ffn_down.weight",
        ".mlp.gate_proj.weight": ".ffn_gate.weight",
        ".mlp.up_proj.weight": ".ffn_up.weight",
        ".post_attention_layernorm.weight": ".ffn_norm.weight",
    }

    for layer_id in range(64):
        for k in translation_llama.keys():
            safetensor_name = "layers." + str(layer_id) + k
            gguf_name = "blk." + str(layer_id) + translation_llama[k]
            state_dict[safetensor_name] = llama_gguf_loader.load_gguf_tensor(
                name=gguf_name, target_dtype=torch.bfloat16
            )

    return state_dict


def quant_fp8(x: torch.Tensor, block_size: int = 128):
    m = x.shape[0]
    n = x.shape[1]
    # assert (m % block_size == 0) and (n % block_size == 0)
    qm = (m + block_size - 1) // block_size
    qn = (n + block_size - 1) // block_size
    zx = torch.zeros([qm * block_size, qn * block_size], dtype=x.dtype, device=x.device)
    zx[:m, :n] = x
    qx = zx.view(qm, block_size, qn, block_size).transpose(1, 2)
    scale = torch.max(torch.max(torch.abs(qx), dim=-1)[0], dim=-1)[0]
    scale = scale.to(torch.float32) / 448
    xscale = torch.stack([torch.stack([scale] * block_size, dim=1)] * block_size, dim=3)
    xscale = xscale.reshape(qm * block_size, qn * block_size)
    qx = zx / xscale
    qx = qx[:m, :n]
    qx = qx.clip(-448, 448)
    return qx.to(torch.float8_e4m3fn), scale


def load_state_dict_deepseek_v3_gguf_mlp_layer(
    ds_gguf_loader: GGUFLoader, main_weight_dtype="float8_e4m3fn"
):
    torch.set_num_threads(8)

    device = torch.device("cuda")
    state_dict = {}

    state_dict["embed_tokens.weight"] = ds_gguf_loader.load_gguf_tensor(
        name="token_embd.weight", target_dtype=torch.bfloat16
    )
    state_dict["lm_head.weight"] = ds_gguf_loader.load_gguf_tensor(
        name="output.weight", target_dtype=torch.bfloat16
    )
    state_dict["norm.weight"] = ds_gguf_loader.load_gguf_tensor(
        name="output_norm.weight", target_dtype=torch.bfloat16
    )

    translation_attn = {
        ".input_layernorm.weight": ".attn_norm.weight",
        ".self_attn.kv_a_layernorm.weight": ".attn_kv_a_norm.weight",
        ".self_attn.kv_a_proj_with_mqa.weight": ".attn_kv_a_mqa.weight",
        ".self_attn.kv_b_proj.weight": ".attn_kv_b.weight",
        ".self_attn.o_proj.weight": ".attn_output.weight",
        ".self_attn.q_a_layernorm.weight": ".attn_q_a_norm.weight",
        ".self_attn.q_a_proj.weight": ".attn_q_a.weight",
        ".self_attn.q_b_proj.weight": ".attn_q_b.weight",
    }

    translation_mlp = {
        ".mlp.down_proj.weight": ".ffn_down.weight",
        ".mlp.gate_proj.weight": ".ffn_gate.weight",
        ".mlp.up_proj.weight": ".ffn_up.weight",
        ".post_attention_layernorm.weight": ".ffn_norm.weight",
    }

    for layer_id in range(3):
        logger.info(f"loading layer : {layer_id}")
        for k in translation_attn.keys():
            safetensor_name = "layers." + str(layer_id) + k
            gguf_name = "blk." + str(layer_id) + translation_attn[k]
            if main_weight_dtype == "float8_e4m3fn" and not safetensor_name.endswith(
                "norm.weight"
            ):
                safetensor_scale = safetensor_name[:-6] + "scale"
                weight = ds_gguf_loader.load_gguf_tensor(
                    gguf_name, device, torch.bfloat16
                )
                weight, scale = quant_fp8(weight, block_size=128)
                state_dict[safetensor_name] = weight.cpu()
                state_dict[safetensor_scale] = scale.cpu()

            else:
                state_dict[safetensor_name] = ds_gguf_loader.load_gguf_tensor(
                    name=gguf_name, target_dtype=torch.bfloat16
                )

        for k in translation_mlp.keys():
            safetensor_name = "layers." + str(layer_id) + k
            gguf_name = "blk." + str(layer_id) + translation_mlp[k]
            if main_weight_dtype == "float8_e4m3fn" and not safetensor_name.endswith(
                "norm.weight"
            ):
                safetensor_scale = safetensor_name[:-6] + "scale"
                weight, scale = quant_fp8(
                    ds_gguf_loader.load_gguf_tensor(gguf_name, device, torch.bfloat16),
                    block_size=128,
                )
                state_dict[safetensor_name] = weight.cpu()
                state_dict[safetensor_scale] = scale.cpu()

            else:
                state_dict[safetensor_name] = ds_gguf_loader.load_gguf_tensor(
                    name=gguf_name, target_dtype=torch.bfloat16
                )

    return state_dict


def load_state_dict_deepseek_v3_gguf_moe_layer(
    ds_gguf_loader: GGUFLoader,
    cpu_layers,
    start_layer: int,
    end_layer: int,
    parallel_moe_load=True,
    main_weight_dtype="float8_e4m3fn",
):
    torch.set_num_threads(8)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        memory_used()
    device = torch.device("cuda")
    state_dict = {}

    translation_attn = {
        ".input_layernorm.weight": ".attn_norm.weight",
        ".self_attn.kv_a_layernorm.weight": ".attn_kv_a_norm.weight",
        ".self_attn.kv_a_proj_with_mqa.weight": ".attn_kv_a_mqa.weight",
        ".self_attn.kv_b_proj.weight": ".attn_kv_b.weight",
        ".self_attn.o_proj.weight": ".attn_output.weight",
        ".self_attn.q_a_layernorm.weight": ".attn_q_a_norm.weight",
        ".self_attn.q_a_proj.weight": ".attn_q_a.weight",
        ".self_attn.q_b_proj.weight": ".attn_q_b.weight",
    }

    translation_gate = {
        ".mlp.gate.e_score_correction_bias": ".exp_probs_b.bias",
        ".mlp.gate.weight": ".ffn_gate_inp.weight",
        ".post_attention_layernorm.weight": ".ffn_norm.weight",
    }

    translation_shared_experts = {
        ".mlp.shared_experts.down_proj.weight": ".ffn_down_shexp.weight",
        ".mlp.shared_experts.gate_proj.weight": ".ffn_gate_shexp.weight",
        ".mlp.shared_experts.up_proj.weight": ".ffn_up_shexp.weight",
    }

    translation_experts = {
        ".down_proj.weight": ".ffn_down_exps.weight",
        ".gate_proj.weight": ".ffn_gate_exps.weight",
        ".up_proj.weight": ".ffn_up_exps.weight",
    }

    # cpu_layer = list(range(100))
    global_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    for layer_id in range(start_layer, end_layer):
        cpu_offload = layer_id in cpu_layers
        if local_rank == 0:
            logger.info(f"loading layer : {layer_id}")
            memory_used()
        for k in translation_attn.keys():
            safetensor_name = "layers." + str(layer_id) + k
            gguf_name = "blk." + str(layer_id) + translation_attn[k]
            if main_weight_dtype == "float8_e4m3fn" and not safetensor_name.endswith(
                "norm.weight"
            ):
                safetensor_scale = safetensor_name[:-6] + "scale"
                weight, scale = quant_fp8(
                    ds_gguf_loader.load_gguf_tensor(gguf_name, device, torch.bfloat16),
                    block_size=128,
                )
                state_dict[safetensor_name] = weight.cpu()
                state_dict[safetensor_scale] = scale.cpu()

            else:
                state_dict[safetensor_name] = ds_gguf_loader.load_gguf_tensor(
                    gguf_name, device, torch.bfloat16
                ).cpu()

        for k in translation_gate.keys():
            safetensor_name = "layers." + str(layer_id) + k
            gguf_name = "blk." + str(layer_id) + translation_gate[k]
            if "bias" in safetensor_name:
                state_dict[safetensor_name] = ds_gguf_loader.load_gguf_tensor(
                    gguf_name, device, torch.float32
                ).cpu()
            else:
                state_dict[safetensor_name] = ds_gguf_loader.load_gguf_tensor(
                    gguf_name, device, torch.bfloat16
                ).cpu()

        for k in translation_shared_experts.keys():
            safetensor_name = "layers." + str(layer_id) + k
            gguf_name = "blk." + str(layer_id) + translation_shared_experts[k]
            if main_weight_dtype == "float8_e4m3fn" and not safetensor_name.endswith(
                "norm.weight"
            ):
                safetensor_scale = safetensor_name[:-6] + "scale"
                weight, scale = quant_fp8(
                    ds_gguf_loader.load_gguf_tensor(gguf_name, device, torch.bfloat16),
                    block_size=128,
                )
                state_dict[safetensor_name] = weight.cpu()
                state_dict[safetensor_scale] = scale.cpu()

            else:
                state_dict[safetensor_name] = ds_gguf_loader.load_gguf_tensor(
                    gguf_name, device, torch.bfloat16
                ).cpu()

        if not cpu_offload:
            if parallel_moe_load:
                for k in translation_experts.keys():
                    gguf_name = "blk." + str(layer_id) + translation_experts[k]
                    tinfo = ds_gguf_loader.tensor_info[gguf_name]
                    data = ds_gguf_loader.get_mmap_tensor(gguf_name)
                    shape = tinfo["shape"]
                    ggml_type = tinfo["ggml_type"]

                    expert_tensor = ds_gguf_loader.load_gguf_tensor_dist(
                        data,
                        shape,
                        ggml_type,
                        "cpu",
                        torch.bfloat16,
                        global_rank,
                        world_size,
                    )

                    if (
                        main_weight_dtype == "float8_e4m3fn"
                        and not safetensor_name.endswith("norm.weight")
                    ):
                        experts_weight, experts_scale = quant_fp8(
                            expert_tensor,
                            block_size=128,
                        )
                        experts_weight = experts_weight.cuda()
                        experts_scale = experts_scale.cuda()
                        gathered_weight = [
                            torch.zeros_like(experts_weight, device=device)
                            for _ in range(world_size)
                        ]
                        gathered_scale = [
                            torch.zeros_like(experts_scale, device=device)
                            for _ in range(world_size)
                        ]
                        dist.all_gather(gathered_weight, experts_weight)
                        dist.all_gather(gathered_scale, experts_scale)
                        gathered_experts_weight = torch.concat(
                            gathered_weight, dim=0
                        ).cpu()
                        gathered_experts_scale = torch.concat(
                            gathered_scale, dim=0
                        ).cpu()
                        torch.cuda.empty_cache()
                    else:
                        gathered_experts = [
                            torch.zeros_like(expert_tensor, device=device)
                            for _ in range(world_size)
                        ]
                        dist.all_gather(gathered_experts, expert_tensor)
                        gathered_experts = torch.concat(gathered_experts, dim=0).cpu()

                    safetensor_name = "layers." + str(layer_id) + k
                    for expert_id in range(256):
                        safetensor_name = (
                            "layers."
                            + str(layer_id)
                            + ".mlp.experts."
                            + str(expert_id)
                            + k
                        )
                        if (
                            main_weight_dtype == "float8_e4m3fn"
                            and not safetensor_name.endswith("norm.weight")
                        ):

                            safetensor_scale = safetensor_name[:-6] + "scale"
                            state_dict[safetensor_name] = gathered_experts_weight[
                                expert_id
                            ].cpu()
                            state_dict[safetensor_scale] = gathered_experts_scale[
                                expert_id
                            ].cpu()

                        else:
                            state_dict[safetensor_name] = (
                                ds_gguf_loader.load_gguf_tensor(
                                    name=gguf_name, target_dtype=torch.bfloat16
                                )
                            )

            else:
                for k in translation_experts.keys():
                    gguf_name = "blk." + str(layer_id) + translation_experts[k]
                    expert_tensor = ds_gguf_loader.load_gguf_tensor(
                        name=gguf_name, target_dtype=torch.bfloat16
                    )
                    safetensor_name = "layers." + str(layer_id) + k
                    for expert_id in range(256):
                        safetensor_name = (
                            "layers."
                            + str(layer_id)
                            + ".mlp.experts."
                            + str(expert_id)
                            + k
                        )
                        if (
                            main_weight_dtype == "float8_e4m3fn"
                            and not safetensor_name.endswith("norm.weight")
                        ):
                            safetensor_scale = safetensor_name[:-6] + "scale"
                            weight, scale = quant_fp8(
                                expert_tensor[expert_id],
                                block_size=128,
                            )
                            state_dict[safetensor_name] = weight
                            state_dict[safetensor_scale] = scale

                        else:
                            state_dict[safetensor_name] = expert_tensor[expert_id]

        else:
            if local_rank == 0:
                gate_proj, gate_type = (
                    ds_gguf_loader.get_undequanted_tensor_and_ggml_type(
                        f"blk.{layer_id}.ffn_gate_exps.weight"
                    )
                )
                up_proj, up_type = ds_gguf_loader.get_undequanted_tensor_and_ggml_type(
                    f"blk.{layer_id}.ffn_up_exps.weight"
                )
                down_proj, down_type = (
                    ds_gguf_loader.get_undequanted_tensor_and_ggml_type(
                        f"blk.{layer_id}.ffn_down_exps.weight"
                    )
                )

                state_dict[
                    "layers." + str(layer_id) + ".mlp.experts.gguf_gate_proj"
                ] = gate_proj
                state_dict["layers." + str(layer_id) + ".mlp.experts.gguf_up_proj"] = (
                    up_proj
                )
                state_dict[
                    "layers." + str(layer_id) + ".mlp.experts.gguf_down_proj"
                ] = down_proj
                state_dict["layers." + str(layer_id) + ".mlp.experts.gate_type"] = (
                    torch.tensor(gate_type)
                )
                state_dict["layers." + str(layer_id) + ".mlp.experts.up_type"] = (
                    torch.tensor(up_type)
                )
                state_dict["layers." + str(layer_id) + ".mlp.experts.down_type"] = (
                    torch.tensor(down_type)
                )

    return state_dict
