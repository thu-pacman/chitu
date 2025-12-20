# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import gc
import itertools
import functools
import os
import time
import re
from collections import deque
from enum import Enum
from glob import glob
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Deque, Optional, Iterable
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from safetensors.torch import safe_open
from tqdm import tqdm
from chitu.attn_backend import (
    FlashAttnBackend,
    FlashInferBackend,
    FlashMLABackend,
    NpuAttnBackend,
    RefAttnBackend,
    TritonAttnBackend,
    NpuAttnBackend,
    HybridAttnBackend,
)
from chitu.cache_manager import (
    DenseKVCacheManager,
    PagedKVCacheManager,
    SingletonPagedKVCacheManager,
    GlobalLocalMap,
)
from chitu.custom_gguf import *
from chitu.device_type import is_ascend, is_muxi
from chitu.distributed.parallel_state import (
    get_world_group,
    get_pp_group,
    initialize_parallel_groups,
)
from chitu.hybrid_device import CPUParameter
from chitu.models.registry import ModelType, get_model_class
from chitu.quantization import (
    QuantizationRegistry,
    QuantizedMoeExpertsBase,
    get_quant_from_checkpoint_prefix,
    utils,
)
from chitu.tokenizer import ChatFormat, ChatFormatHF, Tokenizer, TokenizerHF, Processor
from chitu.utils import (
    compute_layer_dist_in_pipe,
    parse_dtype,
    try_import_opt_dep,
    ceil_div,
)

# from chitu.distributed.moe_token_dispatcher import init_token_dispatcher
from chitu.moe import init_moe_impl

if TYPE_CHECKING:
    from chitu.executor import Executor
    from chitu.scheduler import Scheduler
    from chitu.task import BatchResult

numa, has_numa = try_import_opt_dep("numa", "cpu")
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
    cache_manager = None
    formatter = None
    processor = None
    args = None
    # --- cache_manager related (not used in the current code)
    curr_req_ids = None
    cache_type = ""
    # ---
    use_gloo = True
    group_gloo = None
    pp_stage = None
    pp_end_stage = None
    pp_main_rank = None

    # components
    scheduler: Optional["Scheduler"] = None
    executor: Optional["Executor"] = None

    # mutable
    state = BackendState.Running
    last_batch_results: Deque["BatchResult"] = deque()
    indexer_cache_manager = None

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

            register_moe_weight_accessor(accessor)
            logger.info("Backend: MoE weight accessor installed and registered")
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
                import torch

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
                logger.info(
                    "Backend: auto-built ModelExpertsAccessor and registered to planner"
                )
            except Exception as e:
                logger.warning(
                    f"Backend: failed to build/register ModelExpertsAccessor: {e}"
                )

    @staticmethod
    def build_model(args, cache, *extra_args, **extra_kwargs):
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
        return model_cls(args, cache, *extra_args, **extra_kwargs)

    # FIXME: When cache type is "skew", gloo backend cannot be used.
    @staticmethod
    def _init_distributed(args):
        """
        Initialize distributed training environment with tensor and pipeline parallelism.

        Arguments:
            args: Configuration object with distributed parameters
        """
        is_router_process = os.environ.get("CHITU_ROUTER_PROCESS", "0") == "1"
        if is_router_process:
            # Router process: as independent subprocess, skip CUDA device binding
            logger.info(f"[Router] Router subprocess skip CUDA device binding")
            return

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Bind process to GPU. Please put it before init_process_group
        if args.infer.op_impl != "cpu":
            torch.cuda.set_device(local_rank)

        if not torch.distributed.is_initialized():
            if args.infer.op_impl == "cpu":
                torch.distributed.init_process_group("gloo")
            else:
                torch.distributed.init_process_group("nccl")
        if Backend.use_gloo:
            Backend.group_gloo = torch.distributed.new_group(backend="gloo")

        tensor_parallel_size = args.infer.tp_size
        pipeline_parallel_size = args.infer.pp_size
        non_expert_data_parallel_size = args.infer.dp_size
        expert_parallel_size = args.infer.ep_size
        assert (
            tensor_parallel_size * non_expert_data_parallel_size % expert_parallel_size
            == 0
        )
        expert_tensor_parallel_size = (
            tensor_parallel_size * non_expert_data_parallel_size // expert_parallel_size
        )

        global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        if (
            world_size
            != tensor_parallel_size
            * non_expert_data_parallel_size
            * pipeline_parallel_size
        ):
            raise ValueError(
                f"Inconsistent parallelism: world_size({world_size}) should be equal to "
                f"tensor_parallel_size({tensor_parallel_size}) "
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
        )
        Backend.ip_port_list = get_world_group().gather_all_rank_ip_port()

        Backend.pp_stage = (
            global_rank
            % (world_size // non_expert_data_parallel_size)
            // tensor_parallel_size
        )
        Backend.pp_end_stage = (
            world_size // non_expert_data_parallel_size - 1
        ) // tensor_parallel_size
        Backend.pp_main_rank = (
            global_rank // tensor_parallel_size
        ) * tensor_parallel_size

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
        trust_remote_code = args.models.name.startswith("glm-4")
        force_full_seq_decode = (
            args.models.tokenizer_force_full_seq_decode
            if hasattr(args.models, "tokenizer_force_full_seq_decode")
            else False
        )

        if args.models.tokenizer_type == "hf":
            tokenizer = TokenizerHF(
                path=args.models.tokenizer_path,
                trust_remote_code=trust_remote_code,
                force_full_seq_decode=force_full_seq_decode,
            )
        else:
            tokenizer = Tokenizer(
                model_path=args.models.tokenizer_path,
                force_full_seq_decode=force_full_seq_decode,
            )
            assert (
                args.models.vocab_size == tokenizer.n_words
            ), f"{args.models.vocab_size} vs. {tokenizer.n_words}"

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

        if not hasattr(args.models, "vision_config"):
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
        if args.models.tokenizer_type == "hf":
            return ChatFormatHF(Backend.tokenizer, Backend.processor)
        else:
            return ChatFormat(Backend.tokenizer)

    @staticmethod
    def _init_cache_manager(
        args,
        attn_backend_type,
        layer_filter_fn=lambda x: x,
        num_blocks: int = None,
    ):
        """
        Initialize the appropriate KV cache manager based on configuration.

        Arguments:
            args: Configuration with cache and model settings

        Returns:
            Initialized cache manager
        """
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if args.infer.op_impl == "cpu":
            local_rank = "cpu"

        pipeline_parallel_size = args.infer.pp_size

        # Determine layer distribution for pipeline parallelism
        if pipeline_parallel_size > 1:
            pipe_stage = get_pp_group().rank_in_group
            num_layers_of_each_rank = compute_layer_dist_in_pipe(
                args.models.n_layers, pipeline_parallel_size
            )
            first_layer_id_of_each_rank = list(
                itertools.accumulate([0] + num_layers_of_each_rank)
            )
            local_begin_layer_id = first_layer_id_of_each_rank[pipe_stage]
            local_end_layer_id = first_layer_id_of_each_rank[pipe_stage + 1]
        else:
            local_begin_layer_id = 0
            local_end_layer_id = args.models.n_layers

        local_layers = layer_filter_fn(range(local_begin_layer_id, local_end_layer_id))
        layer_id_map = GlobalLocalMap.from_list(local_layers)

        # Configure KV cache parameters based on model type
        kv_cache_kvargs = Backend._get_kv_cache_params(args, attn_backend_type)

        # Create appropriate cache manager
        if args.infer.cache_type == "paged":
            block_size = 64 if args.infer.mla_absorb != "none" else 256
            if args.infer.attn_type == "npu":
                block_size = 128

            return PagedKVCacheManager(
                layer_id_map,
                max_seq_len=args.infer.max_seq_len,
                num_hot_req=ceil_div(args.infer.max_reqs, args.infer.dp_size),
                block_size=block_size,
                num_blocks=args.infer.num_blocks if num_blocks is None else num_blocks,
                device=local_rank,
                **kv_cache_kvargs,
            )
        elif args.infer.cache_type == "skew":
            return DenseKVCacheManager(
                layer_id_map,
                max_seq_len=args.infer.max_seq_len,
                num_hot_req=ceil_div(args.infer.max_reqs, args.infer.dp_size),
                device=local_rank,
                **kv_cache_kvargs,
            )
        else:
            raise ValueError(f"Unknown cache type {args.infer.cache_type}")

    @staticmethod
    def _init_linear_attn_cache_manager(
        args, layer_filter_fn=lambda x: x, num_blocks: int = None
    ):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if args.infer.op_impl == "cpu":
            local_rank = "cpu"

        pipeline_parallel_size = args.infer.pp_size

        # Determine layer distribution for pipeline parallelism
        if pipeline_parallel_size > 1:
            pipe_stage = get_pp_group().rank_in_group
            num_layers_of_each_rank = compute_layer_dist_in_pipe(
                args.models.n_layers, pipeline_parallel_size
            )
            first_layer_id_of_each_rank = list(
                itertools.accumulate([0] + num_layers_of_each_rank)
            )
            local_begin_layer_id = first_layer_id_of_each_rank[pipe_stage]
            local_end_layer_id = first_layer_id_of_each_rank[pipe_stage + 1]
        else:
            local_begin_layer_id = 0
            local_end_layer_id = args.models.n_layers

        local_layers = layer_filter_fn(range(local_begin_layer_id, local_end_layer_id))
        layer_id_map = GlobalLocalMap.from_list(local_layers)

        return SingletonPagedKVCacheManager(
            layer_id_map,
            num_hot_req=ceil_div(args.infer.max_reqs, args.infer.dp_size),
            shape_per_token_dict=Backend._get_linear_attn_cache_params(args),
            device=local_rank,
        )

    @staticmethod
    def _init_indexer_cache_manager(
        args, layer_filter_fn=lambda x: x, num_blocks: int = None
    ):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if args.infer.op_impl == "cpu":
            local_rank = "cpu"

        pipeline_parallel_size = args.infer.pp_size

        if pipeline_parallel_size > 1:
            pipe_stage = get_pp_group().rank_in_group
            num_layers_of_each_rank = compute_layer_dist_in_pipe(
                args.models.n_layers, pipeline_parallel_size
            )
            first_layer_id_of_each_rank = list(
                itertools.accumulate([0] + num_layers_of_each_rank)
            )
            local_begin_layer_id = first_layer_id_of_each_rank[pipe_stage]
            local_end_layer_id = first_layer_id_of_each_rank[pipe_stage + 1]
        else:
            local_begin_layer_id = 0
            local_end_layer_id = args.models.n_layers

        local_layers = layer_filter_fn(range(local_begin_layer_id, local_end_layer_id))
        layer_id_map = GlobalLocalMap.from_list(local_layers)

        # Shapes for indexer caches
        index_head_dim = getattr(args.models, "index_head_dim", None)
        if index_head_dim is None or index_head_dim <= 0:
            logger.warning(
                f"Index head dim is not set or is not positive, skipping indexer cache manager"
            )
            return None

        shape_per_token_dict = {
            "indexer_k": (int(index_head_dim),),
            "indexer_ks": (int(index_head_dim) // 128,),
        }
        dtype_dict = {
            "indexer_k": torch.float8_e4m3fn,
            "indexer_ks": torch.float32,
        }

        # Align page size with main paged KV
        block_size = 64 if args.infer.mla_absorb != "none" else 256
        if args.infer.attn_type == "npu":
            block_size = 128

        # FIXME: for now, we use the same block size as the main paged KV cache manager
        # Compute sufficient num_blocks for indexer cache when not provided
        # Ensure enough pages for max_seq_len per hot request to avoid OOB when crossing pages
        num_hot_req = ceil_div(args.infer.max_reqs, args.infer.dp_size)
        auto_num_blocks = ceil_div(args.infer.max_seq_len, block_size) * num_hot_req
        num_blocks = (
            args.infer.num_blocks if args.infer.num_blocks != -1 else auto_num_blocks
        )

        return PagedKVCacheManager(
            layer_id_map,
            max_seq_len=args.infer.max_seq_len,
            num_hot_req=(args.infer.max_reqs + args.infer.dp_size - 1)
            // args.infer.dp_size,
            shape_per_token_dict=shape_per_token_dict,
            dtype_dict=dtype_dict,
            block_size=block_size,
            num_blocks=num_blocks,
            device=local_rank,
        )

    @staticmethod
    def _get_kv_cache_params(args, attn_backend_type):
        """
        Calculate the KV cache parameters based on model type and configuration.

        Arguments:
            args: Configuration with model settings

        Returns:
            Dictionary of parameters for KV cache initialization
        """
        tensor_parallel_size = args.infer.tp_size

        kv_cache_kvargs = {}

        if args.models.type == "deepseek-v3":
            if args.infer.mla_absorb in ["absorb", "absorb-without-precomp"]:
                use_separated_kv_lora_k_pe = attn_backend_type in [
                    FlashInferBackend,
                    TritonAttnBackend,
                ]
                if use_separated_kv_lora_k_pe:
                    kv_cache_kvargs["shape_per_token_dict"] = {
                        "kv_lora": (args.models.kv_lora_rank,),
                        "k_pe": (args.models.qk_rope_head_dim,),
                    }
                else:
                    kv_cache_kvargs["shape_per_token_dict"] = {
                        "kv_lora_k_pe": (
                            args.models.kv_lora_rank + args.models.qk_rope_head_dim,
                        )
                    }
            elif args.infer.mla_absorb == "none":
                n_local_heads = args.models.n_heads // tensor_parallel_size
                k_head_dim = args.models.qk_nope_head_dim + args.models.qk_rope_head_dim
                v_head_dim = args.models.v_head_dim
                kv_cache_kvargs["shape_per_token_dict"] = {
                    "k": (n_local_heads, k_head_dim),
                    "v": (n_local_heads, v_head_dim),
                }
            else:
                raise NotImplementedError(
                    f"Unsupported mla_absorb {args.infer.mla_absorb}"
                )
        else:
            n_kv_heads = (
                args.models.n_kv_heads
                if hasattr(args.models, "n_kv_heads")
                else args.models.n_heads
            )
            n_local_kv_heads = (
                n_kv_heads // tensor_parallel_size
                if n_kv_heads > tensor_parallel_size
                else 1
            )  # Compatible with tp_size>n_kv_heads
            head_dim = (
                args.models.head_dim
                if hasattr(args.models, "head_dim")
                else args.models.dim // args.models.n_heads
            )
            kv_cache_kvargs["n_local_kv_heads"] = n_local_kv_heads
            kv_cache_kvargs["head_dim"] = head_dim

        return kv_cache_kvargs

    @staticmethod
    def _get_linear_attn_cache_params(args):
        tensor_parallel_size = args.infer.tp_size

        n_v_heads = args.models.linear_n_v_heads
        n_qk_heads = args.models.linear_n_qk_heads
        head_dim = args.models.linear_head_dim
        conv_kernel_size = args.models.linear_conv_kernel_dim

        n_local_v_heads = n_v_heads // tensor_parallel_size
        local_conv_dim = (n_qk_heads * 2 + n_v_heads) * head_dim // tensor_parallel_size

        return {
            "conv_state": (local_conv_dim, conv_kernel_size),
            "recurrent_state": (n_local_v_heads, head_dim, head_dim),
        }

    # @staticmethod
    # def _init_linear_attn_cache(args):
    #     return Qwen3LinearAttnCacheManager()

    @staticmethod
    def _get_attention_backend_type(args):
        if args.infer.attn_type == "auto":
            if is_ascend():
                return NpuAttnBackend
            elif args.infer.op_impl == "cpu":
                return RefAttnBackend
            elif "deepseek-v3" in args.models.type:
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
        elif args.infer.attn_type == "triton":
            return TritonAttnBackend
        elif args.infer.attn_type == "npu":
            return NpuAttnBackend
        elif args.infer.attn_type == "ref":
            return RefAttnBackend
        else:
            raise ValueError(f"Unknown attn type {args.infer.attn_type}")

    @staticmethod
    def _init_attention_backend(attn_backend_type):
        # Yes, use `type` instead of `isinstance` here, because `AttnBackend`s inherit each other
        if attn_backend_type is FlashInferBackend:
            assert isinstance(Backend.cache_manager, PagedKVCacheManager)
            return attn_backend_type(Backend.cache_manager.get_max_num_blocks())
        else:
            return attn_backend_type()

    @staticmethod
    def _move_one_module_to_device(
        m: torch.nn.Module, non_blocking: bool = True, ignore_not_loaded: bool = False
    ):
        # NOTE: m._parameters contains parameters in this module (non-recursive),
        # while m.parameters() returns all parameters in this module and its submodules
        # (recursive).
        for key in m._parameters:
            param = m._parameters[key]
            if param is not None:
                if not isinstance(param, CPUParameter):
                    if param.device == torch.device("meta"):
                        if not ignore_not_loaded:
                            assert False, f"Unexpected unloaded parameter {key}"
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
        if not args.debug.skip_model_load:
            # Build the model. Don't allocate memory yet.
            with torch.device("meta"):
                model = Backend._build_model_architecture(args, attn_backend)

            # Load model parameters
            Backend._load_checkpoint(model, args)

        else:
            # Use initialized weights
            model = Backend._build_model_architecture(args, attn_backend)

        # Move model to appropriate device
        if args.infer.op_impl != "cpu":
            model.apply(Backend._move_one_module_to_device)

        if torch.distributed.get_rank() == 0:
            logger.debug(f"Model structure: \n{model}")

        Backend.model = model
        Backend.args = args

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
        if args.models.type in ["deepseek-v3", "hf-qwen-3-moe"]:
            QuantizationRegistry._allowed_quant_for_merge_gate_up.append("blockfp4")

        tensor_parallel_size = args.infer.tp_size
        pipeline_parallel_size = args.infer.pp_size

        if args.models.type == "hf-qwen3-next":
            model = Backend.build_model(
                args.models,
                Backend.cache_manager,
                max_position_embeddings=args.infer.max_seq_len,
                pipeline_parallel_size=pipeline_parallel_size,
                tensor_parallel_size=tensor_parallel_size,
                attn_backend=attn_backend,
                op_impl=args.infer.op_impl,
                mla_absorb=args.infer.mla_absorb,
                linear_attn_cache=Backend.linear_attn_cache_manager,
            )
        elif (
            args.models.type == "deepseek-v3"
            and Backend.indexer_cache_manager is not None
        ):
            model = Backend.build_model(
                args.models,
                Backend.cache_manager,
                max_position_embeddings=args.infer.max_seq_len,
                pipeline_parallel_size=pipeline_parallel_size,
                tensor_parallel_size=tensor_parallel_size,
                attn_backend=attn_backend,
                op_impl=args.infer.op_impl,
                mla_absorb=args.infer.mla_absorb,
                indexer_cache=Backend.indexer_cache_manager,
            )
        else:
            model = Backend.build_model(
                args.models,
                Backend.cache_manager,
                max_position_embeddings=args.infer.max_seq_len,
                pipeline_parallel_size=pipeline_parallel_size,
                tensor_parallel_size=tensor_parallel_size,
                attn_backend=attn_backend,
                op_impl=args.infer.op_impl,
                mla_absorb=args.infer.mla_absorb,
            )

        return model

    @staticmethod
    def _load_checkpoint(model, args):
        """
        Load model parameters from checkpoint files.

        Arguments:
            model: The model to load parameters into
            args: Configuration with checkpoint settings
        """
        start_time = time.time()

        if args.models.type == "deepseek-v3" and args.models.quant_config.type in [
            "gguf",
            "q4km",
        ]:
            logger.info(f"loading gguf file : {args.models.ckpt_dir}")
            ds_gguf_loader = GGUFLoader(args.models.ckpt_dir)
            load_gguf_deepseek_v3_gguf(model, ds_gguf_loader, 10, args)

        else:
            quant_config = getattr(args.models, "quant_config", None)
            quant_name = getattr(quant_config, "name", None)
            quant_type = getattr(quant_config, "type", None)
            if args.models.type == "llama":
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
                "hf-llama",
                "hf-qwen-3-moe",
                "hf-glm-z1",
                "hf-glm-4-moe",
                "hf-gpt-oss",
                "hf-mixtral",
                "deepseek-v3",
                "hf-qwen2-vl",
                "hf-qwen3-next",
            }:
                checkpoint = Backend._load_hf_checkpoint(model, args)
            else:
                raise NotImplementedError(f"Unsupported model type {args.models.type}")

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
                quant = get_quant_from_checkpoint_prefix(
                    k, args.models.quant_config.rules
                )
                if parse_dtype(args.infer.raise_lower_bit_float_to).itemsize > 1:
                    if quant == "blockfp8" and checkpoint[k].element_size() == 1:
                        checkpoint[k] = checkpoint[k].view(dtype=torch.uint8)
                if (
                    quant in ("blockfp4", "blockfp4_merged")
                    and checkpoint[k].element_size() == 1
                ):
                    checkpoint[k] = checkpoint[k].view(dtype=torch.uint8)
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
    def _load_hf_checkpoint(model, args):
        """
        Load checkpoint for Hugging Face model types.

        Arguments:
            args: Configuration with checkpoint settings

        Returns:
            Loaded checkpoint dictionary
        """

        def key_filter(k: str) -> bool:
            if (
                is_ascend()
                and args.models.type == "deepseek-v3"
                and k.endswith(".weight_offset")
            ):
                return False
            if (
                args.models.type == "deepseek-v3"
                and "model.layers.61" in k
                and args.infer.mtp_size == 1
            ):
                return False
            if (
                args.models.name == "GLM-4.5-Air"
                and "model.layers.46" in k
                and args.infer.mtp_size == 1
            ):
                return False
            if (
                args.models.name in ["GLM-4.5", "GLM-4.6"]
                and "model.layers.92" in k
                and args.infer.mtp_size == 1
            ):
                return False
            if args.models.quant_config.type == "blockfp4" and (
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
            match = re.search(r"model\.layers\.(\d+)\.", k)
            if match and int(match.group(1)) >= args.models.n_layers:
                return False
            return True

        params = load_state_dict(
            args.models.ckpt_dir,
            skip_preprocess=args.skip_preprocess,
            filter_key=key_filter,
        )
        return Backend._remove_prefix(params, "model.")

    @staticmethod
    def build(args):
        """
        Build and initialize the model, tokenizer, cache manager, and other components required for inference.

        Arguments:
            args: Configuration object containing model and training related configurations.
        """
        # Initialize distributed environment
        Backend._init_distributed(args)

        init_moe_impl(args)

        # Setup environment and basic configuration
        Backend._setup_environment(args)

        # Initialize tokenizer and formatter
        Backend.tokenizer = Backend._init_tokenizer(args)
        Backend.processor = Backend._init_processor(args)
        Backend.formatter = Backend._init_formatter(args)

        attn_backend_type = Backend._get_attention_backend_type(args)

        # Initialize cache manager
        if args.models.type == "hf-qwen3-next":

            def is_full_attention(layer_id):
                return (layer_id + 1) % args.models.full_attention_interval == 0

            def filter_full_attn_layer(layers: Iterable[int]):
                return [idx for idx in layers if is_full_attention(idx)]

            def filter_linear_attn_layer(layers: Iterable[int]):
                return [idx for idx in layers if not is_full_attention(idx)]

            num_full_attn_blocks = (
                args.infer.num_blocks
                if args.models.num_full_attention_blocks == -1
                else args.models.num_full_attention_blocks
            )
            num_linear_attn_blocks = (
                args.infer.num_blocks
                if args.models.num_linear_attention_blocks == -1
                else args.models.num_linear_attention_blocks
            )

            Backend.cache_type = args.infer.cache_type
            Backend.cache_manager = Backend._init_cache_manager(
                args,
                attn_backend_type,
                layer_filter_fn=filter_full_attn_layer,
                num_blocks=num_full_attn_blocks,
            )
            Backend.linear_attn_cache_manager = Backend._init_linear_attn_cache_manager(
                args,
                layer_filter_fn=filter_linear_attn_layer,
                num_blocks=num_linear_attn_blocks,
            )
        elif getattr(args.models, "type", "") == "deepseek-v3" and getattr(
            args.models, "index_head_dim", None
        ):
            Backend.cache_type = args.infer.cache_type
            Backend.cache_manager = Backend._init_cache_manager(args, attn_backend_type)
            Backend.indexer_cache_manager = Backend._init_indexer_cache_manager(args)
        else:
            Backend.cache_manager = Backend._init_cache_manager(args, attn_backend_type)
            Backend.cache_type = args.infer.cache_type

        # Initialize attention backend
        attn_backend = Backend._init_attention_backend(attn_backend_type)

        Backend._build_and_setup_model(args, attn_backend)

        # After model/expert modules are fully built and registered, optionally run MoE P2P self-test
        try:
            from chitu.moe.load_balancer import warmup_for_moe_schema

            warmup_for_moe_schema()
        except Exception as _e:
            logger.debug(f"Skip/failed running warmup for MoE schema: {_e}")

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        logger.info(
            f"rank {local_rank} Backend initialized with CUDA mem at {torch.cuda.memory_allocated()/1024**3:.2f} GB"
        )
        logger.info(
            f"Using {len(c10d._pg_map)} communication gruops. If this number is too high, there may be too much memory reserved for underlying communication libraries."
        )
        return Backend

    @staticmethod
    def stop():
        setattr(Backend, "model", None)
        setattr(Backend, "cache_manager", None)
        gc.collect()
        torch.cuda.empty_cache()


def load_state_dict(
    hf_ckpt_path, *, skip_preprocess=False, filter_key: Callable[[str], bool] = None
):
    if not skip_preprocess:
        path = os.path.join(hf_ckpt_path, "*.safetensors")
    else:
        rank = torch.distributed.get_rank()
        path = os.path.join(hf_ckpt_path, f"model.rank{rank}.safetensors")

    state_dict = {}
    ignored_params = []
    for file_path in tqdm(glob(path)):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if filter_key is None or filter_key(name):
                    param: torch.Tensor = f.get_tensor(name)
                    state_dict[name] = param
                else:
                    ignored_params.append(name)

    if ignored_params:
        logger.warning(f"Ignored {len(ignored_params)} params: {ignored_params}")

    return state_dict


def memory_used():
    logger.debug(
        f"gpu memory usage: {torch.cuda.memory_allocated()/(1024**3)} GB"
    )  # torch.cuda.max_memory_allocated()/(1024**3)) #, torch.cuda.memory_reserved()/(1024**3))
    import resource

    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logger.debug(f"cpu memory usage: {memory_usage / 1024} MB")


def load_gguf_deepseek_v3_gguf(
    model, ds_gguf_loader: GGUFLoader, layer_load_per_iter=10, args=None
):
    logger.debug(f"loading layer : from 0 to 3")
    checkpoint0 = load_state_dict_deepseek_v3_gguf_mlp_layer(
        ds_gguf_loader, main_weight_dtype=args.models.main_weight_dtype
    )
    model.load_state_dict_parallel(
        checkpoint0,
        strict=False,
        replace=False,
        assign=True,  # Replacing "meta" tensors in the model with tensors from the checkpoint
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
    for layer_id in range(3, 61, layer_load_per_iter):
        end_layer = min(61, layer_id + layer_load_per_iter)
        checkpoint = load_state_dict_deepseek_v3_gguf_moe_layer(
            ds_gguf_loader,
            cpu_layers,
            layer_id,
            end_layer,
            parallel_moe_load=True,
            main_weight_dtype=args.models.main_weight_dtype,
        )
        model.load_state_dict_parallel(
            checkpoint,
            strict=False,
            replace=False,
            assign=True,  # Replacing "meta" tensors in the model with tensors from the checkpoint
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

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
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
    device = f"cuda:{local_rank}"
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
