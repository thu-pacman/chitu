from typing import Callable
import gc
import itertools
import json
import os
import sys
import time
from enum import Enum
from glob import glob
from logging import getLogger
from pathlib import Path

import torch
from safetensors.torch import safe_open
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM

from chitu.attn_backend import (
    FlashAttnBackend,
    FlashInferBackend,
    FlashMLABackend,
    RefAttnBackend,
    TritonAttnBackend,
)
from chitu.cache_manager import (
    KVCacheManager,
    KVCacheManagerNop,
    KVCacheManagerSkewAware,
    PagedKVCacheManager,
)
from chitu.models.model_deepseek_v3 import TransformerDeepSeekV3
from chitu.models.model_hf_llama import TransformerHFLlama
from chitu.models.model_hf_mixtral import TransformerHFMixtral
from chitu.models.model_llama import TransformerLlama
from chitu.tensor_parallel import get_tp_size, init_tp
from chitu.tokenizer import ChatFormat, ChatFormatHF, Tokenizer, TokenizerHF
from chitu.utils import compute_layer_dist_in_pipe

logger = getLogger(__name__)


def check_checkpoint_path(args):
    if args.models.ckpt_dir is None:
        raise ValueError(
            f"No checkpoint path provided. You can set it in command line by adding `models.ckpt_dir=<path>`. The model {args.models.name} can be downloaded from {args.models.source}"
        )
    if args.models.tokenizer_path is None:
        logger.info(
            f"Using {args.models.ckpt_dir} as the path to tokenizer. If the tokenizer has a different path, please set in command line by adding `models.tokenizer_path=<path>`"
        )
        args.models.tokenizer_path = args.models.ckpt_dir


class BackendState(Enum):
    Running = 1
    Terminating = 2  # All tasks done, but rank 0 should tell others to terminate
    Terminated = 3


class Backend:
    model = None
    tokenizer = None
    formatter = None
    args = None
    curr_varlens = None
    curr_req_ids = None
    ongoing_reqs = []
    cache_type = ""
    state = BackendState.Running
    pp_stage = None
    pp_end_stage = None
    pp_main_rank = None

    @staticmethod
    def build_model(args, cache, *extra_args, **extra_kwargs):
        if args.type == "hf-llama":
            if args.name.startswith("glm-4"):
                extra_kwargs["rotary_type"] = "glm4"
            return TransformerHFLlama(args, cache, *extra_args, **extra_kwargs)
        elif args.type == "hf-mixtral":
            return TransformerHFMixtral(args, cache, *extra_args, **extra_kwargs)
        elif args.type == "llama":
            return TransformerLlama(args, cache, *extra_args, **extra_kwargs)
        elif args.type == "deepseek-v3":
            return TransformerDeepSeekV3(args, cache, *extra_args, **extra_kwargs)
        else:
            assert False, f"Unknown model type {args.models.type}"

    @staticmethod
    def _init_distributed(args):
        """
        Initialize distributed training environment with tensor and pipeline parallelism.

        Arguments:
            args: Configuration object with distributed parameters
        """
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")

        model_parallel_size = args.infer.tp_size
        pipeline_parallel_size = args.infer.pp_size
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        assert (
            world_size == model_parallel_size * pipeline_parallel_size
        ), "World size not match"

        torch.cuda.set_device(local_rank)
        init_tp(model_parallel_size, pipeline_parallel_size)

        Backend.pp_stage = global_rank // model_parallel_size
        Backend.pp_end_stage = (world_size - 1) // model_parallel_size
        Backend.pp_main_rank = (
            global_rank // model_parallel_size
        ) * model_parallel_size

    @staticmethod
    def _setup_environment(args):
        """
        Set up random seed, default dtype, and check prerequisites.

        Arguments:
            args: Configuration with seed and dtype settings
        """
        torch.manual_seed(args.infer.seed)

        # Set default_dtype
        if args.dtype == "float16":
            torch.set_default_dtype(torch.float16)
        elif args.dtype == "bfloat16":
            torch.set_default_dtype(torch.bfloat16)
        else:
            raise NotImplementedError(f"Unsupported dtype {args.dtype}")

        # Check checkpoint exists
        check_checkpoint_path(args)

    @staticmethod
    def _init_tokenizer(args):
        """
        Initialize the appropriate tokenizer based on model type.

        Arguments:
            args: Configuration with tokenizer settings

        Returns:
            Initialized tokenizer
        """
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        trust_remote_code = args.models.name.startswith("glm-4")
        force_full_seq_decode = (
            args.models.tokenizer_force_full_seq_decode
            if hasattr(args.models, "tokenizer_force_full_seq_decode")
            else False
        )

        if (
            args.models.type == "hf-llama"
            or args.models.type == "hf-mixtral"
            or args.models.type == "deepseek-v3"
        ):
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

        tokenizer.stop_tokens = torch.tensor(
            list(
                [tokenizer.stop_tokens]
                if isinstance(tokenizer.stop_tokens, int)
                else tokenizer.stop_tokens
            ),
            device=local_rank,
        )

        return tokenizer

    @staticmethod
    def _init_formatter(args):
        """
        Initialize the chat formatter based on model type.

        Arguments:
            args: Configuration with model settings

        Returns:
            Appropriate chat formatter instance
        """
        if args.models.type in ["hf-llama", "hf-mixtral", "deepseek-v3"]:
            return ChatFormatHF(Backend.tokenizer)
        else:
            return ChatFormat(Backend.tokenizer)

    @staticmethod
    def _init_cache_manager(args):
        """
        Initialize the appropriate KV cache manager based on configuration.

        Arguments:
            args: Configuration with cache and model settings

        Returns:
            Initialized cache manager
        """
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        pipeline_parallel_size = args.infer.pp_size

        # Determine layer distribution for pipeline parallelism
        if pipeline_parallel_size > 1:
            num_layers_of_each_rank = compute_layer_dist_in_pipe(
                args.models.n_layers, pipeline_parallel_size
            )
            first_layer_id_of_each_rank = list(
                itertools.accumulate([0] + num_layers_of_each_rank)
            )
            local_begin_layer_id = first_layer_id_of_each_rank[Backend.pp_stage]
            local_end_layer_id = first_layer_id_of_each_rank[Backend.pp_stage + 1]
        else:
            local_begin_layer_id = 0
            local_end_layer_id = args.models.n_layers

        # Configure KV cache parameters based on model type
        kv_cache_kvargs = Backend._get_kv_cache_params(args)

        # Create appropriate cache manager
        if args.infer.cache_type == "normal":
            return KVCacheManager(
                local_begin_layer_id,
                local_end_layer_id,
                **kv_cache_kvargs,
            )
        elif args.infer.cache_type == "nop":
            return KVCacheManagerNop(
                local_begin_layer_id,
                local_end_layer_id,
                max_seq_len=args.infer.max_seq_len,
                num_hot_req=args.infer.max_reqs,
                device=local_rank,
                **kv_cache_kvargs,
            )
        elif args.infer.cache_type == "paged":
            block_size = 64 if args.infer.mla_absorb != "none" else 256
            return PagedKVCacheManager(
                local_begin_layer_id,
                local_end_layer_id,
                max_seq_len=args.infer.max_seq_len,
                num_hot_req=args.infer.max_reqs,
                block_size=block_size,
                device=local_rank,
                **kv_cache_kvargs,
            )
        elif args.infer.cache_type == "skew":
            return KVCacheManagerSkewAware(
                local_begin_layer_id,
                local_end_layer_id,
                max_seq_len=args.infer.max_seq_len,
                num_hot_req=args.infer.max_reqs,
                device=local_rank,
                **kv_cache_kvargs,
            )
        else:
            raise ValueError(f"Unknown cache type {args.infer.cache_type}")

    @staticmethod
    def _get_kv_cache_params(args):
        """
        Calculate the KV cache parameters based on model type and configuration.

        Arguments:
            args: Configuration with model settings

        Returns:
            Dictionary of parameters for KV cache initialization
        """
        model_parallel_size = args.infer.tp_size

        kv_cache_kvargs = {}

        if args.models.type == "deepseek-v3":
            if args.infer.mla_absorb in ["absorb", "absorb-without-precomp"]:
                if args.infer.cache_type == "paged":
                    kv_cache_kvargs["kv_shape_per_sample"] = (
                        args.models.kv_lora_rank + args.models.qk_rope_head_dim,
                    )
                else:
                    kv_cache_kvargs["k_shape_per_sample"] = (args.models.kv_lora_rank,)
                    kv_cache_kvargs["v_shape_per_sample"] = (
                        args.models.qk_rope_head_dim,
                    )
            elif args.infer.mla_absorb == "none":
                n_local_heads = args.models.n_heads // model_parallel_size
                k_head_dim = args.models.qk_nope_head_dim + args.models.qk_rope_head_dim
                v_head_dim = args.models.v_head_dim
                kv_cache_kvargs["k_shape_per_sample"] = (n_local_heads, k_head_dim)
                kv_cache_kvargs["v_shape_per_sample"] = (n_local_heads, v_head_dim)
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
            n_local_kv_heads = n_kv_heads // model_parallel_size
            head_dim = args.models.dim // args.models.n_heads
            kv_cache_kvargs["n_local_kv_heads"] = n_local_kv_heads
            kv_cache_kvargs["head_dim"] = head_dim

        return kv_cache_kvargs

    @staticmethod
    def _init_attention_backend(args):
        """
        Initialize the appropriate attention backend based on configuration.

        Arguments:
            args: Configuration with attention settings

        Returns:
            Initialized attention backend
        """
        if args.infer.attn_type == "flash_attn":
            return FlashAttnBackend()
        elif args.infer.attn_type == "flash_mla":
            return FlashMLABackend()
        elif args.infer.attn_type == "flash_infer":
            assert isinstance(Backend.cache_manager, PagedKVCacheManager)
            return FlashInferBackend(Backend.cache_manager.get_num_blocks())
        elif args.infer.attn_type == "triton":
            return TritonAttnBackend()
        elif args.infer.attn_type == "ref":
            return RefAttnBackend()
        else:
            raise ValueError(f"Unknown attn type {args.infer.attn_type}")

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
        # Build the model
        model = Backend._build_model_architecture(args, attn_backend)

        # Load model parameters if needed
        if args.infer.do_load:
            Backend._load_checkpoint(model, args)

        # Move model to appropriate device
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = model.to(local_rank)
        Backend.model = model
        Backend.args = args

        return model

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
        model_parallel_size = args.infer.tp_size
        pipeline_parallel_size = args.infer.pp_size

        # Determine whether to merge QKV, gate, and up projections
        merge_qkv_gate_up = True
        if args.models.type == "llama":
            merge_qkv_gate_up = False  # Not yet supported

        if hasattr(args.models, "quant") and args.models.quant is not None:
            # Merge weights for offline-scaled quantized models is non-trivial, because we can
            # only merge weights but NOT the scales on input dimensions, and this will break the
            # assumption of the fused quantized kernels. So we only merge weights for supported
            # quantization methods.
            merge_qkv_gate_up = False

        model = Backend.build_model(
            args.models,
            Backend.cache_manager,
            max_position_embeddings=args.infer.max_seq_len,
            pipeline_parallel_size=pipeline_parallel_size,
            model_parallel_size=model_parallel_size,
            attn_backend=attn_backend,
            op_impl=args.infer.op_impl,
            merge_qkv_gate_up=merge_qkv_gate_up,
            mla_absorb=args.infer.mla_absorb,
        )

        # Handle model precision
        if hasattr(args.models, "quant") and args.models.quant in [
            "awq",
            "llmint8",
            "gptq",
            "w8a16",
            "simple_w8a8",
            "simple_w8a8_muxi",
        ]:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

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

        if args.models.type == "llama":
            checkpoints = sorted(Path(args.models.ckpt_dir).glob("*.pth"))
            assert (
                len(checkpoints) > 0
            ), f"no checkpoint files found in {args.models.ckpt_dir}"
            ckpt_path = checkpoints[0]
            checkpoint = torch.load(ckpt_path, map_location="cpu")
        elif (
            args.models.type == "hf-llama"
            or args.models.type == "hf-mixtral"
            or args.models.type == "deepseek-v3"
        ):
            checkpoint = Backend._load_hf_checkpoint(args)
        else:
            raise NotImplementedError(f"Unsupported model type {args.models.type}")

        model.load_state_dict_parallel(
            checkpoint,
            strict=True,
            assign=args.keep_dtype_in_checkpoint,
            skip_preprocess=args.skip_preprocess,
        )

        logger.info(f"Checkpoint loaded in {time.time() - start_time:.2f} seconds")

    @staticmethod
    def _load_hf_checkpoint(args):
        """
        Load checkpoint for Hugging Face model types.

        Arguments:
            args: Configuration with checkpoint settings

        Returns:
            Loaded checkpoint dictionary
        """
        trust_remote_code = args.models.name.startswith("glm-4")

        if hasattr(args.models, "quant") and args.models.quant == "awq":
            params = torch.load(args.models.ckpt_dir, map_location="cpu")
            replace_list = [
                ("model.", ""),
                ("embed_tokens.weight", "embed_tokens.tok_embeddings.weight"),
            ]

            def rep(s):
                for p in replace_list:
                    s = s.replace(p[0], p[1], 1)
                return s

            checkpoint = dict((rep(k), v) for k, v in params.items())
        elif hasattr(args.models, "quant") and args.models.quant == "gptq":
            params = AutoModelForCausalLM.from_pretrained(
                args.models.ckpt_dir,
                torch_dtype="auto",
                device_map="cpu",
                trust_remote_code=trust_remote_code,
            ).state_dict()

            def transform_key(key):
                if key.startswith("model."):
                    return key[len("model.") :]
                return key

            checkpoint = dict((transform_key(k), v) for k, v in params.items())
        elif hasattr(args.models, "quant") and args.models.quant == "w8a16":
            params = torch.load(
                args.models.ckpt_dir + "/pytorch_model.bin", map_location="cpu"
            )
            replace_list = [
                ("model.", ""),
            ]

            def rep(s):
                for p in replace_list:
                    s = s.replace(p[0], p[1], 1)
                return s

            checkpoint = dict((rep(k), v) for k, v in params.items())
        else:
            filter_key = None
            if args.models.type == "deepseek-v3":
                filter_key = lambda key: "model.layers.61" not in key
            params = load_state_dict(
                args.models.ckpt_dir,
                skip_preprocess=args.skip_preprocess,
                filter_key=filter_key,
            )

            def transform_key(key):
                if key.startswith("model."):
                    return key[len("model.") :]
                return key

            checkpoint = dict((transform_key(k), v) for k, v in params.items())

        return checkpoint

    @staticmethod
    def build(args):
        """
        Build and initialize the model, tokenizer, cache manager, and other components required for inference.

        Arguments:
            args: Configuration object containing model and training related configurations.
        """
        # Initialize distributed environment
        Backend._init_distributed(args)

        # Setup environment and basic configuration
        Backend._setup_environment(args)

        # Initialize tokenizer and formatter
        Backend.tokenizer = Backend._init_tokenizer(args)
        Backend.formatter = Backend._init_formatter(args)

        # Initialize cache manager
        Backend.cache_manager = Backend._init_cache_manager(args)
        Backend.cache_type = args.infer.cache_type

        # Initialize attention backend
        attn_backend = Backend._init_attention_backend(args)

        # Build and setup model
        Backend._build_and_setup_model(args, attn_backend)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        logger.info(
            f"rank {local_rank} Backend initialized with CUDA mem at {torch.cuda.memory_allocated()}"
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
    for file_path in tqdm(glob(path)):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if filter_key is None or filter_key(name):
                    param: torch.Tensor = f.get_tensor(name)
                    state_dict[name] = param
    return state_dict
