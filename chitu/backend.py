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

import gc
import sys
from chitu.custom_gguf import *
import torch.distributed as dist


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
    cpu_infer = None

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

        if args.models.type in ["hf-llama", "hf-mixtral", "deepseek-v3"]:
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
                kv_cache_kvargs["kv_shape_per_sample"] = (
                    args.models.kv_lora_rank + args.models.qk_rope_head_dim,
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
        Backend.model = model.to(local_rank)
        Backend.args = args

        gc.collect()
        torch.cuda.empty_cache()

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
        if args.models.name == "QwQ-32B-FP8":
            merge_qkv_gate_up = False  # FIXME

        if hasattr(args.models, "quant") and args.models.quant not in [
            None,
            "blockfp8",
            "blockfp4",
        ]:
            # Merge weights for offline-scaled quantized models is non-trivial, because we can
            # only merge weights but NOT the scales on input dimensions, and this will break the
            # assumption of the fused quantized kernels. So we only merge weights for supported
            # quantization methods.
            merge_qkv_gate_up = False

        if args.models.type == "deepseek-v3" and args.quant == "gguf":
            import cpuinfer

            cpu_layer_num = (
                args.cpu_layer_num if hasattr(args.models, "cpu_layer_num") else 58
            )
            Backend.cpu_infer = cpuinfer.CPUInfer(args.models.cpu_num_thread)
            Backend.cpu_layers = list(range(61 - cpu_layer_num, 61))
            Backend.ggml_type = [
                0,
                0,
                0,
                14,
                14,
                14,
                14,
                12,
                12,
                14,
                12,
                12,
                14,
                12,
                12,
                14,
                12,
                12,
                14,
                12,
                12,
                14,
                12,
                12,
                14,
                12,
                12,
                14,
                12,
                12,
                14,
                12,
                12,
                14,
                12,
                12,
                14,
                12,
                12,
                14,
                12,
                12,
                14,
                12,
                12,
                14,
                12,
                12,
                14,
                12,
                12,
                14,
                12,
                14,
                14,
                14,
                14,
                14,
                14,
                14,
                14,
            ]
        else:
            Backend.cpu_infer = None
            Backend.cpu_layers = []
            Backend.ggml_type = []
        model = Backend.build_model(
            args.models,
            Backend.cache_manager,
            cpu_infer=Backend.cpu_infer,
            cpu_layers=Backend.cpu_layers,
            ggml_type=Backend.ggml_type,
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
        elif args.models.type == "hf-llama" or args.models.type == "hf-mixtral":
            checkpoint = Backend._load_hf_checkpoint(model, args)
        elif args.models.type == "deepseek-v3":
            if args.quant is None:
                checkpoint = load_state_dict_deepseek_v3(
                    args.models.ckpt_dir, skip_preprocess=args.skip_preprocess
                )
        else:
            raise NotImplementedError(f"Unsupported model type {args.models.type}")

        if args.models.type == "deepseek-v3" and args.quant == "gguf":
            logger.info(f"loading gguf file : {args.models.ckpt_dir}")
            ds_gguf_loader = GGUFLoader(args.models.ckpt_dir)
            load_gguf_deepseek_v3_gguf(
                model, ds_gguf_loader, Backend.cpu_layers, 10, args
            )
        else:
            model.load_state_dict_parallel(
                checkpoint,
                strict=True,
                assign=args.keep_dtype_in_checkpoint,
                skip_preprocess=args.skip_preprocess,
            )

        logger.info(f"Checkpoint loaded in {time.time() - start_time:.2f} seconds")

    def _load_hf_checkpoint(model, args):
        """
        Load checkpoint for Hugging Face model types.

        Arguments:
            args: Configuration with checkpoint settings

        Returns:
            Loaded checkpoint dictionary
        """
        trust_remote_code = args.models.name.startswith("glm-4")

        if args.quant == "awq":
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
        if args.quant == "autoawq":
            params = load_state_dict(args.models.ckpt_dir)
            replace_list = [
                ("model.", ""),
            ]

            def rep(s):
                for p in replace_list:
                    s = s.replace(p[0], p[1], 1)
                return s

            checkpoint = dict((rep(k), v) for k, v in params.items())
        elif args.quant == "gptq":
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
        elif args.quant == "gguf":
            llama_gguf_loader = llama_gguf_loader = GGUFLoader(args.models.ckpt_dir)
            checkpoint = load_state_dict_llama_gguf_mlp_layers(
                llama_gguf_loader, len(model.layers)
            )
        else:
            model_path = args.models.ckpt_dir
            params = load_state_dict(model_path, skip_preprocess=args.skip_preprocess)

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
                param: torch.Tensor = f.get_tensor(name)
                state_dict[name] = param
    return state_dict


def memory_used():
    logger.debug(
        f"gpu memory usage : RANK : {torch.cuda.current_device()} {torch.cuda.memory_allocated()/(1024**3)} GB"
    )  # torch.cuda.max_memory_allocated()/(1024**3)) #, torch.cuda.memory_reserved()/(1024**3))
    import resource

    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logger.debug(f"cpu memory usage: {memory_usage / 1024} MB")


def load_gguf_deepseek_v3_gguf(
    model, ds_gguf_loader: GGUFLoader, cpu_layers, layer_load_per_iter=10, args=None
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    logger.debug(f"loading layer : from 0 to 3")
    checkpoint0 = load_state_dict_deepseek_v3_gguf_mlp_layer(
        ds_gguf_loader, main_weight_dtype=args.models.main_weight_dtype
    )
    model.load_state_dict_parallel(
        checkpoint0,
        strict=False,
        replace=False,
        assign=args.keep_dtype_in_checkpoint,
        skip_preprocess=args.skip_preprocess,
    )
    model = model.to(local_rank)
    del checkpoint0
    gc.collect()
    torch.cuda.empty_cache()

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
            assign=args.keep_dtype_in_checkpoint,
            skip_preprocess=args.skip_preprocess,
        )
        # model = model.to(local_rank)
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("initing cpu tensors!")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    for layer_id in range(3, 61):
        if layer_id in cpu_layers:
            if model.layers[layer_id].ffn.moe == None:
                model.layers[layer_id].ffn.init_weights()


def load_state_dict_deepseek_v3(hf_ckpt_path, skip_preprocess=False):
    torch.set_num_threads(8)

    if not skip_preprocess:
        path = os.path.join(hf_ckpt_path, "*.safetensors")
    else:
        rank = torch.distributed.get_rank()
        path = os.path.join(hf_ckpt_path, f"model.rank{rank}.safetensors")

    state_dict = {}

    for file_path in tqdm(glob(path)):
        # memory_used()
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if "model.layers.61" in name:
                    continue
                param: torch.Tensor = f.get_tensor(name)
                if not skip_preprocess:
                    if name.startswith("model."):
                        name = name[len("model.") :]
                    name = name.replace("self_attn", "attn")
                    name = name.replace("mlp", "ffn")
                    name = name.replace("weight_scale_inv", "scale")
                    name = name.replace("weight_scale", "scale")
                    name = name.replace("e_score_correction_bias", "bias")
                    key = name.split(".")[-2]
                    mapping = {
                        "embed_tokens": ("embed", 0),
                        "input_layernorm": ("attn_norm", None),
                        "post_attention_layernorm": ("ffn_norm", None),
                        "q_proj": ("wq", 0),
                        "q_a_proj": ("wq_a", None),
                        "q_a_layernorm": ("q_norm", None),
                        "q_b_proj": ("wq_b", 0),
                        "kv_a_proj_with_mqa": ("wkv_a", None),
                        "kv_a_layernorm": ("kv_norm", None),
                        "kv_b_proj": ("wkv_b", 0),
                        "o_proj": ("wo", 1),
                        "gate": ("gate", None),
                        "gate_proj": ("w1", 0),
                        "down_proj": ("w2", 1),
                        "up_proj": ("w3", 0),
                        "norm": ("norm", None),
                        "lm_head": ("head", 0),
                        "scale": ("scale", None),
                    }
                    assert key in mapping, f"Key {key} not found in mapping"
                    new_key, dim = mapping[key]
                    name = name.replace(key, new_key)
                state_dict[name] = param
            # memory_used()

    return state_dict


def print_dict(d, name):
    print(name, d[name].shape, d[name].dtype)


def check_equal(d0, d1, name):
    print_dict(d0, name)
    print(d1[name])
    assert d0[name].shape == d1[name][0] and d0[name].dtype == d1[name][1]


def load_state_dict_llama_gguf_mlp_layers(llama_gguf_loader: GGUFLoader, layer_num=64):
    state_dict = {}

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"

    state_dict["embed_tokens.weight"] = llama_gguf_loader.load_gguf_tensor(
        "token_embd.weight", device, torch.bfloat16
    ).cpu()
    state_dict["lm_head.weight"] = llama_gguf_loader.load_gguf_tensor(
        "output.weight", device, torch.bfloat16
    ).cpu()
    state_dict["norm.weight"] = llama_gguf_loader.load_gguf_tensor(
        "output_norm.weight", device, torch.bfloat16
    ).cpu()

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
                gguf_name, device, torch.bfloat16
            ).cpu()

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

    state_dict["embed.weight"] = ds_gguf_loader.load_gguf_tensor(
        "token_embd.weight", device, torch.bfloat16
    ).cpu()
    state_dict["head.weight"] = ds_gguf_loader.load_gguf_tensor(
        "output.weight", device, torch.bfloat16
    ).cpu()
    state_dict["norm.weight"] = ds_gguf_loader.load_gguf_tensor(
        "output_norm.weight", device, torch.bfloat16
    ).cpu()

    translation_attn = {
        ".attn_norm.weight": ".attn_norm.weight",
        ".attn.kv_norm.weight": ".attn_kv_a_norm.weight",
        ".attn.wkv_a.weight": ".attn_kv_a_mqa.weight",
        ".attn.wkv_b.weight": ".attn_kv_b.weight",
        ".attn.wo.weight": ".attn_output.weight",
        ".attn.q_norm.weight": ".attn_q_a_norm.weight",
        ".attn.wq_a.weight": ".attn_q_a.weight",
        ".attn.wq_b.weight": ".attn_q_b.weight",
    }

    translation_mlp = {
        ".ffn.w2.weight": ".ffn_down.weight",
        ".ffn.w1.weight": ".ffn_gate.weight",
        ".ffn.w3.weight": ".ffn_up.weight",
        ".ffn_norm.weight": ".ffn_norm.weight",
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
                    gguf_name, device, torch.bfloat16
                ).cpu()

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
                    gguf_name, device, torch.bfloat16
                ).cpu()

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
    ds_gguf_loader = GGUFLoader("/data/nfs/DeepSeek-R1-Q4_K_M")
    translation_attn = {
        ".attn_norm.weight": ".attn_norm.weight",
        ".attn.kv_norm.weight": ".attn_kv_a_norm.weight",
        ".attn.wkv_a.weight": ".attn_kv_a_mqa.weight",
        ".attn.wkv_b.weight": ".attn_kv_b.weight",
        ".attn.wo.weight": ".attn_output.weight",
        ".attn.q_norm.weight": ".attn_q_a_norm.weight",
        ".attn.wq_a.weight": ".attn_q_a.weight",
        ".attn.wq_b.weight": ".attn_q_b.weight",
    }

    translation_mlp = {
        ".ffn.w2.weight": ".ffn_down.weight",
        ".ffn.w1.weight": ".ffn_gate.weight",
        ".ffn.w3.weight": ".ffn_up.weight",
        ".ffn_norm.weight": ".ffn_norm.weight",
    }

    translation_gate = {
        ".ffn.gate.bias": ".exp_probs_b.bias",
        ".ffn.gate.weight": ".ffn_gate_inp.weight",
        ".ffn_norm.weight": ".ffn_norm.weight",
    }

    translation_shared_experts = {
        ".ffn.shared_experts.w2.weight": ".ffn_down_shexp.weight",
        ".ffn.shared_experts.w1.weight": ".ffn_gate_shexp.weight",
        ".ffn.shared_experts.w3.weight": ".ffn_up_shexp.weight",
    }

    translation_shared_experts_cpu = {
        ".ffn.w2.weight": ".ffn_down_shexp.weight",
        ".ffn.w1.weight": ".ffn_gate_shexp.weight",
        ".ffn.w3.weight": ".ffn_up_shexp.weight",
    }

    translation_experts = {
        ".w2.weight": ".ffn_down_exps.weight",
        ".w1.weight": ".ffn_gate_exps.weight",
        ".w3.weight": ".ffn_up_exps.weight",
    }

    # cpu_layer = list(range(100))
    global_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    for layer_id in range(start_layer, end_layer):
        cpu_offload = layer_id in cpu_layers
        shared_cpu_offload = False
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
            # check_equal(state_dict, tensor_dict_st, safetensor_name)

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
            # check_equal(state_dict, tensor_dict_st, safetensor_name)

        if shared_cpu_offload and cpu_offload:
            gate_proj, gate_type = ds_gguf_loader.get_undequanted_tensor_and_ggml_type(
                f"blk.{layer_id}.ffn_gate_shexp.weight"
            )
            up_proj, up_type = ds_gguf_loader.get_undequanted_tensor_and_ggml_type(
                f"blk.{layer_id}.ffn_up_shexp.weight"
            )
            down_proj, down_type = ds_gguf_loader.get_undequanted_tensor_and_ggml_type(
                f"blk.{layer_id}.ffn_down_shexp.weight"
            )

            state_dict["layers." + str(layer_id) + ".ffn.shared_gate_proj"] = gate_proj
            state_dict["layers." + str(layer_id) + ".ffn.shared_up_proj"] = up_proj
            state_dict["layers." + str(layer_id) + ".ffn.shared_down_proj"] = down_proj
        elif not cpu_offload:
            for k in translation_shared_experts.keys():
                safetensor_name = "layers." + str(layer_id) + k
                gguf_name = "blk." + str(layer_id) + translation_shared_experts[k]
                if (
                    main_weight_dtype == "float8_e4m3fn"
                    and not safetensor_name.endswith("norm.weight")
                ):
                    safetensor_scale = safetensor_name[:-6] + "scale"
                    weight, scale = quant_fp8(
                        ds_gguf_loader.load_gguf_tensor(
                            gguf_name, device, torch.bfloat16
                        ),
                        block_size=128,
                    )
                    state_dict[safetensor_name] = weight.cpu()
                    state_dict[safetensor_scale] = scale.cpu()

                else:
                    state_dict[safetensor_name] = ds_gguf_loader.load_gguf_tensor(
                        gguf_name, device, torch.bfloat16
                    ).cpu()
                # check_equal(state_dict, tensor_dict_st, safetensor_name)
        else:
            for k in translation_shared_experts_cpu.keys():
                safetensor_name = "layers." + str(layer_id) + k
                gguf_name = "blk." + str(layer_id) + translation_shared_experts_cpu[k]
                if (
                    main_weight_dtype == "float8_e4m3fn"
                    and not safetensor_name.endswith("norm.weight")
                ):
                    safetensor_scale = safetensor_name[:-6] + "scale"
                    weight, scale = quant_fp8(
                        ds_gguf_loader.load_gguf_tensor(
                            gguf_name, device, torch.bfloat16
                        ),
                        block_size=128,
                    )
                    state_dict[safetensor_name] = weight.cpu()
                    state_dict[safetensor_scale] = scale.cpu()

                else:
                    state_dict[safetensor_name] = ds_gguf_loader.load_gguf_tensor(
                        gguf_name, device, torch.bfloat16
                    ).cpu()
                # check_equal(state_dict, tensor_dict_st, safetensor_name)

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
                        device,
                        torch.bfloat16,
                        global_rank,
                        world_size,
                    ).cpu()

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
                            + ".ffn.experts."
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
                                    gguf_name, device, torch.bfloat16
                                ).cpu()
                            )

            else:
                for k in translation_experts.keys():
                    gguf_name = "blk." + str(layer_id) + translation_experts[k]
                    expert_tensor = ds_gguf_loader.load_gguf_tensor(
                        gguf_name, device, torch.bfloat16
                    ).cpu()
                    safetensor_name = "layers." + str(layer_id) + k
                    for expert_id in range(256):
                        safetensor_name = (
                            "layers."
                            + str(layer_id)
                            + ".ffn.experts."
                            + str(expert_id)
                            + k
                        )
                        # state_dict[safetensor_name] = expert_tensor[expert_id]
                        # check_equal(state_dict, tensor_dict_st, safetensor_name)
                        if (
                            main_weight_dtype == "float8_e4m3fn"
                            and not safetensor_name.endswith("norm.weight")
                        ):
                            safetensor_scale = safetensor_name[:-6] + "scale"
                            weight, scale = quant_fp8(
                                expert_tensor[expert_id],
                                block_size=128,
                            )
                            state_dict[safetensor_name] = weight.cpu()
                            state_dict[safetensor_scale] = scale.cpu()

                        else:
                            state_dict[safetensor_name] = expert_tensor[expert_id].cpu()

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

                state_dict["layers." + str(layer_id) + ".ffn.gate_proj"] = gate_proj
                state_dict["layers." + str(layer_id) + ".ffn.up_proj"] = up_proj
                state_dict["layers." + str(layer_id) + ".ffn.down_proj"] = down_proj
                state_dict["layers." + str(layer_id) + ".ffn.gate_type"] = torch.tensor(
                    gate_type
                ).view(1)
                state_dict["layers." + str(layer_id) + ".ffn.up_type"] = torch.tensor(
                    up_type
                ).view(1)
                state_dict["layers." + str(layer_id) + ".ffn.down_type"] = torch.tensor(
                    down_type
                ).view(1)

    return state_dict
