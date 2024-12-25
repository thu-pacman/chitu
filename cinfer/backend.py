from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)
import torch
import gc
from enum import Enum
from .tokenizer import Tokenizer, ChatFormat, TokenizerHF, ChatFormatHF
from pathlib import Path
import os, sys, json, time
from transformers import AutoModelForCausalLM

from .cache_manager import (
    KVCacheManager,
    KVCacheManagerSkewAware,
    PagedKVCacheManager,
    KVCacheManagerNop,
)
from .attn_backend import FlashAttnBackend, RefAttnBackend
from .model_llama import TransformerLlama
from .model_qwen import TransformerQwen
from .utils import load_pipe, load_tensor_parallel
import fairscale.nn.model_parallel.initialize as fs_init

from logging import getLogger

logger = getLogger(__name__)


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
    parallel_type = ""
    state = BackendState.Running

    @staticmethod
    def build_model(args, cache, *extra_args, **extra_kwargs):
        if args.type == "qwen":
            return TransformerQwen(args, cache, *extra_args, **extra_kwargs)
        elif args.type == "llama":
            return TransformerLlama(args, cache, *extra_args, **extra_kwargs)
        else:
            assert False, f"Unknown model type {args.models.type}"

    @staticmethod
    def build(args):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if args.infer.parallel_type == "tensor":
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            pipeline_parallel_size = 1
        else:
            model_parallel_size = 1
            pipeline_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        if not model_parallel_is_initialized():
            initialize_model_parallel(model_parallel_size)

        Backend.parallel_type = args.infer.parallel_type

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)

        torch.manual_seed(args.infer.seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        # Init tokenizer
        if args.models.type == "qwen":
            tokenizer = TokenizerHF(path=args.models.tokenizer_path)
        else:
            tokenizer = Tokenizer(model_path=args.models.tokenizer_path)
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
        Backend.tokenizer = tokenizer
        if args.models.type == "qwen":
            Backend.formatter = ChatFormatHF(tokenizer)
        else:
            Backend.formatter = ChatFormat(tokenizer)

        # Init cache
        local_n_layers = args.models.n_layers // pipeline_parallel_size
        assert (
            args.models.n_layers % pipeline_parallel_size == 0
        ), f"n_layers {args.models.n_layers} not divisible by pipeline_parallel_size {pipeline_parallel_size}"
        model_parallel_size = fs_init.get_model_parallel_world_size()
        n_kv_heads = args.models.n_kv_heads
        n_local_kv_heads = n_kv_heads // model_parallel_size
        logger.warning(f"n local kv heads {n_local_kv_heads}")
        head_dim = args.models.dim // args.models.n_heads
        if args.infer.cache_type == "normal":
            Backend.cache_manager = KVCacheManager(
                local_n_layers, n_local_kv_heads, head_dim
            )
        elif args.infer.cache_type == "nop":
            Backend.cache_manager = KVCacheManagerNop(
                local_n_layers,
                n_local_kv_heads,
                head_dim,
                max_seq_len=args.infer.max_seq_len,
                num_hot_req=args.infer.max_reqs,
                device=local_rank,
            )
        elif args.infer.cache_type == "paged":
            Backend.cache_manager = PagedKVCacheManager(
                local_n_layers,
                n_local_kv_heads,
                head_dim,
                max_seq_len=args.infer.max_seq_len,
                num_hot_req=args.infer.max_reqs,
            )
        elif args.infer.cache_type == "skew":
            Backend.cache_manager = KVCacheManagerSkewAware(
                local_n_layers,
                n_local_kv_heads,
                head_dim,
                max_seq_len=args.infer.max_seq_len,
                num_hot_req=args.infer.max_reqs,
                device=local_rank,
            )
        else:
            assert False, f"Unknown cache type {args.infer.cache_type}"
        Backend.cache_type = args.infer.cache_type
        if args.infer.attn_type == "flash":
            attn_backend = FlashAttnBackend()
        elif args.infer.attn_type == "ref":
            attn_backend = RefAttnBackend()
        else:
            assert False, f"Unknown attn type {args.infer.attn_type}"

        # Init model
        model = Backend.build_model(
            args.models,
            Backend.cache_manager,
            pipeline_parallel_size,
            model_parallel_size,
            attn_backend,
        )
        if args.quant == "None":
            if args.dtype == "float16":
                model = model.to(torch.float16)
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
            elif args.dtype == "bfloat16":
                model = model.to(torch.bfloat16)
                torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
            else:
                raise NotImplementedError(f"Unsupported dtype {args.dtype}")
        if (
            (args.quant == "awq")
            or (args.quant == "llmint8")
            or (args.quant == "gptq")
            or (args.quant == "w8a16")
        ):
            from .quantize import quant

            torch.set_default_tensor_type(torch.cuda.HalfTensor)
            model = model.to(torch.float16)
        if args.quant == "awq":
            quant(model, method="awq", name="qwen")
        elif args.quant == "gptq":
            quant(model, method="gptq", name="qwen")
        elif args.quant == "w8a16":
            quant(model, method="w8a16", name="qwen")
        # print(model)

        # Init model parameters
        if args.infer.do_load:
            start_time = time.time()
            if args.models.type == "llama":
                checkpoints = sorted(Path(args.models.ckpt_dir).glob("*.pth"))
                assert (
                    len(checkpoints) > 0
                ), f"no checkpoint files found in {args.models.ckpt_dir}"
                ckpt_path = checkpoints[0]
                # logger.warning(f"Loading checkpoint from {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location="cpu")
            elif args.models.type == "qwen":
                if args.quant == "awq":
                    params = torch.load(args.quant_ckpt_dir, map_location="cpu")
                    replace_list = [
                        ("model.", ""),
                        ("embed_tokens.weight", "embed_tokens.tok_embeddings.weight"),
                    ]

                    def rep(s):
                        for p in replace_list:
                            s = s.replace(p[0], p[1], 1)
                        return s

                    checkpoint = dict((rep(k), v) for k, v in params.items())
                elif args.quant == "gptq":
                    params = AutoModelForCausalLM.from_pretrained(
                        args.quant_ckpt_dir, torch_dtype="auto", device_map="cpu"
                    ).state_dict()

                    def transform_key(key):
                        if key.startswith("model."):
                            return key[len("model.") :]
                        return key

                    checkpoint = dict((transform_key(k), v) for k, v in params.items())
                elif args.quant == "w8a16":
                    params = torch.load(
                        args.quant_ckpt_dir + "/pytorch_model.bin", map_location="cpu"
                    )
                    replace_list = [
                        ("model.", ""),
                        ("embed_tokens.weight", "embed_tokens.tok_embeddings.weight"),
                    ]
                    replace_list = [
                        ("model.", ""),
                    ]

                    def rep(s):
                        for p in replace_list:
                            s = s.replace(p[0], p[1], 1)
                        return s

                    checkpoint = dict((rep(k), v) for k, v in params.items())
                    # print(checkpoint.keys())
                else:
                    params = AutoModelForCausalLM.from_pretrained(
                        args.models.ckpt_dir, torch_dtype="auto", device_map="cpu"
                    ).state_dict()

                    def transform_key(key):
                        if key.startswith("model."):
                            return key[len("model.") :]
                        return key

                    checkpoint = dict((transform_key(k), v) for k, v in params.items())

                # Fuse q_proj, k_proj, v_proj into qkv_proj
                new_checkpoint = {}
                for k in checkpoint.keys():
                    if k.endswith("q_proj.weight"):
                        prefix = k[: -len("q_proj.weight")]
                        assert prefix + "k_proj.weight" in checkpoint
                        assert prefix + "v_proj.weight" in checkpoint
                        q_weight = checkpoint[prefix + "q_proj.weight"]
                        k_weight = checkpoint[prefix + "k_proj.weight"]
                        v_weight = checkpoint[prefix + "v_proj.weight"]

                        # The fused projected shape should be concatenated after the model parallel dimension.
                        # See model_qwen.py for details.

                        # q_weight, k_weight, v_weight are from ColumnParallelLinear, so their shape[0] are
                        # output_size
                        assert q_weight.shape[0] % model_parallel_size == 0
                        assert k_weight.shape[0] % model_parallel_size == 0
                        assert v_weight.shape[0] % model_parallel_size == 0
                        q_weight = q_weight.reshape(
                            model_parallel_size, -1, q_weight.shape[-1]
                        )
                        k_weight = k_weight.reshape(
                            model_parallel_size, -1, k_weight.shape[-1]
                        )
                        v_weight = v_weight.reshape(
                            model_parallel_size, -1, v_weight.shape[-1]
                        )
                        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=1)
                        qkv_weight = qkv_weight.reshape(-1, qkv_weight.shape[-1])
                        new_checkpoint[prefix + "qkv_proj.weight"] = qkv_weight
                    elif k.endswith("k_proj.weight") or k.endswith("v_proj.weight"):
                        continue
                    elif k.endswith("q_proj.bias"):
                        prefix = k[: -len("q_proj.bias")]
                        assert prefix + "k_proj.bias" in checkpoint
                        assert prefix + "v_proj.bias" in checkpoint
                        q_bias = checkpoint[prefix + "q_proj.bias"]
                        k_bias = checkpoint[prefix + "k_proj.bias"]
                        v_bias = checkpoint[prefix + "v_proj.bias"]

                        # The fused projected shape should be concatenated after the model parallel dimension.
                        # See model_qwen.py for details.
                        assert q_bias.shape[0] % model_parallel_size == 0
                        assert k_bias.shape[0] % model_parallel_size == 0
                        assert v_bias.shape[0] % model_parallel_size == 0
                        q_bias = q_bias.reshape(model_parallel_size, -1)
                        k_bias = k_bias.reshape(model_parallel_size, -1)
                        v_bias = v_bias.reshape(model_parallel_size, -1)
                        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=1)
                        qkv_bias = qkv_bias.reshape(-1)
                        new_checkpoint[prefix + "qkv_proj.bias"] = qkv_bias
                    elif k.endswith("k_proj.bias") or k.endswith("v_proj.bias"):
                        continue
                    else:
                        new_checkpoint[k] = checkpoint[k]
                checkpoint = new_checkpoint

            # logger.warning(checkpoint.keys())
            if pipeline_parallel_size > 1:
                load_pipe(
                    checkpoint,
                    model,
                    args.models.n_layers,
                    local_rank,
                    world_size,
                    args.models.type,
                )
            elif model_parallel_size > 1:
                load_tensor_parallel(
                    checkpoint,
                    model,
                    args.models.n_layers,
                    local_rank,
                    world_size,
                    args.models.type,
                )
            else:
                model.load_state_dict(checkpoint, strict=True)
            logger.warning(f"Loaded in {time.time() - start_time:.2f} seconds")

        if args.quant == "llmint8":
            quant(model, "llmint8", "qwen")
        model = model.to(local_rank)
        Backend.model = model

        Backend.args = args
        logger.warning(
            f"rank {local_rank} Backend initialized with CUDA mem at {torch.cuda.memory_allocated()}"
        )

    @staticmethod
    def stop():
        setattr(Backend, "model", None)
        setattr(Backend, "cache_manager", None)
        gc.collect()
        torch.cuda.empty_cache()
