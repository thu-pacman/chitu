from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)
import torch
import gc
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
from .model_llama import TransformerLlama
from .model_qwen import TransformerQwen
from .utils import load_pipe, load_tensor_parallel
import fairscale.nn.model_parallel.initialize as fs_init

from logging import getLogger

logger = getLogger(__name__)


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

        # Init model
        model = Backend.build_model(
            args.models,
            Backend.cache_manager,
            pipeline_parallel_size,
            model_parallel_size,
        )
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = model.to(torch.bfloat16)

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
                params = AutoModelForCausalLM.from_pretrained(
                    args.models.ckpt_dir, torch_dtype="auto", device_map="cpu"
                ).state_dict()

                def transform_key(key):
                    if key.startswith("model."):
                        return key[len("model.") :]
                    return key

                checkpoint = dict((transform_key(k), v) for k, v in params.items())
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
