import torch
import gc
import itertools
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
from .tensor_parallel import init_tp, get_tp_size
from .attn_backend import FlashAttnBackend, RefAttnBackend
from .model_llama import TransformerLlama
from .model_hf_llama import TransformerHFLlama
from .model_hf_mixtral import TransformerHFMixtral
from .utils import compute_layer_dist_in_pipe

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
    state = BackendState.Running
    pp_stage = None
    pp_end_stage = None
    pp_main_rank = None

    @staticmethod
    def build_model(args, cache, *extra_args, **extra_kwargs):
        if args.type == "hf-llama":
            if args.name.startswith("glm4"):
                extra_kwargs["rotary_type"] = "glm4"
            return TransformerHFLlama(args, cache, *extra_args, **extra_kwargs)
        elif args.type == "hf-mixtral":
            return TransformerHFMixtral(args, cache, *extra_args, **extra_kwargs)
        elif args.type == "llama":
            return TransformerLlama(args, cache, *extra_args, **extra_kwargs)
        else:
            assert False, f"Unknown model type {args.models.type}"

    @staticmethod
    def build(args):
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

        torch.manual_seed(args.infer.seed)

        if global_rank > 0:
            sys.stdout = open(os.devnull, "w")

        trust_remote_code = False
        if args.models.name.startswith("glm4"):
            trust_remote_code = True  # Blame the glm4 folks for this

        # Init tokenizer
        force_full_seq_decode = (
            args.models.tokenizer_force_full_seq_decode
            if hasattr(args.models, "tokenizer_force_full_seq_decode")
            else False
        )
        if args.models.type == "hf-llama" or args.models.type == "hf-mixtral":
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
        Backend.tokenizer = tokenizer
        if args.models.type == "hf-llama" or args.models.type == "hf-mixtral":
            Backend.formatter = ChatFormatHF(tokenizer)
        else:
            Backend.formatter = ChatFormat(tokenizer)

        # Init cache
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

        n_kv_heads = args.models.n_kv_heads
        n_local_kv_heads = n_kv_heads // model_parallel_size
        head_dim = args.models.dim // args.models.n_heads
        if args.infer.cache_type == "normal":
            Backend.cache_manager = KVCacheManager(
                local_begin_layer_id,
                local_end_layer_id,
                n_local_kv_heads=n_local_kv_heads,
                head_dim=head_dim,
            )
        elif args.infer.cache_type == "nop":
            Backend.cache_manager = KVCacheManagerNop(
                local_begin_layer_id,
                local_end_layer_id,
                n_local_kv_heads=n_local_kv_heads,
                head_dim=head_dim,
                max_seq_len=args.infer.max_seq_len,
                num_hot_req=args.infer.max_reqs,
                device=local_rank,
            )
        elif args.infer.cache_type == "paged":
            Backend.cache_manager = PagedKVCacheManager(
                local_begin_layer_id,
                local_end_layer_id,
                n_local_kv_heads=n_local_kv_heads,
                head_dim=head_dim,
                max_seq_len=args.infer.max_seq_len,
                num_hot_req=args.infer.max_reqs,
                device=local_rank,
            )
        elif args.infer.cache_type == "skew":
            Backend.cache_manager = KVCacheManagerSkewAware(
                local_begin_layer_id,
                local_end_layer_id,
                n_local_kv_heads=n_local_kv_heads,
                head_dim=head_dim,
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
        merge_qkv_gate_up = True
        if args.models.type == "llama":
            merge_qkv_gate_up = False  # Not yet supported
        if args.quant != "None":
            # Merge weights for quantized models is non-trivial, because we can
            # only merge weights but NOT the scales on input dimensions, and this
            # will break the assumption of the fused quantized kernels.
            merge_qkv_gate_up = False
        model = Backend.build_model(
            args.models,
            Backend.cache_manager,
            pipeline_parallel_size,
            model_parallel_size,
            attn_backend=attn_backend,
            op_impl=args.infer.op_impl,
            merge_qkv_gate_up=merge_qkv_gate_up,
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
            quant(model, method="awq", name="hf-llama")
        elif args.quant == "gptq":
            quant(model, method="gptq", name="hf-llama")
        elif args.quant == "w8a16":
            quant(model, method="w8a16", name="hf-llama")
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
            elif args.models.type == "hf-llama" or args.models.type == "hf-mixtral":
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
                        args.quant_ckpt_dir,
                        torch_dtype="auto",
                        device_map="cpu",
                        trust_remote_code=trust_remote_code,
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

            model.load_state_dict(checkpoint, strict=True)
            logger.warning(f"Loaded in {time.time() - start_time:.2f} seconds")

        if args.quant == "llmint8":
            quant(model, "llmint8", "hf-llama")
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
