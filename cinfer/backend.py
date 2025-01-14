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
from .model_hf_llama import TransformerHFLlama
from .model_hf_mixtral import TransformerHFMixtral
from .utils import (
    compute_layer_dist_in_pipe,
    load_pipe,
    load_tensor_parallel,
    merge_column_parallel_weights,
    merge_column_parallel_biases,
)
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
        local_n_layers = (
            compute_layer_dist_in_pipe(args.models.n_layers, pipeline_parallel_size)[
                local_rank
            ]
            if args.infer.parallel_type == "pipe"
            else args.models.n_layers
        )
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

                if args.models.name.startswith("glm4"):
                    # glm4 has non-standard key names because they use "custom code" in model files instead of
                    # using code in transformers' repo.

                    def map_glm4_key(k):
                        k = k.replace(
                            "transformer.embedding.word_embeddings.", "embed_tokens."
                        )
                        k = k.replace("transformer.encoder.layers.", "layers.")
                        k = k.replace(".self_attention.", ".self_attn.")
                        k = k.replace(".query_key_value.", ".qkv_proj.")
                        k = k.replace(".dense.", ".o_proj.")
                        k = k.replace(".dense_h_to_4h.", ".gate_up_proj.")
                        k = k.replace(".dense_4h_to_h.", ".down_proj.")
                        k = k.replace("transformer.encoder.final_layernorm.", "norm.")
                        k = k.replace("transformer.output_layer.", "lm_head.")
                        return k

                    del checkpoint["transformer.rotary_pos_emb.inv_freq"]
                    checkpoint = {map_glm4_key(k): v for k, v in checkpoint.items()}

                if args.models.name.startswith("mixtral"):

                    def map_mixtral_key(k):
                        k = k.replace(".block_sparse_moe.", ".mlp.")
                        k = k.replace(".w1.", ".gate_proj.")
                        k = k.replace(".w3.", ".up_proj.")
                        k = k.replace(".w2.", ".down_proj.")
                        return k

                    checkpoint = {map_mixtral_key(k): v for k, v in checkpoint.items()}

                # Fuse q_proj, k_proj, v_proj into qkv_proj
                new_checkpoint = {}
                for k in checkpoint.keys():
                    if k.endswith(".qkv_proj.weight"):
                        # Already fused, but need to transpose for model parallel
                        qkv_weight = checkpoint[k]
                        if model_parallel_size > 1:
                            n_heads = args.models.n_heads
                            n_kv_heads = (
                                args.models.n_heads
                                if args.models.n_kv_heads is None
                                else args.models.n_kv_heads
                            )
                            head_dim = args.models.dim // n_heads
                            q_weight, k_weight, v_weight = qkv_weight.split(
                                [
                                    n_heads * head_dim,
                                    n_kv_heads * head_dim,
                                    n_kv_heads * head_dim,
                                ],
                                dim=0,
                            )
                            qkv_weight = merge_column_parallel_weights(
                                [q_weight, k_weight, v_weight], model_parallel_size
                            )
                        new_checkpoint[k] = qkv_weight
                    elif k.endswith(".q_proj.weight"):
                        prefix = k[: -len("q_proj.weight")]
                        assert prefix + "k_proj.weight" in checkpoint
                        assert prefix + "v_proj.weight" in checkpoint
                        assert prefix + "qkv_proj.weight" not in checkpoint
                        q_weight = checkpoint[prefix + "q_proj.weight"]
                        k_weight = checkpoint[prefix + "k_proj.weight"]
                        v_weight = checkpoint[prefix + "v_proj.weight"]
                        new_checkpoint[prefix + "qkv_proj.weight"] = (
                            merge_column_parallel_weights(
                                [q_weight, k_weight, v_weight], model_parallel_size
                            )
                        )
                    elif k.endswith(".k_proj.weight") or k.endswith(".v_proj.weight"):
                        continue
                    elif k.endswith(".qkv_proj.bias"):
                        # Already fused, but need to transpose for model parallel
                        qkv_bias = checkpoint[k]
                        if model_parallel_size > 1:
                            n_heads = args.models.n_heads
                            n_kv_heads = (
                                args.models.n_heads
                                if args.models.n_kv_heads is None
                                else args.models.n_kv_heads
                            )
                            head_dim = args.models.dim // n_heads
                            q_bias, k_bias, v_bias = qkv_bias.split(
                                [
                                    n_heads * head_dim,
                                    n_kv_heads * head_dim,
                                    n_kv_heads * head_dim,
                                ],
                                dim=0,
                            )
                            qkv_bias = merge_column_parallel_biases(
                                [q_bias, k_bias, v_bias], model_parallel_size
                            )
                        new_checkpoint[k] = qkv_bias
                    elif k.endswith(".q_proj.bias"):
                        prefix = k[: -len("q_proj.bias")]
                        assert prefix + "k_proj.bias" in checkpoint
                        assert prefix + "v_proj.bias" in checkpoint
                        assert prefix + "qkv_proj.bias" not in checkpoint
                        q_bias = checkpoint[prefix + "q_proj.bias"]
                        k_bias = checkpoint[prefix + "k_proj.bias"]
                        v_bias = checkpoint[prefix + "v_proj.bias"]
                        new_checkpoint[prefix + "qkv_proj.bias"] = (
                            merge_column_parallel_biases(
                                [q_bias, k_bias, v_bias], model_parallel_size
                            )
                        )
                    elif k.endswith(".k_proj.bias") or k.endswith(".v_proj.bias"):
                        continue
                    else:
                        new_checkpoint[k] = checkpoint[k]
                checkpoint = new_checkpoint

                # Fuse gate_proj and up_proj into gate_up_proj
                new_checkpoint = {}
                for k in checkpoint.keys():
                    if k.endswith(".gate_up_proj.weight"):
                        # Already fused, but need to transpose for model parallel
                        gate_up_weight = checkpoint[k]
                        if model_parallel_size > 1:
                            gate_weight, up_weight = torch.chunk(
                                gate_up_weight, 2, dim=0
                            )
                            gate_up_weight = merge_column_parallel_weights(
                                [gate_weight, up_weight], model_parallel_size
                            )
                        new_checkpoint[k] = gate_up_weight
                    elif k.endswith(".gate_proj.weight"):
                        prefix = k[: -len("gate_proj.weight")]
                        assert prefix + "up_proj.weight" in checkpoint
                        assert prefix + "gate_up_proj.weight" not in checkpoint
                        gate_weight = checkpoint[prefix + "gate_proj.weight"]
                        up_weight = checkpoint[prefix + "up_proj.weight"]

                        # The fused projected shape should be concatenated after the model parallel dimension.
                        # See model_hf_llama.py for details.
                        assert gate_weight.shape[0] % model_parallel_size == 0
                        assert up_weight.shape[0] % model_parallel_size == 0
                        gate_weight = gate_weight.reshape(
                            model_parallel_size, -1, gate_weight.shape[-1]
                        )
                        up_weight = up_weight.reshape(
                            model_parallel_size, -1, up_weight.shape[-1]
                        )
                        gate_up_weight = torch.cat([gate_weight, up_weight], dim=1)
                        gate_up_weight = gate_up_weight.reshape(
                            -1, gate_up_weight.shape[-1]
                        )
                        new_checkpoint[prefix + "gate_up_proj.weight"] = gate_up_weight
                    elif k.endswith(".up_proj.weight"):
                        continue
                    elif k.endswith(".gate_up_proj.bias"):
                        # Already fused, but need to transpose for model parallel
                        gate_up_bias = checkpoint[k]
                        if model_parallel_size > 1:
                            gate_bias, up_bias = torch.chunk(gate_up_bias, 2, dim=0)
                            gate_up_bias = merge_column_parallel_biases(
                                [gate_bias, up_bias], model_parallel_size
                            )
                        new_checkpoint[k] = gate_up_bias
                    elif k.endswith(".gate_proj.bias"):
                        prefix = k[: -len("gate_proj.bias")]
                        assert prefix + "up_proj.bias" in checkpoint
                        assert prefix + "gate_up_proj.bias" not in checkpoint
                        gate_bias = checkpoint[prefix + "gate_proj.bias"]
                        up_bias = checkpoint[prefix + "up_proj.bias"]

                        # The fused projected shape should be concatenated after the model parallel dimension.
                        # See model_hf_llama.py for details.
                        assert gate_bias.shape[0] % model_parallel_size == 0
                        assert up_bias.shape[0] % model_parallel_size == 0
                        gate_bias = gate_bias.reshape(model_parallel_size, -1)
                        up_bias = up_bias.reshape(model_parallel_size, -1)
                        gate_up_bias = torch.cat([gate_bias, up_bias], dim=1)
                        gate_up_bias = gate_up_bias.reshape(-1)
                        new_checkpoint[prefix + "gate_up_proj.bias"] = gate_up_bias
                    elif k.endswith(".up_proj.bias"):
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
