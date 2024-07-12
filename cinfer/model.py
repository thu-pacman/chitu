import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch.distributed
from .global_vars import set_global_variables, get_timers
from .cache_manager import KVCacheManager, KVCacheManagerSkewAware, KVCacheManagerNop

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn
import numpy as np

# from vllm import _custom_ops as vllm_ops
# import cinfer_backend
import flash_attn

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from .tokenizer import Tokenizer, ChatFormat, TokenizerHF, ChatFormatHF
from pathlib import Path
import os, sys, json, time

from xformers.ops import fmha


from logging import getLogger


logger = getLogger(__name__)


class OngoingRequests:
    def __init__(self, reqs, tasks, handle, logits):
        self.reqs = reqs
        self.tasks = tasks
        self.handle = handle
        self.logits = logits


class Backend:
    model = None
    tokenizer = None
    formatter = None
    args = None
    curr_varlens = None
    curr_req_ids = None
    ongoing_reqs = []
    parallel_type = ""

    @staticmethod
    def build(args):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if args.infer.parallel_type == "tensor":
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            else:
                model_parallel_size = 1
            initialize_model_parallel(model_parallel_size)
        Backend.parallel_type = args.infer.parallel_type 

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)

        torch.manual_seed(args.infer.seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        if args.models.type == "qwen":
            tokenizer = TokenizerHF(path=args.models.tokenizer_path)
        else:
            tokenizer = Tokenizer(model_path=args.models.tokenizer_path)
        # assert (
        #     args.models.vocab_size == tokenizer.n_words
        # ), f"{args.models.vocab_size} vs. {tokenizer.n_words}"
        model = Transformer.build(args.models)
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = model.to(torch.bfloat16)

        if args.infer.do_load:
            start_time = time.time()
            checkpoints = sorted(Path(args.models.ckpt_dir).glob("model1.pth"))
            assert (
                len(checkpoints) > 0
            ), f"no checkpoint files found in {args.models.ckpt_dir}"
            # assert model_parallel_size == len(
            #     checkpoints
            # ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
            # ckpt_path = checkpoints[get_model_parallel_rank()]
            ckpt_path = checkpoints[0]
            logger.warning(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            logger.warning(checkpoint.keys())
            from .utils import load

            if world_size > 1 and args.infer.parallel_type == "pipe":
                load(
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

        model_parallel_size = fs_init.get_model_parallel_world_size()
        n_kv_heads = args.models.n_kv_heads
        n_local_kv_heads = n_kv_heads // model_parallel_size
        logger.warning(f"n local kv heads {n_local_kv_heads}")
        head_dim = args.models.dim // args.models.n_heads

        if args.infer.cache_type == "normal":
            Backend.cache_manager = KVCacheManager(
                model.n_layers, n_local_kv_heads, head_dim
            )
        elif args.infer.cache_type == "nop":
            Backend.cache_manager = KVCacheManagerNop(
                model.n_layers,
                n_local_kv_heads,
                head_dim,
                max_seq_len=args.infer.max_seq_len,
                num_hot_req=args.infer.max_reqs,
                device=local_rank
            )
        else:
            Backend.cache_manager = KVCacheManagerSkewAware(
                model.n_layers,
                n_local_kv_heads,
                head_dim,
                max_seq_len=args.infer.max_seq_len,
                num_hot_req=args.infer.max_reqs,
                device=local_rank
            )
        Backend.args = args
        logger.warning(f"rank {local_rank} Backend initialized with {torch.cuda.memory_allocated()}")


class VarLens:
    def __init__(self, tokens, device) -> None:
        self.lens = torch.tensor(
            [len(t) for t in tokens], device=device, dtype=torch.int32
        )
        self.cpu_prefix_lens = [0]
        for t in tokens:
            self.cpu_prefix_lens.append(self.cpu_prefix_lens[-1] + len(t))
        self.prefix_lens = torch.tensor(
            self.cpu_prefix_lens, device=device, dtype=torch.int32
        )
        self.cpu_lens = [len(t) for t in tokens]
        self.max_len = int(torch.max(self.lens))
        self.total_len = int(torch.sum(self.lens))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if ndim == 4:
        assert freqs_cis.shape == (
            x.shape[1],
            x.shape[-1],
        ), f"{freqs_cis.shape} {x.shape}"
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    elif ndim == 3:
        assert freqs_cis.shape == (
            x.shape[0],
            x.shape[-1],
        ), f"{freqs_cis.shape} {x.shape}"
        shape = [d if i == 0 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    else:
        assert False
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(xq.dim() - 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(xk.dim() - 1)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def _run_linear(self, x):
        raise NotImplementedError

    def _run_output_linear(self, x):
        raise NotImplementedError

    def prefill_forward(
        self,
        x: torch.Tensor,
        varlens,
        freqs_cis: torch.Tensor,
    ):
        bs_seq, _ = x.shape
        xq, xk, xv = self._run_linear(x)
        xq = xq.view(bs_seq, self.n_local_heads, self.head_dim)
        xk = xk.view(bs_seq, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs_seq, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        Backend.cache_manager.finalize_cache_bylayer_prefill(
            xk,
            xv,
            Backend.curr_req_ids,
            Backend.curr_varlens,
        )
        output = flash_attn.flash_attn_varlen_func(
            xq,
            xk,
            xv,
            varlens.prefix_lens,
            varlens.prefix_lens,
            varlens.max_len,
            varlens.max_len,
            causal=True,
        ).view(bs_seq, -1)
        return self._run_output_linear(output)

    def decode_forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        assert seqlen == 1, "decode_forward only supports single token decoding"
        xq, xk, xv = self._run_linear(x)

        xq = xq.view(-1, self.n_local_heads, self.head_dim)
        xk = xk.view(-1, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(-1, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        cache = Backend.cache_manager.update_cache_decode(xk, xv)
        cache_k = cache[0]
        cache_v = cache[1]
        max_seq_len = cache.shape[2]

        if self.n_local_heads != self.n_local_kv_heads:
            group_size = self.n_local_heads // self.n_local_kv_heads
            assert group_size > 1
            xq = xq.view(bsz, seqlen, self.n_local_kv_heads, group_size, self.head_dim)
            cache_k = cache_k.view(
                bsz, max_seq_len, self.n_local_kv_heads, 1, self.head_dim
            ).expand(bsz, max_seq_len, self.n_local_kv_heads, group_size, self.head_dim)
            cache_v = cache_v.view(
                bsz, max_seq_len, self.n_local_kv_heads, 1, self.head_dim
            ).expand(bsz, max_seq_len, self.n_local_kv_heads, group_size, self.head_dim)
        output = fmha.memory_efficient_attention_forward(xq, cache_k, cache_v).view(
            bsz, seqlen, -1
        )
        output = xq.view(bsz, seqlen, -1)
        return self._run_output_linear(output)

    def forward(self, x, freqs_cis, varlens=None):
        if varlens is not None:  # prefill
            return self.prefill_forward(x, varlens, freqs_cis)
        else:  # decode
            return self.decode_forward(x, freqs_cis)


def GEMV(x, w):
    # w = w.transpose(1, 0).contiguous()
    output = torch.zeros(x.shape[:-1] + w.shape[-1:], device=x.device, dtype=x.dtype)
    cinfer_backend.gemv(x, w, output)
    return output


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):

    def __init__(self, layer_id: int, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.layer_id = layer_id
        self.timers = get_timers()

    def forward(self):
        raise NotImplementedError


class Transformer(nn.Module):
    @staticmethod
    def build(args):
        if args.type == "qwen":
            return TransformerQwen(args)
        elif args.type == "llama":
            return TransformerLlama(args)
        else:
            assert False, f"Unknown model type {args.models.type}"

    def __init__(self, params):
        super().__init__()
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.device = torch.device(self.rank)

        # self.pipeline_exec = True # self.world_size > 1
        # self.tensor_exec = False
        self.pipeline_exec = False
        self.tensor_exec = True

        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        if self.pipeline_exec:
            self.n_layers = self.n_layers // self.world_size

        if not self.pipeline_exec or self.rank == 0:
            self._init_pre_layers()
        self._init_layers()
        if not self.pipeline_exec or self.rank == self.world_size - 1:
            self._init_post_layers()

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        ).cuda()

    def _init_pre_layers(self):
        raise NotImplementedError

    def _init_layers(self):
        raise NotImplementedError

    def _init_post_layers(self):
        raise NotImplementedError

    def _pre_layers(self, h):
        raise NotImplementedError

    def _post_layers(self, h):
        raise NotImplementedError

    def prepare_freqs_cis_prefill(self, varlens, device):
        prepared_freqs_cis = torch.empty(
            [varlens.total_len, self.freqs_cis.shape[1]],
            device=device,
            dtype=torch.complex64,
        )
        start = 0
        for length in varlens.cpu_lens:
            prepared_freqs_cis[start : start + length] = self.freqs_cis[:length]
            start += length
        return prepared_freqs_cis

    def prepare_freqs_cis_decode(self, seq_lens, device):
        prepared_freqs_cis = torch.empty(
            [len(seq_lens), self.freqs_cis.shape[1]],
            device=device,
            dtype=torch.complex64,
        )
        for i, seq_len in enumerate(seq_lens):
            prepared_freqs_cis[i] = self.freqs_cis[seq_len]
        return prepared_freqs_cis

    @torch.inference_mode()
    def prefill_single_device(self, tokens: list[int]):
        varlens = VarLens(tokens, self.device)
        tokens = torch.from_numpy(np.concatenate(tokens)).to(self.device)
        freqs_cis = self.prepare_freqs_cis_prefill(varlens, self.device)
        h = self._pre_layers(tokens)
        for it, layer in enumerate(self.layers):
            h = layer(h, freqs_cis, varlens)
        h = self._post_layers(h)
        tmp = varlens.cpu_prefix_lens[1:]
        h = h[[item - 1 for item in tmp]]
        h = h.float()
        return h

    @torch.inference_mode()
    def decode_single_device(self, tokens, seq_lens):
        # generate different freqs_cis for each request, [num_req, other_freq_dim]
        freqs_cis = self.prepare_freqs_cis_decode(seq_lens, self.device)
        h = self._pre_layers(tokens)
        for it, layer in enumerate(self.layers):
            h = layer(h, freqs_cis)
        h = self._post_layers(h)
        h = h.float()
        return h

    @torch.inference_mode()
    def prefill_pipeline(self, tokens):
        varlens = Backend.curr_varlens
        freqs_cis = self.prepare_freqs_cis_prefill(varlens, self.device)

        # start of model
        if self.rank == 0:
            tokens = torch.from_numpy(np.concatenate(tokens)).to(self.device)
            h = self._pre_layers(tokens)
        else:
            h = tokens
        # layers
        for it, layer in enumerate(self.layers):
            h = layer(h, freqs_cis, varlens)
        # end of model
        if self.rank == self.world_size - 1:
            h = self._post_layers(h)
            tmp = varlens.cpu_prefix_lens[1:]
            h = h[[item - 1 for item in tmp]]
            h = h.float()
        return h

    @torch.inference_mode()
    def decode_pipeline(self, tokens, seq_lens):
        # generate different freqs_cis for each request, [num_req, other_freq_dim]
        freqs_cis = self.prepare_freqs_cis_decode(seq_lens, self.device)
        if self.rank == 0:
            h = self._pre_layers(tokens)
        else:
            h = tokens
        for it, layer in enumerate(self.layers):
            h = layer(h, freqs_cis)
        if self.rank == self.world_size - 1:
            h = self._post_layers(h)
            h = h.float()
        return h

    @torch.inference_mode()
    def prefill(self, tokens):
        if self.pipeline_exec:
            return self.prefill_pipeline(tokens)
        else:
            return self.prefill_single_device(tokens)

    @torch.inference_mode()
    def decode(self, tokens, seq_lens):
        if self.pipeline_exec:
            return self.decode_pipeline(tokens, seq_lens)
        else:
            return self.decode_single_device(tokens, seq_lens)


class AttentionQwen(Attention):
    def __init__(self, args):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # self.q_proj = ColumnParallelLinear(
        #     args.dim,
        #     self.n_local_heads * self.head_dim,
        #     bias=True,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.k_proj = ColumnParallelLinear(
        #     args.dim,
        #     self.n_local_kv_heads * self.head_dim,
        #     bias=True,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.v_proj = ColumnParallelLinear(
        #     args.dim,
        #     self.n_local_kv_heads * self.head_dim,
        #     bias=True,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        self.qkv_proj = ColumnParallelLinear(
            args.dim,
            (args.n_heads + args.n_kv_heads + args.n_kv_heads) * self.head_dim,
            bias=True,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.o_proj = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

    def _run_linear(self, x):
        # return self.q_proj(x), self.k_proj(x), self.v_proj(x)
        qkv = self.qkv_proj(x)
        if len(qkv.shape) == 3:
            xq = qkv[:,:,:self.n_local_heads * self.head_dim]
            xk = qkv[:,:,self.n_local_heads * self.head_dim:(self.n_local_heads + self.n_local_kv_heads) * self.head_dim]
            xv = qkv[:,:,(self.n_local_heads + self.n_local_kv_heads) * self.head_dim : ]
            return xq, xk, xv
        else:
            xq = qkv[:,:self.n_local_heads * self.head_dim]
            xk = qkv[:,self.n_local_heads * self.head_dim:(self.n_local_heads + self.n_local_kv_heads) * self.head_dim]
            xv = qkv[:,(self.n_local_heads + self.n_local_kv_heads) * self.head_dim : ]
            return xq, xk, xv

    def _run_output_linear(self, x):
        return self.o_proj(x)


class FeedForwardQwen(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.down_proj = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.up_proj = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlockQwen(TransformerBlock):
    def __init__(self, layer_id: int, args):
        super().__init__(layer_id, args)
        self.self_attn = AttentionQwen(args)
        self.mlp = FeedForwardQwen(dim=args.dim, hidden_dim=args.intermediate_dim)
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, varlens=None):
        # h = self.input_layernorm(x)
        h = self.self_attn(self.input_layernorm(x), freqs_cis, varlens)
        h += x
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class TransformerQwen(Transformer):
    def __init__(self, params):
        super().__init__(params)

    def _init_pre_layers(self):
        self.embed_tokens = VocabParallelEmbedding(
            self.params.vocab_size, self.params.dim, init_method=lambda x: x
        )

    def _init_layers(self):
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlockQwen(layer_id, self.params))

    def _init_post_layers(self):
        self.norm = RMSNorm(self.params.dim, eps=self.params.norm_eps)
        self.lm_head = ColumnParallelLinear(
            self.params.dim, self.params.vocab_size, bias=False, init_method=lambda x: x
        )

    def _pre_layers(self, h):
        return self.embed_tokens(h)

    def _post_layers(self, h):
        h = self.norm(h)
        h = self.lm_head(h)
        return h


class AttentionLlama(Attention):
    def __init__(self, args):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

    def _run_linear(self, x):
        return self.wq(x), self.wk(x), self.wv(x)

    def _run_output_linear(self, x):
        return self.wo(x)


class TransformerLlama(Transformer):
    def __init__(self, params):
        super().__init__(params)

    def _init_pre_layers(self):
        self.tok_embeddings = VocabParallelEmbedding(
            self.params.vocab_size, self.params.dim, init_method=lambda x: x
        )

    def _init_layers(self):
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlockLlama(layer_id, self.params))

    def _init_post_layers(self):
        self.norm = RMSNorm(self.params.dim, eps=self.params.norm_eps)
        self.output = ColumnParallelLinear(
            self.params.dim, self.params.vocab_size, bias=False, init_method=lambda x: x
        )

    def _pre_layers(self, h):
        return self.tok_embeddings(h)

    def _post_layers(self, h):
        h = self.norm(h)
        h = self.output(h)
        return h


class FeedForwardLlama(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlockLlama(TransformerBlock):
    def __init__(self, layer_id: int, args):
        super().__init__(layer_id, args)
        self.attention = AttentionLlama(args)
        self.feed_forward = FeedForwardLlama(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, varlens=None):
        h = self.attention(self.attention_norm(x), freqs_cis, varlens)
        h += x
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
