import math
from dataclasses import dataclass
from typing import Optional, Tuple
from .global_vars import set_global_variables, get_timers
from .cache_manager import KVCacheManager, KVCacheManagerSkewAware

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
import cinfer_backend
import flash_attn

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from .tokenizer import Tokenizer, ChatFormat
from pathlib import Path
import os, sys, json, time

from xformers.ops import fmha


from logging import getLogger


logger = getLogger(__name__)


class Backend:
    model = None
    tokenizer = None
    formatter = None
    args = None

    @staticmethod
    def build(args):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        torch.manual_seed(args.seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        with open(Path(args.ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            **params,
        )
        tokenizer = Tokenizer(model_path=args.tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words
        model = Transformer(model_args)
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = model.to(torch.bfloat16)

        if args.do_load:
            start_time = time.time()
            checkpoints = sorted(Path(args.ckpt_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {args.ckpt_dir}"
            assert model_parallel_size == len(
                checkpoints
            ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
            ckpt_path = checkpoints[get_model_parallel_rank()]
            logger.warning(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
            logger.warning(f"Loaded in {time.time() - start_time:.2f} seconds")
        model = model.to(local_rank)

        Backend.model = model
        tokenizer.stop_tokens = torch.tensor(
            list(tokenizer.stop_tokens), device=local_rank
        )
        Backend.tokenizer = tokenizer
        Backend.formatter = ChatFormat(tokenizer)

        model_parallel_size = fs_init.get_model_parallel_world_size()
        n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        n_local_kv_heads = n_kv_heads // model_parallel_size
        head_dim = model_args.dim // model_args.n_heads

        if args.cache_type == "normal":
            Backend.cache_manager = KVCacheManager(
                model_args.n_layers, n_local_kv_heads, head_dim
            )
        else:
            Backend.cache_manager = KVCacheManagerSkewAware(
                model_args.n_layers,
                n_local_kv_heads,
                head_dim,
                max_seq_length=1024,
            )
        Backend.args = args
        logger.warning(f"Backend initialized with {torch.cuda.memory_allocated()}")


class VarLens:
    def __init__(self, tokens, device) -> None:
        self.lens = torch.tensor(
            [len(t) for t in tokens], device=device, dtype=torch.int32
        )
        self.prefix_lens = [0]
        for t in tokens:
            self.prefix_lens.append(self.prefix_lens[-1] + len(t))
        self.prefix_lens = torch.tensor(
            self.prefix_lens, device=device, dtype=torch.int32
        )
        self.cpu_lens = [len(t) for t in tokens]
        self.max_len = int(torch.max(self.lens))
        self.total_len = int(torch.sum(self.lens))


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


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


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
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

    def prefill_forward(
        self,
        x: torch.Tensor,
        varlens,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bs_seq, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bs_seq, self.n_local_heads, self.head_dim)
        xk = xk.view(bs_seq, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs_seq, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)  # TODO
        Backend.cache_manager.tmp_store(xk, xv)
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
        return self.wo(output)

    def decode_forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        assert seqlen == 1, "decode_forward only supports single token decoding"
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(-1, self.n_local_heads, self.head_dim)
        xk = xk.view(-1, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(-1, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        cache = Backend.cache_manager.update_prepare_cache(xk, xv)
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

        return self.wo(output)

    def forward(self, x, start_pos, freqs_cis, mask, varlens=None):
        if start_pos == 0:
            return self.prefill_forward(x, varlens, start_pos, freqs_cis, mask)
        else:
            return self.decode_forward(x, freqs_cis)


def GEMV(x, w):
    # w = w.transpose(1, 0).contiguous()
    output = torch.zeros(x.shape[:-1] + w.shape[-1:], device=x.device, dtype=x.dtype)
    cinfer_backend.gemv(x, w, output)
    return output


class FeedForward(nn.Module):
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
        # if x.dim() != 3:
        #     return self.w2(F.silu(self.w1(x)) * self.w3(x))

        # def prune_zeros(x):
        #     if x.dim() == 3:
        #         abs_x = torch.abs(x)
        #         med_value, med_indices = torch.median(abs_x, dim=-1)
        #         # logger.warning(f"med {med_value} mean {torch.mean(abs_x)}")
        #         x[abs_x < med_value.item()] = 0
        #     return x

        # # import datetime
        # # import os, time

        # # current_time = datetime.datetime.now()
        # # p = current_time.strftime("dumps/%Y-%m-%d_%H-%M-%S")
        # # p = p + f"_{time.time()}"
        # # os.makedirs(p, exist_ok=False)

        # # x = prune_zeros(x)
        # o0 = self.w1(x)
        # o1 = F.silu(o0)
        # o2 = self.w3(x)
        # o3 = o1 * o2
        # o3 = prune_zeros(o3)
        # # torch.save(x, p + "/x.pth")
        # # torch.save(o3, p + "/o3.pth")
        # # torch.save(self.w1.weight.data, p + "/w1.pth")
        # # torch.save(self.w3.weight.data, p + "/w3.pth")
        # # torch.save(self.w2.weight.data, p + "/w2.pth")
        # return self.w2(o3)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.timers = get_timers()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        varlens=None,
    ):
        h = self.attention(self.attention_norm(x), start_pos, freqs_cis, mask, varlens)
        h += x
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        ).cuda()

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output

    def prepare_freqs_cis_prefill(self, varlens):
        prepared_freqs_cis = torch.empty(
            [varlens.total_len, self.freqs_cis.shape[1]],
            device="cuda",
            dtype=torch.complex64,
        )
        start = 0
        for length in varlens.cpu_lens:
            prepared_freqs_cis[start : start + length] = self.freqs_cis[:length]
            start += length
        return prepared_freqs_cis

    def prepare_freqs_cis_decode(self, seq_lens):
        prepared_freqs_cis = torch.empty(
            [len(seq_lens), self.freqs_cis.shape[1]],
            device="cuda",
            dtype=torch.complex64,
        )
        for i, seq_len in enumerate(seq_lens):
            prepared_freqs_cis[i] = self.freqs_cis[seq_len]
        return prepared_freqs_cis

    @torch.inference_mode()
    def prefill(self, tokens: list[int]):
        varlens = VarLens(tokens, "cuda")
        tokens = torch.from_numpy(np.concatenate(tokens)).to("cuda")
        freqs_cis = self.prepare_freqs_cis_prefill(varlens)
        h = self.tok_embeddings(tokens)
        for it, layer in enumerate(self.layers):
            h = layer(h, 0, freqs_cis, None, varlens)
        h = self.norm(h)
        output = self.output(h).float()
        return output

    @torch.inference_mode()
    def decode(self, tokens, seq_lens):
        # generate different freqs_cis for each request, [num_req, other_freq_dim]
        freqs_cis = self.prepare_freqs_cis_decode(seq_lens)
        h = self.tok_embeddings(tokens)
        for it, layer in enumerate(self.layers):
            h = layer(h, 1, freqs_cis, None)
        h = self.norm(h)
        output = self.output(h).float()
        return output
