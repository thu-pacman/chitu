import math
import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Mapping, List, Any

import torch.distributed
from .global_vars import set_global_variables, get_timers
from .utils import (
    VarLens,
    compute_layer_dist_in_pipe,
    is_layer,
)
from .cache_manager import PagedKVCacheManager


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

from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from .tokenizer import Tokenizer, ChatFormat, TokenizerHF, ChatFormatHF
from pathlib import Path
import os, sys, json, time

from logging import getLogger


logger = getLogger(__name__)


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
    def __init__(self, layer_id, cache, attn_backend):
        super().__init__()
        self.layer_id = layer_id
        self.cache = cache
        self.attn_backend = attn_backend

    def _run_linear(self, x):
        raise NotImplementedError

    def _run_output_linear(self, x):
        raise NotImplementedError

    def prefill_forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        varlens,
    ):
        bs_seq, _ = x.shape
        xq, xk, xv = self._run_linear(x)
        xq = xq.view(bs_seq, self.n_local_heads, self.head_dim)
        xk = xk.view(bs_seq, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs_seq, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        self.cache.finalize_cache_bylayer_prefill(
            xk, xv, self.cache.curr_req_ids, self.cache.curr_varlens, self.layer_id
        )
        output = self.attn_backend.attn_varlen_func(
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

    def decode_forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
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

        cache = self.cache.get_cache_decode(self.layer_id)
        cache_k = cache[0]
        cache_v = cache[1]
        cache_seqlens = self.cache.get_gpu_seq_lens()
        output = self.attn_backend.attn_with_kvcache(
            xq,
            cache_k,
            cache_v,
            xk,
            xv,
            cache_seqlens=cache_seqlens,
        ).view(bsz, seqlen, -1)
        return self._run_output_linear(output)

    def decode_forward_paged(self, x: torch.Tensor, freqs_cis: torch.Tensor):
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

        block_table = self.cache.get_gpu_block_table(self.layer_id)
        cache_seqlens = self.cache.get_gpu_seq_lens()
        paged_k_cache, paged_v_cache = self.cache.get_paged_kv_cache()
        output = self.attn_backend.attn_with_kvcache(
            xq,
            paged_k_cache,
            paged_v_cache,
            xk,
            xv,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
        ).view(bsz, seqlen, -1)
        return self._run_output_linear(output)

    def forward(self, x, freqs_cis, varlens=None):
        if varlens is not None:  # prefill
            return self.prefill_forward(x, freqs_cis, varlens)
        elif isinstance(self.cache, PagedKVCacheManager):
            return self.decode_forward_paged(x, freqs_cis)
        else:
            return self.decode_forward(x, freqs_cis)


# def GEMV(x, w):
#     # w = w.transpose(1, 0).contiguous()
#     output = torch.zeros(x.shape[:-1] + w.shape[-1:], device=x.device, dtype=x.dtype)
#     cinfer_backend.gemv(x, w, output)
#     return output


class TransformerBlock(nn.Module):

    def __init__(self, layer_id: int, args, cache, attn_backend, op_impl):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.layer_id = layer_id
        self.timers = get_timers()

    def forward(self):
        raise NotImplementedError


class Transformer(nn.Module):
    def __init__(
        self,
        params,
        cache,
        pipeline_parallel_size,
        model_parallel_size,
        attn_backend,
        op_impl,
    ):
        super().__init__()
        self.cache = cache
        self.attn_backend = attn_backend
        self.op_impl = op_impl
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.device = torch.device(self.rank)

        self.pipeline_parallel_size = pipeline_parallel_size
        self.model_parallel_size = model_parallel_size
        self.pipeline_exec = pipeline_parallel_size > 1
        self.tensor_exec = model_parallel_size > 1

        self.params = params
        self.vocab_size = params.vocab_size
        self.global_n_layers = params.n_layers

        if self.pipeline_exec:
            num_layers_of_each_rank = compute_layer_dist_in_pipe(
                self.global_n_layers, self.world_size
            )
            first_layer_id_of_each_rank = list(
                itertools.accumulate([0] + num_layers_of_each_rank)
            )
            self.local_begin_layer_id = first_layer_id_of_each_rank[self.rank]
            self.local_end_layer_id = first_layer_id_of_each_rank[self.rank + 1]
        else:
            self.local_begin_layer_id = 0
            self.local_end_layer_id = self.global_n_layers

        if not self.pipeline_exec or self.rank == 0:
            self._init_pre_layers()
        self._init_layers(cache, attn_backend=attn_backend, op_impl=op_impl)
        if not self.pipeline_exec or self.rank == self.world_size - 1:
            self._init_post_layers()

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        ).cuda()

    def _get_tensor_column_parallel_layer_names(self) -> List[str]:
        raise NotImplementedError

    def _get_tensor_row_parallel_layer_names(self) -> List[str]:
        raise NotImplementedError

    def _get_pre_layer_prefixes(self) -> List[str]:
        raise NotImplementedError

    def _get_post_layer_prefixes(self) -> List[str]:
        raise NotImplementedError

    def _get_layer_i_prefixes(self, i: int) -> List[str]:
        raise NotImplementedError

    def _chunk_checkpoint_for_pipeline_parallel(
        self,
        checkpoint,
        num_layers: int,
        rank: int,
        world_size: int,
    ):
        keys = checkpoint.keys()
        # logger.warning(f"Loading checkpoint {keys}")
        partial_checkpoint = {}

        num_layers_of_each_rank = compute_layer_dist_in_pipe(num_layers, world_size)
        first_layer_id_of_each_rank = list(
            itertools.accumulate([0] + num_layers_of_each_rank)
        )

        for i in range(
            first_layer_id_of_each_rank[rank], first_layer_id_of_each_rank[rank + 1]
        ):
            for key in keys:
                if i == 0:
                    for prefix in self._get_pre_layer_prefixes():
                        if key.startswith(prefix):
                            partial_checkpoint[key] = checkpoint[key]
                for prefix in self._get_layer_i_prefixes(i):
                    if key.startswith(prefix):
                        local_i = i - first_layer_id_of_each_rank[rank]
                        partial_checkpoint[
                            key.replace(f"layers.{i}.", f"layers.{local_i}.", 1)
                        ] = checkpoint[key]
                if i == num_layers - 1:
                    for prefix in self._get_post_layer_prefixes():
                        if key.startswith(prefix):
                            partial_checkpoint[key] = checkpoint[key]
        return partial_checkpoint

    def _chunk_checkpoint_for_tensor_parallel(
        self,
        checkpoint,
        rank: int,
        world_size: int,
    ):
        keys = checkpoint.keys()
        partial_checkpoint = {}

        cpl_names = self._get_tensor_column_parallel_layer_names()
        rpl_names = self._get_tensor_row_parallel_layer_names()

        for name, param in checkpoint.items():
            if any(is_layer(s, name) for s in cpl_names):
                chunks = torch.chunk(param, world_size, dim=0)
                partial_checkpoint[name] = chunks[rank]
            elif any(is_layer(s, name) for s in rpl_names):
                chunks = torch.chunk(param, world_size, dim=1)
                partial_checkpoint[name] = chunks[rank]
            else:
                partial_checkpoint[name] = param
        return partial_checkpoint

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs):
        if self.pipeline_exec and not self.tensor_exec:
            state_dict = self._chunk_checkpoint_for_pipeline_parallel(
                state_dict, self.global_n_layers, self.rank, self.world_size
            )
        elif self.tensor_exec and not self.pipeline_exec:
            state_dict = self._chunk_checkpoint_for_tensor_parallel(
                state_dict, self.rank, self.world_size
            )
        elif not self.pipeline_exec and not self.tensor_exec:
            pass
        else:
            raise NotImplementedError  # TODO: Hybrid parallelism

        super().load_state_dict(state_dict, *args, **kwargs)

    def _init_pre_layers(self):
        raise NotImplementedError

    def _init_layers(self, cache, attn_backend):
        raise NotImplementedError

    def _init_post_layers(self):
        raise NotImplementedError

    def _pre_layers(self, h):
        raise NotImplementedError

    def _post_layers(self, h):
        """NOTE: _post_layers is assumed to be a token-wise computation"""
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
    def prefill_single_device(self, tokens, varlens=None):
        if isinstance(
            tokens, list
        ):  # else use tensor variable passed by TensorExecutor
            varlens = VarLens(tokens, self.device)
            tokens = torch.from_numpy(np.concatenate(tokens)).to(self.device)
        freqs_cis = self.prepare_freqs_cis_prefill(varlens, self.device)
        h = self._pre_layers(tokens)
        for it, layer in enumerate(self.layers):
            h = layer(h, freqs_cis, varlens)
        tmp = varlens.cpu_prefix_lens[1:]
        h = h[[item - 1 for item in tmp]]
        h = self._post_layers(h)  # Exec post layers AFTER cutting the last token off
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
        varlens = self.cache.curr_varlens
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
            tmp = varlens.cpu_prefix_lens[1:]
            h = h[[item - 1 for item in tmp]]
            h = self._post_layers(
                h
            )  # Exec post layers AFTER cutting the last token off
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
    def prefill(self, tokens, varlens=None):
        if self.pipeline_exec:
            return self.prefill_pipeline(tokens)
        elif self.tensor_exec:
            return self.prefill_single_device(tokens, varlens)
        else:
            return self.prefill_single_device(tokens)

    @torch.inference_mode()
    def decode(self, tokens, seq_lens):
        if self.pipeline_exec:
            return self.decode_pipeline(tokens, seq_lens)
        else:
            return self.decode_single_device(tokens, seq_lens)
