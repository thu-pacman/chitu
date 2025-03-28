import itertools
import math
import os
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn

from chitu.attn_backend import AttnBackend
from chitu.cache_manager import PagedKVCacheManager
from chitu.global_vars import get_global_args, get_timers, set_global_variables
from chitu.muxi_utils import has_tbsgemm, tbsgemm
from chitu.ops import apply_rotary_pos_emb
from chitu.tensor_parallel import get_tp_group, get_tp_rank
from chitu.tokenizer import ChatFormat, ChatFormatHF, Tokenizer, TokenizerHF
from chitu.utils import VarLens, compute_layer_dist_in_pipe, is_layer
from chitu.cuda_graph import make_dispatched_graphed_callables
from chitu.device_type import is_muxi, get_device_name

logger = getLogger(__name__)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _naive_norm(self, x, compute_dtype):
        dtype = x.dtype
        x = x.to(compute_dtype)
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.to(dtype) * self.weight

    def forward(self, x: torch.Tensor, compute_dtype=None):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.
            compute_dtype (torch.dtype, optional): The dtype to use for computation. Defaults to the
                dtype of the input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        # NOTE: Although F.rms_norm uses different dtypes inside itself, and some models directly
        # pass float16 tensors to it, our CI shows it does not work for some models, especially GPTQ
        # quantized models. Maybe we should make the dtype optional.

        if has_tbsgemm and x.dtype == torch.float16:
            output = tbsgemm.norm(x, self.weight)
            return output
        else:
            if compute_dtype is None:
                compute_dtype = torch.float32
            if hasattr(F, "rms_norm"):
                dtype = x.dtype
                return F.rms_norm(
                    x.to(compute_dtype), (self.dim,), self.weight, self.eps
                ).to(dtype)
            else:  # Old PyTorch versions
                return self._naive_norm(x, compute_dtype=compute_dtype)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device=None):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


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
        freqs_cis_cos: torch.Tensor,
        freqs_cis_sin: torch.Tensor,
        varlens,
    ):
        bs_seq, _ = x.shape
        xq, xk, xv = self._run_linear(x)
        xq = xq.view(bs_seq, self.n_local_heads, self.head_dim)
        xk = xk.view(bs_seq, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs_seq, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_pos_emb(
            xq, xk, freqs_cis_cos, freqs_cis_sin, rotary_type="llama"
        )
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

    def decode_forward(
        self, x: torch.Tensor, freqs_cis_cos: torch.Tensor, freqs_cis_sin: torch.Tensor
    ):
        bsz, seqlen, _ = x.shape
        assert seqlen == 1, "decode_forward only supports single token decoding"
        xq, xk, xv = self._run_linear(x)

        xq = xq.view(-1, self.n_local_heads, self.head_dim)
        xk = xk.view(-1, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(-1, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_pos_emb(
            xq, xk, freqs_cis_cos, freqs_cis_sin, rotary_type="llama"
        )

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        cache = self.cache.get_cache_decode(self.layer_id)
        cache_k = cache[0]
        cache_v = cache[1]
        cache_seqlens_excl_this_decode = self.cache.get_gpu_seq_lens_excl_this_decode()
        output = self.attn_backend.attn_with_kvcache(
            xq,
            cache_k,
            cache_v,
            xk,
            xv,
            cache_seqlens=cache_seqlens_excl_this_decode,
        ).view(bsz, seqlen, -1)
        return self._run_output_linear(output)

    def decode_forward_paged(
        self, x: torch.Tensor, freqs_cis_cos: torch.Tensor, freqs_cis_sin: torch.Tensor
    ):
        bsz, seqlen, _ = x.shape
        assert seqlen == 1, "decode_forward only supports single token decoding"
        xq, xk, xv = self._run_linear(x)

        xq = xq.view(-1, self.n_local_heads, self.head_dim)
        xk = xk.view(-1, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(-1, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_pos_emb(
            xq, xk, freqs_cis_cos, freqs_cis_sin, rotary_type="llama"
        )

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        block_table = self.cache.get_gpu_block_table()
        cache_seqlens_excl_this_decode = self.cache.get_gpu_seq_lens_excl_this_decode()
        paged_k_cache, paged_v_cache = self.cache.get_paged_kv_cache(self.layer_id)
        output = self.attn_backend.attn_with_kvcache(
            xq,
            paged_k_cache,
            paged_v_cache,
            xk,
            xv,
            cache_seqlens=cache_seqlens_excl_this_decode,
            block_table=block_table,
        ).view(bsz, seqlen, -1)
        return self._run_output_linear(output)

    def forward(self, x, freqs_cis_cos, freqs_cis_sin, varlens=None):
        if varlens is not None:  # prefill
            return self.prefill_forward(x, freqs_cis_cos, freqs_cis_sin, varlens)
        elif isinstance(self.cache, PagedKVCacheManager):
            return self.decode_forward_paged(x, freqs_cis_cos, freqs_cis_sin)
        else:
            return self.decode_forward(x, freqs_cis_cos, freqs_cis_sin)


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
        *,
        max_position_embeddings: int,
        pipeline_parallel_size: int,
        model_parallel_size: int,
        attn_backend: AttnBackend,
        op_impl: str,
        **kvargs,
    ):
        super().__init__()
        self.cache = cache
        self.attn_backend = attn_backend
        self.op_impl = op_impl
        self.rank = torch.distributed.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = torch.distributed.get_world_size()
        self.device = torch.device(self.local_rank)

        self.pipeline_parallel_size = pipeline_parallel_size
        self.model_parallel_size = model_parallel_size
        self.pipeline_exec = pipeline_parallel_size > 1
        self.tensor_exec = model_parallel_size > 1

        self.tp_size = model_parallel_size
        self.pp_size = pipeline_parallel_size
        self.pp_stage = self.rank // self.model_parallel_size
        self.pp_main_rank = (self.rank // model_parallel_size) * model_parallel_size
        self.pp_end_stage = (self.world_size - 1) // model_parallel_size
        self.tp_group = get_tp_group()

        self.params = params
        self.vocab_size = params.vocab_size
        self.global_n_layers = params.n_layers

        if self.pipeline_exec:
            num_layers_of_each_rank = compute_layer_dist_in_pipe(
                self.global_n_layers, self.pipeline_parallel_size
            )
            first_layer_id_of_each_rank = list(
                itertools.accumulate([0] + num_layers_of_each_rank)
            )
            self.local_begin_layer_id = first_layer_id_of_each_rank[self.pp_stage]
            self.local_end_layer_id = first_layer_id_of_each_rank[self.pp_stage + 1]
        else:
            self.local_begin_layer_id = 0
            self.local_end_layer_id = self.global_n_layers

        if not self.pipeline_exec or self.pp_stage == 0:
            self._init_pre_layers()
        self._init_layers(cache, attn_backend=attn_backend, op_impl=op_impl)
        if not self.pipeline_exec or self.pp_stage == self.pipeline_parallel_size - 1:
            self._init_post_layers()

        self.precompute_freqs_cis(max_position_embeddings, self.device)

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
                if name.endswith("weight") or (
                    self.params.type == "deepseek-v3" and name.endswith("scale")
                ):
                    chunks = torch.chunk(param, world_size, dim=0)
                    partial_checkpoint[name] = chunks[rank]
                elif name.endswith("bias"):
                    chunks = torch.chunk(param, world_size, dim=-1)
                    partial_checkpoint[name] = chunks[rank]
                else:
                    assert False, f"Illegal parallel tensor {name}"
            elif any(is_layer(s, name) for s in rpl_names):
                if name.endswith("weight") or (
                    self.params.type == "deepseek-v3" and name.endswith("scale")
                ):
                    chunks = torch.chunk(param, world_size, dim=1)
                    partial_checkpoint[name] = chunks[rank]
                elif name.endswith("bias"):
                    # Rank 0 needs a full bias and only rank 0 needs it
                    if get_tp_rank() == 0:
                        partial_checkpoint[name] = param
                else:
                    assert False, f"Illegal parallel tensor {name}"
            else:
                partial_checkpoint[name] = param
        return partial_checkpoint

    def load_state_dict_parallel(
        self,
        state_dict: Mapping[str, Any],
        skip_preprocess: bool = False,
        *args,
        **kwargs,
    ):
        if not skip_preprocess:
            if self.pipeline_exec:
                state_dict = self._chunk_checkpoint_for_pipeline_parallel(
                    state_dict, self.global_n_layers, self.pp_stage, self.pp_size
                )
            if self.tensor_exec:
                state_dict = self._chunk_checkpoint_for_tensor_parallel(
                    state_dict, self.rank % self.tp_size, self.tp_size
                )
        self.load_state_dict(
            state_dict, skip_preprocess=skip_preprocess, *args, **kwargs
        )

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        skip_preprocess: bool = False,
        *args,
        **kwargs,
    ):
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

    def precompute_freqs_cis(self, max_position_embeddings, device):
        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads,
            max_position_embeddings * 2,
            self.params.rope_theta,
            device=device,
        )

    def prepare_freqs_cis_prefill(self, varlens):
        curr_freqs_cis = self.freqs_cis[self.cache.curr_varlens.position_ids]
        return curr_freqs_cis.real.contiguous(), curr_freqs_cis.imag.contiguous()

    def prepare_freqs_cis_decode(self):
        curr_freqs_cis = self.freqs_cis[self.cache.get_gpu_seq_lens_excl_this_decode()]
        return curr_freqs_cis.real.contiguous(), curr_freqs_cis.imag.contiguous()

    @torch.inference_mode()
    def prefill_single_device(self, tokens, varlens=None):
        if isinstance(
            tokens, list
        ):  # else use tensor variable passed by TensorExecutor
            varlens = VarLens(tokens, self.device)
            tokens = torch.from_numpy(np.concatenate(tokens)).to(self.device)
        freqs_cis_cos, freqs_cis_sin = self.prepare_freqs_cis_prefill(varlens)
        h = self._pre_layers(tokens)
        for it, layer in enumerate(self.layers):
            h = layer(h, freqs_cis_cos, freqs_cis_sin, varlens)
        tmp = varlens.cpu_prefix_lens[1:]
        h = h[[item - 1 for item in tmp]]
        h = self._post_layers(h)  # Exec post layers AFTER cutting the last token off
        h = h.float()
        return h

    @torch.inference_mode()
    def decode_single_device(self, tokens, freqs_cis_cos, freqs_cis_sin):
        h = self._pre_layers(tokens)
        for it, layer in enumerate(self.layers):
            h = layer(h, freqs_cis_cos, freqs_cis_sin)
        h = self._post_layers(h)
        h = h.float()
        return h

    @torch.inference_mode()
    def prefill_pipeline(self, tokens):

        varlens = self.cache.curr_varlens
        freqs_cis_cos, freqs_cis_sin = self.prepare_freqs_cis_prefill(varlens)

        # start of model
        if self.pp_stage == 0:
            h = self._pre_layers(tokens)
        else:
            h = tokens
        # layers
        for it, layer in enumerate(self.layers):
            h = layer(h, freqs_cis_cos, freqs_cis_sin, varlens)
        # end of model
        if self.pp_stage == self.pp_end_stage:
            tmp = varlens.cpu_prefix_lens[1:]
            h = h[[item - 1 for item in tmp]]
            h = self._post_layers(
                h
            )  # Exec post layers AFTER cutting the last token off

            h = h.float()

        return h

    @torch.inference_mode()
    def decode_pipeline(self, tokens, freqs_cis_cos, freqs_cis_sin):
        if self.pp_stage == 0:
            h = self._pre_layers(tokens)
        else:
            h = tokens
        for it, layer in enumerate(self.layers):
            h = layer(h, freqs_cis_cos, freqs_cis_sin)
        if self.pp_stage == self.pp_end_stage:
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

    def prepare_decoding_attn(self):
        cache_seqlens_excl_this_decode = self.cache.get_gpu_seq_lens_excl_this_decode()
        cache_seqlens_incl_this_decode = self.cache.get_gpu_seq_lens_incl_this_decode()
        block_table = self.cache.get_gpu_block_table()
        block_size = self.cache.get_block_size()
        self.attn_backend.prepare_metadata_for_decode(
            cache_seqlens_excl_this_decode,
            cache_seqlens_incl_this_decode,
            block_table,
            block_size,
        )

    @torch.inference_mode()
    def decode(self, tokens, seq_lens):
        self.prepare_decoding_attn()

        batch_size = len(seq_lens)
        max_batch_size = get_global_args().infer.max_reqs

        use_cuda_graph = get_global_args().infer.use_cuda_graph

        if use_cuda_graph:
            if get_global_args().infer.cache_type == "skew":
                raise NotImplementedError(
                    'CUDA graph is currently not supported for infer.cache_type="skew"'
                )

        @make_dispatched_graphed_callables(
            sample_args=(tokens,),
            sample_kwargs={},
            args_max_nelem=(tokens.numel() // batch_size * max_batch_size,),
            kwargs_max_nelem={},
            output_max_nelem_callback=lambda n: n // batch_size * max_batch_size,
            enable=use_cuda_graph,
        )
        def do_decode(tokens):
            freqs_cis_cos, freqs_cis_sin = self.prepare_freqs_cis_decode()
            if self.pipeline_exec:
                return self.decode_pipeline(tokens, freqs_cis_cos, freqs_cis_sin)
            else:
                return self.decode_single_device(tokens, freqs_cis_cos, freqs_cis_sin)

        return do_decode(batch_size, tokens)
