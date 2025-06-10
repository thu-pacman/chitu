import itertools
import math
import os
import re
import functools
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, List, Mapping, Optional, Dict, Type, Set
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from chitu.attn_backend import AttnBackend
from chitu.cache_manager import PagedKVCacheManager
from chitu.global_vars import get_global_args, get_timers, set_global_variables
from chitu.muxi_utils import has_tbsgemm, tbsgemm
from chitu.ops import apply_rotary_pos_emb, rms_norm, silu_and_mul, topk_softmax
from chitu.ops import weight_dequant_soft_fp8_deepseek_v3 as weight_dequant_soft_fp8
from chitu.tensor_parallel import get_tp_group, get_tp_rank, get_tp_size
from chitu.tokenizer import ChatFormat, ChatFormatHF, Tokenizer, TokenizerHF
from chitu.utils import VarLens, compute_layer_dist_in_pipe, is_layer, ceil_div
from chitu.cuda_graph import make_dispatched_graphed_callables
from chitu.device_type import is_muxi, get_device_name, is_nvidia, is_ascend
from chitu.utils import try_import_opt_dep, parse_dtype
from chitu.muxi_utils import grouped_topk, muxi_fused_experts
from chitu.layers.gate import fused_sigmoid_gate
from chitu.quantization import (
    linear_block_fp8,
    linear_block_fp4,
)

torch_npu, has_torch_npu = try_import_opt_dep("torch_npu", "torch_npu")
chitu_backend, has_chitu_backend = try_import_opt_dep("chitu_backend", "chitu_backend")
triton, has_triton = try_import_opt_dep("triton", "triton")
torch_npu, has_torch_npu = try_import_opt_dep("torch_npu", "torch_npu")

if has_torch_npu:
    from chitu.npu_utils import fused_experts_npu

if has_triton:
    from chitu.fused_moe import fused_experts

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

    def _ref_norm(self, x, compute_dtype):
        dtype = x.dtype
        x = x.to(compute_dtype)
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.to(dtype) * self.weight

    def forward(
        self,
        x: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        compute_dtype=None,
        impl: str = "auto",
    ):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.
            out (torch.Tensor, optional): If set, the output will be written to this tensor.
            compute_dtype (torch.dtype, optional): The dtype to use for computation. Defaults to the
                dtype of the input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        # NOTE: Although F.rms_norm uses different dtypes inside itself, and some models directly
        # pass float16 tensors to it, our CI shows it does not work for some models, especially GPTQ
        # quantized models. Maybe we should make the dtype optional.

        if compute_dtype is None:
            compute_dtype = torch.float32

        if impl == "auto":
            triton, has_triton = try_import_opt_dep("triton", "triton")
            if out is not None and has_chitu_backend:
                impl = "cuda"
            elif (
                has_tbsgemm
                and get_global_args().dtype == "float16"
                and self.eps == 1e-6
            ):
                impl = "muxi_w8a8_kernels"
            elif has_triton:
                impl = "triton"
            elif has_torch_npu:
                impl = "torch_npu"
            elif hasattr(F, "rms_norm"):
                impl = "torch"
            else:
                impl = "ref"

        if impl == "triton":
            assert out is None
            return rms_norm(x, self.weight, self.eps, compute_dtype=compute_dtype)
        elif impl == "cuda":
            # Currently, this kernel always raise to float32 to compute
            return chitu_backend.cuda_rms_norm(x, self.weight, eps=self.eps, out=out)
        elif impl == "muxi_w8a8_kernels":
            assert out is None
            assert self.eps == 1e-6
            assert x.dtype == torch.float16
            return tbsgemm.norm(x, self.weight)
        elif impl == "torch_npu":
            dtype = x.dtype
            tmp_out = torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0].to(
                dtype
            )
            if out is not None:
                out.copy_(tmp_out)
            else:
                out = tmp_out
            return out
        elif impl == "torch":
            dtype = x.dtype
            tmp_out = F.rms_norm(
                x.to(compute_dtype), (self.dim,), self.weight, self.eps
            ).to(dtype)
            if out is not None:
                out.copy_(tmp_out)
            else:
                out = tmp_out
            return out
        elif impl == "ref":
            tmp_out = self._ref_norm(x, compute_dtype)
            if out is not None:
                out.copy_(tmp_out)
            else:
                out = tmp_out
            return out
        else:
            raise ValueError(f"Invalid RMSNorm implementation: {impl}")


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
        is_fp4=False,
        **kvargs,
    ):
        super().__init__()
        self.cache = cache
        self.attn_backend = attn_backend
        self.op_impl = op_impl
        self.is_fp4 = is_fp4
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

        self.do_decode_callable = None
        self.args = get_global_args()
        self.max_batch_size = self.args.infer.max_reqs
        self.model_type = self.args.models.type
        self.use_cuda_graph = self.args.infer.use_cuda_graph
        if self.use_cuda_graph and is_ascend() and self.model_type == "deepseek-v3":
            raise NotImplementedError(
                "Graph capturing is not yet implemented for deepseek models on Ascend NPU"
            )

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

    def _get_2d_out_x_in_tensor_names(self, quant) -> List[str]:
        ret = ["weight"]
        if quant == "blockfp8" or quant == "gguf-blockfp8":
            ret += ["scale"]
        elif quant == "blockfp4":
            ret += ["weight_scale", "weight_scale_2", "input_scale"]
        return ret

    def _get_2d_in_x_out_tensor_names(self, quant) -> List[str]:
        ret = []
        if quant == "autoawq":
            ret += ["qweight", "qzeros", "scales"]
        elif quant == "gptqmodel":
            ret += ["qweight", "qzeros", "scales"]
        return ret

    def _get_1d_in_tensor_names(self, quant) -> List[str]:
        ret = []
        if quant == "gptqmodel":
            ret += ["g_idx"]
        return ret

    def _get_1d_out_tensor_names(self, quant) -> List[str]:
        ret = ["bias"]
        if quant == "simple_w8a8":
            ret += ["scale_channel"]
        if quant == "simple_w8a8_muxi":
            ret += ["scale_channel"]
        return ret

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
            quant = None
            for rule in self.params.quant_config.rules:
                pattern = rule.get("regex")
                if pattern and re.search(pattern, name):
                    quant = rule.type
                    break
            if any(is_layer(s, name) for s in cpl_names):
                if name.split(".")[-1] in self._get_1d_in_tensor_names(
                    quant
                ) + self._get_1d_out_tensor_names(quant):
                    assert (
                        param.dim() == 1
                    ), f"{name} is expected to be 1D, but got {param.dim()}D"
                    if param.shape[0] == 1:  # Broadcast
                        partial_checkpoint[name] = param
                    else:
                        assert param.shape[0] % world_size == 0
                        chunks = torch.chunk(param, world_size, dim=0)
                        partial_checkpoint[name] = chunks[rank]
                elif name.split(".")[-1] in self._get_2d_out_x_in_tensor_names(quant):
                    assert (
                        param.dim() == 2
                    ), f"{name} is expected to be 2D, but got {param.dim()}D"
                    if param.shape[0] == 1:  # Broadcast
                        partial_checkpoint[name] = param
                    else:
                        assert param.shape[0] % world_size == 0
                        chunks = torch.chunk(param, world_size, dim=0)
                        partial_checkpoint[name] = chunks[rank]
                elif name.split(".")[-1] in self._get_2d_in_x_out_tensor_names(quant):
                    assert (
                        param.dim() == 2
                    ), f"{name} is expected to be 2D, but got {param.dim()}D"
                    if param.shape[1] == 1:  # Broadcast
                        partial_checkpoint[name] = param
                    else:
                        assert param.shape[1] % world_size == 0
                        chunks = torch.chunk(param, world_size, dim=1)
                        partial_checkpoint[name] = chunks[rank]
                else:
                    # FIXME: Support quant=llmint8 for TP
                    assert False, f"Illegal parallel tensor {name}"

            elif any(is_layer(s, name) for s in rpl_names):
                if name.split(".")[-1] in self._get_1d_in_tensor_names(
                    quant
                ) + self._get_1d_out_tensor_names(quant):
                    assert (
                        param.dim() == 1
                    ), f"{name} is expected to be 1D, but got {param.dim()}D"
                    if param.shape[0] == 1:  # Broadcast
                        partial_checkpoint[name] = param
                    else:
                        assert param.shape[0] % world_size == 0
                        chunks = torch.chunk(param, world_size, dim=0)
                        partial_checkpoint[name] = chunks[rank]
                elif name.split(".")[-1] in self._get_2d_out_x_in_tensor_names(quant):
                    assert (
                        param.dim() == 2
                    ), f"{name} is expected to be 2D, but got {param.dim()}D"
                    if param.shape[1] == 1:  # Broadcast
                        partial_checkpoint[name] = param
                    else:
                        assert param.shape[1] % world_size == 0
                        chunks = torch.chunk(param, world_size, dim=1)
                        partial_checkpoint[name] = chunks[rank]
                elif name.split(".")[-1] in self._get_2d_in_x_out_tensor_names(quant):
                    assert (
                        param.dim() == 2
                    ), f"{name} is expected to be 2D, but got {param.dim()}D"
                    if param.shape[0] == 1:  # Broadcast
                        partial_checkpoint[name] = param
                    else:
                        assert param.shape[0] % world_size == 0
                        chunks = torch.chunk(param, world_size, dim=0)
                        partial_checkpoint[name] = chunks[rank]
                else:
                    # FIXME: Support quant=llmint8 for TP
                    assert False, f"Illegal parallel tensor {name}"

            else:
                partial_checkpoint[name] = param
        return partial_checkpoint

    def process_state_dict_for_blockfp4_before_chunk(self, state_dict):
        quant = (
            self.params.quant_config.type
            if hasattr(self.params, "quant_config")
            else None
        )
        if quant == "blockfp4":
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.endswith(".weight_scale_2") or key.endswith(".input_scale"):
                    new_state_dict[key] = value.view(1, 1)
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        return state_dict

    def process_state_dict_for_blockfp4_after_chunk(self, state_dict):
        quant = (
            self.params.quant_config.type
            if hasattr(self.params, "quant_config")
            else None
        )
        if quant == "blockfp4":
            new_state_dict = {}
            for k in state_dict.keys():
                param = state_dict[k]
                if param.dtype == torch.uint8 and k.endswith("weight"):
                    param.data = chitu_backend.weight_layout_change(
                        param.data.cuda()
                    ).cpu()
                new_state_dict[k] = param
            state_dict = new_state_dict
        return state_dict

    def process_state_dict_for_renaming_linear_layer(
        self, checkpoint, merge_gate_up, n_dense_layers
    ):
        """
        重命名专家权重结构的函数以消除冗余的 gate,up,down 层
        参数格式示例：
        输入键：'layers.3.mlp.gate_proj.weight'
        输出键：'layers.3.mlp.gate_proj_weight'
        """
        from collections import defaultdict

        new_checkpoint = {}
        if not merge_gate_up:
            pattern_lists = [
                r"layers\.(\d+)\.mlp\.gate_proj\.([^.]+)",
                r"layers\.(\d+)\.mlp\.down_proj\.([^.]+)",
                r"layers\.(\d+)\.mlp\.up_proj\.([^.]+)",
            ]
            tensor_names = ["gate_proj", "down_proj", "up_proj"]
        else:
            pattern_lists = [
                r"layers\.(\d+)\.mlp\.gate_up_proj\.([^.]+)",
                r"layers\.(\d+)\.mlp\.down_proj\.([^.]+)",
            ]
            tensor_names = ["gate_up_proj", "down_proj"]

        for key in checkpoint:
            matched = False
            for tensor_name, pattern in zip(tensor_names, pattern_lists):
                match = re.fullmatch(pattern, key)
                if match:
                    layer_idx = int(match.group(1))
                    suffix = match.group(2)
                    if layer_idx < n_dense_layers:
                        break
                    new_key = f"layers.{layer_idx}.mlp.{tensor_name}_{suffix}"
                    new_checkpoint[new_key] = checkpoint[key]
                    matched = True
                    break
            if not matched:
                new_checkpoint[key] = checkpoint[key]

        return new_checkpoint

    def load_state_dict_parallel(
        self,
        state_dict: Mapping[str, Any],
        skip_preprocess: bool = False,
        *args,
        **kwargs,
    ):
        if not skip_preprocess:
            state_dict = self.process_state_dict_for_blockfp4_before_chunk(state_dict)
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
        if not skip_preprocess:
            state_dict = self.process_state_dict_for_blockfp4_after_chunk(state_dict)
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
        self.attn_backend.prepare_metadata_for_prefill(self.cache.curr_varlens)
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

        if self.do_decode_callable is None:

            before_replay_callback = None
            if is_ascend():
                before_replay_callback = lambda graph: graph.update(
                    cpu_update_input=[
                        {"actual_seq_lengths_kv": self.attn_backend.seq_lens_incl_list}
                    ]
                )

            @make_dispatched_graphed_callables(
                args_max_nelem=(tokens.numel() // batch_size * self.max_batch_size,),
                kwargs_max_nelem={},
                output_max_nelem_callback=lambda bs, n: n // bs * self.max_batch_size,
                before_replay_callback=before_replay_callback,
                enable=self.use_cuda_graph,
            )
            def do_decode(tokens):
                freqs_cis_cos, freqs_cis_sin = self.prepare_freqs_cis_decode()
                if self.pipeline_exec:
                    return self.decode_pipeline(tokens, freqs_cis_cos, freqs_cis_sin)
                else:
                    return self.decode_single_device(
                        tokens, freqs_cis_cos, freqs_cis_sin
                    )

            self.do_decode_callable = do_decode

        return self.do_decode_callable(batch_size, tokens)


class MoeGate(nn.Module):
    def __init__(
        self,
        op_impl,
        dim,
        topk,
        n_groups,
        topk_groups,
        score_func,
        route_scale,
        n_experts,
        bias,
        norm_prob,
    ):
        """
        Initializes the Gate module.
        """
        super().__init__()
        self.op_impl = op_impl
        self.dim = dim
        self.topk = topk
        self.n_groups = n_groups
        self.topk_groups = topk_groups
        self.score_func = score_func
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty((n_experts, self.dim)))
        self.bias = bias
        self.norm_prob = norm_prob

    def is_fused_sigmoid_gate(self):
        return self.score_func == "sigmoid" and self.n_groups > 1

    def forward(self, x):
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        if self.op_impl == "muxi_custom_kernel":
            scores = F.linear(x, self.weight)
            weights, indices = grouped_topk(
                x,
                scores,
                self.topk,
                renormalize=self.score_func == "sigmoid",
                num_expert_group=self.n_groups,
                topk_group=self.topk_groups,
                scoring_func=self.score_func,
                e_score_correction_bias=(
                    None if self.bias is None else self.bias.type_as(scores)
                ),
            )
        elif self.is_fused_sigmoid_gate() and (is_nvidia() or is_muxi()):
            scores = F.linear(x, self.weight)
            indices, weights = fused_sigmoid_gate(
                scores, self.topk, self.n_groups, self.topk_groups, self.bias
            )
            weights /= weights.sum(dim=-1, keepdim=True)
        elif self.score_func == "softmax" and (is_nvidia()):
            scores = F.linear(x, self.weight)
            weights, indices, expert_idx = topk_softmax(
                scores, self.topk, self.norm_prob, torch.int32
            )
        else:
            scores = F.linear(x, self.weight)
            if self.score_func == "softmax":
                scores = scores.softmax(dim=-1, dtype=torch.float32)
            else:
                scores = scores.sigmoid()
            original_scores = scores
            if self.bias is not None:
                scores = scores + self.bias
            if self.n_groups > 1:
                scores = scores.view(x.size(0), self.n_groups, -1)
                if self.bias is None:
                    group_scores = scores.amax(dim=-1)
                else:
                    group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
                indices = group_scores.topk(self.topk_groups, dim=-1)[1]
                mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(
                    1, indices, False
                )
                scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(
                    1
                )
            indices = torch.topk(scores, self.topk, dim=-1)[1]
            weights = original_scores.gather(1, indices)

            if self.score_func == "sigmoid" or self.norm_prob:
                weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices.to(torch.int32)


class MoeBlock(nn.Module):
    """
    Basic Moe Block.
    Example:
    >>> class derivedMoe(MoeBlock):
    >>>     super().__init__(
                dim=1024,
                hidden_dim=4096,
                n_routed_experts=8,
                n_shared_experts=2,
                moe_world_size=4,
                moe_rank=0,
                merge_gate_up=True,
                ...
            )
    """

    def __init__(
        self,
        dim: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_shared_experts: int,
        n_activated_experts: int,
        moe_world_size: int,
        moe_rank: int,
        do_gather_output: bool,
        merge_gate_up: bool,
        dtype: str,
        op_impl: str,
        gate: MoeGate,
        fuse_shared_experts: bool,
        shared_experts: nn.Module,
        checkpoint_prefix: str,
        build_weight: bool = True,
    ):
        super().__init__()
        self.op_impl = op_impl
        self.merge_gate_up = merge_gate_up
        self.gate = gate
        self.dim = dim
        self.fuse_shared_experts = fuse_shared_experts
        assert (
            n_routed_experts % moe_world_size == 0
        ), f"Number of experts must be divisible by world size (world_size={moe_world_size})"
        self.n_shared_experts = n_shared_experts
        self.n_fused_shared_experts = (
            n_shared_experts if self.fuse_shared_experts else 0
        )
        self.n_routed_experts = n_routed_experts
        self.n_local_experts = n_routed_experts // moe_world_size
        self.n_activated_experts = n_activated_experts
        self.experts_start_idx = moe_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.group_size = (
            self.experts_end_idx - self.experts_start_idx + self.n_fused_shared_experts
        )
        self.tp_group = get_tp_group()
        self.tp_size = get_tp_size()
        self.do_gather_output = do_gather_output
        self.checkpoint_prefix = checkpoint_prefix
        # Non-fused shared experts
        if not self.fuse_shared_experts:
            self.shared_experts = shared_experts

        self.linear_dtype = (
            torch.uint8
            if (
                parse_dtype(dtype).itemsize == 1
                and parse_dtype(
                    get_global_args().infer.raise_lower_bit_float_to
                ).itemsize
                > 1
            )
            else parse_dtype(dtype)
        )
        if build_weight:
            if not self.merge_gate_up:
                self.gate_proj_weight = nn.Parameter(
                    torch.empty(
                        (self.group_size, moe_inter_dim // self.tp_size, self.dim),
                        dtype=self.linear_dtype,
                    ),
                    requires_grad=False,
                )
                self.up_proj_weight = nn.Parameter(
                    torch.empty(
                        (self.group_size, moe_inter_dim // self.tp_size, self.dim),
                        dtype=self.linear_dtype,
                    ),
                    requires_grad=False,
                )
            else:
                self.gate_up_proj_weight = nn.Parameter(
                    torch.empty(
                        (self.group_size, moe_inter_dim * 2 // self.tp_size, self.dim),
                        dtype=self.linear_dtype,
                    ),
                    requires_grad=False,
                )
            self.down_proj_weight = nn.Parameter(
                torch.empty(
                    (self.group_size, self.dim, moe_inter_dim // self.tp_size),
                    dtype=self.linear_dtype,
                ),
                requires_grad=False,
            )

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """

        shape = x.size()
        x = x.view(-1, self.dim)

        weights, indices = self.gate(x)

        if self.op_impl == "muxi_custom_kernel":
            y = self._compute_muxi_fused_experts(x, weights, indices)
        elif has_torch_npu:  # or use op_impl ?
            y = self._compute_npu_fused_experts(x, weights, indices)
        elif has_triton:
            assert self.merge_gate_up
            if (
                parse_dtype(get_global_args().infer.raise_lower_bit_float_to).itemsize
                == 1
                or is_nvidia()
                or is_muxi()
            ):
                fused_soft_fp8 = (
                    parse_dtype(
                        get_global_args().infer.raise_lower_bit_float_to
                    ).itemsize
                    != 1
                )
                gate_up_proj_weight = self.gate_up_proj_weight
                down_proj_weight = self.down_proj_weight
            else:
                logger.warning(
                    f"Soft-fp8 fused gemm not implemented for {get_device_name()}, falling back to soft-fp8 conversion"
                )
                block_size = 128
                gate_up_proj_weight = weight_dequant_soft_fp8(
                    self.gate_up_proj_weight,
                    None,
                    block_size,
                )
                down_proj_weight = weight_dequant_soft_fp8(
                    self.down_proj_weight,
                    None,
                    block_size,
                )
                fused_soft_fp8 = False

            if not self.fuse_shared_experts:

                y = fused_experts(
                    x,
                    gate_up_proj_weight,
                    down_proj_weight,
                    topk_weights=weights,
                    topk_ids=indices,
                    use_fp8_w8a8=False,
                    use_fp4_w4a8=False,
                    inplace=True,
                    global_num_experts=self.n_routed_experts + self.n_shared_experts,
                    expert_map=None,  # use when ep > 1
                    w1_scale=None,
                    w2_scale=None,
                    w1w3_scale_2=None,
                    w2_scale_2=None,
                    block_shape=[128, 128],
                    soft_fp8=fused_soft_fp8,
                )

                if self.n_shared_experts > 0:
                    y1 = self.shared_experts(x)
                    y += y1
            else:

                indice_shape = indices.shape
                new_indices = torch.empty(
                    (indice_shape[0], indice_shape[1] + 1),
                    dtype=indices.dtype,
                    device=indices.device,
                )

                new_weights = torch.empty(
                    (weights.shape[0], weights.shape[1] + 1),
                    dtype=weights.dtype,
                    device=weights.device,
                )

                chitu_backend.cuda_add_shared_experts(
                    new_weights,
                    new_indices,
                    weights,
                    indices,
                    self.n_routed_experts,
                    self.n_shared_experts,
                )
                del weights, indices
                y = fused_experts(
                    x,
                    gate_up_proj_weight,
                    down_proj_weight,
                    topk_weights=new_weights,
                    topk_ids=new_indices,
                    use_fp8_w8a8=False,
                    use_fp4_w4a8=False,
                    inplace=True,
                    global_num_experts=self.n_routed_experts + self.n_shared_experts,
                    expert_map=None,  # use when ep > 1
                    w1_scale=None,
                    w2_scale=None,
                    w1w3_scale_2=None,
                    w2_scale_2=None,
                    block_shape=[128, 128],
                    soft_fp8=fused_soft_fp8,
                )

            if self.tp_size > 1:
                torch.distributed.all_reduce(y, group=get_tp_group())
        else:
            y = torch.zeros_like(x)
            counts = torch.bincount(
                indices.flatten(), minlength=self.n_routed_experts
            ).tolist()

            xs = []
            for i in range(self.experts_start_idx, self.experts_end_idx):
                this_x = None
                if counts[i]:
                    idx, top = torch.where(indices == i)
                    this_x = x[idx]
                xs.append(this_x)
            if self.fuse_shared_experts:
                xs += [x] * self.n_fused_shared_experts

            assert len(xs) == self.group_size
            if self.merge_gate_up:
                gate_up_proj_outs = []
                for i in range(self.group_size):
                    out = None
                    if xs[i] is not None:
                        out = F.linear(xs[i], self.gate_up_proj_weight[i], bias=None)
                        if self.do_gather_output and self.tp_size > 1:
                            out = self.gather_output(
                                out, self.tp_size, tp_group=self.tp_group
                            )
                    gate_up_proj_outs.append(out)
                act = [
                    (
                        silu_and_mul(gate_up_proj_out)
                        if gate_up_proj_out is not None
                        else None
                    )
                    for gate_up_proj_out in gate_up_proj_outs
                ]
            else:
                gate_proj_outs = []
                up_proj_outs = []
                for i in range(self.group_size):
                    gate_proj_out = None
                    up_proj_out = None
                    if xs[i] is not None:
                        gate_proj_out = F.linear(
                            xs[i], self.gate_proj_weight[i], bias=None
                        )
                        up_proj_out = F.linear(xs[i], self.up_proj_weight[i], bias=None)
                        if self.do_gather_output and self.tp_size > 1:
                            gate_proj_out = self.gather_output(
                                gate_proj_out, self.tp_size, tp_group=self.tp_group
                            )
                            up_proj_out = self.gather_output(
                                up_proj_out, self.tp_size, tp_group=self.tp_group
                            )
                    gate_proj_outs.append(gate_proj_out)
                    up_proj_outs.append(up_proj_out)

                act = [
                    (
                        F.silu(gate_proj_out) * up_proj_out
                        if gate_proj_out is not None
                        else None
                    )
                    for gate_proj_out, up_proj_out in zip(gate_proj_outs, up_proj_outs)
                ]

            down_proj_outs = []
            for i in range(self.group_size):
                down_proj_out = None
                if act[i] is not None:
                    down_proj_out = F.linear(
                        act[i], self.down_proj_weight[i], bias=None
                    )
                down_proj_outs.append(down_proj_out)

            for i in range(self.experts_start_idx, self.experts_end_idx):
                if counts[i]:
                    idx, top = torch.where(indices == i)
                    y[idx] += (
                        down_proj_outs[i - self.experts_start_idx]
                        * weights[idx, top, None]
                    )
            if self.fuse_shared_experts:
                for i in range(
                    self.experts_end_idx - self.experts_start_idx,
                    self.experts_end_idx
                    - self.experts_start_idx
                    + self.n_fused_shared_experts,
                ):
                    y += down_proj_outs[i]
            else:
                if self.n_shared_experts > 0:
                    y += self.shared_experts(x)
            if self.tp_size > 1:
                dist.all_reduce(y, group=self.tp_group)
        return y.view(shape)

    def _compute_muxi_fused_experts(self, x, weights, indices):
        if self.fuse_shared_experts:
            raise NotImplementedError(
                "Fused shared experts is not supported for muxi_layout_kernels"
            )
        if not self.merge_gate_up:
            raise NotImplementedError(
                "muxi_layout_kernels for fused MoE requires merge_gate_up=True"
            )

        y = muxi_fused_experts(
            hidden_states=x,
            w1=self.gate_up_proj_weight,
            w2=self.down_proj_weight,
            topk_weights=weights,
            topk_ids=indices,
            inplace=True,
            w1_scale=None,
            w2_scale=None,
            block_shape=[128, 128],
            soft_fp8=False,
        )
        if self.n_shared_experts > 0:
            y1 = self.shared_experts(x)
            y += y1
        torch.distributed.all_reduce(y, group=get_tp_group())
        return y

    def _compute_npu_fused_experts(self, x, weights, indices):
        y = fused_experts_npu(
            hidden_states=x,
            w1=self.gate_up_proj_weight,
            w2=self.down_proj_weight,
            topk_weights=weights,
            topk_ids=indices,
        )
        if self.n_shared_experts > 0:
            y1 = self.shared_experts(x)
            y += y1
        torch.distributed.all_reduce(y, group=get_tp_group())
        return y

    def gather_output(
        x: torch.Tensor,
        tp_size: int,
        tp_group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        """
        Gather output tensor across multiple devices.

        Args:
            x (torch.Tensor): Input tensor.
            gather_output (bool): Flag to indicate if gathering is needed.
            tp_group (Optional[torch.distributed.ProcessGroup]): Process group for gathering.

        Returns:
            torch.Tensor: Gathered output tensor.
        """
        x = x.permute(-1, *range(x.dim() - 1)).contiguous()
        shape = list(x.shape)
        shape[0] *= tp_size
        x_gathered = x.new_empty(shape)
        torch.distributed.all_gather_into_tensor(x_gathered, x, group=tp_group)
        return x_gathered.permute(*range(1, x.dim()), 0)


class MoeBlockRegistry:
    """
    Registry of available quantization methods and their implementations.
    """

    _registry: Dict[str, Type[MoeBlock]] = {}

    @classmethod
    def get_all_methods(cls) -> Set[str]:
        """
        Get all registered MoeBlock methods.

        Returns:
            Set of MoeBlock method names
        """
        ret = set(cls._registry.keys())
        ret.remove(None)
        return ret

    @classmethod
    def get_MoeBlock_class(
        cls,
        method: Optional[str],
        *,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
    ) -> Optional[Type[MoeBlock]]:
        """
        Get the quantized moe implementation for the specified method.

        Arguments:
            method: Quantization method name, or None for no quantization
            quant_kwargs: Nested mapping for additional arguments for specific
                quantization methods. E.g., `{"quant_method_x": {"arg1": value1, ...}}`
        Returns:
            The quantized moe class, or None if method is None or not found
        """

        impl = cls._registry.get(method)
        if impl is None:
            raise ValueError(f"Unknown quantization method in `method`: {method}")

        for key in quant_kwargs:
            if key not in cls._registry:
                raise ValueError(
                    f"Unknown quantization method in `quant_kwargs`: {key}"
                )

        if method in quant_kwargs:

            class QuantMoeImpl(impl):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **quant_kwargs[method], **kwargs)

            impl = QuantMoeImpl

        return impl

    @classmethod
    def get_quantized_MoeBlock_class_from_global_args(
        cls,
        *,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        checkpoint_prefix="",
    ) -> Optional[Type[MoeBlock]]:
        args = get_global_args()
        quant_cfg = getattr(args.models, "quant_config", None)
        if quant_cfg is None:
            return cls.get_MoeBlock_class(None, quant_kwargs=quant_kwargs)

        rules = getattr(quant_cfg, "rules", [])
        for rule in rules:
            pattern = rule.get("regex")
            if not pattern or not re.search(pattern, checkpoint_prefix):
                continue

            method = getattr(rule, "type", None)
            if not method:
                method = quant_cfg.type
            rule_kwargs = rule.get("kwargs", {})
            method_kwargs = quant_kwargs.get(method, {})
            merged_kwargs = {**rule_kwargs, **method_kwargs}
            return cls.get_MoeBlock_class(
                method,
                quant_kwargs={method: merged_kwargs},
            )

        return cls.get_MoeBlock_class(
            None,
            quant_kwargs=quant_kwargs,
        )

    @classmethod
    def register_method(
        cls,
        name: Optional[str],
        implementation: Optional[Type[MoeBlock]] = None,
    ) -> None:
        """
        Register a new MoeBlock method.

        Arguments:
            name: Name of the MoeBlock method. None for non-quantized layer.
            implementation: Implementation class. If None, return a partial function as
                a decorator.
        """
        if implementation is None:
            return functools.partial(cls.register_method, name)
        cls._registry[name] = implementation
        return implementation


@MoeBlockRegistry.register_method(None)
class MoeBlock_without_quant(MoeBlock):
    pass


@MoeBlockRegistry.register_method("blockfp4")
class MoE_blockfp4(MoeBlock):
    """
    blockfp4 quantized Mixture-of-Experts (MoE) module.
    """

    def __init__(
        self,
        dim: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_shared_experts: int,
        n_activated_experts: int,
        moe_world_size: int,
        moe_rank: int,
        do_gather_output: bool,
        merge_gate_up: bool,
        dtype: torch.dtype,
        op_impl: str,
        gate: MoeGate,
        fuse_shared_experts: bool,
        shared_experts: nn.Module,
        checkpoint_prefix: str,
    ):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__(
            dim,
            moe_inter_dim,
            n_routed_experts,
            n_shared_experts,
            n_activated_experts,
            moe_world_size,
            moe_rank,
            do_gather_output,
            merge_gate_up,
            dtype,
            op_impl,
            gate,
            fuse_shared_experts,
            shared_experts,
            checkpoint_prefix,
            build_weight=False,
        )

        self.linear_dtype = torch.uint8

        quant_scale_stride = 16

        gate_up_proj_in_features = dim

        if merge_gate_up:
            out_features = moe_inter_dim * 2
            assert (
                out_features % self.tp_size == 0
            ), "gate_up_proj_out_features must be divisible by tp_size"
            local_out_features = out_features // self.tp_size
            scale_in_features = ceil_div(gate_up_proj_in_features, quant_scale_stride)
            scale_out_features = local_out_features
            self.gate_up_proj_weight = nn.Parameter(
                torch.empty(
                    self.group_size,
                    local_out_features,
                    gate_up_proj_in_features // 2,
                    dtype=self.linear_dtype,
                ),
                requires_grad=False,
            )
            self.gate_up_proj_weight_scale = nn.Parameter(
                torch.empty(
                    self.group_size,
                    scale_out_features,
                    scale_in_features,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            self.gate_up_proj_weight_scale_2 = nn.Parameter(
                torch.empty(
                    self.group_size,
                    2,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.gate_up_proj_input_scale = nn.Parameter(
                torch.empty(
                    self.group_size,
                    2,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        else:
            out_features = moe_inter_dim
            assert (
                out_features % self.tp_size == 0
            ), "gate_proj_out_features must be divisible by tp_size"
            local_out_features = out_features // self.tp_size
            scale_in_features = ceil_div(gate_up_proj_in_features, quant_scale_stride)
            scale_out_features = local_out_features
            self.gate_proj_weight = nn.Parameter(
                torch.empty(
                    self.group_size,
                    local_out_features,
                    gate_up_proj_in_features // 2,
                    dtype=self.linear_dtype,
                ),
                requires_grad=False,
            )
            self.gate_proj_weight_scale = nn.Parameter(
                torch.empty(
                    self.group_size,
                    scale_out_features,
                    scale_in_features,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            self.gate_proj_weight_scale_2 = nn.Parameter(
                torch.empty(
                    self.group_size,
                    1,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.gate_proj_input_scale = nn.Parameter(
                torch.empty(
                    self.group_size,
                    1,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.up_proj_weight = nn.Parameter(
                torch.empty(
                    self.group_size,
                    local_out_features,
                    gate_up_proj_in_features // 2,
                    dtype=self.linear_dtype,
                ),
                requires_grad=False,
            )
            self.up_proj_weight_scale = nn.Parameter(
                torch.empty(
                    self.group_size,
                    scale_out_features,
                    scale_in_features,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            self.up_proj_weight_scale_2 = nn.Parameter(
                torch.empty(
                    self.group_size,
                    1,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.up_proj_input_scale = nn.Parameter(
                torch.empty(
                    self.group_size,
                    1,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        assert (
            moe_inter_dim % self.tp_size == 0
        ), "down_proj_infeatures must be divisible by tp_size"
        down_proj_local_in_features = moe_inter_dim // self.tp_size
        down_proj_out_features = dim
        down_proj_scale_in_features = ceil_div(
            down_proj_local_in_features, quant_scale_stride
        )
        down_proj_scale_out_features = down_proj_out_features
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.group_size,
                down_proj_out_features,
                down_proj_local_in_features // 2,
                dtype=self.linear_dtype,
            ),
            requires_grad=False,
        )
        self.down_proj_weight_scale = nn.Parameter(
            torch.empty(
                self.group_size,
                down_proj_scale_out_features,
                down_proj_scale_in_features,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.down_proj_weight_scale_2 = nn.Parameter(
            torch.empty(
                self.group_size,
                1,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.down_proj_input_scale = nn.Parameter(
            torch.empty(
                self.group_size,
                1,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """

        shape = x.size()
        x = x.view(-1, self.dim)

        weights, indices = self.gate(x)

        if self.op_impl == "muxi_custom_kernel":
            y = self._compute_muxi_fused_experts(x, weights, indices)
        elif has_torch_npu:  # or use op_impl ?
            y = self._compute_npu_fused_experts(x, weights, indices)
        elif has_triton and self.merge_gate_up:
            if (
                parse_dtype(get_global_args().infer.raise_lower_bit_float_to).itemsize
                == 1
                or is_nvidia()
                or is_muxi()
            ):
                fused_soft_fp8 = (
                    parse_dtype(
                        get_global_args().infer.raise_lower_bit_float_to
                    ).itemsize
                    != 1
                )
                gate_up_proj_weight = self.gate_up_proj_weight
                gate_up_proj_scale = self.gate_up_proj_weight_scale
                gate_up_proj_scale_2 = self.gate_up_proj_weight_scale_2
                down_proj_weight = self.down_proj_weight
                down_proj_scale = self.down_proj_weight_scale
                down_proj_scale_2 = self.down_proj_weight_scale_2
            else:
                logger.warning(
                    f"Soft-fp8 fused gemm not implemented for {get_device_name()}, falling back to soft-fp8 conversion"
                )
                block_size = 128
                gate_up_proj_weight = weight_dequant_soft_fp8(
                    self.gate_up_proj_weight,
                    self.gate_up_proj_weight_scale,
                    block_size,
                )
                gate_up_proj_scale = None
                gate_up_proj_scale_2 = None
                down_proj_weight = weight_dequant_soft_fp8(
                    self.down_proj_weight,
                    self.down_proj_weight_scale,
                    block_size,
                )
                down_proj_scale = None
                down_proj_scale_2 = None
                fused_soft_fp8 = False

            if not self.fuse_shared_experts:
                y1 = self.shared_experts(x)

                y = fused_experts(
                    x,
                    gate_up_proj_weight,
                    down_proj_weight,
                    topk_weights=weights,
                    topk_ids=indices,
                    use_fp8_w8a8=False,
                    use_fp4_w4a8=True,
                    inplace=True,
                    global_num_experts=self.n_routed_experts + self.n_shared_experts,
                    expert_map=None,  # use when ep > 1
                    w1_scale=gate_up_proj_scale,
                    w2_scale=down_proj_scale,
                    w1w3_scale_2=gate_up_proj_scale_2,
                    w2_scale_2=down_proj_scale_2,
                    block_shape=[128, 128],
                    soft_fp8=fused_soft_fp8,
                )

                y += y1
            else:

                indice_shape = indices.shape
                new_indices = torch.empty(
                    (indice_shape[0], indice_shape[1] + 1),
                    dtype=indices.dtype,
                    device=indices.device,
                )

                new_weights = torch.empty(
                    (weights.shape[0], weights.shape[1] + 1),
                    dtype=weights.dtype,
                    device=weights.device,
                )

                chitu_backend.cuda_add_shared_experts(
                    new_weights,
                    new_indices,
                    weights,
                    indices,
                    self.n_routed_experts,
                    self.n_shared_experts,
                )
                del weights, indices
                y = fused_experts(
                    x,
                    gate_up_proj_weight,
                    down_proj_weight,
                    topk_weights=new_weights,
                    topk_ids=new_indices,
                    use_fp8_w8a8=False,
                    use_fp4_w4a8=True,
                    inplace=True,
                    global_num_experts=self.n_routed_experts + self.n_shared_experts,
                    expert_map=None,  # use when ep > 1
                    w1_scale=gate_up_proj_scale,
                    w2_scale=down_proj_scale,
                    w1w3_scale_2=gate_up_proj_scale_2,
                    w2_scale_2=down_proj_scale_2,
                    block_shape=[128, 128],
                    soft_fp8=fused_soft_fp8,
                )

            if self.tp_size > 1:
                torch.distributed.all_reduce(y, group=get_tp_group())
        else:
            y = torch.zeros_like(x)
            counts = torch.bincount(
                indices.flatten(), minlength=self.n_routed_experts
            ).tolist()

            xs = []
            for i in range(self.experts_start_idx, self.experts_end_idx):
                this_x = None
                if counts[i]:
                    idx, top = torch.where(indices == i)
                    this_x = x[idx]
                xs.append(this_x)
            if self.fuse_shared_experts:
                xs += [x] * self.n_fused_shared_experts

            if self.merge_gate_up:
                assert len(xs) == self.group_size
                gate_up_proj_outs = []
                for i in range(self.group_size):
                    out = None
                    if xs[i] is not None:
                        out = linear_block_fp4(
                            xs[i],
                            self.gate_up_proj_weight[i],
                            self.gate_up_proj_weight_scale[i],
                            self.gate_up_proj_weight_scale_2[i],
                            128,
                            None,
                        )
                        if self.do_gather_output and self.tp_size > 1:
                            out = self.gather_output(
                                out, self.tp_size, tp_group=self.tp_group
                            )
                    gate_up_proj_outs.append(out)
                act = [
                    (
                        silu_and_mul(gate_up_proj_out)
                        if gate_up_proj_out is not None
                        else None
                    )
                    for gate_up_proj_out in gate_up_proj_outs
                ]
            else:
                assert len(xs) == self.group_size
                gate_proj_outs = []
                up_proj_outs = []
                for i in range(self.group_size):
                    gate_proj_out = None
                    up_proj_out = None
                    if xs[i] is not None:
                        gate_proj_out = linear_block_fp4(
                            xs[i],
                            self.gate_proj_weight[i],
                            self.gate_proj_weight_scale[i],
                            self.gate_proj_weight_scale_2[i],
                            128,
                            None,
                        )
                        up_proj_out = linear_block_fp4(
                            xs[i],
                            self.up_proj_weight[i],
                            self.up_proj_weight_scale[i],
                            self.up_proj_weight_scale_2[i],
                            128,
                            None,
                        )
                        if self.do_gather_output and self.tp_size > 1:
                            gate_proj_out = self.gather_output(
                                gate_proj_out, self.tp_size, tp_group=self.tp_group
                            )
                            up_proj_out = self.gather_output(
                                up_proj_out, self.tp_size, tp_group=self.tp_group
                            )
                    gate_proj_outs.append(gate_proj_out)
                    up_proj_outs.append(up_proj_out)

                act = [
                    (
                        F.silu(gate_proj_out) * up_proj_out
                        if gate_proj_out is not None
                        else None
                    )
                    for gate_proj_out, up_proj_out in zip(gate_proj_outs, up_proj_outs)
                ]

            down_proj_outs = []
            for i in range(self.group_size):
                down_proj_out = None
                if act[i] is not None:
                    down_proj_out = linear_block_fp4(
                        act[i],
                        self.down_proj_weight[i],
                        self.down_proj_weight_scale[i],
                        self.down_proj_weight_scale_2[i],
                        128,
                        None,
                    )
                down_proj_outs.append(down_proj_out)

            for i in range(self.experts_start_idx, self.experts_end_idx):
                if counts[i]:
                    idx, top = torch.where(indices == i)
                    y[idx] += (
                        down_proj_outs[i - self.experts_start_idx]
                        * weights[idx, top, None]
                    )
            if self.fuse_shared_experts:
                for i in range(
                    self.experts_end_idx - self.experts_start_idx,
                    self.experts_end_idx
                    - self.experts_start_idx
                    + self.n_fused_shared_experts,
                ):
                    y += down_proj_outs[i]
            else:
                for i in range(self.n_shared_experts):
                    y += self.shared_experts(x)
            if self.tp_size > 1:
                dist.all_reduce(y, group=self.tp_group)
        return y.view(shape)

    def _compute_muxi_fused_experts(self, x, weights, indices):
        if self.fuse_shared_experts:
            raise NotImplementedError(
                "Fused shared experts is not supported for muxi_layout_kernels"
            )
        if not self.merge_gate_up:
            raise NotImplementedError(
                "muxi_layout_kernels for fused MoE requires merge_gate_up=True"
            )

        y = self.shared_experts(x)
        y1 = muxi_fused_experts(
            hidden_states=x,
            w1=self.gate_up_proj_weight,
            w2=self.down_proj_weight,
            topk_weights=weights,
            topk_ids=indices,
            inplace=True,
            w1_scale=self.gate_up_proj_weight_scale,
            w2_scale=self.down_proj_weight_scale,
            block_shape=[128, 128],
            soft_fp8=True,
        )
        y += y1
        torch.distributed.all_reduce(y, group=get_tp_group())
        return y


@MoeBlockRegistry.register_method("blockfp8")
class MoE_blockfp8(MoeBlock):
    """
    blockfp8 quantized Mixture-of-Experts (MoE) module.
    """

    def __init__(
        self,
        dim: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_shared_experts: int,
        n_activated_experts: int,
        moe_world_size: int,
        moe_rank: int,
        do_gather_output: bool,
        merge_gate_up: bool,
        dtype: torch.dtype,
        op_impl: str,
        gate: MoeGate,
        fuse_shared_experts: bool,
        shared_experts: nn.Module,
        checkpoint_prefix: str,
    ):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__(
            dim,
            moe_inter_dim,
            n_routed_experts,
            n_shared_experts,
            n_activated_experts,
            moe_world_size,
            moe_rank,
            do_gather_output,
            merge_gate_up,
            dtype,
            op_impl,
            gate,
            fuse_shared_experts,
            shared_experts,
            checkpoint_prefix,
        )
        gate_up_proj_in_features = dim
        block_size = 128

        if merge_gate_up:
            out_features = moe_inter_dim * 2
            assert (
                out_features % self.tp_size == 0
            ), "gate_up_proj_out_features must be divisible by tp_size"
            local_out_features = out_features // self.tp_size
            scale_out_features = (local_out_features + block_size - 1) // block_size
            scale_in_features = (
                gate_up_proj_in_features + block_size - 1
            ) // block_size
            self.gate_up_proj_scale = nn.Parameter(
                torch.empty(
                    self.group_size,
                    scale_out_features,
                    scale_in_features,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        else:
            out_features = moe_inter_dim
            assert (
                out_features % self.tp_size == 0
            ), "gate_proj_out_features must be divisible by tp_size"
            local_out_features = out_features // self.tp_size
            scale_out_features = (local_out_features + block_size - 1) // block_size
            scale_in_features = (
                gate_up_proj_in_features + block_size - 1
            ) // block_size
            self.gate_proj_scale = nn.Parameter(
                torch.empty(
                    self.group_size,
                    scale_out_features,
                    scale_in_features,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.up_proj_scale = nn.Parameter(
                torch.empty(
                    self.group_size,
                    scale_out_features,
                    scale_in_features,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        assert (
            moe_inter_dim % self.tp_size == 0
        ), "down_proj_infeatures must be divisible by tp_size"
        down_proj_local_in_features = moe_inter_dim // self.tp_size
        down_proj_out_features = dim
        down_proj_scale_out_features = (
            down_proj_out_features + block_size - 1
        ) // block_size
        down_proj_scale_in_features = (
            down_proj_local_in_features + block_size - 1
        ) // block_size
        self.down_proj_scale = nn.Parameter(
            torch.empty(
                self.group_size,
                down_proj_scale_out_features,
                down_proj_scale_in_features,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """

        shape = x.size()
        x = x.view(-1, self.dim)

        weights, indices = self.gate(x)

        if self.op_impl == "muxi_custom_kernel":
            y = self._compute_muxi_fused_experts(x, weights, indices)
        elif has_torch_npu:  # or use op_impl ?
            y = self._compute_npu_fused_experts(x, weights, indices)
        elif has_triton:
            assert self.merge_gate_up
            if (
                parse_dtype(get_global_args().infer.raise_lower_bit_float_to).itemsize
                == 1
                or is_nvidia()
                or is_muxi()
            ):
                fused_soft_fp8 = (
                    parse_dtype(
                        get_global_args().infer.raise_lower_bit_float_to
                    ).itemsize
                    != 1
                )
                gate_up_proj_weight = self.gate_up_proj_weight
                gate_up_proj_scale = self.gate_up_proj_scale
                down_proj_weight = self.down_proj_weight
                down_proj_scale = self.down_proj_scale
            else:
                logger.warning(
                    f"Soft-fp8 fused gemm not implemented for {get_device_name()}, falling back to soft-fp8 conversion"
                )
                block_size = 128
                gate_up_proj_weight = weight_dequant_soft_fp8(
                    self.gate_up_proj_weight,
                    self.gate_up_proj_scale,
                    block_size,
                )
                gate_up_proj_scale = None
                down_proj_weight = weight_dequant_soft_fp8(
                    self.down_proj_weight,
                    self.down_proj_scale,
                    block_size,
                )
                down_proj_scale = None
                fused_soft_fp8 = False

            if not self.fuse_shared_experts:
                y1 = self.shared_experts(x)

                y = fused_experts(
                    x,
                    gate_up_proj_weight,
                    down_proj_weight,
                    topk_weights=weights,
                    topk_ids=indices,
                    use_fp8_w8a8=True,
                    use_fp4_w4a8=False,
                    inplace=True,
                    global_num_experts=self.n_routed_experts + self.n_shared_experts,
                    expert_map=None,  # use when ep > 1
                    w1_scale=gate_up_proj_scale,
                    w2_scale=down_proj_scale,
                    w1w3_scale_2=None,
                    w2_scale_2=None,
                    block_shape=[128, 128],
                    soft_fp8=fused_soft_fp8,
                )

                y += y1
            else:

                indice_shape = indices.shape
                new_indices = torch.empty(
                    (indice_shape[0], indice_shape[1] + 1),
                    dtype=indices.dtype,
                    device=indices.device,
                )

                new_weights = torch.empty(
                    (weights.shape[0], weights.shape[1] + 1),
                    dtype=weights.dtype,
                    device=weights.device,
                )

                chitu_backend.cuda_add_shared_experts(
                    new_weights,
                    new_indices,
                    weights,
                    indices,
                    self.n_routed_experts,
                    self.n_shared_experts,
                )
                del weights, indices
                y = fused_experts(
                    x,
                    gate_up_proj_weight,
                    down_proj_weight,
                    topk_weights=new_weights,
                    topk_ids=new_indices,
                    use_fp8_w8a8=True,
                    use_fp4_w4a8=False,
                    inplace=True,
                    global_num_experts=self.n_routed_experts + self.n_shared_experts,
                    expert_map=None,  # use when ep > 1
                    w1_scale=gate_up_proj_scale,
                    w2_scale=down_proj_scale,
                    w1w3_scale_2=None,
                    w2_scale_2=None,
                    block_shape=[128, 128],
                    soft_fp8=fused_soft_fp8,
                )

            torch.distributed.all_reduce(y, group=get_tp_group())
        else:
            y = torch.zeros_like(x)
            counts = torch.bincount(
                indices.flatten(), minlength=self.n_routed_experts
            ).tolist()

            xs = []
            for i in range(self.experts_start_idx, self.experts_end_idx):
                this_x = None
                if counts[i]:
                    idx, top = torch.where(indices == i)
                    this_x = x[idx]
                xs.append(this_x)
            if self.fuse_shared_experts:
                xs += [x] * self.n_fused_shared_experts

            if self.merge_gate_up:
                assert len(xs) == self.group_size
                gate_up_proj_outs = []
                for i in range(self.group_size):
                    out = None
                    if xs[i] is not None:
                        out = linear_block_fp8(
                            xs[i],
                            self.gate_up_proj_weight[i],
                            self.gate_up_proj_scale[i],
                            None,
                            128,
                        )
                        if self.do_gather_output and self.tp_size > 1:
                            out = self.gather_output(
                                out, self.tp_size, tp_group=self.tp_group
                            )
                    gate_up_proj_outs.append(out)
                act = [
                    (
                        silu_and_mul(gate_up_proj_out)
                        if gate_up_proj_out is not None
                        else None
                    )
                    for gate_up_proj_out in gate_up_proj_outs
                ]
            else:
                assert len(xs) == self.group_size
                gate_proj_outs = []
                up_proj_outs = []
                for i in range(self.group_size):
                    gate_proj_out = None
                    up_proj_out = None
                    if xs[i] is not None:
                        gate_proj_out = linear_block_fp8(
                            xs[i],
                            self.gate_proj_weight[i],
                            self.gate_proj_scale[i],
                            None,
                            128,
                        )
                        up_proj_out = linear_block_fp8(
                            xs[i],
                            self.up_proj_weight[i],
                            self.up_proj_scale[i],
                            None,
                            128,
                        )
                        if self.do_gather_output and self.tp_size > 1:
                            gate_proj_out = self.gather_output(
                                gate_proj_out, self.tp_size, tp_group=self.tp_group
                            )
                            up_proj_out = self.gather_output(
                                up_proj_out, self.tp_size, tp_group=self.tp_group
                            )
                    gate_proj_outs.append(gate_proj_out)
                    up_proj_outs.append(up_proj_out)

                act = [
                    (
                        F.silu(gate_proj_out) * up_proj_out
                        if gate_proj_out is not None
                        else None
                    )
                    for gate_proj_out, up_proj_out in zip(gate_proj_outs, up_proj_outs)
                ]

            down_proj_outs = []
            for i in range(self.group_size):
                down_proj_out = None
                if act[i] is not None:
                    down_proj_out = linear_block_fp8(
                        act[i],
                        self.down_proj_weight[i],
                        self.down_proj_scale[i],
                        None,
                        128,
                    )
                down_proj_outs.append(down_proj_out)

            for i in range(self.experts_start_idx, self.experts_end_idx):
                if counts[i]:
                    idx, top = torch.where(indices == i)
                    y[idx] += (
                        down_proj_outs[i - self.experts_start_idx]
                        * weights[idx, top, None]
                    )
            if self.fuse_shared_experts:
                for i in range(
                    self.experts_end_idx - self.experts_start_idx,
                    self.experts_end_idx
                    - self.experts_start_idx
                    + self.n_fused_shared_experts,
                ):
                    y += down_proj_outs[i]
            else:
                for i in range(self.n_shared_experts):
                    y += self.shared_experts(x)
            if self.tp_size > 1:
                dist.all_reduce(y, group=self.tp_group)
        return y.view(shape)

    def _compute_muxi_fused_experts(self, x, weights, indices):
        if self.fuse_shared_experts:
            raise NotImplementedError(
                "Fused shared experts is not supported for muxi_layout_kernels"
            )
        if not self.merge_gate_up:
            raise NotImplementedError(
                "muxi_layout_kernels for fused MoE requires merge_gate_up=True"
            )

        y = self.shared_experts(x)
        y1 = muxi_fused_experts(
            hidden_states=x,
            w1=self.gate_up_proj_weight,
            w2=self.down_proj_weight,
            topk_weights=weights,
            topk_ids=indices,
            inplace=True,
            w1_scale=self.gate_up_proj_scale,
            w2_scale=self.down_proj_scale,
            block_shape=[128, 128],
            soft_fp8=True,
        )
        y += y1
        torch.distributed.all_reduce(y, group=get_tp_group())
        return y
