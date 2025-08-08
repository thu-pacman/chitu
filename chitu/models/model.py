# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
from logging import getLogger
from typing import Any, List, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from chitu.attn_backend import AttnBackend
from chitu.cache_manager import PagedKVCacheManager
from chitu.cuda_graph import make_dispatched_graphed_callables
from chitu.device_type import is_ascend, is_muxi, is_nvidia
from chitu.global_vars import get_global_args, get_timers
from chitu.muxi_utils import (
    has_tbsgemm,
    tbsgemm,
    Blockfp8LinearMuxiLayoutContigY,
    LinearMuxiLayoutContigY,
    LinearMuxiLayoutNativeY,
)
from chitu.ops import apply_rotary_pos_emb, rms_norm, moe_gate
from chitu.distributed.parallel_state import (
    get_tp_group,
    get_tp_size,
    get_ep_group,
    get_ep_size,
)
from chitu.distributed.moe_token_dispatcher import get_token_dispatcher
from chitu.utils import (
    compute_layer_dist_in_pipe,
    is_layer,
    try_import_platform_dep,
)
from chitu.quantization import (
    QuantizationRegistry,
    QuantizedMoeExpertsBase,
    get_quant_from_checkpoint_prefix,
    get_backend_from_checkpoint_prefix,
)

torch_npu, has_torch_npu = try_import_platform_dep("torch_npu")
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
triton, has_triton = try_import_platform_dep("triton")


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
            if self.weight.dtype != dtype:
                self.weight.data = self.weight.data.to(dtype)
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
            xq, xk, freqs_cis_cos, freqs_cis_sin, rotary_type="interleaved"
        )
        self.cache.finalize_cache_bylayer_prefill(
            xk, xv, self.cache.curr_req_ids, self.cache.curr_varlens, self.layer_id
        )
        output = self.attn_backend.prefill_ragged_qkvo(
            xq, xk, xv, varlens, causal=True
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
            xq, xk, freqs_cis_cos, freqs_cis_sin, rotary_type="interleaved"
        )

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        cache = self.cache.get_cache_decode(self.layer_id)
        cache_k = cache[0]
        cache_v = cache[1]
        cache_seqlens_excl_this_decode = self.cache.get_gpu_seq_lens_excl_this_decode()
        output = self.attn_backend.decode_dense_kv(
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
            xq, xk, freqs_cis_cos, freqs_cis_sin, rotary_type="interleaved"
        )

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        block_table = self.cache.get_gpu_block_table()
        cache_seqlens_excl_this_decode = self.cache.get_gpu_seq_lens_excl_this_decode()
        paged_k_cache, paged_v_cache = self.cache.get_paged_kv_cache(self.layer_id)
        output = self.attn_backend.decode_paged_kv(
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
        self.ep_group = get_ep_group()
        self.ep_size = self.ep_group.group_size
        self.pp_stage = self.rank // self.model_parallel_size
        self.pp_main_rank = (self.rank // model_parallel_size) * model_parallel_size
        self.pp_end_stage = (self.world_size - 1) // model_parallel_size

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

        if hasattr(self.params, "n_routed_experts"):
            n_routed_experts = self.params.n_routed_experts
        elif hasattr(self.params, "num_experts"):
            n_routed_experts = self.params.num_experts
        else:
            n_routed_experts = 0

        # if self.ep_size > 1:
        n_local_experts = n_routed_experts // self.ep_size
        remainder = n_routed_experts % self.ep_size
        self.experts_start_idx = self.ep_group.rank_in_group * n_local_experts
        self.experts_end_idx = self.experts_start_idx + n_local_experts
        if self.ep_group.is_last_rank:
            self.experts_end_idx += remainder

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
        if quant == "blockfp8" or quant == "q4km":
            ret += ["scale"]
        elif quant == "blockfp4":
            ret += ["weight_scale", "weight_scale_2", "input_scale"]
        elif quant == "w4a8_per_token_per_channel_asymm":
            ret += ["qweight"]
        elif quant == "mixq":
            ret += ["fp_weight"]
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
        elif quant == "simple_w8a8_muxi":
            ret += ["scale_channel"]
        elif quant == "w4a8_per_token_per_channel_asymm":
            ret += ["s1_scales", "s1_szeros"]
        elif quant == "mixq":
            ret += ["fp_idx", "weight_scale"]
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
        partial_checkpoint = {}

        cpl_names = self._get_tensor_column_parallel_layer_names()
        rpl_names = self._get_tensor_row_parallel_layer_names()

        enable_expert_parallel = get_ep_size() > 1

        for name, param in checkpoint.items():
            quant = get_quant_from_checkpoint_prefix(name)
            backend = get_backend_from_checkpoint_prefix(name)
            if backend == "cpuinfer":
                if rank == 0:
                    partial_checkpoint[name] = param
            elif enable_expert_parallel and ".experts." in name:
                partial_checkpoint[name] = param
            elif any(is_layer(s, name) for s in cpl_names):
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
        state_dict_keys = list(state_dict.keys())
        for key in state_dict_keys:
            value = state_dict[key]
            quant = get_quant_from_checkpoint_prefix(
                key, self.params.quant_config.rules
            )
            if quant == "blockfp4" and (
                key.endswith(".weight_scale_2") or key.endswith(".input_scale")
            ):
                state_dict[key] = value.view(1, 1)
            else:
                continue
        return state_dict

    def anti_quant_fp8(self, scale1, scale2):
        shape_w = scale1.shape
        shape_nw = list(shape_w)
        scale_fp8_to_32 = torch.tensor(0x7B80, dtype=torch.uint16)
        scale1 = scale1.to(torch.int16)
        new_weight = ((scale1 & 0x0080) << 8) | ((scale1 & 0x007F) << 4)
        new_weight = new_weight.view(torch.bfloat16) * scale_fp8_to_32.view(
            torch.bfloat16
        )
        new_weight = new_weight.to(torch.float32)
        if scale2.shape[-2] == 1:
            new_weight *= scale2
        else:
            new_weight[..., : shape_nw[-2] // 2, :] *= scale2[..., 0, :].unsqueeze(-1)
            new_weight[..., shape_nw[-2] // 2 :, :] *= scale2[..., 1, :].unsqueeze(-1)
        return new_weight.to(torch.bfloat16)

    def _process_fp4_weight_scale_for_npu_fusion(self, param, scale_2):
        param.data = self.anti_quant_fp8(
            param.data.to(device="npu"), scale_2.data.to(device="npu")
        ).cpu()
        param.data = param.data.transpose(-2, -1).contiguous().transpose(-2, -1)
        return param

    def process_state_dict_for_blockfp4_after_chunk(self, state_dict):
        state_dict_keys = list(state_dict.keys())
        for k in state_dict_keys:
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            if quant == "blockfp4":
                param = state_dict.pop(k)
                if get_global_args().infer.npu_fusion_fp4 and k.endswith(
                    "weight_scale"
                ):
                    scale_name = k + "_2"
                    scale_2 = state_dict[scale_name]
                    param = self._process_fp4_weight_scale_for_npu_fusion(
                        param, scale_2
                    )
                state_dict[k] = param
        return state_dict

    def process_state_dict_for_int4_after_chunk(self, state_dict):
        state_dict_keys = list(state_dict.keys())
        for key in state_dict_keys:
            quant = get_quant_from_checkpoint_prefix(
                key, self.params.quant_config.rules
            )
            if quant == "w4a8_per_token_per_channel_asymm":
                param = state_dict.pop(key)
                if param.dtype == torch.int8 and key.endswith("qweight"):
                    n, half_k = param.shape
                    k = half_k * 2

                    # Unpack from qserve format. See
                    # https://github.com/mit-han-lab/deepcompressor/blob/main/deepcompressor/backend/qserve/utils.py#L18
                    # for the format details
                    assert n % 32 == 0
                    assert k % 32 == 0
                    weight = param.data.view(
                        n // 32, k // 32, 1, 8, 4, 2, 2, 1, 4
                    ).view(torch.uint8)
                    weight = torch.stack([weight & 0x0F, weight >> 4], dim=0)
                    weight = (
                        weight.permute(1, 0, 7, 4, 8, 2, 3, 6, 5, 9)
                        .contiguous()
                        .view(n, k)
                    )

                    # Do our packing
                    assert k % 128 == 0
                    weight = (
                        weight.view(n, k // 128, 2, 64).permute(2, 0, 1, 3).contiguous()
                    )
                    weight = weight[0] + (weight[1] << 4)
                    param.data = weight.view(n, half_k)
                state_dict[key] = param

        return state_dict

    def process_state_dict_for_merging_qkv(self, checkpoint: Mapping[str, Any]):
        return checkpoint  # Inherit to preprocess. Leave it empty if not needed.

    def process_state_dict_for_merging_gate_up(self, checkpoint: Mapping[str, Any]):
        return checkpoint  # Inherit to preprocess. Leave it empty if not needed.

    def process_state_dict_for_merging_experts(self, checkpoint: Mapping[str, Any]):
        return checkpoint  # Inherit to preprocess. Leave it empty if not needed.

    def load_state_dict_parallel(
        self,
        state_dict: Mapping[str, Any],
        *args,
        skip_preprocess: bool = False,
        **kwargs,
    ):
        if not skip_preprocess:
            state_dict = self.process_state_dict_for_blockfp4_before_chunk(state_dict)

            # handle ep param
            if self.ep_size > 1:
                state_dict_keys = list(state_dict.keys())
                for key in state_dict_keys:
                    if (".experts." in key) and all(
                        f".experts.{x}." not in key
                        for x in range(self.experts_start_idx, self.experts_end_idx)
                    ):
                        state_dict.pop(key, None)

            if self.pipeline_exec:
                state_dict = self._chunk_checkpoint_for_pipeline_parallel(
                    state_dict, self.global_n_layers, self.pp_stage, self.pp_size
                )
            if self.tensor_exec:
                state_dict = self._chunk_checkpoint_for_tensor_parallel(
                    state_dict, self.rank % self.tp_size, self.tp_size
                )
        self.load_state_dict(
            state_dict, *args, skip_preprocess=skip_preprocess, **kwargs
        )

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        skip_preprocess: bool = False,
        **kwargs,
    ):
        if not skip_preprocess:
            state_dict = self.process_state_dict_for_merging_qkv(state_dict)
            state_dict = self.process_state_dict_for_merging_gate_up(state_dict)
            state_dict = self.process_state_dict_for_merging_experts(state_dict)
            state_dict = self.process_state_dict_for_blockfp4_after_chunk(state_dict)
            state_dict = self.process_state_dict_for_int4_after_chunk(state_dict)

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
        curr_freqs_cis = self.freqs_cis[
            self.cache.curr_varlens.position_ids_tensor_device
        ]
        return curr_freqs_cis.real.contiguous(), curr_freqs_cis.imag.contiguous()

    def prepare_freqs_cis_decode(self):
        curr_freqs_cis = self.freqs_cis[self.cache.get_gpu_seq_lens_excl_this_decode()]
        return curr_freqs_cis.real.contiguous(), curr_freqs_cis.imag.contiguous()

    @torch.inference_mode()
    def prefill_single_device(self, tokens):
        varlens = self.cache.curr_varlens
        freqs_cis_cos, freqs_cis_sin = self.prepare_freqs_cis_prefill(varlens)
        h = self._pre_layers(tokens)
        for it, layer in enumerate(self.layers):
            h = layer(h, freqs_cis_cos, freqs_cis_sin, varlens)
        tmp = varlens.prefix_lens_list[1:]
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
            tmp = varlens.prefix_lens_list[1:]
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
    def prefill(self, tokens):
        self.attn_backend.prepare_metadata_for_prefill(self.cache.curr_varlens)
        if self.pipeline_exec:
            return self.prefill_pipeline(tokens)
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
        infer_args = get_global_args().infer
        current_cuda_graph_enabled = self.use_cuda_graph and (
            infer_args.cache_type != "paged" or infer_args.num_blocks != -1
        )

        if (
            hasattr(self, "_last_cuda_graph_enabled")
            and self._last_cuda_graph_enabled != current_cuda_graph_enabled
        ):
            self.do_decode_callable = None
        self._last_cuda_graph_enabled = current_cuda_graph_enabled

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
                enable=current_cuda_graph_enabled,
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

    def forward(self, x):
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        if x.shape[0] == 0:
            return torch.empty(
                (0, self.topk),
                dtype=self.weight.dtype,
                device=self.weight.device,
            ), torch.empty((0, self.topk), dtype=torch.int32, device=self.weight.device)

        scores = F.linear(x, self.weight)
        indices, weights = moe_gate(
            scores,
            self.topk,
            self.n_groups,
            self.topk_groups,
            self.bias,
            self.score_func,
        )
        if self.norm_prob:
            weights /= weights.sum(dim=-1, keepdim=True)

        weights *= self.route_scale
        return weights.type_as(x), indices.to(torch.int32)


class ParallelMoeBlock(nn.Module):
    """
    Mixture-of-Experts (MoE) block.

    An object of this class includes MoeGate, MoeExperts, and maybe shared experts in a separated object
    (if fuse_shared_experts=False). This object maybe in parallel.

    Args:
        gate (MoeGate): The gating layer.
        experts (QuantizedMoeExpertsBase): The layer containing routed experts + fused shared experts
        non_fused_shared_experts (Optional[nn.Module]): Optional layer for shared experts if not fused.
    """

    def __init__(
        self,
        gate: MoeGate,
        experts: QuantizedMoeExpertsBase,
        non_fused_shared_experts: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.shared_experts = non_fused_shared_experts

        self.token_dispatcher = get_token_dispatcher()
        self.is_tp_mode = get_tp_size() > 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.shape  # [TODO] unify decode hidden states shape
        x = x.view(-1, x.shape[-1])
        weights, indices = self.gate(x)

        shared_y = None
        if self.shared_experts is not None:
            # Do this before `self.experts`, because `self.experts` may modify `x` in-place
            shared_y = self.shared_experts(x)

        if self.token_dispatcher is not None:
            x, weights, indices = self.token_dispatcher.token_permutation(
                x, weights, indices
            )

        y = self.experts(x, weights, indices)

        # Fuse allreduce to improve performance in TP mode
        if self.is_tp_mode:
            if shared_y is not None:
                y += shared_y
            if not self.token_dispatcher:
                torch.distributed.all_reduce(y, group=get_tp_group().gpu_group)

        if self.token_dispatcher is not None:
            y = self.token_dispatcher.token_unpermutation(y)

        if shared_y is not None and not self.is_tp_mode:
            y += shared_y
        return y.view(shape)


def get_linear_layout_native_y(
    op_impl: str,
    checkpoint_prefix: str,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
):
    if op_impl == "muxi_custom_kernel":
        args = get_global_args()
        quant_method = (
            None
            if not hasattr(args.models, "quant_config")
            else args.models.quant_config.type
        )
        if quant_method is None:
            assert (
                len(quant_kwargs.get(None, {})) == 0
            ), "quant_kwargs is not supported for muxi_custom_kernel"
            return LinearMuxiLayoutNativeY
        elif quant_method == "blockfp8":
            assert (
                len(quant_kwargs.get("blockfp8", {})) == 0
            ), "quant_kwargs is not supported for muxi_custom_kernel"
            # Blockfp8LinearMuxiLayoutNativeY is not implemented. Fall back.
            return Blockfp8LinearMuxiLayoutContigY
        else:
            raise NotImplementedError(
                f'Quantization method {quant_method} is not implemented for "muxi_custom_kernel"'
            )

    else:
        return QuantizationRegistry.get_quantized_linear_class_from_global_args(
            quant_kwargs=quant_kwargs, checkpoint_prefix=checkpoint_prefix
        )


def get_linear_layout_contig_y(
    op_impl: str,
    checkpoint_prefix: str,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
):
    if op_impl == "muxi_custom_kernel":
        args = get_global_args()
        quant_method = (
            None
            if not hasattr(args.models, "quant_config")
            else args.models.quant_config.type
        )
        # FIXME: get layer-specifc quant_method via get_quant_from_checkpoint_prefix

        if quant_method is None:
            assert (
                len(quant_kwargs.get(None, {})) == 0
            ), "quant_kwargs is not supported for muxi_custom_kernel"
            return LinearMuxiLayoutContigY
        elif quant_method == "blockfp8":
            assert (
                len(quant_kwargs.get("blockfp8", {})) == 0
            ), "quant_kwargs is not supported for muxi_custom_kernel"
            return Blockfp8LinearMuxiLayoutContigY
        else:
            raise NotImplementedError(
                f'Quantization method {quant_method} is not implemented for "muxi_custom_kernel"'
            )

    else:
        return QuantizationRegistry.get_quantized_linear_class_from_global_args(
            quant_kwargs=quant_kwargs, checkpoint_prefix=checkpoint_prefix
        )
