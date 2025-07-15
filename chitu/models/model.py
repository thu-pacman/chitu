import itertools
import os
import re
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
from chitu.layers.gate import fused_sigmoid_gate
from chitu.muxi_utils import has_tbsgemm, grouped_topk, tbsgemm
from chitu.ops import apply_rotary_pos_emb, rms_norm, topk_softmax

from chitu.distributed.parallel_state import get_tp_group, get_tp_size
from chitu.utils import (
    VarLens,
    compute_layer_dist_in_pipe,
    is_layer,
    try_import_opt_dep,
)
from chitu.quantization import QuantizedMoeExpertsBase, get_quant_from_checkpoint_prefix

torch_npu, has_torch_npu = try_import_opt_dep("torch_npu", "torch_npu")
chitu_backend, has_chitu_backend = try_import_opt_dep("chitu_backend", "chitu_backend")
triton, has_triton = try_import_opt_dep("triton", "triton")


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
        if quant == "blockfp8" or quant == "q4km":
            ret += ["scale"]
        elif quant == "blockfp4":
            ret += ["weight_scale", "weight_scale_2", "input_scale"]
        elif quant == "w4a8_per_token_per_channel_asymm":
            ret += ["qweight"]
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

        for name, param in checkpoint.items():
            quant = get_quant_from_checkpoint_prefix(
                name, self.params.quant_config.rules
            )
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
        new_state_dict = {}
        for key, value in state_dict.items():
            quant = get_quant_from_checkpoint_prefix(
                key, self.params.quant_config.rules
            )
            if quant == "blockfp4" and (
                key.endswith(".weight_scale_2") or key.endswith(".input_scale")
            ):
                new_state_dict[key] = value.view(1, 1)
            else:
                new_state_dict[key] = value
        return new_state_dict

    def anti_quant_fp8(self, scale1, scale2):
        """
        Pytorch native 反量化函数，用于将FP8 scale1 转换为BF16

        Args:
            scale1: 输入的FP8 scale1
            scale2: 输入的FP32 scale2

        """
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

    def _process_weight_scale_for_npu_fusion(self, param, scale_2):
        """处理NPU fusion mode下的权重scale数据

        Args:
            param: scale参数
            scale_2: 第二级scale参数

        Returns:
            处理后的scale参数
        """
        param.data = self.anti_quant_fp8(
            param.data.to(device="npu"), scale_2.data.to(device="npu")
        ).cpu()
        param.data = param.data.transpose(-2, -1).contiguous().transpose(-2, -1)
        return param

    def process_state_dict_for_blockfp4_after_chunk(self, state_dict):
        new_state_dict = {}
        for k in state_dict.keys():
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            if quant == "blockfp4":
                param = state_dict[k]
                # 处理scale参数
                if get_global_args().infer.npu_fusion_fp4 and k.endswith(
                    "weight_scale"
                ):
                    scale_name = k + "_2"
                    scale_2 = state_dict[scale_name]
                    param = self._process_weight_scale_for_npu_fusion(param, scale_2)
                new_state_dict[k] = param
            else:
                new_state_dict[k] = state_dict[k]
        return new_state_dict

    def process_state_dict_for_int4_after_chunk(self, state_dict):
        new_state_dict = {}
        for key in state_dict.keys():
            quant = get_quant_from_checkpoint_prefix(
                key, self.params.quant_config.rules
            )
            if quant == "w4a8_per_token_per_channel_asymm":
                param = state_dict[key]
                if param.dtype == torch.int8 and key.endswith("qweight"):
                    n, half_k = param.shape
                    k = half_k * 2

                    # Unpack from qserve format
                    # (https://github.com/mit-han-lab/deepcompressor/blob/main/deepcompressor/backend/qserve/utils.py#L18)
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
                new_state_dict[key] = param
            else:
                new_state_dict[key] = state_dict[key]
        return new_state_dict

    def process_state_dict_for_renaming_linear_layer(self, checkpoint, n_dense_layers):
        """
        重命名专家权重结构的函数以消除冗余的 gate,up,down 层
        参数格式示例：
        输入键：'layers.3.mlp.experts.gate_proj.weight'
        输出键：'layers.3.mlp.experts.gate_proj_weight'
        """
        new_checkpoint = {}
        for key in checkpoint:
            pattern_lists = [
                r"layers\.(\d+)\.mlp\.experts\.gate_up_proj\.([^.]+)",
                r"layers\.(\d+)\.mlp\.experts\.gate_proj\.([^.]+)",
                r"layers\.(\d+)\.mlp\.experts\.down_proj\.([^.]+)",
                r"layers\.(\d+)\.mlp\.experts\.up_proj\.([^.]+)",
            ]
            tensor_names = ["gate_up_proj", "gate_proj", "down_proj", "up_proj"]
            matched = False
            for tensor_name, pattern in zip(tensor_names, pattern_lists):
                match = re.fullmatch(pattern, key)
                if match:
                    layer_idx = int(match.group(1))
                    suffix = match.group(2)
                    if layer_idx < n_dense_layers and self.pp_stage == 0:
                        break
                    new_key = f"layers.{layer_idx}.mlp.experts.{tensor_name}_{suffix}"
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
        curr_freqs_cis = self.freqs_cis[self.cache.curr_varlens.position_ids]
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
        if (
            self.op_impl == "muxi_custom_kernel"
            and self.n_groups == 8
            and self.topk_groups == 4
            and self.topk == 8
            and self.weight.shape[0] == 256
            and self.score_func in ["sigmoid", "softmax"]
        ):
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
        elif self.score_func == "softmax" and is_nvidia():
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        weights, indices = self.gate(x.view(-1, x.shape[-1]))
        if self.shared_experts is not None:
            # Do this before `self.experts`, because `self.experts` may modify `x` in-place
            shared_y = self.shared_experts(x)
        y = self.experts(x, weights, indices)
        if self.shared_experts is not None:
            y += shared_y
        if get_tp_size() > 1:
            torch.distributed.all_reduce(y, group=get_tp_group().gpu_group)
        return y
