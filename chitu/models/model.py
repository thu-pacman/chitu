# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import functools
import operator
from logging import getLogger
import os
from collections import OrderedDict
from typing import Any, Mapping, Optional, Callable
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from chitu.task_type import TaskType
from chitu.device_type import is_muxi
from chitu.attn_backend import AttnBackend, NpuAttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.cache_manager import (
    KVCacheManagerBase,
    PagedKVCacheManager,
    DenseKVCacheManager,
)
from chitu.cuda_graph import (
    make_dispatched_graphed_callables,
    cuda_graph_safe_cached_property,
)
from chitu.device_type import is_ascend
from chitu.global_vars import get_global_args, get_timers
from chitu.muxi_utils import (
    Blockfp8LinearMuxiLayoutContigY,
    LinearMuxiLayoutContigY,
    LinearMuxiLayoutNativeY,
)
from chitu.ops import apply_rotary_pos_emb, rms_norm, moe_gate
from chitu.distributed.parallel_state import (
    get_tp_group,
    get_tp_size,
    get_etp_size,
    get_ep_group,
    get_ep_size,
    get_dp_group,
    get_dp_size,
    get_pp_group,
    get_pp_size,
)
from chitu.distributed.partition import compute_layer_dist_in_pp
from chitu.moe import get_moe_impl, MoEImplBase
from chitu.moe.batched_routed_activation import IndexedBatchedRoutedActivation
from chitu.moe.load_balancer import get_moe_load_planner
from chitu.utils import (
    is_layer,
    try_import_platform_dep,
    try_import_opt_dep,
    ceil_div,
    proportion_split,
)
from chitu.quantization import (
    QuantizationRegistry,
    QuantizedMoeExpertsBase,
    get_quant_from_checkpoint_prefix,
    get_quant_kwargs_from_checkpoint_prefix,
    get_backend_from_checkpoint_prefix,
)
from chitu.hybrid_device import CPUParameter
from chitu.static_tensor import StaticTensor

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
triton, has_triton = try_import_platform_dep("triton")
cinfer_ascendc, _ = try_import_opt_dep("cinfer_ascendc", "ascend_kernels")


logger = getLogger(__name__)


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype=None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(dim, dtype=dtype), requires_grad=False)

    def forward(self, x: torch.Tensor, compute_dtype=None):
        if compute_dtype is None:
            compute_dtype = torch.float32
        else:
            compute_dtype = self.weight.dtype
        return torch.nn.functional.layer_norm(
            x.to(compute_dtype),
            (self.dim,),
            self.weight.to(compute_dtype),
            self.bias.to(compute_dtype),
            self.eps,
        ).type_as(x)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6, dtype=None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype), requires_grad=False)

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

        return rms_norm(
            x,
            self.weight,
            eps=self.eps,
            out=out,
            compute_dtype=compute_dtype,
            impl=impl,
        )


class RMSNormBias(RMSNorm):
    def __init__(self, dim: int, eps: float = 1e-6, dtype=None, bias_dtype=None):
        super().__init__(dim=dim, eps=eps, dtype=dtype)
        self.bias = nn.Parameter(
            torch.zeros(self.dim, dtype=bias_dtype), requires_grad=False
        )

    def forward(
        self,
        x: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        compute_dtype=None,
        impl: str = "auto",
    ):
        return (
            super()
            .forward(x, out, compute_dtype if compute_dtype else torch.float32, impl)
            .add_(self.bias)
        )


class Attention(nn.Module):
    def __init__(self, layer_id, cache: KVCacheManagerBase, attn_backend):
        super().__init__()
        self.layer_id = layer_id
        self.cache = cache
        self.attn_backend = attn_backend

    def _run_linear(self, x):
        raise NotImplementedError

    def _run_output_linear(self, x):
        raise NotImplementedError

    def forward(self, x: torch.Tensor, freqs_cis: BatchedFreqsCis):
        bs_seq, _ = x.shape
        xq, xk, xv = self._run_linear(x)
        xq = xq.view(bs_seq, self.n_local_heads, self.head_dim)
        xk = xk.view(bs_seq, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs_seq, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_pos_emb(xq, xk, freqs_cis, rotary_type="interleaved")
        output = self.attn_backend(
            xq,
            self.cache.get_accessor(self.layer_id),
            xk,
            xv,
            seq_len_delta=self.cache.seq_len_delta,
            causal=True,
        ).view(bs_seq, -1)
        return self._run_output_linear(output)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        layer_id: int,
        args,
        cache_managers: dict[str, KVCacheManagerBase],
        attn_backend,
        op_impl,
    ):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.layer_id = layer_id
        self.timers = get_timers()

    def forward(self, x: torch.Tensor, freqs_cis: BatchedFreqsCis):
        raise NotImplementedError


class Transformer(nn.Module):
    def __init__(
        self,
        params,
        cache_managers: dict[str, KVCacheManagerBase],
        *,
        max_position_embeddings: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        attn_backend: AttnBackend,
        op_impl: str,
        **kvargs,
    ):
        super().__init__()
        self.cache_managers = cache_managers
        self.attn_backend = attn_backend
        self.op_impl = op_impl
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.device = torch.device(
            "cpu" if get_global_args().infer.op_impl == "cpu" else "cuda"
        )

        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_exec = pipeline_parallel_size > 1
        self.tensor_exec = tensor_parallel_size > 1

        self.tp_size = tensor_parallel_size
        self.pp_size = pipeline_parallel_size
        self.dp_size = get_dp_size()
        self.ep_group = get_ep_group()
        self.ep_size = self.ep_group.group_size
        self.pp_stage = get_pp_group().rank_in_group
        self.pp_main_rank = (self.rank // tensor_parallel_size) * tensor_parallel_size
        self.pp_end_stage = get_pp_size() - 1

        # `get_global_args()` can be a Hydra/OmegaConf object; force to plain int for type checkers.
        self.mtp_size = int(getattr(get_global_args().infer, "mtp_size", 1))

        self.params = params
        self.vocab_size = params.vocab_size
        self.global_n_layers = params.n_layers + (1 if self.mtp_size > 1 else 0)
        if self.pipeline_exec:
            num_layers_of_each_rank = compute_layer_dist_in_pp(
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
        self._init_layers(cache_managers, attn_backend=attn_backend, op_impl=op_impl)
        if not self.pipeline_exec or self.pp_stage == self.pipeline_parallel_size - 1:
            self._init_post_layers()

        with self.device:
            # precomputed freqs_cis has real data, so we can't put it on "meta" device
            self.precompute_freqs_cis(max_position_embeddings, self.device)

        self.do_decode_callable = None
        self.args = get_global_args()
        self.max_batch_size_per_dp = ceil_div(self.args.infer.max_reqs, get_dp_size())
        self.model_type = self.args.models.type
        self.use_cuda_graph = self.args.infer.use_cuda_graph

        self.moe_impl = get_moe_impl()

        if self.mtp_size > 1:
            self.token_offset_list = None
            self.mtp_token_list = None
            self.prefill_main_last_hidden_states = None
            self.last_hidden_states_4_postprocess = None
            self.main_last_hidden_states_static = StaticTensor(
                max_nelem=self.max_batch_size_per_dp * self.params.dim * self.mtp_size,
                dtype=torch.bfloat16,
                device=self.device,
            )
            self.mtp_last_hidden_states_static = StaticTensor(
                max_nelem=self.max_batch_size_per_dp * self.params.dim,
                dtype=torch.bfloat16,
                device=self.device,
            )
            self.main_last_hidden_states_up_to_date = False
            self.lhs_ = None

        dummy_input_shape = [0, self.params.dim]
        self.dummy_input = torch.empty(
            dummy_input_shape,
            dtype=torch.get_default_dtype(),
            device=self.device,
        )

        self.graph_dummy_output = torch.empty(
            [1],
            dtype=torch.get_default_dtype(),
            device=self.device,
        )

    def _get_tensor_column_parallel_layer_names(self) -> list[str]:
        raise NotImplementedError

    def _get_tensor_row_parallel_layer_names(self) -> list[str]:
        raise NotImplementedError

    def _get_pre_layer_prefixes(self) -> list[str]:
        raise NotImplementedError

    def _get_post_layer_prefixes(self) -> list[str]:
        raise NotImplementedError

    def _get_layer_i_prefixes(self, i: int) -> list[str]:
        raise NotImplementedError

    def _get_non_layer_prefix_mappings(self) -> list[tuple[str, str]]:
        raise NotImplementedError

    def _get_layer_i_prefix_mapping(self, i: int) -> tuple[str, str]:
        raise NotImplementedError

    def _get_2d_out_x_in_tensor_names(self, quant) -> list[str]:
        ret = ["weight"]
        if quant == "blockfp8" or quant == "q4km":
            ret += ["scale"]
        elif quant == "blockfp4" or quant == "blockfp4_merged":
            ret += ["weight_scale", "weight_scale_2", "input_scale"]
        elif quant == "w4a8_per_token_per_channel_asymm":
            ret += ["qweight"]
        elif quant == "w4a8_per_token_per_group_asymm":
            ret += ["qweight"]
        elif quant == "w4_g128_symm_a8":
            ret += ["weight"]
        elif quant == "mixq":
            ret += ["fp_weight"]
        elif quant == "ascend_w8a8_dynamic":
            ret += ["weight_scale", "weight_offset"]
        return ret

    def _get_2d_in_x_out_tensor_names(self, quant) -> list[str]:
        ret = []
        if quant == "autoawq":
            ret += ["qweight", "qzeros", "scales"]
        elif quant == "gptqmodel":
            ret += ["qweight", "qzeros", "scales"]
        return ret

    def _get_1d_in_tensor_names(self, quant) -> list[str]:
        ret = []
        if quant == "gptqmodel":
            ret += ["g_idx"]
        return ret

    def _get_1d_out_tensor_names(self, quant) -> list[str]:
        ret = ["bias"]
        if quant == "simple_w8a8":
            ret += ["scale_channel"]
        elif quant == "simple_w8a8_muxi":
            ret += ["scale_channel"]
        elif quant == "w4a8_per_token_per_channel_asymm":
            ret += ["s1_scales", "s1_szeros"]
        elif quant == "w4a8_per_token_per_group_asymm":
            ret += ["s1_scales", "s2_scales", "s2_zeros"]
        elif quant == "w4_g128_symm_a8_symm":
            ret += ["s2_scales", "s1_scales"]
        elif quant == "mixq":
            ret += ["fp_idx", "weight_scale"]
        elif quant == "ascend_w8a8":
            ret += ["input_scale", "input_offset", "quant_bias", "deq_scale"]
        return ret

    def _get_module_by_prefix(self, prefix: str) -> nn.Module | None:
        prefix = prefix[:-1] if prefix.endswith(".") else prefix
        module = self
        for part in prefix.split("."):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part, None)
        return module

    def load_state_dict_by_prefix(
        self, state_dict: dict[str, Any], prefix: str, skip_preprocess: bool = False
    ) -> nn.Module:
        state_dict = self.preprocess_state_dict_parallel(
            state_dict, skip_preprocess=skip_preprocess, is_layerwise=True
        )
        module_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                module_state_dict[key[len(prefix) :]] = value
        state_dict = module_state_dict
        module = self._get_module_by_prefix(prefix)
        assert module is not None, f"Module {prefix} not found"
        module.load_state_dict(state_dict, strict=True, assign=True)
        return module

    def _chunk_checkpoint_for_pipeline_parallel(
        self,
        checkpoint: dict[str, Any],
        num_layers: int,
        rank: int,
        pp_size: int,
    ):
        keys = checkpoint.keys()
        partial_checkpoint = {}

        num_layers_of_each_rank = compute_layer_dist_in_pp(num_layers, pp_size)
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
        checkpoint: dict[str, Any],
        rank: int,
        tp_size: int,
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
                if name.split(".")[-1] in self._get_1d_out_tensor_names(quant):
                    assert (
                        param.dim() == 1
                    ), f"{name} is expected to be 1D, but got {param.dim()}D"
                    if param.shape[-1] == 1:  # Broadcast
                        partial_checkpoint[name] = param
                    else:
                        if param.shape[-1] % tp_size != 0:
                            raise RuntimeError(
                                f"Tensor {name}'s last dim {param.shape[-1]} should be divisible by tp_size {tp_size}"
                            )
                        chunks = torch.chunk(param, tp_size, dim=-1)
                        partial_checkpoint[name] = chunks[rank]
                elif name.split(".")[-1] in self._get_1d_in_tensor_names(quant):
                    assert (
                        param.dim() == 1
                    ), f"{name} is expected to be 1D, but got {param.dim()}D"
                    if get_tp_group().rank_in_group == 0:
                        partial_checkpoint[name] = param
                elif name.split(".")[-1] in self._get_2d_out_x_in_tensor_names(quant):
                    assert (
                        param.dim() >= 2
                    ), f"{name} is expected to be >=2D, but got {param.dim()}D"
                    if param.shape[-2] == 1:  # Broadcast
                        partial_checkpoint[name] = param
                    else:
                        if param.shape[-2] % tp_size != 0:
                            raise RuntimeError(
                                f"Tensor {name}'s out dim {param.shape[-2]} should be divisible by tp_size {tp_size}"
                            )
                        chunks = torch.chunk(param, tp_size, dim=-2)
                        partial_checkpoint[name] = chunks[rank]
                elif name.split(".")[-1] in self._get_2d_in_x_out_tensor_names(quant):
                    assert (
                        param.dim() >= 2
                    ), f"{name} is expected to be >=2D, but got {param.dim()}D"
                    if param.shape[-1] == 1:  # Broadcast
                        partial_checkpoint[name] = param
                    else:
                        if param.shape[-1] % tp_size != 0:
                            raise RuntimeError(
                                f"Tensor {name}'s out dim {param.shape[-1]} should be divisible by tp_size {tp_size}"
                            )
                        chunks = torch.chunk(param, tp_size, dim=-1)
                        partial_checkpoint[name] = chunks[rank]
                else:
                    # FIXME: Support quant=llmint8 for TP
                    assert False, f"Illegal parallel tensor {name}"

            elif any(is_layer(s, name) for s in rpl_names):
                if name.split(".")[-1] in self._get_1d_in_tensor_names(quant):
                    assert (
                        param.dim() == 1
                    ), f"{name} is expected to be 1D, but got {param.dim()}D"
                    if param.shape[-1] == 1:  # Broadcast
                        partial_checkpoint[name] = param
                    else:
                        if param.shape[-1] % tp_size != 0:
                            raise RuntimeError(
                                f"Tensor {name}'s last dim {param.shape[-1]} should be divisible by tp_size {tp_size}"
                            )
                        chunks = torch.chunk(param, tp_size, dim=-1)
                        partial_checkpoint[name] = chunks[rank]
                elif name.split(".")[-1] in self._get_1d_out_tensor_names(quant):
                    assert (
                        param.dim() == 1
                    ), f"{name} is expected to be 1D, but got {param.dim()}D"
                    if name.split(".")[-1] == "bias":
                        if get_tp_group().rank_in_group != 0:
                            continue
                    partial_checkpoint[name] = param
                elif name.split(".")[-1] in self._get_2d_out_x_in_tensor_names(quant):
                    assert (
                        param.dim() >= 2
                    ), f"{name} is expected to be >=2D, but got {param.dim()}D"
                    if param.shape[-1] == 1:  # Broadcast
                        partial_checkpoint[name] = param
                    else:
                        if param.shape[-1] % tp_size != 0:
                            raise RuntimeError(
                                f"Tensor {name}'s in dim {param.shape[-1]} should be divisible by tp_size {tp_size}"
                            )
                        chunks = torch.chunk(param, tp_size, dim=-1)
                        partial_checkpoint[name] = chunks[rank]
                elif name.split(".")[-1] in self._get_2d_in_x_out_tensor_names(quant):
                    assert (
                        param.dim() >= 2
                    ), f"{name} is expected to be >=2D, but got {param.dim()}D"
                    if param.shape[-2] == 1:  # Broadcast
                        partial_checkpoint[name] = param
                    else:
                        assert param.shape[-2] % tp_size == 0
                        chunks = torch.chunk(param, tp_size, dim=-2)
                        partial_checkpoint[name] = chunks[rank]
                else:
                    # FIXME: Support quant=llmint8 for TP
                    assert False, f"Illegal parallel tensor {name}"

            else:
                partial_checkpoint[name] = param

        return partial_checkpoint

    def process_state_dict_for_blockfp4_before_chunk(self, state_dict: dict[str, Any]):
        # TODO: move it into utils
        BLOCKFP4_VARIANTS = ("blockfp4", "blockfp4_merged")

        state_dict_keys = list(state_dict.keys())
        for key in state_dict_keys:
            value = state_dict[key]
            quant = get_quant_from_checkpoint_prefix(
                key, self.params.quant_config.rules
            )
            if quant in BLOCKFP4_VARIANTS and (
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
        old_device = param.device
        param.data = self.anti_quant_fp8(
            param.data.to(device="npu"), scale_2.data.to(device="npu")
        ).to(old_device)
        param.data = param.data.transpose(-2, -1).contiguous().transpose(-2, -1)
        return param

    def process_state_dict_for_blockfp4_after_chunk(self, state_dict: dict[str, Any]):
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

    def process_state_dict_for_hygon_mixq_index_select(
        self, state_dict: dict[str, Any]
    ):
        hygon_mixq_kernels, has_hygon = try_import_platform_dep("sugon_mixQ4_kernels")
        if not has_hygon:
            return state_dict

        TILE = 512
        state_dict_keys = list(state_dict.keys())

        for k in state_dict_keys:
            # Only care about mixq quantized tensors
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            if quant != "mixq":
                continue

            if not k.endswith(".fp_idx"):
                continue

            # Pop the original fp_idx tensor, derive its prefix
            fp_idx = state_dict.pop(k)
            if "." in k:
                prefix, _name = k.rsplit(".", 1)
            else:
                prefix = ""

            # Sort indices then group them
            sorted_idx = torch.sort(fp_idx).values
            outliers_idx_grouped, outliers_idx_start = (
                hygon_mixq_kernels.group_outliers(sorted_idx, TILE)
            )

            state_dict[f"{prefix}.outliers_idx_grouped"] = outliers_idx_grouped
            state_dict[f"{prefix}.outliers_idx_start"] = outliers_idx_start
            state_dict[f"{prefix}.fp_idx"] = fp_idx

        return state_dict

    def process_state_dict_for_splitting_tensors(
        self,
        checkpoint: dict[str, Any],
        src_layer: str,
        *,
        tgt_layer_to_proportion: Optional[OrderedDict[str, int]] = None,
        equally_split_tgt_layers: Optional[list[str]] = None,
        dim_type: str | int = "out",
    ):
        """
        Split tensors in the checkpoint, useful for splitting merged Q/K/V or gate/up.

        All the tensors with named {prefix...}.{src_layer}.{tensor_name} will be split into
        {prefix...}.{tgt_layer}.{tensor_name}, where `prefix` (e.g. "layer.0") can be any
        strings joined by ".", `tgt_layer` (e.g. "q_proj") are set by `tgt_layer_to_proportion`
        or `equally_split_tgt_layers`, `src_layer` (e.g. "qkv_proj") is set by `src_layer`,
        and `tensor_name` (e.g. "weight") are the tensors in the layers according to the
        quantization config.

        Args:
            checkpoint: Checkpoint to process.
            src_layer: Source layer name.
            tgt_layer_to_proportion: Map from source layer names to their proportions, if
                you want to split with proportions. The proportions have no need to be
                exact sizes (for the sake of blocked quantizations). One of
                `tgt_layer_to_proportion` and `equally_split_tgt_layers` must be provided.
            equally_split_tgt_layers: Source layer names, if you want to split with equal
                sizes. One of `tgt_layer_to_proportion` and `equally_split_tgt_layers` must be
                provided.
            dim_type: Can be one of: 1) "in" means the input dimension of a linear layer,
                which dimension ID follows quantization config; 2) "out" means the output
                dimension of a linear layer, which dimension ID follows quantization config;
                3) int means the explicit dimension ID to split.
        """

        if tgt_layer_to_proportion is None and equally_split_tgt_layers is None:
            raise ValueError(
                "Either tgt_layer_to_proportion or equally_split_tgt_layers must be provided."
            )
        if tgt_layer_to_proportion is not None and equally_split_tgt_layers is not None:
            raise ValueError(
                "Only one of tgt_layer_to_proportion or equally_split_tgt_layers can be provided."
            )

        if tgt_layer_to_proportion is not None:
            tgt_layers = list(tgt_layer_to_proportion.keys())
        elif equally_split_tgt_layers is not None:
            tgt_layers = equally_split_tgt_layers
        else:
            assert False

        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)

            _2d_out_x_in_tensor_names = self._get_2d_out_x_in_tensor_names(quant)
            _2d_in_x_out_tensor_names = self._get_2d_in_x_out_tensor_names(quant)
            _1d_out_tensor_names = self._get_1d_out_tensor_names(quant)
            _1d_in_tensor_names = self._get_1d_in_tensor_names(quant)
            all_tensor_names = (
                _2d_out_x_in_tensor_names
                + _2d_in_x_out_tensor_names
                + _1d_out_tensor_names
                + _1d_in_tensor_names
            )

            if any(
                k.endswith(f".{src_layer}.{tensor_name}")
                for tensor_name in all_tensor_names
            ):
                tensor_name = k.split(".")[-1]

                if dim_type == "out":
                    if tensor_name in _2d_in_x_out_tensor_names + _1d_out_tensor_names:
                        split_dim = -1
                    elif tensor_name in _2d_out_x_in_tensor_names:
                        split_dim = -2
                    else:
                        continue
                elif dim_type == "in":
                    if tensor_name in _2d_out_x_in_tensor_names + _1d_in_tensor_names:
                        split_dim = -1
                    elif tensor_name in _2d_in_x_out_tensor_names:
                        split_dim = -2
                    else:
                        continue
                elif isinstance(dim_type, int):
                    split_dim = dim_type
                else:
                    raise ValueError(f"Invalid dim_type: {dim_type}")

                prefix = k[: -len(f".{src_layer}.{tensor_name}")]
                for tgt_layer in tgt_layers:
                    assert f"{prefix}.{tgt_layer}.{tensor_name}" not in checkpoint
                src_weight = checkpoint.pop(k)
                if src_weight.shape[split_dim] == 1:
                    checkpoint[k] = src_weight
                    continue
                if tgt_layer_to_proportion is not None:
                    tgt_weights = proportion_split(
                        src_weight,
                        list(tgt_layer_to_proportion.values()),
                        dim=split_dim,
                    )
                else:
                    tgt_weights = src_weight.chunk(len(tgt_layers), dim=split_dim)
                for tgt_layer, tgt_weight in zip(tgt_layers, tgt_weights):
                    checkpoint[f"{prefix}.{tgt_layer}.{tensor_name}"] = tgt_weight
        return checkpoint

    def process_state_dict_for_merging_tensors(
        self,
        checkpoint: dict[str, Any],
        tgt_layer: str,
        src_layers: list[str],
        *,
        enable_callback: Callable[[str], bool] = lambda _: True,
        dim_type: str | int = "out",
    ):
        """
        Merge tensors in the checkpoint, useful for merging Q/K/V or gate/up.

        All the tensors with named {prefix...}.{src_layer}.{tensor_name} will be merged into
        {prefix...}.{tgt_layer}.{tensor_name}, where `prefix` (e.g. "layer.0") can be any
        strings joined by ".", `src_layer` (e.g. "q_proj") are set by `src_layers`,
        `tgt_layer` (e.g. "qkv_proj") is set by `tgt_layer`, and `tensor_name` (e.g. "weight")
        are the tensors in the layers according to the quantization config.
        """

        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)

            _2d_out_x_in_tensor_names = self._get_2d_out_x_in_tensor_names(quant)
            _2d_in_x_out_tensor_names = self._get_2d_in_x_out_tensor_names(quant)
            _1d_out_tensor_names = self._get_1d_out_tensor_names(quant)
            _1d_in_tensor_names = self._get_1d_in_tensor_names(quant)
            all_tensor_names = (
                _2d_out_x_in_tensor_names
                + _2d_in_x_out_tensor_names
                + _1d_out_tensor_names
                + _1d_in_tensor_names
            )

            if not enable_callback(k):
                continue
            elif any(
                k.endswith(f".{src_layers[0]}.{tensor_name}")
                for tensor_name in all_tensor_names
            ):
                tensor_name = k.split(".")[-1]
                prefix = k[: -len(f".{src_layers[0]}.{tensor_name}")]
                src_weights = [
                    checkpoint.pop(f"{prefix}.{src_layer}.{tensor_name}")
                    for src_layer in src_layers
                ]

                if dim_type == "out":
                    if tensor_name in _2d_in_x_out_tensor_names + _1d_out_tensor_names:
                        cat_dim = -1
                    elif tensor_name in _2d_out_x_in_tensor_names:
                        cat_dim = -2
                    else:
                        continue
                elif dim_type == "in":
                    if tensor_name in _2d_out_x_in_tensor_names + _1d_in_tensor_names:
                        cat_dim = -1
                    elif tensor_name in _2d_in_x_out_tensor_names:
                        cat_dim = -2
                    else:
                        continue
                elif isinstance(dim_type, int):
                    cat_dim = dim_type
                else:
                    raise ValueError(f"Invalid dim_type: {dim_type}")

                # For MixQ quantized models, all the tensors share the same fp_idx
                merged_weight = (
                    src_weights[0]
                    if tensor_name == "fp_idx"
                    else torch.cat(src_weights, dim=cat_dim)
                )
                checkpoint[f"{prefix}.{tgt_layer}.{tensor_name}"] = merged_weight
        return checkpoint

    def process_state_dict_for_splitting_qkv(self, checkpoint: dict[str, Any]):
        return checkpoint  # Inherit to preprocess. Leave it empty if not needed.

    def process_state_dict_for_splitting_gate_up(self, checkpoint: dict[str, Any]):
        return checkpoint  # Inherit to preprocess. Leave it empty if not needed.

    def process_state_dict_for_merging_qkv(self, checkpoint: dict[str, Any]):
        return checkpoint  # Inherit to preprocess. Leave it empty if not needed.

    def process_state_dict_for_merging_gate_up(self, checkpoint: dict[str, Any]):
        return checkpoint  # Inherit to preprocess. Leave it empty if not needed.

    def process_state_dict_for_merging_experts(self, checkpoint: dict[str, Any]):
        return checkpoint  # Inherit to preprocess. Leave it empty if not needed.

    def process_state_dict_for_repeat_kv_head(self, checkpoint: dict[str, Any]):
        return checkpoint  # Inherit to preprocess. Leave it empty if not needed.

    def preprocess_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *,
        skip_preprocess: bool = False,
        is_layerwise: bool = False,
        replace: bool = True,
    ) -> dict[str, Any]:
        if not skip_preprocess:
            state_dict = self.process_state_dict_for_blockfp4_before_chunk(state_dict)
            # handle ep param
            if self.ep_size > 1:
                local_experts = [
                    self.moe_impl.load_balancer[layer_id].get_local_experts(
                        self.moe_impl.ep_group.rank_in_group
                    )
                    for layer_id in self.moe_impl.moe_layer_id_list
                ]
                state_dict_keys = list(state_dict.keys())

                for key in state_dict_keys:
                    key_split = key.split(".")
                    if key_split[0] != "layers":
                        continue
                    layer_id = int(key_split[1])
                    if (".experts." in key) and all(
                        f"{layer_id}.mlp.experts.{x}." not in key
                        for x in local_experts[layer_id - self.moe_impl.n_dense_layers]
                    ):
                        state_dict.pop(key, None)

            if self.pipeline_exec and not is_layerwise:
                state_dict = self._chunk_checkpoint_for_pipeline_parallel(
                    state_dict, self.global_n_layers, self.pp_stage, self.pp_size
                )
            if self.tensor_exec:
                # QKV and gate/up layers might already be merged in the checkpoint, but they should be split
                # for TP. After we process for TP, we merge them back.
                state_dict = self.process_state_dict_for_splitting_qkv(state_dict)
                state_dict = self.process_state_dict_for_splitting_gate_up(state_dict)

                # Repeat kv_head weights in case tensor_parallel_size > n_kv_heads
                state_dict = self.process_state_dict_for_repeat_kv_head(state_dict)

                state_dict = self._chunk_checkpoint_for_tensor_parallel(
                    state_dict, self.rank % self.tp_size, self.tp_size
                )

        return self.preprocess_state_dict(state_dict, skip_preprocess=skip_preprocess)

    def preprocess_state_dict(
        self, state_dict: dict[str, Any], *, skip_preprocess: bool = False
    ) -> dict[str, Any]:
        # TODO: Move `state_dict` to GPU and preprocess on GPU if there is no `CPUParameter`s
        # Problems:
        # - Processing on GPU laeds to sever memory fragmentation (13.44 GiB fragements in 94.93
        #   GiB allocated memory). Disabling torch allocator with `PYTORCH_NO_CUDA_MEMORY_CACHING=1`
        #   works but may lead to too much performance degradation.

        if not skip_preprocess:
            state_dict = self.process_state_dict_for_merging_qkv(state_dict)
            state_dict = self.process_state_dict_for_merging_gate_up(state_dict)
            state_dict = self.process_state_dict_for_merging_experts(state_dict)
            state_dict = self.process_state_dict_for_blockfp4_after_chunk(state_dict)
            state_dict = self.process_state_dict_for_hygon_mixq_index_select(state_dict)

        # Check inconsistent dtype
        keep_dtype_in_checkpoint = get_global_args().keep_dtype_in_checkpoint
        for name, param in self.named_parameters():
            if name in state_dict and param.dtype != state_dict[name].dtype:
                if keep_dtype_in_checkpoint:
                    logger.info(
                        f"Parameter {name} has inconsistent dtype in the checkpoint "
                        f"({state_dict[name].dtype}) and the model ({param.dtype}), "
                        f"using the dtype in the checkpoint. Set `keep_dtype_in_checkpoint=False` "
                        f"when starting chitu if you want to use the dtype in the model."
                    )
                else:
                    logger.info(
                        f"Parameter {name} has inconsistent dtype in the checkpoint "
                        f"({state_dict[name].dtype}) and the model ({param.dtype}), "
                        f"converting the checkpoint dtype to the model dtype. Set "
                        f"`keep_dtype_in_checkpoint=True` when starting chitu if you "
                        f"want to use the dtype in the checkpoint."
                    )
                    state_dict[name] = state_dict[name].to(param.dtype)

        for k in state_dict:
            if isinstance(self.get_parameter(k), CPUParameter):
                state_dict[k] = CPUParameter(state_dict[k], requires_grad=False)
            else:
                # Work around a bug on torch<2.2.2 that creates requires_grad=True inside
                # `super().load_state_dict`:
                # See https://github.com/pytorch/pytorch/pull/121157.
                state_dict[k] = torch.nn.Parameter(state_dict[k], requires_grad=False)

        return state_dict

    def load_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *args,
        skip_preprocess: bool = False,
        replace: bool = True,
        **kwargs,
    ):
        state_dict = self.preprocess_state_dict_parallel(
            state_dict, skip_preprocess=skip_preprocess, replace=replace
        )
        super().load_state_dict(state_dict, *args, **kwargs)

    def _init_pre_layers(self):
        raise NotImplementedError

    def _init_layers(
        self, cache_managers: dict[str, KVCacheManagerBase], attn_backend, op_impl
    ):
        raise NotImplementedError

    def _init_post_layers(self):
        raise NotImplementedError

    def _pre_layers(self, h, **args):
        raise NotImplementedError

    def _pre_layers_mtp(self, h, **args):
        raise NotImplementedError

    def _post_layers(self, h):
        """NOTE: _post_layers is assumed to be a token-wise computation"""
        raise NotImplementedError

    def _get_prefill_previous_hidden_states(self, h):
        raise NotImplementedError

    def _post_layers_mtp(self, h):
        raise NotImplementedError

    def precompute_freqs_cis(self, max_position_embeddings, device):
        dim = self.params.dim // self.params.n_heads
        freqs = 1.0 / (
            self.params.rope_theta
            ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
        )
        t = torch.arange(
            max_position_embeddings * 2, device=device, dtype=torch.float32
        )
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        rotary_dtype = (
            torch.float32
            if get_global_args().use_float32_rotary
            else torch.get_default_dtype()
        )
        self.freqs_cis_real = freqs_cis.real.contiguous().to(rotary_dtype)
        self.freqs_cis_imag = freqs_cis.imag.contiguous().to(rotary_dtype)

    def prepare_freqs_cis(self) -> BatchedFreqsCis:
        return BatchedFreqsCis(
            self.freqs_cis_real[
                self.cache_managers[
                    "main"
                ].seq_len_delta.delta_position_ids_tensor_device
            ],
            self.freqs_cis_imag[
                self.cache_managers[
                    "main"
                ].seq_len_delta.delta_position_ids_tensor_device
            ],
        )

    def prepare_freqs_cis_mtp(self) -> BatchedFreqsCis:
        return BatchedFreqsCis(
            self.freqs_cis_real[
                self.cache_managers[
                    "main"
                ].mtp_seq_len_delta.delta_position_ids_tensor_device
            ],
            self.freqs_cis_imag[
                self.cache_managers[
                    "main"
                ].mtp_seq_len_delta.delta_position_ids_tensor_device
            ],
        )

    @cuda_graph_safe_cached_property(
        "main_last_hidden_states_static", "main_last_hidden_states_up_to_date"
    )
    def set_main_last_hidden_states_static(self):
        return self.lhs_

    @torch.inference_mode()
    def prefill_no_pipeline(
        self, tokens, output_token_offsets: torch.Tensor, **args
    ) -> torch.Tensor:
        freqs_cis = self.prepare_freqs_cis()
        if self.moe_impl is not None:
            self.moe_impl.prepare(TaskType.Prefill, int(tokens.shape[0]))
        h = self._pre_layers(tokens, **args)
        if self.mtp_size > 1:
            for mgr in self.cache_managers.values():
                mgr.seq_len_delta.is_decode_stage = False
            self.token_offset_list = None
            self.mtp_token_list = None
            for it, layer in enumerate(self.layers[0:-1]):
                h = layer(h, freqs_cis, False)
            prefill_previous_hidden_states = self._get_prefill_previous_hidden_states(h)
            h_mtp = self._pre_layers_mtp(tokens, **args)
            h_mtp[
                self.cache_managers[
                    "main"
                ].mtp_seq_len_delta.delta_position_ids_tensor_device
                == 0
            ] = 0
            h_mtp = self.layers[-1](
                h_mtp, freqs_cis, prefill_previous_hidden_states, False
            )
            self.last_hidden_states_4_postprocess = (
                self.prefill_main_last_hidden_states[output_token_offsets]
            )
        else:
            for it, layer in enumerate(self.layers):
                h = layer(h, freqs_cis)
        # Exec post layers AFTER cutting the last token off
        h = h[output_token_offsets]
        h = self._post_layers(h)
        h = h.float()
        return h

    @torch.inference_mode()
    def decode_no_pipeline(self, tokens, freqs_cis: BatchedFreqsCis):
        h = self._pre_layers(tokens)
        if not self.mtp_size > 1:
            for it, layer in enumerate(self.layers):
                h = layer(h, freqs_cis)
        else:
            for it, layer in enumerate(self.layers[0:-1]):
                h = layer(h, freqs_cis, False)
            self.lhs_ = self.norm(h, compute_dtype=h.dtype)
            self.set_main_last_hidden_states_static
        h = self._post_layers(h)
        h = h.float()
        return h

    @torch.inference_mode()
    def mtp_decode_no_pipeline(self, tokens, freqs_cis: BatchedFreqsCis):
        h = self._pre_layers_mtp(tokens)
        h = self.layers[-1](
            h, freqs_cis, self.mtp_last_hidden_states_static.get(), True
        )
        self.mtp_last_hidden_states_static.set(h)
        h = self._post_layers_mtp(h)
        h = h.float()
        return h

    @torch.inference_mode()
    def mtp_decode_no_pipeline_total(
        self,
        tokens,
        func,
        key,
        func_mtp,
        key_mtp,
        extra_inputs: tuple[torch.Tensor, ...] = (),
        extra_inputs_mtp: tuple[torch.Tensor, ...] = (),
    ):
        token_list = []
        token_list.append(tokens)
        for i in range(0, self.mtp_size):
            for mgr in self.cache_managers.values():
                mgr.prepare_mtp_cache_decode(i)
                if isinstance(mgr, PagedKVCacheManager):
                    mgr.update_page_offs()
            self.prepare_decoding_attn_mtp()
            h = func_mtp(key_mtp, tokens, *extra_inputs_mtp)
            tokens = torch.argmax(h, dim=-1)
            token_list.append(tokens)
        for mgr in self.cache_managers.values():
            if isinstance(mgr, PagedKVCacheManager):
                mgr.update_page_offs()
        self.main_last_hidden_states_up_to_date = False
        tokens_proposal = torch.stack(token_list[:-1], dim=1).view(-1)
        if self.use_cuda_graph:
            for mgr in self.cache_managers.values():
                mgr.seq_len_delta.is_decode_stage = True
            self.prepare_decoding_attn()
        else:
            self.attn_backend.prepare_metadata_for_prefill(
                self.cache_managers["main"].seq_len_delta
            )
            if (
                self.moe_impl is not None
                and self.moe_impl.ep_size > 1
                and self.moe_impl.decode_token_dispatcher_impl == "allgather"
            ):
                self.moe_impl.prepare(TaskType.Decode, tokens_proposal.shape[0])
        h = func(key, tokens_proposal, *extra_inputs)
        tokens_proposal = tokens_proposal.view(-1, self.mtp_size)
        tokens_verify = torch.argmax(h, dim=-1).view(-1, self.mtp_size)
        h = h.view(-1, self.mtp_size, h.shape[-1])
        mlh_ = self.main_last_hidden_states_static.get()
        mtp_last_hidden_states = mlh_.view(-1, self.mtp_size, mlh_.shape[-1])
        assert h.shape[0] == mtp_last_hidden_states.shape[0]
        matches = tokens_proposal[:, 1:] == tokens_verify[:, :-1]

        all_accept = matches.all(dim=1)
        first_mismatch_idx = torch.argmax((~matches).int(), dim=1)
        accept_idx = torch.where(
            all_accept,
            torch.full_like(first_mismatch_idx, self.mtp_size - 1),
            first_mismatch_idx,
        )

        batch_indices = torch.arange(tokens_proposal.shape[0], device=h.device)
        h_selected = h[batch_indices, accept_idx]
        mtp_selected = mtp_last_hidden_states[batch_indices, accept_idx]
        token_offset = (accept_idx + 1).tolist()
        tokens_proposal_accepted = [
            tokens_proposal[i, 1 : accept_idx[i] + 1].tolist()
            for i in range(tokens_proposal.size(0))
        ]

        for mgr in self.cache_managers.values():
            mgr.update_mtp_cache_decode(token_offset)
        self.token_offset_list = token_offset
        self.mtp_token_list = tokens_proposal_accepted
        self.last_hidden_states_4_postprocess = mtp_selected
        return h_selected

    @torch.inference_mode()
    def prefill_pipeline(
        self, tokens, output_token_offsets: torch.Tensor, **args
    ) -> torch.Tensor:
        freqs_cis = self.prepare_freqs_cis()

        # start of model
        if self.pp_stage == 0:
            h = self._pre_layers(tokens, **args)
        else:
            h = tokens

        # Ensure MoE impl is primed before layer execution in prefill.
        if self.moe_impl is not None:
            self.moe_impl.prepare(TaskType.Prefill, int(tokens.shape[0]))

        # layers
        for it, layer in enumerate(self.layers):
            h = layer(h, freqs_cis)

        # end of model
        if self.pp_stage == self.pp_end_stage:
            # Exec post layers AFTER cutting the last token off
            h = h[output_token_offsets]
            h = self._post_layers(h)
            h = h.float()

        return h

    @torch.inference_mode()
    def decode_pipeline(self, tokens, freqs_cis: BatchedFreqsCis):
        if self.pp_stage == 0:
            h = self._pre_layers(tokens)
        else:
            h = tokens
        for it, layer in enumerate(self.layers):
            h = layer(h, freqs_cis)
        if self.pp_stage == self.pp_end_stage:
            h = self._post_layers(h)
            h = h.float()

        return h

    @torch.inference_mode()
    def empty_prefill(self) -> torch.Tensor:
        if self.ep_size > 1:
            for it, layer in enumerate(self.layers):
                if it < self.moe_impl.n_dense_layers:
                    continue
                layer.mlp(self.dummy_input)
        return None

    @torch.inference_mode()
    def empty_decode(self):
        layer_main = self.layers[0:-1] if self.mtp_size > 1 else self.layers
        for it, layer in enumerate(layer_main):
            if it < self.moe_impl.n_dense_layers:
                continue
            layer.mlp(self.dummy_input)
        return self.graph_dummy_output

    @torch.inference_mode()
    def empty_mtp_decode(self):
        self.layers[-1].mlp(self.dummy_input)
        return self.graph_dummy_output

    @torch.inference_mode()
    def empty_mtp_decode_total(self, func, key, func_mtp, key_mtp):
        for i in range(0, self.mtp_size):
            func_mtp(key_mtp)

        if (
            self.moe_impl is not None
            and self.moe_impl.decode_token_dispatcher_impl == "allgather"
        ):
            self.moe_impl.prepare(TaskType.Decode, self.dummy_input.shape[0])

        return func(key)

    @torch.inference_mode()
    def prefill(
        self, tokens, output_token_offsets: torch.Tensor, **args
    ) -> torch.Tensor:
        if tokens.shape[0] == 0:
            return self.empty_prefill()

        self.attn_backend.prepare_metadata_for_prefill(
            self.cache_managers["main"].seq_len_delta
        )
        if self.pipeline_exec:
            return self.prefill_pipeline(tokens, output_token_offsets, **args)
        else:
            return self.prefill_no_pipeline(tokens, output_token_offsets, **args)

    def prepare_decoding_attn(self):
        self.attn_backend.prepare_metadata_for_decode(
            self.cache_managers["main"].seq_len_delta,
            self.cache_managers["main"].get_gpu_block_table(),
            self.cache_managers["main"].get_block_size(),
        )

    def prepare_decoding_attn_mtp(self):
        self.attn_backend.prepare_metadata_for_decode(
            self.cache_managers["main"].mtp_seq_len_delta,
            self.cache_managers["main"].get_gpu_block_table(),
            self.cache_managers["main"].get_block_size(),
        )

    def _decode_graph_extra_inputs(
        self, tokens: torch.Tensor, batch_size: int
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
        """
        Optional extra tensor inputs for CUDA-graphed decode.

        Why this exists:
        - In CUDA graph replay, Python-side logic in `prepare_freqs_cis()` is NOT re-executed.
        - If a model needs per-step dynamic values (e.g. per-request RoPE deltas) to affect freqs,
          it must be provided as a tensor argument to the graphed callable so StaticTensor can update it.

        Returns:
            (extra_inputs, extra_inputs_max_nelem)
            - extra_inputs: tuple of tensors passed to the graphed callable after `tokens`
            - extra_inputs_max_nelem: matching tuple of maximum nelem for StaticTensor allocation
        """
        return (), ()

    def _decode_graph_extra_inputs_mtp(
        self, tokens: torch.Tensor, batch_size: int
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
        """
        Optional extra tensor inputs for CUDA-graphed MTP decode.

        Same motivation as `_decode_graph_extra_inputs()`, but for the MTP decode callable.
        """
        return (), ()

    def _prepare_freqs_cis_for_decode(
        self, *extra_inputs: torch.Tensor
    ) -> BatchedFreqsCis:
        """
        Build freqs_cis for decode inside the CUDA-graphed callable.

        Default behavior: ignore extra inputs and defer to `prepare_freqs_cis()`.
        Models may override this to incorporate extra tensor inputs (e.g. RoPE deltas).
        """
        return self.prepare_freqs_cis()

    def _prepare_freqs_cis_for_decode_mtp(
        self, *extra_inputs: torch.Tensor
    ) -> BatchedFreqsCis:
        """
        Build freqs_cis for MTP decode inside the CUDA-graphed callable.

        Default behavior: ignore extra inputs and defer to `prepare_freqs_cis_mtp()`.
        Models may override this to incorporate extra tensor inputs (e.g. RoPE deltas).
        """
        return self.prepare_freqs_cis_mtp()

    @torch.inference_mode()
    def decode(self, tokens, batch_size):
        if isinstance(self.cache_managers["main"], DenseKVCacheManager):
            key = (batch_size, self.cache_managers["main"].get_start_and_end_idx()[0])
        elif isinstance(self.cache_managers["main"], PagedKVCacheManager):
            key = (batch_size,)
        else:
            assert False

        if batch_size != 0 and not self.mtp_size > 1:
            self.prepare_decoding_attn()

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

        extra_inputs, extra_inputs_max_nelem = self._decode_graph_extra_inputs(
            tokens, batch_size
        )
        if self.mtp_size > 1:
            extra_inputs_mtp, extra_inputs_mtp_max_nelem = (
                self._decode_graph_extra_inputs_mtp(tokens, batch_size)
            )
        else:
            extra_inputs_mtp, extra_inputs_mtp_max_nelem = (), ()

        if self.do_decode_callable is None:

            before_replay_callback = None

            if is_ascend() and not (
                infer_args.cache_type == "skew"
                and NpuAttnBackend.should_use_attn_from_cinfer_ascendc(
                    self.args.models.type, infer_args.max_reqs
                )
            ):
                before_replay_callback = lambda graph: graph.update(
                    cpu_update_input=[
                        {
                            "actual_seq_lengths_kv": self.cache_managers[
                                "main"
                            ].seq_len_delta.new.lens_list
                        }
                    ]
                )

            def numel_per_seq(batch_size, x):
                if batch_size > 0:
                    return x.numel() // batch_size
                else:
                    assert x.shape[0] == 0
                    return functools.reduce(operator.mul, x.shape[1:], 1)

            tokens_max_nelem = self.max_batch_size_per_dp * numel_per_seq(
                batch_size, tokens
            )
            output_max_nelem_callback = (
                lambda key, out: numel_per_seq(key[0], out) * self.max_batch_size_per_dp
            )

            @make_dispatched_graphed_callables(
                args_max_nelem=(
                    self.mtp_size * tokens_max_nelem,
                    *extra_inputs_max_nelem,
                ),
                kwargs_max_nelem={},
                output_max_nelem_callback=output_max_nelem_callback,
                before_capture_callback=lambda: self.prepare_decoding_attn(),
                before_replay_callback=before_replay_callback,
                enable=current_cuda_graph_enabled,
            )
            def do_decode(tokens, *extra_inputs):
                freqs_cis = self._prepare_freqs_cis_for_decode(*extra_inputs)
                if self.pipeline_exec:
                    return self.decode_pipeline(tokens, freqs_cis)
                else:
                    return self.decode_no_pipeline(tokens, freqs_cis)

            self.do_decode_callable = do_decode

            if self.mtp_size > 1:

                @make_dispatched_graphed_callables(
                    args_max_nelem=(tokens_max_nelem, *extra_inputs_mtp_max_nelem),
                    kwargs_max_nelem={},
                    output_max_nelem_callback=output_max_nelem_callback,
                    before_capture_callback=lambda: self.prepare_decoding_attn_mtp(),
                    before_replay_callback=before_replay_callback,
                    enable=current_cuda_graph_enabled,
                )
                def do_decode_mtp(tokens, *extra_inputs_mtp):
                    freqs_cis = self._prepare_freqs_cis_for_decode_mtp(
                        *extra_inputs_mtp
                    )
                    return self.mtp_decode_no_pipeline(tokens, freqs_cis)

                self.do_decode_callable_mtp = do_decode_mtp

            if self.ep_size > 1:

                @make_dispatched_graphed_callables(
                    args_max_nelem=(),
                    kwargs_max_nelem={},
                    output_max_nelem_callback=lambda key, n: 1,
                    before_replay_callback=None,
                    # empty decode 仅用于 EP sync，没有真实 token
                    # 在 empty decode 上capture graph 会生成 zero-size buffer，
                    # 后续非空 replay 会失败，因此关闭graphed，经过测试发现这部分对性能影响很小
                    enable=False,
                )
                def do_empty_decode():
                    return self.empty_decode()

                self.do_empty_decode_callable = do_empty_decode

                if self.mtp_size > 1:

                    @make_dispatched_graphed_callables(
                        args_max_nelem=(),
                        kwargs_max_nelem={},
                        output_max_nelem_callback=lambda key, n: 1,
                        before_replay_callback=None,
                        # empty MTP decode 仅用于 EP sync，没有真实 token
                        # 在 empty decode 上捕获 CUDA graph 会生成 zero-size buffer，
                        # 后续非空 replay 会失败，因此保持 non-graphed。
                        enable=False,
                    )
                    def do_empty_decode_mtp():
                        return self.empty_mtp_decode()

                    self.do_empty_decode_callable_mtp = do_empty_decode_mtp

        if batch_size != 0:

            if self.mtp_size > 1:
                return self.mtp_decode_no_pipeline_total(
                    tokens,
                    self.do_decode_callable,
                    key + ("main",),
                    self.do_decode_callable_mtp,
                    key + ("mtp",),
                    extra_inputs,
                    extra_inputs_mtp,
                )
            else:
                return self.do_decode_callable(key, tokens, *extra_inputs)
        else:
            if not self.ep_size > 1:
                return None
            if self.mtp_size > 1:
                return self.empty_mtp_decode_total(
                    self.do_empty_decode_callable,
                    (1,) + ("empty_main",),
                    self.do_empty_decode_callable_mtp,
                    (1,) + ("empty_mtp",),
                )
            else:
                return self.do_empty_decode_callable((1,) + ("empty",))


class MoeGate(nn.Module):
    def __init__(
        self,
        op_impl,
        dim,
        topk,
        *,
        n_groups,
        topk_groups,
        topk_as_topk_group_criteria,
        score_func,
        route_scale,
        n_experts,
        bias,
        e_score_correction_bias,
        norm_prob,
        n_fused_shared_experts: int,
        _debug_force_moe_balance: Optional[bool] = None,
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
        self.topk_as_topk_group_criteria = topk_as_topk_group_criteria
        self.score_func = score_func
        self.route_scale = route_scale
        self.n_experts = n_experts
        self.weight = nn.Parameter(torch.empty((n_experts, self.dim)))
        self.bias = bias
        self.e_score_correction_bias = e_score_correction_bias
        self.norm_prob = norm_prob
        self.n_fused_shared_experts = n_fused_shared_experts

        if _debug_force_moe_balance is None:
            _debug_force_moe_balance = get_global_args().debug.force_moe_balance
        self._debug_force_moe_balance = _debug_force_moe_balance
        if self._debug_force_moe_balance:
            self._debug_force_moe_balance_mask_cache = (
                self._debug_gen_force_moe_balance_mask(
                    ceil_div(get_global_args().infer.max_reqs, get_dp_size())
                )
            )

    def _debug_gen_force_moe_balance_mask(self, bs):
        # Strategy: For token i, pick ((i to i + topk) % n_experts)-th expert.
        # Note that the picked experts should have contiguous ids, so it is compatible
        # with expert grouping.
        mask = torch.ones((bs, self.n_experts), dtype=torch.bool, device="cuda")
        for k in range(self.topk):
            r = torch.arange(bs, device=mask.device)
            mask[r, (bs * get_dp_group().rank_in_group + r + k) % self.n_experts] = (
                False
            )
        return mask

    def forward(self, x):
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        if x.shape[0] == 0:
            return torch.empty(
                (0, self.topk),
                dtype=self.weight.dtype,
                device=self.weight.device,
            ), torch.empty((0, self.topk), dtype=torch.int32, device=self.weight.device)
        scores = F.linear(x, self.weight, self.bias)

        e_score_correction_bias = self.e_score_correction_bias
        if self._debug_force_moe_balance:
            if x.shape[0] <= self._debug_force_moe_balance_mask_cache.shape[0]:
                # decode
                mask = self._debug_force_moe_balance_mask_cache[: x.shape[0]]
            else:
                # prefill
                mask = self._debug_gen_force_moe_balance_mask(x.shape[0])
            scores.masked_fill_(mask, float("-inf"))
            e_score_correction_bias = None

        indices, weights = moe_gate(
            scores,
            self.topk,
            num_expert_group=self.n_groups,
            topk_group=self.topk_groups,
            topk_as_topk_group_criteria=self.topk_as_topk_group_criteria,
            e_score_correction_bias=e_score_correction_bias,
            score_func=self.score_func,
            norm_prob=self.norm_prob,
        )
        if self.route_scale != 1:
            weights *= self.route_scale
        weights = weights.type_as(x)
        indices = indices.to(torch.int32)

        if self.n_fused_shared_experts > 0:
            indice_shape = indices.shape
            final_indices = torch.empty(
                (indice_shape[0], indice_shape[1] + 1),
                dtype=indices.dtype,
                device=indices.device,
            )

            final_weights = torch.empty(
                (weights.shape[0], weights.shape[1] + 1),
                dtype=weights.dtype,
                device=weights.device,
            )

            chitu_backend.cuda_add_shared_experts(
                final_weights,
                final_indices,
                weights,
                indices,
                self.n_experts,
                self.n_fused_shared_experts,
            )
            weights, indices = final_weights, final_indices

        return weights, indices


class ParallelMoeBlock(nn.Module):
    """
    Mixture-of-Experts (MoE) block.

    An object of this class includes MoeGate, MoeExperts, and maybe shared experts in a separated object
    (if fuse_shared_experts=False). This object maybe in parallel.

    Args:
        gate (MoeGate): The gating layer.
        experts (QuantizedMoeExpertsBase): The layer containing routed experts + fused shared experts
        non_fused_shared_experts (Optional[nn.Module]): Optional layer for shared experts if not fused.
        layer_id (int): The layer id of this MoE block.
        enable_dynamic_load_balance: If True, enable dynamic load balancing. If None, the value will
            be read from global args.
    """

    def __init__(
        self,
        gate: MoeGate,
        experts: QuantizedMoeExpertsBase,
        non_fused_shared_experts: Optional[nn.Module] = None,
        layer_id: int = 0,
        moe_impl: Optional[MoEImplBase] = None,
        *,
        enable_dynamic_load_balance: Optional[bool] = None,
        prefill_memory_tolerance: Optional[float] = None,
        checkpoint_prefix: str,
    ):
        super().__init__()

        if moe_impl is None:
            moe_impl = get_moe_impl()

        self.gate = gate
        self.experts = experts
        self.shared_experts = non_fused_shared_experts

        self.shared_experts_stream = None
        if self.shared_experts is not None and not is_muxi():
            self.shared_experts_stream = torch.cuda.Stream()

        self.moe_impl = moe_impl
        if self.moe_impl is not None and self.moe_impl.ep_size > 1:
            self.expert_mapping = self.moe_impl.get_expert_mapping(layer_id=layer_id)
        else:
            self.expert_mapping = None

        self.checkpoint_prefix = checkpoint_prefix
        self.layer_id = layer_id

        from chitu.backend import Backend

        if self.layer_id is not None:
            Backend.register_moe_layer_experts(self.layer_id, self.experts)
        self.layer_id = layer_id

        if enable_dynamic_load_balance is None:
            enable_dynamic_load_balance = get_global_args().infer.moe_lb_trigger > 0
        self.enable_dynamic_load_balance = enable_dynamic_load_balance

        if prefill_memory_tolerance is None:
            prefill_memory_tolerance = (
                get_global_args().infer.moe.prefill_memory_tolerance
            )
        self.prefill_memory_tolerance = prefill_memory_tolerance

    def forward(self, x: torch.Tensor, inplace: bool = True) -> torch.Tensor:
        """
        Forward pass for the MoE block.

        Args:
            x (torch.Tensor): Input tensor.
            inplace (bool): If True, this function may touch `x`.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.shape  # [TODO] unify decode hidden states shape
        hidden_size = x.shape[-1]
        x = x.view(-1, hidden_size)

        weights, indices = self.gate(x)
        rerouted_indices = None

        if self.enable_dynamic_load_balance:
            planner = get_moe_load_planner()
            if planner is not None:
                global_slot_idx = planner.route_expert_ids(self.layer_id, indices)
                rerouted_indices = global_slot_idx.to(
                    dtype=indices.dtype, device=indices.device
                ).contiguous()
        elif self.expert_mapping is not None:
            rerouted_indices = self.expert_mapping[indices].contiguous()
        else:
            rerouted_indices = None
        if rerouted_indices is None:
            rerouted_indices = indices

        routed_x = IndexedBatchedRoutedActivation(
            x, rerouted_indices, expert_ids_are_local=self.moe_impl.ep_size == 1
        )

        shared_y = None
        x_in_use_simultenously = False
        if self.shared_experts is not None:
            ctx = nullcontext()
            if self.shared_experts_stream is not None:
                self.shared_experts_stream.wait_stream(torch.cuda.current_stream())
                ctx = torch.cuda.stream(self.shared_experts_stream)
                x_in_use_simultenously = True
            with ctx:
                shared_y = self.shared_experts(x)

        experts_impl = "auto"
        if self.moe_impl.ep_size > 1:
            experts_impl = self.moe_impl.get_experts_impl()
            routed_x_old = routed_x
            routed_x, weights = self.moe_impl.enter_moe(
                routed_x,
                weights,
                may_fuse_quant=get_quant_from_checkpoint_prefix(
                    f"{self.checkpoint_prefix}.experts"
                ),
                may_fuse_quant_kwargs=get_quant_kwargs_from_checkpoint_prefix(
                    f"{self.checkpoint_prefix}.experts"
                ),
                layer_id=self.layer_id,
            )
            x_in_use_simultenously = x_in_use_simultenously and (
                routed_x_old is routed_x
            )
        elif self.moe_impl.ep_size == 1:
            experts_impl = self.moe_impl.get_experts_impl()

        if (
            self.moe_impl.ep_size > 1
            and self.moe_impl.exit_moe_prefer_before_local_sum()
        ):
            if (
                self.moe_impl.task_type == TaskType.Prefill
                and self.prefill_memory_tolerance < self.moe_impl.ep_size
                and get_global_args().infer.prefill_chunk_size is not None
            ):
                logger.warning_once(
                    "`prefill_memory_tolerance` is not implemented when `exit_moe_prefer_before_local_sum` is True, ignoring."
                )

            y_before_local_sum = self.experts.forward_no_sum(
                routed_x, impl=experts_impl
            )
            y = self.moe_impl.exit_moe_before_local_sum(y_before_local_sum)

        else:
            if (
                self.moe_impl.task_type == TaskType.Prefill
                and self.moe_impl.ep_size > 1
                and self.prefill_memory_tolerance < self.moe_impl.ep_size
                and get_global_args().infer.prefill_chunk_size is not None
            ):
                max_n_tokens_per_chunk = int(
                    get_global_args().infer.prefill_chunk_size
                    / self.moe_impl.ep_size
                    * self.prefill_memory_tolerance
                )
                try:
                    chunks = routed_x.get_chunks_no_larger_than(
                        weights, max_n_tokens_per_chunk
                    )
                except Exception as e:
                    logger.warning(
                        f"Unable to chunk {type(routed_x)}: {e}. Ignoring `prefill_memory_tolerance`."
                    )
                    chunks = [(routed_x, weights)]
            else:
                chunks = [(routed_x, weights)]

            y_list = []
            for routed_x_item, weights_item in chunks:
                y_item = self.experts(
                    routed_x_item,
                    weights_item,
                    inplace=inplace and not x_in_use_simultenously,
                    impl=experts_impl,
                )
                assert y_item.ndim == 2 and y_item.shape[-1] == hidden_size
                y_list.append(y_item)
            assert len(y_list) > 0
            if len(y_list) == 1:
                y = y_list[0]
            else:
                y = torch.cat(y_list, dim=0)

            if shared_y is not None and self.moe_impl.tp_size > 1:
                # we need to reduce shared_y on tp group, if this group equals the group reduce y later, we can merge them together
                if self.moe_impl and self.moe_impl.ep_size > 1:
                    y_reduce_rank_list = self.moe_impl.exit_moe_reduce_rank_list()
                else:
                    y_reduce_rank_list = self.moe_impl.tp_group.rank_list
                if self.moe_impl.tp_group.rank_list == y_reduce_rank_list:
                    if self.shared_experts_stream:
                        torch.cuda.current_stream().wait_stream(
                            self.shared_experts_stream
                        )
                    y += shared_y
                    shared_y = None

            if self.moe_impl.ep_size > 1:
                y = self.moe_impl.exit_moe_after_local_sum(y)
            elif self.moe_impl.tp_size > 1:
                self.moe_impl.tp_group.all_reduce(y)

        if shared_y is not None:
            if self.shared_experts_stream:
                torch.cuda.current_stream().wait_stream(self.shared_experts_stream)
            if self.moe_impl.tp_size > 1:
                self.moe_impl.tp_group.all_reduce(shared_y)
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
