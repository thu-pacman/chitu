# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
from logging import getLogger
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from chitu.device_type import is_muxi
from chitu.attn_backend import AttnBackend, NpuAttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.cache_manager import PagedKVCacheManager, DenseKVCacheManager
from chitu.cuda_graph import make_dispatched_graphed_callables
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
    get_ep_group,
    get_ep_size,
    get_dp_group,
    get_dp_size,
)
from chitu.moe import get_moe_impl
from chitu.moe.batched_routed_activation import IndexedBatchedRoutedActivation
from chitu.utils import (
    compute_layer_dist_in_pipe,
    is_layer,
    try_import_platform_dep,
    try_import_opt_dep,
    ceil_div,
)
from chitu.quantization import (
    QuantizationRegistry,
    QuantizedMoeExpertsBase,
    get_quant_from_checkpoint_prefix,
    get_quant_kwargs_from_checkpoint_prefix,
    get_backend_from_checkpoint_prefix,
)
from chitu.hybrid_device import CPUParameter

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
        return torch.nn.functional.layer_norm(
            x.to(compute_dtype), (self.dim,), self.weight, self.bias, self.eps
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
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(dim=dim, eps=eps)
        self.bias = nn.Parameter(
            torch.zeros(self.dim, dtype=torch.get_default_dtype()), requires_grad=False
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
    def __init__(self, layer_id, cache, attn_backend):
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

    def __init__(self, layer_id: int, args, cache, attn_backend, op_impl):
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
        if get_global_args().infer.op_impl == "cpu":
            self.local_rank = "cpu"
        self.world_size = torch.distributed.get_world_size()
        self.device = torch.device(self.local_rank)

        self.pipeline_parallel_size = pipeline_parallel_size
        self.model_parallel_size = model_parallel_size
        self.pipeline_exec = pipeline_parallel_size > 1
        self.tensor_exec = model_parallel_size > 1

        self.tp_size = model_parallel_size
        self.pp_size = pipeline_parallel_size
        self.dp_size = get_dp_size()
        self.ep_group = get_ep_group()
        self.ep_size = self.ep_group.group_size
        self.pp_stage = (
            self.rank % (self.world_size // self.dp_size) // self.model_parallel_size
        )
        self.pp_main_rank = (self.rank // model_parallel_size) * model_parallel_size
        self.pp_end_stage = (self.world_size // self.dp_size - 1) // model_parallel_size

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

        with self.device:
            # precomputed freqs_cis has real data, so we can't put it on "meta" device
            self.precompute_freqs_cis(max_position_embeddings, self.device)

        self.do_decode_callable = None
        self.args = get_global_args()
        self.max_batch_size = self.args.infer.max_reqs
        self.model_type = self.args.models.type
        self.use_cuda_graph = self.args.infer.use_cuda_graph

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
        self.moe_impl = get_moe_impl()
        if self.ep_group.is_last_rank:
            self.experts_end_idx += remainder

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

    def _chunk_checkpoint_for_pipeline_parallel(
        self,
        checkpoint: dict[str, Any],
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
        checkpoint: dict[str, Any],
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
                if name.split(".")[-1] in self._get_1d_out_tensor_names(quant):
                    assert (
                        param.dim() == 1
                    ), f"{name} is expected to be 1D, but got {param.dim()}D"
                    if param.shape[0] == 1:  # Broadcast
                        partial_checkpoint[name] = param
                    else:
                        assert param.shape[0] % world_size == 0
                        chunks = torch.chunk(param, world_size, dim=0)
                        partial_checkpoint[name] = chunks[rank]
                elif name.split(".")[-1] in self._get_1d_in_tensor_names(quant):
                    assert (
                        param.dim() == 1
                    ), f"{name} is expected to be 1D, but got {param.dim()}D"
                    if get_tp_group().rank_in_group == 0:
                        partial_checkpoint[name] = param
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
                if name.split(".")[-1] in self._get_1d_in_tensor_names(quant):
                    assert (
                        param.dim() == 1
                    ), f"{name} is expected to be 1D, but got {param.dim()}D"
                    if param.shape[0] == 1:  # Broadcast
                        partial_checkpoint[name] = param
                    else:
                        assert param.shape[0] % world_size == 0
                        chunks = torch.chunk(param, world_size, dim=0)
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

    def process_state_dict_for_merging_qkv(self, checkpoint: dict[str, Any]):
        return checkpoint  # Inherit to preprocess. Leave it empty if not needed.

    def process_state_dict_for_merging_gate_up(self, checkpoint: dict[str, Any]):
        return checkpoint  # Inherit to preprocess. Leave it empty if not needed.

    def process_state_dict_for_merging_experts(self, checkpoint: dict[str, Any]):
        return checkpoint  # Inherit to preprocess. Leave it empty if not needed.

    def load_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *args,
        skip_preprocess: bool = False,
        **kwargs,
    ):
        if not skip_preprocess:
            state_dict = self.process_state_dict_for_blockfp4_before_chunk(state_dict)
            # handle ep param
            if self.ep_size > 1:
                local_experts = [
                    self.moe_impl.load_balancer[layer_id].get_local_experts(
                        self.moe_impl.ep_rank
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

            if self.pipeline_exec:
                state_dict = self._chunk_checkpoint_for_pipeline_parallel(
                    state_dict, self.global_n_layers, self.pp_stage, self.pp_size
                )
            if self.tensor_exec:
                state_dict = self._chunk_checkpoint_for_tensor_parallel(
                    state_dict, self.rank % self.tp_size, self.tp_size
                )

        # TODO: Move `state_dict` to GPU and preprocess on GPU if there is no `CPUParameter`s
        # Problems:
        # - Processing on GPU laeds to sever memory fragmentation (13.44 GiB fragements in 94.93
        #   GiB allocated memory). Disabling torch allocator with `PYTORCH_NO_CUDA_MEMORY_CACHING=1`
        #   works but may lead to too much performance degradation.

        self.load_state_dict(
            state_dict, *args, skip_preprocess=skip_preprocess, **kwargs
        )

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        *args,
        skip_preprocess: bool = False,
        **kwargs,
    ):
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

        super().load_state_dict(state_dict, *args, **kwargs)

    def _init_pre_layers(self):
        raise NotImplementedError

    def _init_layers(self, cache, attn_backend):
        raise NotImplementedError

    def _init_post_layers(self):
        raise NotImplementedError

    def _pre_layers(self, h, **args):
        raise NotImplementedError

    def _post_layers(self, h):
        """NOTE: _post_layers is assumed to be a token-wise computation"""
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
                self.cache.seq_len_delta.delta_position_ids_tensor_device
            ],
            self.freqs_cis_imag[
                self.cache.seq_len_delta.delta_position_ids_tensor_device
            ],
        )

    @torch.inference_mode()
    def prefill_no_pipeline(
        self, tokens, output_token_offsets: torch.Tensor, **args
    ) -> torch.Tensor:
        freqs_cis = self.prepare_freqs_cis()
        h = self._pre_layers(tokens, **args)
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
        for it, layer in enumerate(self.layers):
            h = layer(h, freqs_cis)
        h = self._post_layers(h)
        h = h.float()
        return h

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
    def prefill(
        self, tokens, output_token_offsets: torch.Tensor, **args
    ) -> torch.Tensor:
        self.attn_backend.prepare_metadata_for_prefill(self.cache.seq_len_delta)
        if self.pipeline_exec:
            return self.prefill_pipeline(tokens, output_token_offsets, **args)
        else:
            return self.prefill_no_pipeline(tokens, output_token_offsets, **args)

    def prepare_decoding_attn(self):
        block_table = self.cache.get_gpu_block_table()
        block_size = self.cache.get_block_size()
        self.attn_backend.prepare_metadata_for_decode(
            self.cache.seq_len_delta,
            block_table,
            block_size,
        )

    @torch.inference_mode()
    def decode(self, tokens, batch_size):
        if isinstance(self.cache, DenseKVCacheManager):
            key = (batch_size, self.cache.get_start_and_end_idx()[0])
        elif isinstance(self.cache, PagedKVCacheManager):
            key = (batch_size,)
        else:
            assert False

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
                            "actual_seq_lengths_kv": self.cache.seq_len_delta.new.lens_list
                        }
                    ]
                )

            @make_dispatched_graphed_callables(
                args_max_nelem=(tokens.numel() // batch_size * self.max_batch_size,),
                kwargs_max_nelem={},
                output_max_nelem_callback=lambda key, n: n
                // key[0]
                * self.max_batch_size,
                before_replay_callback=before_replay_callback,
                enable=current_cuda_graph_enabled,
            )
            def do_decode(tokens):
                freqs_cis = self.prepare_freqs_cis()
                if self.pipeline_exec:
                    return self.decode_pipeline(tokens, freqs_cis)
                else:
                    return self.decode_no_pipeline(tokens, freqs_cis)

            self.do_decode_callable = do_decode

        return self.do_decode_callable(key, tokens)


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

        if get_global_args().debug.force_moe_balance:
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
        if get_global_args().debug.force_moe_balance:
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
        layer_id: int = 0,
        *,
        checkpoint_prefix: str,
    ):
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.shared_experts = non_fused_shared_experts

        if self.shared_experts is not None:
            self.shared_experts_stream = torch.cuda.Stream()

        self.moe_impl = get_moe_impl()
        self.is_tp_mode = get_tp_size() > 1
        if self.moe_impl is not None:
            self.expert_mapping = self.moe_impl.get_expert_mapping(layer_id=layer_id)
        else:
            self.expert_mapping = None

        self.checkpoint_prefix = checkpoint_prefix
        self.layer_id = layer_id
        self.experts_stats = torch.zeros(self.experts.n_routed_experts, device="cuda")

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
        indices = (
            self.expert_mapping[indices] if self.expert_mapping is not None else indices
        )
        routed_x = IndexedBatchedRoutedActivation(x, indices)
        shared_y = None
        x_in_use_simultenously = False
        if self.shared_experts is not None:
            if not is_muxi():
                self.shared_experts_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.shared_experts_stream):
                    shared_y = self.shared_experts(x)
                    x_in_use_simultenously = True
            else:
                shared_y = self.shared_experts(x)
                x_in_use_simultenously = False

        experts_impl = "auto"
        tokens_per_expert = None
        if self.moe_impl is not None:
            experts_impl = self.moe_impl.get_experts_impl()
            routed_x, weights = self.moe_impl.token_permutation(
                routed_x,
                weights,
                may_fuse_quant=get_quant_from_checkpoint_prefix(
                    f"{self.checkpoint_prefix}.experts"
                ),
                may_fuse_quant_kwargs=get_quant_kwargs_from_checkpoint_prefix(
                    f"{self.checkpoint_prefix}.experts"
                ),
            )
            x_in_use_simultenously = False

        y = self.experts(
            routed_x, weights, inplace=not x_in_use_simultenously, impl=experts_impl
        )

        # Fuse allreduce to improve performance in TP mode
        if self.is_tp_mode:
            if shared_y is not None:
                if not is_muxi():
                    torch.cuda.current_stream().wait_stream(self.shared_experts_stream)
                y += shared_y
            if not self.moe_impl:
                torch.distributed.all_reduce(y, group=get_tp_group().gpu_group)

        if self.moe_impl is not None:
            y = self.moe_impl.token_unpermutation(y)

        if shared_y is not None and not self.is_tp_mode:
            if not is_muxi():
                torch.cuda.current_stream().wait_stream(self.shared_experts_stream)
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


def get_rmsnorm(dim: int, *, use_bias: bool, eps: float = 1e-6) -> RMSNorm:
    return RMSNormBias(dim, eps=eps) if use_bias else RMSNorm(dim, eps=eps)
