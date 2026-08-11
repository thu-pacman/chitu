# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Tuple
from typing_extensions import override
import functools
import logging
import math

import torch

from chitu.device_type import get_device_name, is_hygon
from chitu.import_utils import (
    try_import_platform_dep,
    try_import_opt_dep,
    try_import_and_setup_torch_npu,
)
from chitu.moe.batched_expert_result import (
    BatchedExpertResult,
    ExpertBlockPermutedBatchedExpertResult,
    PerExpertDenseBatchedExpertResultMinimal,
    PerTokenBatchedExpertResult,
)
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    ConcatPermutedBatchedRoutedActivationMinimal,
    ExpertBlockIndexedBatchedRoutedActivation,
    ExpertBlockPermutedBatchedRoutedActivationWithScale,
    IndexedBatchedRoutedActivationWithScale,
    IndexedBatchedRoutedActivationWithScaleAndPaddedPerExpertCnt,
    PerExpertDenseBatchedRoutedActivationMinimal,
    PerExpertDenseBatchedRoutedActivationWithScaleMinimal,
    IndexedBatchedRoutedActivationWithPaddedPerExpertCnt,
)
from chitu.native_layout import (
    AiterMoeCInt8Gemm1Weight,
    AiterMoeCInt8Gemm2Weight,
    HygonDeepGemmW8A8LegacyMarlinWeight,
    HygonDeepGemmW8A8Marlin2Weight,
    NativeLayoutMixin,
    NpuFractalZnTensor,
)
from chitu.ops import silu_and_mul
from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.base import QuantizedLinearBase, QuantizedMoeExpertsMerged
from chitu.ops.quant import w8a8_gemm_per_token_per_channel, a8_per_token_act_quant
from chitu.ops.utils import make_op_dispatcher
from chitu.lazy import LazyTensor, eval_lazy
from chitu.global_vars import get_global_args
from chitu.utils import parse_dtype

_, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
lmslim_quant_ops, has_lmslim_quant_ops = try_import_platform_dep(
    "lmslim.quantize.quant_ops"
)
aiter, has_aiter = try_import_platform_dep("aiter.moe")
hygon_deepgemm, has_deepgemm = try_import_opt_dep("deepgemm", "deep_gemm")
lightop, has_lightop = try_import_platform_dep("lightop")

if has_triton:
    from chitu.moe.experts.triton_fused_experts import fused_experts_int8
if has_torch_npu:
    from chitu.moe.experts import fused_experts_no_sum_npu, fused_experts_npu_for_ep

W8A8_PER_TOKEN_PER_CHANNEL_DYN = "w8a8_per_token_per_channel_dyn"
_DEEPGEMM_MASKED_MIN_LAYOUT_M = 64
_DEEPGEMM_MOE_BLOCK_SIZE = 256

logger = logging.getLogger(__name__)


def _hygon_backend_w8a8_per_token_per_channel_dyn_use_deepgemm_moe(
    _: dict[str, Any],
) -> bool:
    if not (is_hygon() and has_deepgemm):
        return False
    try:
        ep_size = int(getattr(get_global_args().infer, "ep_size", 1))
    except (TypeError, ValueError):
        ep_size = 1
    return ep_size > 1


def _prepare_w8a8_per_token_per_channel_dyn_per_expert_activation(
    routed_x: PerExpertDenseBatchedRoutedActivationWithScaleMinimal,
) -> tuple[torch.Tensor, torch.Tensor]:
    if routed_x.quant_method != W8A8_PER_TOKEN_PER_CHANNEL_DYN:
        raise RuntimeError(
            "DeepGEMM MoE only supports w8a8_per_token_per_channel_dyn "
            f"pre-quantized activations, got {routed_x.quant_method}."
        )
    activation_per_expert = routed_x.activation_per_expert.contiguous()
    if activation_per_expert.dtype != torch.int8:
        raise RuntimeError(
            "w8a8_per_token_per_channel_dyn pre-quantized DeepGEMM MoE "
            f"activation must be torch.int8, got {activation_per_expert.dtype}."
        )
    activation_scale_per_expert = routed_x.activation_scale_per_expert.contiguous()
    if activation_scale_per_expert.ndim == 2:
        activation_scale_per_expert = activation_scale_per_expert.unsqueeze(-1)
    E, M, _ = activation_per_expert.shape
    if (
        activation_scale_per_expert.ndim != 3
        or tuple(activation_scale_per_expert.shape[:2]) != (E, M)
        or activation_scale_per_expert.shape[-1] != 1
    ):
        raise RuntimeError(
            "w8a8_per_token_per_channel_dyn pre-quantized DeepGEMM MoE "
            f"activation scale must have shape ({E}, {M}, 1), got "
            f"{tuple(activation_scale_per_expert.shape)}."
        )
    return activation_per_expert, activation_scale_per_expert


def _lightop_gemm_w8a8_smooth_available() -> bool:
    return has_lightop and callable(getattr(lightop, "gemm_w8a8_smooth", None))


def _lightop_fuse_silu_mul_quant_ep_available() -> bool:
    return has_lightop and callable(getattr(lightop, "fuse_silu_mul_quant_ep", None))


def _lightop_moe_gemm_w8a8_available() -> bool:
    return has_lightop and callable(getattr(lightop, "moe_gemm_w8a8", None))


def _a8_per_token_act_quant_with_expert_mask(
    activation_per_expert: torch.Tensor,
    n_tokens_per_expert: torch.Tensor,
    *,
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    E, M, hidden = activation_per_expert.shape
    q_activation = torch.zeros(
        (E, M, hidden), dtype=torch.int8, device=activation_per_expert.device
    )
    activation_scale = torch.zeros(
        (E, M, 1), dtype=scale_dtype, device=activation_per_expert.device
    )
    for expert_id, n_tokens in enumerate(n_tokens_per_expert.detach().cpu().tolist()):
        n_tokens = min(max(int(n_tokens), 0), M)
        if n_tokens == 0:
            continue
        q_valid, scale_valid = a8_per_token_act_quant(
            activation_per_expert[expert_id, :n_tokens].contiguous(),
            scale_dtype=scale_dtype,
        )
        q_activation[expert_id, :n_tokens].copy_(q_valid.view(n_tokens, hidden))
        activation_scale[expert_id, :n_tokens].copy_(scale_valid.reshape(n_tokens, 1))
    return q_activation.contiguous(), activation_scale.contiguous()


def _pad_deepgemm_masked_m(tensor: torch.Tensor, layout_m: int) -> torch.Tensor:
    if tensor.shape[1] >= layout_m:
        return tensor.contiguous()
    return torch.nn.functional.pad(
        tensor, (0, 0, 0, layout_m - tensor.shape[1]), "constant", 0
    ).contiguous()


def _call_deepgemm_w8a8_grouped_gemm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    m_indices: torch.Tensor,
    out: torch.Tensor,
) -> None:
    hygon_deepgemm.m_grouped_i8_gemm_nt_contiguous(
        (a, a_scale), (weight, weight_scale), out, m_indices, {"MODE": 1000}
    )


def _fill_deepgemm_padding_m_indices(
    block_to_expert_indices: torch.Tensor,
    *,
    num_experts: int,
) -> torch.Tensor:
    """
    Replace block padding markers with a valid expert id for DeepGEMM.

    Padding outputs are not gathered back to real tokens, so padded rows can use
    the block's expert id as dummy work.
    """
    valid = (block_to_expert_indices >= 0) & (block_to_expert_indices < num_experts)
    has_valid = valid.any(dim=1, keepdim=True)
    first_valid_pos = valid.to(torch.int32).argmax(dim=1, keepdim=True)
    block_expert_indices = block_to_expert_indices.gather(1, first_valid_pos)
    block_expert_indices = torch.where(
        has_valid, block_expert_indices, torch.zeros_like(block_expert_indices)
    )
    return torch.where(
        block_to_expert_indices < 0,
        block_expert_indices.expand_as(block_to_expert_indices),
        block_to_expert_indices,
    )


def _lightop_get_moe_cuda_config(
    *,
    E: int,
    M: int,
    N1: int,
    K1: int,
    N2: int,
    K2: int,
    topk: int,
) -> Tuple[dict[str, Any], dict[str, Any], bool]:
    device_name = get_device_name()
    device_name_for_cfg = ""
    if "BW" in device_name:
        device_name_for_cfg = "BW200"
    elif "K100" in device_name:
        device_name_for_cfg = "K100"

    num_cus = 0
    if torch.cuda.is_available():
        try:
            num_cus = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        except Exception:
            num_cus = 0

    try:
        cfg1, cfg2, status = lightop.get_moe_cuda_config(
            E, M, N1, K1, N2, K2, topk, device_name_for_cfg, str(num_cus), None
        )
    except Exception as e:
        logger.warning_once(
            f"lightop.get_moe_cuda_config failed, falling back to defaults: {e}"
        )
        status = False
        cfg1, cfg2 = {}, {}

    cfg1 = dict(cfg1 or {})
    cfg2 = dict(cfg2 or {})
    placeholder = {"BLOCK_SIZE_M": 16, "MODE": 33, "DELTA": 8}
    for cfg in (cfg1, cfg2):
        for key, value in placeholder.items():
            cfg.setdefault(key, value)
    return cfg1, cfg2, bool(status)


def _validate_indexed_moe_shapes(
    *,
    label: str,
    activation: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    dim: int,
    moe_inter_dim: int,
    w1_plain_shape: Optional[torch.Size] = None,
    w2_plain_shape: Optional[torch.Size] = None,
) -> tuple[int, int, int, int, int]:
    w1_shape = tuple(w1_plain_shape) if w1_plain_shape is not None else tuple(w1.shape)
    w2_shape = tuple(w2_plain_shape) if w2_plain_shape is not None else tuple(w2.shape)
    if len(w1_shape) != 3 or len(w2_shape) != 3:
        raise ValueError(
            f"{label} weights must be 3D: w1={tuple(w1.shape)} plain_w1={w1_shape} "
            f"w2={tuple(w2.shape)} plain_w2={w2_shape}"
        )
    E, N1, K1 = w1_shape
    _, N2, K2 = w2_shape
    if (
        activation.shape[-1] != K1
        or K1 != dim
        or N2 != dim
        or K2 != moe_inter_dim
        or N1 != moe_inter_dim * 2
    ):
        raise ValueError(
            f"{label} shape mismatch: "
            f"activation={tuple(activation.shape)} "
            f"w1={tuple(w1.shape)} plain_w1={w1_shape} "
            f"w2={tuple(w2.shape)} plain_w2={w2_shape} "
            f"expected dim={dim} moe_inter_dim={moe_inter_dim}"
        )
    return E, N1, K1, N2, K2


@QuantizationRegistry.register_linear("w8a8_per_token_per_channel_dyn")
class W8A8PerTokenPerChannelDynLinear(QuantizedLinearBase):
    """
    int8 weight + int8 dynamic activation quantized linear layer.
    """

    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        ############################################
        # Parameters specific to this quantization
        weight_scale_dtype: Optional[torch.dtype | str] = None,
        weight_scale_has_singleton_last_dim: bool = False,
    ):
        super().__init__(in_features, out_features, has_bias)

        self.weight = torch.nn.Parameter(
            torch.zeros(
                self.out_features,
                self.in_features,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )

        if weight_scale_dtype is None:
            weight_scale_dtype = torch.get_default_dtype()
        elif isinstance(weight_scale_dtype, str):
            weight_scale_dtype = parse_dtype(weight_scale_dtype)
        self.weight_scale = torch.nn.Parameter(
            torch.ones(
                (
                    (self.out_features, 1)
                    if weight_scale_has_singleton_last_dim
                    else (self.out_features,)
                ),
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )

        if has_bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(
                    (self.out_features,),
                    dtype=torch.get_default_dtype(),
                ),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_x, act_scale = a8_per_token_act_quant(x)
        out = w8a8_gemm_per_token_per_channel(
            q_x,
            act_scale,
            self.weight,
            self.weight_scale.view(self.out_features),
        ).view(*x.shape[:-1], self.out_features)
        if self.bias is not None:
            out += self.bias
        return out


@QuantizationRegistry.register_linear(
    "w8a8_per_token_per_channel_dyn", when=lambda _: has_torch_npu, priority=1
)
class AscendW8A8PerTokenPerChannelDynLinear(
    NativeLayoutMixin, W8A8PerTokenPerChannelDynLinear
):
    """
    Ascend implementation of W8A8PerTokenPerChannelDynLinear using NpuFuFractalZnTensor
    layout.
    """

    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(self.weight, NpuFractalZnTensor)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return torch.empty([0, self.out_features], dtype=x.dtype, device=x.device)
        quantized_x, dynamic_scale = a8_per_token_act_quant(
            x.view(-1, self.in_features)
        )
        y = w8a8_gemm_per_token_per_channel(
            quantized_x,
            dynamic_scale,
            self.get_native_layout_weight(),
            self.weight_scale.view(self.out_features),
        )
        if self.bias is not None:
            y += self.bias
        return y.view(*x.shape[:-1], y.shape[-1])


def _finalize_fused_experts_sum_output(
    output, hidden_states, topk_weights: torch.Tensor, inplace: bool
):
    if hasattr(output, "weighted_sum"):
        out = hidden_states.activation if inplace else None
        return output.weighted_sum(topk_weights, out=out)
    return output


@make_op_dispatcher
def fused_experts_no_sum_w8a8_per_token_per_channel_dyn_indexed(
    hidden_states,
    w1,
    w2,
    *,
    impl: str = "auto",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
    use_int8_w8a8: bool = False,
    swiglu_limit: Optional[float] = None,
): ...


@fused_experts_no_sum_w8a8_per_token_per_channel_dyn_indexed.register_auto
def _auto_fused_experts_no_sum_w8a8_per_token_per_channel_dyn_indexed():
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


fused_experts_no_sum_w8a8_per_token_per_channel_dyn_indexed.register_candidate(
    "torch_npu"
)
if has_torch_npu:
    fused_experts_no_sum_w8a8_per_token_per_channel_dyn_indexed.register("torch_npu")(
        fused_experts_no_sum_npu
    )


@make_op_dispatcher
def fused_experts_no_sum_w8a8_per_token_per_channel_dyn_concat_permuted(
    hidden_states,
    w1,
    w2,
    *,
    impl: str = "auto",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    experts_start_idx: int = 0,
    use_int8_w8a8: bool = False,
    swiglu_limit: Optional[float] = None,
) -> BatchedExpertResult: ...


@fused_experts_no_sum_w8a8_per_token_per_channel_dyn_concat_permuted.register_auto
def _auto_fused_experts_no_sum_w8a8_per_token_per_channel_dyn_concat_permuted():
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


fused_experts_no_sum_w8a8_per_token_per_channel_dyn_concat_permuted.register_candidate(
    "torch_npu"
)
if has_torch_npu:
    fused_experts_no_sum_w8a8_per_token_per_channel_dyn_concat_permuted.register(
        "torch_npu"
    )(fused_experts_npu_for_ep)


@make_op_dispatcher
def fused_experts_sum_w8a8_per_token_per_channel_dyn_indexed(
    hidden_states,
    w1,
    w2,
    topk_weights: torch.Tensor,
    *,
    inplace: bool = False,
    impl: str = "auto",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
    use_int8_w8a8: bool = False,
    swiglu_limit: Optional[float] = None,
) -> torch.Tensor: ...


@fused_experts_sum_w8a8_per_token_per_channel_dyn_indexed.register_auto
def _auto_fused_experts_sum_w8a8_per_token_per_channel_dyn_indexed():
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


@fused_experts_sum_w8a8_per_token_per_channel_dyn_indexed.register("torch_npu")
def _run_sum_indexed_torch_npu(
    hidden_states,
    w1,
    w2,
    topk_weights: torch.Tensor,
    *,
    inplace: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
    use_int8_w8a8: bool = False,
    swiglu_limit: Optional[float] = None,
    impl: str,
) -> torch.Tensor:
    output = fused_experts_no_sum_w8a8_per_token_per_channel_dyn_indexed(
        hidden_states,
        w1,
        w2,
        impl=impl,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        global_num_experts=global_num_experts,
        experts_start_idx=experts_start_idx,
        use_int8_w8a8=use_int8_w8a8,
        swiglu_limit=swiglu_limit,
    )
    return _finalize_fused_experts_sum_output(
        output, hidden_states, topk_weights=topk_weights, inplace=inplace
    )


@make_op_dispatcher
def fused_experts_sum_w8a8_per_token_per_channel_dyn_concat_permuted(
    hidden_states,
    w1,
    w2,
    topk_weights: torch.Tensor,
    *,
    inplace: bool = False,
    impl: str = "auto",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    experts_start_idx: int = 0,
    use_int8_w8a8: bool = False,
    swiglu_limit: Optional[float] = None,
) -> torch.Tensor: ...


@fused_experts_sum_w8a8_per_token_per_channel_dyn_concat_permuted.register_auto
def _auto_fused_experts_sum_w8a8_per_token_per_channel_dyn_concat_permuted():
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


@fused_experts_sum_w8a8_per_token_per_channel_dyn_concat_permuted.register("torch_npu")
def _run_sum_concat_torch_npu(
    hidden_states,
    w1,
    w2,
    topk_weights: torch.Tensor,
    *,
    inplace: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    experts_start_idx: int = 0,
    use_int8_w8a8: bool = False,
    swiglu_limit: Optional[float] = None,
    impl: str,
) -> torch.Tensor:
    output = fused_experts_no_sum_w8a8_per_token_per_channel_dyn_concat_permuted(
        hidden_states,
        w1,
        w2,
        impl=impl,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        experts_start_idx=experts_start_idx,
        use_int8_w8a8=use_int8_w8a8,
        swiglu_limit=swiglu_limit,
    )
    return _finalize_fused_experts_sum_output(
        output, hidden_states, topk_weights=topk_weights, inplace=inplace
    )


class W8A8PerTokenPerChannelDynMoeExpertsMergedBase(QuantizedMoeExpertsMerged):
    """Merged W8A8 MoE experts with shared parameter layout."""

    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        dim: int,
        moe_inter_dim: int,
        global_n_experts: int,
        experts_start_idx: int,
        experts_end_idx: int,
        n_activated_experts: int,
        checkpoint_prefix: str,
        ############################################
        # Parameters specific to this quantization
        weight_scale_dtype: Optional[torch.dtype | str] = None,
        weight_scale_has_singleton_last_dim: bool = False,
    ):
        super().__init__(
            dim,
            moe_inter_dim,
            global_n_experts,
            experts_start_idx,
            experts_end_idx,
            n_activated_experts,
            checkpoint_prefix,
        )
        if weight_scale_dtype is None:
            weight_scale_dtype = torch.get_default_dtype()
        elif isinstance(weight_scale_dtype, str):
            weight_scale_dtype = parse_dtype(weight_scale_dtype)
        self.gate_up_proj_weight = torch.nn.Parameter(
            torch.empty(
                (self.group_size, moe_inter_dim * 2, self.dim), dtype=torch.int8
            ),
            requires_grad=False,
        )
        self.gate_up_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                (
                    (self.group_size, moe_inter_dim * 2, 1)
                    if weight_scale_has_singleton_last_dim
                    else (self.group_size, moe_inter_dim * 2)
                ),
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        self.down_proj_weight = torch.nn.Parameter(
            torch.empty((self.group_size, self.dim, moe_inter_dim), dtype=torch.int8),
            requires_grad=False,
        )
        self.down_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                (
                    (self.group_size, self.dim, 1)
                    if weight_scale_has_singleton_last_dim
                    else (self.group_size, self.dim)
                ),
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )


@QuantizationRegistry.register_moe_experts(
    "w8a8_per_token_per_channel_dyn", merge_gate_up=True
)
class AscendW8A8PerTokenPerChannelDynMoeExperts(
    NativeLayoutMixin, W8A8PerTokenPerChannelDynMoeExpertsMergedBase
):
    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(self.gate_up_proj_weight, NpuFractalZnTensor)
        self.apply_native_layout(self.down_proj_weight, NpuFractalZnTensor)

    @override
    @functools.singledispatchmethod
    def forward_no_sum(
        self, routed_x: BatchedRoutedActivation, impl: str = "torch_npu"
    ) -> BatchedExpertResult:
        return super().forward_no_sum(routed_x, impl=impl)

    @forward_no_sum.register
    def _(
        self, routed_x: IndexedBatchedRoutedActivation, impl: str = "torch_npu"
    ) -> BatchedExpertResult:
        return fused_experts_no_sum_w8a8_per_token_per_channel_dyn_indexed(
            routed_x,
            w1=self.get_native_layout_gate_up_proj_weight(),
            w1_scale=self.gate_up_proj_weight_scale.view(
                self.group_size, self.moe_inter_dim * 2
            ),
            w2=self.get_native_layout_down_proj_weight(),
            w2_scale=self.down_proj_weight_scale.view(self.group_size, self.dim),
            use_int8_w8a8=True,
            impl=impl,
            global_num_experts=self.global_n_experts,
            experts_start_idx=self.experts_start_idx,
            swiglu_limit=self.swiglu_limit,
        )

    @override
    @functools.singledispatchmethod
    def forward(
        self,
        routed_x: BatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "torch_npu",
    ) -> torch.Tensor:
        return super().forward(routed_x, weights, inplace=inplace, impl=impl)

    @forward.register
    def _(
        self,
        routed_x: IndexedBatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "torch_npu",
    ) -> torch.Tensor:
        return fused_experts_sum_w8a8_per_token_per_channel_dyn_indexed(
            routed_x,
            w1=self.get_native_layout_gate_up_proj_weight(),
            w1_scale=self.gate_up_proj_weight_scale.view(
                self.group_size, self.moe_inter_dim * 2
            ),
            w2=self.get_native_layout_down_proj_weight(),
            w2_scale=self.down_proj_weight_scale.view(self.group_size, self.dim),
            topk_weights=weights,
            use_int8_w8a8=True,
            impl=impl,
            global_num_experts=self.global_n_experts,
            experts_start_idx=self.experts_start_idx,
            swiglu_limit=self.swiglu_limit,
        )

    @forward_no_sum.register
    def _(
        self,
        routed_x: ConcatPermutedBatchedRoutedActivationMinimal,
        impl: str = "torch_npu",
    ) -> BatchedExpertResult:
        return fused_experts_no_sum_w8a8_per_token_per_channel_dyn_concat_permuted(
            routed_x,
            w1=self.get_native_layout_gate_up_proj_weight(),
            w1_scale=self.gate_up_proj_weight_scale.view(
                self.group_size, self.moe_inter_dim * 2
            ),
            w2=self.get_native_layout_down_proj_weight(),
            w2_scale=self.down_proj_weight_scale.view(self.group_size, self.dim),
            use_int8_w8a8=True,
            impl=impl,
            experts_start_idx=self.experts_start_idx,
            swiglu_limit=self.swiglu_limit,
        )

    @forward.register
    def _(
        self,
        routed_x: ConcatPermutedBatchedRoutedActivationMinimal,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "torch_npu",
    ) -> torch.Tensor:
        return fused_experts_sum_w8a8_per_token_per_channel_dyn_concat_permuted(
            routed_x,
            w1=self.get_native_layout_gate_up_proj_weight(),
            w1_scale=self.gate_up_proj_weight_scale.view(
                self.group_size, self.moe_inter_dim * 2
            ),
            w2=self.get_native_layout_down_proj_weight(),
            w2_scale=self.down_proj_weight_scale.view(self.group_size, self.dim),
            topk_weights=weights,
            use_int8_w8a8=True,
            impl=impl,
            experts_start_idx=self.experts_start_idx,
            swiglu_limit=self.swiglu_limit,
        )


@QuantizationRegistry.register_moe_experts(
    W8A8_PER_TOKEN_PER_CHANNEL_DYN,
    merge_gate_up=True,
    when=lambda _: is_hygon() and has_aiter,
    priority=10,
)
class HygonAiterW8A8PerTokenPerChannelDynMoeExpertsMerged(
    NativeLayoutMixin, W8A8PerTokenPerChannelDynMoeExpertsMergedBase
):
    """
    W8A8 per-token/per-channel dynamic MoE on Hygon using Aiter MOE_C.
    """

    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(self.gate_up_proj_weight, AiterMoeCInt8Gemm1Weight)
        self.apply_native_layout(self.down_proj_weight, AiterMoeCInt8Gemm2Weight)

    @override
    @functools.singledispatchmethod
    def forward(
        self,
        routed_x: BatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        raise RuntimeError(
            "Aiter MoE only supports IndexedBatchedRoutedActivation, "
            f"got {type(routed_x).__name__}."
        )

    @forward.register
    def _(
        self,
        routed_x: IndexedBatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        return self._forward_aiter_indexed(routed_x, weights)

    @override
    def forward_no_sum(
        self, routed_x: BatchedRoutedActivation, impl: str = "auto"
    ) -> BatchedExpertResult:
        raise RuntimeError("Aiter MoE does not support forward_no_sum.")

    def _aiter_weights_for_config(
        self, moe_config: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        solution_type = str(getattr(moe_config, "solution_type", "")).lower()
        if solution_type != "moe_c":
            raise RuntimeError(
                "Hygon W8A8 Aiter MoE only supports MOE_C: "
                f"solution_type={solution_type}"
            )
        if not (
            AiterMoeCInt8Gemm1Weight.check_tensor(self.gate_up_proj_weight)
            and AiterMoeCInt8Gemm2Weight.check_tensor(self.down_proj_weight)
        ):
            raise RuntimeError(
                "Aiter MOE_C weight layout is not prepared. Load the plain "
                "checkpoint state_dict before running moe_backend='aiter'."
            )
        return self.gate_up_proj_weight, self.down_proj_weight

    def _forward_aiter_indexed(
        self,
        routed_x: IndexedBatchedRoutedActivation,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.swiglu_limit is not None:
            raise NotImplementedError(
                "swiglu_limit is not implemented for Chitu's Aiter W8A8 MoE wrapper"
            )
        topk = weights.shape[1]
        if not (
            self.experts_start_idx == 0
            and self.experts_end_idx == self.global_n_experts
        ):
            raise RuntimeError("Aiter MoE requires all experts to be local.")

        M = routed_x.activation.shape[0]
        if M == 0:
            return torch.empty(
                (0, self.dim),
                dtype=routed_x.activation.dtype,
                device=routed_x.activation.device,
            )
        E, N1, K, N2, _ = _validate_indexed_moe_shapes(
            label="Aiter MoE",
            activation=routed_x.activation,
            w1=self.gate_up_proj_weight,
            w2=self.down_proj_weight,
            dim=self.dim,
            moe_inter_dim=self.moe_inter_dim,
        )
        found, moe_config = aiter.get_aiter_moe_config(
            M=M,
            E=E,
            N1=N1,
            N2=N2,
            K=K,
            top_k=topk,
            block_size=0,
            dtype=routed_x.activation.dtype,
            quant_type=aiter.MoeQuantType.W8A8,
            activation="silu",
            gated=True,
        )
        if not found:
            raise RuntimeError("Aiter MoE config was not found.")

        w1, w2 = self._aiter_weights_for_config(moe_config)
        topk_weights = weights.to(
            device=routed_x.activation.device,
            dtype=torch.float32,
        ).contiguous()
        return aiter.aiter_moe(
            hidden_states=routed_x.activation,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=routed_x.token_to_expert_indices,
            moe_config=moe_config,
            inplace=False,
            activation="silu",
            w1_scale=self.gate_up_proj_weight_scale.view(
                self.group_size, self.moe_inter_dim * 2, 1
            )
            .to(torch.float32)
            .contiguous(),
            w2_scale=self.down_proj_weight_scale.view(self.group_size, self.dim, 1)
            .to(torch.float32)
            .contiguous(),
            w1_zp=None,
            w2_zp=None,
            a1_scale=None,
            a2_scale=None,
            block_shape=None,
            global_num_experts=self.global_n_experts,
            expert_map=None,
            routed_scaling_factor=1.0,
            use_weight_shuffle=False,
            output_dtype=routed_x.activation.dtype,
        )


@QuantizationRegistry.register_moe_experts(
    W8A8_PER_TOKEN_PER_CHANNEL_DYN,
    merge_gate_up=True,
    when=lambda _: has_triton,
)
class TritonW8A8PerTokenPerChannelDynMoeExpertsMerged(
    W8A8PerTokenPerChannelDynMoeExpertsMergedBase
):
    """W8A8 per-token/per-channel dynamic MoE using Triton kernels."""

    @override
    @functools.singledispatchmethod
    def forward_no_sum(
        self, routed_x: BatchedRoutedActivation, impl: str = "triton"
    ) -> BatchedExpertResult:
        return super().forward_no_sum(routed_x, impl=impl)

    @forward_no_sum.register
    def _(
        self,
        routed_x: IndexedBatchedRoutedActivation,
        impl: str = "triton",
    ) -> PerTokenBatchedExpertResult:
        return fused_experts_int8(
            hidden_states=routed_x,
            w1=self.gate_up_proj_weight,
            w2=self.down_proj_weight,
            activation="silu",
            use_int8_w8a16=False,
            w1_scale=self.gate_up_proj_weight_scale,
            w2_scale=self.down_proj_weight_scale,
            a1_scale=None,
            a2_scale=None,
            use_int8_w8a8=True,
            swiglu_limit=self.swiglu_limit,
            experts_start_idx=self.experts_start_idx,
        )


@QuantizationRegistry.register_moe_experts(
    W8A8_PER_TOKEN_PER_CHANNEL_DYN,
    merge_gate_up=True,
    when=lambda _: is_hygon()
    and _lightop_gemm_w8a8_smooth_available()
    and _lightop_moe_gemm_w8a8_available(),
    priority=1,
)
class HygonLightopW8A8PerTokenPerChannelDynMoeExpertsMerged(
    W8A8PerTokenPerChannelDynMoeExpertsMergedBase
):
    """W8A8 per-token/per-channel dynamic MoE on Hygon using LightOP."""

    @override
    @functools.singledispatchmethod
    def forward_no_sum(self, routed_x: Any, impl: str = "auto") -> BatchedExpertResult:
        return super().forward_no_sum(routed_x, impl=impl)

    @torch.no_grad()
    def _w8a8_expert_gemm(
        self,
        x_fp: torch.Tensor,
        weight_int8: torch.Tensor,
        weight_scale: torch.Tensor,
        x_scale: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if x_scale is not None:
            x_scale = eval_lazy(x_scale)
        out_dtype = out_dtype or (
            weight_scale.dtype if x_fp.dtype == torch.int8 else x_fp.dtype
        )
        if math.prod(x_fp.shape) == 0 or x_fp.shape[0] == 0:
            return torch.empty(
                (x_fp.shape[0], weight_int8.shape[0]),
                dtype=out_dtype,
                device=x_fp.device,
            )
        if isinstance(x_fp, LazyTensor):
            x2 = (
                x_fp
                if len(x_fp.shape) == 2
                else eval_lazy(x_fp).reshape(-1, x_fp.shape[-1])
            )
        else:
            x2 = x_fp.reshape(-1, x_fp.shape[-1])
        if x_scale is None:
            q_x, act_scale = a8_per_token_act_quant(x2)
            scale_a = act_scale.to(torch.float32).view(q_x.shape[0], 1)
        else:
            if x2.dtype != torch.int8:
                raise ValueError("Pre-quantized W8A8 activation must be torch.int8")
            q_x = x2.contiguous()
            scale_a = x_scale.to(torch.float32).reshape(q_x.shape[0], -1)
            if scale_a.shape[1] != 1:
                raise NotImplementedError(
                    "Hygon W8A8 LightOP currently expects per-token INT8 scales"
                )
            scale_a = scale_a.contiguous()
        scale_b = weight_scale.to(torch.float32).view(weight_int8.shape[0], 1)
        status, out = lightop.gemm_w8a8_smooth(
            q_x,
            weight_int8.t(),
            scale_a,
            scale_b,
            None,
            torch.get_default_dtype(),
        )
        if status is False or out is None:
            raise RuntimeError(
                "lightop.gemm_w8a8_smooth failed: "
                f"{q_x.shape=} {weight_int8.shape=} {scale_a.shape=} {scale_b.shape=}"
            )
        if out.dtype != out_dtype:
            out = out.to(out_dtype)
        return out.view(*x_fp.shape[:-1], weight_int8.shape[0])

    @override
    def forward_ith_expert_gate_up(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self._w8a8_expert_gemm(
            x,
            self.gate_up_proj_weight[i],
            self.gate_up_proj_weight_scale[i],
            x_scale=x_scale,
        )

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return self._w8a8_expert_gemm(
            x,
            self.down_proj_weight[i],
            self.down_proj_weight_scale[i],
        )

    @override
    @functools.singledispatchmethod
    def forward(
        self,
        routed_x: Any,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        return super().forward(routed_x, weights, inplace=inplace, impl=impl)

    @forward.register
    def _(
        self,
        routed_x: IndexedBatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        return self._forward_indexed(routed_x, weights)

    @forward.register
    def _(
        self,
        routed_x: IndexedBatchedRoutedActivationWithPaddedPerExpertCnt,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        return self._forward_indexed(routed_x, weights)

    @forward_no_sum.register
    def _(
        self,
        routed_x: PerExpertDenseBatchedRoutedActivationWithScaleMinimal,
        impl: str = "auto",
    ) -> PerExpertDenseBatchedExpertResultMinimal:
        routed_x = routed_x.as_local_expert_ids(
            self.experts_start_idx, self.experts_end_idx
        )
        if routed_x.quant_method != W8A8_PER_TOKEN_PER_CHANNEL_DYN:
            raise NotImplementedError(
                "Hygon W8A8 LightOP only supports "
                "w8a8_per_token_per_channel_dyn per-expert scaled activations."
            )
        output_dtype = routed_x.output_dtype or torch.get_default_dtype()
        activation_per_expert = routed_x.activation_per_expert.contiguous()
        activation_scale_per_expert = routed_x.activation_scale_per_expert.contiguous()
        row_ids = torch.arange(
            activation_per_expert.shape[1],
            device=activation_per_expert.device,
            dtype=routed_x.n_tokens_per_expert.dtype,
        ).unsqueeze(0)
        valid_token_mask = (
            row_ids < routed_x.n_tokens_per_expert.unsqueeze(1)
        ).unsqueeze(-1)
        activation_per_expert = torch.where(
            valid_token_mask,
            activation_per_expert,
            torch.zeros_like(activation_per_expert),
        )
        activation_scale_per_expert = torch.where(
            valid_token_mask,
            activation_scale_per_expert,
            torch.zeros_like(activation_scale_per_expert),
        )
        y = torch.empty(
            (activation_per_expert.shape[0], activation_per_expert.shape[1], self.dim),
            dtype=output_dtype,
            device=activation_per_expert.device,
        )
        for i in range(self.group_size):
            gate_up = self._w8a8_expert_gemm(
                activation_per_expert[i],
                self.gate_up_proj_weight[i],
                self.gate_up_proj_weight_scale[i],
                x_scale=activation_scale_per_expert[i],
                out_dtype=output_dtype,
            )
            act = self.forward_act_fn_merged(gate_up)
            y[i] = self._w8a8_expert_gemm(
                act,
                self.down_proj_weight[i],
                self.down_proj_weight_scale[i],
                out_dtype=output_dtype,
            )
        return PerExpertDenseBatchedExpertResultMinimal(y)

    @forward_no_sum.register
    def _(
        self,
        routed_x: IndexedBatchedRoutedActivation,
        impl: str = "auto",
    ) -> PerTokenBatchedExpertResult:
        return self._forward_no_sum_indexed(routed_x)

    @forward_no_sum.register
    def _(
        self,
        routed_x: IndexedBatchedRoutedActivationWithPaddedPerExpertCnt,
        impl: str = "auto",
    ) -> PerTokenBatchedExpertResult:
        return self._forward_no_sum_indexed(routed_x)

    def _get_lightop_sorted_and_expert_ids(
        self,
        routed_x: IndexedBatchedRoutedActivation,
        *,
        block_size_m: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not routed_x.expert_ids_are_local:
            routed_x = routed_x.as_local_expert_ids(
                self.experts_start_idx, self.experts_end_idx
            )
        hidden_states = ExpertBlockIndexedBatchedRoutedActivation.convert_from(
            routed_x, n_experts=self.group_size, block_size=block_size_m
        )
        sorted_token_ids_flat = hidden_states.block_to_token_x_topk_indices.flatten()
        sorted_token_ids_flat = sorted_token_ids_flat.to(torch.int32).contiguous()
        expert_ids = hidden_states.block_to_expert_indices.to(torch.int32).contiguous()
        num_tokens_post_pad = torch.tensor(
            [sorted_token_ids_flat.numel()],
            dtype=torch.int32,
            device=sorted_token_ids_flat.device,
        )
        return sorted_token_ids_flat, expert_ids, num_tokens_post_pad

    def _run_lightop_indexed(
        self,
        routed_x: IndexedBatchedRoutedActivation,
        *,
        topk_weights: torch.Tensor,
    ) -> tuple[IndexedBatchedRoutedActivation, Optional[torch.Tensor]]:
        if not routed_x.expert_ids_are_local:
            routed_x = routed_x.as_local_expert_ids(
                self.experts_start_idx, self.experts_end_idx
            )
        M = routed_x.activation.shape[0]
        topk = topk_weights.shape[1]
        if M == 0:
            return routed_x, torch.empty(
                (0, topk, self.dim),
                dtype=routed_x.activation.dtype,
                device=routed_x.activation.device,
            )
        w1 = self.gate_up_proj_weight
        w1_scale_3d = (
            self.gate_up_proj_weight_scale.to(torch.float32)
            .view(self.group_size, self.moe_inter_dim * 2, 1)
            .contiguous()
        )
        w2 = self.down_proj_weight
        w2_scale_3d = (
            self.down_proj_weight_scale.to(torch.float32)
            .view(self.group_size, self.dim, 1)
            .contiguous()
        )
        cfg1, cfg2, has_tuned_cfg = _lightop_get_moe_cuda_config(
            E=w1.shape[0],
            M=M,
            N1=w1.shape[1],
            K1=w1.shape[2],
            N2=w2.shape[1],
            K2=w2.shape[2],
            topk=topk,
        )
        if not has_tuned_cfg:
            return routed_x, None
        sorted_token_ids_flat, expert_ids, num_tokens_post_pad = (
            self._get_lightop_sorted_and_expert_ids(
                routed_x, block_size_m=int(cfg1["BLOCK_SIZE_M"])
            )
        )
        q_x, a1_scale_1d = a8_per_token_act_quant(routed_x.activation)
        q_x = q_x.contiguous()
        a1_scale = a1_scale_1d.to(torch.float32).reshape(M, 1).contiguous()
        intermediate_cache1 = torch.zeros(
            (M, topk, w1.shape[1]),
            device=routed_x.activation.device,
            dtype=torch.bfloat16,
        )
        lightop.moe_gemm_w8a8(
            q_x,
            w1,
            intermediate_cache1,
            a1_scale,
            w1_scale_3d,
            None,
            sorted_token_ids_flat,
            expert_ids,
            num_tokens_post_pad,
            topk,
            cfg1,
        )
        intermediate_cache2 = eval_lazy(silu_and_mul(intermediate_cache1))
        intermediate_cache2_2d = intermediate_cache2.reshape(
            -1, intermediate_cache2.shape[-1]
        )
        q_intermediate_cache2, a2_scale_1d = a8_per_token_act_quant(
            intermediate_cache2_2d
        )
        q_intermediate_cache2 = q_intermediate_cache2.contiguous()
        a2_scale = a2_scale_1d.to(torch.float32).reshape(-1, 1).contiguous()
        intermediate_cache3 = torch.zeros(
            (M, topk, w2.shape[1]),
            device=routed_x.activation.device,
            dtype=torch.bfloat16,
        )
        lightop.moe_gemm_w8a8(
            q_intermediate_cache2,
            w2,
            intermediate_cache3,
            a2_scale,
            w2_scale_3d,
            topk_weights.to(torch.float32).contiguous(),
            sorted_token_ids_flat,
            expert_ids,
            num_tokens_post_pad,
            1,
            cfg2,
        )
        return routed_x, intermediate_cache3

    def _forward_indexed(
        self,
        routed_x: IndexedBatchedRoutedActivation,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        routed_x, per_token_topk = self._run_lightop_indexed(
            routed_x,
            topk_weights=weights,
        )
        if per_token_topk is None:
            raise RuntimeError("LightOP W8A8 MoE tuned config is not available.")
        return per_token_topk.sum(dim=1).to(routed_x.activation.dtype)

    def _forward_no_sum_indexed(
        self,
        routed_x: IndexedBatchedRoutedActivation,
    ) -> PerTokenBatchedExpertResult:
        topk = routed_x.token_to_expert_indices.shape[1]
        routed_x, per_token_topk = self._run_lightop_indexed(
            routed_x,
            topk_weights=torch.ones(
                (routed_x.activation.shape[0], topk),
                dtype=torch.float32,
                device=routed_x.activation.device,
            ),
        )
        if per_token_topk is None:
            raise RuntimeError("LightOP W8A8 MoE tuned config is not available.")
        return PerTokenBatchedExpertResult(per_token_topk)


@QuantizationRegistry.register_moe_experts(
    W8A8_PER_TOKEN_PER_CHANNEL_DYN,
    merge_gate_up=True,
    when=_hygon_backend_w8a8_per_token_per_channel_dyn_use_deepgemm_moe,
    priority=13,
)
class HygonDeepGemmW8A8PerTokenPerChannelDynMoeExpertsMerged(
    NativeLayoutMixin, W8A8PerTokenPerChannelDynMoeExpertsMergedBase
):
    """W8A8 per-token/per-channel dynamic MoE on Hygon using DeepGEMM."""

    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(
            self.gate_up_proj_weight, HygonDeepGemmW8A8Marlin2Weight
        )
        self.apply_native_layout(self.down_proj_weight, HygonDeepGemmW8A8Marlin2Weight)

    @override
    @functools.singledispatchmethod
    def forward(
        self,
        routed_x: BatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        raise RuntimeError(
            "DeepGEMM MoE only supports IndexedBatchedRoutedActivation "
            f"in prefill path, got {type(routed_x).__name__}."
        )

    @forward.register
    def _(
        self,
        routed_x: IndexedBatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        q_activation, act_scale = a8_per_token_act_quant(routed_x.activation)
        quantized_routed_x = IndexedBatchedRoutedActivationWithScale(
            activation=q_activation,
            activation_scale=act_scale.unsqueeze(-1),
            token_to_expert_indices=routed_x.token_to_expert_indices,
            quant_method=W8A8_PER_TOKEN_PER_CHANNEL_DYN,
            expert_ids_are_local=routed_x.expert_ids_are_local,
            expected_n_tokens_per_expert=routed_x.expected_n_tokens_per_expert,
        )
        return self._forward_deepgemm_indexed(quantized_routed_x, weights)

    @forward.register
    def _(
        self,
        routed_x: IndexedBatchedRoutedActivationWithScale,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        return self._forward_deepgemm_indexed(routed_x, weights)

    @override
    def forward_no_sum(
        self, routed_x: BatchedRoutedActivation, impl: str = "auto"
    ) -> BatchedExpertResult:
        if isinstance(routed_x, PerExpertDenseBatchedRoutedActivationMinimal):
            return self._forward_deepgemm_masked_per_expert_dense(routed_x)
        raise RuntimeError(f"DeepGEMM MoE does not support {type(routed_x).__name__}.")

    def _forward_deepgemm_indexed(
        self,
        routed_x: IndexedBatchedRoutedActivationWithScale,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        output_dtype = torch.get_default_dtype()
        local_routed_x = routed_x.as_local_expert_ids(
            self.experts_start_idx,
            self.experts_end_idx,
        )
        padded_routed_x = (
            IndexedBatchedRoutedActivationWithScaleAndPaddedPerExpertCnt.convert_from(
                local_routed_x,
                n_experts=self.group_size,
                pad_block_size=_DEEPGEMM_MOE_BLOCK_SIZE,
            )
        )
        blocked_routed_x = (
            ExpertBlockPermutedBatchedRoutedActivationWithScale.convert_from(
                padded_routed_x,
                block_size=_DEEPGEMM_MOE_BLOCK_SIZE,
                num_experts=self.group_size,
            )
        )
        blocked_activation = blocked_routed_x.blocked_activation.contiguous()
        n_blocks, block_size, _ = blocked_activation.shape
        block_to_expert_indices = blocked_routed_x.block_to_expert_indices.to(
            torch.int32
        ).contiguous()
        block_to_expert_indices = _fill_deepgemm_padding_m_indices(
            block_to_expert_indices,
            num_experts=self.group_size,
        )
        m_indices = block_to_expert_indices.reshape(-1).contiguous()
        q_x = blocked_activation.view(-1, self.dim)
        blocked_scale = blocked_routed_x.blocked_activation_scale
        act_scale = (
            blocked_scale.contiguous()
            .view(-1, blocked_scale.shape[-1])[:, 0]
            .to(torch.float32)
            .contiguous()
        )
        if q_x.shape[0] == 0:
            return torch.zeros(
                (weights.shape[0], self.dim),
                dtype=output_dtype,
                device=routed_x.activation.device,
            )
        gate_up_weight = HygonDeepGemmW8A8LegacyMarlinWeight.convert_from(
            self.get_native_layout_gate_up_proj_weight()
        ).layout_tensor
        # Hygon DeepGEMM W8A8 requires fp32 scale tensors.
        gate_up_scale = (
            self.gate_up_proj_weight_scale.squeeze(-1).to(torch.float32).contiguous()
        )
        gate_up_out = torch.empty(
            (q_x.shape[0], self.moe_inter_dim * 2),
            dtype=output_dtype,
            device=routed_x.activation.device,
        )
        _call_deepgemm_w8a8_grouped_gemm(
            q_x,
            act_scale,
            gate_up_weight,
            gate_up_scale,
            m_indices,
            gate_up_out,
        )
        del gate_up_weight
        intermediate = silu_and_mul(gate_up_out, swiglu_limit=self.swiglu_limit)
        q_intermediate, intermediate_scale = a8_per_token_act_quant(intermediate)
        down_weight = HygonDeepGemmW8A8LegacyMarlinWeight.convert_from(
            self.get_native_layout_down_proj_weight()
        ).layout_tensor
        # Hygon DeepGEMM W8A8 requires fp32 scale tensors.
        down_scale = (
            self.down_proj_weight_scale.squeeze(-1).to(torch.float32).contiguous()
        )
        down_out = torch.empty(
            (q_intermediate.shape[0], self.dim),
            dtype=output_dtype,
            device=routed_x.activation.device,
        )
        _call_deepgemm_w8a8_grouped_gemm(
            q_intermediate.contiguous(),
            intermediate_scale.reshape(-1).to(torch.float32).contiguous(),
            down_weight,
            down_scale,
            m_indices,
            down_out,
        )
        expert_result = ExpertBlockPermutedBatchedExpertResult(
            blocked_activation=down_out.view(n_blocks, block_size, self.dim),
            token_comma_topk_to_block_x_item_indices=blocked_routed_x.token_comma_topk_to_block_x_item_indices,
        )
        return expert_result.weighted_sum(
            weights.to(device=routed_x.activation.device, dtype=torch.float32),
            out=None,
        )

    def _forward_deepgemm_masked_per_expert_dense(
        self,
        routed_x: PerExpertDenseBatchedRoutedActivationMinimal,
    ) -> PerExpertDenseBatchedExpertResultMinimal:
        routed_x = routed_x.as_local_expert_ids(
            self.experts_start_idx, self.experts_end_idx
        )
        gate_up_native = self.get_native_layout_gate_up_proj_weight()
        down_native = self.get_native_layout_down_proj_weight()
        gate_up_weight = gate_up_native.layout_tensor.contiguous()
        down_weight = down_native.layout_tensor.contiguous()
        _validate_indexed_moe_shapes(
            label="DeepGEMM masked MoE",
            activation=routed_x.activation_per_expert,
            w1=gate_up_weight,
            w2=down_weight,
            dim=self.dim,
            moe_inter_dim=self.moe_inter_dim,
            w1_plain_shape=gate_up_native.plain_shape,
            w2_plain_shape=down_native.plain_shape,
        )
        E, output_m, _ = routed_x.activation_per_expert.shape
        device = routed_x.activation_per_expert.device
        if output_m == 0:
            return PerExpertDenseBatchedExpertResultMinimal(
                torch.empty((E, 0, self.dim), dtype=torch.bfloat16, device=device)
            )
        masked_m = routed_x.n_tokens_per_expert.to(torch.int32).contiguous()
        layout_m = max(
            output_m,
            int(routed_x.expected_n_tokens_per_expert),
            _DEEPGEMM_MASKED_MIN_LAYOUT_M,
            1,
        )
        if isinstance(routed_x, PerExpertDenseBatchedRoutedActivationWithScaleMinimal):
            activation_per_expert, activation_scale_per_expert = (
                _prepare_w8a8_per_token_per_channel_dyn_per_expert_activation(routed_x)
            )
            q_x = _pad_deepgemm_masked_m(activation_per_expert, layout_m)
            act_scale = (
                _pad_deepgemm_masked_m(activation_scale_per_expert, layout_m)
                .to(torch.float32)
                .contiguous()
            )
        else:
            q_x, act_scale = _a8_per_token_act_quant_with_expert_mask(
                routed_x.activation_per_expert.contiguous(),
                masked_m,
                scale_dtype=torch.float32,
            )
            q_x = _pad_deepgemm_masked_m(q_x, layout_m)
            act_scale = _pad_deepgemm_masked_m(act_scale, layout_m)
        M = layout_m
        infer_args = get_global_args().infer
        expected_m_per_group = min(
            M,
            int(infer_args.max_batch_size)
            * max(1, int(getattr(infer_args, "mtp_size", 1) or 1)),
        )
        # Hygon DeepGEMM W8A8 requires fp32 scale tensors.
        gate_up_scale = self.gate_up_proj_weight_scale.to(torch.float32).contiguous()
        gate_up_out = torch.empty(
            (E, M, self.moe_inter_dim * 2),
            dtype=torch.bfloat16,
            device=device,
        )
        hygon_deepgemm.m_grouped_w8a8_gemm_nt_masked(
            (q_x, act_scale),
            (gate_up_weight, gate_up_scale),
            gate_up_out,
            masked_m,
            expected_m_per_group,
        )
        if self.swiglu_limit is None and _lightop_fuse_silu_mul_quant_ep_available():
            q_intermediate, intermediate_scale = lightop.fuse_silu_mul_quant_ep(
                gate_up_out,
                tokens_per_expert=masked_m,
                expect_m=M,
            )
            q_intermediate = q_intermediate.contiguous()
            intermediate_scale = intermediate_scale.to(torch.float32)
            if intermediate_scale.ndim == 2:
                intermediate_scale = intermediate_scale.unsqueeze(-1)
            intermediate_scale = intermediate_scale.contiguous()
        else:
            intermediate = eval_lazy(
                silu_and_mul(
                    gate_up_out,
                    expert_n_tokens=masked_m,
                    swiglu_limit=self.swiglu_limit,
                )
            )
            q_intermediate, intermediate_scale = (
                _a8_per_token_act_quant_with_expert_mask(
                    intermediate,
                    masked_m,
                    scale_dtype=torch.float32,
                )
            )
        # Hygon DeepGEMM W8A8 requires fp32 scale tensors.
        down_scale = self.down_proj_weight_scale.to(torch.float32).contiguous()
        down_out = torch.empty((E, M, self.dim), dtype=torch.bfloat16, device=device)
        hygon_deepgemm.m_grouped_w8a8_gemm_nt_masked(
            (q_intermediate, intermediate_scale),
            (down_weight, down_scale),
            down_out,
            masked_m,
            expected_m_per_group,
        )
        if output_m != M:
            down_out = down_out[:, :output_m, :].contiguous()
        return PerExpertDenseBatchedExpertResultMinimal(down_out)
