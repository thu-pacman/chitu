# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Hygon DCU W8A8 kernels for imported W8A8 linear and MoE layers.

The external GLM-5 W8A8 checkpoint uses Chitu's `w8a8_dynamic`
tensor naming. 
"""

import functools
import logging
import math
from typing import Any, Optional, Tuple

import torch
from typing_extensions import override

from chitu.device_type import get_device_name, is_hygon
from chitu.import_utils import try_import_platform_dep
from chitu.moe.batched_expert_result import (
    BatchedExpertResult,
    ExpertBlockPermutedBatchedExpertResult,
    PerExpertDenseBatchedExpertResultMinimal,
    PerTokenBatchedExpertResult,
)
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    ExpertBlockIndexedBatchedRoutedActivation,
    ExpertBlockPermutedBatchedRoutedActivationWithScale,
    IndexedBatchedRoutedActivation,
    IndexedBatchedRoutedActivationWithScale,
    IndexedBatchedRoutedActivationWithScaleAndPaddedPerExpertCnt,
    PerExpertDenseBatchedRoutedActivationMinimal,
    PerExpertDenseBatchedRoutedActivationWithScaleMinimal,
    IndexedBatchedRoutedActivationWithPaddedPerExpertCnt,
)
from chitu.native_layout import (
    AiterMoeCInt8Gemm1Weight,
    AiterMoeCInt8Gemm2Weight,
    HygonDeepGemmW8A8MarlinWeight,
    enable_native_layout_weight,
)
from chitu.ops import silu_and_mul
from chitu.ops.quant import a8_per_token_act_quant
from chitu.quantization.base import QuantizedMoeExpertsMerged
from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.w8a8_per_token_per_channel_dyn import (
    W8A8PerTokenPerChannelDynLinear,
)
from chitu.lazy import LazyTensor, eval_lazy

lmslim_quant_ops, has_lmslim_quant_ops = try_import_platform_dep(
    "lmslim.quantize.quant_ops"
)
aiter, has_aiter = try_import_platform_dep("aiter.moe")
deepgemm, has_deepgemm = try_import_platform_dep("deepgemm")
lightop, has_lightop = try_import_platform_dep("lightop")

_DEEPGEMM_MASKED_MIN_LAYOUT_M = 64
_DEEPGEMM_MOE_BLOCK_SIZE = 256
logger = logging.getLogger(__name__)


def _prepare_w8a8_dynamic_per_expert_activation(
    routed_x: PerExpertDenseBatchedRoutedActivationWithScaleMinimal,
) -> tuple[torch.Tensor, torch.Tensor]:
    if routed_x.quant_method != "w8a8_dynamic":
        raise RuntimeError(
            "DeepGEMM MoE only supports w8a8_dynamic pre-quantized "
            f"activations, got {routed_x.quant_method}."
        )

    activation_per_expert = routed_x.activation_per_expert.contiguous()
    if activation_per_expert.dtype != torch.int8:
        raise RuntimeError(
            "w8a8_dynamic pre-quantized DeepGEMM MoE activation must be "
            f"torch.int8, got {activation_per_expert.dtype}."
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
            "w8a8_dynamic pre-quantized DeepGEMM MoE activation scale must "
            f"have shape ({E}, {M}, 1), got "
            f"{tuple(activation_scale_per_expert.shape)}."
        )

    return activation_per_expert, activation_scale_per_expert


def _lightop_gemm_w8a8_smooth_available() -> bool:
    if not has_lightop:
        return False
    return callable(getattr(lightop, "gemm_w8a8_smooth", None))


def _lightop_fuse_silu_mul_quant_ep_available() -> bool:
    if not has_lightop:
        return False
    return callable(getattr(lightop, "fuse_silu_mul_quant_ep", None))


def _lightop_moe_gemm_w8a8_available() -> bool:
    if not has_lightop:
        return False
    return callable(getattr(lightop, "moe_gemm_w8a8", None))


def _a8_per_token_act_quant_with_expert_mask(
    activation_per_expert: torch.Tensor,
    n_tokens_per_expert: torch.Tensor,
    *,
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Fallback for non-prequantized per-expert inputs: only valid rows are
    # quantized, while padded rows are left zero for masked GEMM.
    E, M, hidden = activation_per_expert.shape
    q_activation = torch.zeros(
        (E, M, hidden),
        dtype=torch.int8,
        device=activation_per_expert.device,
    )
    activation_scale = torch.zeros(
        (E, M, 1),
        dtype=scale_dtype,
        device=activation_per_expert.device,
    )
    if M == 0:
        return q_activation, activation_scale

    n_tokens_per_expert_cpu = n_tokens_per_expert.detach().cpu().tolist()
    for expert_id, n_tokens in enumerate(n_tokens_per_expert_cpu):
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
        tensor,
        (0, 0, 0, layout_m - tensor.shape[1]),
        "constant",
        0,
    ).contiguous()


def _call_deepgemm_w8a8_grouped_gemm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    m_indices: torch.Tensor,
    out: torch.Tensor,
) -> None:
    deepgemm.m_grouped_i8_gemm_nt_contiguous(
        (a, a_scale),
        (weight, weight_scale),
        out,
        m_indices,
        {"MODE": 1000},
    )


def _fill_deepgemm_padding_m_indices(
    block_to_expert_indices: torch.Tensor,
    *,
    num_experts: int,
) -> torch.Tensor:
    """
    Replace block padding markers with a valid expert id for DeepGEMM.

    Chitu uses -1 to mark padded rows in the block-permuted routed activation.
    Hygon DeepGEMM's contiguous grouped GEMM indexes weight by m_indices, so -1
    becomes an invalid weight address instead of a skip marker. Padding outputs
    are not gathered back to real tokens, therefore it is safe to compute them
    against the block's expert as dummy rows.
    """

    valid = (block_to_expert_indices >= 0) & (block_to_expert_indices < num_experts)
    has_valid = valid.any(dim=1, keepdim=True)
    first_valid_pos = valid.to(torch.int32).argmax(dim=1, keepdim=True)
    block_expert_indices = block_to_expert_indices.gather(1, first_valid_pos)
    block_expert_indices = torch.where(
        has_valid,
        block_expert_indices,
        torch.zeros_like(block_expert_indices),
    )
    return torch.where(
        block_to_expert_indices < 0,
        block_expert_indices.expand_as(block_to_expert_indices),
        block_to_expert_indices,
    )


@QuantizationRegistry.register_linear(
    "w8a8_dynamic",
    when=lambda kwargs: is_hygon(),
    priority=12,
)
class HygonW8A8Linear(W8A8PerTokenPerChannelDynLinear):
    """
    Imported W8A8 linear on Hygon.

    Dense GEMM backend selection is delegated to
    `w8a8_gemm_per_token_per_channel(impl="auto")`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        weight_scale_dtype: torch.dtype = None,
    ):
        super().__init__(
            in_features,
            out_features,
            has_bias,
            weight_scale_dtype=weight_scale_dtype,
            weight_scale_has_singleton_last_dim=True,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            output_shape = (*x.shape[:-1], self.out_features)
            return torch.empty(output_shape, dtype=x.dtype, device=x.device)
        return super().forward(x)


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
            E,
            M,
            N1,
            K1,
            N2,
            K2,
            topk,
            device_name_for_cfg,
            str(num_cus),
            None,
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


@QuantizationRegistry.register_moe_experts(
    "w8a8_dynamic",
    merge_gate_up=True,
    when=lambda _: is_hygon()
    and _lightop_gemm_w8a8_smooth_available()
    and _lightop_moe_gemm_w8a8_available(),
    priority=1,
)
class W8A8MoeExpertsMergedHygonLightop(QuantizedMoeExpertsMerged):
    @override
    @functools.singledispatchmethod
    def forward_no_sum(self, routed_x: Any, impl: str = "auto") -> BatchedExpertResult:
        return super().forward_no_sum(routed_x, impl=impl)

    def __init__(
        self,
        dim: int,
        moe_inter_dim: int,
        global_n_experts: int,
        experts_start_idx: int,
        experts_end_idx: int,
        n_activated_experts: int,
        checkpoint_prefix: str,
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

        self.gate_up_proj_weight = torch.nn.Parameter(
            torch.empty(
                (self.group_size, moe_inter_dim * 2, self.dim),
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        self.gate_up_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                (self.group_size, moe_inter_dim * 2, 1),
                dtype=torch.get_default_dtype(),
            ),
            requires_grad=False,
        )
        self.down_proj_weight = torch.nn.Parameter(
            torch.empty(
                (self.group_size, self.dim, moe_inter_dim),
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        self.down_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                (self.group_size, self.dim, 1),
                dtype=torch.get_default_dtype(),
            ),
            requires_grad=False,
        )

    def _ensure_lightop_scales(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        w1_scale = self.gate_up_proj_weight_scale.to(torch.float32).contiguous()
        w2_scale = self.down_proj_weight_scale.to(torch.float32).contiguous()
        return self.gate_up_proj_weight, w1_scale, self.down_proj_weight, w2_scale

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
            if len(x_fp.shape) != 2:
                x_fp = eval_lazy(x_fp)
                x2 = x_fp.reshape(-1, x_fp.shape[-1])
            else:
                x2 = x_fp
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
        if routed_x.quant_method != "w8a8_dynamic":
            raise NotImplementedError(
                "Hygon W8A8 LightOP only supports w8a8_dynamic per-expert "
                "scaled activations."
            )
        output_dtype = routed_x.output_dtype or torch.get_default_dtype()

        activation_per_expert = routed_x.activation_per_expert.contiguous()
        activation_scale_per_expert = routed_x.activation_scale_per_expert.contiguous()
        row_ids = torch.arange(
            activation_per_expert.shape[1],
            device=activation_per_expert.device,
            dtype=routed_x.n_tokens_per_expert.dtype,
        ).unsqueeze(0)
        valid_token_mask = row_ids < routed_x.n_tokens_per_expert.unsqueeze(1)
        valid_token_mask = valid_token_mask.unsqueeze(-1)
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
            (
                activation_per_expert.shape[0],
                activation_per_expert.shape[1],
                self.dim,
            ),
            dtype=output_dtype,
            device=activation_per_expert.device,
        )

        for i in range(self.group_size):
            x_i = activation_per_expert[i]
            x_scale_i = activation_scale_per_expert[i]
            gate_up = self._w8a8_expert_gemm(
                x_i,
                self.gate_up_proj_weight[i],
                self.gate_up_proj_weight_scale[i],
                x_scale=x_scale_i,
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

        w1, w1_scale_3d, w2, w2_scale_3d = self._ensure_lightop_scales()
        N1 = w1.shape[1]
        N2 = w2.shape[1]
        K1 = w1.shape[2]
        K2 = w2.shape[2]

        cfg1, cfg2, has_tuned_cfg = _lightop_get_moe_cuda_config(
            E=w1.shape[0],
            M=M,
            N1=N1,
            K1=K1,
            N2=N2,
            K2=K2,
            topk=topk,
        )
        if not has_tuned_cfg:
            return routed_x, None

        block_size_m = int(cfg1["BLOCK_SIZE_M"])
        device = routed_x.activation.device
        cache_dtype = torch.bfloat16

        sorted_token_ids_flat, expert_ids, num_tokens_post_pad = (
            self._get_lightop_sorted_and_expert_ids(routed_x, block_size_m=block_size_m)
        )

        q_x, a1_scale_1d = a8_per_token_act_quant(routed_x.activation)
        q_x = q_x.contiguous()
        a1_scale = a1_scale_1d.to(torch.float32).reshape(M, 1).contiguous()

        intermediate_cache1 = torch.zeros(
            (M, topk, N1),
            device=device,
            dtype=cache_dtype,
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

        intermediate_cache2 = silu_and_mul(intermediate_cache1)
        q_intermediate_cache2, a2_scale_1d = a8_per_token_act_quant(intermediate_cache2)
        q_intermediate_cache2 = q_intermediate_cache2.contiguous()
        a2_scale = a2_scale_1d.to(torch.float32).reshape(-1, 1).contiguous()

        intermediate_cache3 = torch.zeros(
            (M, topk, N2),
            device=device,
            dtype=cache_dtype,
        )
        topk_weights_fp32 = topk_weights.to(torch.float32).contiguous()
        lightop.moe_gemm_w8a8(
            q_intermediate_cache2,
            w2,
            intermediate_cache3,
            a2_scale,
            w2_scale_3d,
            topk_weights_fp32,
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
            return self._forward_indexed_triton_fallback(routed_x, weights)
        return per_token_topk.sum(dim=1).to(routed_x.activation.dtype)

    def _triton_fallback_no_sum(
        self,
        routed_x: IndexedBatchedRoutedActivation,
    ) -> PerTokenBatchedExpertResult:
        from chitu.moe.experts.triton_fused_experts import fused_experts_int8

        w1, w1_scale_3d, w2, w2_scale_3d = self._ensure_lightop_scales()
        n_local = w1.shape[0]
        if not routed_x.expert_ids_are_local:
            routed_x = routed_x.as_local_expert_ids(
                self.experts_start_idx, self.experts_start_idx + n_local
            )
        return fused_experts_int8(
            hidden_states=routed_x,
            w1=w1,
            w2=w2,
            activation="silu",
            use_int8_w8a16=False,
            w1_scale=w1_scale_3d,
            w2_scale=w2_scale_3d,
            a1_scale=None,
            a2_scale=None,
            use_int8_w8a8=True,
            experts_start_idx=0,
        )

    def _forward_indexed_triton_fallback(
        self,
        routed_x: IndexedBatchedRoutedActivation,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        per_token_topk = self._triton_fallback_no_sum(routed_x).activation
        weighted = per_token_topk * weights.to(per_token_topk.dtype).unsqueeze(-1)
        return weighted.sum(dim=1).to(routed_x.activation.dtype)

    def _forward_no_sum_indexed_triton_fallback(
        self,
        routed_x: IndexedBatchedRoutedActivation,
    ) -> PerTokenBatchedExpertResult:
        return self._triton_fallback_no_sum(routed_x)

    def _forward_no_sum_indexed(
        self,
        routed_x: IndexedBatchedRoutedActivation,
    ) -> PerTokenBatchedExpertResult:
        topk = routed_x.token_to_expert_indices.shape[1]
        routed_x, per_token_topk = self._run_lightop_indexed(
            routed_x,
            topk_weights=torch.ones(
                (
                    routed_x.activation.shape[0],
                    topk,
                ),
                dtype=torch.float32,
                device=routed_x.activation.device,
            ),
        )
        if per_token_topk is None:
            return self._forward_no_sum_indexed_triton_fallback(routed_x)
        return PerTokenBatchedExpertResult(per_token_topk)


@QuantizationRegistry.register_moe_experts(
    "w8a8_dynamic",
    merge_gate_up=True,
    when=lambda _: is_hygon() and has_aiter,
    priority=10,
)
class HygonW8A8AiterMoeExpertsMerged(
    enable_native_layout_weight("gate_up_proj_weight", AiterMoeCInt8Gemm1Weight),
    enable_native_layout_weight("down_proj_weight", AiterMoeCInt8Gemm2Weight),
    QuantizedMoeExpertsMerged,
):
    """
    Imported W8A8 MoE on Hygon using Aiter MOE_C.

    `w8a8_dynamic` still describes checkpoint tensor layout. Native
    layout preprocessing prepares the Aiter MOE_C weight layouts at load time.
    """

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

        # The imported GLM-5 W8A8 checkpoint uses the same tensor naming/layout as
        # `w8a8_dynamic` after gate/up merging. Aiter MOE_C requires fp32
        # weight scales, so checkpoint bf16 scales are loaded into fp32 params.
        # - gate_up_proj_weight: [E, 2*moe_inter_dim, dim]
        # - gate_up_proj_weight_scale: [E, 2*moe_inter_dim, 1]
        # - down_proj_weight: [E, dim, moe_inter_dim]
        # - down_proj_weight_scale: [E, dim, 1]
        self.gate_up_proj_weight = torch.nn.Parameter(
            torch.empty(
                (self.group_size, moe_inter_dim * 2, self.dim),
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        self.gate_up_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                (self.group_size, moe_inter_dim * 2, 1),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        self.down_proj_weight = torch.nn.Parameter(
            torch.empty(
                (self.group_size, self.dim, moe_inter_dim),
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        self.down_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                (self.group_size, self.dim, 1),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

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
        out = aiter.aiter_moe(
            hidden_states=routed_x.activation,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=routed_x.token_to_expert_indices,
            moe_config=moe_config,
            inplace=False,
            activation="silu",
            w1_scale=self.gate_up_proj_weight_scale,
            w2_scale=self.down_proj_weight_scale,
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
        return out


@QuantizationRegistry.register_moe_experts(
    "w8a8_dynamic",
    merge_gate_up=True,
    when=lambda _: is_hygon() and has_deepgemm,
    priority=13,
)
class HygonW8A8DeepGemmMoeExpertsMerged(
    enable_native_layout_weight("gate_up_proj_weight", HygonDeepGemmW8A8MarlinWeight),
    enable_native_layout_weight("down_proj_weight", HygonDeepGemmW8A8MarlinWeight),
    QuantizedMoeExpertsMerged,
):
    """
    Imported W8A8 MoE on Hygon using DeepGEMM contiguous grouped GEMM.

    MORI/DeepEP low-latency dispatch feeds bf16 per-expert dense activations,
    therefore this backend quantizes activations locally and runs DeepGEMM's
    masked W8A8 grouped GEMM.
    """

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

        self.gate_up_proj_weight = torch.nn.Parameter(
            torch.empty(
                (self.group_size, moe_inter_dim * 2, self.dim),
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        self.gate_up_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                (self.group_size, moe_inter_dim * 2, 1),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        self.down_proj_weight = torch.nn.Parameter(
            torch.empty(
                (self.group_size, self.dim, moe_inter_dim),
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        self.down_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                (self.group_size, self.dim, 1),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

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
        routed_x: IndexedBatchedRoutedActivation,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        local_routed_x = routed_x.as_local_expert_ids(
            self.experts_start_idx,
            self.experts_end_idx,
        )
        q_activation, act_scale = a8_per_token_act_quant(local_routed_x.activation)
        act_scale = act_scale.unsqueeze(-1)
        quantized_routed_x = IndexedBatchedRoutedActivationWithScale(
            activation=q_activation,
            activation_scale=act_scale,
            token_to_expert_indices=local_routed_x.token_to_expert_indices,
            quant_method="w8a8_dynamic",
            expert_ids_are_local=local_routed_x.expert_ids_are_local,
            expected_n_tokens_per_expert=local_routed_x.expected_n_tokens_per_expert,
        )
        padded_routed_x = (
            IndexedBatchedRoutedActivationWithScaleAndPaddedPerExpertCnt.convert_from(
                quantized_routed_x,
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
        blocked_activation_scale = (
            blocked_routed_x.blocked_activation_scale.reshape(-1)
            .to(torch.float32)
            .contiguous()
        )
        n_blocks, block_size, _ = blocked_activation.shape
        block_to_expert_indices = blocked_routed_x.block_to_expert_indices.to(
            torch.int32
        ).contiguous()
        block_to_expert_indices = _fill_deepgemm_padding_m_indices(
            block_to_expert_indices,
            num_experts=self.group_size,
        )
        m_indices = block_to_expert_indices.reshape(-1).contiguous()
        q_x = blocked_activation.view(-1, self.dim).contiguous()

        gate_up_weight = (
            self.get_native_layout_gate_up_proj_weight().layout_tensor.contiguous()
        )
        gate_up_scale = self.gate_up_proj_weight_scale.squeeze(-1).contiguous()
        gate_up_out = torch.empty(
            (q_x.shape[0], self.moe_inter_dim * 2),
            dtype=routed_x.activation.dtype,
            device=routed_x.activation.device,
        )
        _call_deepgemm_w8a8_grouped_gemm(
            q_x,
            blocked_activation_scale,
            gate_up_weight,
            gate_up_scale,
            m_indices,
            gate_up_out,
        )

        intermediate = silu_and_mul(gate_up_out, swiglu_limit=self.swiglu_limit)
        q_intermediate, intermediate_scale = a8_per_token_act_quant(intermediate)
        down_weight = (
            self.get_native_layout_down_proj_weight().layout_tensor.contiguous()
        )
        down_scale = self.down_proj_weight_scale.squeeze(-1).contiguous()
        down_out = torch.empty(
            (q_intermediate.shape[0], self.dim),
            dtype=routed_x.activation.dtype,
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
            blocked_activation=down_out.view(
                n_blocks,
                block_size,
                self.dim,
            ),
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
        _, _, _, _, _ = _validate_indexed_moe_shapes(
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
                torch.empty(
                    (E, 0, self.dim),
                    dtype=torch.bfloat16,
                    device=device,
                )
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
                _prepare_w8a8_dynamic_per_expert_activation(routed_x)
            )
            activation_per_expert = _pad_deepgemm_masked_m(
                activation_per_expert, layout_m
            )
            activation_scale_per_expert = _pad_deepgemm_masked_m(
                activation_scale_per_expert, layout_m
            )
            q_x = activation_per_expert
            act_scale = activation_scale_per_expert.to(torch.float32).contiguous()
        else:
            q_x, act_scale = _a8_per_token_act_quant_with_expert_mask(
                routed_x.activation_per_expert.contiguous(),
                masked_m,
                scale_dtype=torch.float32,
            )
            q_x = _pad_deepgemm_masked_m(q_x, layout_m)
            act_scale = _pad_deepgemm_masked_m(act_scale, layout_m)

        M = layout_m
        expected_m = M
        gate_up_scale = self.gate_up_proj_weight_scale.to(torch.float32).contiguous()
        gate_up_out = torch.empty(
            (E, M, self.moe_inter_dim * 2),
            dtype=torch.bfloat16,
            device=device,
        )
        deepgemm.m_grouped_w8a8_gemm_nt_masked_impl(
            (q_x, act_scale),
            (gate_up_weight, gate_up_scale),
            gate_up_out,
            masked_m,
            expected_m,
            0,
        )

        if self.swiglu_limit is None and _lightop_fuse_silu_mul_quant_ep_available():
            # Hygon lightop EP fused path consumes the per-expert token counts
            # and avoids doing SiLU+quant work on layout padding.
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
        down_scale = self.down_proj_weight_scale.to(torch.float32).contiguous()
        down_out = torch.empty(
            (E, M, self.dim),
            dtype=torch.bfloat16,
            device=device,
        )
        deepgemm.m_grouped_w8a8_gemm_nt_masked_impl(
            (q_intermediate, intermediate_scale),
            (down_weight, down_scale),
            down_out,
            masked_m,
            expected_m,
            0,
        )
        if output_m != M:
            down_out = down_out[:, :output_m, :].contiguous()
        return PerExpertDenseBatchedExpertResultMinimal(down_out)
