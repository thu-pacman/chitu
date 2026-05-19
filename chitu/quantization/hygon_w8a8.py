# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Hygon DCU W8A8 kernels for imported W8A8 linear and MoE layers.

The external GLM-5 W8A8 checkpoint uses Chitu's `w8a8_dynamic`
tensor naming. Backend selection is explicit in quant config:
- `linear_backend`: dense linear GEMM backend, currently `blaslt`
- `moe_backend`: MoE/group GEMM backend, currently `aiter` or `deepgemm`
"""

import functools
from typing import Any, Optional, Tuple

import torch
from typing_extensions import override

from chitu.device_type import is_hygon
from chitu.import_utils import try_import_platform_dep
from chitu.lazy import eval_lazy
from chitu.moe.batched_expert_result import (
    BatchedExpertResult,
    ExpertBlockPermutedBatchedExpertResult,
)
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    ExpertBlockPermutedBatchedRoutedActivationNormal,
    IndexedBatchedRoutedActivation,
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
from chitu.quantization.base import QuantizedLinearBase, QuantizedMoeExpertsMerged
from chitu.quantization.registry import QuantizationRegistry

_lmslim_quant_ops, _has_lmslim_quant_ops = try_import_platform_dep(
    "lmslim.quantize.quant_ops"
)
_aiter, _has_aiter = try_import_platform_dep("aiter.moe")
_deepgemm, _has_deepgemm = try_import_platform_dep("deepgemm")

_SUPPORTED_LINEAR_BACKENDS = {"blaslt"}
_SUPPORTED_MOE_BACKENDS = {"aiter", "deepgemm"}
_DEEPGEMM_MOE_BLOCK_SIZE = 256


def _call_hipblaslt_w8a8_gemm(
    a: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> Tuple[bool, Optional[torch.Tensor]]:
    """BLASLt contract: input [M, K], weight [N, K], trans="NT"."""
    m = a.shape[0]
    k = a.shape[1]
    n = weight.shape[0]
    status, out = _lmslim_quant_ops.hipblaslt_w8a8_gemm(
        a,
        weight,
        scale_a,
        scale_b,
        m,
        n,
        k,
        "NT",
        out_dtype,
    )
    if status is False or out is None:
        return False, None
    return True, out


def _call_deepgemm_w8a8_grouped_gemm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    m_indices: torch.Tensor,
    out: torch.Tensor,
) -> None:
    _deepgemm.m_grouped_i8_gemm_nt_contiguous(
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
    when=lambda kwargs: kwargs.get("linear_backend") == "blaslt",
    priority=10,
)
class HygonW8A8BlasltLinear(QuantizedLinearBase):
    """
    Imported W8A8 linear on Hygon using BLASLt.

    Checkpoint layout:
    - `weight`: int8 `[out, in]`
    - `weight_scale`: bf16 `[out, 1]`, loaded as fp32 for BLASLt
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        weight_scale_dtype: torch.dtype = None,
        linear_backend: str = "blaslt",
    ):
        super().__init__(in_features, out_features, has_bias)
        self.linear_backend = linear_backend
        self._check_backend_available()

        self.weight = torch.nn.Parameter(
            torch.zeros(
                self.out_features,
                self.in_features,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        # BLASLt requires fp32 scale tensors. Keep `weight_scale_dtype` in the
        # signature for quant-config compatibility, but store scale_b as fp32.
        self.weight_scale = torch.nn.Parameter(
            torch.ones(
                self.out_features,
                1,
                dtype=torch.float32,
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

    def _check_backend_available(self) -> None:
        if self.linear_backend != "blaslt":
            raise ValueError(
                "Unsupported Hygon W8A8 linear_backend="
                f"{self.linear_backend!r}. Supported backends are "
                f"{sorted(_SUPPORTED_LINEAR_BACKENDS)}."
            )
        if not is_hygon():
            raise ValueError(
                "Hygon W8A8 linear backend is unavailable for explicit "
                "linear_backend='blaslt'. Check that the current platform is Hygon."
            )
        if not _has_lmslim_quant_ops or not callable(
            getattr(_lmslim_quant_ops, "hipblaslt_w8a8_gemm", None)
        ):
            raise ValueError(
                "Hygon W8A8 linear backend is unavailable for explicit "
                "linear_backend='blaslt'. Check that "
                "lmslim.quantize.quant_ops.hipblaslt_w8a8_gemm is installed."
            )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = eval_lazy(x)
        out_leading = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        q_x, act_scale = a8_per_token_act_quant(x2)
        m = q_x.shape[0]
        scale_a = act_scale.reshape(m, 1).to(torch.float32)
        scale_b = self.weight_scale
        ok, out = _call_hipblaslt_w8a8_gemm(
            q_x,
            self.weight,
            scale_a,
            scale_b,
            x.dtype,
        )
        if not ok:
            raise RuntimeError(
                "hipblaslt_w8a8_gemm failed for explicit "
                f"linear_backend=blaslt: q_x={tuple(q_x.shape)} "
                f"weight={tuple(self.weight.shape)} "
                f"scale_a={tuple(scale_a.shape)} scale_b={tuple(scale_b.shape)}"
            )
        if self.bias is not None:
            out += self.bias
        return out.view(*out_leading, self.out_features)


@QuantizationRegistry.register_linear(
    "w8a8_dynamic",
    when=lambda kwargs: "linear_backend" in kwargs
    and kwargs.get("linear_backend") not in _SUPPORTED_LINEAR_BACKENDS,
    priority=9,
)
class HygonW8A8LinearUnavailable(QuantizedLinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        *,
        linear_backend: str,
        **kwargs,
    ):
        del kwargs
        super().__init__(in_features, out_features, has_bias)
        raise ValueError(
            "Hygon W8A8 linear backend is unavailable for explicit "
            f"linear_backend={linear_backend!r}. Supported backends are "
            f"{sorted(_SUPPORTED_LINEAR_BACKENDS)}."
        )


def _validate_indexed_moe_shapes(
    *,
    label: str,
    activation: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    dim: int,
    moe_inter_dim: int,
) -> tuple[int, int, int, int, int]:
    if w1.dim() != 3 or w2.dim() != 3:
        raise ValueError(
            f"{label} weights must be 3D: w1={tuple(w1.shape)} w2={tuple(w2.shape)}"
        )
    E, N1, K1 = w1.shape
    _, N2, K2 = w2.shape
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
            f"w1={tuple(w1.shape)} w2={tuple(w2.shape)} "
            f"expected dim={dim} moe_inter_dim={moe_inter_dim}"
        )
    return E, N1, K1, N2, K2


@QuantizationRegistry.register_moe_experts(
    "w8a8_dynamic",
    merge_gate_up=True,
    when=lambda kwargs: kwargs.get("moe_backend") == "aiter",
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
        moe_backend: str = "aiter",
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
        self.moe_backend = moe_backend
        self._check_backend_available()

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

    def _check_backend_available(self) -> None:
        if self.moe_backend != "aiter":
            raise ValueError(
                "Unsupported Hygon W8A8 moe_backend="
                f"{self.moe_backend!r}. Supported backends are "
                f"{sorted(_SUPPORTED_MOE_BACKENDS)}."
            )
        if not is_hygon():
            raise ValueError(
                "Hygon W8A8 MoE backend is unavailable for explicit "
                "moe_backend='aiter'. Check that the current platform is Hygon."
            )
        if not _has_aiter or not (
            callable(getattr(_aiter, "get_aiter_moe_config", None))
            and callable(getattr(_aiter, "aiter_moe", None))
        ):
            raise ValueError(
                "Hygon W8A8 MoE backend is unavailable for explicit "
                "moe_backend='aiter'. Check that aiter.moe.get_aiter_moe_config "
                "and aiter.moe.aiter_moe are installed."
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
        found, moe_config = _aiter.get_aiter_moe_config(
            M=M,
            E=E,
            N1=N1,
            N2=N2,
            K=K,
            top_k=topk,
            block_size=0,
            dtype=routed_x.activation.dtype,
            quant_type=_aiter.MoeQuantType.W8A8,
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
        out = _aiter.aiter_moe(
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
    when=lambda kwargs: kwargs.get("moe_backend") == "deepgemm",
    priority=11,
)
class HygonW8A8DeepGemmMoeExpertsMerged(
    enable_native_layout_weight("gate_up_proj_weight", HygonDeepGemmW8A8MarlinWeight),
    enable_native_layout_weight("down_proj_weight", HygonDeepGemmW8A8MarlinWeight),
    QuantizedMoeExpertsMerged,
):
    """
    Imported W8A8 MoE on Hygon using DeepGEMM contiguous grouped GEMM.

    This follows the existing Aiter prefill interface: only indexed routed
    activations are supported and EP-local execution is intentionally rejected.
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
        moe_backend: str = "deepgemm",
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
        self.moe_backend = moe_backend
        self._check_backend_available()

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

    def _check_backend_available(self) -> None:
        if self.moe_backend != "deepgemm":
            raise ValueError(
                "Unsupported Hygon W8A8 moe_backend="
                f"{self.moe_backend!r}. Supported backends are "
                f"{sorted(_SUPPORTED_MOE_BACKENDS)}."
            )
        if not is_hygon():
            raise ValueError(
                "Hygon W8A8 DeepGEMM MoE backend is unavailable for explicit "
                "moe_backend='deepgemm'. Check that the current platform is Hygon."
            )
        if not _has_deepgemm or not callable(
            getattr(_deepgemm, "m_grouped_i8_gemm_nt_contiguous", None)
        ):
            raise ValueError(
                "Hygon W8A8 DeepGEMM MoE backend requires "
                "deepgemm.m_grouped_i8_gemm_nt_contiguous."
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
        raise RuntimeError(
            "DeepGEMM MoE does not support forward_no_sum in prefill path."
        )

    def _forward_deepgemm_indexed(
        self,
        routed_x: IndexedBatchedRoutedActivation,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if not (
            self.experts_start_idx == 0
            and self.experts_end_idx == self.global_n_experts
        ):
            raise RuntimeError(
                "DeepGEMM MoE requires all experts to be local in prefill path."
            )
        if routed_x.activation.shape[-1] != self.dim:
            raise RuntimeError(
                f"DeepGEMM MoE activation dim mismatch: "
                f"got {routed_x.activation.shape[-1]}, expected {self.dim}."
            )

        local_routed_x = routed_x.as_local_expert_ids(
            self.experts_start_idx,
            self.experts_end_idx,
        )
        padded_routed_x = (
            IndexedBatchedRoutedActivationWithPaddedPerExpertCnt.convert_from(
                local_routed_x,
                n_experts=self.group_size,
                pad_block_size=_DEEPGEMM_MOE_BLOCK_SIZE,
            )
        )
        blocked_routed_x = (
            ExpertBlockPermutedBatchedRoutedActivationNormal.convert_from(
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
        flat_activation = blocked_activation.view(-1, self.dim)
        q_x, act_scale = a8_per_token_act_quant(flat_activation)

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
            q_x.contiguous(),
            act_scale.reshape(-1).to(torch.float32).contiguous(),
            gate_up_weight,
            gate_up_scale,
            m_indices,
            gate_up_out,
        )

        intermediate = eval_lazy(silu_and_mul(gate_up_out))
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


@QuantizationRegistry.register_moe_experts(
    "w8a8_dynamic",
    merge_gate_up=True,
    when=lambda kwargs: "moe_backend" in kwargs
    and kwargs.get("moe_backend") not in _SUPPORTED_MOE_BACKENDS,
    priority=9,
)
class HygonW8A8MoeExpertsUnavailable(QuantizedMoeExpertsMerged):
    def __init__(
        self,
        dim: int,
        moe_inter_dim: int,
        global_n_experts: int,
        experts_start_idx: int,
        experts_end_idx: int,
        n_activated_experts: int,
        checkpoint_prefix: str,
        *,
        moe_backend: str,
        **kwargs,
    ):
        del kwargs
        super().__init__(
            dim,
            moe_inter_dim,
            global_n_experts,
            experts_start_idx,
            experts_end_idx,
            n_activated_experts,
            checkpoint_prefix,
        )
        raise ValueError(
            "Hygon W8A8 MoE backend is unavailable for explicit "
            f"moe_backend={moe_backend!r}. Supported backends are "
            f"{sorted(_SUPPORTED_MOE_BACKENDS)}."
        )
