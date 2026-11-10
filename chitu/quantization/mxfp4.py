# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from typing import Optional

from typing_extensions import override

from chitu.checkpoint_prefix import CheckpointPrefix
from chitu.global_vars import get_global_args
from chitu.utils import (
    ceil_div,
)
from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.base import (
    QuantizedLinearBase,
    QuantizedMoeExpertsMerged,
    QuantizedMoeExpertsUnmerged,
)
from chitu.native_layout import (
    NativeLayoutMixin,
    Packed4BitWeightAlongKContig,
)
from chitu.ops import (
    mxfp4_weight_dequant,
    situ_and_mul,
    linear,
)


MXFP4_GROUP_SIZE = 32


def _require_contiguous_packed_mxfp4_weight(weight: torch.nn.Parameter) -> None:
    """
    Guard for the dequant-based fallbacks in this file.

    The fallbacks dequantize `weight` as packed contiguously along K, which is the
    `Packed4BitWeightAlongKContig` layout set up by `MXFP4LinearBase` and
    `MXFP4MoeExpertsBase`. Optimized subclasses apply further native layouts
    (swizzled, tiled, ...) on top for their fused kernels; dequantizing those
    layouts as contiguous packing silently produces wrong values, so reject them.
    """

    if Packed4BitWeightAlongKContig.check_tensor(weight):
        return
    layout_cls = getattr(getattr(weight, "native_layout", None), "layout_cls", None)
    raise NotImplementedError(
        f"The MXFP4 dequant fallback requires the Packed4BitWeightAlongKContig "
        f"weight layout, but the weight is in {layout_cls}."
    )


@QuantizationRegistry.register_linear("mxfp4", when=lambda _: True, priority=-1000)
class MXFP4LinearBase(NativeLayoutMixin, QuantizedLinearBase):
    """
    MXFP4 quantized linear layer.

    This class itself is a portable but slow fallback. Inherit from this class
    to implement an optimized subclass.
    """

    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        *,
        ############################################
        # Parameters specific to this quantization
        block_size: int = 32,
    ):
        super().__init__(in_features, out_features, has_bias)
        assert block_size == 32, "MXFP4 block size must be 32"

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        self.weight = torch.nn.Parameter(
            torch.zeros(
                self.out_features,
                self.in_features,
            ),
            requires_grad=False,
        )

        self.weight_scale = torch.nn.Parameter(
            torch.ones(
                self.out_features,
                self.in_features // self.block_size,
                dtype=torch.uint8,
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

    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(
            self.weight, Packed4BitWeightAlongKContig, state_dict_convert=False
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _require_contiguous_packed_mxfp4_weight(self.weight)
        weight = mxfp4_weight_dequant(
            self.weight, self.weight_scale, self.block_size
        ).to(torch.get_default_dtype())
        return linear(x, weight, self.bias)


@QuantizationRegistry.register_moe_experts(
    "mxfp4", merge_gate_up=True, when=lambda _: True, priority=-1000
)
class MXFP4MoeExpertsBase(NativeLayoutMixin, QuantizedMoeExpertsMerged):
    """
    MXFP4 quantized MoeExperts

    This class itself is a portable but slow fallback. Inherit from this class
    to implement an optimized subclass.
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
        checkpoint_prefix: str | CheckpointPrefix,
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

        self.block_size = 32
        self.moe_inter_dim = moe_inter_dim
        scale_in_features = ceil_div(dim, self.block_size)
        self.gate_up_proj_weight = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                moe_inter_dim * 2,
                dim,
            ),
            requires_grad=False,
        )
        self.gate_up_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                moe_inter_dim * 2,
                scale_in_features,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        down_proj_scale_in_features = ceil_div(moe_inter_dim, self.block_size)
        down_proj_scale_out_features = dim
        self.down_proj_weight = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                dim,
                moe_inter_dim,
            ),
            requires_grad=False,
        )
        self.down_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                down_proj_scale_out_features,
                down_proj_scale_in_features,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )

    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(
            self.gate_up_proj_weight,
            Packed4BitWeightAlongKContig,
            state_dict_convert=False,
        )
        self.apply_native_layout(
            self.down_proj_weight,
            Packed4BitWeightAlongKContig,
            state_dict_convert=False,
        )

    def forward_ith_expert_gate_up(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Routed activations are never quantized for mxfp4 experts: the token
        # dispatchers only fuse-quant for blockfp8/blockfp4 experts.
        assert x_scale is None
        _require_contiguous_packed_mxfp4_weight(self.gate_up_proj_weight)
        weight = mxfp4_weight_dequant(
            self.gate_up_proj_weight[i],
            self.gate_up_proj_weight_scale[i],
            self.block_size,
        ).to(torch.get_default_dtype())
        return linear(x, weight)

    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        _require_contiguous_packed_mxfp4_weight(self.down_proj_weight)
        weight = mxfp4_weight_dequant(
            self.down_proj_weight[i],
            self.down_proj_weight_scale[i],
            self.block_size,
        ).to(torch.get_default_dtype())
        return linear(x, weight)

    def forward_act_fn_merged(self, gate_up_out: torch.Tensor) -> torch.Tensor:
        activation, situ_beta, situ_linear_beta = _moe_activation_params()
        if activation == "situ":
            return situ_and_mul(
                gate_up_out, beta=situ_beta, linear_beta=situ_linear_beta
            )
        return super().forward_act_fn_merged(gate_up_out)


class MXFP4MoeExpertsUnmergedBase(QuantizedMoeExpertsUnmerged):
    """Shared implementation of unmerged MXFP4 routed experts.

    Not registered by itself: backends select it via
    ``QuantizationRegistry.register_moe_experts(..., merge_gate_up=False)``
    (see :mod:`chitu.quantization.mxfp4_gcu`), so device-specific behaviour is
    expressed by the registration rather than by a separate implementation.
    """

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

        scale_in_features = ceil_div(dim, MXFP4_GROUP_SIZE)
        down_scale_in_features = ceil_div(moe_inter_dim, MXFP4_GROUP_SIZE)

        self.gate_proj_weight = torch.nn.Parameter(
            torch.empty(self.group_size, moe_inter_dim, dim // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.gate_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                self.group_size, moe_inter_dim, scale_in_features, dtype=torch.uint8
            ),
            requires_grad=False,
        )
        self.up_proj_weight = torch.nn.Parameter(
            torch.empty(self.group_size, moe_inter_dim, dim // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.up_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                self.group_size, moe_inter_dim, scale_in_features, dtype=torch.uint8
            ),
            requires_grad=False,
        )
        self.down_proj_weight = torch.nn.Parameter(
            torch.empty(self.group_size, dim, moe_inter_dim // 2, dtype=torch.uint8),
            requires_grad=False,
        )
        self.down_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                dim,
                down_scale_in_features,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )

    @staticmethod
    def _dequant_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return mxfp4_weight_dequant(weight, scale, MXFP4_GROUP_SIZE).to(
            torch.get_default_dtype()
        )

    @override
    def forward_ith_expert_gate(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x_scale is not None:
            raise ValueError(
                "The MXFP4 BF16 fallback does not accept quantized activations"
            )
        return F.linear(
            x,
            self._dequant_weight(
                self.gate_proj_weight[i], self.gate_proj_weight_scale[i]
            ),
        )

    @override
    def forward_ith_expert_up(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x_scale is not None:
            raise ValueError(
                "The MXFP4 BF16 fallback does not accept quantized activations"
            )
        return F.linear(
            x,
            self._dequant_weight(
                self.up_proj_weight[i], self.up_proj_weight_scale[i]
            ),
        )

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self._dequant_weight(
                self.down_proj_weight[i], self.down_proj_weight_scale[i]
            ),
        )


def _moe_activation_params() -> tuple[str, float, Optional[float]]:
    """Activation name + situ betas for MoE experts, from the model config.

    Returns ("silu", 1.0, None) when the model does not configure situ (also the
    safe fallback for unit tests that never set global args).
    """
    try:
        models = get_global_args().models
        if getattr(models, "hidden_act", "silu") == "situ":
            beta = getattr(models, "activation_situ_beta", None)
            linear_beta = getattr(models, "activation_situ_linear_beta", None)
            return "situ", float(beta) if beta is not None else 1.0, linear_beta
    except Exception:
        pass
    return "silu", 1.0, None
