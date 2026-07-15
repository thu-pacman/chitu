# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import functools

import torch
import torch.nn as nn

from chitu.lazy import eval_lazy
from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.base import (
    QuantizedMoeExpertsMerged,
    QuantizedMoeExpertsUnmerged,
)
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    ExpertBlockIndexedBatchedRoutedActivation,
)
from chitu.moe.batched_expert_result import (
    BatchedExpertResult,
    PerTokenBatchedExpertResult,
)
from chitu.utils import try_import_platform_dep
from chitu.native_layout import (
    NativeLayoutMixin,
    Packed4BitWeightAlongKContigInt32,
    BlockInt4MarlinQWeight,
    BlockInt4MarlinScale,
)
from chitu.quantization.gptqmodel import marlin_make_empty_g_idx

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
triton, has_triton = try_import_platform_dep("triton")
has_marlin = has_chitu_backend and hasattr(chitu_backend, "gptq_marlin_gemm")
has_marlin_moe = has_chitu_backend and hasattr(chitu_backend, "moe_wna16_marlin_gemm")
aiter, has_aiter = try_import_platform_dep("aiter")

if has_aiter:
    from aiter.moe import aiter_moe, get_aiter_moe_config, MoeQuantType, MoeSolutionType
    from aiter.ops.shuffle import w4a16_marlin_weight_1, w4a16_marlin_weight_2
if has_triton:
    from chitu.moe.experts.triton_batched_experts import (
        invoke_fused_moe_wna16_triton_kernel,
    )

# ScalarTypeId for ScalarType::uint4b8.
# Computed from scalar_type.hpp: ScalarType(exponent=0, mantissa=4, signed_=false, bias=8)
# with NAN_IEEE_754=1, packed as: exponent(8b)|mantissa(8b)|signed_(1b)|bias(32b)|finite(1b)|nan_repr(8b)
UINT4B8_TYPE_ID = 1125899907892224


class BlockInt4MoeExpertsUnmergedBase(NativeLayoutMixin, QuantizedMoeExpertsUnmerged):
    """blockint4 MoE experts with separated gate and up projections."""

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
        group_size: int = 128,
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
        self.quant_group_size = group_size

        self.gate_proj_qweight = nn.Parameter(
            torch.empty(self.group_size, moe_inter_dim, dim), requires_grad=False
        )
        self.gate_proj_scales = nn.Parameter(
            torch.empty(
                self.group_size,
                moe_inter_dim,
                dim // group_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.up_proj_qweight = nn.Parameter(
            torch.empty(self.group_size, moe_inter_dim, dim), requires_grad=False
        )
        self.up_proj_scales = nn.Parameter(
            torch.empty(
                self.group_size,
                moe_inter_dim,
                dim // group_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.down_proj_qweight = nn.Parameter(
            torch.empty(self.group_size, dim, moe_inter_dim), requires_grad=False
        )
        self.down_proj_scales = nn.Parameter(
            torch.empty(
                self.group_size,
                dim,
                moe_inter_dim // group_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )

    @override
    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(
            self.gate_proj_qweight,
            Packed4BitWeightAlongKContigInt32,
            state_dict_convert=False,
        )
        self.apply_native_layout(
            self.up_proj_qweight,
            Packed4BitWeightAlongKContigInt32,
            state_dict_convert=False,
        )
        self.apply_native_layout(
            self.down_proj_qweight,
            Packed4BitWeightAlongKContigInt32,
            state_dict_convert=False,
        )


class BlockInt4MoeExpertsMergedBase(NativeLayoutMixin, QuantizedMoeExpertsMerged):
    """blockint4 MoE experts with merged gate and up projection."""

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
        group_size: int = 128,
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
        self.quant_group_size = group_size

        self.gate_up_proj_qweight = nn.Parameter(
            torch.empty(self.group_size, 2 * moe_inter_dim, dim), requires_grad=False
        )
        self.gate_up_proj_scales = nn.Parameter(
            torch.empty(
                self.group_size,
                2 * moe_inter_dim,
                dim // group_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        self.down_proj_qweight = nn.Parameter(
            torch.empty(self.group_size, dim, moe_inter_dim), requires_grad=False
        )
        self.down_proj_scales = nn.Parameter(
            torch.empty(
                self.group_size,
                dim,
                moe_inter_dim // group_size,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )

    @override
    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(
            self.gate_up_proj_qweight,
            Packed4BitWeightAlongKContigInt32,
            state_dict_convert=False,
        )
        self.apply_native_layout(
            self.down_proj_qweight,
            Packed4BitWeightAlongKContigInt32,
            state_dict_convert=False,
        )


class BlockInt4ExpertBlockDispatchMixin:
    """Shared conversion from indexed routing to expert-block indexed routing."""

    moe_block_size = 64

    @override
    @functools.singledispatchmethod
    def forward_no_sum(
        self, routed_x: BatchedRoutedActivation, impl="auto"
    ) -> BatchedExpertResult:
        return super().forward_no_sum(routed_x, impl=impl)

    @forward_no_sum.register
    def _(
        self, routed_x: IndexedBatchedRoutedActivation, impl="auto"
    ) -> PerTokenBatchedExpertResult:
        routed_x = routed_x.as_local_expert_ids(
            self.experts_start_idx, self.experts_end_idx
        )
        block_routed = ExpertBlockIndexedBatchedRoutedActivation.convert_from(
            routed_x,
            n_experts=self.group_size,
            block_size=self.moe_block_size,
        )
        return self.forward_no_sum(block_routed, impl=impl)

    @forward_no_sum.register
    def _(
        self, routed_x: ExpertBlockIndexedBatchedRoutedActivation, impl="auto"
    ) -> PerTokenBatchedExpertResult:
        routed_x = routed_x.as_local_expert_ids(
            self.experts_start_idx, self.experts_end_idx
        )
        return self._forward_expert_block_indexed(routed_x)

    def _forward_expert_block_indexed(
        self, routed_x: ExpertBlockIndexedBatchedRoutedActivation
    ) -> PerTokenBatchedExpertResult:
        raise NotImplementedError


class BlockInt4TritonMixin(BlockInt4ExpertBlockDispatchMixin):
    """Triton blockint4 MoE consumes the int32-packed checkpoint layout directly."""

    def _invoke_triton(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        B_scale: torch.Tensor,
        routed_x: ExpertBlockIndexedBatchedRoutedActivation,
        topk_weights: torch.Tensor,
        top_k: int,
        config: dict,
    ):
        A = eval_lazy(A)
        invoke_fused_moe_wna16_triton_kernel(
            A=A,
            B=B,
            C=C,
            B_scale=B_scale,
            B_zp=None,
            topk_weights=topk_weights,
            sorted_token_ids=routed_x.block_to_token_x_topk_indices.flatten()
            .contiguous()
            .to(torch.int64),
            expert_ids=routed_x.block_to_expert_indices.contiguous().to(torch.int64),
            num_tokens_post_padded=routed_x.n_blocks_scalar_tensor
            * self.moe_block_size,
            mul_routed_weight=False,
            top_k=top_k,
            config=config,
            use_int4_w4a16=True,
            use_int8_w8a16=False,
            group_size=self.quant_group_size,
        )


@QuantizationRegistry.register_moe_experts(
    "blockint4", merge_gate_up=False, when=lambda _: has_triton, priority=0
)
class TritonBlockInt4MoeExpertsUnmerged(
    BlockInt4TritonMixin, BlockInt4MoeExpertsUnmergedBase
):
    @override
    def _forward_expert_block_indexed(
        self, routed_x: ExpertBlockIndexedBatchedRoutedActivation
    ) -> PerTokenBatchedExpertResult:
        M = routed_x.activation.shape[0]
        topk = routed_x.topk
        device = routed_x.activation.device
        if M == 0:
            return PerTokenBatchedExpertResult(
                torch.zeros(
                    M,
                    topk,
                    self.dim,
                    device=device,
                    dtype=routed_x.activation.dtype,
                )
            )

        activation = routed_x.activation
        topk_weights = torch.empty(M, topk, dtype=torch.float32, device=device)
        config = {
            "BLOCK_SIZE_M": self.moe_block_size,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }

        gate_out = torch.zeros(
            M, topk, self.moe_inter_dim, dtype=activation.dtype, device=device
        )
        self._invoke_triton(
            activation,
            self.gate_proj_qweight,
            gate_out,
            self.gate_proj_scales,
            routed_x,
            topk_weights,
            topk,
            config,
        )
        up_out = torch.zeros(
            M, topk, self.moe_inter_dim, dtype=activation.dtype, device=device
        )
        self._invoke_triton(
            activation,
            self.up_proj_qweight,
            up_out,
            self.up_proj_scales,
            routed_x,
            topk_weights,
            topk,
            config,
        )
        intermediate = torch.nn.functional.silu(gate_out) * up_out
        del gate_out, up_out

        down_out = torch.zeros(M, topk, self.dim, dtype=activation.dtype, device=device)
        self._invoke_triton(
            intermediate.view(M * topk, self.moe_inter_dim),
            self.down_proj_qweight,
            down_out,
            self.down_proj_scales,
            routed_x,
            topk_weights,
            1,
            config,
        )
        del intermediate
        return PerTokenBatchedExpertResult(down_out)


@QuantizationRegistry.register_moe_experts(
    "blockint4", merge_gate_up=True, when=lambda _: has_triton, priority=0
)
class TritonBlockInt4MoeExpertsMerged(
    BlockInt4TritonMixin, BlockInt4MoeExpertsMergedBase
):
    @override
    def _forward_expert_block_indexed(
        self, routed_x: ExpertBlockIndexedBatchedRoutedActivation
    ) -> PerTokenBatchedExpertResult:
        M = routed_x.activation.shape[0]
        topk = routed_x.topk
        device = routed_x.activation.device
        if M == 0:
            return PerTokenBatchedExpertResult(
                torch.zeros(
                    M,
                    topk,
                    self.dim,
                    device=device,
                    dtype=routed_x.activation.dtype,
                )
            )

        activation = routed_x.activation
        topk_weights = torch.empty(M, topk, dtype=torch.float32, device=device)
        config = {
            "BLOCK_SIZE_M": self.moe_block_size,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }

        gate_up_out = torch.zeros(
            M, topk, 2 * self.moe_inter_dim, dtype=activation.dtype, device=device
        )
        self._invoke_triton(
            activation,
            self.gate_up_proj_qweight,
            gate_up_out,
            self.gate_up_proj_scales,
            routed_x,
            topk_weights,
            topk,
            config,
        )
        intermediate = self.forward_act_fn_merged(gate_up_out)
        del gate_up_out

        down_out = torch.zeros(M, topk, self.dim, dtype=activation.dtype, device=device)
        self._invoke_triton(
            intermediate.view(M * topk, self.moe_inter_dim),
            self.down_proj_qweight,
            down_out,
            self.down_proj_scales,
            routed_x,
            topk_weights,
            1,
            config,
        )
        del intermediate
        return PerTokenBatchedExpertResult(down_out)


class BlockInt4MarlinMixin(BlockInt4ExpertBlockDispatchMixin):
    """Marlin blockint4 MoE stores qweights and scales in Marlin runtime layout."""

    def _ensure_marlin_workspace(self):
        if not hasattr(self, "workspace"):
            device = self.down_proj_qweight.device
            non_moe_ws_size = max(self.dim, self.moe_inter_dim * 2) // 64 * 16
            sms = torch.cuda.get_device_properties(device).multi_processor_count
            moe_ws_size = sms * 4
            ws_size = max(non_moe_ws_size, moe_ws_size)
            self.workspace = torch.zeros(
                ws_size, dtype=torch.int, device=device, requires_grad=False
            )
            self._g_idx = marlin_make_empty_g_idx(device)
            self._g_idx_sort_indices = marlin_make_empty_g_idx(device)
            self._zp = marlin_make_empty_g_idx(device)

    def _marlin_gemm(
        self,
        x: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        in_features: int,
        out_features: int,
    ) -> torch.Tensor:
        activation = eval_lazy(x).reshape(-1, x.shape[-1])
        output = chitu_backend.gptq_marlin_gemm(
            activation,  # a
            None,  # c_or_none
            qweight,  # b_q_weight
            None,  # b_bias_or_none
            scales,  # b_scales
            None,  # global_scale_or_none
            None,  # b_zeros_or_none
            None,  # g_idx_or_none
            None,  # perm_or_none
            self.workspace,  # workspace
            UINT4B8_TYPE_ID,  # b_q_type_id
            activation.shape[0],  # size_m
            out_features,  # size_n
            in_features,  # size_k
            True,  # is_k_full
            False,  # use_atomic_add
            True,  # use_fp32_reduce
            False,  # is_zp_float
            False,  # is_block_fp8
        )
        return output.reshape(x.shape[:-1] + (out_features,))

    def _invoke_marlin_moe(
        self,
        activation: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        routed_x: ExpertBlockIndexedBatchedRoutedActivation,
        topk_weights: torch.Tensor,
        size_m: int,
        size_n: int,
        size_k: int,
        top_k: int,
    ) -> torch.Tensor:
        activation = eval_lazy(activation)
        output = torch.zeros(
            size_m * top_k,
            size_n,
            dtype=activation.dtype,
            device=activation.device,
        )
        return chitu_backend.moe_wna16_marlin_gemm(
            activation,  # a
            output,  # c_or_none
            qweight,  # b_q_weight
            None,  # b_bias_or_none
            scales,  # b_scales
            None,  # global_scale_or_none
            None,  # b_zeros_or_none
            None,  # g_idx_or_none
            None,  # perm_or_none
            self.workspace,  # workspace
            routed_x.block_to_token_x_topk_indices.flatten().contiguous(),  # sorted_token_ids
            routed_x.block_to_expert_indices.contiguous(),  # expert_ids
            routed_x.n_blocks_scalar_tensor
            * self.moe_block_size,  # num_tokens_past_padded
            topk_weights,  # topk_weights
            self.moe_block_size,  # moe_block_size
            top_k,  # top_k
            False,  # mul_topk_weights
            False,  # is_ep
            UINT4B8_TYPE_ID,  # b_q_type_id
            size_m,  # size_m
            size_n,  # size_n
            size_k,  # size_k
            True,  # is_k_full
            False,  # use_atomic_add
            True,  # use_fp32_reduce
            False,  # is_zp_float
        )

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        self._ensure_marlin_workspace()
        return self._marlin_gemm(
            x,
            self.down_proj_qweight[i],
            self.down_proj_scales[i],
            self.moe_inter_dim,
            self.dim,
        )


@QuantizationRegistry.register_moe_experts(
    "blockint4",
    merge_gate_up=False,
    when=lambda _: has_marlin and has_marlin_moe,
    priority=1,
)
class MarlinBlockInt4MoeExpertsUnmerged(
    BlockInt4MarlinMixin, BlockInt4MoeExpertsUnmergedBase
):
    @override
    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(self.gate_proj_qweight, BlockInt4MarlinQWeight)
        self.apply_native_layout(self.gate_proj_scales, BlockInt4MarlinScale)
        self.apply_native_layout(self.up_proj_qweight, BlockInt4MarlinQWeight)
        self.apply_native_layout(self.up_proj_scales, BlockInt4MarlinScale)
        self.apply_native_layout(self.down_proj_qweight, BlockInt4MarlinQWeight)
        self.apply_native_layout(self.down_proj_scales, BlockInt4MarlinScale)

    @override
    def _forward_expert_block_indexed(
        self, routed_x: ExpertBlockIndexedBatchedRoutedActivation
    ) -> PerTokenBatchedExpertResult:
        self._ensure_marlin_workspace()
        M = routed_x.activation.shape[0]
        topk = routed_x.topk
        device = routed_x.activation.device
        if M == 0:
            return PerTokenBatchedExpertResult(
                torch.zeros(
                    M,
                    topk,
                    self.dim,
                    device=device,
                    dtype=routed_x.activation.dtype,
                )
            )

        a = routed_x.activation
        if a.dtype != self.down_proj_scales.dtype:
            a = a.to(self.down_proj_scales.dtype)
        topk_weights = torch.empty(M, topk, dtype=torch.float32, device=device)

        gate_out = self._invoke_marlin_moe(
            a,
            self.gate_proj_qweight,
            self.gate_proj_scales,
            routed_x,
            topk_weights,
            M,
            self.moe_inter_dim,
            self.dim,
            topk,
        )
        up_out = self._invoke_marlin_moe(
            a,
            self.up_proj_qweight,
            self.up_proj_scales,
            routed_x,
            topk_weights,
            M,
            self.moe_inter_dim,
            self.dim,
            topk,
        )
        intermediate = torch.nn.functional.silu(gate_out) * up_out
        del gate_out, up_out

        down_out = self._invoke_marlin_moe(
            intermediate.view(M * topk, self.moe_inter_dim),
            self.down_proj_qweight,
            self.down_proj_scales,
            routed_x,
            topk_weights,
            M * topk,
            self.dim,
            self.moe_inter_dim,
            1,
        )
        del intermediate
        return PerTokenBatchedExpertResult(down_out.view(M, topk, self.dim))

    @override
    def forward_ith_expert_gate(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert x_scale is None
        self._ensure_marlin_workspace()
        return self._marlin_gemm(
            x,
            self.gate_proj_qweight[i],
            self.gate_proj_scales[i],
            self.dim,
            self.moe_inter_dim,
        )

    @override
    def forward_ith_expert_up(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert x_scale is None
        self._ensure_marlin_workspace()
        return self._marlin_gemm(
            x,
            self.up_proj_qweight[i],
            self.up_proj_scales[i],
            self.dim,
            self.moe_inter_dim,
        )


@QuantizationRegistry.register_moe_experts(
    "blockint4",
    merge_gate_up=True,
    when=lambda _: has_marlin and has_marlin_moe,
    priority=1,
)
class MarlinBlockInt4MoeExpertsMerged(
    BlockInt4MarlinMixin, BlockInt4MoeExpertsMergedBase
):
    @override
    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(self.gate_up_proj_qweight, BlockInt4MarlinQWeight)
        self.apply_native_layout(self.gate_up_proj_scales, BlockInt4MarlinScale)
        self.apply_native_layout(self.down_proj_qweight, BlockInt4MarlinQWeight)
        self.apply_native_layout(self.down_proj_scales, BlockInt4MarlinScale)

    @override
    def _forward_expert_block_indexed(
        self, routed_x: ExpertBlockIndexedBatchedRoutedActivation
    ) -> PerTokenBatchedExpertResult:
        self._ensure_marlin_workspace()
        M = routed_x.activation.shape[0]
        topk = routed_x.topk
        device = routed_x.activation.device
        if M == 0:
            return PerTokenBatchedExpertResult(
                torch.zeros(
                    M,
                    topk,
                    self.dim,
                    device=device,
                    dtype=routed_x.activation.dtype,
                )
            )

        a = routed_x.activation
        if a.dtype != self.down_proj_scales.dtype:
            a = a.to(self.down_proj_scales.dtype)
        topk_weights = torch.empty(M, topk, dtype=torch.float32, device=device)

        gate_up_out = self._invoke_marlin_moe(
            a,
            self.gate_up_proj_qweight,
            self.gate_up_proj_scales,
            routed_x,
            topk_weights,
            M,
            2 * self.moe_inter_dim,
            self.dim,
            topk,
        )
        intermediate = self.forward_act_fn_merged(gate_up_out.view(M, topk, -1))
        del gate_up_out

        down_out = self._invoke_marlin_moe(
            intermediate.view(M * topk, self.moe_inter_dim),
            self.down_proj_qweight,
            self.down_proj_scales,
            routed_x,
            topk_weights,
            M * topk,
            self.dim,
            self.moe_inter_dim,
            1,
        )
        del intermediate
        return PerTokenBatchedExpertResult(down_out.view(M, topk, self.dim))

    @override
    def forward_ith_expert_gate_up(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert x_scale is None
        self._ensure_marlin_workspace()
        return self._marlin_gemm(
            x,
            self.gate_up_proj_qweight[i],
            self.gate_up_proj_scales[i],
            self.dim,
            2 * self.moe_inter_dim,
        )


@QuantizationRegistry.register_moe_experts(
    "blockint4", merge_gate_up=True, when=lambda _: has_aiter, priority=1
)
class AiterBlockInt4MoeExpertsMerged(BlockInt4MoeExpertsMergedBase):
    """aiter blockint4 MoE consumes merged gate/up weights repacked to uint8 pack2."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aiter_repacked = False

    def _repack_to_aiter(self):
        self._w1 = self.gate_up_proj_qweight.data.view(torch.uint8).contiguous()
        self._w1_scale = self.gate_up_proj_scales.data.contiguous()
        self._w2 = self.down_proj_qweight.data.view(torch.uint8).contiguous()
        self._w2_scale = self.down_proj_scales.data.contiguous()

        E = self.gate_up_proj_qweight.shape[0]
        n_groups_k = self.dim // self.quant_group_size
        n_groups_inter = self.moe_inter_dim // self.quant_group_size
        device = self.gate_up_proj_qweight.device
        self._w1_zp_dummy = torch.full(
            (E, self.moe_inter_dim, n_groups_k),
            0x88,
            dtype=torch.uint8,
            device=device,
        ).contiguous()
        self._w2_zp_dummy = torch.full(
            (E, self.dim // 2, n_groups_inter),
            0x88,
            dtype=torch.uint8,
            device=device,
        ).contiguous()

        self._aiter_config_valid = False
        self._aiter_repacked = True

    @override
    def forward(
        self,
        routed_x: BatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        if not isinstance(routed_x, IndexedBatchedRoutedActivation):
            return QuantizedMoeExpertsMerged.forward(
                self, routed_x, weights, inplace=inplace, impl=impl
            )
        if not self._aiter_repacked:
            self._repack_to_aiter()

        activation = routed_x.activation
        if activation.dtype != self.gate_up_proj_scales.dtype:
            activation = activation.to(self.gate_up_proj_scales.dtype)

        M = activation.shape[0]
        topk = routed_x.token_to_expert_indices.shape[1]
        device = activation.device
        if M == 0:
            return torch.zeros(
                M,
                self.dim,
                device=device,
                dtype=activation.dtype,
            )

        n_local_experts = self.experts_end_idx - self.experts_start_idx
        topk_ids = routed_x.token_to_expert_indices.contiguous()

        if not self._aiter_config_valid:
            status, moe_cfg = get_aiter_moe_config(
                M=M,
                E=n_local_experts,
                N1=2 * self.moe_inter_dim,
                N2=self.dim,
                K=self.dim,
                top_k=topk,
                block_size=self.quant_group_size,
                dtype=activation.dtype,
                quant_type=MoeQuantType.W4A16,
            )
            if not status:
                raise NotImplementedError(
                    "aiter config not found for blockint4 MoE: "
                    f"M={M} E={n_local_experts} N1={2 * self.moe_inter_dim} "
                    f"N2={self.dim} K={self.dim}"
                )
            self._moe_config = moe_cfg
            self._aiter_config_valid = True

        aiter_global_num_experts = self.global_n_experts
        expert_map = None
        if self.group_size != self.global_n_experts:
            if routed_x.expert_ids_are_local:
                raise RuntimeError(
                    "aiter blockint4 EP expects global expert IDs. "
                    "ChiTu local expert IDs with sentinel are not supported."
                )
            if self._moe_config.solution_type == MoeSolutionType.ASM:
                # aiter names this argument ``expert_map``, but ASM EP treats
                # it as a binary expert mask: 1 for local global experts, 0
                # otherwise.
                expert_map = torch.zeros(
                    self.global_n_experts,
                    dtype=torch.int32,
                    device=device,
                )
                expert_map[self.experts_start_idx : self.experts_end_idx] = 1
            else:
                # aiter non-ASM EP backends use the documented global-to-local
                # mapping: local global experts map to [0, n_local_experts),
                # and non-local experts map to -1.
                expert_map = torch.full(
                    (self.global_n_experts,),
                    -1,
                    dtype=torch.int32,
                    device=device,
                )
                expert_map[self.experts_start_idx : self.experts_end_idx] = (
                    torch.arange(n_local_experts, dtype=torch.int32, device=device)
                )

        if self._moe_config.solution_type == MoeSolutionType.MOE_C:
            w1 = w4a16_marlin_weight_1(self._w1)
            w1 = w1.view(-1).view(torch.uint8).view(*self._w1.shape)
            w2 = w4a16_marlin_weight_2(self._w2)
            w2 = w2.view(-1).view(torch.uint8).view(*self._w2.shape)
            zp1 = self._w1_zp_dummy
            zp2 = self._w2_zp_dummy
        elif self._moe_config.solution_type == MoeSolutionType.ASM:
            w1, w2 = self._w1, self._w2
            zp1 = self._w1_zp_dummy
            zp2 = self._w2_zp_dummy
        else:
            w1 = self._w1
            w2 = self._w2
            zp1 = None
            zp2 = None

        topk_weights = weights.to(torch.float32)

        out = aiter_moe(
            activation,
            w1,
            w2,
            topk_weights,
            topk_ids,
            self._moe_config,
            inplace=False,
            activation="silu",
            w1_scale=self._w1_scale,
            w2_scale=self._w2_scale,
            w1_zp=zp1,
            w2_zp=zp2,
            a1_scale=None,
            a2_scale=None,
            block_shape=[0, self.quant_group_size],
            global_num_experts=aiter_global_num_experts,
            expert_map=expert_map,
            routed_scaling_factor=1.0,
            output_dtype=activation.dtype,
        )

        if out.dim() != 2:
            raise RuntimeError(
                f"aiter blockint4 MoE expected weighted 2D output, got {out.shape}"
            )
        return out
