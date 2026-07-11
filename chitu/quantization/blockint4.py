# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import logging
import plum

import torch
import torch.nn as nn

from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.base import QuantizedMoeExpertsUnmerged
from chitu.moe.batched_routed_activation import (
    IndexedBatchedRoutedActivation,
    ExpertBlockIndexedBatchedRoutedActivation,
)
from chitu.moe.batched_expert_result import PerTokenBatchedExpertResult
from chitu.utils import try_import_platform_dep
from chitu.native_layout import (
    NativeLayoutMixin,
    Packed4BitWeightAlongKContigInt32,
    BlockInt4MarlinQWeight,
    BlockInt4MarlinScale,
)
from chitu.quantization.gptqmodel import marlin_make_empty_g_idx
from chitu.device_type import is_hygon

logger = logging.getLogger(__name__)

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
triton, has_triton = try_import_platform_dep("triton")
has_marlin = has_chitu_backend and hasattr(chitu_backend, "gptq_marlin_gemm")

# aiter (AMD ROCm backend)
aiter, has_aiter = try_import_platform_dep("aiter")
if has_aiter:
    from aiter.moe import aiter_moe, get_aiter_moe_config, MoeQuantType, MoeSolutionType
    from aiter.ops.shuffle import w4a16_marlin_weight_1, w4a16_marlin_weight_2
if has_triton:
    from chitu.moe.experts.triton_batched_experts import (
        invoke_fused_moe_wna16_triton_kernel,
    )
# ScalarTypeId for ScalarType::uint4b8
# Computed from scalar_type.hpp: ScalarType(exponent=0, mantissa=4, signed_=false, bias=8)
# with NAN_IEEE_754=1, packed as: exponent(8b)|mantissa(8b)|signed_(1b)|bias(32b)|finite(1b)|nan_repr(8b)
UINT4B8_TYPE_ID = 1125899907892224


@QuantizationRegistry.register_moe_experts(
    "blockint4", merge_gate_up=False, when=lambda _: has_chitu_backend or has_aiter
)
class BlockInt4MoeExpertsUnmerged(NativeLayoutMixin, QuantizedMoeExpertsUnmerged):
    """
    Marlin INT4 quantized MoE experts with unmerged gate and up projection.
    Supports compressed-tensors pack-quantized format (4-bit symmetric, group quantization).
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

        # Checkpoint layout: (num_experts, out_features, packed_in_features)
        # gate/up: out=moe_inter_dim, in=dim
        self.gate_proj_qweight = nn.Parameter(
            torch.empty(
                self.group_size,
                moe_inter_dim,
                dim,
            ),
            requires_grad=False,
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
            torch.empty(
                self.group_size,
                moe_inter_dim,
                dim,
            ),
            requires_grad=False,
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
        # down: out=dim, in=moe_inter_dim
        self.down_proj_qweight = nn.Parameter(
            torch.empty(
                self.group_size,
                dim,
                moe_inter_dim,
            ),
            requires_grad=False,
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

        self._aiter_repacked = False

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
        if has_marlin:
            self.apply_native_layout(self.gate_proj_qweight, BlockInt4MarlinQWeight)
            self.apply_native_layout(self.gate_proj_scales, BlockInt4MarlinScale)
            self.apply_native_layout(self.up_proj_qweight, BlockInt4MarlinQWeight)
            self.apply_native_layout(self.up_proj_scales, BlockInt4MarlinScale)
            self.apply_native_layout(self.down_proj_qweight, BlockInt4MarlinQWeight)
            self.apply_native_layout(self.down_proj_scales, BlockInt4MarlinScale)

    def _ensure_marlin_workspace(self):
        """Allocate Marlin workspace and sentinel tensors lazily."""
        if not hasattr(self, "workspace"):
            device = self.gate_proj_qweight.device
            non_moe_ws_size = max(self.dim, self.moe_inter_dim) // 64 * 16
            sms = torch.cuda.get_device_properties(device).multi_processor_count
            moe_ws_size = sms * 4
            ws_size = max(non_moe_ws_size, moe_ws_size)
            self.workspace = torch.zeros(
                ws_size, dtype=torch.int, device=device, requires_grad=False
            )
            self._g_idx = marlin_make_empty_g_idx(device)
            self._g_idx_sort_indices = marlin_make_empty_g_idx(device)
            self._zp = marlin_make_empty_g_idx(device)

    def _repack_to_aiter(self):
        """Convert weights from chitu int32 pack8 to aiter uint8 pack2.

        Stores a cached unshuffled copy; Marlin shuffle is applied lazily
        in ``_forward_aiter`` once the actual backend is known.

        Cached tensors:
          _w1:       ``[E, 2*inter, K//2]`` uint8 (gate+up concatenated)
          _w1_scale: ``[E, 2*inter, K/G]`` bf16
          _w2:       ``[E, dim, inter//2]`` uint8
          _w2_scale: ``[E, dim, inter/G]`` bf16
        """
        gate_q = self.gate_proj_qweight.data.view(torch.uint8)
        up_q = self.up_proj_qweight.data.view(torch.uint8)
        self._w1 = torch.cat([gate_q, up_q], dim=1).contiguous()
        self._w1_scale = torch.cat(
            [self.gate_proj_scales.data, self.up_proj_scales.data], dim=1
        ).contiguous()
        self._w2 = self.down_proj_qweight.data.view(torch.uint8).contiguous()
        self._w2_scale = self.down_proj_scales.data.contiguous()

        # Dummy zero-points for symmetric quant — the Marlin kernel
        # requires ``b_zeros`` to be a non-None tensor even when all
        # values are the midpoint (8).  Shape: ``[E, N//2, K/G]`` uint8.
        E = self.gate_proj_qweight.shape[0]
        n_groups_k = self.dim // self.quant_group_size
        n_groups_inter = self.moe_inter_dim // self.quant_group_size
        self._w1_zp_dummy = torch.full(
            (E, self.moe_inter_dim, n_groups_k),
            0x88,
            dtype=torch.uint8,
            device=self.gate_proj_qweight.device,
        ).contiguous()
        self._w2_zp_dummy = torch.full(
            (E, self.dim // 2, n_groups_inter),
            0x88,
            dtype=torch.uint8,
            device=self.gate_proj_qweight.device,
        ).contiguous()

        self._aiter_config_valid = False
        self._aiter_repacked = True

    def _forward_aiter(
        self, routed_x: IndexedBatchedRoutedActivation
    ) -> PerTokenBatchedExpertResult:
        """MoE forward using ``aiter.moe.aiter_moe`` (AMD ROCm).

        Delegates to ``get_aiter_moe_config`` + ``aiter_moe``, which
        select the best available backend (mo_c / asm / triton).  The
        output shape is adjusted back to ``[M, topk, dim]`` because the
        downstream caller always applies ``weighted_sum``.
        """
        if not self._aiter_repacked:
            self._repack_to_aiter()

        routed_x = routed_x.as_local_expert_ids(
            self.experts_start_idx, self.experts_end_idx
        )

        M = routed_x.activation.shape[0]
        topk = routed_x.token_to_expert_indices.shape[1]
        n_local_experts = self.experts_end_idx - self.experts_start_idx
        device = routed_x.activation.device

        if M == 0:
            y = torch.zeros(
                M,
                topk,
                self.dim,
                device=device,
                dtype=routed_x.activation.dtype,
            )
            return PerTokenBatchedExpertResult(y)

        # Fetch (or validate) aiter config
        if not self._aiter_config_valid:
            status, moe_cfg = get_aiter_moe_config(
                M=M,
                E=n_local_experts,
                N1=2 * self.moe_inter_dim,
                N2=self.dim,
                K=self.dim,
                top_k=topk,
                block_size=self.quant_group_size,
                dtype=routed_x.activation.dtype,
                quant_type=MoeQuantType.W4A16,
            )
            if not status:
                logger.warning(
                    "aiter config not found for M=%d E=%d N1=%d N2=%d K=%d, "
                    "falling back to triton",
                    M,
                    n_local_experts,
                    2 * self.moe_inter_dim,
                    self.dim,
                    self.dim,
                )
                return self._forward_triton(routed_x)
            self._moe_config = moe_cfg
            self._aiter_config_valid = True

        # Apply Marlin shuffle when the backend requires it (moe_c);
        # asm / triton use the unshuffled uint8 pack2 format.
        if self._moe_config.solution_type == MoeSolutionType.MOE_C:
            w1 = w4a16_marlin_weight_1(self._w1)
            w1 = w1.view(-1).view(torch.uint8).view(*self._w1.shape)
            w2 = w4a16_marlin_weight_2(self._w2)
            w2 = w2.view(-1).view(torch.uint8).view(*self._w2.shape)
            zp1 = self._w1_zp_dummy
            zp2 = self._w2_zp_dummy
        elif self._moe_config.solution_type == MoeSolutionType.ASM:
            # ASM kernel also requires non-None ZP tensors (packed along K
            # dim).  Symmetric quant → all nibbles = 8 (midpoint).
            w1, w2 = self._w1, self._w2
            zp1 = self._w1_zp_dummy
            zp2 = self._w2_zp_dummy
        else:
            w1 = self._w1
            w2 = self._w2
            zp1 = None
            zp2 = None

        # Prepare inputs for aiter_moe
        topk_weights = torch.ones(
            M,
            topk,
            device=device,
            dtype=torch.float32,
        )
        topk_ids = routed_x.token_to_expert_indices.contiguous()

        a = routed_x.activation
        if a.dtype != self.gate_proj_scales.dtype:
            a = a.to(self.gate_proj_scales.dtype)

        out = aiter_moe(
            a,
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
            global_num_experts=n_local_experts,
            expert_map=None,
            routed_scaling_factor=1.0,
            output_dtype=a.dtype,
        )

        # aiter_moe output shape depends on backend:
        #   MOE_C → [M, topk, dim]  (per-expert, before weighted_sum)
        #   ASM   → [M, dim]        (weighted_sum already applied)
        if out.dim() == 2:
            # Already weighted — unsqueeze to [M, 1, dim] so that
            # ``PerTokenBatchedExpertResult.weighted_sum`` with
            # 1.0 weights is a no-op.
            out = out.unsqueeze(1)  # [M, 1, dim]

        return PerTokenBatchedExpertResult(out)

    def _marlin_gemm(
        self,
        x: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        in_features: int,
        out_features: int,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        output = chitu_backend.gptq_marlin_gemm(
            reshaped_x,
            None,  # c_or_none
            qweight,  # b_q_weight
            None,  # b_bias_or_none
            scales,  # b_scales
            None,  # global_scale_or_none
            None,  # b_zeros_or_none (symmetric)
            None,  # g_idx_or_none (no act reorder)
            None,  # perm_or_none
            self.workspace,  # workspace
            UINT4B8_TYPE_ID,
            reshaped_x.shape[0],
            out_features,
            in_features,
            True,  # is_k_full
            False,  # use_atomic_add
            True,  # use_fp32_reduce
            False,  # is_zp_float
            False,  # is_block_fp8
        )
        return output.reshape(x.shape[:-1] + (out_features,))

    # ------------------------------------------------------------------ #
    #  Batch group GEMM using moe_wna16_marlin_gemm kernel                #
    # ------------------------------------------------------------------ #

    MOE_BLOCK_SIZE = 64

    def _moe_marlin_gemm(
        self,
        a: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_past_padded: torch.Tensor,
        topk_weights: torch.Tensor,
        size_m: int,
        size_n: int,
        size_k: int,
        top_k: int,
    ) -> torch.Tensor:
        return chitu_backend.moe_wna16_marlin_gemm(
            a,  # a
            None,  # c_or_none
            qweight,  # b_q_weight
            None,  # b_bias_or_none
            scales,  # b_scales
            None,  # global_scale_or_none
            None,  # b_zeros_or_none
            None,  # g_idx_or_none
            None,  # perm_or_none
            self.workspace,  # workspace
            sorted_token_ids,  # sorted_token_ids
            expert_ids,  # expert_ids
            num_tokens_past_padded,  # num_tokens_past_padded
            topk_weights,  # topk_weights
            self.MOE_BLOCK_SIZE,  # moe_block_size
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
    @plum.dispatch
    def forward_no_sum(
        self, routed_x: IndexedBatchedRoutedActivation, impl="auto"
    ) -> PerTokenBatchedExpertResult:
        """
        Batch group GEMM forward.

        Args:
            routed_x: Input routed activations.
            impl: Backend selection — ``"auto"`` (default, automatic),
                  ``"aiter"``, ``"triton"``, or ``"marlin"``.
        """
        if impl == "auto":
            if has_aiter:
                impl = "aiter"
            elif (
                has_triton
                and not has_chitu_backend
                or not hasattr(chitu_backend, "moe_wna16_marlin_gemm")
            ):
                impl = "triton"
            else:
                impl = "marlin"

        if impl == "aiter":
            return self._forward_aiter(routed_x)
        if impl == "triton":
            return self._forward_triton(routed_x)
        if impl == "marlin":
            return self._forward_marlin(routed_x)

    def _forward_marlin(
        self, routed_x: IndexedBatchedRoutedActivation
    ) -> PerTokenBatchedExpertResult:
        """
        MoE forward using Marlin CUDA kernel (for NVIDIA platform).
        """
        self._ensure_marlin_workspace()

        routed_x = routed_x.as_local_expert_ids(
            self.experts_start_idx, self.experts_end_idx
        )

        M = routed_x.activation.shape[0]
        topk = routed_x.token_to_expert_indices.shape[1]
        n_local_experts = self.experts_end_idx - self.experts_start_idx
        device = routed_x.activation.device

        if M == 0:
            y = torch.zeros(
                M,
                topk,
                self.dim,
                device=device,
                dtype=routed_x.activation.dtype,
            )
            return PerTokenBatchedExpertResult(y)

        # Convert to ExpertBlockIndexed format (sorted_token_ids, expert_ids)
        block_routed = ExpertBlockIndexedBatchedRoutedActivation.convert_from(
            routed_x,
            n_experts=n_local_experts,
            block_size=self.MOE_BLOCK_SIZE,
        )

        sorted_token_ids = (
            block_routed.block_to_token_x_topk_indices.flatten().contiguous()
        )
        expert_ids = block_routed.block_to_expert_indices.contiguous()
        num_tokens_past_padded = (
            block_routed.n_blocks_scalar_tensor * self.MOE_BLOCK_SIZE
        )

        # Ensure activation dtype matches scales
        a = routed_x.activation
        if a.dtype != self.gate_proj_scales.dtype:
            a = a.to(self.gate_proj_scales.dtype)

        # Dummy topk_weights (mul_topk_weights=False, not used)
        topk_weights = torch.empty(M, topk, dtype=torch.float32, device=device)

        # Gate projection: (M, dim) -> (M*topk, moe_inter_dim)
        gate_out = self._moe_marlin_gemm(
            a,
            self.gate_proj_qweight,
            self.gate_proj_scales,
            sorted_token_ids,
            expert_ids,
            num_tokens_past_padded,
            topk_weights,
            M,
            self.moe_inter_dim,
            self.dim,
            topk,
        )

        # Up projection: (M, dim) -> (M*topk, moe_inter_dim)
        up_out = self._moe_marlin_gemm(
            a,
            self.up_proj_qweight,
            self.up_proj_scales,
            sorted_token_ids,
            expert_ids,
            num_tokens_past_padded,
            topk_weights,
            M,
            self.moe_inter_dim,
            self.dim,
            topk,
        )

        # Activation: silu(gate) * up
        intermediate = torch.nn.functional.silu(gate_out) * up_out
        del gate_out, up_out

        # Down projection: (M*topk, moe_inter_dim) -> (M*topk, dim)
        # top_k=1 because input is already topk-expanded
        down_out = self._moe_marlin_gemm(
            intermediate,
            self.down_proj_qweight,
            self.down_proj_scales,
            sorted_token_ids,
            expert_ids,
            num_tokens_past_padded,
            topk_weights,
            M * topk,
            self.dim,
            self.moe_inter_dim,
            1,
        )
        del intermediate

        return PerTokenBatchedExpertResult(down_out.view(M, topk, self.dim))

    def _forward_triton(
        self, routed_x: IndexedBatchedRoutedActivation
    ) -> PerTokenBatchedExpertResult:
        """
        MoE forward using Triton kernel (for Hygon platform).

        Uses fused_moe_kernel_gptq_awq from vllm, which works with
        AWQ format weights: [E, N, K//8] for int4 packed weights.
        """

        routed_x = routed_x.as_local_expert_ids(
            self.experts_start_idx, self.experts_end_idx
        )

        activation_shape = routed_x.activation.shape
        M = activation_shape[0]
        topk = routed_x.token_to_expert_indices.shape[1]

        n_local_experts = self.experts_end_idx - self.experts_start_idx
        device = routed_x.activation.device

        if M == 0:
            y = torch.zeros(
                M,
                topk,
                self.dim,
                device=device,
                dtype=routed_x.activation.dtype,
            )
            return PerTokenBatchedExpertResult(y)

        # Convert to ExpertBlockIndexed format (sorted_token_ids, expert_ids)
        block_routed = ExpertBlockIndexedBatchedRoutedActivation.convert_from(
            routed_x,
            n_experts=n_local_experts,
            block_size=self.MOE_BLOCK_SIZE,
        )

        sorted_token_ids = (
            block_routed.block_to_token_x_topk_indices.flatten().contiguous()
        ).to(
            torch.int64
        )  # Convert to int64 for kernel compatibility
        expert_ids = block_routed.block_to_expert_indices.contiguous().to(torch.int64)

        num_tokens_past_padded = (
            block_routed.n_blocks_scalar_tensor * self.MOE_BLOCK_SIZE
        )

        # Ensure activation dtype matches scales
        a = routed_x.activation
        if a.dtype != self.gate_proj_scales.dtype:
            a = a.to(self.gate_proj_scales.dtype)

        # Default config for Triton kernel
        config = {
            "BLOCK_SIZE_M": self.MOE_BLOCK_SIZE,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }

        # Dummy topk_weights (mul_routed_weight=False, not used)
        topk_weights = torch.empty(M, topk, dtype=torch.float32, device=device)

        # Gate projection: (M, dim) -> (M, topk, moe_inter_dim)
        # Weight shape: [E, N, K//8] = [E, moe_inter_dim, dim//8]
        # Scale shape: [E, N, num_groups] = [E, moe_inter_dim, dim//group_size]
        # Output C: 3D tensor [M, topk, N]
        gate_out = torch.empty(
            M, topk, self.moe_inter_dim, dtype=a.dtype, device=device
        )

        invoke_fused_moe_wna16_triton_kernel(
            A=a,
            B=self.gate_proj_qweight,  # [E, N, K//8]
            C=gate_out,
            B_scale=self.gate_proj_scales,  # [E, N, num_groups]
            B_zp=None,  # Symmetric quantization, no zero point
            topk_weights=topk_weights,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_past_padded,
            mul_routed_weight=False,
            top_k=topk,
            config=config,
            use_int4_w4a16=True,
            use_int8_w8a16=False,
            group_size=self.quant_group_size,
        )

        # Up projection: (M, dim) -> (M, topk, moe_inter_dim)
        up_out = torch.empty(M, topk, self.moe_inter_dim, dtype=a.dtype, device=device)
        invoke_fused_moe_wna16_triton_kernel(
            A=a,
            B=self.up_proj_qweight,
            C=up_out,
            B_scale=self.up_proj_scales,
            B_zp=None,
            topk_weights=topk_weights,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_past_padded,
            mul_routed_weight=False,
            top_k=topk,
            config=config,
            use_int4_w4a16=True,
            use_int8_w8a16=False,
            group_size=self.quant_group_size,
        )

        # Activation: silu(gate) * up
        intermediate = torch.nn.functional.silu(gate_out) * up_out
        del gate_out, up_out

        # Down projection: (M, topk, moe_inter_dim) -> (M, topk, dim)
        # Weight shape: [E, dim, moe_inter_dim//8]
        # Scale shape: [E, dim, moe_inter_dim//group_size]
        # For down projection, top_k=1 because we process M*topk tokens as M tokens each with top_k=1
        down_out = torch.empty(M, topk, self.dim, dtype=a.dtype, device=device)
        invoke_fused_moe_wna16_triton_kernel(
            A=intermediate.view(
                M * topk, self.moe_inter_dim
            ),  # 2D input [M*topk, moe_inter_dim]
            B=self.down_proj_qweight,
            C=down_out,
            B_scale=self.down_proj_scales,
            B_zp=None,
            topk_weights=topk_weights,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_past_padded,
            mul_routed_weight=False,
            top_k=1,  # Input is already topk-expanded
            config=config,
            use_int4_w4a16=True,
            use_int8_w8a16=False,
            group_size=self.quant_group_size,
        )
        del intermediate

        return PerTokenBatchedExpertResult(down_out)

    # ------------------------------------------------------------------ #
    #  Per-expert iterative forward (fallback)                            #
    # ------------------------------------------------------------------ #

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
