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
from chitu.quantization.gptqmodel import (
    marlin_make_empty_g_idx,
    marlin_permute_scales,
    replace_tensor,
)

logger = logging.getLogger(__name__)

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")

# ScalarTypeId for ScalarType::uint4b8
# Computed from scalar_type.hpp: ScalarType(exponent=0, mantissa=4, signed_=false, bias=8)
# with NAN_IEEE_754=1, packed as: exponent(8b)|mantissa(8b)|signed_(1b)|bias(32b)|finite(1b)|nan_repr(8b)
UINT4B8_TYPE_ID = 1125899907892224


@QuantizationRegistry.register_moe_experts(
    "blockint4", merge_gate_up=False, when=lambda _: has_chitu_backend
)
class BlockInt4MoeExpertsUnmerged(
    QuantizedMoeExpertsUnmerged
):  # TODO: Extract Marlin repack logic into a NativeLayoutTensor subclass
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
        n_shared_experts: int,
        n_activated_experts: int,
        fuse_shared_experts: bool,
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
            n_shared_experts,
            n_activated_experts,
            fuse_shared_experts,
            checkpoint_prefix,
        )

        self.bits = 4
        self.pack_factor = 32 // self.bits  # 8 x 4-bit values per int32
        self.quant_group_size = group_size

        # Checkpoint layout: (num_experts, out_features, packed_in_features)
        # gate/up: out=moe_inter_dim, in=dim
        self.gate_proj_qweight = nn.Parameter(
            torch.empty(
                self.group_size,
                moe_inter_dim,
                dim // self.pack_factor,
                dtype=torch.int32,
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
                dim // self.pack_factor,
                dtype=torch.int32,
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
                moe_inter_dim // self.pack_factor,
                dtype=torch.int32,
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

        self._marlin_repacked = False

    def _repack_to_marlin(self):
        """Repack all expert weights from compressed-tensors to Marlin tiled layout."""
        device = self.gate_proj_qweight.device
        empty_g_idx = marlin_make_empty_g_idx(device)

        # MoE kernel workspace: min(max_n_tiles * n_blocks, sms * 4).
        # Allocate max(non_moe_workspace, sms * 4) to cover both paths.
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

        proj_configs = [
            ("gate_proj", self.dim, self.moe_inter_dim),
            ("up_proj", self.dim, self.moe_inter_dim),
            ("down_proj", self.moe_inter_dim, self.dim),
        ]

        for proj_name, in_features, out_features in proj_configs:
            qweight_3d = getattr(self, f"{proj_name}_qweight")
            scales_3d = getattr(self, f"{proj_name}_scales")

            repacked_list = []
            permuted_scales_list = []
            for i in range(self.group_size):
                # Transpose from (out, packed_in) to (packed_in, out) for Marlin repack
                qw = qweight_3d[i].T.contiguous()
                sc = scales_3d[i].T.contiguous()

                repacked = chitu_backend.gptq_marlin_repack(
                    qw, empty_g_idx, in_features, out_features, self.bits
                )
                perm_scales = marlin_permute_scales(
                    sc, in_features, out_features, self.quant_group_size
                )

                repacked_list.append(repacked)
                permuted_scales_list.append(perm_scales)

            replace_tensor(
                self, f"{proj_name}_qweight", torch.stack(repacked_list, dim=0)
            )
            replace_tensor(
                self, f"{proj_name}_scales", torch.stack(permuted_scales_list, dim=0)
            )

        self._marlin_repacked = True

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
        Batch group GEMM forward using moe_wna16_marlin_gemm kernel.

        Computes gate/up/down projections for all experts in parallel via the
        Marlin MoE grouped GEMM kernel, enabling CUDA graph compatibility.
        """
        # Fused shared experts require a separate computation path that the
        # grouped GEMM kernel does not support; fall back to iterative.
        if self.fuse_shared_experts:
            return QuantizedMoeExpertsUnmerged.forward_no_sum(self, routed_x, impl)

        if not self._marlin_repacked:
            self._repack_to_marlin()

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

    # ------------------------------------------------------------------ #
    #  Per-expert iterative forward (fallback)                            #
    # ------------------------------------------------------------------ #

    @override
    def forward_ith_expert_gate(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert x_scale is None
        if not self._marlin_repacked:
            self._repack_to_marlin()
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
        if not self._marlin_repacked:
            self._repack_to_marlin()
        return self._marlin_gemm(
            x,
            self.up_proj_qweight[i],
            self.up_proj_scales[i],
            self.dim,
            self.moe_inter_dim,
        )

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        if not self._marlin_repacked:
            self._repack_to_marlin()
        return self._marlin_gemm(
            x,
            self.down_proj_qweight[i],
            self.down_proj_scales[i],
            self.moe_inter_dim,
            self.dim,
        )
