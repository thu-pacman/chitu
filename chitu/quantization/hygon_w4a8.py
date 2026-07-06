# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing_extensions import override
import functools
import torch

from chitu.utils import try_import_platform_dep
from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.base import QuantizedMoeExpertsMerged
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
)

lmslim, has_lmslim = try_import_platform_dep("lmslim")
if has_lmslim:
    from lmslim.layers.fused_moe.fuse_moe_w4a8 import fused_experts_impl_w4a8


@QuantizationRegistry.register_moe_experts(
    "hygon_w4a8", merge_gate_up=True, when=lambda _: has_lmslim
)
class HygonW4A8MoeExpertsMerged(QuantizedMoeExpertsMerged):
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

        # gate_up_proj_weight shape: (group_size, moe_inter_dim * 2, dim)
        # For w4a8, weight is stored as int8, but actual bits are packed
        # The actual shape depends on how w4 is packed
        #
        # TODO: Use Packed4BitWeightAlongK
        self.gate_up_proj_weight = torch.nn.Parameter(
            torch.empty(
                (
                    self.group_size,
                    moe_inter_dim * 2,
                    dim // 2,
                ),  # w4 packing: dim // 2
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        # Scale per channel for w4a8
        self.gate_up_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                (self.group_size, moe_inter_dim * 2, 1),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        # down_proj_weight shape: (group_size, dim, moe_inter_dim)
        self.down_proj_weight = torch.nn.Parameter(
            torch.empty(
                (self.group_size, dim, moe_inter_dim // 2),  # w4 packing: dim // 2
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        # Scale per channel for w4a8
        self.down_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                (self.group_size, dim, 1),
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
        return super().forward(routed_x, weights, inplace=inplace, impl=impl)

    @forward.register
    def forward(
        self,
        routed_x: IndexedBatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        routed_x = routed_x.as_local_expert_ids(
            self.experts_start_idx, self.experts_end_idx
        )

        num_tokens = routed_x.activation.size(0)
        E, N, _ = self.gate_up_proj_weight.size()
        K = self.down_proj_weight.size(1)

        global_num_experts = E
        top_k_num = routed_x.token_to_expert_indices.size(1)

        cache13 = torch.empty(
            num_tokens * top_k_num * max(N, K),
            device=routed_x.activation.device,
            dtype=routed_x.activation.dtype,
        )

        ret = fused_experts_impl_w4a8(
            routed_x.activation,
            w1=self.gate_up_proj_weight,
            w2=self.down_proj_weight,
            topk_weights=weights,
            topk_ids=routed_x.token_to_expert_indices,
            inplace=True,
            cache13=cache13,
            activation="silu",
            apply_router_weight_on_input=False,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            use_int4_w4a8=True,
            per_channel_quant=True,
            global_num_experts=global_num_experts,
            expert_map=None,
            w1_scale=self.gate_up_proj_weight_scale,
            w2_scale=self.down_proj_weight_scale,
            w1_zp=None,
            w2_zp=None,
            a1_scale=self.gate_up_proj_weight_scale,
            a2_scale=self.down_proj_weight_scale,
            block_shape=None,
            use_nn_moe=False,
        )
        return ret
