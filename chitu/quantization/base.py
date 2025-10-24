# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import functools
import torch

from chitu.ops import silu_and_mul
from chitu.distributed.parallel_state import get_ep_group
from chitu.moe import get_moe_impl
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
)


class QuantizedLinearBase(torch.nn.Module):
    """
    Base class for all quantized linear layers.

    Defines the interface that all quantized linear implementations must follow.
    """

    pass


class QuantizedMoeExpertsBase(torch.nn.Module):
    """
    MoE experts after the gate. This module runs locally on one device.

    Inherit from this class for quantization.
    """

    def __init__(
        self,
        dim: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_shared_experts: int,
        n_activated_experts: int,
        fuse_shared_experts: bool,
        checkpoint_prefix: str,
        merge_gate_up: bool,
        layer_id: int,
    ):
        super().__init__()

        self.dim = dim
        self.moe_inter_dim = moe_inter_dim
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_activated_experts = n_activated_experts
        self.fuse_shared_experts = fuse_shared_experts
        self.checkpoint_prefix = checkpoint_prefix
        self.merge_gate_up = merge_gate_up

        self.ep_group = get_ep_group()
        moe_rank = self.ep_group.rank_in_group
        moe_world_size = self.ep_group.group_size
        self.n_fused_shared_experts = (
            n_shared_experts if self.fuse_shared_experts else 0
        )

        self.n_local_experts = n_routed_experts // moe_world_size
        remainder = n_routed_experts % moe_world_size
        self.experts_start_idx = moe_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        if self.ep_group.is_last_rank:
            self.experts_end_idx += remainder
        self.moe_impl = get_moe_impl()
        self.group_size = (
            self.experts_end_idx - self.experts_start_idx + self.n_fused_shared_experts
        )
        if self.moe_impl is not None:
            num_local_slots = self.moe_impl.load_balancer[
                layer_id
            ].get_num_local_slots()
            self.group_size = num_local_slots

    def forward_ith_expert_gate_up(self, i: int, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the i-th expert's merged gate_up_proj layer only.

        Override this method to support `self.forward_iterative`. You can safely ignore
        this method if you only do fused forward for all experts altogether.
        """

        raise NotImplementedError()

    def forward_ith_expert_gate(self, i: int, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the i-th expert's separated gate_proj layer only.

        Override this method to support `self.forward_iterative`. You can safely ignore
        this method if you only do fused forward for all experts altogether.
        """

        raise NotImplementedError()

    def forward_ith_expert_up(self, i: int, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the i-th expert's separated up_proj layer only.

        Override this method to support `self.forward_iterative`. You can safely ignore
        this method if you only do fused forward for all experts altogether.
        """

        raise NotImplementedError()

    def forward_act_fn_unmerged(
        self, gate_out: torch.Tensor, up_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute a single expert's activation function only if there is NO merge_gate_up.

        Override this method to support `self.forward_iterative`. You can safely ignore
        this method if you only do fused forward for all experts altogether.
        """

        return torch.nn.functional.silu(gate_out) * up_out

    def forward_act_fn_merged(self, gate_up_out: torch.Tensor) -> torch.Tensor:
        """
        Compute a single expert's activation function only if there is merge_gate_up.

        Override this method to support `self.forward_iterative`. You can safely ignore
        this method if you only do fused forward for all experts altogether.
        """

        return silu_and_mul(gate_up_out)

    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the i-th expert's down_proj layer only.

        Override this method to support `self.forward_iterative`. You can safely ignore
        this method if you only do fused forward for all experts altogether.
        """

        raise NotImplementedError()

    @functools.singledispatchmethod
    def forward_iterative(
        self, routed_x: BatchedRoutedActivation, weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Sequantially iterate through each expert and compute the output.

        This is a fallback method in case there is no fused forward implementation.
        This method requires the `forward_ith_expert_*` methods to be implemented.
        """

        raise NotImplementedError(
            f"{type(routed_x)} not supported for QuantizedMoeExpertsBase.forward_iterative"
        )

    @forward_iterative.register
    def _(
        self, routed_x: IndexedBatchedRoutedActivation, weights: torch.Tensor
    ) -> torch.Tensor:
        x, indices = routed_x.activation, routed_x.token_to_expert_indices

        shape = x.size()
        y = torch.zeros_like(x)
        counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).tolist()

        xs = []
        for i in range(self.experts_start_idx, self.experts_end_idx):
            this_x = None
            if counts[i]:
                idx, top = torch.where(indices == i)
                this_x = x[idx]
            xs.append(this_x)
        if self.fuse_shared_experts:
            xs += [x] * self.n_fused_shared_experts

        assert len(xs) == self.group_size
        if self.merge_gate_up:
            act = []
            for i, xsi in enumerate(xs):
                out = None
                if xsi is not None:
                    out = self.forward_act_fn_merged(
                        self.forward_ith_expert_gate_up(i, xsi)
                    )
                act.append(out)
        else:
            act = []
            for i, xsi in enumerate(xs):
                out = None
                if xsi is not None:
                    out = self.forward_act_fn_unmerged(
                        self.forward_ith_expert_gate(i, xsi),
                        self.forward_ith_expert_up(i, xsi),
                    )
                act.append(out)

        down_proj_outs = []
        for i, acti in enumerate(act):
            down_proj_out = None
            if acti is not None:
                down_proj_out = self.forward_ith_expert_down(i, acti)
            down_proj_outs.append(down_proj_out)

        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i]:
                idx, top = torch.where(indices == i)
                y[idx] += (
                    down_proj_outs[i - self.experts_start_idx] * weights[idx, top, None]
                )
        if self.fuse_shared_experts:
            for i in range(
                self.experts_end_idx - self.experts_start_idx,
                self.experts_end_idx
                - self.experts_start_idx
                + self.n_fused_shared_experts,
            ):
                y += down_proj_outs[i]
        return y.view(shape)

    def forward(
        self,
        routed_x: BatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ):
        """
        Forward pass for the MoE module.

        Args:
            routed_x (torch.Tensor): Input BatchedRoutedActivation.
            weights (torch.Tensor): Routing weights from the gate.
            inplace (bool): If true, `x` may be modified in-place.

        Returns:
            torch.Tensor: Output tensor.
        """

        return self.forward_iterative(routed_x, weights)


class QuantizedAbsorbGemmBase(torch.nn.Module):
    """
    The two group GeMMs in "absorb-without-precomp" mode for MLA. This module runs locally on one device.

    Inherit from this class for quantization.
    """

    pass
