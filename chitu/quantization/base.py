# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import torch

from chitu.ops import silu_and_mul
from chitu.moe.batched_expert_result import (
    BatchedExpertResult,
    PerTokenBatchedExpertResult,
    PerExpertDenseBatchedExpertResultMinimal,
)
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    PerExpertDenseBatchedRoutedActivation,
)


class QuantizedLinearBase(torch.nn.Module):
    """
    Base class for all quantized linear layers.

    Defines the interface that all quantized linear implementations must follow.
    """

    def __init__(self, in_features: int, out_features: int, has_bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias

    def __repr__(self):
        inheritance_order = []
        for cls in self.__class__.__mro__:
            if cls is torch.nn.Module:
                break
            inheritance_order.append(cls.__name__)
        inheritance_order_str = " <- ".join(inheritance_order)
        return f"{inheritance_order_str}(in_features={self.in_features}, out_features={self.out_features}, has_bias={self.has_bias})"


class QuantizedMoeExpertsBase(torch.nn.Module):
    """
    MoE experts after the gate. This module runs locally on one device.

    Inherit from this class for quantization.
    """

    def __init__(
        self,
        dim: int,
        moe_inter_dim: int,
        global_n_experts: int,
        experts_start_idx: int,
        experts_end_idx: int,
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
        self.global_n_experts = global_n_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_end_idx
        self.n_shared_experts = n_shared_experts
        self.n_activated_experts = n_activated_experts
        self.fuse_shared_experts = fuse_shared_experts
        self.checkpoint_prefix = checkpoint_prefix
        self.merge_gate_up = merge_gate_up
        self.layer_id = layer_id

        self.n_fused_shared_experts = (
            n_shared_experts if self.fuse_shared_experts else 0
        )
        self.n_routed_experts = self.experts_end_idx - self.experts_start_idx
        self.group_size = self.n_routed_experts + self.n_fused_shared_experts

    def __repr__(self):
        inheritance_order = []
        for cls in self.__class__.__mro__:
            if cls is torch.nn.Module:
                break
            inheritance_order.append(cls.__name__)
        inheritance_order_str = " <- ".join(inheritance_order)
        return f"{inheritance_order_str}(dim={self.dim}, moe_inter_dim={self.moe_inter_dim}, global_n_experts={self.global_n_experts}, n_shared_experts={self.n_shared_experts}, n_activated_experts={self.n_activated_experts})"

    def forward_ith_expert_gate_up(self, i: int, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the i-th expert's merged gate_up_proj layer only.

        Override this method to support `self.forward_no_sum_iterative`. You can safely ignore
        this method if you only do fused forward for all experts altogether.
        """

        raise NotImplementedError()

    def forward_ith_expert_gate(self, i: int, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the i-th expert's separated gate_proj layer only.

        Override this method to support `self.forward_no_sum_iterative`. You can safely ignore
        this method if you only do fused forward for all experts altogether.
        """

        raise NotImplementedError()

    def forward_ith_expert_up(self, i: int, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the i-th expert's separated up_proj layer only.

        Override this method to support `self.forward_no_sum_iterative`. You can safely ignore
        this method if you only do fused forward for all experts altogether.
        """

        raise NotImplementedError()

    def forward_act_fn_unmerged(
        self, gate_out: torch.Tensor, up_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute a single expert's activation function only if there is NO merge_gate_up.

        Override this method to support `self.forward_no_sum_iterative`. You can safely ignore
        this method if you only do fused forward for all experts altogether.
        """

        return torch.nn.functional.silu(gate_out) * up_out

    def forward_act_fn_merged(self, gate_up_out: torch.Tensor) -> torch.Tensor:
        """
        Compute a single expert's activation function only if there is merge_gate_up.

        Override this method to support `self.forward_no_sum_iterative`. You can safely ignore
        this method if you only do fused forward for all experts altogether.
        """

        return silu_and_mul(gate_up_out)

    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the i-th expert's down_proj layer only.

        Override this method to support `self.forward_no_sum_iterative`. You can safely ignore
        this method if you only do fused forward for all experts altogether.
        """

        raise NotImplementedError()

    @functools.singledispatchmethod
    def forward_no_sum_iterative(
        self, routed_x: BatchedRoutedActivation
    ) -> BatchedExpertResult:
        """
        Sequantially iterate through each expert and compute the output.

        This is a fallback method in case there is no fused forward implementation.
        This method requires the `forward_ith_expert_*` methods to be implemented.
        """

        raise NotImplementedError(
            f"{type(routed_x)} not supported for QuantizedMoeExpertsBase.forward_no_sum_iterative"
        )

    @forward_no_sum_iterative.register
    def _(
        self, routed_x: IndexedBatchedRoutedActivation
    ) -> PerTokenBatchedExpertResult:
        routed_x = routed_x.as_local_expert_ids(
            self.experts_start_idx, self.experts_end_idx
        )

        x, indices = routed_x.activation, routed_x.token_to_expert_indices

        flattened_indices = indices.flatten()
        in_range_indices = flattened_indices[
            (flattened_indices >= 0)
            & (flattened_indices < self.experts_end_idx - self.experts_start_idx)
        ]
        activated_expert_ids = {
            i
            for i, cnt in enumerate(torch.bincount(in_range_indices).tolist())
            if cnt > 0
        }

        xs = []
        for i in range(self.experts_end_idx - self.experts_start_idx):
            this_x = None
            if i in activated_expert_ids:
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

        y = torch.zeros(
            indices.shape[0],
            indices.shape[1],
            x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        for i in range(self.experts_end_idx - self.experts_start_idx):
            if i in activated_expert_ids:
                idx, top = torch.where(indices == i)
                y[idx, top] = down_proj_outs[i]
        if self.fuse_shared_experts:
            for i in range(self.n_fused_shared_experts):
                y[:, self.experts_end_idx - self.experts_start_idx + i] = (
                    down_proj_outs[i]
                )
        return PerTokenBatchedExpertResult(y)

    @forward_no_sum_iterative.register
    def _(
        self, routed_x: PerExpertDenseBatchedRoutedActivation
    ) -> PerExpertDenseBatchedExpertResultMinimal:
        routed_x = routed_x.as_local_expert_ids(
            self.experts_start_idx, self.experts_end_idx
        )

        n_tokens_per_expert_cpu = routed_x.n_tokens_per_expert.cpu()
        xs = []
        for i in range(self.group_size):
            this_x = None
            if n_tokens_per_expert_cpu[i] > 0:
                this_x = routed_x.activation_per_expert[i, : n_tokens_per_expert_cpu[i]]
            xs.append(this_x)

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

        y = torch.empty_like(routed_x.activation_per_expert)
        for i, acti in enumerate(act):
            if acti is not None:
                y[i, : n_tokens_per_expert_cpu[i]] = self.forward_ith_expert_down(
                    i, acti
                )

        return PerExpertDenseBatchedExpertResultMinimal(y)

    def forward_no_sum(
        self, routed_x: BatchedRoutedActivation, impl="auto"
    ) -> BatchedExpertResult:
        """
        Compute all experts but without summing across multiple experts for each token.
        """

        return self.forward_no_sum_iterative(routed_x)

    def forward(
        self,
        routed_x: BatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            routed_x (torch.Tensor): Input BatchedRoutedActivation.
            weights (torch.Tensor): Routing weights from the gate.
            inplace (bool): If true, `x` may be modified in-place.

        Returns:
             BatchedExpertResult: Output tensor (without local sum).
        """

        y = self.forward_no_sum_iterative(routed_x)
        if (
            inplace
            and isinstance(routed_x, IndexedBatchedRoutedActivation)
            and routed_x.activation.dtype == torch.get_default_dtype()
        ):
            out = routed_x.activation
        else:
            out = None
        return y.weighted_sum(weights, out=out)


class QuantizedAbsorbGemmBase(torch.nn.Module):
    """
    The two group GeMMs in "absorb-without-precomp" mode for MLA. This module runs locally on one device.

    Inherit from this class for quantization.
    """

    pass
