import torch

from chitu.ops import silu_and_mul


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

    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the i-th expert's down_proj layer only.

        Override this method to support `self.forward_iterative`. You can safely ignore
        this method if you only do fused forward for all experts altogether.
        """

        raise NotImplementedError()

    def forward_iterative(
        self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Sequantially iterate through each expert and compute the output.

        This is a fallback method in case there is no fused forward implementation.
        This method requires the `forward_ith_expert_*` methods to be implemented.
        """

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
            gate_up_proj_outs = []
            for i in range(self.group_size):
                out = None
                if xs[i] is not None:
                    out = self.forward_ith_expert_gate_up(i, xs[i])
                gate_up_proj_outs.append(out)
            act = [
                (
                    silu_and_mul(gate_up_proj_out)
                    if gate_up_proj_out is not None
                    else None
                )
                for gate_up_proj_out in gate_up_proj_outs
            ]
        else:
            gate_proj_outs = []
            up_proj_outs = []
            for i in range(self.group_size):
                gate_proj_out = None
                up_proj_out = None
                if xs[i] is not None:
                    gate_proj_out = self.forward_ith_expert_gate(i, xs[i])
                    up_proj_out = self.forward_ith_expert_up(i, xs[i])
                gate_proj_outs.append(gate_proj_out)
                up_proj_outs.append(up_proj_out)

            act = [
                (
                    torch.nn.functional.silu(gate_proj_out) * up_proj_out
                    if gate_proj_out is not None
                    else None
                )
                for gate_proj_out, up_proj_out in zip(gate_proj_outs, up_proj_outs)
            ]

        down_proj_outs = []
        for i in range(self.group_size):
            down_proj_out = None
            if act[i] is not None:
                down_proj_out = self.forward_ith_expert_down(i, act[i])
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


class QuantizedAbsorbGemmBase(torch.nn.Module):
    """
    The two group GeMMs in "absorb-without-precomp" mode for MLA. This module runs locally on one device.

    Inherit from this class for quantization.
    """

    pass
