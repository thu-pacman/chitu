from typing import Optional

import torch

from chitu.quantization.registry import (
    QuantizedLinearBase,
    QuantizedMoeExpertsBase,
    QuantizedAbsorbGemmBase,
    QuantizationRegistry,
)
from chitu.global_vars import get_global_args
from chitu.ops import silu_and_mul
from chitu.utils import try_import_opt_dep

triton, has_triton = try_import_opt_dep("triton", "triton")
torch_npu, has_torch_npu = try_import_opt_dep("torch_npu", "torch_npu")
chitu_backend, has_chitu_backend = try_import_opt_dep("chitu_backend", "chitu_backend")
if has_torch_npu:
    from chitu.npu_utils import fused_experts_npu
if has_triton:
    from chitu.fused_moe import fused_experts


@QuantizationRegistry.register_linear(None)
class NormalLinear(QuantizedLinearBase):
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
        dtype=None,
        bias_dtype=None,
    ):
        """
        Non-quantized linear layer.

        Additional parameters are supported based on `torch.nn.Linear`.

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            has_bias: If set to True, the layer will have a bias.
            dtype: The desired data type of the parameters.
            bias_dtype: The desired data type of the bias. Defaults to `dtype`.
        """

        super().__init__()

        # These attributes are unused, but keep them compatible with nn.Linear
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(
            torch.empty(self.out_features, in_features, dtype=dtype),
            requires_grad=False,
        )
        if has_bias:
            self.bias = torch.nn.Parameter(
                torch.empty(self.out_features, dtype=bias_dtype or dtype),
                requires_grad=False,
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


@QuantizationRegistry.register_moe_experts(None)
class NormalMoeExperts(QuantizedMoeExpertsBase):
    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        dim: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_shared_experts: int,
        n_activated_experts: int,
        moe_world_size: int,
        moe_rank: int,
        op_impl: str,
        fuse_shared_experts: bool,
        checkpoint_prefix: str,
        merge_gate_up: bool,
        *,
        ############################################
        # Parameters specific to this quantization
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.op_impl = op_impl
        self.dim = dim
        self.fuse_shared_experts = fuse_shared_experts
        assert (
            n_routed_experts % moe_world_size == 0
        ), f"Number of experts must be divisible by world size (world_size={moe_world_size})"
        self.n_shared_experts = n_shared_experts
        self.n_fused_shared_experts = (
            n_shared_experts if self.fuse_shared_experts else 0
        )
        self.n_routed_experts = n_routed_experts
        self.n_local_experts = n_routed_experts // moe_world_size
        self.experts_start_idx = moe_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.group_size = (
            self.experts_end_idx - self.experts_start_idx + self.n_fused_shared_experts
        )
        self.checkpoint_prefix = checkpoint_prefix
        self.merge_gate_up = merge_gate_up

        if not self.merge_gate_up:
            self.gate_proj_weight = torch.nn.Parameter(
                torch.empty(
                    (self.group_size, moe_inter_dim, self.dim),
                    dtype=dtype,
                ),
                requires_grad=False,
            )
            self.up_proj_weight = torch.nn.Parameter(
                torch.empty(
                    (self.group_size, moe_inter_dim, self.dim),
                    dtype=dtype,
                ),
                requires_grad=False,
            )
        else:
            self.gate_up_proj_weight = torch.nn.Parameter(
                torch.empty(
                    (self.group_size, moe_inter_dim * 2, self.dim),
                    dtype=dtype,
                ),
                requires_grad=False,
            )
        self.down_proj_weight = torch.nn.Parameter(
            torch.empty(
                (self.group_size, self.dim, moe_inter_dim),
                dtype=dtype,
            ),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor):
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.
            weights (torch.Tensor): Routing weights from the gate.
            indices (torch.Tensor): Indices of the selected experts.

        Returns:
            torch.Tensor: Output tensor.
        """

        shape = x.size()
        x = x.view(-1, self.dim)

        if has_torch_npu:  # or use op_impl ?
            y = self._compute_npu_fused_experts(x, weights, indices)
        elif has_triton:
            assert self.merge_gate_up

            if not self.fuse_shared_experts:

                y = fused_experts(
                    x,
                    self.gate_up_proj_weight,
                    self.down_proj_weight,
                    topk_weights=weights,
                    topk_ids=indices,
                    inplace=True,
                    global_num_experts=self.n_routed_experts,
                    block_shape=[128, 128],
                )

            else:

                indice_shape = indices.shape
                new_indices = torch.empty(
                    (indice_shape[0], indice_shape[1] + 1),
                    dtype=indices.dtype,
                    device=indices.device,
                )

                new_weights = torch.empty(
                    (weights.shape[0], weights.shape[1] + 1),
                    dtype=weights.dtype,
                    device=weights.device,
                )

                chitu_backend.cuda_add_shared_experts(
                    new_weights,
                    new_indices,
                    weights,
                    indices,
                    self.n_routed_experts,
                    self.n_shared_experts,
                )
                del weights, indices
                y = fused_experts(
                    x,
                    self.gate_up_proj_weight,
                    self.down_proj_weight,
                    topk_weights=new_weights,
                    topk_ids=new_indices,
                    inplace=True,
                    global_num_experts=self.n_routed_experts + self.n_shared_experts,
                    block_shape=[128, 128],
                )

        else:
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
                        out = torch.nn.functional.linear(
                            xs[i], self.gate_up_proj_weight[i], bias=None
                        )
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
                        gate_proj_out = torch.nn.functional.linear(
                            xs[i], self.gate_proj_weight[i], bias=None
                        )
                        up_proj_out = torch.nn.functional.linear(
                            xs[i], self.up_proj_weight[i], bias=None
                        )
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
                    down_proj_out = torch.nn.functional.linear(
                        act[i], self.down_proj_weight[i], bias=None
                    )
                down_proj_outs.append(down_proj_out)

            for i in range(self.experts_start_idx, self.experts_end_idx):
                if counts[i]:
                    idx, top = torch.where(indices == i)
                    y[idx] += (
                        down_proj_outs[i - self.experts_start_idx]
                        * weights[idx, top, None]
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

    def _compute_npu_fused_experts(self, x, weights, indices):
        return fused_experts_npu(
            hidden_states=x,
            w1=self.gate_up_proj_weight,
            w2=self.down_proj_weight,
            topk_weights=weights,
            topk_ids=indices,
        )


@QuantizationRegistry.register_absorb_gemm(None)
class NormalAbsorbGemm(QuantizedAbsorbGemmBase):
    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        n_heads: int,
        in_features_per_head: int,
        out_features_per_head: int,
        *,
        ############################################
        # Parameters specific to this quantization
        dtype=None,
    ):
        super().__init__()

        self.weight = torch.nn.Parameter(
            torch.empty(
                n_heads, out_features_per_head, in_features_per_head, dtype=dtype
            ),
            requires_grad=False,
        )

        self.n_heads = n_heads
        self.in_features_per_head = in_features_per_head
        self.out_features_per_head = out_features_per_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            seq, n_head, n_hidden = x.shape
            bs = None
        else:
            bs, seq, n_head, n_hidden = x.shape
            x = x.view(bs * seq, n_head, n_hidden)

        y = torch.einsum("shc,hdc->shd", x, self.weight)

        if bs is not None:
            y = y.view(bs, seq, y.shape[-2], y.shape[-1])
        return y
