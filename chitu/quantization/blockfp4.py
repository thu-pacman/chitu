from typing import Optional, Tuple
from logging import getLogger

import torch

from chitu.quantization.registry import (
    QuantizedLinearBase,
    QuantizedMoeExpertsBase,
    QuantizationRegistry,
)
from chitu.ops import (
    soft_fp4_raise_to_fp8_gemm_deepseek_v3,
    soft_fp4_raise_to_bf16_gemm_deepseek_v3,
    act_quant_deepseek_v3,
)
from chitu.device_type import get_device_name, is_muxi, is_nvidia
from chitu.utils import ceil_div, try_import_opt_dep, parse_dtype
from chitu.global_vars import get_global_args
from chitu.ops import silu_and_mul

chitu_backend, has_chitu_backend = try_import_opt_dep("chitu_backend", "chitu_backend")
triton, has_triton = try_import_opt_dep("triton", "triton")
if has_triton:
    from chitu.fused_moe import fused_experts
torch_npu, has_torch_npu = try_import_opt_dep("torch_npu", "torch_npu")
if has_torch_npu:
    from chitu.npu_utils import fused_experts_npu
grouped_gemm, _ = try_import_opt_dep("grouped_gemm", "ascend_kernels")


logger = getLogger(__name__)


def linear_block_fp4_npu(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert (
        weight.shape[-2] % 2 == 0
    ), f"Weight shape[-2] must be even, but got {weight.shape[-2]}"
    assert (
        weight.shape[-1] % 2 == 0
    ), f"Weight shape[-1] must be even, but got {weight.shape[-1]}"
    # 针对反量化矩阵乘算子做的 shape 适配
    weight = weight.reshape(weight.shape[-1] * 2, weight.shape[-2] // 2)
    weight = weight.unsqueeze(0)
    weight_scale = weight_scale.unsqueeze(0)
    scale = weight_scale.transpose(-2, -1)
    if not scale.is_contiguous():
        scale = scale.contiguous()
    scale_off = torch.empty_like(scale)
    output = torch.empty(
        [x.shape[0], weight.shape[-1] * 2], dtype=x.dtype, device=x.device
    )
    # NOTE: 生成一个仅有一个元素的 Tensor，值为 N,并且需要保证 export tokens 是一个一维的 Tensor
    expert_tokens = torch.full([1], x.shape[0], device=x.device, dtype=torch.int64)

    if x.dim() == 3:
        # 三维的 x 需要squeeze到二维,在NpuAttnBackend mla_attn_with_kvcache中 x 会被 unsqueeze 到三维
        x = x.squeeze(1)
        grouped_gemm.grouped_gemm(
            x,
            weight,
            antiquantOffsetOptional=scale_off,
            antiquantScaleOptional=scale,
            groupListOptional=expert_tokens,
            output=output,
        )
        output = output.unsqueeze(1)
    else:
        grouped_gemm.grouped_gemm(
            x,
            weight,
            antiquantOffsetOptional=scale_off,
            antiquantScaleOptional=scale,
            groupListOptional=expert_tokens,
            output=output,
        )

    if bias is not None:
        output += bias
    return output


def linear_block_fp4(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    act_block_size: int,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and
            requires dequantization for certain cases.
        weight_scale (torch.Tensor): The first-level scale tensor.
        weight_scale_2 (torch.Tensor): The second-level scale tensor.
        act_block_size (int): The block size for activation quantization.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve
        quantization-aware computations depending on the input parameters.
    """

    assert weight.element_size() == 1

    if get_global_args().infer.raise_lower_bit_float_to == "bfloat16":
        if is_nvidia() or is_muxi():
            y = soft_fp4_raise_to_bf16_gemm_deepseek_v3(
                x, weight, weight_scale, weight_scale_2
            )
            if bias is not None:
                y += bias
            return y
        elif get_global_args().infer.npu_fusion_fp4:
            return linear_block_fp4_npu(x, weight, weight_scale, weight_scale_2, bias)
        else:
            raise NotImplementedError(
                f"Soft-fp8 fused gemm not implemented for {get_device_name()}"
            )
            # FIXME: Use a dequant-then-compute approach
    else:
        x_dtype = x.dtype
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        x, act_scale = act_quant_deepseek_v3(x, act_block_size)
        assert weight_scale is not None
        y = soft_fp4_raise_to_fp8_gemm_deepseek_v3(
            x,
            act_scale,
            weight,
            weight_scale,
            weight_scale_2,
            act_block_size=act_block_size,
        )
        if bias is not None:
            y += bias
        return y.view(x_shape[:-1] + y.shape[-1:]).to(x_dtype)


@QuantizationRegistry.register_linear("blockfp4")
class Blockfp4Linear(QuantizedLinearBase):
    """
    block 4-bit weight and activation quantized linear layer.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        has_bias: If set to True, the layer will have a bias.
        bias_dtype: The desired data type of the bias.
        block_shape: The block shape (in, out) of first-level scaling. Defaults to
            (16, 1).
        block_shape_2: The block shape (in, out) of second-level scaling. Defaults
            to the same shape as the full weight tensor.
        act_block_size: The block size for activation quantization.
    """

    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        in_features: int,
        out_features: int,
        has_bias: bool = False,
        *,
        ############################################
        # Parameters specific to this quantization
        bias_dtype=None,
        block_shape: Tuple[int, int] = (16, 1),
        block_shape_2: Optional[Tuple[int, int]] = None,
        act_block_size: int = 128,
    ):
        super().__init__()

        if block_shape_2 is None:
            block_shape_2 = (in_features, out_features)

        self.in_features = in_features
        self.out_features = out_features
        self.act_block_size = act_block_size

        self.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty(
                    (
                        out_features,
                        in_features // 2,  # Every 2 float4 is packed into 1 uint8
                    ),
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            ),
        )

        block_in, block_out = block_shape
        if (
            get_global_args().models.type == "hf-llama"
            and get_global_args().infer.npu_fusion_fp4
        ):
            dtype = torch.bfloat16
        else:
            dtype = torch.uint8
        self.register_parameter(
            "weight_scale",
            torch.nn.Parameter(
                torch.empty(
                    ceil_div(out_features, block_out),
                    ceil_div(in_features, block_in),
                    dtype=dtype,
                ),
                requires_grad=False,
            ),
        )

        block_2_in, block_2_out = block_shape_2
        assert out_features % block_2_out == 0, f"{out_features=}, {block_2_out=}"
        assert in_features % block_2_in == 0, f"{in_features=}, {block_2_in=}"
        self.register_parameter(
            "input_scale",
            torch.nn.Parameter(
                torch.empty(
                    out_features // block_2_out,
                    in_features // block_2_in,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "weight_scale_2",
            torch.nn.Parameter(
                torch.empty(
                    out_features // block_2_out,
                    in_features // block_2_in,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            ),
        )

        if has_bias:
            self.register_parameter(
                "bias",
                torch.nn.Parameter(
                    torch.empty(out_features, dtype=bias_dtype),
                    requires_grad=False,
                ),
            )
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        return linear_block_fp4(
            x,
            self.weight,
            self.weight_scale,
            self.weight_scale_2,
            act_block_size=self.act_block_size,
            bias=self.bias,
        )


@QuantizationRegistry.register_moe_experts("blockfp4")
class Blockfp4MoeExperts(QuantizedMoeExpertsBase):
    """
    blockfp4 quantized MoeExperts
    """

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
        ############################################
        # No parameters specific to this quantization
    ):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
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

        quant_scale_stride = 16

        if self.merge_gate_up:
            scale_in_features = ceil_div(dim, quant_scale_stride)
            self.gate_up_proj_weight = torch.nn.Parameter(
                torch.empty(
                    self.group_size,
                    moe_inter_dim * 2,
                    dim // 2,
                    dtype=torch.uint8,
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
            self.gate_up_proj_weight_scale_2 = torch.nn.Parameter(
                torch.empty(
                    self.group_size,
                    2,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.gate_up_proj_input_scale = torch.nn.Parameter(
                torch.empty(
                    self.group_size,
                    2,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        else:
            scale_in_features = ceil_div(dim, quant_scale_stride)
            self.gate_proj_weight = torch.nn.Parameter(
                torch.empty(
                    self.group_size,
                    moe_inter_dim,
                    dim // 2,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            self.gate_proj_weight_scale = torch.nn.Parameter(
                torch.empty(
                    self.group_size,
                    moe_inter_dim,
                    scale_in_features,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            self.gate_proj_weight_scale_2 = torch.nn.Parameter(
                torch.empty(
                    self.group_size,
                    1,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.gate_proj_input_scale = torch.nn.Parameter(
                torch.empty(
                    self.group_size,
                    1,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.up_proj_weight = torch.nn.Parameter(
                torch.empty(
                    self.group_size,
                    moe_inter_dim,
                    dim // 2,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            self.up_proj_weight_scale = torch.nn.Parameter(
                torch.empty(
                    self.group_size,
                    moe_inter_dim,
                    scale_in_features,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            self.up_proj_weight_scale_2 = torch.nn.Parameter(
                torch.empty(
                    self.group_size,
                    1,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.up_proj_input_scale = torch.nn.Parameter(
                torch.empty(
                    self.group_size,
                    1,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        down_proj_scale_in_features = ceil_div(moe_inter_dim, quant_scale_stride)
        down_proj_scale_out_features = dim
        self.down_proj_weight = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                dim,
                moe_inter_dim // 2,
                dtype=torch.uint8,
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
        self.down_proj_weight_scale_2 = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                1,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.down_proj_input_scale = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                1,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def _compute_npu_fused_experts(self, x, weights, indices):
        y = fused_experts_npu(
            hidden_states=x,
            w1=self.gate_up_proj_weight,
            w2=self.down_proj_weight,
            topk_weights=weights,
            topk_ids=indices,
            w1_scale=self.gate_up_proj_weight_scale,
            w2_scale=self.down_proj_weight_scale,
        )
        return y

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
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

        if self.op_impl == "muxi_custom_kernel":
            raise NotImplementedError(
                "muxi_custom_kernel is not supported for blockfp4 MoeExperts"
            )
        elif has_torch_npu:
            y = self._compute_npu_fused_experts(x, weights, indices)
        elif has_triton and self.merge_gate_up:
            raise_to_16 = (
                parse_dtype(get_global_args().infer.raise_lower_bit_float_to).itemsize
                != 1
            )

            if not self.fuse_shared_experts:

                y = fused_experts(
                    x,
                    self.gate_up_proj_weight,
                    self.down_proj_weight,
                    topk_weights=weights,
                    topk_ids=indices,
                    use_fp4_w4a8=True,
                    inplace=True,
                    global_num_experts=self.n_routed_experts,
                    w1_scale=self.gate_up_proj_weight_scale,
                    w2_scale=self.down_proj_weight_scale,
                    w1w3_scale_2=self.gate_up_proj_weight_scale_2,
                    w2_scale_2=self.down_proj_weight_scale_2,
                    block_shape=[128, 128],
                    soft_fp8=raise_to_16,
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
                    use_fp4_w4a8=True,
                    inplace=True,
                    global_num_experts=self.n_routed_experts + self.n_shared_experts,
                    w1_scale=self.gate_up_proj_weight_scale,
                    w2_scale=self.down_proj_weight_scale,
                    w1w3_scale_2=self.gate_up_proj_weight_scale_2,
                    w2_scale_2=self.down_proj_weight_scale_2,
                    block_shape=[128, 128],
                    soft_fp8=raise_to_16,
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

            if self.merge_gate_up:
                assert len(xs) == self.group_size
                gate_up_proj_outs = []
                for i in range(self.group_size):
                    out = None
                    if xs[i] is not None:
                        out = linear_block_fp4(
                            xs[i],
                            self.gate_up_proj_weight[i],
                            self.gate_up_proj_weight_scale[i],
                            self.gate_up_proj_weight_scale_2[i],
                            128,
                            None,
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
                assert len(xs) == self.group_size
                gate_proj_outs = []
                up_proj_outs = []
                for i in range(self.group_size):
                    gate_proj_out = None
                    up_proj_out = None
                    if xs[i] is not None:
                        gate_proj_out = linear_block_fp4(
                            xs[i],
                            self.gate_proj_weight[i],
                            self.gate_proj_weight_scale[i],
                            self.gate_proj_weight_scale_2[i],
                            128,
                            None,
                        )
                        up_proj_out = linear_block_fp4(
                            xs[i],
                            self.up_proj_weight[i],
                            self.up_proj_weight_scale[i],
                            self.up_proj_weight_scale_2[i],
                            128,
                            None,
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
                    down_proj_out = linear_block_fp4(
                        act[i],
                        self.down_proj_weight[i],
                        self.down_proj_weight_scale[i],
                        self.down_proj_weight_scale_2[i],
                        128,
                        None,
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
