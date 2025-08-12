# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from typing_extensions import override
from logging import getLogger

import torch

from chitu.quantization.base import (
    QuantizedLinearBase,
    QuantizedMoeExpertsBase,
)
from chitu.quantization.registry import QuantizationRegistry
from chitu.ops import (
    soft_fp4_raise_to_fp8_gemm_deepseek_v3,
    soft_fp4_raise_to_bf16_gemm_deepseek_v3,
    act_quant_deepseek_v3,
)
from chitu.device_type import get_device_name, is_muxi, is_nvidia
from chitu.utils import (
    ceil_div,
    try_import_opt_dep,
    try_import_platform_dep,
    parse_dtype,
)
from chitu.global_vars import get_global_args
from chitu.native_layout import (
    enable_native_layout_weight,
    Packed4BitWeightAlongK,
    Packed4BitWeightNPUNative,
)

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
triton, has_triton = try_import_platform_dep("triton")
if has_triton:
    from chitu.moe.experts import fused_experts
torch_npu, has_torch_npu = try_import_platform_dep("torch_npu")
if has_torch_npu:
    from chitu.npu_utils import fused_experts_npu
grouped_gemm, _ = try_import_opt_dep("grouped_gemm", "ascend_kernels")


logger = getLogger(__name__)


def linear_block_fp4_npu(
    x: torch.Tensor,
    weight: Packed4BitWeightNPUNative,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    act_block_size: int,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert isinstance(weight, Packed4BitWeightNPUNative)
    weight = weight.layout_tensor

    if get_global_args().infer.raise_lower_bit_float_to != "bfloat16":
        raise NotImplementedError(
            "infer.raise_lower_bit_float_to must be 'bfloat16' for NPU linear_block_fp4_npu"
        )

    assert act_block_size == 128

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
        # 三维的 x 需要squeeze到二维,在NpuAttnBackend mla_decode_paged_kv 中 x 会被 unsqueeze 到三维
        x = x.squeeze(1)
        if x.shape[0] <= 2:
            grouped_gemm.grouped_gemv(
                x,
                weight,
                scale=scale,
                groupList=expert_tokens,
                output=output,
                type=grouped_gemm.GroupedGemmType.FP4,
            )
        else:
            grouped_gemm.grouped_gemm(
                x,
                weight,
                antiquantOffsetOptional=scale_off,
                antiquantScaleOptional=scale,
                groupListOptional=expert_tokens,
                output=output,
                type=grouped_gemm.GroupedGemmType.FP4,
            )
        output = output.unsqueeze(1)
    else:
        if x.shape[0] <= 2:
            grouped_gemm.grouped_gemv(
                x,
                weight,
                scale=scale,
                groupList=expert_tokens,
                output=output,
                type=grouped_gemm.GroupedGemmType.FP4,
            )
        else:
            grouped_gemm.grouped_gemm(
                x,
                weight,
                antiquantOffsetOptional=scale_off,
                antiquantScaleOptional=scale,
                groupListOptional=expert_tokens,
                output=output,
                type=grouped_gemm.GroupedGemmType.FP4,
            )

    if bias is not None:
        output += bias
    return output


def linear_block_fp4(
    x: torch.Tensor,
    weight: Packed4BitWeightAlongK,
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
        weight (Packed4BitWeightAlongK): The weight tensor.
        weight_scale (torch.Tensor): The first-level scale tensor.
        weight_scale_2 (torch.Tensor): The second-level scale tensor.
        act_block_size (int): The block size for activation quantization.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve
        quantization-aware computations depending on the input parameters.
    """

    if get_global_args().infer.raise_lower_bit_float_to == "bfloat16":
        if is_nvidia() or is_muxi():
            y = soft_fp4_raise_to_bf16_gemm_deepseek_v3(
                x, weight, weight_scale, weight_scale_2
            )
            if bias is not None:
                y += bias
            return y
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


class Blockfp4LinearBase(QuantizedLinearBase):
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
        no_input_scale: bool = False,
    ):
        super().__init__()

        if block_shape_2 is None:
            block_shape_2 = (in_features, out_features)

        self.in_features = in_features
        self.out_features = out_features
        self.act_block_size = act_block_size

        # In the checkpoint, self.weight is in Packed4BitWeightAlongK layout with
        # `k_stride = 1`. Here we mark the layout via `self._weight_layout_class`
        # and `self._weight_plain_shape`, so `enable_native_layout_weight` can recognize
        # it. After loading, `enable_native_layout_weight` will convert it to
        # other layouts.
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
        self._weight_layout_class = Packed4BitWeightAlongK
        self._weight_plain_shape = (out_features, in_features)

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
        if not no_input_scale:
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


class Blockfp4LinearPackKStride64(
    enable_native_layout_weight("weight", Packed4BitWeightAlongK, k_stride=64),
    Blockfp4LinearBase,
):
    """
    Blockfp4Linear with weight in Packed4BitWeightAlongK (k_stride=64) layout.
    """

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        return linear_block_fp4(
            x,
            self.get_native_layout_weight(),
            self.weight_scale,
            self.weight_scale_2,
            act_block_size=self.act_block_size,
            bias=self.bias,
        )


class Blockfp4LinearPackNPUNative(
    enable_native_layout_weight("weight", Packed4BitWeightNPUNative),
    Blockfp4LinearBase,
):
    """
    Blockfp4Linear with weight in Packed4BitWeightNPUNative layout.
    """

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        return linear_block_fp4_npu(
            x,
            self.get_native_layout_weight(),
            self.weight_scale,
            self.weight_scale_2,
            act_block_size=self.act_block_size,
            bias=self.bias,
        )


class Blockfp4MoeExpertsBase(QuantizedMoeExpertsBase):
    """
    blockfp4 quantized MoeExperts with weights in Packed4BitWeightAlongK (k_stride=1) layout.
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
        fuse_shared_experts: bool,
        checkpoint_prefix: str,
        merge_gate_up: bool,
        ############################################
        # No parameters specific to this quantization
        no_input_scale: bool = False,
    ):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__(
            dim,
            moe_inter_dim,
            n_routed_experts,
            n_shared_experts,
            n_activated_experts,
            fuse_shared_experts,
            checkpoint_prefix,
            merge_gate_up,
        )

        quant_scale_stride = 16

        # In the checkpoint, the weights are in Packed4BitWeightAlongK layout with
        # `stride = 1`. Here we mark the layout via `self._{key}_layout_class` and
        # `self._{key}_plain_shape`, so `enable_native_layout_weight` can recognize it.
        # After loading, `enable_native_layout_weight` will convert them to other
        # layouts.
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
            self._gate_up_proj_weight_layout_class = Packed4BitWeightAlongK
            self._gate_up_proj_weight_plain_shape = (
                self.group_size,
                moe_inter_dim * 2,
                dim,
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
            self._gate_proj_weight_layout_class = Packed4BitWeightAlongK
            self._gate_proj_weight_plain_shape = (self.group_size, moe_inter_dim, dim)
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
            if not no_input_scale:
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
            self._up_proj_weight_layout_class = Packed4BitWeightAlongK
            self._up_proj_weight_plain_shape = (self.group_size, moe_inter_dim, dim)
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
            if not no_input_scale:
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
        self._down_proj_weight_layout_class = Packed4BitWeightAlongK
        self._down_proj_weight_plain_shape = (self.group_size, dim, moe_inter_dim)
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
        if not no_input_scale:
            self.down_proj_input_scale = torch.nn.Parameter(
                torch.empty(
                    self.group_size,
                    1,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )


class Blockfp4MoeExpertsPackKStride64(
    enable_native_layout_weight(
        "gate_up_proj_weight", Packed4BitWeightAlongK, allow_missing=True, k_stride=64
    ),
    enable_native_layout_weight(
        "gate_proj_weight", Packed4BitWeightAlongK, allow_missing=True, k_stride=64
    ),
    enable_native_layout_weight(
        "up_proj_weight", Packed4BitWeightAlongK, allow_missing=True, k_stride=64
    ),
    enable_native_layout_weight(
        "down_proj_weight", Packed4BitWeightAlongK, k_stride=64
    ),
    Blockfp4MoeExpertsBase,
):
    """
    blockfp4 quantized MoeExperts with weights in Packed4BitWeightAlongK (k_stride=64) layout.
    """

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        tokens_per_expert: Optional[torch.Tensor] = None,
        impl: str = "auto",
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

        if has_triton and self.merge_gate_up:
            raise_to_16 = (
                parse_dtype(get_global_args().infer.raise_lower_bit_float_to).itemsize
                != 1
            )

            final_indices = indices
            final_weights = weights
            if self.fuse_shared_experts:
                indice_shape = indices.shape
                final_indices = torch.empty(
                    (indice_shape[0], indice_shape[1] + 1),
                    dtype=indices.dtype,
                    device=indices.device,
                )

                final_weights = torch.empty(
                    (weights.shape[0], weights.shape[1] + 1),
                    dtype=weights.dtype,
                    device=weights.device,
                )

                chitu_backend.cuda_add_shared_experts(
                    final_weights,
                    final_indices,
                    weights,
                    indices,
                    self.n_routed_experts,
                    self.n_shared_experts,
                )
                del weights, indices

            y = fused_experts(
                hidden_states=x,
                w1=self.get_native_layout_gate_up_proj_weight().layout_tensor,
                w2=self.get_native_layout_down_proj_weight().layout_tensor,
                topk_weights=final_weights,
                topk_ids=final_indices,
                inplace=False,
                use_fp4_w4a8=True,
                expert_map=self.expert_map,
                w1_scale=self.gate_up_proj_weight_scale,
                w2_scale=self.down_proj_weight_scale,
                w1_scale_2=self.gate_up_proj_weight_scale_2,
                w2_scale_2=self.down_proj_weight_scale_2,
                block_shape=[128, 128],
                soft_fp8=raise_to_16,
                experts_start_idx=self.experts_start_idx,
                impl=impl,
            )

        else:
            y = self.forward_iterative(x, weights, indices)

        return y.view(shape)

    @override
    def forward_ith_expert_gate_up(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear_block_fp4(
            x,
            self.get_native_layout_gate_up_proj_weight()[i],
            self.gate_up_proj_weight_scale[i],
            self.gate_up_proj_weight_scale_2[i],
            128,
            None,
        )

    @override
    def forward_ith_expert_gate(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear_block_fp4(
            x,
            self.get_native_layout_gate_proj_weight()[i],
            self.gate_proj_weight_scale[i],
            self.gate_proj_weight_scale_2[i],
            128,
            None,
        )

    @override
    def forward_ith_expert_up(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear_block_fp4(
            x,
            self.get_native_layout_up_proj_weight()[i],
            self.up_proj_weight_scale[i],
            self.up_proj_weight_scale_2[i],
            128,
            None,
        )

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear_block_fp4(
            x,
            self.get_native_layout_down_proj_weight()[i],
            self.down_proj_weight_scale[i],
            self.down_proj_weight_scale_2[i],
            128,
            None,
        )


class Blockfp4MoeExpertsPackNPUNative(
    enable_native_layout_weight(
        "gate_up_proj_weight", Packed4BitWeightNPUNative, allow_missing=True
    ),
    enable_native_layout_weight(
        "gate_proj_weight", Packed4BitWeightNPUNative, allow_missing=True
    ),
    enable_native_layout_weight(
        "up_proj_weight", Packed4BitWeightNPUNative, allow_missing=True
    ),
    enable_native_layout_weight("down_proj_weight", Packed4BitWeightNPUNative),
    Blockfp4MoeExpertsBase,
):
    """
    blockfp4 quantized MoeExperts with weights in Packed4BitWeightNPUNative layout.
    """

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        tokens_per_expert: Optional[torch.Tensor] = None,
        impl: str = "auto",
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

        if self.merge_gate_up:
            y = fused_experts_npu(
                hidden_states=x,
                w1=self.gate_up_proj_weight,
                w2=self.down_proj_weight,
                topk_weights=weights,
                topk_ids=indices,
                w1_scale=self.gate_up_proj_weight_scale,
                w2_scale=self.down_proj_weight_scale,
            )

        else:
            y = self.forward_iterative(x, weights, indices)

        return y.view(shape)

    @override
    def forward_ith_expert_gate_up(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear_block_fp4_npu(
            x,
            self.get_native_layout_gate_up_proj_weight()[i],
            self.gate_up_proj_weight_scale[i],
            self.gate_up_proj_weight_scale_2[i],
            128,
            None,
        )

    @override
    def forward_ith_expert_gate(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear_block_fp4_npu(
            x,
            self.get_native_layout_gate_proj_weight()[i],
            self.gate_proj_weight_scale[i],
            self.gate_proj_weight_scale_2[i],
            128,
            None,
        )

    @override
    def forward_ith_expert_up(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear_block_fp4_npu(
            x,
            self.get_native_layout_up_proj_weight()[i],
            self.up_proj_weight_scale[i],
            self.up_proj_weight_scale_2[i],
            128,
            None,
        )

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear_block_fp4_npu(
            x,
            self.get_native_layout_down_proj_weight()[i],
            self.down_proj_weight_scale[i],
            self.down_proj_weight_scale_2[i],
            128,
            None,
        )


if has_torch_npu:
    QuantizationRegistry.register_linear("blockfp4", Blockfp4LinearPackNPUNative)
    QuantizationRegistry.register_moe_experts(
        "blockfp4", Blockfp4MoeExpertsPackNPUNative
    )
else:
    QuantizationRegistry.register_linear("blockfp4", Blockfp4LinearPackKStride64)
    QuantizationRegistry.register_moe_experts(
        "blockfp4", Blockfp4MoeExpertsPackKStride64
    )
