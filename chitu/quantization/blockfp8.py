# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
from logging import getLogger

import torch

from chitu.quantization.base import (
    QuantizedLinearBase,
    QuantizedMoeExpertsBase,
    QuantizedAbsorbGemmBase,
)
from chitu.quantization.registry import QuantizationRegistry
from chitu.ops.quant import (
    linear,
    blockfp8_gemm,
    soft_fp8_blockfp8_gemm,
    soft_fp8_blockfp8_weight_dequant,
    blockfp8_act_quant,
    blockfp8_einsum_shc_hdc_shd,
    soft_fp8_blockfp8_gemm_marlin,
)
from chitu.device_type import get_device_name, is_muxi, is_nvidia
from chitu.utils import try_import_platform_dep, parse_dtype
from chitu.global_vars import get_global_args
from chitu.native_layout import (
    enable_native_layout_weight,
    MarlinNativeLayoutWeight,
    MarlinNativeLayoutScale,
)
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
)

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
triton, has_triton = try_import_platform_dep("triton")
if has_triton and torch.cuda.is_available():
    from chitu.moe.experts import fused_experts

has_marlin = has_chitu_backend and hasattr(chitu_backend, "gptq_marlin_gemm")

logger = getLogger(__name__)


def linear_block_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve
        quantization-aware computations depending on the input parameters.
    """

    assert weight.element_size() == 1

    if get_global_args().infer.raise_lower_bit_float_to == "bfloat16":
        try:
            y = soft_fp8_blockfp8_gemm(x, weight, weight_scale)
            if bias is not None:
                y += bias
            return y
        except NotImplementedError:
            logger.warning(
                f"Soft-fp8 fused gemm not implemented for {get_device_name()}, falling back to soft-fp8 conversion"
            )
            weight_dequanted = soft_fp8_blockfp8_weight_dequant(
                weight, weight_scale, block_size
            )
            return linear(x, weight_dequanted, bias)
    else:
        x_dtype = x.dtype
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        x, act_scale = blockfp8_act_quant(x, block_size)
        assert weight_scale is not None
        y = blockfp8_gemm(x, act_scale, weight, weight_scale)
        if bias is not None:
            y += bias
        return y.view(x_shape[:-1] + y.shape[-1:]).to(x_dtype)


@QuantizationRegistry.register_linear("blockfp8")
class Blockfp8Linear(QuantizedLinearBase):
    """
    block 8-bit weight and activation quantized linear layer.
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
        block_size=128,
    ):
        super().__init__()

        # Some platforms do not support float8, but we can run them with `infer.raise_lower_bit_float_to=bfloat16`.
        # However, we need to treat float8 items as uint8 first, to avoid the missing ops on these platforms.
        args = get_global_args()
        if parse_dtype(args.infer.raise_lower_bit_float_to).itemsize > 1:
            dtype = torch.uint8
        else:
            dtype = torch.float8_e4m3fn
        assert dtype.itemsize == 1

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        self.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty((out_features, in_features), dtype=dtype),
                requires_grad=False,
            ),
        )

        scale_out_features = (out_features + block_size - 1) // block_size
        scale_in_features = (in_features + block_size - 1) // block_size
        self.register_parameter(
            "scale",
            torch.nn.Parameter(
                torch.empty(
                    scale_out_features,
                    scale_in_features,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            ),
        )

        if has_bias:
            self.register_parameter(
                "bias",
                torch.nn.Parameter(
                    torch.empty(out_features, dtype=bias_dtype), requires_grad=False
                ),
            )
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        return linear_block_fp8(
            x, self.weight, self.scale, self.bias, block_size=self.block_size
        )


@QuantizationRegistry.register_linear(
    "blockfp8",
    when=lambda: has_marlin
    and parse_dtype(get_global_args().infer.raise_lower_bit_float_to).itemsize > 1,
    priority=1,
)
class Blockfp8LinearMarlinLayout(
    enable_native_layout_weight("weight", MarlinNativeLayoutWeight),
    enable_native_layout_weight("scale", MarlinNativeLayoutScale),
    Blockfp8Linear,
):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return soft_fp8_blockfp8_gemm_marlin(
            x,
            self.get_native_layout_weight(),
            self.get_native_layout_scale(),
        )


@QuantizationRegistry.register_moe_experts("blockfp8")
class Blockfp8MoeExperts(QuantizedMoeExpertsBase):
    """
    blockfp8 quantized MoeExperts
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
        layer_id: int,
        ############################################
        # No parameters specific to this quantization
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
            layer_id,
        )

        # Some platforms do not support float8, but we can run them with `infer.raise_lower_bit_float_to=bfloat16`.
        # However, we need to treat float8 items as uint8 first, to avoid the missing ops on these platforms.
        args = get_global_args()
        if parse_dtype(args.infer.raise_lower_bit_float_to).itemsize > 1:
            dtype = torch.uint8
        else:
            dtype = torch.float8_e4m3fn
        assert dtype.itemsize == 1

        gate_up_proj_in_features = dim
        block_size = 128

        if self.merge_gate_up:
            self.gate_up_proj_weight = torch.nn.Parameter(
                torch.empty(
                    (self.group_size, moe_inter_dim * 2, self.dim),
                    dtype=dtype,
                ),
                requires_grad=False,
            )
            scale_out_features = (moe_inter_dim * 2 + block_size - 1) // block_size
            scale_in_features = (
                gate_up_proj_in_features + block_size - 1
            ) // block_size
            self.gate_up_proj_scale = torch.nn.Parameter(
                torch.empty(
                    self.group_size,
                    scale_out_features,
                    scale_in_features,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        else:
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
            scale_out_features = (moe_inter_dim + block_size - 1) // block_size
            scale_in_features = (
                gate_up_proj_in_features + block_size - 1
            ) // block_size
            self.gate_proj_scale = torch.nn.Parameter(
                torch.empty(
                    self.group_size,
                    scale_out_features,
                    scale_in_features,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.up_proj_scale = torch.nn.Parameter(
                torch.empty(
                    self.group_size,
                    scale_out_features,
                    scale_in_features,
                    dtype=torch.float32,
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
        down_proj_scale_out_features = (dim + block_size - 1) // block_size
        down_proj_scale_in_features = (moe_inter_dim + block_size - 1) // block_size
        self.down_proj_scale = torch.nn.Parameter(
            torch.empty(
                self.group_size,
                down_proj_scale_out_features,
                down_proj_scale_in_features,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    @override
    def forward(
        self,
        routed_x: BatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        if has_triton and self.merge_gate_up:
            fused_soft_fp8 = False
            use_fp8_w8a8 = False
            if (
                parse_dtype(get_global_args().infer.raise_lower_bit_float_to).itemsize
                == 1
                or is_nvidia()
                or is_muxi()
            ):
                fused_soft_fp8 = (
                    parse_dtype(
                        get_global_args().infer.raise_lower_bit_float_to
                    ).itemsize
                    != 1
                )
                gate_up_proj_weight = self.gate_up_proj_weight
                gate_up_proj_scale = self.gate_up_proj_scale
                down_proj_weight = self.down_proj_weight
                down_proj_scale = self.down_proj_scale
                use_fp8_w8a8 = True
            else:
                logger.warning(
                    f"Soft-fp8 fused gemm not implemented for {get_device_name()}, falling back to soft-fp8 conversion"
                )
                block_size = 128
                gate_up_proj_weight = soft_fp8_blockfp8_weight_dequant(
                    self.gate_up_proj_weight,
                    self.gate_up_proj_scale,
                    block_size,
                )
                gate_up_proj_scale = None
                down_proj_weight = soft_fp8_blockfp8_weight_dequant(
                    self.down_proj_weight,
                    self.down_proj_scale,
                    block_size,
                )
                down_proj_scale = None

            if self.fuse_shared_experts:
                if isinstance(routed_x, IndexedBatchedRoutedActivation):
                    x, indices = routed_x.activation, routed_x.token_to_expert_indices
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
                    weights, indices = final_weights, final_indices
                    routed_x = IndexedBatchedRoutedActivation(x, indices)
                else:
                    raise NotImplementedError()

            return fused_experts(
                routed_x,
                w1=gate_up_proj_weight,
                w2=down_proj_weight,
                topk_weights=weights,
                inplace=inplace,
                use_fp8_w8a8=use_fp8_w8a8,
                w1_scale=gate_up_proj_scale,
                w2_scale=down_proj_scale,
                block_shape=[128, 128],
                soft_fp8=fused_soft_fp8,
                experts_start_idx=self.experts_start_idx,
                impl=impl,
            )

        else:
            return super().forward(routed_x, weights, inplace=inplace, impl=impl)

    @override
    def forward_ith_expert_gate_up(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear_block_fp8(
            x,
            self.gate_up_proj_weight[i],
            self.gate_up_proj_scale[i],
            None,
            128,
        )

    @override
    def forward_ith_expert_gate(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear_block_fp8(
            x,
            self.gate_proj_weight[i],
            self.gate_proj_scale[i],
            None,
            128,
        )

    @override
    def forward_ith_expert_up(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear_block_fp8(
            x,
            self.up_proj_weight[i],
            self.up_proj_scale[i],
            None,
            128,
        )

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear_block_fp8(
            x,
            self.down_proj_weight[i],
            self.down_proj_scale[i],
            None,
            128,
        )


@QuantizationRegistry.register_absorb_gemm("blockfp8")
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
        block_size: int = 128,
    ):
        super().__init__()

        # Some platforms do not support float8, but we can run them with `infer.raise_lower_bit_float_to=bfloat16`.
        # However, we need to treat float8 items as uint8 first, to avoid the missing ops on these platforms.
        args = get_global_args()
        if parse_dtype(args.infer.raise_lower_bit_float_to).itemsize > 1:
            dtype = torch.uint8
        else:
            dtype = torch.float8_e4m3fn

        self.weight = torch.nn.Parameter(
            torch.empty(
                n_heads, out_features_per_head, in_features_per_head, dtype=dtype
            ),
            requires_grad=False,
        )

        assert out_features_per_head % block_size == 0
        assert in_features_per_head % block_size == 0
        self.scale = torch.nn.Parameter(
            torch.empty(
                n_heads,
                out_features_per_head // block_size,
                in_features_per_head // block_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        self.n_heads = n_heads
        self.in_features_per_head = in_features_per_head
        self.out_features_per_head = out_features_per_head
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            seq, n_head, n_hidden = x.shape
            bs = None
        else:
            bs, seq, n_head, n_hidden = x.shape
            x = x.view(bs * seq, n_head, n_hidden)

        y = blockfp8_einsum_shc_hdc_shd(
            x,
            self.weight,
            self.scale,
            block_size=self.block_size,
            soft_fp8=(get_global_args().infer.raise_lower_bit_float_to == "bfloat16"),
        )

        if bs is not None:
            y = y.view(bs, seq, y.shape[-2], y.shape[-1])
        return y
