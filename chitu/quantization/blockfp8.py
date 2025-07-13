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
from chitu.ops import (
    fp8_gemm_deepseek_v3,
    soft_fp8_gemm_deepseek_v3,
    weight_dequant_soft_fp8_deepseek_v3,
    act_quant_deepseek_v3,
    quant_einsum_shc_hdc_shd,
)
from chitu.device_type import get_device_name, is_muxi, is_nvidia
from chitu.utils import try_import_opt_dep, parse_dtype
from chitu.global_vars import get_global_args
from chitu.ops import weight_dequant_soft_fp8_deepseek_v3

chitu_backend, has_chitu_backend = try_import_opt_dep("chitu_backend", "chitu_backend")
triton, has_triton = try_import_opt_dep("triton", "triton")
if has_triton:
    from chitu.fused_moe import fused_experts


logger = getLogger(__name__)


def linear_block_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    block_size: Optional[int] = 128,
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
        if is_nvidia() or is_muxi():
            y = soft_fp8_gemm_deepseek_v3(x, weight, weight_scale)
            if bias is not None:
                y += bias
            return y
        else:
            logger.warning(
                f"Soft-fp8 fused gemm not implemented for {get_device_name()}, falling back to soft-fp8 conversion"
            )
            weight_dequanted = weight_dequant_soft_fp8_deepseek_v3(
                weight, weight_scale, block_size
            )
            return torch.nn.functional.linear(x, weight_dequanted, bias)
    else:
        x_dtype = x.dtype
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        x, act_scale = act_quant_deepseek_v3(x, block_size)
        assert weight_scale is not None
        y = fp8_gemm_deepseek_v3(x, act_scale, weight, weight_scale)
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

        if has_triton and self.merge_gate_up:
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
            else:
                logger.warning(
                    f"Soft-fp8 fused gemm not implemented for {get_device_name()}, falling back to soft-fp8 conversion"
                )
                block_size = 128
                gate_up_proj_weight = weight_dequant_soft_fp8_deepseek_v3(
                    self.gate_up_proj_weight,
                    self.gate_up_proj_scale,
                    block_size,
                )
                gate_up_proj_scale = None
                down_proj_weight = weight_dequant_soft_fp8_deepseek_v3(
                    self.down_proj_weight,
                    self.down_proj_scale,
                    block_size,
                )
                down_proj_scale = None
                fused_soft_fp8 = False

            if not self.fuse_shared_experts:
                y = fused_experts(
                    x,
                    gate_up_proj_weight,
                    down_proj_weight,
                    topk_weights=weights,
                    topk_ids=indices,
                    use_fp8_w8a8=True,
                    inplace=True,
                    global_num_experts=self.n_routed_experts,
                    w1_scale=gate_up_proj_scale,
                    w2_scale=down_proj_scale,
                    block_shape=[128, 128],
                    soft_fp8=fused_soft_fp8,
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
                    gate_up_proj_weight,
                    down_proj_weight,
                    topk_weights=new_weights,
                    topk_ids=new_indices,
                    use_fp8_w8a8=True,
                    inplace=True,
                    global_num_experts=self.n_routed_experts + self.n_shared_experts,
                    w1_scale=gate_up_proj_scale,
                    w2_scale=down_proj_scale,
                    block_shape=[128, 128],
                    soft_fp8=fused_soft_fp8,
                )

        else:
            y = self.forward_iterative(x, weights, indices)

        return y.view(shape)

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

        y = quant_einsum_shc_hdc_shd(
            x,
            self.weight,
            self.scale,
            block_size=self.block_size,
            soft_fp8=(get_global_args().infer.raise_lower_bit_float_to == "bfloat16"),
        )

        if bs is not None:
            y = y.view(bs, seq, y.shape[-2], y.shape[-1])
        return y
