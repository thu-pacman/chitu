# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import functools
from typing_extensions import override
from logging import getLogger

import torch

from chitu.quantization.base import (
    QuantizedLinearBase,
    QuantizedMoeExpertsUnmerged,
    QuantizedMoeExpertsMerged,
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
from chitu.moe.experts import make_op_dispatcher
from chitu.device_type import get_device_name, is_muxi, is_nvidia
from chitu.utils import (
    try_import_platform_dep,
    parse_dtype,
    try_import_opt_dep,
    try_import_and_setup_torch_npu,
)
from chitu.utils import parse_dtype, ceil_div
from chitu.import_utils import try_import_platform_dep, try_import_opt_dep
from chitu.global_vars import get_global_args
from chitu.cuda_graph import is_warming_up_or_cuda_graph_capture
from chitu.native_layout import (
    enable_native_layout_weight,
    MarlinNativeLayoutWeight,
    MarlinNativeLayoutScale,
    DeepGemmScale,
)
from chitu.moe.batched_expert_result import BatchedExpertResult
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    IndexedBatchedRoutedActivationBlockfp8,
    IndexedBatchedRoutedActivationWithPaddedPerExpertCnt,
    IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt,
    ConcatPermutedBatchedRoutedActivationMinimal,
    PerExpertDenseBatchedRoutedActivationMinimal,
)

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")

has_marlin = has_chitu_backend and hasattr(chitu_backend, "gptq_marlin_gemm")

if has_torch_npu:
    from chitu.npu_utils import fused_experts_no_sum_npu, fused_experts_npu_for_ep
if has_triton:
    from chitu.moe.experts import fused_experts_fp8
if has_deep_gemm:
    from chitu.moe.experts import deepgemm_masked_fused_expert
    from chitu.moe.experts import deepgemm_contiguous_fused_expert

logger = getLogger(__name__)


def linear_blockfp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    x_scale: Optional[torch.Tensor] = None,
    block_size: int,
    round_scale_to_pow2: bool,
) -> torch.Tensor:
    """
    Quantized linear with blockfp8 quantization.

    Args:
        x (torch.Tensor): The input tensor, maybe in fp16, bf16 or fp8. If in fp8,
            `x_scale` should also be set.
        weight (torch.Tensor): The weight tensor, in fp8.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.
        x_scale: The scale of the input tensor, if the input tensor is quantized.
        round_scale_to_pow2: Round scale to powers of 2. But it does not necessarily
            mean the scale must be stored as a 8-bit integer. Implementations are
            free to pick a storage data type for it.

    Returns:
        torch.Tensor: The result of the linear transformation.
    """

    assert weight.element_size() == 1

    if get_global_args().infer.raise_lower_bit_float_to == "bfloat16":
        if x.dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError(f"Unsupported input type: {x.dtype}")
        if x_scale is not None:
            raise ValueError(f"No x_scale is supported for {x.dtype=}")
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
        x_shape = x.shape
        if x.dtype in {torch.float16, torch.bfloat16}:
            x = x.view(-1, x_shape[-1])
            x, x_scale = blockfp8_act_quant(
                x, block_size=block_size, round_scale_to_pow2=round_scale_to_pow2
            )
        elif x.dtype == torch.float8_e4m3fn:
            assert x_scale is not None
            x = x.view(-1, x_shape[-1])
            x_scale = x_scale.view(-1, x_scale.shape[-1])
        else:
            raise ValueError(f"Unsupported input type: {x.dtype}")
        assert weight_scale is not None
        y = blockfp8_gemm(
            x,
            x_scale,
            weight,
            weight_scale,
            block_size=block_size,
            round_scale_to_pow2=round_scale_to_pow2,
        )
        if bias is not None:
            y = (y + bias).to(y.dtype)
        return y.view(x_shape[:-1] + y.shape[-1:])


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
        block_size: int = 128,
        round_scale_to_pow2: bool = False,
    ):
        """
        Linear layer with blockfp8 quantization

        Additional args of this inheritance:
            bias_dtype: Data type for bias. Only applied when `has_bias` is True.
            block_size: Size of the quantization group along both of the weight
                dimensions, which means each group is a square.
            round_scale_to_pow2: Round scale to powers of 2. But it does not necessarily
                mean the scale must be stored as a 8-bit integer. Implementations are
                free to pick a storage data type for it.
        """

        super().__init__(in_features, out_features, has_bias)

        # Some platforms do not support float8, but we can run them with `infer.raise_lower_bit_float_to=bfloat16`.
        # However, we need to treat float8 items as uint8 first, to avoid the missing ops on these platforms.
        args = get_global_args()
        if parse_dtype(args.infer.raise_lower_bit_float_to).itemsize > 1:
            dtype = torch.uint8
        else:
            dtype = torch.float8_e4m3fn
        assert dtype.itemsize == 1

        self.block_size = block_size
        self.round_scale_to_pow2 = round_scale_to_pow2

        self.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty((out_features, in_features), dtype=dtype),
                requires_grad=False,
            ),
        )

        scale_out_features = ceil_div(out_features, block_size)
        scale_in_features = ceil_div(in_features, block_size)
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
        return linear_blockfp8(
            x,
            self.weight,
            self.scale,
            self.bias,
            block_size=self.block_size,
            round_scale_to_pow2=self.round_scale_to_pow2,
        )


@QuantizationRegistry.register_linear(
    "blockfp8",
    when=lambda _: has_marlin
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


@QuantizationRegistry.register_linear(
    "blockfp8",
    when=lambda quant_kwargs: quant_kwargs.get("block_size", 128) == 128
    and has_deep_gemm
    and torch.get_default_dtype() == torch.bfloat16
    and (
        torch.cuda.get_device_capability()[0] == 9
        or (
            torch.cuda.get_device_capability()[0] == 10
            and quant_kwargs.get("round_scale_to_pow2", False)
        )
    ),
    priority=1,
)
class Blockfp8LinearDeepGemm(
    enable_native_layout_weight(
        "scale",
        DeepGemmScale,
        mn=lambda m: m.out_features,
        k=lambda m: m.in_features,
        disable_ue8m0_cast=lambda m: not m.round_scale_to_pow2,
    ),
    Blockfp8Linear,
):
    @override
    def forward(self, x) -> torch.Tensor:
        return linear_blockfp8(
            x,
            self.weight,
            self.get_native_layout_scale(),
            self.bias,
            block_size=self.block_size,
            round_scale_to_pow2=self.round_scale_to_pow2,
            # TODO: Call deep_gemm implementation only. No dispatching
        )


def _finalize_fused_experts_sum_output(
    output, hidden_states, topk_weights: Optional[torch.Tensor], inplace: bool
):
    if hasattr(output, "weighted_sum"):
        if (
            inplace
            and isinstance(hidden_states, IndexedBatchedRoutedActivation)
            and hidden_states.activation.dtype == torch.get_default_dtype()
        ):
            out = hidden_states.activation
        else:
            out = None
        return output.weighted_sum(topk_weights, out=out)
    return output


@make_op_dispatcher
def fused_experts_no_sum_blockfp8_indexed(
    hidden_states,
    w1,
    w2,
    *,
    impl: str = "auto",
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    round_scale_to_pow2: bool = False,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
): ...


@fused_experts_no_sum_blockfp8_indexed.register_auto
def _auto_fused_experts_no_sum_blockfp8_indexed():
    if has_triton:
        return "triton"
    if has_deep_gemm:
        return "deepgemm"
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


@fused_experts_no_sum_blockfp8_indexed.register("triton")
def _fused_experts_no_sum_blockfp8_indexed_triton(
    hidden_states,
    w1,
    w2,
    *,
    impl: str = "triton",
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    round_scale_to_pow2: bool = False,
    experts_start_idx: int = 0,
):
    return fused_experts_fp8(
        hidden_states,
        w1=w1,
        w2=w2,
        activation=activation,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
        soft_fp8=soft_fp8,
        round_scale_to_pow2=round_scale_to_pow2,
        experts_start_idx=experts_start_idx,
    )


@fused_experts_no_sum_blockfp8_indexed.register("torch_npu")
def _fused_experts_no_sum_blockfp8_indexed_torch_npu(
    hidden_states,
    w1,
    w2,
    *,
    impl: str = "torch_npu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
):
    return fused_experts_no_sum_npu(
        hidden_states,
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        global_num_experts=global_num_experts,
        experts_start_idx=experts_start_idx,
    )


@fused_experts_no_sum_blockfp8_indexed.register("deepgemm")
def _fused_experts_no_sum_blockfp8_indexed_deepgemm(
    hidden_states,
    w1,
    w2,
    *,
    impl: str = "deepgemm",
    activation: Optional[str] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    round_scale_to_pow2: bool = False,
    experts_start_idx: int = 0,
):
    return deepgemm_contiguous_fused_expert(
        hidden_states,
        w1=w1,
        w2=w2,
        activation=activation,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
        round_scale_to_pow2=round_scale_to_pow2,
        experts_start_idx=experts_start_idx,
    )


@make_op_dispatcher
def fused_experts_sum_blockfp8_indexed(
    hidden_states,
    w1,
    w2,
    topk_weights: Optional[torch.Tensor],
    *,
    inplace: bool = False,
    impl: str = "auto",
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    round_scale_to_pow2: bool = False,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
): ...


@fused_experts_sum_blockfp8_indexed.register_auto
def _auto_fused_experts_sum_blockfp8_indexed():
    if has_triton:
        return "triton"
    if has_deep_gemm:
        return "deepgemm"
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


def _resolve_indexed_blockfp8_impl(
    routed_x: IndexedBatchedRoutedActivation, impl: str, *, soft_fp8: bool = False
) -> str:
    if impl != "auto":
        return impl
    if soft_fp8:
        return "triton"
    # DeepEP prefill may produce indexed inputs with padded per-expert counts.
    # These paths should go through deepgemm contiguous kernels instead of Triton.
    if isinstance(
        routed_x,
        (
            IndexedBatchedRoutedActivationWithPaddedPerExpertCnt,
            IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt,
        ),
    ):
        if has_deep_gemm:
            return "deepgemm"
    # For blockfp8 indexed input:
    # - prefill prefers deepgemm contiguous
    # - decode (CUDA graph warmup/capture phase) prefers triton
    if isinstance(routed_x, IndexedBatchedRoutedActivationBlockfp8):
        if is_warming_up_or_cuda_graph_capture():
            if has_triton:
                return "triton"
        else:
            if has_deep_gemm:
                return "deepgemm"
    # For non-blockfp8 indexed input:
    # - prefill prefers deepgemm contiguous
    # - decode (CUDA graph warmup/capture phase) prefers triton
    if isinstance(routed_x, IndexedBatchedRoutedActivation):
        if is_warming_up_or_cuda_graph_capture():
            if has_triton:
                return "triton"
        else:
            if has_deep_gemm:
                return "deepgemm"
    if has_triton:
        return "triton"
    if has_deep_gemm:
        return "deepgemm"
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError("No available implementation for indexed blockfp8 MoE")


@fused_experts_sum_blockfp8_indexed.register("triton")
@fused_experts_sum_blockfp8_indexed.register("torch_npu")
@fused_experts_sum_blockfp8_indexed.register("deepgemm")
def _fused_experts_sum_blockfp8_indexed_any(
    hidden_states,
    w1,
    w2,
    topk_weights: Optional[torch.Tensor],
    *,
    inplace: bool = False,
    impl: str = "auto",
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    round_scale_to_pow2: bool = False,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
):
    output = fused_experts_no_sum_blockfp8_indexed(
        hidden_states,
        w1,
        w2,
        impl=impl,
        activation=activation,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
        soft_fp8=soft_fp8,
        round_scale_to_pow2=round_scale_to_pow2,
        global_num_experts=global_num_experts,
        experts_start_idx=experts_start_idx,
    )
    return _finalize_fused_experts_sum_output(
        output, hidden_states, topk_weights=topk_weights, inplace=inplace
    )


@make_op_dispatcher
def fused_experts_no_sum_blockfp8_concat_permuted(
    hidden_states,
    w1,
    w2,
    *,
    impl: str = "auto",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    experts_start_idx: int = 0,
): ...


@fused_experts_no_sum_blockfp8_concat_permuted.register_auto
def _auto_fused_experts_no_sum_blockfp8_concat_permuted():
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


@fused_experts_no_sum_blockfp8_concat_permuted.register("torch_npu")
def _fused_experts_no_sum_blockfp8_concat_permuted_torch_npu(
    hidden_states,
    w1,
    w2,
    *,
    impl: str = "torch_npu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    experts_start_idx: int = 0,
):
    return fused_experts_npu_for_ep(
        hidden_states,
        w1=w1,
        w1_scale=w1_scale,  # fp32
        w2=w2,
        w2_scale=w2_scale,  # bf16
        experts_start_idx=experts_start_idx,
    )


@make_op_dispatcher
def fused_experts_sum_blockfp8_concat_permuted(
    hidden_states,
    w1,
    w2,
    topk_weights: Optional[torch.Tensor],
    *,
    inplace: bool = False,
    impl: str = "auto",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    experts_start_idx: int = 0,
): ...


@fused_experts_sum_blockfp8_concat_permuted.register_auto
def _auto_fused_experts_sum_blockfp8_concat_permuted():
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


@fused_experts_sum_blockfp8_concat_permuted.register("torch_npu")
def _fused_experts_sum_blockfp8_concat_permuted_torch_npu(
    hidden_states,
    w1,
    w2,
    topk_weights: Optional[torch.Tensor],
    *,
    inplace: bool = False,
    impl: str = "torch_npu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    experts_start_idx: int = 0,
):
    output = fused_experts_no_sum_blockfp8_concat_permuted(
        hidden_states,
        w1,
        w2,
        impl=impl,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        experts_start_idx=experts_start_idx,
    )
    return _finalize_fused_experts_sum_output(
        output, hidden_states, topk_weights=topk_weights, inplace=inplace
    )


@make_op_dispatcher
def fused_experts_no_sum_blockfp8_per_expert_dense(
    hidden_states,
    w1,
    w2,
    *,
    impl: str = "auto",
    activation: Optional[str] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    round_scale_to_pow2: bool = False,
    experts_start_idx: int = 0,
): ...


@fused_experts_no_sum_blockfp8_per_expert_dense.register_auto
def _auto_fused_experts_no_sum_blockfp8_per_expert_dense():
    if has_deep_gemm:
        return "deepgemm"
    raise NotImplementedError


@fused_experts_no_sum_blockfp8_per_expert_dense.register("deepgemm")
def _fused_experts_no_sum_blockfp8_per_expert_dense_deepgemm(
    hidden_states,
    w1,
    w2,
    *,
    impl: str = "deepgemm",
    activation: Optional[str] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    round_scale_to_pow2: bool = False,
    experts_start_idx: int = 0,
):
    return deepgemm_masked_fused_expert(
        hidden_states,
        w1=w1,
        w2=w2,
        activation=activation,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
        round_scale_to_pow2=round_scale_to_pow2,
        experts_start_idx=experts_start_idx,
    )


@make_op_dispatcher
def fused_experts_sum_blockfp8_per_expert_dense(
    hidden_states,
    w1,
    w2,
    topk_weights: Optional[torch.Tensor],
    *,
    inplace: bool = False,
    impl: str = "auto",
    activation: Optional[str] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    round_scale_to_pow2: bool = False,
    experts_start_idx: int = 0,
): ...


@fused_experts_sum_blockfp8_per_expert_dense.register_auto
def _auto_fused_experts_sum_blockfp8_per_expert_dense():
    if has_deep_gemm:
        return "deepgemm"
    raise NotImplementedError


@fused_experts_sum_blockfp8_per_expert_dense.register("deepgemm")
def _fused_experts_sum_blockfp8_per_expert_dense_deepgemm(
    hidden_states,
    w1,
    w2,
    topk_weights: Optional[torch.Tensor],
    *,
    inplace: bool = False,
    impl: str = "deepgemm",
    activation: Optional[str] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    round_scale_to_pow2: bool = False,
    experts_start_idx: int = 0,
):
    output = fused_experts_no_sum_blockfp8_per_expert_dense(
        hidden_states,
        w1,
        w2,
        impl=impl,
        activation=activation,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
        round_scale_to_pow2=round_scale_to_pow2,
        experts_start_idx=experts_start_idx,
    )
    return _finalize_fused_experts_sum_output(
        output, hidden_states, topk_weights=topk_weights, inplace=inplace
    )


@QuantizationRegistry.register_moe_experts("blockfp8", merge_gate_up=False)
class Blockfp8MoeExpertsUnmerged(QuantizedMoeExpertsUnmerged):
    """
    blockfp8 quantized MoeExperts with unmerged gate and up projection
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
        block_size: int = 128,
        round_scale_to_pow2: bool = False,
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

        self.block_size = block_size
        self.round_scale_to_pow2 = round_scale_to_pow2

        # Some platforms do not support float8, but we can run them with `infer.raise_lower_bit_float_to=bfloat16`.
        # However, we need to treat float8 items as uint8 first, to avoid the missing ops on these platforms.
        args = get_global_args()
        if parse_dtype(args.infer.raise_lower_bit_float_to).itemsize > 1:
            dtype = torch.uint8
        else:
            dtype = torch.float8_e4m3fn
        assert dtype.itemsize == 1

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
        scale_out_features = ceil_div(moe_inter_dim, block_size)
        scale_in_features = ceil_div(dim, block_size)
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
        down_proj_scale_out_features = ceil_div(dim, block_size)
        down_proj_scale_in_features = ceil_div(moe_inter_dim, block_size)
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
    def forward_ith_expert_gate(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return linear_blockfp8(
            x,
            self.gate_proj_weight[i],
            self.gate_proj_scale[i],
            None,
            x_scale=x_scale,
            block_size=self.block_size,
            round_scale_to_pow2=self.round_scale_to_pow2,
        )

    @override
    def forward_ith_expert_up(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return linear_blockfp8(
            x,
            self.up_proj_weight[i],
            self.up_proj_scale[i],
            None,
            x_scale=x_scale,
            block_size=self.block_size,
            round_scale_to_pow2=self.round_scale_to_pow2,
        )

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear_blockfp8(
            x,
            self.down_proj_weight[i],
            self.down_proj_scale[i],
            None,
            block_size=self.block_size,
            round_scale_to_pow2=self.round_scale_to_pow2,
        )


@QuantizationRegistry.register_moe_experts("blockfp8", merge_gate_up=True)
class Blockfp8MoeExpertsMerged(QuantizedMoeExpertsMerged):
    """
    blockfp8 quantized MoeExperts with merged gate and up projection
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
        block_size: int = 128,
        round_scale_to_pow2: bool = False,
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

        self.block_size = block_size
        self.round_scale_to_pow2 = round_scale_to_pow2

        # Some platforms do not support float8, but we can run them with `infer.raise_lower_bit_float_to=bfloat16`.
        # However, we need to treat float8 items as uint8 first, to avoid the missing ops on these platforms.
        args = get_global_args()
        if parse_dtype(args.infer.raise_lower_bit_float_to).itemsize > 1:
            dtype = torch.uint8
        else:
            dtype = torch.float8_e4m3fn
        assert dtype.itemsize == 1

        self.gate_up_proj_weight = torch.nn.Parameter(
            torch.empty(
                (self.group_size, moe_inter_dim * 2, self.dim),
                dtype=dtype,
            ),
            requires_grad=False,
        )
        scale_out_features = ceil_div(moe_inter_dim * 2, block_size)
        scale_in_features = ceil_div(dim, block_size)
        self.gate_up_proj_scale = torch.nn.Parameter(
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
        down_proj_scale_out_features = ceil_div(dim, block_size)
        down_proj_scale_in_features = ceil_div(moe_inter_dim, block_size)
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
    @functools.singledispatchmethod
    def forward_no_sum(
        self, routed_x: BatchedRoutedActivation, impl: str = "auto"
    ) -> BatchedExpertResult:
        return super().forward_no_sum(routed_x, impl=impl)

    def _resolve_runtime_weights(self):
        if has_triton:
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

            return (
                gate_up_proj_weight,
                gate_up_proj_scale,
                down_proj_weight,
                down_proj_scale,
                fused_soft_fp8,
                use_fp8_w8a8,
            )
        return None

    @forward_no_sum.register
    def _(
        self, routed_x: IndexedBatchedRoutedActivation, impl: str = "auto"
    ) -> BatchedExpertResult:
        resolved = self._resolve_runtime_weights()
        if resolved is None:
            return super().forward_no_sum(routed_x, impl=impl)
        (
            gate_up_proj_weight,
            gate_up_proj_scale,
            down_proj_weight,
            down_proj_scale,
            fused_soft_fp8,
            _,
        ) = resolved
        impl = _resolve_indexed_blockfp8_impl(routed_x, impl, soft_fp8=fused_soft_fp8)
        return fused_experts_no_sum_blockfp8_indexed(
            routed_x,
            w1=gate_up_proj_weight,
            w2=down_proj_weight,
            activation="silu",
            w1_scale=gate_up_proj_scale,
            w2_scale=down_proj_scale,
            block_shape=[self.block_size, self.block_size],
            round_scale_to_pow2=self.round_scale_to_pow2,
            soft_fp8=fused_soft_fp8,
            global_num_experts=self.global_n_experts,
            experts_start_idx=self.experts_start_idx,
            impl=impl,
        )

    @forward_no_sum.register
    def _(
        self,
        routed_x: ConcatPermutedBatchedRoutedActivationMinimal,
        impl: str = "auto",
    ) -> BatchedExpertResult:
        resolved = self._resolve_runtime_weights()
        if resolved is None:
            return super().forward_no_sum(routed_x, impl=impl)
        impl = "torch_npu" if impl == "auto" else impl
        (
            gate_up_proj_weight,
            gate_up_proj_scale,
            down_proj_weight,
            down_proj_scale,
            fused_soft_fp8,
            _,
        ) = resolved
        return fused_experts_no_sum_blockfp8_concat_permuted(
            routed_x,
            w1=gate_up_proj_weight,
            w2=down_proj_weight,
            w1_scale=gate_up_proj_scale,
            w2_scale=down_proj_scale,
            experts_start_idx=self.experts_start_idx,
            impl=impl,
        )

    @forward_no_sum.register
    def _(
        self,
        routed_x: PerExpertDenseBatchedRoutedActivationMinimal,
        impl: str = "auto",
    ) -> BatchedExpertResult:
        resolved = self._resolve_runtime_weights()
        if resolved is None:
            return super().forward_no_sum(routed_x, impl=impl)
        if impl == "auto":
            if has_deep_gemm:
                impl = "deepgemm"
            else:
                return super().forward_no_sum(routed_x, impl=impl)
        (
            gate_up_proj_weight,
            gate_up_proj_scale,
            down_proj_weight,
            down_proj_scale,
            fused_soft_fp8,
            _,
        ) = resolved
        return fused_experts_no_sum_blockfp8_per_expert_dense(
            routed_x,
            w1=gate_up_proj_weight,
            w2=down_proj_weight,
            activation="silu",
            w1_scale=gate_up_proj_scale,
            w2_scale=down_proj_scale,
            block_shape=[self.block_size, self.block_size],
            round_scale_to_pow2=self.round_scale_to_pow2,
            experts_start_idx=self.experts_start_idx,
            impl=impl,
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
    def _(
        self,
        routed_x: IndexedBatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        resolved = self._resolve_runtime_weights()
        if resolved is None:
            return super().forward(routed_x, weights, inplace=inplace, impl=impl)
        (
            gate_up_proj_weight,
            gate_up_proj_scale,
            down_proj_weight,
            down_proj_scale,
            fused_soft_fp8,
            _,
        ) = resolved
        impl = _resolve_indexed_blockfp8_impl(routed_x, impl, soft_fp8=fused_soft_fp8)
        return fused_experts_sum_blockfp8_indexed(
            routed_x,
            w1=gate_up_proj_weight,
            w2=down_proj_weight,
            topk_weights=weights,
            activation="silu",
            inplace=inplace,
            w1_scale=gate_up_proj_scale,
            w2_scale=down_proj_scale,
            block_shape=[self.block_size, self.block_size],
            round_scale_to_pow2=self.round_scale_to_pow2,
            soft_fp8=fused_soft_fp8,
            global_num_experts=self.global_n_experts,
            experts_start_idx=self.experts_start_idx,
            impl=impl,
        )

    @forward.register
    def _(
        self,
        routed_x: ConcatPermutedBatchedRoutedActivationMinimal,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        resolved = self._resolve_runtime_weights()
        if resolved is None:
            return super().forward(routed_x, weights, inplace=inplace, impl=impl)
        impl = "torch_npu" if impl == "auto" else impl
        (
            gate_up_proj_weight,
            gate_up_proj_scale,
            down_proj_weight,
            down_proj_scale,
            fused_soft_fp8,
            _,
        ) = resolved
        return fused_experts_sum_blockfp8_concat_permuted(
            routed_x,
            w1=gate_up_proj_weight,
            w2=down_proj_weight,
            topk_weights=weights,
            inplace=inplace,
            w1_scale=gate_up_proj_scale,
            w2_scale=down_proj_scale,
            experts_start_idx=self.experts_start_idx,
            impl=impl,
        )

    @forward.register
    def _(
        self,
        routed_x: PerExpertDenseBatchedRoutedActivationMinimal,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        resolved = self._resolve_runtime_weights()
        if resolved is None:
            return super().forward(routed_x, weights, inplace=inplace, impl=impl)
        if impl == "auto":
            if has_deep_gemm:
                impl = "deepgemm"
            else:
                return super().forward(routed_x, weights, inplace=inplace, impl=impl)
        (
            gate_up_proj_weight,
            gate_up_proj_scale,
            down_proj_weight,
            down_proj_scale,
            _,
            _,
        ) = resolved
        return fused_experts_sum_blockfp8_per_expert_dense(
            routed_x,
            w1=gate_up_proj_weight,
            w2=down_proj_weight,
            topk_weights=weights,
            activation="silu",
            inplace=inplace,
            w1_scale=gate_up_proj_scale,
            w2_scale=down_proj_scale,
            block_shape=[self.block_size, self.block_size],
            round_scale_to_pow2=self.round_scale_to_pow2,
            experts_start_idx=self.experts_start_idx,
            impl=impl,
        )

    @override
    def forward_ith_expert_gate_up(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return linear_blockfp8(
            x,
            self.gate_up_proj_weight[i],
            self.gate_up_proj_scale[i],
            None,
            x_scale=x_scale,
            block_size=self.block_size,
            round_scale_to_pow2=self.round_scale_to_pow2,
        )

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear_blockfp8(
            x,
            self.down_proj_weight[i],
            self.down_proj_scale[i],
            None,
            block_size=self.block_size,
            round_scale_to_pow2=self.round_scale_to_pow2,
        )


@QuantizationRegistry.register_absorb_gemm("blockfp8")
class Blockfp8AbsorbGemm(QuantizedAbsorbGemmBase):
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
        round_scale_to_pow2: bool = False,
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

        if out_features_per_head % block_size != 0:
            raise NotImplementedError(
                f"This model does not support infer.mla_absorb=absorb-without-precomp because otherwise "
                f"out_features_per_head({out_features_per_head}) of the absorbing group gemm will not be "
                f"a multiple of block_size({block_size}). Please use infer.mla_absorb=none or "
                f"infer.mla_absorb=absorb instead."
            )
        if in_features_per_head % block_size != 0:
            raise NotImplementedError(
                f"This model does not support infer.mla_absorb=absorb-without-precomp because otherwise "
                f"in_features_per_head({in_features_per_head}) of the absorbing group gemm will not be "
                f"a multiple of block_size({block_size}). Please use infer.mla_absorb=none or "
                f"infer.mla_absorb=absorb instead."
            )
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
