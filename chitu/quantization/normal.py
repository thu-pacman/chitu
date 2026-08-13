# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import functools
import torch
import ctypes

from chitu.checkpoint_prefix import CheckpointPrefix
from chitu.quantization.base import (
    QuantizedLinearBase,
    QuantizedMoeExpertsUnmerged,
    QuantizedMoeExpertsMerged,
    QuantizedAbsorbGemmBase,
)
from chitu.ops.quant import linear
from chitu.hybrid_device import CPUParameter
from chitu.cpuinfer_singleton import get_cpu_infer
from chitu.quantization.registry import QuantizationRegistry
from chitu.global_vars import get_global_args
from chitu.utils import (
    try_import_platform_dep,
    try_import_opt_dep,
    try_import_and_setup_torch_npu,
)
from chitu.static_tensor import StaticTensor
from chitu.native_layout import (
    NativeLayoutMixin,
    PermutedTensor,
    NpuFractalNzTensor,
    NpuFractalZnTensor,
)
from chitu.native_layout.npu import ACL_FORMAT_FRACTAL_NZ
from chitu.custom_gguf import GGMLQuantizationType, get_ggml_quant_type
from chitu.moe.batched_expert_result import BatchedExpertResult
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    ConcatPermutedBatchedRoutedActivationMinimal,
    PerExpertDenseBatchedRoutedActivationMinimal,
)
from chitu.ops.utils import make_op_dispatcher

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")

if has_triton:
    from chitu.moe.experts.triton_batched_experts import triton_batched_experts
    from chitu.moe.experts import fused_experts

if has_torch_npu:
    from chitu.moe.experts import fused_experts_no_sum_npu, fused_experts_npu_for_ep


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

        super().__init__(in_features, out_features, has_bias)

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

    def forward(
        self, x: torch.Tensor, out_dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        return linear(x, self.weight, self.bias, out_dtype=out_dtype)


class NormalLinearNpuFractalNz(NativeLayoutMixin, NormalLinear):
    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(self.weight, NpuFractalNzTensor)

    @override
    def forward(
        self, x: torch.Tensor, out_dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        assert torch_npu.get_npu_format(self.weight) == ACL_FORMAT_FRACTAL_NZ
        return super().forward(x, out_dtype=out_dtype)


class NormalLinearNpuFractalZn(NativeLayoutMixin, NormalLinear):
    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(self.weight, NpuFractalZnTensor)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "NormalLinearNpuFractalZn is not designed to be run directly. It is supposed to form "
            "a fused operator. Please use NormalLinearNpuFractalNz if you want a stand-alone layer."
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
def fused_experts_no_sum_normal_indexed(
    hidden_states,
    w1,
    w2,
    *,
    activation: str = "silu",
    swiglu_limit: Optional[float] = None,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
    impl: str = "auto",
): ...


@fused_experts_no_sum_normal_indexed.register_auto
def _auto_fused_experts_no_sum_normal_indexed():
    if has_triton:
        return "triton"
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


fused_experts_no_sum_normal_indexed.register_candidate("triton")
if has_triton:
    fused_experts_no_sum_normal_indexed.register("triton")(fused_experts)


fused_experts_no_sum_normal_indexed.register_candidate("torch_npu")
if has_torch_npu:
    fused_experts_no_sum_normal_indexed.register("torch_npu")(fused_experts_no_sum_npu)


@make_op_dispatcher
def fused_experts_sum_normal_indexed(
    hidden_states,
    w1,
    w2,
    topk_weights: Optional[torch.Tensor],
    *,
    inplace: bool = False,
    impl: str = "auto",
    activation: str = "silu",
    swiglu_limit: Optional[float] = None,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
): ...


@fused_experts_sum_normal_indexed.register_auto
def _auto_fused_experts_sum_normal_indexed():
    if has_triton:
        return "triton"
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


@fused_experts_sum_normal_indexed.register("triton", available=has_triton)
@fused_experts_sum_normal_indexed.register("torch_npu", available=has_torch_npu)
def _fused_experts_sum_normal_indexed_any(
    hidden_states,
    w1,
    w2,
    topk_weights: Optional[torch.Tensor],
    *,
    inplace: bool = False,
    activation: str = "silu",
    swiglu_limit: Optional[float] = None,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
    impl: str,
):
    output = fused_experts_no_sum_normal_indexed(
        hidden_states,
        w1,
        w2,
        impl=impl,
        activation=activation,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        global_num_experts=global_num_experts,
        experts_start_idx=experts_start_idx,
    )
    return _finalize_fused_experts_sum_output(
        output, hidden_states, topk_weights=topk_weights, inplace=inplace
    )


@make_op_dispatcher
def fused_experts_no_sum_normal_per_expert_dense(
    hidden_states, w1, w2, *, swiglu_limit: Optional[float] = None, impl: str = "auto"
): ...


@fused_experts_no_sum_normal_per_expert_dense.register_auto
def _auto_fused_experts_no_sum_normal_per_expert_dense():
    if has_triton:
        return "triton"
    raise NotImplementedError


@fused_experts_no_sum_normal_per_expert_dense.register("triton", available=has_triton)
def _fused_experts_no_sum_normal_per_expert_dense_triton(
    hidden_states, w1, w2, *, swiglu_limit: Optional[float] = None, impl: str = "triton"
):
    return triton_batched_experts(
        hidden_states, w1=w1, w2=w2, swiglu_limit=swiglu_limit
    )


@make_op_dispatcher
def fused_experts_sum_normal_per_expert_dense(
    hidden_states,
    w1,
    w2,
    topk_weights: Optional[torch.Tensor],
    *,
    inplace: bool = False,
    swiglu_limit: Optional[float] = None,
    impl: str = "auto",
): ...


@fused_experts_sum_normal_per_expert_dense.register_auto
def _auto_fused_experts_sum_normal_per_expert_dense():
    if has_triton:
        return "triton"
    raise NotImplementedError


@fused_experts_sum_normal_per_expert_dense.register("triton", available=has_triton)
def _fused_experts_sum_normal_per_expert_dense_triton(
    hidden_states,
    w1,
    w2,
    topk_weights: Optional[torch.Tensor],
    *,
    inplace: bool = False,
    swiglu_limit: Optional[float] = None,
    impl: str,
):
    output = fused_experts_no_sum_normal_per_expert_dense(
        hidden_states, w1, w2, swiglu_limit=swiglu_limit, impl=impl
    )
    return _finalize_fused_experts_sum_output(
        output, hidden_states, topk_weights=topk_weights, inplace=inplace
    )


@make_op_dispatcher
def fused_experts_no_sum_normal_concat_permuted(
    hidden_states,
    w1,
    w2,
    *,
    swiglu_limit: Optional[float] = None,
    impl: str = "auto",
    experts_start_idx: int = 0,
): ...


@fused_experts_no_sum_normal_concat_permuted.register_auto
def _auto_fused_experts_no_sum_normal_concat_permuted():
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


fused_experts_no_sum_normal_concat_permuted.register_candidate("torch_npu")
if has_torch_npu:
    fused_experts_no_sum_normal_concat_permuted.register("torch_npu")(
        fused_experts_npu_for_ep
    )


@make_op_dispatcher
def fused_experts_sum_normal_concat_permuted(
    hidden_states,
    w1,
    w2,
    topk_weights: Optional[torch.Tensor],
    *,
    inplace: bool = False,
    swiglu_limit: Optional[float] = None,
    impl: str = "auto",
    experts_start_idx: int = 0,
): ...


@fused_experts_sum_normal_concat_permuted.register_auto
def _auto_fused_experts_sum_normal_concat_permuted():
    if has_torch_npu:
        return "torch_npu"
    raise NotImplementedError


@fused_experts_sum_normal_concat_permuted.register("torch_npu", available=has_torch_npu)
def _fused_experts_sum_normal_concat_permuted_torch_npu(
    hidden_states,
    w1,
    w2,
    topk_weights: Optional[torch.Tensor],
    *,
    inplace: bool = False,
    swiglu_limit: Optional[float] = None,
    experts_start_idx: int = 0,
    impl: str,
):
    output = fused_experts_no_sum_normal_concat_permuted(
        hidden_states,
        w1,
        w2,
        swiglu_limit=swiglu_limit,
        impl=impl,
        experts_start_idx=experts_start_idx,
    )
    return _finalize_fused_experts_sum_output(
        output, hidden_states, topk_weights=topk_weights, inplace=inplace
    )


@QuantizationRegistry.register_moe_experts(None, merge_gate_up=False)
class NormalMoeExpertsUnmerged(QuantizedMoeExpertsUnmerged):
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
        checkpoint_prefix: str | CheckpointPrefix,
        *,
        ############################################
        # Parameters specific to this quantization
        dtype: Optional[torch.dtype] = None,
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
        self.down_proj_weight = torch.nn.Parameter(
            torch.empty(
                (self.group_size, self.dim, moe_inter_dim),
                dtype=dtype,
            ),
            requires_grad=False,
        )

    @override
    def forward_ith_expert_gate(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert x_scale is None
        return linear(x, self.gate_proj_weight[i], bias=None)

    @override
    def forward_ith_expert_up(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert x_scale is None
        return linear(x, self.up_proj_weight[i], bias=None)

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.down_proj_weight[i], bias=None)


@QuantizationRegistry.register_moe_experts(None, merge_gate_up=True)
class NormalMoeExpertsMerged(QuantizedMoeExpertsMerged):
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
        checkpoint_prefix: str | CheckpointPrefix,
        *,
        ############################################
        # Parameters specific to this quantization
        dtype: Optional[torch.dtype] = None,
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

    @override
    @functools.singledispatchmethod
    def forward_no_sum(
        self, routed_x: BatchedRoutedActivation, impl="auto"
    ) -> BatchedExpertResult:
        return super().forward_no_sum(routed_x, impl=impl)

    @forward_no_sum.register
    def _(
        self, routed_x: IndexedBatchedRoutedActivation, impl="auto"
    ) -> BatchedExpertResult:
        if has_triton or has_torch_npu:
            return fused_experts_no_sum_normal_indexed(
                hidden_states=routed_x,
                w1=self.gate_up_proj_weight,
                w2=self.down_proj_weight,
                impl=impl,
                swiglu_limit=self.swiglu_limit,
                global_num_experts=self.global_n_experts,
                experts_start_idx=self.experts_start_idx,
            )
        return super().forward_no_sum(routed_x, impl=impl)

    @forward_no_sum.register
    def _(
        self, routed_x: PerExpertDenseBatchedRoutedActivationMinimal, impl="auto"
    ) -> BatchedExpertResult:
        if has_triton:
            return fused_experts_no_sum_normal_per_expert_dense(
                hidden_states=routed_x,
                w1=self.gate_up_proj_weight,
                w2=self.down_proj_weight,
                impl=impl,
                swiglu_limit=self.swiglu_limit,
            )
        return super().forward_no_sum(routed_x, impl=impl)

    @forward_no_sum.register
    def _(
        self, routed_x: ConcatPermutedBatchedRoutedActivationMinimal, impl="auto"
    ) -> BatchedExpertResult:
        if has_torch_npu:
            impl = fused_experts_no_sum_normal_concat_permuted.resolve_impl(impl=impl)
            return fused_experts_no_sum_normal_concat_permuted(
                hidden_states=routed_x,
                w1=self.gate_up_proj_weight,
                w2=self.down_proj_weight,
                impl=impl,
                swiglu_limit=self.swiglu_limit,
                experts_start_idx=self.experts_start_idx,
            )
        return super().forward_no_sum(routed_x, impl=impl)

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
        if has_triton or has_torch_npu:
            return fused_experts_sum_normal_indexed(
                hidden_states=routed_x,
                topk_weights=weights,
                w1=self.gate_up_proj_weight,
                w2=self.down_proj_weight,
                impl=impl,
                inplace=inplace,
                swiglu_limit=self.swiglu_limit,
                global_num_experts=self.global_n_experts,
                experts_start_idx=self.experts_start_idx,
            )
        return super().forward(routed_x, weights, inplace=inplace, impl=impl)

    @forward.register
    def _(
        self,
        routed_x: PerExpertDenseBatchedRoutedActivationMinimal,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        if has_triton:
            return fused_experts_sum_normal_per_expert_dense(
                hidden_states=routed_x,
                topk_weights=weights,
                w1=self.gate_up_proj_weight,
                w2=self.down_proj_weight,
                impl=impl,
                inplace=inplace,
                swiglu_limit=self.swiglu_limit,
            )
        return super().forward(routed_x, weights, inplace=inplace, impl=impl)

    @forward.register
    def _(
        self,
        routed_x: ConcatPermutedBatchedRoutedActivationMinimal,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        if has_torch_npu:
            return fused_experts_sum_normal_concat_permuted(
                hidden_states=routed_x,
                topk_weights=weights,
                w1=self.gate_up_proj_weight,
                w2=self.down_proj_weight,
                impl=impl,
                inplace=inplace,
                swiglu_limit=self.swiglu_limit,
                experts_start_idx=self.experts_start_idx,
            )
        return super().forward(routed_x, weights, inplace=inplace, impl=impl)

    @override
    def forward_ith_expert_gate_up(
        self, i: int, x: torch.Tensor, x_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert x_scale is None
        return linear(x, self.gate_up_proj_weight[i], bias=None)

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.down_proj_weight[i], bias=None)


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
        super().__init__(n_heads, in_features_per_head, out_features_per_head)

        self.weight = torch.nn.Parameter(
            torch.empty(
                n_heads, out_features_per_head, in_features_per_head, dtype=dtype
            ),
            requires_grad=False,
        )

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

    def forward_token_major(self, x: torch.Tensor) -> torch.Tensor:
        # FlashMLA split-Q consumes q_nope directly, so write it as a contiguous
        # token-major [seq, head, dim] tensor without materializing torch.cat.
        seq, n_head, _ = x.shape
        y = x.new_empty((seq, n_head, self.out_features_per_head))
        torch.bmm(
            x.transpose(0, 1),
            self.weight.transpose(1, 2),
            out=y.transpose(0, 1),
        )
        return y


class NormalAbsorbGemmPermuted021(NativeLayoutMixin, NormalAbsorbGemm):
    def init_native_layout(self):
        super().init_native_layout()
        self.apply_native_layout(self.weight, PermutedTensor, perm=(0, 2, 1))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            seq, n_head, n_hidden = x.shape
            bs = None
        else:
            bs, seq, n_head, n_hidden = x.shape
            x = x.view(bs * seq, n_head, n_hidden)

        y = torch.einsum("shc,hcd->shd", x, self.weight)

        if bs is not None:
            y = y.view(bs, seq, y.shape[-2], y.shape[-1])
        return y


@QuantizationRegistry.register_linear(None, backend_type="cpuinfer")
class NormLinearCPUInfer(QuantizedLinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        *,
        dtype=None,
        bias_dtype=None,
    ):
        super().__init__(in_features, out_features, has_bias)
        self.stride = 64
        self.group_max_len = 1024
        if torch.distributed.get_rank() == 0:
            self.weight = CPUParameter(
                torch.empty(
                    self.out_features, self.in_features, dtype=dtype, device="cpu"
                ),
                requires_grad=False,
            )
            max_batch_size = get_global_args().infer.max_batch_size
            self.input_cpu = StaticTensor(
                max_nelem=max_batch_size * self.in_features,
                device="cpu",
                pin_memory=True,
                dtype=torch.get_default_dtype(),
            )
            self.output_cpu = StaticTensor(
                max_nelem=max_batch_size * self.out_features,
                device="cpu",
                pin_memory=True,
                dtype=torch.get_default_dtype(),
            )

            self.cpu_infer = get_cpu_infer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.distributed.get_rank() == 0:
            # Initialize after __init__ because `data_ptr` may be modified during weight loading
            if not hasattr(self, "linear"):
                linear_config = cpuinfer.linear.LinearConfig(
                    self.in_features,
                    self.out_features,
                    self.stride,
                    self.group_max_len,
                    self.weight.data_ptr(),
                    get_ggml_quant_type(self.weight),
                    get_ggml_quant_type(x),
                )
                self.linear = cpuinfer.linear.Linear(linear_config)

            out_shape = list(x.shape)
            out_shape[-1] = self.out_features

            if x.device.type == "cpu" or not torch.cuda.is_current_stream_capturing():
                inp = x.contiguous().cpu()
                out = torch.empty(
                    out_shape, device="cpu", dtype=torch.get_default_dtype()
                )

                self.cpu_infer.submit(
                    self.linear.forward(x.size(0), inp.data_ptr(), out.data_ptr())
                )
                self.cpu_infer.sync()

                y = out.to(x.device, non_blocking=True)

            else:
                self.input_cpu.set_shape(x.shape)
                self.input_cpu.get().copy_(x, non_blocking=True)
                self.output_cpu.set_shape(out_shape)

                self.cpu_infer.submit(
                    self.linear.forward(
                        x.size(0),
                        self.input_cpu.get().data_ptr(),
                        self.output_cpu.get().data_ptr(),
                    )
                )
                self.cpu_infer.sync()

                y = self.output_cpu.get().to(x.device, non_blocking=True)

        else:
            y = torch.zeros_like(x)
        return y


@QuantizationRegistry.register_moe_experts(
    None, backend_type="cpuinfer", merge_gate_up=False
)
class NormalMoeExpertsCPUInfer(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        moe_inter_dim: int,
        global_n_experts: int,
        experts_start_idx: int,
        experts_end_idx: int,
        n_activated_experts: int,
        checkpoint_prefix: str | CheckpointPrefix,
    ):
        super().__init__()

        from chitu.tensor_parallel import get_tp_size

        self.moe_inter_dim = moe_inter_dim * get_tp_size()
        self.dim = dim
        self.max_batch_size = get_global_args().infer.max_batch_size
        self.group_size = self.experts_end_idx - self.experts_start_idx
        self.n_activated_experts = n_activated_experts
        self.checkpoint_prefix = checkpoint_prefix

        if torch.distributed.get_rank() == 0:
            self.gate_proj_weight = CPUParameter(
                torch.empty(
                    (self.group_size, self.moe_inter_dim, self.dim),
                    dtype=torch.bfloat16,
                    device="cpu",
                ),
                requires_grad=False,
            )
            with torch.device("cpu"):
                # The value matters. Don't put onto "meta" device.
                self.gate_type = torch.tensor(
                    (GGMLQuantizationType.BF16),
                    dtype=torch.int,
                    device="cpu",
                    requires_grad=False,
                )
            self.up_proj_weight = CPUParameter(
                torch.empty(
                    (self.group_size, self.moe_inter_dim, self.dim),
                    dtype=torch.bfloat16,
                    device="cpu",
                ),
                requires_grad=False,
            )
            with torch.device("cpu"):
                # The value matters. Don't put onto "meta" device.
                self.up_type = torch.tensor(
                    (GGMLQuantizationType.BF16),
                    dtype=torch.int,
                    device="cpu",
                    requires_grad=False,
                )
            self.down_proj_weight = CPUParameter(
                torch.empty(
                    (self.group_size, self.dim, self.moe_inter_dim),
                    dtype=torch.bfloat16,
                    device="cpu",
                ),
                requires_grad=False,
            )
            with torch.device("cpu"):
                # The value matters. Don't put onto "meta" device.
                self.down_type = torch.tensor(
                    (GGMLQuantizationType.BF16),
                    dtype=torch.int,
                    device="cpu",
                    requires_grad=False,
                )
            self.input_tensor_cpu = StaticTensor(
                max_nelem=self.max_batch_size * self.dim,
                device="cpu",
                pin_memory=True,
                dtype=torch.bfloat16,
            )
            self.weights_cpu = StaticTensor(
                max_nelem=self.max_batch_size * self.n_activated_experts,
                device="cpu",
                pin_memory=True,
                dtype=torch.float32,
            )
            self.indices_cpu = StaticTensor(
                max_nelem=self.max_batch_size * self.n_activated_experts,
                device="cpu",
                pin_memory=True,
                dtype=torch.int64,
            )
            self.output_cpu = StaticTensor(
                max_nelem=self.max_batch_size * self.dim,
                device="cpu",
                pin_memory=True,
                dtype=torch.bfloat16,
            )
            self.output_gpu = StaticTensor(
                max_nelem=self.max_batch_size * self.dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            self.cpu_infer = get_cpu_infer()

    def warm_up(self):
        if torch.distributed.get_rank() == 0:
            # Initialize after __init__ because `data_ptr` may be modified during weight loading
            gate_ptr = ctypes.addressof(
                ctypes.cast(
                    self.gate_proj_weight.data_ptr(), ctypes.POINTER(ctypes.c_uint64)
                ).contents
            )
            up_ptr = ctypes.addressof(
                ctypes.cast(
                    self.up_proj_weight.data_ptr(), ctypes.POINTER(ctypes.c_uint64)
                ).contents
            )
            down_ptr = ctypes.addressof(
                ctypes.cast(
                    self.down_proj_weight.data_ptr(), ctypes.POINTER(ctypes.c_uint64)
                ).contents
            )
            moe_config = cpuinfer.moe.MOEConfig(
                self.group_size,
                self.n_activated_experts,
                self.dim,
                self.moe_inter_dim,
                64,
                10,
                1024,
                gate_ptr,
                up_ptr,
                down_ptr,
                self.gate_type.item(),
                self.up_type.item(),
                self.down_type.item(),
                GGMLQuantizationType.BF16,
            )
            self.moe = cpuinfer.moe.MOE(moe_config)

            self.cpu_infer.submit(self.moe.warm_up())
            self.cpu_infer.sync()

    @override
    @functools.singledispatchmethod
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
            x (torch.Tensor): Input BatchedRoutedActivation.
            weights (torch.Tensor): Routing weights from the gate.

        Returns:
            torch.Tensor: Output tensor.
        """

        raise NotImplementedError(
            f"{type(routed_x)} not supported for NormalMoeExpertsCPUInfer.forward"
        )

    @forward.register
    def _(
        self,
        routed_x: IndexedBatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        x, indices = routed_x.activation, routed_x.token_to_expert_indices

        shape = x.size()
        capturing = torch.cuda.is_current_stream_capturing()

        if torch.distributed.get_rank() == 0:
            indices = indices.contiguous().to(torch.int64)
            weights = weights.contiguous().to(torch.float32)
            if not capturing:
                input_tensor = x.contiguous().cpu()
                indices = indices.cpu()
                weights = weights.cpu()
                output = torch.empty_like(input_tensor).contiguous().pin_memory()
                self.cpu_infer.submit(
                    self.moe.forward(
                        indices.size(0),
                        indices.size(1),
                        indices.data_ptr(),
                        weights.data_ptr(),
                        input_tensor.data_ptr(),
                        output.data_ptr(),
                    )
                )
            else:
                self.input_tensor_cpu.set_shape(x.shape)
                self.indices_cpu.set_shape(indices.shape)
                self.weights_cpu.set_shape(weights.shape)
                self.output_cpu.set_shape(x.shape)
                self.output_gpu.set_shape(x.shape)
                self.input_tensor_cpu.get().copy_(x, non_blocking=True)
                self.indices_cpu.get().copy_(indices, non_blocking=True)
                self.weights_cpu.get().copy_(weights, non_blocking=True)
                self.cpu_infer.submit_with_cuda_stream(
                    torch.cuda.current_stream().cuda_stream,
                    self.moe.forward(
                        indices.size(0),
                        indices.size(1),
                        self.indices_cpu.get().data_ptr(),
                        self.weights_cpu.get().data_ptr(),
                        self.input_tensor_cpu.get().data_ptr(),
                        self.output_cpu.get().data_ptr(),
                    ),
                )

        if torch.distributed.get_rank() == 0:
            if not capturing:
                self.cpu_infer.sync()
                y = output.to(x.device, non_blocking=True).view(shape)
            else:
                self.cpu_infer.sync_with_cuda_stream(
                    torch.cuda.current_stream().cuda_stream
                )
                self.output_gpu.get().copy_(self.output_cpu.get(), non_blocking=True)
                y = self.output_gpu.get()
        else:
            y = torch.zeros_like(x)

        return y.view(shape)
