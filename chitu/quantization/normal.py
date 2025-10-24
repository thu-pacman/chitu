# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import functools
import torch
import ctypes

from chitu.tensor_parallel import (
    get_tp_size,
)
from chitu.quantization.base import (
    QuantizedLinearBase,
    QuantizedMoeExpertsBase,
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
from chitu.distributed.parallel_state import get_ep_group
from chitu.static_tensor import StaticTensor
from chitu.native_layout import (
    enable_native_layout_weight,
    PermutedTensor,
    NpuFractalNzTensor,
    NpuFractalZnTensor,
    ACL_FORMAT_FRACTAL_NZ,
)
from chitu.custom_gguf import GGMLQuantizationType, get_ggml_quant_type
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
)

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")
if has_triton or has_torch_npu:
    from chitu.moe.experts import fused_experts


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
        return linear(x, self.weight, self.bias)


class NormalLinearNpuFractalNz(
    enable_native_layout_weight("weight", NpuFractalNzTensor), NormalLinear
):
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert torch_npu.get_npu_format(self.weight) == ACL_FORMAT_FRACTAL_NZ
        return super().forward(x)


class NormalLinearNpuFractalZn(
    enable_native_layout_weight("weight", NpuFractalZnTensor), NormalLinear
):
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "NormalLinearNpuFractalZn is not designed to be run directly. It is supposed to form "
            "a fused operator. Please use NormalLinearNpuFractalNz if you want a stand-alone layer."
        )


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
        fuse_shared_experts: bool,
        checkpoint_prefix: str,
        merge_gate_up: bool,
        layer_id: int,
        *,
        ############################################
        # Parameters specific to this quantization
        dtype: Optional[torch.dtype] = None,
    ):
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

    @override
    def forward(
        self,
        routed_x: BatchedRoutedActivation,
        weights: torch.Tensor,
        inplace: bool = False,
        impl: str = "auto",
    ) -> torch.Tensor:
        if self.merge_gate_up and (has_triton or has_torch_npu):
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
                w1=self.gate_up_proj_weight,
                w2=self.down_proj_weight,
                topk_weights=weights,
                inplace=inplace,
                experts_start_idx=self.experts_start_idx,
                impl=impl,
            )

        else:
            return super().forward(routed_x, weights, inplace=inplace, impl=impl)

    @override
    def forward_ith_expert_gate_up(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.gate_up_proj_weight[i], bias=None)

    @override
    def forward_ith_expert_gate(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.gate_proj_weight[i], bias=None)

    @override
    def forward_ith_expert_up(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.up_proj_weight[i], bias=None)

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


class NormalAbsorbGemmPermuted021(
    enable_native_layout_weight("weight", PermutedTensor, perm=(0, 2, 1)),
    NormalAbsorbGemm,
):
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
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.stride = 64
        self.group_max_len = 1024
        if torch.distributed.get_rank() == 0:
            self.weight = CPUParameter(
                torch.empty(
                    self.out_features, self.in_features, dtype=dtype, device="cpu"
                ),
                requires_grad=False,
            )
            max_reqs = get_global_args().infer.max_reqs
            self.input_cpu = StaticTensor(
                max_nelem=max_reqs * self.in_features,
                device="cpu",
                pin_memory=True,
                dtype=torch.get_default_dtype(),
            )
            self.output_cpu = StaticTensor(
                max_nelem=max_reqs * self.out_features,
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


@QuantizationRegistry.register_moe_experts(None, backend_type="cpuinfer")
class NormalMoeExpertsCPUInfer(torch.nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
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
    ):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.merge_gate_up = merge_gate_up
        self.moe_inter_dim = moe_inter_dim * get_tp_size()
        self.ep_group = get_ep_group()
        self.dim = dim
        self.fuse_shared_experts = fuse_shared_experts
        moe_rank = self.ep_group.rank_in_group
        moe_world_size = self.ep_group.group_size
        self.max_batch_size = get_global_args().infer.max_reqs
        assert (
            moe_world_size == 1
        ), f"moe_world_size must be 1 for this configuration, but got {moe_world_size}"
        assert (
            n_routed_experts % moe_world_size == 0
        ), f"Number of experts must be divisible by world size (world_size={moe_world_size})"
        self.n_shared_experts = n_shared_experts
        self.n_fused_shared_experts = (
            n_shared_experts if self.fuse_shared_experts else 0
        )
        self.n_routed_experts = n_routed_experts
        self.n_local_experts = n_routed_experts // moe_world_size
        self.n_activated_experts = n_activated_experts
        self.experts_start_idx = moe_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.group_size = (
            self.experts_end_idx - self.experts_start_idx + self.n_fused_shared_experts
        )
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
                self.n_routed_experts,
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
