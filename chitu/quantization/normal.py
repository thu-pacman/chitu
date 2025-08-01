from typing import Optional
from typing_extensions import override

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
from chitu.hybrid_device import CPUParameter
from chitu.quantization.cpuinfer_singleton import get_cpu_infer
from chitu.quantization.registry import QuantizationRegistry
from chitu.global_vars import get_global_args
from chitu.utils import try_import_opt_dep
from chitu.distributed.parallel_state import get_ep_group
from chitu.static_tensor import StaticTensor
from chitu.custom_gguf import GGMLQuantizationType

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
        fuse_shared_experts: bool,
        checkpoint_prefix: str,
        merge_gate_up: bool,
        *,
        ############################################
        # Parameters specific to this quantization
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.ep_group = get_ep_group()
        moe_rank = self.ep_group.rank_in_group
        moe_world_size = self.ep_group.group_size
        self.dim = dim
        self.fuse_shared_experts = fuse_shared_experts
        assert (
            n_routed_experts % moe_world_size == 0
        ), f"Number of experts must be divisible by moe world size (world_size={moe_world_size})"
        self.n_shared_experts = n_shared_experts
        self.n_fused_shared_experts = (
            n_shared_experts if self.fuse_shared_experts else 0
        )

        self.n_routed_experts = n_routed_experts
        self.n_local_experts = n_routed_experts // moe_world_size
        remainder = n_routed_experts % moe_world_size
        self.experts_start_idx = moe_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        if self.ep_group.is_last_rank:
            self.experts_end_idx += remainder
        if moe_world_size > 1:
            expert_map = [-1] * self.n_routed_experts
            expert_map[self.experts_start_idx : self.experts_end_idx] = list(
                range(self.n_local_experts)
            )
            self.expert_map = torch.tensor(expert_map, dtype=torch.int32, device="cuda")
        else:
            self.expert_map = None

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

        if has_torch_npu and self.merge_gate_up:
            y = fused_experts_npu(
                hidden_states=x,
                w1=self.gate_up_proj_weight,
                w2=self.down_proj_weight,
                topk_weights=weights,
                topk_ids=indices,
            )

        elif has_triton and self.merge_gate_up:
            if not self.fuse_shared_experts:
                y = fused_experts(
                    x,
                    self.gate_up_proj_weight,
                    self.down_proj_weight,
                    topk_weights=weights,
                    topk_ids=indices,
                    inplace=True,
                    global_num_experts=self.n_routed_experts,
                    expert_map=self.expert_map,
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
                    expert_map=self.expert_map,
                    block_shape=[128, 128],
                )

        else:
            y = self.forward_iterative(x, weights, indices)

        return y.view(shape)

    @override
    def forward_ith_expert_gate_up(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.gate_up_proj_weight[i], bias=None)

    @override
    def forward_ith_expert_gate(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.gate_proj_weight[i], bias=None)

    @override
    def forward_ith_expert_up(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.up_proj_weight[i], bias=None)

    @override
    def forward_ith_expert_down(self, i: int, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.down_proj_weight[i], bias=None)


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


@QuantizationRegistry.register_linear(None, backend_type="cpuinfer")
class NormLinearCPUInfer(QuantizedLinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = False,
        **args,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.stride = 64
        self.group_max_len = 1024
        if torch.distributed.get_rank() == 0:
            self.weight = CPUParameter(
                torch.empty(
                    self.out_features,
                    self.in_features,
                    dtype=torch.bfloat16,
                    device="cpu",
                ),
                requires_grad=False,
            )

            import cpuinfer

            linear_config = cpuinfer.linear.LinearConfig(
                self.in_features,
                self.out_features,
                self.stride,
                self.group_max_len,
                self.weight.data_ptr(),
                GGMLQuantizationType.BF16,
                GGMLQuantizationType.BF16,
            )
            self.linear = cpuinfer.linear.Linear(linear_config)

            max_reqs = 256
            self.input_cpu = StaticTensor(
                max_nelem=max_reqs * self.in_features,
                device="cpu",
                pin_memory=True,
                dtype=torch.bfloat16,
            )
            self.output_cpu = StaticTensor(
                max_nelem=max_reqs * self.out_features,
                device="cpu",
                pin_memory=True,
                dtype=torch.bfloat16,
            )

            self.cpu_infer = get_cpu_infer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.distributed.get_rank() == 0:
            self.input_cpu.set_shape(x.shape)
            out_shape = list(x.shape)
            out_shape[-1] = self.out_features
            self.output_cpu.set_shape(out_shape)

            if x.device.type == "cpu":
                inp = x.contiguous().cpu()
                inp_ptr = inp.data_ptr()
            else:
                self.input_cpu.set_shape(x.shape)
                self.input_cpu.get().copy_(x, non_blocking=True)
                inp_ptr = self.input_cpu.get().data_ptr()

            self.cpu_infer.submit(
                self.linear.forward(
                    x.size(0),
                    inp_ptr,
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
            self.down_type = torch.tensor(
                (GGMLQuantizationType.BF16),
                dtype=torch.int,
                device="cpu",
                requires_grad=False,
            )
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
            import cpuinfer

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
            self.cpu_infer.submit(self.moe.warm_up())
            self.cpu_infer.sync()

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

        if torch.distributed.get_rank() == 0:
            indices = indices.contiguous().to(torch.int64)
            weights = weights.contiguous().to(torch.float32)
            if x.shape[1] > 1:
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
            if x.shape[1] > 1:
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
