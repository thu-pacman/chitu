# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import torch
import ctypes
import functools

from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.base import QuantizedMoeExpertsBase
from chitu.global_vars import get_global_args
from chitu.static_tensor import StaticTensor
from chitu.hybrid_device import CPUParameter
from chitu.cpuinfer_singleton import get_cpu_infer
from chitu.custom_gguf import GGMLQuantizationType
from chitu.utils import try_import_opt_dep
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
)

cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")


@QuantizationRegistry.register_moe_experts("q4km", backend_type="cpuinfer")
class MoeExpertsDeepSeekV3CPUInfer(QuantizedMoeExpertsBase):
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

    cpu_infer = None

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
        ggml_type: str,
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
            layer_id=layer_id,
        )

        self.rank = self.ep_group.rank_in_group
        self.max_batch_size = get_global_args().infer.max_reqs
        self.n_local_experts = n_routed_experts

        if self.rank == 0:

            self.gguf_gate_proj = CPUParameter(
                torch.empty(
                    int(256 * 2048 * 7168 / 256 * 144),
                    dtype=torch.uint8,
                    device="cpu",
                ),
                requires_grad=False,
            )
            with torch.device("cpu"):
                # The value matters. Don't put onto "meta" device.
                self.gate_type = CPUParameter(
                    torch.tensor(
                        (12),
                        dtype=torch.int,
                        device="cpu",
                    ),
                    requires_grad=False,
                )
            self.gguf_up_proj = CPUParameter(
                torch.empty(
                    int(256 * 2048 * 7168 / 256 * 144),
                    dtype=torch.uint8,
                    device="cpu",
                ),
                requires_grad=False,
            )
            with torch.device("cpu"):
                # The value matters. Don't put onto "meta" device.
                self.up_type = CPUParameter(
                    torch.tensor(
                        (12),
                        dtype=torch.int,
                        device="cpu",
                    ),
                    requires_grad=False,
                )
            if ggml_type == "q4k":
                self.gguf_down_proj = CPUParameter(
                    torch.empty(
                        int(256 * 2048 * 7168 / 256 * 144),
                        dtype=torch.uint8,
                        device="cpu",
                    ),
                    requires_grad=False,
                )
                with torch.device("cpu"):
                    # The value matters. Don't put onto "meta" device.
                    self.down_type = CPUParameter(
                        torch.tensor(
                            (12),
                            dtype=torch.int,
                            device="cpu",
                        ),
                        requires_grad=False,
                    )
            elif ggml_type == "q6k":
                self.gguf_down_proj = CPUParameter(
                    torch.empty(
                        int(256 * 2048 * 7168 / 256 * 210),
                        dtype=torch.uint8,
                        device="cpu",
                    ),
                    requires_grad=False,
                )
                with torch.device("cpu"):
                    # The value matters. Don't put onto "meta" device.
                    self.down_type = CPUParameter(
                        torch.tensor(
                            (14),
                            dtype=torch.int,
                            device="cpu",
                        ),
                        requires_grad=False,
                    )
            else:
                raise ValueError("ggml quantization type unimplemented !")

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
        if self.rank == 0:
            # Initialize after __init__ because `data_ptr` may be modified during weight loading
            gate_ptr = ctypes.addressof(
                ctypes.cast(
                    self.gguf_gate_proj.data_ptr(), ctypes.POINTER(ctypes.c_uint64)
                ).contents
            )
            up_ptr = ctypes.addressof(
                ctypes.cast(
                    self.gguf_up_proj.data_ptr(), ctypes.POINTER(ctypes.c_uint64)
                ).contents
            )
            down_ptr = ctypes.addressof(
                ctypes.cast(
                    self.gguf_down_proj.data_ptr(), ctypes.POINTER(ctypes.c_uint64)
                ).contents
            )
            self.moe_config = cpuinfer.moe.MOEConfig(
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
            self.moe = cpuinfer.moe.MOE(self.moe_config)

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
        raise NotImplementedError(
            f"{type(routed_x)} not supported for MoeExpertsDeepSeekV3CPUInfer.forward"
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

        if self.rank == 0:
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

        if self.rank == 0:
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
