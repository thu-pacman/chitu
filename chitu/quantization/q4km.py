import torch
import torch.nn.functional as F

from chitu.quantization.registry import (
    QuantizationRegistry,
)
from chitu.global_vars import get_global_args
from chitu.static_tensor import StaticTensor
from chitu.hybrid_device import CPUParameter
import ctypes


@QuantizationRegistry.register_moe_experts("q4km")
class MoeExpertsDeepSeekV3CPU(torch.nn.Module):
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
        dim: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_shared_experts: int,
        n_activated_experts: int,
        moe_world_size: int,
        moe_rank: int,
        dtype: torch.dtype,
        op_impl: str,
        fuse_shared_experts: bool,
        checkpoint_prefix: str,
        merge_gate_up: bool,
        ggml_type: str,
    ):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.merge_gate_up = merge_gate_up
        self.moe_inter_dim = moe_inter_dim
        self.dim = dim
        self.rank = moe_rank

        moe_world_size = 1
        self.max_batch_size = get_global_args().infer.max_reqs
        assert (
            n_routed_experts % moe_world_size == 0
        ), f"Number of experts must be divisible by world size (world_size={moe_world_size})"
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.n_local_experts = n_routed_experts // moe_world_size
        self.n_activated_experts = n_activated_experts

        if self.rank == 0:

            self.gguf_gate_proj = CPUParameter(
                torch.empty(
                    int(256 * 2048 * 7168 / 256 * 144),
                    dtype=torch.uint8,
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
            if ggml_type == "q4k":
                self.gguf_down_proj = CPUParameter(
                    torch.empty(
                        int(256 * 2048 * 7168 / 256 * 144),
                        dtype=torch.uint8,
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
            else:
                raise ValueError("ggml quantization type unimplemented !")

            self.gate_type = CPUParameter(
                torch.empty(
                    (),
                    dtype=torch.int,
                    device="cpu",
                ),
                requires_grad=False,
            )
            self.up_type = CPUParameter(
                torch.empty(
                    (),
                    dtype=torch.int,
                    device="cpu",
                ),
                requires_grad=False,
            )
            self.down_type = CPUParameter(
                torch.empty(
                    (),
                    dtype=torch.int,
                    device="cpu",
                ),
                requires_grad=False,
            )

        self.stride = 64
        self.moe = None

        if MoeExpertsDeepSeekV3CPU.cpu_infer is None:
            import cpuinfer

            MoeExpertsDeepSeekV3CPU.cpu_infer = cpuinfer.CPUInfer(
                get_global_args().infer.bind_thread_to_cpu
            )

    def init_weights(self):
        if self.rank == 0:
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

            import cpuinfer

            moe_config = cpuinfer.moe.MOEConfig(
                self.n_routed_experts,
                self.n_activated_experts,
                self.dim,
                self.moe_inter_dim,
                self.stride,
                10,
                1024,
                gate_ptr,
                up_ptr,
                down_ptr,
                self.gate_type.item(),
                self.up_type.item(),
                self.down_type.item(),
                30,
            )

            self.moe = cpuinfer.moe.MOE(moe_config)

            # warm up
            MoeExpertsDeepSeekV3CPU.cpu_infer.submit(self.moe.warm_up())
            MoeExpertsDeepSeekV3CPU.cpu_infer.sync()

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

        if self.rank == 0:
            indices = indices.contiguous().to(torch.int64)
            weights = weights.contiguous().to(torch.float32)
            if x.shape[1] > 1:
                input_tensor = x.contiguous().cpu()
                indices = indices.cpu()
                weights = weights.cpu()
                output = torch.empty_like(input_tensor).contiguous().pin_memory()
                MoeExpertsDeepSeekV3CPU.cpu_infer.submit(
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
                MoeExpertsDeepSeekV3CPU.cpu_infer.submit_with_cuda_stream(
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
