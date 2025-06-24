from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Type,
)
import functools
import re

import torch

from chitu.global_vars import get_global_args
from chitu.utils import try_import_opt_dep, parse_dtype
from chitu.ops import silu_and_mul

torch_npu, has_torch_npu = try_import_opt_dep("torch_npu", "torch_npu")
triton, has_triton = try_import_opt_dep("triton", "triton")
chitu_backend, has_chitu_backend = try_import_opt_dep("chitu_backend", "chitu_backend")
if has_torch_npu:
    from chitu.npu_utils import fused_experts_npu
if has_triton:
    from chitu.fused_moe import fused_experts


class QuantizedLinearBase(torch.nn.Module):
    """
    Base class for all quantized linear layers.

    Defines the interface that all quantized linear implementations must follow.
    """

    pass


class QuantizedMoeExpertsBase(torch.nn.Module):
    """
    MoE experts after the gate. This module runs locally on one device.

    Inherit from this class for quantization.
    """

    def __init__(
        self,
        dim: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_shared_experts: int,
        n_activated_experts: int,
        moe_world_size: int,
        moe_rank: int,
        dtype: str,
        op_impl: str,
        fuse_shared_experts: bool,
        checkpoint_prefix: str,
        merge_gate_up: bool,
        build_weight: bool = True,
    ):
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
        self.n_activated_experts = n_activated_experts
        self.experts_start_idx = moe_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.group_size = (
            self.experts_end_idx - self.experts_start_idx + self.n_fused_shared_experts
        )
        self.checkpoint_prefix = checkpoint_prefix

        self.linear_dtype = (
            torch.uint8
            if (
                parse_dtype(dtype).itemsize == 1
                and parse_dtype(
                    get_global_args().infer.raise_lower_bit_float_to
                ).itemsize
                > 1
            )
            else parse_dtype(dtype)
        )
        self.merge_gate_up = merge_gate_up
        if build_weight:
            if not self.merge_gate_up:
                self.gate_proj_weight = torch.nn.Parameter(
                    torch.empty(
                        (self.group_size, moe_inter_dim, self.dim),
                        dtype=self.linear_dtype,
                    ),
                    requires_grad=False,
                )
                self.up_proj_weight = torch.nn.Parameter(
                    torch.empty(
                        (self.group_size, moe_inter_dim, self.dim),
                        dtype=self.linear_dtype,
                    ),
                    requires_grad=False,
                )
            else:
                self.gate_up_proj_weight = torch.nn.Parameter(
                    torch.empty(
                        (self.group_size, moe_inter_dim * 2, self.dim),
                        dtype=self.linear_dtype,
                    ),
                    requires_grad=False,
                )
            self.down_proj_weight = torch.nn.Parameter(
                torch.empty(
                    (self.group_size, self.dim, moe_inter_dim),
                    dtype=self.linear_dtype,
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

        if self.op_impl == "muxi_custom_kernel":
            y = self._compute_muxi_fused_experts(x, weights, indices)
        elif has_torch_npu:  # or use op_impl ?
            y = self._compute_npu_fused_experts(x, weights, indices)
        elif has_triton:
            assert self.merge_gate_up

            if not self.fuse_shared_experts:

                y = fused_experts(
                    x,
                    self.gate_up_proj_weight,
                    self.down_proj_weight,
                    topk_weights=weights,
                    topk_ids=indices,
                    inplace=True,
                    global_num_experts=self.n_routed_experts,
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
                    block_shape=[128, 128],
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

            assert len(xs) == self.group_size
            if self.merge_gate_up:
                gate_up_proj_outs = []
                for i in range(self.group_size):
                    out = None
                    if xs[i] is not None:
                        out = torch.nn.functional.linear(
                            xs[i], self.gate_up_proj_weight[i], bias=None
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
                gate_proj_outs = []
                up_proj_outs = []
                for i in range(self.group_size):
                    gate_proj_out = None
                    up_proj_out = None
                    if xs[i] is not None:
                        gate_proj_out = torch.nn.functional.linear(
                            xs[i], self.gate_proj_weight[i], bias=None
                        )
                        up_proj_out = torch.nn.functional.linear(
                            xs[i], self.up_proj_weight[i], bias=None
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
                    down_proj_out = torch.nn.functional.linear(
                        act[i], self.down_proj_weight[i], bias=None
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

    def _compute_muxi_fused_experts(self, x, weights, indices):
        from chitu.muxi_utils import muxi_fused_experts

        if self.fuse_shared_experts:
            raise NotImplementedError(
                "Fused shared experts is not supported for muxi_layout_kernels"
            )
        if not self.merge_gate_up:
            raise NotImplementedError(
                "muxi_layout_kernels for fused MoE requires merge_gate_up=True"
            )

        return muxi_fused_experts(
            hidden_states=x,
            w1=self.gate_up_proj_weight,
            w2=self.down_proj_weight,
            topk_weights=weights,
            topk_ids=indices,
            inplace=True,
            block_shape=[128, 128],
            soft_fp8=False,
        )

    def _compute_npu_fused_experts(self, x, weights, indices):
        return fused_experts_npu(
            hidden_states=x,
            w1=self.gate_up_proj_weight,
            w2=self.down_proj_weight,
            topk_weights=weights,
            topk_ids=indices,
        )


class QuantizationRegistry:
    """
    Registry of available quantization methods and their implementations.
    """

    _linear_registry: Dict[str, Type[QuantizedLinearBase]] = {}
    _moe_experts_registry: Dict[str, Type[QuantizedMoeExpertsBase]] = {}
    _allowed_quant_for_merge_qkv_gate_up: List = [
        "blockfp8",
        "autoawq",
        "simple_w8a8",
        None,
    ]

    @classmethod
    def get_all_methods(cls) -> Set[str]:
        """
        Get all registered quantization methods.

        Returns:
            Set of quantization method names
        """
        ret = set(cls._linear_registry.keys()).union(
            set(cls._moe_experts_registry.keys())
        )
        ret.remove(None)
        return ret

    @classmethod
    def _get_quantized_class(
        cls,
        class_type: str,
        method: Optional[str],
        *,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
    ):
        if class_type == "linear":
            registry = cls._linear_registry
        elif class_type == "moe_experts":
            registry = cls._moe_experts_registry
        else:
            raise ValueError(f"Unknown class type: {class_type}")
        impl = registry.get(method)

        if impl is None:
            raise ValueError(f"Unknown quantization method in `method`: {method}")

        for key in quant_kwargs:
            if key not in registry:
                raise ValueError(
                    f"Unknown quantization method in `quant_kwargs`: {key}"
                )

        if method in quant_kwargs:

            class QuantLayerImpl(impl):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **quant_kwargs[method], **kwargs)

            impl = QuantLayerImpl

        return impl

    @classmethod
    def get_quantized_linear_class(
        cls,
        method: Optional[str],
        *,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
    ) -> Optional[Type[QuantizedLinearBase]]:
        """
        Get the quantized linear implementation for the specified method.

        Arguments:
            method: Quantization method name, or None for no quantization
            quant_kwargs: Nested mapping for additional arguments for specific
                quantization methods. E.g., `{"quant_method_x": {"arg1": value1, ...}}`
        Returns:
            The quantized linear class, or None if method is None or not found
        """

        return cls._get_quantized_class(
            "linear",
            method,
            quant_kwargs=quant_kwargs,
        )

    @classmethod
    def get_quantized_moe_experts_class(
        cls,
        method: Optional[str],
        *,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
    ) -> Optional[Type[QuantizedMoeExpertsBase]]:
        """
        Get the quantized MoeExperts implementation for the specified method.

        Arguments:
            method: Quantization method name, or None for no quantization
            quant_kwargs: Nested mapping for additional arguments for specific
                quantization methods. E.g., `{"quant_method_x": {"arg1": value1, ...}}`
        Returns:
            The quantized moe class, or None if method is None or not found
        """

        return cls._get_quantized_class(
            "moe_experts",
            method,
            quant_kwargs=quant_kwargs,
        )

    @classmethod
    def _get_quantized_class_from_global_args(
        cls,
        class_type: str,
        *,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        checkpoint_prefix="",
    ) -> Optional[Type[QuantizedLinearBase]]:
        args = get_global_args()
        quant_cfg = getattr(args.models, "quant_config", None)
        if quant_cfg is None:
            return cls._get_quantized_class(class_type, None, quant_kwargs=quant_kwargs)

        rules = getattr(quant_cfg, "rules", [])
        for rule in rules:
            pattern = rule.get("regex")
            if not pattern or not re.search(pattern, checkpoint_prefix):
                continue

            layers = rule.get("layers")
            if layers:
                match = re.search(r"layers\.(\d+)\.", checkpoint_prefix)
                if match:
                    layer_id = int(match.group(1))
                    if layer_id not in layers:
                        continue

            method = getattr(rule, "type", None)
            if not method:
                method = quant_cfg.type
            rule_kwargs = rule.get("kwargs", {})
            method_kwargs = quant_kwargs.get(method, {})
            merged_kwargs = {**rule_kwargs, **method_kwargs}
            return cls._get_quantized_class(
                class_type,
                method,
                quant_kwargs={method: merged_kwargs},
            )

        return cls._get_quantized_class(
            class_type,
            None,
            quant_kwargs=quant_kwargs,
        )

    @classmethod
    def get_quantized_linear_class_from_global_args(
        cls,
        *,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        checkpoint_prefix="",
    ) -> Optional[Type[QuantizedLinearBase]]:
        return cls._get_quantized_class_from_global_args(
            "linear",
            quant_kwargs=quant_kwargs,
            checkpoint_prefix=checkpoint_prefix,
        )

    @classmethod
    def get_quantized_moe_experts_class_from_global_args(
        cls,
        *,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        checkpoint_prefix="",
    ) -> Optional[Type[QuantizedMoeExpertsBase]]:
        return cls._get_quantized_class_from_global_args(
            "moe_experts",
            quant_kwargs=quant_kwargs,
            checkpoint_prefix=checkpoint_prefix,
        )

    @classmethod
    def register_linear(
        cls,
        name: Optional[str],
        implementation: Optional[Type[QuantizedLinearBase]] = None,
    ) -> None:
        """
        Register a new quantization Linear layer.

        Arguments:
            name: Name of the quant. None for non-quantized layer.
            implementation: Implementation class. If None, return a partial function as
                a decorator.
        """
        if implementation is None:
            return functools.partial(cls.register_linear, name)
        cls._linear_registry[name] = implementation
        return implementation

    @classmethod
    def register_moe_experts(
        cls,
        name: Optional[str],
        implementation: Optional[Type[QuantizedMoeExpertsBase]] = None,
    ) -> None:
        """
        Register a new MoeExperts layer.

        Arguments:
            name: Name of the MoeExperts layer. None for non-quantized layer.
            implementation: Implementation class. If None, return a partial function as
                a decorator.
        """
        if implementation is None:
            return functools.partial(cls.register_moe_experts, name)
        cls._moe_experts_registry[name] = implementation
        return implementation
