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

    pass


class QuantizedAbsorbGemmBase(torch.nn.Module):
    """
    The two group GeMMs in "absorb-without-precomp" mode for MLA. This module runs locally on one device.

    Inherit from this class for quantization.
    """

    pass


class QuantizationRegistry:
    """
    Registry of available quantization methods and their implementations.
    """

    _linear_registry: Dict[str, Type[QuantizedLinearBase]] = {}
    _moe_experts_registry: Dict[str, Type[QuantizedMoeExpertsBase]] = {}
    _absorb_gemm_registry: Dict[str, Type[QuantizedAbsorbGemmBase]] = {}

    _allowed_quant_for_merge_gate_up: List = [
        "blockfp8",
        "autoawq",
        "simple_w8a8",
        None,
    ]
    _allowed_quant_for_merge_qkv: List = [
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
        ret = (
            set(cls._linear_registry.keys())
            .union(set(cls._moe_experts_registry.keys()))
            .union(set(cls._absorb_gemm_registry.keys()))
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
        elif class_type == "absorb_gemm":
            registry = cls._absorb_gemm_registry
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
    def get_quantized_absorb_gemm_class(
        cls,
        method: Optional[str],
        *,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
    ) -> Optional[Type[QuantizedAbsorbGemmBase]]:
        """
        Get the quantized AbsorbGemm implementation for the specified method.

        Arguments:
            method: Quantization method name, or None for no quantization
            quant_kwargs: Nested mapping for additional arguments for specific
                quantization methods. E.g., `{"quant_method_x": {"arg1": value1, ...}}`
        Returns:
            The quantized AbsorbGemm class, or None if method is None or not found
        """

        return cls._get_quantized_class(
            "absorb_gemm",
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
    def get_quantized_absorb_gemm_class_from_global_args(
        cls,
        *,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        checkpoint_prefix="",
    ) -> Optional[Type[QuantizedAbsorbGemmBase]]:
        return cls._get_quantized_class_from_global_args(
            "absorb_gemm",
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

    @classmethod
    def register_absorb_gemm(
        cls,
        name: Optional[str],
        implementation: Optional[Type[QuantizedAbsorbGemmBase]] = None,
    ) -> None:
        """
        Register a new quantization AbsorbGemm layer.

        Arguments:
            name: Name of the quant. None for non-quantized layer.
            implementation: Implementation class. If None, return a partial function as
                a decorator.
        """
        if implementation is None:
            return functools.partial(cls.register_absorb_gemm, name)
        cls._absorb_gemm_registry[name] = implementation
        return implementation
