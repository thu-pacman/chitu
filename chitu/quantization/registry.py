from typing import Any, Dict, List, Mapping, Optional, Set, Type, Callable
import functools
import re

import torch

from chitu.global_vars import get_global_args
from chitu.quantization.base import (
    QuantizedLinearBase,
    QuantizedMoeExpertsBase,
    QuantizedAbsorbGemmBase,
)
from chitu.quantization.utils import (
    get_quant_from_checkpoint_prefix,
    get_backend_from_checkpoint_prefix,
)


class QuantizationRegistry:
    """
    Registry of available quantization methods and their implementations.
    """

    # NOTE: The inner dict's key can either be a `str` typed quantization method
    # nane, or `None` for no quantization (a.k.a. "normal" quantization)
    _linear_registry: Dict[str, Dict[str | None, Type[QuantizedLinearBase]]] = {}
    _moe_experts_registry: Dict[
        str, Dict[str | None, Type[QuantizedMoeExpertsBase]]
    ] = {}
    _absorb_gemm_registry: Dict[
        str, Dict[str | None, Type[QuantizedAbsorbGemmBase]]
    ] = {}

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
    def allowed_merge_gate_up(cls, checkpoint):
        quant = get_quant_from_checkpoint_prefix(checkpoint)
        backend = get_backend_from_checkpoint_prefix(checkpoint)
        if backend == "cpuinfer":
            return False
        return quant in QuantizationRegistry._allowed_quant_for_merge_gate_up

    @classmethod
    def allowed_merge_qkv(cls, checkpoint):
        quant = get_quant_from_checkpoint_prefix(checkpoint)
        backend = get_backend_from_checkpoint_prefix(checkpoint)
        if backend == "cpuinfer":
            return False
        return quant in QuantizationRegistry._allowed_quant_for_merge_qkv

    @classmethod
    def _get_quantized_class(
        cls,
        class_type: str,
        method: Optional[str],
        *,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        backend_type: str = "default",
    ) -> Type:
        registry: Dict[str, Dict[str | None, Type]]
        if class_type == "linear":
            registry = cls._linear_registry
        elif class_type == "moe_experts":
            registry = cls._moe_experts_registry
        elif class_type == "absorb_gemm":
            registry = cls._absorb_gemm_registry
        else:
            raise ValueError(f"Unknown class type: {class_type}")

        if backend_type not in registry:
            raise ValueError(f"Unknown backend impls: {backend_type}")
        backend_impls = registry[backend_type]

        if method not in backend_impls:
            raise ValueError(f"Unknown quantization method in `method`: {method}")
        impl: Type = backend_impls[method]

        for key in quant_kwargs:
            if key not in backend_impls:
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
    ) -> Type[QuantizedLinearBase]:
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
    ) -> Type[QuantizedMoeExpertsBase]:
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
    ) -> Type[QuantizedAbsorbGemmBase]:
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
    ) -> Type:
        args = get_global_args()
        quant_cfg = getattr(args.models, "quant_config", None)
        if quant_cfg is None:
            return cls._get_quantized_class(class_type, None, quant_kwargs=quant_kwargs)

        rules = getattr(quant_cfg, "rules", [])
        backend_type = get_backend_from_checkpoint_prefix(checkpoint_prefix)
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
                backend_type=backend_type,
            )

        return cls._get_quantized_class(
            class_type,
            None,
            quant_kwargs=quant_kwargs,
            backend_type=backend_type,
        )

    @classmethod
    def get_quantized_linear_class_from_global_args(
        cls,
        *,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        checkpoint_prefix="",
    ) -> Type[QuantizedLinearBase]:
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
    ) -> Type[QuantizedMoeExpertsBase]:
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
    ) -> Type[QuantizedAbsorbGemmBase]:
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
        backend_type: str = "default",
    ) -> Callable | Type[QuantizedLinearBase]:
        """
        Register a new quantization Linear layer.

        Arguments:
            name: Name of the quant. None for non-quantized layer.
            implementation: Implementation class. If None, return a partial function as
                a decorator.
        """
        if implementation is None:
            return functools.partial(
                cls.register_linear, name, backend_type=backend_type
            )
        if backend_type not in cls._linear_registry:
            cls._linear_registry[backend_type] = {}
        cls._linear_registry[backend_type][name] = implementation
        return implementation

    @classmethod
    def register_moe_experts(
        cls,
        name: Optional[str],
        implementation: Optional[Type[QuantizedMoeExpertsBase]] = None,
        backend_type: str = "default",
    ) -> Callable | Type[QuantizedMoeExpertsBase]:
        """
        Register a new MoeExperts layer.

        Arguments:
            name: Name of the MoeExperts layer. None for non-quantized layer.
            implementation: Implementation class. If None, return a partial function as
                a decorator.
        """
        if implementation is None:
            return functools.partial(
                cls.register_moe_experts, name, backend_type=backend_type
            )
        if backend_type not in cls._moe_experts_registry:
            cls._moe_experts_registry[backend_type] = {}
        cls._moe_experts_registry[backend_type][name] = implementation
        return implementation

    @classmethod
    def register_absorb_gemm(
        cls,
        name: Optional[str],
        implementation: Optional[Type[QuantizedAbsorbGemmBase]] = None,
        backend_type: str = "default",
    ) -> Callable | Type[QuantizedAbsorbGemmBase]:
        """
        Register a new quantization AbsorbGemm layer.

        Arguments:
            name: Name of the quant. None for non-quantized layer.
            implementation: Implementation class. If None, return a partial function as
                a decorator.
        """
        if implementation is None:
            return functools.partial(
                cls.register_absorb_gemm, name, backend_type=backend_type
            )
        if backend_type not in cls._absorb_gemm_registry:
            cls._absorb_gemm_registry[backend_type] = {}
        cls._absorb_gemm_registry[backend_type][name] = implementation
        return implementation
