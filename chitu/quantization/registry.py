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


class QuantizationRegistry:
    """
    Registry of available quantization methods and their implementations.
    """

    _registry: Dict[str, Type[QuantizedLinearBase]] = {}
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
        ret = set(cls._registry.keys())
        ret.remove(None)
        return ret

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

        impl = cls._registry.get(method)
        if impl is None:
            raise ValueError(f"Unknown quantization method in `method`: {method}")

        for key in quant_kwargs:
            if key not in cls._registry:
                raise ValueError(
                    f"Unknown quantization method in `quant_kwargs`: {key}"
                )

        if method in quant_kwargs:

            class QuantLinearImpl(impl):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **quant_kwargs[method], **kwargs)

            impl = QuantLinearImpl

        return impl

    @classmethod
    def get_quantized_linear_class_from_global_args(
        cls,
        *,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        checkpoint_prefix="",
    ) -> Optional[Type[QuantizedLinearBase]]:
        args = get_global_args()
        quant_cfg = getattr(args.models, "quant_config", None)
        if quant_cfg is None:
            return cls.get_quantized_linear_class(None, quant_kwargs=quant_kwargs)

        rules = getattr(quant_cfg, "rules", [])
        for rule in rules:
            pattern = rule.get("regex")
            if not pattern or not re.search(pattern, checkpoint_prefix):
                continue

            method = getattr(rule, "type", None)
            if not method:
                method = quant_cfg.type
            rule_kwargs = rule.get("kwargs", {})
            method_kwargs = quant_kwargs.get(method, {})
            merged_kwargs = {**rule_kwargs, **method_kwargs}
            return cls.get_quantized_linear_class(
                method,
                quant_kwargs={method: merged_kwargs},
            )

        return cls.get_quantized_linear_class(
            None,
            quant_kwargs=quant_kwargs,
        )

    @classmethod
    def register_method(
        cls,
        name: Optional[str],
        implementation: Optional[Type[QuantizedLinearBase]] = None,
    ) -> None:
        """
        Register a new quantization method.

        Arguments:
            name: Name of the quantization method. None for non-quantized layer.
            implementation: Implementation class. If None, return a partial function as
                a decorator.
        """
        if implementation is None:
            return functools.partial(cls.register_method, name)
        cls._registry[name] = implementation
        return implementation
