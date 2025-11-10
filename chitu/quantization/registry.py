# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Mapping, Optional, Type, Callable
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
from chitu.distributed.parallel_state import get_tp_size
from chitu.utils import try_import_and_setup_torch_npu

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


class QuantizationRegistry:
    """
    Registry of available quantization methods and their implementations.
    """

    # NOTE: The inner dict's key can either be a `str` typed quantization method
    # nane, or `None` for no quantization (a.k.a. "normal" quantization)
    _linear_registry: dict[str, dict[str | None, Type[QuantizedLinearBase]]] = {}
    _moe_experts_registry: dict[
        str, dict[str | None, Type[QuantizedMoeExpertsBase]]
    ] = {}
    _absorb_gemm_registry: dict[
        str, dict[str | None, Type[QuantizedAbsorbGemmBase]]
    ] = {}

    _allowed_quant_for_merge_gate_up: list = [
        "blockfp4_merged",
        "blockfp8",
        "autoawq",
        "simple_w8a8",
        "mixq",
        "ascend_w8a8_dynamic",
        "w4_g128_symm_a8_symm",
        None,
    ]
    _allowed_quant_for_merge_qkv: list = [
        "blockfp4_merged",
        "blockfp8",
        "autoawq",
        "simple_w8a8",
        "mixq",
        "ascend_w8a8_dynamic",
        "w4_g128_symm_a8_symm",
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
    def allowed_merge_qkv(cls, checkpoint, can_use_mla_prologue_int8: bool = False):
        quant = get_quant_from_checkpoint_prefix(checkpoint)

        backend = get_backend_from_checkpoint_prefix(checkpoint)
        if backend == "cpuinfer":
            return False

        # This restriction is from
        # https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_mla_prolog_v2.md
        # Should be synchronized in the following files:
        # - chitu/models/model_deepseek_v3.py
        # - chitu/quantization/registry.py
        # - chitu/ops/mla_prologue.py
        args = get_global_args()
        if (
            has_torch_npu
            and (quant is None or can_use_mla_prologue_int8)
            and args.models.type == "deepseek-v3"
            and getattr(args.models, "index_topk", None) is None
            and args.infer.mla_absorb == "absorb-without-precomp"
            and torch.get_default_dtype() == torch.bfloat16
            and args.models.dim == 7168
            and args.models.q_lora_rank == 1536
            and args.models.n_heads // get_tp_size() in [8, 16, 32, 64, 128]
            and args.models.kv_lora_rank == 512
            and args.models.qk_nope_head_dim == 128
            and args.models.qk_rope_head_dim == 64
        ):
            return False  # Not merging, so we can use mla_prologue(impl=torch_npu), which is even better

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
        registry: dict[str, dict[str | None, Type]]
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
            raise ValueError(
                f"Unknown quantization method in `method`: {method}, `backend`: {backend_type}"
            )

        priority = -1
        impl: Type = None
        for impl_, when_, priority_ in backend_impls[method]:
            if when_() and priority_ > priority:
                impl, priority = impl_, priority_
        if impl is None:
            raise ValueError(
                f"No available implementation for quantization method: {method}, backend: {backend_type}"
            )

        if method in quant_kwargs:
            if method not in backend_impls:
                raise ValueError(
                    f"Unknown quantization method in `quant_kwargs`: {method}"
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
        when=lambda: True,
        priority: int = 0,
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
                cls.register_linear,
                name,
                backend_type=backend_type,
                when=when,
                priority=priority,
            )
        if backend_type not in cls._linear_registry:
            cls._linear_registry[backend_type] = {}

        if name not in cls._linear_registry[backend_type]:
            cls._linear_registry[backend_type][name] = []
        cls._linear_registry[backend_type][name].append(
            (implementation, when, priority)
        )

        return implementation

    @classmethod
    def register_moe_experts(
        cls,
        name: Optional[str],
        implementation: Optional[Type[QuantizedMoeExpertsBase]] = None,
        backend_type: str = "default",
        when=lambda: True,
        priority: int = 0,
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
                cls.register_moe_experts,
                name,
                backend_type=backend_type,
                when=when,
                priority=priority,
            )
        if backend_type not in cls._moe_experts_registry:
            cls._moe_experts_registry[backend_type] = {}
        if name not in cls._moe_experts_registry[backend_type]:
            cls._moe_experts_registry[backend_type][name] = []
        cls._moe_experts_registry[backend_type][name].append(
            (implementation, when, priority)
        )

        return implementation

    @classmethod
    def register_absorb_gemm(
        cls,
        name: Optional[str],
        implementation: Optional[Type[QuantizedAbsorbGemmBase]] = None,
        backend_type: str = "default",
        when=lambda: True,
        priority: int = 0,
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
                cls.register_absorb_gemm,
                name,
                backend_type=backend_type,
                when=when,
                priority=priority,
            )
        if backend_type not in cls._absorb_gemm_registry:
            cls._absorb_gemm_registry[backend_type] = {}

        if name not in cls._absorb_gemm_registry[backend_type]:
            cls._absorb_gemm_registry[backend_type][name] = []
        cls._absorb_gemm_registry[backend_type][name].append(
            (implementation, when, priority)
        )

        return implementation
