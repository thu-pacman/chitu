# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Mapping, Optional, Type, Callable
import copy
from dataclasses import dataclass
import functools
import re
from logging import getLogger
import torch
from tabulate import tabulate

from chitu.global_vars import get_global_args
from chitu.quantization.base import (
    QuantizedLinearBase,
    QuantizedMoeExpertsBase,
    QuantizedMoeExpertsUnmerged,
    QuantizedMoeExpertsMerged,
    QuantizedAbsorbGemmBase,
)
from chitu.quantization.utils import (
    get_quant_from_checkpoint_prefix,
    get_quant_kwargs_from_checkpoint_prefix,
    get_backend_from_checkpoint_prefix,
)
from chitu.checkpoint_prefix import CheckpointPrefix
from chitu.distributed.parallel_state import get_tp_size
from chitu.utils import try_import_and_setup_torch_npu

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
logger = getLogger(__name__)


@dataclass(frozen=True)
class _ObservedModuleImplRequest:
    class_type: str
    backend_type: str
    method: str | None
    effective_kwargs: Mapping[str, Any]


_LinearImplEntry = tuple[
    Type[QuantizedLinearBase], Callable[[Mapping[str, Any]], bool], int
]
_MoeExpertsUnmergedImplEntry = tuple[
    Type[QuantizedMoeExpertsUnmerged], Callable[[Mapping[str, Any]], bool], int
]
_MoeExpertsMergedImplEntry = tuple[
    Type[QuantizedMoeExpertsMerged], Callable[[Mapping[str, Any]], bool], int
]
_AbsorbGemmImplEntry = tuple[
    Type[QuantizedAbsorbGemmBase], Callable[[Mapping[str, Any]], bool], int
]


def _copy_observed_kwargs(kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        return copy.deepcopy(kwargs)
    except Exception:
        return dict(kwargs)


class QuantizationRegistry:
    """
    Registry of available quantization methods and their implementations.
    """

    # NOTE: The inner dict's key can either be a `str` typed quantization method
    # nane, or `None` for no quantization (a.k.a. "normal" quantization)
    _linear_registry: dict[str, dict[str | None, list[_LinearImplEntry]]] = {}
    _moe_experts_unmerged_registry: dict[
        str, dict[str | None, list[_MoeExpertsUnmergedImplEntry]]
    ] = {}
    _moe_experts_merged_registry: dict[
        str, dict[str | None, list[_MoeExpertsMergedImplEntry]]
    ] = {}
    _absorb_gemm_registry: dict[str, dict[str | None, list[_AbsorbGemmImplEntry]]] = {}

    _observed_module_impl_requests: list[_ObservedModuleImplRequest] = []
    _observed_module_impl_summary_emitted = False

    _allowed_quant_for_merge_gate_up: list = [
        "blockfp4_merged",
        "blockfp8",
        "fp8_per_channel",
        "autoawq",
        "w8a8_per_token_per_channel_dyn",
        "mixq",
        "w8a8_per_token_per_channel_dyn",
        "w4_g128_symm_a8_symm",
        "blockint4",
        "hygon_w4a8",
        None,
    ]
    _allowed_quant_for_merge_qkv: list = [
        "blockfp4_merged",
        "blockfp8",
        "fp8_per_channel",
        "autoawq",
        "w8a8_per_token_per_channel_dyn",
        "mixq",
        "w8a8_per_token_per_channel_dyn",
        "w4_g128_symm_a8_symm",
        "hygon_w4a8",
        None,
    ]

    @classmethod
    def _get_module_registry(cls, class_type: str):
        if class_type == "linear":
            return cls._linear_registry
        if class_type == "moe_experts_unmerged":
            return cls._moe_experts_unmerged_registry
        if class_type == "moe_experts_merged":
            return cls._moe_experts_merged_registry
        if class_type == "absorb_gemm":
            return cls._absorb_gemm_registry
        raise ValueError(f"Unknown class type: {class_type}")

    @staticmethod
    def _select_module_impl(backend_impl, effective_kwargs: Mapping[str, Any]):
        priority = -1
        impl: Type | None = None
        availability: dict[Type, bool] = {}
        for impl_, when_, priority_ in backend_impl:
            available = when_(effective_kwargs)
            availability[impl_] = available
            if available and priority_ > priority:
                impl, priority = impl_, priority_
        return impl, availability

    @staticmethod
    def _format_module_selection_key(
        class_type: str, backend_type: str, method: str | None
    ) -> str:
        quant = "none" if method is None else method
        return f"{class_type}[backend={backend_type}, quant={quant}]"

    @classmethod
    def format_observed_quantized_module_impl_summary_lines(cls) -> list[str]:
        table = []
        seen_selection_keys = set()
        for request in cls._observed_module_impl_requests:
            selection_key = cls._format_module_selection_key(
                request.class_type, request.backend_type, request.method
            )
            if selection_key in seen_selection_keys:
                continue
            seen_selection_keys.add(selection_key)

            registry = cls._get_module_registry(request.class_type)
            backend_impl = registry.get(request.backend_type, {}).get(request.method)
            if not backend_impl:
                continue

            selected_impl, availability = cls._select_module_impl(
                backend_impl, request.effective_kwargs
            )
            if selected_impl is None:
                continue

            row = [selection_key]
            for impl_, _, _ in backend_impl:
                if impl_ is selected_impl:
                    marker = "✓"
                elif availability.get(impl_, False):
                    marker = "·"
                else:
                    marker = "×"
                row.append(f"{impl_.__qualname__} {marker}")
            table.append(row)

        if not table:
            return []

        max_columns = max(len(row) for row in table)
        for row in table:
            row += [""] * (max_columns - len(row))
        return tabulate(table, tablefmt="github").splitlines()

    @classmethod
    def emit_observed_quantized_module_impl_summary(cls, target_logger=None) -> bool:
        lines = cls.format_observed_quantized_module_impl_summary_lines()
        if not lines:
            return False

        if cls._observed_module_impl_summary_emitted:
            return False
        cls._observed_module_impl_summary_emitted = True

        if target_logger is None:
            target_logger = logger
        target_logger.info(
            "Quantized module implementations selected during model build (✓ = selected; · = not selected due to priority; × = unavailable):"
        )
        for line in lines:
            target_logger.info(line)
        target_logger.info(
            "Set CHITU_LOGGING_LEVEL=chitu.backend:DEBUG to see the full model structure, including where each selected module is used."
        )
        return True

    @classmethod
    def allowed_merge_gate_up(cls, checkpoint: str | CheckpointPrefix):
        quant = get_quant_from_checkpoint_prefix(checkpoint)
        backend = get_backend_from_checkpoint_prefix(checkpoint)
        if backend == "cpuinfer":
            return False
        return quant in QuantizationRegistry._allowed_quant_for_merge_gate_up

    @classmethod
    def allowed_merge_qkv(
        cls,
        checkpoint: str | CheckpointPrefix,
        can_use_mla_prologue_int8: bool = False,
    ):
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
        from chitu.models.registry import ModelType

        args = get_global_args()
        if (
            has_torch_npu
            and (quant is None or can_use_mla_prologue_int8)
            and args.models.type == ModelType.DEEPSEEK_V3
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
        registry = cls._get_module_registry(class_type)

        if backend_type not in registry:
            raise ValueError(f"Unknown backend impls: {backend_type}")
        backend_impls = registry[backend_type]
        if method not in backend_impls:
            raise ValueError(
                f"Unknown quantization: `method` {method}, `backend` {backend_type}"
            )
        backend_impl = backend_impls[method]
        if method is not None:
            effective_kwargs = quant_kwargs.get(method, {})
        else:
            effective_kwargs = {}
        cls._observed_module_impl_requests.append(
            _ObservedModuleImplRequest(
                class_type=class_type,
                backend_type=backend_type,
                method=method,
                effective_kwargs=_copy_observed_kwargs(effective_kwargs),
            )
        )

        impl, _ = cls._select_module_impl(backend_impl, effective_kwargs)
        if impl is None:
            raise ValueError(
                f"No available implementation for quantization method: {method}, backend: {backend_type}"
            )

        if method is not None and method in quant_kwargs:

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
        merge_gate_up: bool,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
    ) -> Type[QuantizedMoeExpertsBase]:
        """
        Get the quantized MoeExperts implementation for the specified method.

        Arguments:
            method: Quantization method name, or None for no quantization
            quant_kwargs: Nested mapping for additional arguments for specific
                quantization methods. E.g., `{"quant_method_x": {"arg1": value1, ...}}`
            merge_gate_up: Whether to merge gate and up projection.
        Returns:
            The quantized moe class, or None if method is None or not found
        """

        return cls._get_quantized_class(
            "moe_experts_merged" if merge_gate_up else "moe_experts_unmerged",
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
        checkpoint_prefix: str | CheckpointPrefix = "",
    ) -> Type:
        method = get_quant_from_checkpoint_prefix(checkpoint_prefix)
        if method is not None:
            kwargs_of_method_from_user = quant_kwargs.get(method, {})
        else:
            kwargs_of_method_from_user = {}
        kwargs_of_method_from_rules = get_quant_kwargs_from_checkpoint_prefix(
            checkpoint_prefix
        )
        joined_kwargs_of_method = {
            **kwargs_of_method_from_rules,
            **kwargs_of_method_from_user,
        }
        if method is not None:
            joined_quant_kwargs = {method: joined_kwargs_of_method}
        else:
            joined_quant_kwargs = {}
        return cls._get_quantized_class(
            class_type,
            method,
            quant_kwargs=joined_quant_kwargs,
            backend_type=get_backend_from_checkpoint_prefix(checkpoint_prefix),
        )

    @classmethod
    def get_quantized_linear_class_from_global_args(
        cls,
        *,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        checkpoint_prefix: str | CheckpointPrefix = "",
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
        merge_gate_up: bool,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        checkpoint_prefix: str | CheckpointPrefix = "",
    ) -> Type[QuantizedMoeExpertsBase]:
        return cls._get_quantized_class_from_global_args(
            "moe_experts_merged" if merge_gate_up else "moe_experts_unmerged",
            quant_kwargs=quant_kwargs,
            checkpoint_prefix=checkpoint_prefix,
        )

    @classmethod
    def get_quantized_absorb_gemm_class_from_global_args(
        cls,
        *,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        checkpoint_prefix: str | CheckpointPrefix = "",
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
        *,
        backend_type: str = "default",
        when: Callable[[Mapping[str, Any]], bool] = lambda _: True,
        priority: int = 0,
    ) -> Callable | Type[QuantizedLinearBase]:
        """
        Register a new quantization Linear layer.

        Arguments:
            name: Name of the quant. None for non-quantized layer.
            implementation: Implementation class. If None, return a partial function as
                a decorator.
            when: Only select this implementation if the `when` function returns True.
                The `when` function accepts the kwargs passed to the model as its only
                argument.
            proiority: When multiple implementations match, the one with the highest
                priority will be selected.
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
        *,
        merge_gate_up: bool,
        backend_type: str = "default",
        when: Callable[[Mapping[str, Any]], bool] = lambda _: True,
        priority: int = 0,
    ) -> Callable | Type[QuantizedMoeExpertsBase]:
        """
        Register a new MoeExperts layer.

        Arguments:
            name: Name of the MoeExperts layer. None for non-quantized layer.
            implementation: Implementation class. If None, return a partial function as
                a decorator.
            when: Only select this implementation if the `when` function returns True.
                The `when` function accepts the kwargs passed to the model as its only
                argument.
            proiority: When multiple implementations match, the one with the highest
                priority will be selected.
        """
        if implementation is None:
            return functools.partial(
                cls.register_moe_experts,
                name,
                merge_gate_up=merge_gate_up,
                backend_type=backend_type,
                when=when,
                priority=priority,
            )
        if merge_gate_up:
            registry = cls._moe_experts_merged_registry
        else:
            registry = cls._moe_experts_unmerged_registry
        if backend_type not in registry:
            registry[backend_type] = {}
        if name not in registry[backend_type]:
            registry[backend_type][name] = []
        registry[backend_type][name].append((implementation, when, priority))

        return implementation

    @classmethod
    def register_absorb_gemm(
        cls,
        name: Optional[str],
        implementation: Optional[Type[QuantizedAbsorbGemmBase]] = None,
        *,
        backend_type: str = "default",
        when: Callable[[Mapping[str, Any]], bool] = lambda _: True,
        priority: int = 0,
    ) -> Callable | Type[QuantizedAbsorbGemmBase]:
        """
        Register a new quantization AbsorbGemm layer.

        Arguments:
            name: Name of the quant. None for non-quantized layer.
            implementation: Implementation class. If None, return a partial function as
                a decorator.
            when: Only select this implementation if the `when` function returns True.
                The `when` function accepts the kwargs passed to the model as its only
                argument.
            proiority: When multiple implementations match, the one with the highest
                priority will be selected.
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
