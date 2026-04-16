# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from chitu.utils import (
    try_import_opt_dep,
    try_import_platform_dep,
    try_import_and_setup_torch_npu,
)

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")
flashinfer, has_flashinfer = try_import_opt_dep("flashinfer", "flashinfer")
hygon_w4a8_kernels, has_hygon_w4a8 = try_import_platform_dep("sugon_w4a8_kernels")
hard_fp4_kernels, has_hard_fp4_kernels = try_import_platform_dep("hard_fp4_kernels")
metax_soft_fp4_kernels, has_metax_soft_fp4 = try_import_platform_dep(
    "metax_soft_fp4_kernels"
)

if has_torch_npu:
    from chitu.npu_utils import fused_experts_no_sum_npu, fused_experts_npu_for_ep
if has_triton:
    from .triton_batched_experts import triton_batched_experts
    from .triton_fused_experts import (
        fused_experts,
        fused_experts_int8,
        fused_experts_fp8,
        fused_experts_soft_fp4,
    )
if has_deep_gemm:
    from .deepgemm_masked import deepgemm_masked_fused_expert
    from .deepgemm_contiguous import deepgemm_contiguous_fused_expert


def make_op_dispatcher(func: Callable):
    """
    Make a function-level impl dispatcher.

    The returned callable keeps the original function metadata/type hint surface by
    `functools.wraps`, and exposes:
      - `@op.register("<impl>")`
      - `@op.register_auto`
    """

    handlers: Dict[str, Callable] = {}
    auto_resolver: Optional[Callable] = None

    @functools.wraps(func)
    def dispatcher(*args, **kwargs):
        nonlocal auto_resolver
        impl = kwargs.get("impl", "auto")

        if impl == "auto":
            if auto_resolver is None:
                raise NotImplementedError(
                    f"{func.__name__}: auto impl is not registered"
                )
            if len(inspect.signature(auto_resolver).parameters) == 0:
                impl = auto_resolver()
            else:
                impl = auto_resolver(*args, **kwargs)

        if impl not in handlers:
            raise NotImplementedError(
                f"{func.__name__}: impl '{impl}' is not registered"
            )

        target = handlers[impl]
        call_kwargs = dict(kwargs)
        call_kwargs["impl"] = impl
        target_sig = inspect.signature(target)
        params = target_sig.parameters
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if accepts_var_kwargs:
            return target(*args, **call_kwargs)
        filtered_kwargs = {k: v for k, v in call_kwargs.items() if k in params}
        return target(*args, **filtered_kwargs)

    def register(name: str):
        def decorator(impl_func: Callable):
            handlers[name] = impl_func
            return impl_func

        return decorator

    def register_auto(auto_func: Callable):
        nonlocal auto_resolver
        auto_resolver = auto_func
        return auto_func

    dispatcher.register = register
    dispatcher.register_auto = register_auto
    return dispatcher
