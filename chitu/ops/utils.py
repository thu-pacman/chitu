# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
from dataclasses import dataclass, field
from logging import getLogger
from threading import Lock
from typing import Callable, Dict, Optional
from collections import OrderedDict

import torch

logger = getLogger(__name__)


@dataclass
class _ObservedOpImplState:
    impl_availability: Dict[str, bool] = field(default_factory=dict)
    selected_impls: set[str] = field(default_factory=set)


_observed_op_impls: Dict[str, _ObservedOpImplState] = {}
_observed_op_impl_summary_emitted = False


def compatible_with_inplace(fn):
    """
    Make an out-of-place-only op compatible with in-place usage via `out` argument.

    This is a fallback wrapper with performance degradation. DO NOT use it on ops that
    already supports in-place usage.
    """

    @functools.wraps(fn)
    def wrapper(*args, out: Optional[torch.Tensor] = None, **kwargs):
        tmp_out = fn(*args, **kwargs)
        if out is not None:
            if out.shape != tmp_out.shape:
                raise ValueError(
                    f"Illegal in-place operation destination: the destination has shape "
                    f"{out.shape}, but the result has shape {tmp_out.shape}"
                )
            if out.dtype != tmp_out.dtype:
                raise ValueError(
                    f"Illegal in-place operation destination: the destination has dtype "
                    f"{out.dtype}, but the result has dtype {tmp_out.dtype}"
                )
            out.copy_(tmp_out)
        else:
            out = tmp_out
        return out

    sig = inspect.signature(wrapper)
    params = list(sig.parameters.values())
    out_param = inspect.Parameter("out", inspect.Parameter.KEYWORD_ONLY)
    params.append(out_param)
    wrapper.__signature__ = sig.replace(parameters=params)

    return wrapper


def _get_observed_op_impl_state(op_name: str) -> _ObservedOpImplState:
    state = _observed_op_impls.get(op_name)
    if state is None:
        state = _ObservedOpImplState()
        _observed_op_impls[op_name] = state
    return state


def _record_op_impl_availability(op_name: str, impl_name: str, available: bool):
    state = _get_observed_op_impl_state(op_name)
    state.impl_availability[impl_name] = available


def _record_selected_op_impl(op_name: str, impl_name: str):
    state = _get_observed_op_impl_state(op_name)
    state.selected_impls.add(impl_name)


def format_observed_op_impl_summary_lines(pretty: bool = True) -> list[str]:
    observed_states = [
        (
            op_name,
            dict(state.impl_availability),
            set(state.selected_impls),
        )
        for op_name, state in _observed_op_impls.items()
        if state.selected_impls
    ]

    table = []
    for op_name, avail_map, selected_impls in observed_states:
        impl_names = list(avail_map)
        for impl_name in selected_impls:
            if impl_name not in avail_map:
                impl_names.append(impl_name)

        row = [op_name]
        for name in impl_names:
            if name in selected_impls:
                marker = "✓"
            elif avail_map.get(name, True):
                marker = "·"
            else:
                marker = "×"
            row.append(f"{name} {marker}")
        table.append(row)

    if pretty:
        max_columns = max(len(row) for row in table)
        for row in table:
            row += [""] * (max_columns - len(row))
        for i in range(max_columns):
            max_width = max(len(row[i]) for row in table)
            for row in table:
                row[i] = row[i].ljust(max_width)

    lines = []
    for row in table:
        lines.append(" | ".join(row))
    return lines


def emit_observed_op_impl_summary(target_logger=None) -> bool:
    global _observed_op_impl_summary_emitted

    lines = format_observed_op_impl_summary_lines()
    if not lines:
        return False

    if _observed_op_impl_summary_emitted:
        return False
    _observed_op_impl_summary_emitted = True

    if target_logger is None:
        target_logger = logger
    target_logger.info(
        "Operator implementations used during warmup (✓ = used at least once; · = not used; × = not installed):"
    )
    for line in lines:
        target_logger.info(line)
    return True


def clear_observed_op_impl_selections():
    global _observed_op_impl_summary_emitted

    for state in _observed_op_impls.values():
        state.selected_impls.clear()
    _observed_op_impl_summary_emitted = False


def reset_observed_op_impl_state():
    global _observed_op_impl_summary_emitted

    _observed_op_impls.clear()
    _observed_op_impl_summary_emitted = False


def make_op_dispatcher(
    func: Optional[Callable] = None, *, op_name: Optional[str] = None
):
    """
    Make a function-level impl dispatcher.

    The returned callable keeps the original function metadata/type hint surface by
    `functools.wraps`, and exposes:
      - `@op.register("<impl>")`: Register a specific implementation.
      - `@op.register_auto`: Register an algorithm to select from implementations.
      - `@op.register_candidate("<impl>")`: Mark an implementation as maybe-available.
        Then you can optionally register it using `@op.register("<impl>")` later.

    The registered implementation should accpet all parameters of the dispatcher, and
    all the non-keyword-only parameters should be in the same order. The registered
    implementation may accept more parameters than the dispatcher, with default values.
    The restrictions above are checked, unless `check_params=False`.
    """

    def decorator(dispatch_func: Callable):
        handlers: Dict[str, Callable] = {}
        impl_availability: Dict[str, bool] = {}
        # None  -> pass *args/**kwargs directly (exact match or variadic handler)
        # frozenset -> filter bound args by these names (handler is a name-subset)
        # dict  -> rename+filter via {dispatcher_name: handler_name} mapping
        auto_resolver: Optional[Callable] = None
        _auto_pass_args: bool = False
        _auto_kw_filter: Optional[frozenset] = frozenset()
        dispatch_name = op_name or dispatch_func.__name__
        dispatch_sig = inspect.signature(dispatch_func)
        dispatch_all_param_names = frozenset(
            p.name
            for p in dispatch_sig.parameters.values()
            if p.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        )
        dispatch_param_names = dispatch_all_param_names - {"impl"}

        def resolve_impl(*args, **kwargs):
            impl = kwargs.get("impl", "auto")

            if impl == "auto":
                if auto_resolver is None:
                    raise NotImplementedError(
                        f"{dispatch_name}: auto impl is not registered"
                    )
                if _auto_kw_filter is None:
                    fwd = {k: v for k, v in kwargs.items() if k != "impl"}
                else:
                    fwd = {k: v for k, v in kwargs.items() if k in _auto_kw_filter}
                if _auto_pass_args:
                    impl = auto_resolver(*args, **fwd)
                else:
                    impl = auto_resolver(**fwd)

            if impl not in handlers:
                raise NotImplementedError(
                    f"{dispatch_name}: impl '{impl}' is not registered"
                )
            if not impl_availability.get(impl, True):
                raise NotImplementedError(
                    f"{dispatch_name}: impl '{impl}' is not available in current env"
                )
            return impl

        @functools.wraps(dispatch_func)
        def dispatcher(*args, **kwargs):
            impl = resolve_impl(*args, **kwargs)
            _record_selected_op_impl(dispatch_name, impl)

            params = inspect.signature(handlers[impl]).parameters
            if "impl" in params:
                kwargs["impl"] = impl
            elif "impl" in kwargs:
                del kwargs["impl"]

            return handlers[impl](*args, **kwargs)

        def register_candidate(name: str):
            """
            Register that an implementation MAY be available.

            If the implementation is available, call `.register` later to actually register it.
            If the implementation is not avaiable, there is no need to call `.register` later, and
            the implementation will be still on the list but marked as unavailable.

            This method is optional, but useful for query unavailable implementations in a unified
            way. If your implemenation is always available, you can safely ignore this method.
            """

            if name not in impl_availability:
                impl_availability[name] = False
                _record_op_impl_availability(dispatch_name, name, False)

        def register(name: str, *, available: bool = True, check_params: bool = True):
            def register_decorator(impl_func: Callable = None):
                if check_params:
                    dispatcher_params = OrderedDict(
                        inspect.signature(dispatch_func).parameters
                    )
                    impl_params = OrderedDict(inspect.signature(impl_func).parameters)
                    if "impl" in dispatcher_params:
                        del dispatcher_params["impl"]
                    if "impl" in impl_params:
                        del impl_params["impl"]

                    # The registered implementation should accpet all parameters of the dispatcher
                    dispatcher_params_set = set(dispatcher_params)
                    impl_params_set = set(impl_params)
                    if not dispatcher_params_set.issubset(impl_params_set):
                        raise ValueError(
                            f"{impl_func} should accepet all parameters of {dispatch_func}: "
                            f"{dispatcher_params_set - impl_params_set}"
                        )

                    # Unless keyword-only, the parameters should in the same order
                    dispatcher_params_list = [
                        name
                        for name, param in dispatcher_params.items()
                        if param.kind != param.KEYWORD_ONLY
                    ]
                    impl_params_list = [
                        name
                        for name, param in impl_params.items()
                        if param.kind != param.KEYWORD_ONLY
                        and name in dispatcher_params_list
                    ]
                    if dispatcher_params_list != impl_params_list:
                        raise ValueError(
                            f"{impl_func} should accept parameters in the same order as {dispatch_func}: "
                            f"{dispatcher_params_list} != {impl_params_list}"
                        )

                handlers[name] = impl_func
                impl_availability[name] = available
                _record_op_impl_availability(dispatch_name, name, available)
                return impl_func

            return register_decorator

        def register_auto(auto_func: Callable):
            nonlocal auto_resolver, _auto_pass_args, _auto_kw_filter
            auto_resolver = auto_func
            sig = inspect.signature(auto_func)
            params = sig.parameters
            _auto_pass_args = any(
                p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                )
                for p in params.values()
            )
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            _auto_kw_filter = None if has_var_kw else frozenset(params.keys())
            return auto_func

        dispatcher.register_candidate = register_candidate
        dispatcher.register = register
        dispatcher.register_auto = register_auto
        dispatcher.resolve_impl = resolve_impl
        return dispatcher

    if func is None:
        return decorator
    return decorator(func)
