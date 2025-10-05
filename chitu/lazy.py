# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Sequence, Mapping, Any
from typing_extensions import override
from dataclasses import dataclass
import abc
import copy
import inspect
import functools
import torch

from chitu.native_layout import NativeLayoutTensor


@dataclass
class LazyTensorEvaluation:
    """
    The result of a LazyTensor evaluation, can be shared by multiple
    LazyTensor views.
    """

    is_evaluated: bool = False
    result: Any = None


class LazyTensor:
    """
    The result of a LazyOp, evaluated in the future.
    """

    def __init__(self, func: Callable, args: Sequence, kwargs: Mapping):
        self.func: Callable = func
        self.kwargs: Mapping[str, Any] = self._merge_to_kwargs(
            func, *args, **kwargs
        )  # args will be merged into kwargs

        meta_tensor = func(**self._move_to_meta(self.kwargs))
        self.shape = (
            meta_tensor.plain_shape
            if isinstance(meta_tensor, NativeLayoutTensor)
            else meta_tensor.shape
        )
        self.dtype = meta_tensor.dtype
        self.device = self._get_first_device(self.kwargs)

        self.evaluation = LazyTensorEvaluation()
        self.result_view_shape = self.shape

    def is_evaluated(self):
        return self.evaluation.is_evaluated

    def evaluate(self):
        if not self.evaluation.is_evaluated:
            self.evaluation.result = self.func(**self.kwargs)
            self.evaluation.is_evaluated = True
        if isinstance(self.evaluation.result, NativeLayoutTensor):
            if tuple(self.evaluation.result.plain_shape) != tuple(
                self.result_view_shape
            ):
                raise NotImplementedError(
                    "`view` on NativeLayoutTensor is not supported"
                )
            return self.evaluation.result
        else:
            return self.evaluation.result.view(self.result_view_shape)

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], Sequence):
            shape = shape[0]
        viewer = copy.copy(self)
        viewer.result_view_shape = shape
        return viewer

    @staticmethod
    def _merge_to_kwargs(func: Callable, *args, **kwargs) -> Mapping[str, Any]:
        # Get the function's signature
        sig = inspect.signature(func)

        # Bind arguments to the signature
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()  # Apply default values

        # Check for incompatible parameters
        for param in sig.parameters.values():
            if param.kind in (param.POSITIONAL_ONLY, param.VAR_POSITIONAL):
                raise TypeError(
                    f"Every parameter must be named to use `_merge_to_kwargs`, but "
                    f"function {func} has unnamed {param.kind.description} parameters"
                )

        # Merge into a single dictionary
        merged_kwargs: dict[str, Any] = {}
        for name, value in bound.arguments.items():
            param = sig.parameters[name]
            if param.kind == param.VAR_KEYWORD:
                merged_kwargs.update(value)  # Merge **kwargs-style dict
            else:
                merged_kwargs[name] = value  # Add named arguments

        return merged_kwargs

    @staticmethod
    def _move_to_meta(kwargs: Mapping[str, Any]):
        # TODO: Check tensors in Sequence or Mapping
        return {
            k: (
                v.to(torch.device("meta"))
                if isinstance(v, torch.Tensor) or isinstance(v, NativeLayoutTensor)
                else v
            )
            for k, v in kwargs.items()
        }

    @staticmethod
    def _get_first_device(kwargs: Mapping[str, Any]) -> torch.device:
        # TODO: Check tensors in Sequence or Mapping
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor) or isinstance(v, NativeLayoutTensor):
                return v.device
        raise ValueError("LazyData can't record a operation that doesn't use tensors")


class LazyOp(abc.ABC):
    """
    Base class for a opertion that returns a LazyTensor instead of a Tensor

    LazyOp is useful for fusing operators across modules.
    """

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> LazyTensor:
        """
        Record the arguments to the op and return a LazyTensor
        """

        raise NotImplementedError

    @abc.abstractmethod
    def lazy_tensor_type(self) -> type:
        """
        Retrun the subclass of LazyTensor that will be returned from __call__
        """

        raise NotImplementedError


def eval_lazy(x):
    if isinstance(x, LazyTensor):
        return x.evaluate()
    else:
        return x


def make_lazy_op(func: Callable) -> LazyOp:
    """
    Turn a function into a LazyOp.

    The function must support shape and dtype inference on "meta" device.

    Example:
    ```
    @make_lazy_op
    def foo(x):
        return torch.sqrt(x)

    y = foo(x)

    # You can check the type of `y` and optionally fuse it with other ops
    assert isinstance(y, foo.lazy_tensor_type)

    # You can also evaluate it instantly
    y.evaluate()  # Returns torch.sqrt(x)
    ```
    """

    class LazyTensorImpl(LazyTensor):
        pass

    class LazyOpImpl(LazyOp):
        @override
        def __call__(self, *args, **kwargs):
            return LazyTensorImpl(func, args, kwargs)

        @override
        def lazy_tensor_type(self):
            return LazyTensorImpl

    return LazyOpImpl()


def single_dispatch_lazy_tensor(func: Callable):
    """
    Make a function accept LazyTensor as input as its 1st argument.

    By calling `.register` on the returned callable, you can register
    the function with a LazyTensor type to dispatch the function. If none
    of the registered types match, the function will evalute the LazyTensor
    immediately.

    By calling `.register`, you can also register non-LazyTensor types,
    just like using the ordinary `functools.singledispatch`.

    Only unevaluated LazyTensor will be dispatched. Evaluated LazyTensor
    will be dispatched as normal tensors.

    If you want to dispatch on the second and subsequent arguments,
    create a closure.

    Example:
    ```
    @make_lazy_op
    def foo(x):
        return torch.sqrt(x)

    @single_dispatch_lazy_tensor
    def bar(y):
        return y ** 2

    # You can optionally fuse `sqrt(x) ** 2` to be `x`:
    @bar.register
    def _(y: foo.lazy_tensor_type()):
        return y.kwargs['x']
    ```

    z = bar(foo(x))  # z == x
    """

    dispatcher = functools.singledispatch(func)

    # functools.singledispatch will match subclasses first, if not found, it
    # will fallback to the base class `LazyTensor`, will we evaluate immedately
    @dispatcher.register
    def _(first_arg: LazyTensor, *args, **kwargs):
        return dispatcher(eval_lazy(first_arg), *args, **kwargs)

    old_register = dispatcher.register

    @functools.wraps(old_register)
    def register_wrapper(dispatched_func):
        @functools.wraps(dispatched_func)
        def dispatched_wrapper(first_arg, *args, **kwargs):
            if isinstance(first_arg, LazyTensor) and first_arg.is_evaluated():
                return func(first_arg.evaluate(), *args, **kwargs)
            else:
                return dispatched_func(first_arg, *args, **kwargs)

        return old_register(dispatched_wrapper)

    dispatcher.register = register_wrapper

    return dispatcher
