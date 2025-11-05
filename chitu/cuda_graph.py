# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Sequence, Mapping, Any, Optional
import functools
import torch

from chitu.static_tensor import StaticTensor
from chitu.device_type import is_ascend

_is_warming_up_before_cuda_graph_capture = False
_currently_capturing_graph_object = None
_post_hook_per_graph_object: dict[torch.cuda.CUDAGraph, list[Callable[[], None]]] = {}


def is_warming_up_before_cuda_graph_capture():
    return _is_warming_up_before_cuda_graph_capture


def add_post_hook_for_currently_capturing_graph_object(hook: Callable[[], None]):
    assert isinstance(_currently_capturing_graph_object, torch.cuda.CUDAGraph)
    if _currently_capturing_graph_object not in _post_hook_per_graph_object:
        _post_hook_per_graph_object[_currently_capturing_graph_object] = []
    _post_hook_per_graph_object[_currently_capturing_graph_object].append(hook)


def make_dispatched_graphed_callables(
    f: Optional[Callable] = None,
    *,
    args_max_nelem: Sequence[int],
    kwargs_max_nelem: Mapping[str, int],
    output_max_nelem_callback: Callable[[Any, int], int],
    before_replay_callback: Optional[Callable[[Any], None]] = None,
    enable: bool = True,
) -> Callable:
    """
    Make a callable to run with CUDA graph but capature different graphs when `key` changes.

    Args:
        f: The function to wrap. Currently all the inputs should be tensors, and there should only be one
            output which is a tensor. If None, return a partial function as an decorator.
        args_max_nelem: The maximum number of elements in the positional arguments, used to hold inputs
            in shared static tensors.
        kwargs_max_nelem: The maximum number of elements in the keyword arguments, used to hold inputs
            in shared static tensors.
        output_max_nelem: A `(key, sample_nelem) -> max_nelem` callback to return the maximum number of
            elements in the output tensor, used to hold outputs in shared static tensors.
        before_replay_callback: An optional `(graph) -> None` callback function to be called before each
            graph replay. Note that this callback is not invoked before warming-up runs, or before graph
            capturing.
        enable: If False, do nothing but only add the `key` argument.

    Returns:
        The wrapped function, which has an additional first argument `key` to dispatch different graphs.
    """

    if f is None:
        return functools.partial(
            make_dispatched_graphed_callables,
            args_max_nelem=args_max_nelem,
            kwargs_max_nelem=kwargs_max_nelem,
            output_max_nelem_callback=output_max_nelem_callback,
            before_replay_callback=before_replay_callback,
            enable=enable,
        )

    if enable:

        graph_dict: dict[Any, torch.cuda.CUDAGraph] = {}
        cuda_graph_pool = None

        args_static_tensors: Optional[Sequence[StaticTensor]] = None
        kwargs_static_tensors: Optional[dict[str, StaticTensor]] = None
        output_static_tensor: Optional[StaticTensor] = None

        output_shape_dict = {}
        output_dtype_dict = {}
        output_device_dict = {}

        def new_callable(key: Any, *args, **kwargs):
            global _is_warming_up_before_cuda_graph_capture
            global _currently_capturing_graph_object

            nonlocal graph_dict
            nonlocal cuda_graph_pool
            nonlocal args_static_tensors
            nonlocal kwargs_static_tensors
            nonlocal output_static_tensor
            nonlocal output_shape_dict
            nonlocal output_dtype_dict
            nonlocal output_device_dict

            if key not in graph_dict:
                # Warmup
                assert _is_warming_up_before_cuda_graph_capture is False
                try:
                    _is_warming_up_before_cuda_graph_capture = True
                    sample_output = f(*args, **kwargs)
                    output_shape_dict[key] = sample_output.shape
                    output_dtype_dict[key] = sample_output.dtype
                    output_device_dict[key] = sample_output.device
                finally:
                    _is_warming_up_before_cuda_graph_capture = False

                # Allocate static tensors
                if args_static_tensors is None:
                    args_static_tensors = [
                        StaticTensor(arg, max_nelem=max_nelem)
                        for arg, max_nelem in zip(args, args_max_nelem)
                    ]
                else:
                    for static_tensor, arg in zip(args_static_tensors, args):
                        static_tensor.set(arg)
                if kwargs_static_tensors is None:
                    kwargs_static_tensors = {}
                    for k in kwargs:
                        kwargs_static_tensors[k] = StaticTensor(
                            kwargs[k], max_nelem=kwargs_max_nelem[k]
                        )
                else:
                    for k in kwargs:
                        kwargs_static_tensors[k].set(kwargs[k])
                if output_static_tensor is None:
                    output_static_tensor = StaticTensor(
                        sample_output,
                        max_nelem=output_max_nelem_callback(key, sample_output.numel()),
                    )
                else:
                    output_static_tensor.set(sample_output)

                # Capture the graph
                graph_dict[key] = torch.cuda.CUDAGraph()
                try:
                    _currently_capturing_graph_object = graph_dict[key]
                    if is_ascend():
                        capturing_stream = torch.npu.Stream(device=sample_output.device)
                        capturing_stream.wait_stream(torch.npu.current_stream())
                        with torch.npu.stream(capturing_stream):
                            with torch.cuda.graph(
                                graph_dict[key],
                                pool=cuda_graph_pool,
                                auto_dispatch_capture=True,
                            ):
                                output = f(
                                    *[
                                        static_tensor.get()
                                        for static_tensor in args_static_tensors
                                    ],
                                    **{
                                        k: static_tensor.get()
                                        for k, static_tensor in kwargs_static_tensors.items()
                                    },
                                )
                                output_static_tensor.set(output)
                    else:
                        with torch.cuda.graph(graph_dict[key], pool=cuda_graph_pool):
                            output = f(
                                *[
                                    static_tensor.get()
                                    for static_tensor in args_static_tensors
                                ],
                                **{
                                    k: static_tensor.get()
                                    for k, static_tensor in kwargs_static_tensors.items()
                                },
                            )
                            output_static_tensor.set(output)
                finally:
                    _currently_capturing_graph_object = None
                if cuda_graph_pool is None:
                    cuda_graph_pool = graph_dict[key].pool()

            else:
                assert args_static_tensors is not None
                assert kwargs_static_tensors is not None
                assert output_static_tensor is not None
                for static_tensor, arg in zip(args_static_tensors, args):
                    static_tensor.set(arg)
                for k in kwargs:
                    kwargs_static_tensors[k].set(kwargs[k])
                output_static_tensor.set(
                    torch.empty(
                        output_shape_dict[key],
                        dtype=output_dtype_dict[key],
                        device=output_device_dict[key],
                    )
                )
                if before_replay_callback is not None:
                    before_replay_callback(graph_dict[key])
                graph_dict[key].replay()

            for hooks in _post_hook_per_graph_object.get(graph_dict[key], []):
                hooks()

            return output_static_tensor.get()

    else:  # not enable

        def new_callable(key: Any, *args, **kwargs):
            return f(*args, **kwargs)

    return functools.update_wrapper(new_callable, f)


def cuda_graph_safe_cached_property(
    static_tensor_name: str,
    up_to_date_flag_name: str,
    *,
    enable_flag_name: Optional[str] = None,
):
    """
    Similar to `functools.cached_property`, but the returned value is a tensor, and can be used
    within or without a CUDA graph.

    This decorator is intended for:
    1. caching a tensor and then reusing it without a CUDA graph.
    2. caching a tensor and then reusing it within a CUDA graph.
    3. caching a tensor before a CUDA graph, and then reusing it within a CUDA graph.
    4. caching a tensor within a CUDA graph, and then reusing it after the CUDA graph.

    Note that the reusing behaviour should be consistent for the same graph object, which means
    you can NOT sometimes do case 2 and sometimes do case 3. If you need to support such cases,
    please add a key to `make_dispatched_graphed_callables` and capture multiple graph objects.
    Currently this function is NOT checking the consistency, and the user should ensure it.

    In order to support 3 and 4, the tensor is stored in an external `StaticTensor` as inputs
    or outputs of the CUDA graph, instead of storing it directly in a property like `functools.cached_property`.

    In order to support 3, the cache will not be updated during the warming up phase before a graph
    capture. Otherwise, the value update will be incorrectly skipped in the graph before it is
    falsefully already "cached".

    Example usage:
    ```
    class A:
        def __init__(self):
            self._xxx_static_tensor = StaticTensor(...)  # Should be large enough to hold the tensor
            self._xxx_up_to_date = False

        @cuda_graph_safe_cached_property("_xxx_static_tensor", "_xxx_up_to_date")
        def xxx(self):
            return ...
    ```
    """

    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if enable_flag_name is None or getattr(self, enable_flag_name):
                static_tensor = getattr(self, static_tensor_name)
                assert isinstance(static_tensor, StaticTensor)
                assert isinstance(getattr(self, up_to_date_flag_name), bool)
                if not getattr(self, up_to_date_flag_name):
                    tensor = fn(self, *args, **kwargs)
                    shape = tuple(tensor.shape)
                    static_tensor.set(tensor)
                    if not is_warming_up_before_cuda_graph_capture():
                        setattr(self, up_to_date_flag_name, True)
                    if torch.cuda.is_current_stream_capturing():

                        def post_hook():
                            setattr(self, up_to_date_flag_name, True)
                            getattr(self, static_tensor_name).set_shape(shape)

                        add_post_hook_for_currently_capturing_graph_object(post_hook)
                return static_tensor.get()
            else:
                return fn(self, *args, **kwargs)

        return property(wrapper)

    return decorator
