# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Sequence, Mapping, Any, Optional
from logging import getLogger
import functools
import gc
import torch

from chitu.static_tensor import StaticTensor
from chitu.device_type import is_ascend

logger = getLogger(__name__)

_is_warming_up_before_cuda_graph_capture = False
_currently_capturing_graph_object = None
_post_hook_per_graph_object: dict[torch.cuda.CUDAGraph, list[Callable[[], None]]] = {}


def weak_ref_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a non-owning alias; its storage must be kept valid by its owner."""
    if tensor.numel() == 0:
        return tensor
    if is_ascend():
        from torch_npu._C import _weak_ref_tensor

        return _weak_ref_tensor(tensor)
    if tensor.device.type == "cuda":
        import chitu_backend

        return chitu_backend.weak_ref_tensor(tensor)
    return tensor


def is_warming_up_before_cuda_graph_capture():
    return _is_warming_up_before_cuda_graph_capture


def is_warming_up_or_cuda_graph_capture():
    return (
        _is_warming_up_before_cuda_graph_capture
        or torch.cuda.is_current_stream_capturing()
    )


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
    before_capture_callback: Optional[Callable[[], None]] = None,
    before_replay_callback: Optional[Callable[[Any], None]] = None,
    enable: bool = True,
    graph_pool: Any = None,
    generators: Optional[Sequence[torch.Generator]] = None,
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
        before_replay_callback: An optional `(graph) -> None` callback function to be called before each
            graph replay. Note that this callback is not invoked before warming-up runs, or before graph
            capturing.
        enable: If False, do nothing but only add the `key` argument.
        graph_pool: An optional CUDA Graph pool handle. Graphs sharing a pool
            must be replayed in a capture-compatible order and never concurrently.
        generators: Optional CUDA generators registered on every captured graph
            via `torch.cuda.CUDAGraph.register_generator_state`. Their RNG state
            is part of the graph and advances on every replay, so the captured
            function may draw random numbers without an external noise input.

    Returns:
        The wrapped function, which has an additional first argument `key` to dispatch different graphs.
    """

    if f is None:
        return functools.partial(
            make_dispatched_graphed_callables,
            args_max_nelem=args_max_nelem,
            kwargs_max_nelem=kwargs_max_nelem,
            before_capture_callback=before_capture_callback,
            before_replay_callback=before_replay_callback,
            enable=enable,
            graph_pool=graph_pool,
            generators=generators,
        )

    if enable:

        graph_dict: dict[Any, torch.cuda.CUDAGraph] = {}
        cuda_graph_pool = graph_pool

        args_static_tensors: Optional[Sequence[StaticTensor]] = None
        kwargs_static_tensors: Optional[dict[str, StaticTensor]] = None
        output_dict: dict[Any, torch.Tensor] = {}

        def new_callable(key: Any, *args, **kwargs):
            global _is_warming_up_before_cuda_graph_capture
            global _currently_capturing_graph_object

            nonlocal graph_dict
            nonlocal cuda_graph_pool
            nonlocal args_static_tensors
            nonlocal kwargs_static_tensors

            is_new_graph = key not in graph_dict
            if is_new_graph:
                # Warmup
                logger.debug(f"Warming-up before capturing new graph with key {key}")
                assert _is_warming_up_before_cuda_graph_capture is False
                try:
                    _is_warming_up_before_cuda_graph_capture = True
                    sample_output = f(*args, **kwargs)
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

                # before capture callback
                # NOTE: For some attn_backends like FlashMLA, the actual intiailization of
                # metadata occurs during the first execution of the attn kernel. In such
                # cases, the graph capture after the warmup fail to capture the metadata
                # intialization. Therefore an additional before_capture_callback is necessary
                if before_capture_callback is not None:
                    before_capture_callback()

                # Capture the graph
                logger.debug(f"Capturing new graph with key {key}")
                graph_dict[key] = torch.cuda.CUDAGraph()
                for generator in generators or ():
                    graph_dict[key].register_generator_state(generator)
                gc.disable()  # Disable GC to prevent mid-capture tensor destruction
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
                finally:
                    _currently_capturing_graph_object = None
                    gc.enable()  # Always re-enable GC
                    gc.collect()  # Clean up anything that was delayed
                output_dict[key] = weak_ref_tensor(output)
                if cuda_graph_pool is None:
                    cuda_graph_pool = graph_dict[key].pool()

            else:
                logger.debug(f"Replaying graph with key {key}")
                assert args_static_tensors is not None
                assert kwargs_static_tensors is not None
                for static_tensor, arg in zip(args_static_tensors, args):
                    static_tensor.set(arg)
                for k in kwargs:
                    kwargs_static_tensors[k].set(kwargs[k])
                if before_replay_callback is not None:
                    before_replay_callback(graph_dict[key])
                graph_dict[key].replay()

            for hooks in _post_hook_per_graph_object.get(graph_dict[key], []):
                hooks()

            # Copy outside the graph so callers own their output even when a later
            # replay overwrites the graph output (e.g. during a PP isend).
            # The first call uses the warmup output; later calls use the graph output.
            return sample_output if is_new_graph else output_dict[key].clone()

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

    When the attribute named by enable_flag_name is False, use a regular tensor
    cache when CUDA graph is disabled, or compute on the fly during capture.

    Otherwise, this decorator is intended for:
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

    In order to support 3, the cache will not be marked up to date during the warming up phase before a graph
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
        regular_tensor_name = f"{static_tensor_name}_regular_cache"

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
                if torch.cuda.is_current_stream_capturing():
                    setattr(self, up_to_date_flag_name, False)
                    return fn(self, *args, **kwargs)
                if not getattr(self, up_to_date_flag_name):
                    tensor = fn(self, *args, **kwargs)
                    setattr(self, regular_tensor_name, tensor)
                    setattr(self, up_to_date_flag_name, True)
                return getattr(self, regular_tensor_name)

        return property(wrapper)

    return decorator
