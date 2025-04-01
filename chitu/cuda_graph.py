from typing import Callable, Sequence, Mapping, Any, Optional
import functools
import torch

from chitu.static_tensor import StaticTensor


def make_dispatched_graphed_callables(
    f: Optional[Callable] = None,
    *,
    args_max_nelem: Sequence[int],
    kwargs_max_nelem: Mapping[str, int],
    output_max_nelem_callback: Callable[[int], int],
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
            enable=enable,
        )

    if enable:

        graph_dict: Dict[Any, torch.cuda.CUDAGraph] = {}
        cuda_graph_pool = None

        args_static_tensors: Optinoal[List[StaticTensor]] = None
        kwargs_static_tensors: Optional[Dict[str, StaticTensor]] = None
        output_static_tensor: Optional[StaticTensor] = None

        output_shape_dict = {}
        output_dtype_dict = {}
        output_device_dict = {}

        def new_callable(key: Any, *args, **kwargs):
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
                sample_output = f(*args, **kwargs)
                output_shape_dict[key] = sample_output.shape
                output_dtype_dict[key] = sample_output.dtype
                output_device_dict[key] = sample_output.device

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
                with torch.cuda.graph(graph_dict[key], pool=cuda_graph_pool):
                    output = f(
                        *[static_tensor.get() for static_tensor in args_static_tensors],
                        **{
                            k: static_tensor.get()
                            for k, static_tensor in kwargs_static_tensors.items()
                        },
                    )
                    output_static_tensor.set(output)
                if cuda_graph_pool is None:
                    cuda_graph_pool = graph_dict[key].pool()

            else:
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
                graph_dict[key].replay()

            return output_static_tensor.get()

    else:  # not enable

        def new_callable(key: Any, *args, **kwargs):
            return f(*args, **kwargs)

    return functools.update_wrapper(new_callable, f)
