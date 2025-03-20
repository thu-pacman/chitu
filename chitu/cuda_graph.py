from typing import Callable, Sequence, Mapping, Any, Optional
import functools
import torch

from chitu.static_tensor import StaticTensor


def make_dispatched_graphed_callables(
    f: Optional[Callable] = None,
    *,
    sample_args: Sequence[torch.Tensor],
    sample_kwargs: Mapping[str, torch.Tensor],
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
        sample_args: The sample positional arguments for capturing the graph.
        sample_kwargs: The sample keyword arguments for capturing the graph.
        args_max_nelem: The maximum number of elements in the positional arguments, used to hold inputs
            in shared static tensors.
        kwargs_max_nelem: The maximum number of elements in the keyword arguments, used to hold inputs
            in shared static tensors.
        output_max_nelem: A `(sample_nelem) -> max_nelem` callback to return the maximum number of elements
            in the output tensor, used to hold outputs in shared static tensors.
        enable: If False, do nothing but only add the `key` argument.

    Returns:
        The wrapped function, which has an additional first argument `key` to dispatch different graphs.
    """

    if f is None:
        return functools.partial(
            make_dispatched_graphed_callables,
            sample_args=sample_args,
            sample_kwargs=sample_kwargs,
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

        def new_callable(key: Any, *args, **kwargs):
            nonlocal graph_dict
            nonlocal cuda_graph_pool
            nonlocal args_static_tensors
            nonlocal kwargs_static_tensors
            nonlocal output_static_tensor

            if key not in graph_dict:
                # Warmup
                sample_output = f(*sample_args, **sample_kwargs)

                # Allocate static tensors
                if args_static_tensors is None:
                    args_static_tensors = [
                        StaticTensor(sample, max_nelem=max_nelem)
                        for sample, max_nelem in zip(sample_args, args_max_nelem)
                    ]
                if kwargs_static_tensors is None:
                    kwargs_static_tensors = {}
                    for k in sample_kwargs:
                        kwargs_static_tensors[k] = StaticTensor(
                            sample_kwargs[k], max_nelem=kwargs_max_nelem[k]
                        )
                if output_static_tensor is None:
                    output_static_tensor = StaticTensor(
                        sample_output,
                        max_nelem=output_max_nelem_callback(sample_output.numel()),
                    )

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

            graph_dict[key].replay()
            return output_static_tensor.get()

    else:  # not enable

        def new_callable(key: Any, *args, **kwargs):
            return f(*args, **kwargs)

    return functools.update_wrapper(new_callable, f)
