from typing import Optional
import torch


class StaticTensor:
    """
    A tensor holder that is allocated only once

    This is useful for reducing the memory allocator overhead, and is required
    for input and output tensors of a CUDA graph.

    Initialize a `StaticTensor` with an initial tensor and/or the maximum buffer
    size (in number of elements). Then, use `.set()` and `.get()` to set and get
    the tensor. The tensor will NOT be reallocated during `.set()`.

    Args:
        tensor: The initial tensor to be stored. Defaults to an empty tensor.
        max_nelem: The maximum number of elements that the tensor can store.
            Defaults to the number of elements in the initial tensor.
        dtype: The data type of the tensor. Defaults to the data type of the
            initial tensor.
        device: The device of the tensor. Defaults to the device of the initial
            tensor.
    """

    def __init__(
        self,
        tensor: Optional[torch.Tensor] = None,
        *,
        max_nelem: Optional[int] = None,
        dtype: Optional[torch.device] = None,
        device: Optional[torch.device] = None,
    ):
        if max_nelem is None:
            if tensor is None:
                raise ValueError(f"max_nelem must be specified if tensor is None")
            max_nelem = tensor.numel()
        if dtype is None:
            if tensor is None:
                raise ValueError(f"dtype must be specified if tensor is None")
            dtype = tensor.dtype
        if device is None:
            if tensor is None:
                raise ValueError(f"device must be specified if tensor is None")
            device = tensor.device

        self._buffer = torch.empty(max_nelem, dtype=dtype, device=device)
        self._cur_nelem = 0
        self._cur_shape = (0,)

        if tensor is not None:
            self.set(tensor)

    def set(self, tensor: torch.Tensor):
        """
        Set the tensor to be stored

        Args:
            tensor: The tensor to be stored
        """
        if tensor.numel() > self._buffer.numel():
            raise ValueError(
                f"The assigned tensor ({tensor.shape}) cannot be larger than the buffer ({self._buffer.shape})"
            )
        if tensor.dtype != self._buffer.dtype:
            raise ValueError(
                f"The assigned tensor ({tensor.dtype}) must have the same data type as the buffer ({self._buffer.dtype})"
            )
        if tensor.device != self._buffer.device:
            raise ValueError(
                f"The assigned tensor ({tensor.device}) must have the same device as the buffer ({self._buffer.device})"
            )
        self._buffer[: tensor.numel()].copy_(tensor.flatten())
        self._cur_nelem = tensor.numel()
        self._cur_shape = tensor.shape

    def get(self) -> torch.Tensor:
        """
        Get the stored tensor

        Returns:
            The stored tensor
        """
        return self._buffer[: self._cur_nelem].reshape(self._cur_shape)
