# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Sequence, Any
from typing_extensions import final

import torch

NATIVE_LAYOUT_TENSOR_PROPERTY_NAME = "native_layout"


@dataclass
class NativeLayoutTensor:
    """
    Base class for a tensor store in a different layout than its mathematical representation.

    Inherit from this class to implement specific layouts.
    """

    plain_shape: torch.Size | Sequence[int]
    """
    The shape of the tensor in its mathematical representation.
    """

    layout_tensor: torch.Tensor
    """
    The tensor stored in a different layout.
    """

    @classmethod
    def check_tensor(cls, tensor: torch.Tensor):
        layout_type = getattr(tensor, NATIVE_LAYOUT_TENSOR_PROPERTY_NAME, None)
        try:
            return issubclass(layout_type, cls)
        except TypeError:  # layout_type is None or not a class
            return False

    @classmethod
    def convert_from(
        cls, plain_tensor: Any, *subclass_args, **subclass_kwargs
    ) -> "NativeLayoutTensor":
        """
        Create a NativeLayoutTensor from a tensor in a plain layout or other layouts.

        The layout of the input tensor is dependent on its type.

        Override this method to implement specific layouts, or you can safely ignore this method if you
        only interact with tensors with a specifc layout.
        """

        raise NotImplementedError(
            f"Unable to convert from {type(plain_tensor)} (with args {subclass_args} and "
            f"kwargs {subclass_kwargs}) to NativeLayoutTensor {cls}"
        )

    @staticmethod
    def _unwrap_tensor(tensor):
        """Return (plain_shape, layout_tensor) from either a raw tensor
        or a NativeLayoutTensor-like object.
        """
        if hasattr(tensor, "shape"):
            return tensor.shape, tensor
        # NativeLayoutTensor object
        return tensor.plain_shape, tensor.layout_tensor

    @final
    def convert_to(self, out_type):
        """
        Convert the layout tensor to a tensor in the specified type.

        Don't override this method. You only need to override `convert_from` and `convert_to_plain`.
        """

        if out_type is torch.Tensor:
            return self.convert_to_plain()
        elif isinstance(out_type, NativeLayoutTensor):
            return out_type.convert_from(self)
        else:
            raise TypeError(
                f"A NativeLayoutTensor can only convert to a plain torch.Tensor or another "
                f"NativeLayoutTensor, but got {out_type}."
            )

    def convert_to_plain(self) -> torch.Tensor:
        """
        Convert the layout tensor to its mathematical representation.

        Override this method to implement specific layouts, or you can safely ignore this method if you
        only interact with tensors with a specifc layout.
        """

        raise NotImplementedError()

    @property
    def device(self):
        return self.layout_tensor.device

    @property
    def dtype(self):
        return self.layout_tensor.dtype

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(device, torch.device):
            return type(self)(
                plain_shape=self.plain_shape,
                layout_tensor=self.layout_tensor.to(device),
            )
        else:
            raise ValueError(
                f"NativeLayoutTensor.to only support moving to a device, but got {type(device)}"
            )

    def is_contiguous(self):
        return self.layout_tensor.is_contiguous()
