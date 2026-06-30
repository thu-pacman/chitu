# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Sequence, Any, Protocol, runtime_checkable
from typing_extensions import final

import torch


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
        if not isinstance(tensor, TensorWithNativeLayout):
            return False
        template = tensor.native_layout
        if not isinstance(template, NativeLayoutTemplate):
            return False
        return issubclass(template.layout_cls, cls)

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


@dataclass
class NativeLayoutTemplate:
    """A node in a chain of native-layout conversions.

    Each template represents one conversion step.  Templates are linked via
    ``previous`` so that a parameter can pass through multiple layouts
    (e.g. checkpoint-format → intermediate → final native layout).
    """

    layout_cls: type[NativeLayoutTensor]
    """Target layout class."""

    plain_shape: torch.Size
    """Mathematical shape (from the first ``apply_native_layout`` call)."""

    args: tuple
    """Positional args forwarded to ``layout_cls.convert_from``."""

    kwargs: dict
    """Keyword args forwarded to ``layout_cls.convert_from``."""

    state_dict_convert: bool
    """Whether this layout participates in state-dict conversion hooks."""

    state_dict_shape: torch.Size
    """Shape of the parameter as it appears in ``state_dict``."""

    state_dict_dtype: torch.dtype
    """Dtype of the parameter as it appears in ``state_dict``."""

    previous: "NativeLayoutTemplate | None" = None
    """Previous template in the chain (for chained registrations)."""

    def build(self, src: torch.Tensor) -> NativeLayoutTensor:
        """Wrap a tensor in this layout's type (no chaining, no conversion).

        Constructs the ``layout_cls`` directly, using *src* as the
        ``layout_tensor``.
        """
        return self.layout_cls(
            plain_shape=self.plain_shape,
            layout_tensor=src,
            *self.args,
            **self.kwargs,
        )

    def convert(self, src: torch.Tensor | NativeLayoutTensor) -> NativeLayoutTensor:
        """Run this conversion step.

        If :attr:`previous` is set, the source is first wrapped via
        ``previous.build`` before being passed to this template's
        ``layout_cls.convert_from``.
        """
        if self.previous is not None:
            src = self.previous.build(src)
        return self.layout_cls.convert_from(src, *self.args, **self.kwargs)


@runtime_checkable
class TensorWithNativeLayout(Protocol):
    native_layout: NativeLayoutTemplate
