from typing import Union, Sequence, Mapping, Any
from typing_extensions import override, final
from dataclasses import dataclass
import functools
import torch


@dataclass
class NativeLayoutTensor:
    """
    Base class for a tensor store in a different layout than its mathematical representation.

    Inherit from this class to implement specific layouts.
    """

    plain_shape: Union[torch.Size, Sequence[int]]
    """
    The shape of the tensor in its mathematical representation.
    """

    layout_tensor: torch.Tensor
    """
    The tensor stored in a different layout.
    """

    @classmethod
    def convert_from(cls, plain_tensor: Any) -> "NativeLayoutTensor":
        """
        Create a NativeLayoutTensor from a tensor in a plain layout or other layouts.

        The layout of the input tensor is dependent on its type.

        Override this method to implement specific layouts, or you can safely ignore this method if you
        only interact with tensors with a specifc layout.
        """

        raise NotImplementedError()

    @final
    def convert_to(self, out_type):
        """
        Convert the layout tensor to a tensor in the specified type.

        Don't override this method. You only need to override `convert_from` and `convert_to_plain`.
        """

        if out_type is torch.Tensor:
            return self.conver_to_plain()
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


def enable_native_layout_weight(
    key: str, native_layout_tensor_class: type, *other_args, **other_kwargs
) -> type:
    """
    Return a mix-in class that can be inherit by a Module class, which will enable the Module class to
    preprocess its weight and use it in a native layout.

    Example usage:
    ```
    class YourModule(enable_native_layout_weight("weight", MyNativeLayout), torch.nn.Module):
        ...
    your_module = YourModule()
    ```
    , where `MyNativeLayout` is a subclass of `NativeLayout`.

    After `your_module` loads state dict, the weight will automatically be transformed into the layout of
    `MyNativeLayout`, still stored in the original `Parameter`. Besides, `your_module.get_native_layout_weight()`
    will be available for getting an `MyNativeLayout` instance for the weight.

    Args:
        key: The module will process `self.{key}` for its weight `Parameter`, and the native layout tensor
             getter will be named after `self.get_native_layout_{key}`.
        native_layout_tensor_class: A subclass of `NativeLayoutTensor` representing the layout.
        other_args: Other positional arguments passed to `NativeLayoutTensor`.
        other_kwargs: Other keyword arguments passed to `NativeLayoutTensor`.
    """

    class EnableNativeLayoutWeightMixIn:
        # NOTE: In Python, super().__init__ calls the next base class in the full inheritance graph
        # of the final class, so we can append a class to the base class, to make it act like a
        # further base class of the original base class.
        # See https://docs.python.org/3/tutorial/classes.html#multiple-inheritance

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # NOTE: Look up parameters (dynamically handled by torch.nn.Module) with __getattr__, but
            # look up real Python attributes with __getattribute__.

            def _preprocess_layout(module, incompatible_keys):
                new_tensor = native_layout_tensor_class.convert_from(
                    module.__getattr__(key).data,
                    *other_args,
                    **other_kwargs,
                )
                module.__setattr__(f"_{key}_plain_shape", new_tensor.plain_shape)
                module.__getattr__(key).data = new_tensor.layout_tensor

            self.register_load_state_dict_post_hook(_preprocess_layout)

            def _get_native_layout_tensor():
                return native_layout_tensor_class(
                    self.__getattribute__(f"_{key}_plain_shape"),
                    self.__getattr__(key).data,
                    *other_args,
                    **other_kwargs,
                )

            self.__setattr__(f"get_native_layout_{key}", _get_native_layout_tensor)

    return EnableNativeLayoutWeightMixIn


@dataclass
class Vector(NativeLayoutTensor):
    """
    Not in a special layout, but assert there is only one batch
    """

    @classmethod
    @override
    def convert_from(cls, tensor: torch.Tensor) -> "Vector":
        # NOTE: @functools.singledispatchmethod has a bug in Python 3.8
        # (https://stackoverflow.com/questions/62696796/singledispatchmethod-and-class-method-decorators-in-python-3-8)
        # Use `if` for now

        if isinstance(tensor, torch.Tensor):
            if tensor.numel() != tensor.shape[-1]:
                raise ValueError(
                    f"Vector expects a tensor with batch size equal to 1, but got {tensor.shape}"
                )
            return cls(
                plain_shape=tensor.shape,
                layout_tensor=tensor.view(tensor.shape[-1]),
            )

        else:
            raise TypeError(f"Cannot convert from {type(tensor)} to Vector")

    @override
    def convert_to_plain(self) -> torch.Tensor:
        return self.layout_tensor.view(self.plain_shape)


@dataclass
class BatchPaddedActivation(NativeLayoutTensor):
    """
    Considering all dimensions except the last one as batch dimensions, this layout padded the batch dimensions
    to the next multiple of `multiple_of`.
    """

    multiple_of: int

    @classmethod
    @override
    def convert_from(
        cls, tensor: torch.Tensor, multiple_of: int
    ) -> "BatchPaddedActivation":
        # NOTE: @functools.singledispatchmethod has a bug in Python 3.8
        # (https://stackoverflow.com/questions/62696796/singledispatchmethod-and-class-method-decorators-in-python-3-8)
        # Use `if` for now

        if isinstance(tensor, torch.Tensor):
            plain_shape = tensor.shape
            plain_batch_size = functools.reduce(lambda x, y: x * y, plain_shape[:-1], 1)
            padded_batch_size = (
                (plain_batch_size + multiple_of - 1) // multiple_of * multiple_of
            )
            padded_shape = [padded_batch_size, plain_shape[-1]]
            if padded_batch_size == plain_batch_size:
                layout_tensor = tensor.view(-1, plain_shape[-1])
            else:
                layout_tensor = torch.zeros(
                    padded_shape, dtype=tensor.dtype, device=tensor.device
                )
                layout_tensor[:plain_batch_size].copy_(tensor.view(-1, plain_shape[-1]))
            return cls(
                plain_shape=plain_shape,
                layout_tensor=layout_tensor,
                multiple_of=multiple_of,
            )

        else:
            raise TypeError(
                f"Cannot convert from {type(tensor)} to BatchPaddedActivation"
            )

    @override
    def convert_to_plain(self) -> torch.Tensor:
        plain_batch_size = functools.reduce(
            lambda x, y: x * y, self.plain_shape[:-1], 1
        )
        return self.layout_tensor[:plain_batch_size].view(self.plain_shape)
