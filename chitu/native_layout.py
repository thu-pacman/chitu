from typing import Union, Sequence, Mapping, Any
from typing_extensions import override, final
from dataclasses import dataclass
import functools
import torch

from chitu.utils import try_import_opt_dep

chitu_backend, has_chitu_backend = try_import_opt_dep("chitu_backend", "chitu_backend")


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
    key: str,
    native_layout_tensor_class: type,
    allow_missing: bool = False,
    *other_args,
    **other_kwargs,
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

    `enable_native_layout_weight` also supports preprocessing the weight from another layout. To do this,
    you can set `_{key}_layout_class`, `_{key}_plain_shape`, `_{key}_layout_args` (optional) and
    `_{key}_layout_kwargs` (optional) attributes in the Module class so the layout can be recognized. For
    example:

    ```
    class YourModule(enable_native_layout_weight("weight", MyNativeLayout), torch.nn.Module):
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.randn(2, 5, 2, 5, 2))  # Native shape
            self._weight_layout_class = MyNativeLayout
            self._weight_plain_shape = (10, 20)  # Mathematical shape
    ```

    Args:
        key: The module will process `self.{key}` for its weight `Parameter`, and the native layout tensor
             getter will be named after `self.get_native_layout_{key}`.
        native_layout_tensor_class: A subclass of `NativeLayoutTensor` representing the layout.
        allow_missing: If True, elegantly skip the preprocessing if `self.{key}` does not exists.
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

            def _get_native_layout_tensor():
                layout_args = (
                    self.__getattribute__(f"_{key}_layout_args")
                    if hasattr(self, f"_{key}_layout_args")
                    else ()
                )
                layout_kwargs = (
                    self.__getattribute__(f"_{key}_layout_kwargs")
                    if hasattr(self, f"_{key}_layout_kwargs")
                    else {}
                )
                return self.__getattribute__(f"_{key}_layout_class")(
                    self.__getattribute__(f"_{key}_plain_shape"),
                    self.__getattr__(key).data,
                    *layout_args,
                    **layout_kwargs,
                )

            def _preprocess_layout(module, incompatible_keys):
                if allow_missing and not hasattr(module, key):
                    return
                if hasattr(module, f"_{key}_layout_class"):
                    old_tensor = _get_native_layout_tensor()
                else:
                    old_tensor = module.__getattr__(key)
                new_tensor = native_layout_tensor_class.convert_from(
                    old_tensor,
                    *other_args,
                    **other_kwargs,
                )
                module.__setattr__(f"_{key}_layout_class", native_layout_tensor_class)
                module.__setattr__(f"_{key}_plain_shape", new_tensor.plain_shape)
                module.__setattr__(f"_{key}_layout_args", other_args)
                module.__setattr__(f"_{key}_layout_kwargs", other_kwargs)
                module.__getattr__(key).data = new_tensor.layout_tensor

            self.register_load_state_dict_post_hook(_preprocess_layout)
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


@dataclass
class Packed4BitWeightAlongK(NativeLayoutTensor):
    """
    Int4 or float4 weight, where every two elements `k_stride` elements away in the K dimension
    are packed into a single uint8.

    `k_stride=1` is a special case, which means packing contiguously along the K dimension.
    """

    k_stride: int = 1

    @classmethod
    @override
    def convert_from(
        cls, tensor: "Packed4BitWeightAlongK", k_stride: int = 1
    ) -> "Packed4BitWeightAlongK":
        # NOTE: @functools.singledispatchmethod has a bug in Python 3.8
        # (https://stackoverflow.com/questions/62696796/singledispatchmethod-and-class-method-decorators-in-python-3-8)
        # Use `if` for now

        if isinstance(tensor, Packed4BitWeightAlongK):
            if tensor.k_stride == k_stride:
                return tensor
            k = tensor.plain_shape[-1]
            assert k % (2 * tensor.k_stride) == 0
            assert k % (2 * k_stride) == 0
            if has_chitu_backend and tensor.k_stride == 1 and k_stride == 64:
                device = tensor.layout_tensor.device
                weight = chitu_backend.weight_layout_change(
                    tensor.layout_tensor.cuda()
                ).to(device)
            else:
                weight = tensor.layout_tensor.view(
                    -1, k // (2 * tensor.k_stride), 1, tensor.k_stride
                ).view(torch.uint8)
                weight = torch.cat([weight & 0x0F, weight >> 4], dim=-2)
                weight = weight.view(-1, k // (2 * k_stride), 2, k_stride)
                weight = weight[..., 0, :] + (weight[..., 1, :] << 4)
            weight = weight.view(*tensor.plain_shape[:-1], k // 2).contiguous()
            return cls(
                plain_shape=tensor.plain_shape,
                layout_tensor=weight,
                k_stride=k_stride,
            )

        else:
            raise TypeError(
                f"Cannot convert from {type(tensor)} to Packed4BitWeightAlongK"
            )

    def __getitem__(self, index):
        """
        Indexing a Packed4BitWeightAlongK is safe is the K dimension is untouched. In such a
        case, this function returns a new Packed4BitWeightAlongK with the same layout.
        """
        if not isinstance(index, int):
            raise NotImplementedError(
                f"Indexing {type(self)} with {type(index)} is not supported."
            )
        if len(self.plain_shape) <= 1:
            raise ValueError(
                "Cannot index a Packed4BitWeightAlongK tensor's K dimension."
            )
        return Packed4BitWeightAlongK(
            self.plain_shape[1:], self.layout_tensor[index], k_stride=self.k_stride
        )


@dataclass
class Packed4BitWeightNPUNative(NativeLayoutTensor):
    @classmethod
    @override
    def convert_from(
        cls, tensor: Packed4BitWeightAlongK
    ) -> "Packed4BitWeightNPUNative":
        # NOTE: @functools.singledispatchmethod has a bug in Python 3.8
        # (https://stackoverflow.com/questions/62696796/singledispatchmethod-and-class-method-decorators-in-python-3-8)
        # Use `if` for now

        if isinstance(tensor, Packed4BitWeightAlongK) and tensor.k_stride == 1:
            return cls(
                plain_shape=tensor.plain_shape,
                layout_tensor=cls._repack_weight(tensor.layout_tensor),
            )

        else:
            raise TypeError(
                f"Cannot convert from {type(tensor)} to Packed4BitWeightAlongK"
            )

    # 针对npu反量化+矩阵乘融合算子设计
    @classmethod
    def _repack_weight(cls, weight):
        old_device = weight.device
        old_shape = weight.shape

        weight = weight.to(device="npu")
        tmp_weight = weight.to(torch.int16)
        tmp_weight = ((tmp_weight & 0x00F0) << 4) | (tmp_weight & 0x000F)
        shape = list(tmp_weight.shape)
        shape[-2] = shape[-1] // 2
        shape[-1] = shape[-2] * 2
        new_weight = tmp_weight.view(torch.uint8)
        new_weight = new_weight.transpose(-2, -1).contiguous()
        new_weight = new_weight.view(torch.int16)
        new_weight = ((new_weight & 0x0F00) >> 4) | (new_weight & 0x000F)
        weight = new_weight.to(torch.uint8).unsqueeze(0)

        weight_shape = weight.shape
        assert weight_shape[-2] % 64 == 0
        assert weight_shape[-1] % 128 == 0
        tmp_weight = weight.reshape(
            weight_shape[-3] * weight_shape[-2] // 64,
            4,
            2,
            8,
            weight_shape[-1] // 128,
            8,
            4,
            4,
        )
        new_weight = tmp_weight.permute(0, 2, 1, 5, 4, 6, 3, 7).contiguous()
        return new_weight.reshape(old_shape).to(old_device)

    def __getitem__(self, index):
        """
        Indexing a Packed4BitWeightNPUNative is safe is the last 2 dimensions are untouched.
        In such a case, this function returns a new Packed4BitWeightNPUNative with the same
        layout.
        """
        if not isinstance(index, int):
            raise NotImplementedError(
                f"Indexing {type(self)} with {type(index)} is not supported."
            )
        if len(self.plain_shape) <= 2:
            raise ValueError(
                "Cannot index a Packed4BitWeightAlongK tensor's last 2 dimensions."
            )
        return Packed4BitWeightAlongK(self.plain_shape[1:], self.layout_tensor[index])
