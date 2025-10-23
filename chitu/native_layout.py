# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence, Any
from typing_extensions import override, final
from dataclasses import dataclass
import functools
import plum
import torch

from chitu.utils import try_import_platform_dep, try_import_and_setup_torch_npu

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
hygon_mixq_kernels, has_hygon = try_import_platform_dep("sugon_mixQ4_kernels")
hygon_w4a8_kernels, has_hygon_w4a8 = try_import_platform_dep("sugon_w4a8_kernels")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


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


def enable_native_layout_weight(
    key: str,
    native_layout_tensor_class: type,
    allow_missing: bool = False,
    *static_args,
    **static_kwargs,
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
        static_args: Other positional arguments passed to `NativeLayoutTensor`.
        static_kwargs: Other keyword arguments passed to `NativeLayoutTensor`.
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

                inst_args = getattr(module, f"_{key}_layout_args", None)
                inst_kwargs = getattr(module, f"_{key}_layout_kwargs", None)
                other_args = inst_args if inst_args is not None else static_args
                other_kwargs = inst_kwargs if inst_kwargs is not None else static_kwargs

                def _eval(p):
                    if callable(p):
                        try:
                            return p(module)  # e.g. lambda m: m.in_features
                        except TypeError:
                            return p()  # e.g. torch.get_default_dtype
                    return p

                other_args = tuple(_eval(a) for a in other_args)
                other_kwargs = {k: _eval(v) for k, v in other_kwargs.items()}

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
    @plum.dispatch
    def convert_from(cls, tensor: torch.Tensor) -> "Vector":
        if tensor.numel() != tensor.shape[-1]:
            raise ValueError(
                f"Vector expects a tensor with batch size equal to 1, but got {tensor.shape}"
            )
        return cls(
            plain_shape=tensor.shape,
            layout_tensor=tensor.view(tensor.shape[-1]),
        )

    @override
    def convert_to_plain(self) -> torch.Tensor:
        return self.layout_tensor.view(self.plain_shape)


@dataclass
class PermutedTensor(NativeLayoutTensor):
    """
    A contiguous tensor with dimensions permuted
    """

    perm: Sequence[int]

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls, tensor: torch.Tensor, *, perm: Sequence[int]
    ) -> "PermutedTensor":
        return cls(
            plain_shape=tensor.shape,
            layout_tensor=tensor.permute(*perm).contiguous(),
            perm=perm,
        )

    @override
    def convert_to_plain(self) -> torch.Tensor:
        return self.layout_tensor.permute(*self._get_inverse_perm())

    def _get_inverse_perm(self) -> Sequence[int]:
        return [self.perm.index(i) for i in range(len(self.perm))]


@dataclass
class BatchPaddedActivation(NativeLayoutTensor):
    """
    Considering all dimensions except the last one as batch dimensions, this layout padded the batch dimensions
    to the next multiple of `multiple_of`.
    """

    multiple_of: int

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls, tensor: torch.Tensor, *, multiple_of: int
    ) -> "BatchPaddedActivation":
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
    @plum.dispatch
    def convert_from(
        cls, tensor: "Packed4BitWeightAlongK", *, k_stride: int = 1
    ) -> "Packed4BitWeightAlongK":
        if tensor.k_stride == k_stride:
            return tensor
        k = tensor.plain_shape[-1]
        assert k % (2 * tensor.k_stride) == 0
        assert k % (2 * k_stride) == 0
        if has_chitu_backend and tensor.k_stride == 1 and k_stride == 64:
            device = tensor.layout_tensor.device
            weight = chitu_backend.weight_layout_change(tensor.layout_tensor.cuda()).to(
                device
            )
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

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls, tensor: "Packed4BitWeightQServe", *, k_stride: int = 1
    ) -> "Packed4BitWeightAlongK":
        assert len(tensor.plain_shape) == 2
        n, k = tensor.plain_shape

        # Unpack from qserve format
        assert n % 32 == 0
        assert k % 32 == 0
        weight = tensor.layout_tensor.view(n // 32, k // 32, 1, 8, 4, 2, 2, 1, 4).view(
            torch.uint8
        )
        weight = torch.stack([weight & 0x0F, weight >> 4], dim=0)
        weight = weight.permute(1, 0, 7, 4, 8, 2, 3, 6, 5, 9).contiguous().view(n, k)

        # Pack to Packed4BitWeightAlongK
        assert k % k_stride == 0
        weight = (
            weight.view(n, k // (2 * k_stride), 2, k_stride)
            .permute(2, 0, 1, 3)
            .contiguous()
        )
        weight = weight[0] + (weight[1] << 4)
        weight = weight.view(n, k // 2)
        return cls(
            plain_shape=tensor.plain_shape,
            layout_tensor=weight,
            k_stride=k_stride,
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
    @plum.dispatch
    def convert_from(
        cls, tensor: Packed4BitWeightAlongK
    ) -> "Packed4BitWeightNPUNative":
        if tensor.k_stride == 1:
            return cls(
                plain_shape=tensor.plain_shape,
                layout_tensor=cls._repack_weight(tensor.layout_tensor),
            )

        else:
            raise TypeError(
                f"Cannot convert from {type(tensor)} to Packed4BitWeightAlongK with k_stride={tensor.k_stride}"
            )

    # Designed for NPU de-quantization + matmul fused operator
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


@dataclass
class Packed4BitWeightQServe(NativeLayoutTensor):
    """
    Layout used in QServe

    See
    https://github.com/mit-han-lab/deepcompressor/blob/main/deepcompressor/backend/qserve/utils.py#L18
    for the format details.
    """

    pass


# See https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/appdevgapi/aclpythondevg_01_0914.html
# for the layout ID
ACL_FORMAT_FRACTAL_NZ = 29
ACL_FORMAT_ND = 2


@dataclass
class NpuFractalNzTensor(NativeLayoutTensor):
    """
    FRACTAL_NZ is a matmul-friendly layout used on Ascend.

    See https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0099.html

    NOTE: FRACTAL_NZ and FRACTAL_ZN are transpositions of each other, which means converting an INxOUT weight to FRACTAL_NZ
    is equivalent to converting an OUTxIN weight to FRACTAL_ZN. NpuFractalNzTensor just convert form what you pass to
    `convert_from`.

    Suppose you use NpuFractalNzTensor on an OUTxIN weight (common in torch.nn.Linear), equivalent to FRACTAL_ZN on an
    INxOUT weight, then the resulting tensor can be used like:
    - The pactice described in https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0099.html.
    - What is done by default for all torch.nn.Linear in torch_npu: https://github.com/Ascend/pytorch/blob/dd2acaaa361cc0937852a26dcbfb5ef604114664/torch_npu/utils/_module.py#L81.
    """

    @classmethod
    @override
    @plum.dispatch
    def convert_from(cls, tensor: torch.Tensor) -> "NpuFractalNzTensor":
        layout_tensor = torch_npu.npu_format_cast(
            tensor.npu().contiguous(), ACL_FORMAT_FRACTAL_NZ
        )

        # The assertion is necessary, because npu_format_cast may fail silently as a no-op on some environments
        assert torch_npu.get_npu_format(layout_tensor) == ACL_FORMAT_FRACTAL_NZ

        # NPU formats only live on NPU. Once we move to CPU and then move back, the format will disappear.
        # Therefore, we force this tensor to be on NPU.
        return cls(plain_shape=tensor.shape, layout_tensor=layout_tensor)

    @override
    def convert_to_plain(self) -> torch.Tensor:
        assert self.layout_tensor.device.type == "npu"
        return torch_npu.npu_format_cast(self.layout_tensor, ACL_FORMAT_ND)


@dataclass
class NpuFractalZnTensor(NativeLayoutTensor):
    """
    FRACTAL_ZN in https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0099.html

    NOTE: FRACTAL_NZ and FRACTAL_ZN are transpositions of each other, which means converting a INxOUT weight to FRACTAL_NZ
    is equivalent to converting an OUTxIN weight to FRACTAL_ZN. NpuFractalZnTensor just convert form what you pass to
    `convert_from`.

    torch_npu only provide an interface for FRACTAL_NZ, so NpuFractalZnTensor is implemented by first transposing the tenor
    and then converting it to FRACTAL_NZ.

    Suppose you use NpuFractalZnTensor on an OUTxIN weight (common in torch.nn.Linear), equivalent to FRACTAL_NZ on an
    INxOUT weight, then the resulting tensor can be used like:
    - What is done for MoE layers in OmniInfer: https://gitee.com/omniai/omniinfer/blob/745842ca9937ad445d56036af5289740287d6c11/omni/models/common/layers/moe/fused_moe/layer.py#L144.
    - What is required by npu_mla_prolog_v2: https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_mla_prolog_v2.md.
    """

    @classmethod
    @override
    @plum.dispatch
    def convert_from(cls, tensor: torch.Tensor) -> "NpuFractalZnTensor":
        layout_tensor = torch_npu.npu_format_cast(
            tensor.npu().transpose(-1, -2).contiguous(), ACL_FORMAT_FRACTAL_NZ
        )

        # The assertion is necessary, because npu_format_cast may fail silently as a no-op on some environments
        assert torch_npu.get_npu_format(layout_tensor) == ACL_FORMAT_FRACTAL_NZ

        # NPU formats only live on NPU. Once we move to CPU and then move back, the format will disappear.
        # Therefore, we force this tensor to be on NPU.
        return cls(plain_shape=tensor.shape, layout_tensor=layout_tensor)

    @override
    def convert_to_plain(self) -> torch.Tensor:
        assert self.layout_tensor.device.type == "npu"
        return torch_npu.npu_format_cast(self.layout_tensor, ACL_FORMAT_ND).transpose(
            -1, -2
        )


@dataclass
class ColumnOddEvenSeparatedTensor(NativeLayoutTensor):
    """
    A tensor with its last dimension's odd and even elements separated.

    Plain tensor: [1, 2, 3, 4, 5, 6]

    Layout tensor: [1, 3, 5, 2, 4, 6]
    """

    @classmethod
    @override
    @plum.dispatch
    def convert_from(cls, tensor: torch.Tensor) -> "ColumnOddEvenSeparatedTensor":
        return cls(
            plain_shape=tensor.shape,
            layout_tensor=tensor.view(*tensor.shape[:-1], tensor.shape[-1] // 2, 2)
            .transpose(-1, -2)
            .contiguous()
            .view(*tensor.shape),
        )

    @override
    def convert_to_plain(self) -> torch.Tensor:
        return (
            self.layout_tensor.view(
                *self.plain_shape[:-1], 2, self.plain_shape[-1] // 2
            )
            .transpose(-1, -2)
            .contiguous()
            .view(*self.plain_shape)
        )


@dataclass
class PartialColumnOddEvenSeparatedTensor(NativeLayoutTensor):
    """
    Similar to `ColumnOddEvenSeparatedTensor`, but only the `[begin_idx, end_idx)`
    part of the last dimension is separated.
    """

    begin_idx: int
    end_idx: int

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls, tensor: torch.Tensor, *, begin_idx, end_idx
    ) -> "PartialColumnOddEvenSeparatedTensor":
        layout_tensor = tensor.clone()
        separated_part = layout_tensor[..., begin_idx:end_idx]
        separated_part = (
            separated_part.view(
                *separated_part.shape[:-1], separated_part.shape[-1] // 2, 2
            )
            .transpose(-1, -2)
            .contiguous()
            .view(*separated_part.shape)
        )
        layout_tensor[..., begin_idx:end_idx] = separated_part
        return cls(
            plain_shape=tensor.shape,
            layout_tensor=layout_tensor,
            begin_idx=begin_idx,
            end_idx=end_idx,
        )

    @override
    def convert_to_plain(self) -> torch.Tensor:
        ret = self.layout_tensor.clone()
        separated_part = ret[..., self.begin_idx : self.end_idx]
        separated_part = (
            separated_part.view(
                *separated_part.shape[:-1], 2, separated_part.shape[-1] // 2
            )
            .transpose(-1, -2)
            .contiguous()
            .view(*separated_part.shape)
        )
        ret[..., self.begin_idx : self.end_idx] = separated_part
        return ret


@dataclass
class HygonW4A8Int4TileTensor(NativeLayoutTensor):
    """
    Hygon tiled layout for w4a8 kernels w4 weight on BW.

    This class only wraps the forward conversion:
      Packed4BitWeightQServe torch.Tensor -> hygon native tiled layout.
    """

    @classmethod
    @override
    @plum.dispatch
    def convert_from(cls, tensor: torch.Tensor):
        assert has_hygon_w4a8, "Hygon/Sugon w4a8 kernels are unavailable."
        assert hasattr(
            hygon_w4a8_kernels, "native_layout_of_weights_tile_int4"
        ), "Kernel 'native_layout_of_weights_tile_int4' not found."
        layout_tensor = hygon_w4a8_kernels.native_layout_of_weights_tile_int4(
            tensor.contiguous()
        )
        return cls(plain_shape=tensor.shape, layout_tensor=layout_tensor)

    @override
    def convert_to_plain(self):
        raise NotImplementedError(
            "No inverse kernel for int tile layout (expected 'plain_layout_of_weights_tile_int')."
        )


@dataclass
class HygonW4A8Int8TileTensor(NativeLayoutTensor):
    """
    Hygon tiled layout for w4a8 kernels i8 group scale on BW.

    This class only wraps the forward conversion:
      Packed4BitWeightQServe torch.Tensor -> hygon native tiled layout.
    """

    @classmethod
    @override
    @plum.dispatch
    def convert_from(cls, tensor: torch.Tensor):
        assert has_hygon_w4a8, "Hygon/Sugon w4a8 kernels are unavailable."
        assert hasattr(
            hygon_w4a8_kernels, "native_layout_of_scale_tile_i8"
        ), "Kernel 'native_layout_of_scale_tile_i8' not found."
        layout_tensor = hygon_w4a8_kernels.native_layout_of_scale_tile_i8(
            tensor.T.contiguous()
        )
        return cls(plain_shape=tensor.shape, layout_tensor=layout_tensor)

    @override
    def convert_to_plain(self):
        raise NotImplementedError(
            "No inverse kernel for int tile layout (expected 'plain_layout_of_weights_tile_int')."
        )


@dataclass
class HygonMixQIntTileTensor(NativeLayoutTensor):
    """
    Hygon tiled layout for mixQ integer weights (e.g., W4/W8).

    This class only wraps the forward conversion:
      plain torch.Tensor -> hygon native tiled layout (int path).
    """

    @classmethod
    @override
    @plum.dispatch
    def convert_from(cls, tensor: torch.Tensor):
        assert has_hygon, "Hygon/Sugon kernels are unavailable."
        assert hasattr(
            hygon_mixq_kernels, "native_layout_of_weights_tile_int"
        ), "Kernel 'native_layout_of_weights_tile_int' not found."
        layout_tensor = hygon_mixq_kernels.native_layout_of_weights_tile_int(
            tensor.contiguous()
        )
        return cls(plain_shape=tensor.shape, layout_tensor=layout_tensor)

    @override
    def convert_to_plain(self):
        raise NotImplementedError(
            "No inverse kernel for int tile layout (expected 'plain_layout_of_weights_tile_int')."
        )


@dataclass
class HygonMixQFp16TileTensor(NativeLayoutTensor):
    """
    Hygon tiled layout for mixQ FP16 weights.
    """

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls,
        tensor: torch.Tensor,
        *,
        perm_index=None,
    ):
        assert has_hygon, "Hygon/Sugon kernels are unavailable."
        assert hasattr(
            hygon_mixq_kernels, "native_layout_of_weights_tile_fp16"
        ), "Kernel 'native_layout_of_weights_tile_fp16' not found."

        src = tensor.contiguous()
        if perm_index is not None:
            if perm_index.dtype != torch.long:
                perm_index = perm_index.to(dtype=torch.long)
            src = torch.index_select(src, dim=1, index=perm_index)

        layout_tensor = hygon_mixq_kernels.native_layout_of_weights_tile_fp16(src)
        return cls(plain_shape=tensor.shape, layout_tensor=layout_tensor)

    @override
    def convert_to_plain(self):
        raise NotImplementedError(
            "No inverse kernel for fp16 tile layout (expected 'plain_layout_of_weights_tile_fp16')."
        )


@dataclass
class Repeat1ToLength(NativeLayoutTensor):
    """
    Repeat a scalar or a tensor with a final dimension of size 1 along the last
    dimension to a specified `length`. Optionally cast the values to `out_dtype`.

    Notes:
      - `plain_shape` stores the original shape of the input tensor.
      - `layout_tensor` stores the repeated 1-D tensor with shape [length].
    """

    length: int
    out_dtype: torch.dtype

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls, tensor: torch.Tensor, *, length: int, out_dtype: torch.dtype = None
    ):
        if tensor.numel() != 1:
            raise ValueError(
                f"Repeat1ToLength expects a scalar/size-1 tensor, but got shape {tuple(tensor.shape)}"
            )

        if out_dtype is None:
            out_dtype = tensor.dtype

        layout = tensor.detach().to(out_dtype).view(1).repeat(length).contiguous()
        return cls(
            plain_shape=tensor.shape,
            layout_tensor=layout,
            length=length,
            out_dtype=out_dtype,
        )

    @override
    def convert_to_plain(self) -> torch.Tensor:
        val = self.layout_tensor[0].to(self.out_dtype)
        return val.view(self.plain_shape)


@dataclass
class SqueezeLastSingleton(NativeLayoutTensor):
    """
    Remove the trailing singleton dimension of a tensor.

    Shape transform: [..., K, 1] -> [..., K]

    This only changes the view (no data copy) and does not alter the underlying data.
    """

    @classmethod
    @override
    @plum.dispatch
    def convert_from(cls, tensor: torch.Tensor) -> "SqueezeLastSingleton":
        if tensor.shape[-1] != 1:
            raise ValueError(
                f"SqueezeLastSingleton expects last dim == 1, but got shape {tuple(tensor.shape)}"
            )
        return cls(
            plain_shape=tensor.shape, layout_tensor=tensor.view(*tensor.shape[:-1])
        )

    @override
    def convert_to_plain(self) -> torch.Tensor:
        return self.layout_tensor.view(*self.plain_shape)


@dataclass
class LinearScaleToSwizzled(NativeLayoutTensor):
    """
    Convert the linear scale (of fp4 quantization) layout to swizzled scale layout.
    Padding the tensor from [..., m, k] to [..., round_up(m, 128), k]

    The linear layout is (..., m / 128, 4, 32, k / 4, 4)
    The swizzled layout is (..., m / 128, k / 4, 32, 4, 4)
    """

    @classmethod
    @override
    @plum.dispatch
    def convert_from(cls, tensor: torch.Tensor) -> "LinearScaleToSwizzled":
        shape = tensor.shape
        m, k = shape[-2], shape[-1]
        # padding
        if m % 128 != 0:
            padded_m = (m + 128 - 1) // 128 * 128
            shape = (*shape[:-2], padded_m, k)
            new_tensor = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
            new_tensor[..., :m, :k] = tensor
            tensor = new_tensor
            m = padded_m
        # transform
        tensor = (
            tensor.reshape((-1, m // 128, 4, 32, k // 4, 4))
            .permute(0, 1, 4, 3, 2, 5)
            .reshape(shape)
        )
        return cls(plain_shape=tensor.shape, layout_tensor=tensor)

    @override
    def convert_to_plain(self) -> torch.Tensor:
        return self.layout_tensor


@dataclass
class SwizzledScaleToLinear(NativeLayoutTensor):
    """
    Convert the swizzled scale (of fp4 quantization) layout to linear scale layout.
    Input tensor should be [..., m = 128 * t, k]

    The swizzled layout is (..., m / 128, k / 4, 32, 4, 4)
    The linear layout is (..., m / 128, 4, 32, k / 4, 4)
    """

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls, tensor: torch.Tensor, m: int = 0, k: int = 0
    ) -> "SwizzledScaleToLinear":
        shape = tensor.shape
        shape_m, shape_k = shape[-2], shape[-1]
        m = m or shape_m
        k = k or shape_k
        # transform
        tensor = (
            tensor.reshape((-1, shape_m // 128, shape_k // 4, 32, 4, 4))
            .permute(0, 1, 4, 3, 2, 5)
            .reshape(shape)[..., :m, :k]
            .contiguous()
        )
        return cls(plain_shape=tensor.shape, layout_tensor=tensor)

    @override
    def convert_to_plain(self) -> torch.Tensor:
        return self.layout_tensor


@dataclass
class MarlinNativeLayoutWeight(NativeLayoutTensor):
    @classmethod
    @override
    def convert_from(cls, tensor: torch.Tensor) -> "MarlinNativeLayoutWeight":

        if isinstance(tensor, torch.Tensor):
            n, k = tensor.shape
            tensor = tensor.view(torch.uint8)
            if n % 128 != 0:
                tensor = torch.cat(
                    [tensor, torch.zeros((128 - n % 128, k), dtype=torch.uint8)], dim=0
                )
            assert tensor.dim() == 2
            b16_tensor = tensor.to(torch.int16)
            b16_shape = b16_tensor.shape
            b16_review = (
                b16_tensor.reshape(b16_shape[0] // 64, 64, b16_shape[1] // 16, 16)
                .permute(2, 0, 1, 3)
                .contiguous()
            )
            b16_review = b16_review[:, :, :, 0:8] | (b16_review[:, :, :, 8:16] << 8)
            b16_repack = b16_review[:, :, :, 0:8]
            b16_repack = (
                b16_repack.reshape(
                    b16_repack.shape[0], b16_repack.shape[1], 64 // 8, 8, 4, 2
                )
                .permute(0, 1, 3, 4, 2, 5)
                .contiguous()
            )
            new_weight = b16_repack.view(b16_repack.shape[0], -1).view(torch.uint32)
            return cls(
                [n, k],
                new_weight,
            )
        else:
            raise TypeError(
                f"Cannot convert from {type(tensor)} to MarlinNativeLayoutWeight"
            )


@dataclass
class MarlinNativeLayoutScale(NativeLayoutTensor):
    @classmethod
    @override
    def convert_from(cls, tensor: torch.Tensor) -> "MarlinNativeLayoutScale":

        if isinstance(tensor, torch.Tensor):
            new_scale = tensor.t().contiguous().to(torch.float32)
            return cls(
                new_scale.shape,
                new_scale,
            )
        else:
            raise TypeError(
                f"Cannot convert from {type(tensor)} to MarlinNativeLayoutWeight"
            )


@dataclass
class MarlinNativeLayoutGroupWeight(NativeLayoutTensor):
    @classmethod
    @override
    def convert_from(cls, tensor: torch.Tensor) -> "MarlinNativeLayoutGroupWeight":

        if isinstance(tensor, torch.Tensor):
            e, n, k = tensor.shape
            tensor = tensor.view(torch.uint8)
            assert tensor.dim() == 3
            b16_tensor = tensor.to(torch.int16)
            b16_shape = b16_tensor.shape
            b16_review = (
                b16_tensor.reshape(
                    b16_shape[0], b16_shape[1] // 64, 64, b16_shape[2] // 16, 16
                )
                .permute(0, 3, 1, 2, 4)
                .contiguous()
            )
            b16_review = b16_review[:, :, :, :, 0:8] | (
                b16_review[:, :, :, :, 8:16] << 8
            )
            b16_repack = b16_review[:, :, :, 0:8]
            b16_repack = (
                b16_repack.reshape(
                    b16_repack.shape[0],
                    b16_repack.shape[1],
                    b16_repack.shape[2],
                    64 // 8,
                    8,
                    4,
                    2,
                )
                .permute(0, 2, 4, 5, 3, 6)
                .contiguous()
            )
            return cls(
                tensor.shape,
                b16_repack.view(torch.uint8).view(e, n, k),
            )
        else:
            raise TypeError(
                f"Cannot convert from {type(tensor)} to MarlinNativeLayoutGroupWeight"
            )


@dataclass
class InXOutWeight(NativeLayoutTensor):
    @classmethod
    @override
    def convert_from(cls, tensor: torch.Tensor) -> "InXOutWeight":

        if isinstance(tensor, torch.Tensor):
            tensor = tensor.t().contiguous()
            return cls(
                tensor.shape,
                tensor,
            )
        else:
            raise TypeError(f"Cannot convert from {type(tensor)} to InXOutWeight")
