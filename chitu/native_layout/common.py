# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Sequence
from typing_extensions import override
import functools
import plum

import torch

from chitu.native_layout.base import NativeLayoutTensor
from chitu.import_utils import try_import_platform_dep

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")


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
class TransposeLastTwoDim(NativeLayoutTensor):
    @classmethod
    @override
    def convert_from(cls, tensor: torch.Tensor) -> "TransposeLastTwoDim":
        return cls(
            plain_shape=tensor.shape,
            layout_tensor=tensor.transpose(-1, -2).contiguous(),
        )


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
        cls,
        tensor: torch.Tensor,
        *,
        k_stride: int = 1,
    ) -> "Packed4BitWeightAlongK":
        """
        Build a ``Packed4BitWeightAlongK`` from a raw uint8 tensor storing packed 4-bit
        values along the K dimension.

        This overload assumes the input uses the canonical ``k_stride = 1`` layout, i.e.:

            - Mathematical shape: (..., K)
            - Storage shape:      (..., K / 2)  (two 4-bit values per byte)

        It first wraps the raw tensor into a ``Packed4BitWeightAlongK`` with
        ``k_stride = 1`` and then, if a different ``k_stride`` is requested, delegates
        to the existing ``convert_from(Packed4BitWeightAlongK, k_stride=...)`` overload.

        Args:
            tensor:
                Raw packed 4-bit weights with shape ``[..., K/2]`` and dtype
                ``torch.uint8``. Each byte encodes two 4-bit values along K.
            k_stride:
                Desired stride (in bytes) along the K dimension in the internal layout.
                A value of ``1`` corresponds to the canonical tightly packed layout.

        Returns:
            Packed4BitWeightAlongK:
                Layout object representing weights with mathematical shape ``[..., K]``
                and the requested ``k_stride``.

        Raises:
            AssertionError:
                If ``tensor`` does not have dtype ``torch.uint8``.
        """
        assert (
            tensor.dtype == torch.uint8
        ), "Packed4BitWeightAlongK expects a uint8 tensor for packed 4-bit weights"

        # Storage shape: [..., K_half] where K_half = K / 2 (two 4-bit values per byte).
        *batch_dims, k_half = tensor.shape
        k = k_half * 2
        plain_shape = (*batch_dims, k)

        # Wrap the raw storage as a canonical k_stride = 1 layout.
        base_k1 = cls(
            plain_shape=plain_shape,
            layout_tensor=tensor.contiguous(),
            k_stride=1,
        )

        # Fast path: the caller wants the canonical layout.
        if k_stride == 1:
            return base_k1

        # Otherwise, reuse the Packed4BitWeightAlongK to Packed4BitWeightAlongK
        # converter that handles arbitrary k_stride layouts.
        return cls.convert_from(base_k1, k_stride=k_stride)

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
class Packed4BitWeightAlongN(NativeLayoutTensor):
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
        cls,
        tensor: torch.Tensor,
        *,
        k_stride: int = 1,
    ) -> "Packed4BitWeightAlongN":
        """
        Build a ``Packed4BitWeightAlongN`` from a raw uint8 tensor storing packed 4-bit
        values along the K dimension.

        This overload assumes the input uses the canonical ``k_stride = 1`` layout, i.e.:

            - Mathematical shape: (..., K)
            - Storage shape:      (..., K / 2)  (two 4-bit values per byte)

        It first wraps the raw tensor into a ``Packed4BitWeightAlongN`` with
        ``k_stride = 1`` and then, if a different ``k_stride`` is requested, delegates
        to the existing ``convert_from(Packed4BitWeightAlongN, k_stride=...)`` overload.

        Args:
            tensor:
                Raw packed 4-bit weights with shape ``[..., K/2]`` and dtype
                ``torch.uint8``. Each byte encodes two 4-bit values along K.
            k_stride:
                Desired stride (in bytes) along the K dimension in the internal layout.
                A value of ``1`` corresponds to the canonical tightly packed layout.

        Returns:
            Packed4BitWeightAlongN:
                Layout object representing weights with mathematical shape ``[..., K]``
                and the requested ``k_stride``.

        Raises:
            AssertionError:
                If ``tensor`` does not have dtype ``torch.uint8``.
        """
        assert (
            tensor.dtype == torch.uint8
        ), "Packed4BitWeightAlongN expects a uint8 tensor for packed 4-bit weights"

        # Storage shape: [..., K_half] where K_half = K / 2 (two 4-bit values per byte).
        *batch_dims, k_half = tensor.shape
        k = k_half * 2
        plain_shape = (*batch_dims, k)

        # Wrap the raw storage as a canonical k_stride = 1 layout.
        base_k1 = cls(
            plain_shape=plain_shape,
            layout_tensor=tensor.contiguous(),
            k_stride=1,
        )

        # Fast path: the caller wants the canonical layout.
        if k_stride == 1:
            return base_k1

        # Otherwise, reuse the Packed4BitWeightAlongN to Packed4BitWeightAlongN
        # converter that handles arbitrary k_stride layouts.
        return cls.convert_from(base_k1, k_stride=k_stride)

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls, tensor: "Packed4BitWeightAlongN", *, k_stride: int = 1
    ) -> "Packed4BitWeightAlongN":
        n_size_for_128B_swizzle = 8
        weight_uint8 = tensor.layout_tensor.view(torch.uint8)
        weight_int16 = weight_uint8.to(torch.int16)
        weight_int16 = (weight_int16 << 4) | weight_int16
        weight_int8 = (weight_int16 & 0x0F0F).view(torch.int8)
        weight_int16 = (
            weight_int8.view(
                weight_uint8.shape[0] // n_size_for_128B_swizzle // 2,
                2,
                n_size_for_128B_swizzle,
                -1,
            )
            .permute(0, 2, 3, 1)
            .contiguous()
            .reshape(weight_uint8.shape[0] // 2, -1)
            .view(torch.int16)
        )
        weight_repack = (
            (weight_int16 | (weight_int16 >> 4))
            .to(torch.uint8)
            .view(torch.float8_e4m3fn)
        )
        return cls(
            plain_shape=tensor.plain_shape,
            layout_tensor=weight_repack,
            k_stride=k_stride,
        )

    def __getitem__(self, index):
        """
        Indexing a Packed4BitWeightAlongN is safe is the K dimension is untouched. In such a
        case, this function returns a new Packed4BitWeightAlongN with the same layout.
        """
        if not isinstance(index, int):
            raise NotImplementedError(
                f"Indexing {type(self)} with {type(index)} is not supported."
            )
        if len(self.plain_shape) <= 1:
            raise ValueError(
                "Cannot index a Packed4BitWeightAlongN tensor's K dimension."
            )
        return Packed4BitWeightAlongN(
            self.plain_shape[1:], self.layout_tensor[index], k_stride=self.k_stride
        )


@dataclass
class Packed4BitWeightQServe(NativeLayoutTensor):
    """
    Layout used in QServe

    See
    https://github.com/mit-han-lab/deepcompressor/blob/main/deepcompressor/backend/qserve/utils.py#L18
    for the format details.
    """

    pass


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
