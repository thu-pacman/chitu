# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import math
from typing import Optional
from typing_extensions import override
import plum

import torch
import torch.nn.functional as F

from chitu.native_layout.base import NativeLayoutTensor
from chitu.native_layout.common import Packed4BitWeightAlongKContig


def nvfp4_moe_pad_n_for_group_mm_b_scale(n: int, k_logical: int) -> int:
    if n <= 0 or k_logical <= 0:
        return n
    if k_logical % 16 != 0:
        return n
    group_k = k_logical // 16
    step = 128 // math.gcd(group_k, 128)
    return (n + step - 1) // step * step


def nvfp4_moe_down_proj_n_padded(n: int, packed_half_k: int) -> int:
    """Pad local N so Blackwell nvfp4 MoE down_proj B-scales keep 128 B task alignment."""
    k_logical_padded = (packed_half_k * 2 + 255) // 256 * 256
    return nvfp4_moe_pad_n_for_group_mm_b_scale(n, k_logical_padded)


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
        shape = self.layout_tensor.shape
        shape_m, shape_k = shape[-2], shape[-1]
        m, k = self.plain_shape[-2], self.plain_shape[-1]
        return (
            self.layout_tensor.reshape((-1, shape_m // 128, shape_k // 4, 32, 4, 4))
            .permute(0, 1, 4, 3, 2, 5)
            .reshape(shape)[..., :m, :k]
            .contiguous()
        )


def _pad_blockfp4_linear_scale_to_shape(
    tensor: torch.Tensor, padded_shape: tuple
) -> torch.Tensor:
    """2D MoE ``HygonMXFP4MOEPadTensor`` down_proj-style: pad last dim only with neutral 128 / 0."""
    r, c = tensor.shape[0], tensor.shape[1]
    padded_r, padded_c = padded_shape[0], padded_shape[1]
    assert padded_r == r, f"scale row dim mismatch: {r} vs {padded_r}"
    if c != padded_c:
        if c > padded_c:
            raise ValueError(
                f"weight_scale last dim {c} exceeds padded target {padded_c}"
            )
        pad_c = padded_c - c
        fill = 128 if tensor.dtype == torch.uint8 else 0
        tensor = F.pad(tensor, (0, pad_c), value=fill)
    return tensor.contiguous()


@dataclass
class Blockfp4LinearScalePadToSwizzled(NativeLayoutTensor):
    """
    Pad 2D first-level scales to ``padded_shape`` (same idea as ``HygonMXFP4MOEPadTensor``
    last-dim pad), then ``LinearScaleToSwizzled``.
    """

    padded_shape: Optional[tuple] = None

    @classmethod
    @override
    @plum.dispatch
    def convert_from(cls, tensor: torch.Tensor, *, padded_shape: tuple):
        tensor = _pad_blockfp4_linear_scale_to_shape(tensor, padded_shape)
        sw = LinearScaleToSwizzled.convert_from(tensor)
        return cls(
            plain_shape=sw.plain_shape,
            layout_tensor=sw.layout_tensor,
            padded_shape=tuple(padded_shape),
        )

    @override
    def convert_to_plain(self) -> torch.Tensor:
        return LinearScaleToSwizzled(
            plain_shape=self.plain_shape, layout_tensor=self.layout_tensor
        ).convert_to_plain()


def _mxfp4_moe_pad_weight_to_shape(
    tensor: torch.Tensor, padded_shape: tuple
) -> torch.Tensor:
    """Pad packed MoE fp4 weights like HygonMXFP4MOETileTensor, without vendor tiling."""
    e, m, n = tensor.shape[0], tensor.shape[1], tensor.shape[2]
    padded_e, padded_m, padded_n = padded_shape[0], padded_shape[1], padded_shape[2]
    assert padded_e == e, f"expert dim must match: {e} vs {padded_e}"
    if m != padded_m or n != padded_n:
        pad_m = padded_m - m
        pad_n = padded_n - n
        if pad_m != 0:
            pad_m_half = pad_m // 2
            m_split = m // 2
            tensor_part1 = tensor[:, :m_split, :]
            tensor_part2 = tensor[:, m_split:, :]
            pad_tensor = torch.zeros(
                e, pad_m_half, n, dtype=tensor.dtype, device=tensor.device
            )
            tensor = torch.cat(
                [tensor_part1, pad_tensor, tensor_part2, pad_tensor], dim=1
            )
            m = padded_m
        if pad_n != 0:
            tensor = F.pad(tensor, (0, pad_n), value=0)
        tensor = tensor.contiguous()
    return tensor


def _mxfp4_moe_pad_scale_to_shape(
    tensor: torch.Tensor, padded_shape: tuple
) -> torch.Tensor:
    """Pad MoE MXFP4 e8m0 scales like HygonMXFP4MOEPadTensor (neutral fill 128)."""
    e, m, n = tensor.shape[0], tensor.shape[1], tensor.shape[2]
    padded_e, padded_m, padded_n = padded_shape[0], padded_shape[1], padded_shape[2]
    assert padded_e == e, f"expert dim must match: {e} vs {padded_e}"
    if m != padded_m or n != padded_n:
        pad_m = padded_m - m
        pad_n = padded_n - n
        if pad_m != 0:
            pad_m_half = pad_m // 2
            m_split = m // 2
            tensor_part1 = tensor[:, :m_split, :]
            tensor_part2 = tensor[:, m_split:, :]
            pad_tensor = (
                torch.ones(e, pad_m_half, n, dtype=tensor.dtype, device=tensor.device)
                * 128
            )
            tensor = torch.cat(
                [tensor_part1, pad_tensor, tensor_part2, pad_tensor], dim=1
            )
            m = padded_m
        if pad_n != 0:
            tensor = F.pad(tensor, (0, pad_n), value=128)
        tensor = tensor.contiguous()
    return tensor


@dataclass
class BlackwellMXFP4MOEPadWeight(NativeLayoutTensor):
    """
    Pad Blackwell MXFP4 MoE packed weights for TP with hard_fp4 kernels.
    """

    padded_shape: Optional[tuple] = None

    @classmethod
    @override
    @plum.dispatch
    def convert_from(cls, packed: Packed4BitWeightAlongKContig, *, padded_shape: tuple):

        assert packed.layout_tensor.ndim == 3, packed.layout_tensor.shape
        tensor = _mxfp4_moe_pad_weight_to_shape(packed.layout_tensor, padded_shape)
        return cls(
            plain_shape=packed.plain_shape,
            layout_tensor=tensor,
            padded_shape=tuple(padded_shape),
        )

    @override
    def convert_to_plain(self):
        raise NotImplementedError(
            "BlackwellMXFP4MOEPadWeight does not support convert_to_plain."
        )

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise NotImplementedError(
                f"Indexing {type(self)} with {type(index)} is not supported."
            )
        sl = self.layout_tensor[index]
        plain = tuple(self.plain_shape[1:])
        return Packed4BitWeightAlongKContig(
            plain_shape=plain,
            layout_tensor=sl,
        )


@dataclass
class BlackwellMXFP4MOEScalePadToSwizzled(NativeLayoutTensor):
    """
    Pad MoE scales like HygonMXFP4MOEPadTensor, then apply LinearScaleToSwizzled.
    """

    padded_shape: Optional[tuple] = None

    @classmethod
    @override
    @plum.dispatch
    def convert_from(cls, tensor: torch.Tensor, *, padded_shape: tuple):
        tensor = _mxfp4_moe_pad_scale_to_shape(tensor, padded_shape)
        sw = LinearScaleToSwizzled.convert_from(tensor)
        return cls(
            plain_shape=sw.plain_shape,
            layout_tensor=sw.layout_tensor,
            padded_shape=tuple(padded_shape),
        )

    @override
    def convert_to_plain(self) -> torch.Tensor:
        return LinearScaleToSwizzled(
            plain_shape=self.plain_shape, layout_tensor=self.layout_tensor
        ).convert_to_plain()
