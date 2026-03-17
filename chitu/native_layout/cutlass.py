# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing_extensions import override
import plum

import torch

from chitu.native_layout.base import NativeLayoutTensor


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
