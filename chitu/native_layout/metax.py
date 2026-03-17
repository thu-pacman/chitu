# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing_extensions import override
import plum

import torch

from chitu.native_layout.base import NativeLayoutTensor
from chitu.native_layout.common import BatchPaddedActivation
from chitu.import_utils import try_import_opt_dep

muxi_layout_kernels, has_muxi_layout_kernels = try_import_opt_dep(
    "muxi_layout_kernels", "muxi_layout_kernels"
)
metax_soft_fp4_kernels, has_metax_soft_fp4 = try_import_opt_dep(
    "metax_soft_fp4_kernels", "metax_soft_fp4_kernels"
)


class MuxiNativeLayoutActivation(NativeLayoutTensor):
    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls, tensor: BatchPaddedActivation
    ) -> "MuxiNativeLayoutActivation":
        assert tensor.multiple_of == 16
        return cls(
            tensor.plain_shape, muxi_layout_kernels.layoutB(tensor.layout_tensor)
        )


class MuxiNativeLayoutWeight(NativeLayoutTensor):
    @classmethod
    @override
    @plum.dispatch
    def convert_from(cls, tensor: torch.Tensor) -> "MuxiNativeLayoutWeight":
        m, k = tensor.shape
        assert m % 128 == 0
        assert k % 128 == 0
        return cls(
            tensor.shape,
            tensor.reshape(m // 16, 16, k // 8, 8).permute(0, 2, 1, 3).contiguous(),
        )


class MuxiNativeLayoutGroupWeight(NativeLayoutTensor):
    @classmethod
    @override
    @plum.dispatch
    def convert_from(cls, tensor: torch.Tensor) -> "MuxiNativeLayoutGroupWeight":
        e, m, k = tensor.shape
        assert m % 128 == 0
        assert k % 128 == 0
        return cls(
            tensor.shape,
            tensor.reshape(e, m // 16, 16, k // 8, 8)
            .permute(0, 1, 3, 2, 4)
            .contiguous(),
        )
