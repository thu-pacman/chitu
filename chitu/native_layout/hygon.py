# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing_extensions import override
import plum

import torch

from chitu.native_layout.base import NativeLayoutTensor
from chitu.import_utils import try_import_opt_dep
from chitu.utils import ceil_div

hygon_mixq_kernels, has_hygon = try_import_opt_dep(
    "sugon_mixQ4_kernels", "sugon_mixq4_kernels"
)
hygon_w4a8_kernels, has_hygon_w4a8 = try_import_opt_dep(
    "sugon_w4a8_kernels", "sugon_w4a8_kernels"
)


@dataclass
class HygonW4A8Int4TileTensor(NativeLayoutTensor):
    """
    Hygon tiled layout for w4a8 kernels w4 weight on BW.

    This class only wraps the forward conversion:
      Packed4BitWeightAlongK torch.Tensor -> hygon native tiled layout.
    """

    @classmethod
    @override
    @plum.dispatch
    def convert_from(cls, tensor: torch.Tensor):
        assert has_hygon_w4a8, "Hygon/Sugon w4a8 kernels are unavailable."
        assert hasattr(
            hygon_w4a8_kernels, "native_layout_of_weights_tile_int4_opt"
        ), "Kernel 'native_layout_of_weights_tile_int4_opt' not found."

        layout_tensor = hygon_w4a8_kernels.native_layout_of_weights_tile_int4_opt(
            tensor
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
    def convert_from(
        cls, tensor: torch.Tensor, *, weight_bits: int
    ):  # weights_bits for debug more easy
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
        if tensor.numel() == 0:
            return cls(plain_shape=tensor.shape, layout_tensor=tensor)
        layout_tensor = hygon_mixq_kernels.native_layout_of_weights_tile_fp16(src)
        return cls(plain_shape=tensor.shape, layout_tensor=layout_tensor)

    @override
    def convert_to_plain(self):
        raise NotImplementedError(
            "No inverse kernel for fp16 tile layout (expected 'plain_layout_of_weights_tile_fp16')."
        )
