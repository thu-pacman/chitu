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


def _validate_aiter_moe_c_int8_weight(
    weight: torch.Tensor,
    *,
    gemm_name: str,
    require_n_multiple_of_256: bool = False,
) -> tuple[int, int, int]:
    if weight.dim() != 3:
        raise ValueError(
            f"Aiter MOE_C {gemm_name} weight must be 3D [E, N, K], "
            f"got {tuple(weight.shape)}"
        )
    if weight.dtype != torch.int8:
        raise ValueError(
            f"Aiter MOE_C {gemm_name} weight must be int8, got {weight.dtype}"
        )
    e, n, k = weight.shape
    if require_n_multiple_of_256:
        if k % 64 != 0 or n % 256 != 0:
            raise ValueError(
                f"Aiter MOE_C {gemm_name} weight requires K multiple of 64 and "
                f"N multiple of 256, got N={n} K={k}"
            )
    elif k % 64 != 0:
        raise ValueError(
            f"Aiter MOE_C {gemm_name} weight K must be a multiple of 64, got {k}"
        )
    return e, n, k


@dataclass
class AiterMoeCInt8Gemm1Weight(NativeLayoutTensor):
    """Aiter MOE_C W8A8 GEMM1 native layout for plain [E, N, K] int8 weight."""

    @classmethod
    @override
    def convert_from(cls, tensor):
        if isinstance(tensor, cls):
            return tensor
        if isinstance(tensor, NativeLayoutTensor):
            tensor = tensor.convert_to_plain()

        e, n, k = _validate_aiter_moe_c_int8_weight(
            tensor,
            gemm_name="GEMM1",
        )
        layout_tensor = (
            tensor.transpose(1, 2)
            .contiguous()
            .view(e, k // 64, 64, n)
            .transpose(2, 3)
            .contiguous()
            .view(e, n, k)
        )
        return cls(plain_shape=tensor.shape, layout_tensor=layout_tensor)

    @override
    def convert_to_plain(self) -> torch.Tensor:
        e, n, k = _validate_aiter_moe_c_int8_weight(
            self.layout_tensor,
            gemm_name="GEMM1",
        )
        return (
            self.layout_tensor.view(e, k // 64, n, 64)
            .transpose(2, 3)
            .contiguous()
            .view(e, k, n)
            .transpose(1, 2)
            .contiguous()
        )


@dataclass
class AiterMoeCInt8Gemm2Weight(NativeLayoutTensor):
    """Aiter MOE_C W8A8 GEMM2 native layout for plain [E, N, K] int8 weight."""

    @classmethod
    @override
    def convert_from(cls, tensor):
        if isinstance(tensor, cls):
            return tensor
        if isinstance(tensor, NativeLayoutTensor):
            tensor = tensor.convert_to_plain()

        e, n, k = _validate_aiter_moe_c_int8_weight(
            tensor,
            gemm_name="GEMM2",
            require_n_multiple_of_256=True,
        )
        layout_tensor = (
            tensor.transpose(1, 2)
            .contiguous()
            .view(e, k // 64, 64, n // 256, 256)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(e, k // 64, n // 256, 16, 16, 4, 16)
            .permute(0, 1, 2, 3, 5, 4, 6)
            .contiguous()
            .view(e, n, k)
        )
        return cls(plain_shape=tensor.shape, layout_tensor=layout_tensor)

    @override
    def convert_to_plain(self) -> torch.Tensor:
        e, n, k = _validate_aiter_moe_c_int8_weight(
            self.layout_tensor,
            gemm_name="GEMM2",
            require_n_multiple_of_256=True,
        )
        return (
            self.layout_tensor.view(e, k // 64, n // 256, 16, 4, 16, 16)
            .permute(0, 1, 2, 3, 5, 4, 6)
            .contiguous()
            .view(e, k // 64, n // 256, 256, 64)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
            .view(e, k, n)
            .transpose(1, 2)
            .contiguous()
        )


_DEEPGEMM_W8A8_TILE = 16


def _check_deepgemm_w8a8_plain_shape(shape: torch.Size):
    if len(shape) == 2:
        n, k = shape
    elif len(shape) == 3:
        _, n, k = shape
    else:
        raise ValueError(
            f"DeepGEMM W8A8 weight should be 2D or 3D, got shape {tuple(shape)}"
        )

    if n % _DEEPGEMM_W8A8_TILE != 0 or k % _DEEPGEMM_W8A8_TILE != 0:
        raise ValueError(
            "DeepGEMM W8A8 weight requires N and K to be multiples of "
            f"{_DEEPGEMM_W8A8_TILE}, got N={n}, K={k}"
        )


def _pack_deepgemm_w8a8_2d(weight: torch.Tensor):
    n, k = weight.shape
    return (
        weight.contiguous()
        .view(
            n // _DEEPGEMM_W8A8_TILE,
            _DEEPGEMM_W8A8_TILE,
            k // _DEEPGEMM_W8A8_TILE,
            _DEEPGEMM_W8A8_TILE,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(n // _DEEPGEMM_W8A8_TILE, k * _DEEPGEMM_W8A8_TILE)
    )


def _pack_deepgemm_w8a8_3d(weight: torch.Tensor):
    e, n, k = weight.shape
    return (
        weight.contiguous()
        .view(
            e,
            n // _DEEPGEMM_W8A8_TILE,
            _DEEPGEMM_W8A8_TILE,
            k // _DEEPGEMM_W8A8_TILE,
            _DEEPGEMM_W8A8_TILE,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(e, n // _DEEPGEMM_W8A8_TILE, k * _DEEPGEMM_W8A8_TILE)
    )


def _unpack_deepgemm_w8a8_2d(weight: torch.Tensor, plain_shape: torch.Size):
    n, k = plain_shape
    return (
        weight.contiguous()
        .view(
            n // _DEEPGEMM_W8A8_TILE,
            k // _DEEPGEMM_W8A8_TILE,
            _DEEPGEMM_W8A8_TILE,
            _DEEPGEMM_W8A8_TILE,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(n, k)
    )


def _unpack_deepgemm_w8a8_3d(weight: torch.Tensor, plain_shape: torch.Size):
    e, n, k = plain_shape
    return (
        weight.contiguous()
        .view(
            e,
            n // _DEEPGEMM_W8A8_TILE,
            k // _DEEPGEMM_W8A8_TILE,
            _DEEPGEMM_W8A8_TILE,
            _DEEPGEMM_W8A8_TILE,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(e, n, k)
    )


@dataclass
class HygonDeepGemmW8A8MarlinWeight(NativeLayoutTensor):
    """
    DeepGEMM W8A8 grouped GEMM weight layout.

    Plain weight is NT-style [N, K]. DeepGEMM expects 16-wide Marlin tiles:
    [N, K] -> [N // 16, K * 16]. Grouped weights keep the expert dimension:
    [E, N, K] -> [E, N // 16, K * 16].
    """

    @classmethod
    @override
    def convert_from(cls, tensor: torch.Tensor | NativeLayoutTensor):
        if isinstance(tensor, cls):
            return tensor
        if isinstance(tensor, NativeLayoutTensor):
            tensor = tensor.convert_to_plain()
        if tensor.dtype != torch.int8:
            raise ValueError(f"DeepGEMM W8A8 weight should be int8, got {tensor.dtype}")

        plain_shape = torch.Size(tensor.shape)
        _check_deepgemm_w8a8_plain_shape(plain_shape)
        if tensor.dim() == 2:
            layout_tensor = _pack_deepgemm_w8a8_2d(tensor)
        else:
            layout_tensor = _pack_deepgemm_w8a8_3d(tensor)
        return cls(plain_shape=plain_shape, layout_tensor=layout_tensor)

    @override
    def convert_to_plain(self):
        # Used only for tests, debugging, and layout-to-layout conversion.
        # DeepGEMM inference consumes the packed layout directly.
        if len(self.plain_shape) == 2:
            return _unpack_deepgemm_w8a8_2d(
                self.layout_tensor,
                torch.Size(self.plain_shape),
            )
        return _unpack_deepgemm_w8a8_3d(
            self.layout_tensor,
            torch.Size(self.plain_shape),
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
