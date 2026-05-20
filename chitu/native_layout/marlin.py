# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing_extensions import override

import torch

from chitu.native_layout.base import NativeLayoutTensor
from chitu.import_utils import try_import_platform_dep

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")


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
class BlockInt4MarlinQWeight(NativeLayoutTensor):
    """
    Convert blockint4 3D qweight [E, N, K//8] int32 to Marlin tiled format.
    E = number of expert groups, N = out_features, K = in_features.
    """

    @classmethod
    @override
    def convert_from(cls, tensor: torch.Tensor) -> "BlockInt4MarlinQWeight":
        if isinstance(tensor, torch.Tensor):
            if not has_marlin:
                raise RuntimeError(
                    "Marlin backend (chitu_backend) is not available for blockint4 repack"
                )
            e, n, k_packed = tensor.shape
            in_features = k_packed * 8
            out_features = n

            repacked_list = []
            for i in range(e):
                qw = tensor[i].T.contiguous()
                # gptq_marlin_repack requires GPU
                if qw.device.type != "cuda":
                    qw = qw.cuda()
                empty_g_idx = torch.empty(0, dtype=torch.int, device=qw.device)
                repacked = gptq_marlin_repack(
                    qw, empty_g_idx, in_features, out_features, 4
                )
                repacked_list.append(repacked)

            return cls(tensor.shape, torch.stack(repacked_list, dim=0))
        else:
            raise TypeError(
                f"Cannot convert from {type(tensor)} to BlockInt4MarlinQWeight"
            )


@dataclass
class BlockInt4MarlinScale(NativeLayoutTensor):
    """
    Convert blockint4 3D scales [E, N, K//G] bf16 to Marlin permuted scale format.
    E = number of expert groups, N = out_features, K = in_features, G = group_size.
    """

    @classmethod
    @override
    def convert_from(cls, tensor: torch.Tensor) -> "BlockInt4MarlinScale":
        if isinstance(tensor, torch.Tensor):
            e, n, num_groups = tensor.shape
            scale_perm, scale_perm_single = get_scale_perms()
            perm = scale_perm if num_groups > 1 else scale_perm_single

            permuted_list = []
            for i in range(e):
                sc = tensor[i].T.contiguous()
                sc = sc.reshape((-1, len(perm)))[:, perm]
                sc = sc.reshape((-1, n)).contiguous()
                permuted_list.append(sc)

            return cls(tensor.shape, torch.stack(permuted_list, dim=0))
        else:
            raise TypeError(
                f"Cannot convert from {type(tensor)} to BlockInt4MarlinScale"
            )
