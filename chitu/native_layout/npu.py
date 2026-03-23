# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing_extensions import override
import plum

import torch

from chitu.native_layout.base import NativeLayoutTensor
from chitu.native_layout.common import Packed4BitWeightAlongK
from chitu.import_utils import try_import_and_setup_torch_npu

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


@dataclass
class Packed4BitWeightNPUNative(NativeLayoutTensor):
    """
    weight (..., N, K / 2)
        1. unpack, transpose, repack to (..., K, N / 2)
        2. view as    (..., K3, K2, K1, K0, N3, N2, N1, N0)
        3. permute to (..., K3, K1, K2, N2, N3, N1, K0, N0)
    """

    K_SHAPE = (4, 2, 8)  # K2, K1, K0
    N_SHAPE = (8, 4, 4)  # N2, N1, N0

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

        weight = weight.to(device="npu", non_blocking=True)
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
            *cls.K_SHAPE,
            weight_shape[-1] // 128,
            *cls.N_SHAPE,
        )
        new_weight = tmp_weight.permute(0, 2, 1, 5, 4, 6, 3, 7).contiguous()
        # FIXME this old_shape is incorrect for its logical layout
        return new_weight.reshape(old_shape).to(old_device, non_blocking=True)

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
