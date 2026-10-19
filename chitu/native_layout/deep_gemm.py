# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import dataclasses
from dataclasses import dataclass
import plum

import torch

from chitu.native_layout.base import NativeLayoutTensor
from chitu.import_utils import try_import_opt_dep

deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")


@dataclass
class DeepGemmScale(NativeLayoutTensor):
    # This marks all following properties must be set via kwargs
    _: dataclasses.KW_ONLY

    mn: int
    k: int
    num_groups: Optional[int] = None
    disable_ue8m0_cast: bool

    @classmethod
    @override
    @plum.dispatch
    def convert_from(
        cls,
        tensor: torch.Tensor,
        *,
        mn: int,
        k: int,
        num_groups: Optional[int] = None,
        disable_ue8m0_cast: bool,
    ) -> "DeepGemmScale":
        if tensor.device.type == "meta":
            if not disable_ue8m0_cast and torch.cuda.get_device_capability()[0] == 10:
                # Blackwell broadcasts each 128-row block scale to every row,
                # packs four UE8M0 exponents per int32, and stores MN-major
                # columns aligned to 16 bytes. Meta parameters must have the
                # same layout as the converted checkpoint before load_state_dict.
                packed_k = (k + 511) // 512
                aligned_mn = ((mn + 3) // 4) * 4
                shape = (mn, packed_k)
                stride = (1, aligned_mn)
                if num_groups is not None:
                    shape = (num_groups, *shape)
                    stride = (aligned_mn * packed_k, *stride)
                layout_tensor = torch.empty_strided(
                    shape, stride, device="meta", dtype=torch.int32
                )
            else:
                layout_tensor = torch.empty_like(tensor, dtype=torch.float32)
            return cls(
                plain_shape=tensor.shape,
                layout_tensor=layout_tensor,
                mn=mn,
                k=k,
                num_groups=num_groups,
                disable_ue8m0_cast=disable_ue8m0_cast,
            )
        device = tensor.device
        layout_tensor = deep_gemm.transform_sf_into_required_layout(
            tensor.float().cuda(),
            mn,
            k,
            num_groups=num_groups,
            disable_ue8m0_cast=disable_ue8m0_cast,
            recipe=(1, 128, 128),
            is_sfa=False,  # sfa = scale of activation, sfb = scale of weight
        ).to(device)
        return cls(
            plain_shape=tensor.shape,
            layout_tensor=layout_tensor,
            mn=mn,
            k=k,
            num_groups=num_groups,
            disable_ue8m0_cast=disable_ue8m0_cast,
        )
