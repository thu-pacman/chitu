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
        device = tensor.device
        layout_tensor = deep_gemm.transform_sf_into_required_layout(
            tensor.cuda(),
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
