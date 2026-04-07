# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from chitu.native_layout.base import NativeLayoutTensor
from chitu.native_layout.helper import enable_native_layout_weight
from chitu.native_layout.common import (
    Vector,
    PermutedTensor,
    TransposeLastTwoDim,
    BatchPaddedActivation,
    Blockfp4LinearPackedWeightPadToShape,
    Packed4BitWeightAlongK,
    Packed4BitWeightAlongN,
    Packed4BitWeightQServe,
    ColumnOddEvenSeparatedTensor,
    PartialColumnOddEvenSeparatedTensor,
    Repeat1ToLength,
    InXOutWeight,
)
from chitu.native_layout.npu import (
    Packed4BitWeightNPUNative,
    NpuFractalNzTensor,
    NpuFractalZnTensor,
)
from chitu.native_layout.metax import (
    MuxiNativeLayoutActivation,
    MuxiNativeLayoutWeight,
    MuxiNativeLayoutGroupWeight,
)
from chitu.native_layout.hygon import (
    AiterMoeCInt8Gemm1Weight,
    AiterMoeCInt8Gemm2Weight,
    HygonDeepGemmW8A8MarlinWeight,
    HygonW4A8Int4TileTensor,
    HygonW4A8Int8TileTensor,
    HygonMixQIntTileTensor,
    HygonMixQFp16TileTensor,
)
from chitu.native_layout.cutlass import (
    BlackwellMXFP4MOEPadWeight,
    BlackwellMXFP4MOEScalePadToSwizzled,
    Blockfp4LinearScalePadToSwizzled,
    LinearScaleToSwizzled,
    nvfp4_moe_down_proj_n_padded,
    nvfp4_moe_pad_n_for_group_mm_b_scale,
)
from chitu.native_layout.marlin import (
    MarlinNativeLayoutWeight,
    MarlinNativeLayoutScale,
    MarlinNativeLayoutGroupWeight,
    BlockInt4MarlinQWeight,
    BlockInt4MarlinScale,
)
from chitu.native_layout.deep_gemm import DeepGemmScale
