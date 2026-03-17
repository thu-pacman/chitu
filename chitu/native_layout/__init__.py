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
    Packed4BitWeightAlongK,
    Packed4BitWeightAlongN,
    Packed4BitWeightQServe,
    ColumnOddEvenSeparatedTensor,
    PartialColumnOddEvenSeparatedTensor,
    Repeat1ToLength,
    SqueezeLastSingleton,
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
    HygonW4A8Int4TileTensor,
    HygonW4A8Int8TileTensor,
    HygonMixQIntTileTensor,
    HygonMixQFp16TileTensor,
)
from chitu.native_layout.cutlass import (
    LinearScaleToSwizzled,
)
from chitu.native_layout.marlin import (
    MarlinNativeLayoutWeight,
    MarlinNativeLayoutScale,
    MarlinNativeLayoutGroupWeight,
)
