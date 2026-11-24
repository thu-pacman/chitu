# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Type stub for `chitu.optdep`.

`chitu/optdep/__init__.py` resolves dependencies dynamically with a module-level
`__getattr__`, which static analyzers cannot see. This stub lists every name so
that `from chitu.optdep import triton` autocompletes and `triton.` gets member
completion from the real package.

The imports below are never executed: a `.pyi` file is only read by type
checkers and IDEs. If a dependency is not installed on the developer's machine,
the corresponding import shows as unresolved; that is expected and does not
affect runtime.

Keep this list in sync with the `_load_<name>` functions in
`chitu/optdep/__init__.py`. Nothing checks it at CI time, so a missing entry
only shows up as missing completion in the editor.
"""

import aiter
import aiter.moe as aiter_moe
import amdsmi
import bitsandbytes
import boltops.fla.kda as boltops_kda
import boltops.fla.kda.triton.fused_recurrent as boltops_kda_gate
import boltops.generic.triton.causal_conv1d as boltops_conv
import chitu_backend
import cinfer_ascendc
import cpuinfer
import deep_ep
import deep_gemm
import deepgemm
import fast_hadamard_transform
import fla
import flash_attn
import flash_attn_interface
import flash_mla
import flashinfer
import gptqmodel_marlin_kernels
import hard_fp4_kernels
import hpc
import hygon_mxfp4_decode
import hygon_w4a16_kernel
import hygon_w8a8_kernel
import hygon_w8a8_kernel.dense as hygon_w8a8_kernel_dense
import lightop
import lmslim
import lmslim.quantize.quant_ops as lmslim_quantize_quant_ops
import metax_soft_fp4_kernels
import mooncake
import muxi_layout_kernels
import numa
import pynvml
import scipy
import sugon_mixQ4_kernels
import sugon_w4a8_kernels
import tbsgemm
import tilelang
import torch_gcu
import torch_npu
import torchada
import triton
