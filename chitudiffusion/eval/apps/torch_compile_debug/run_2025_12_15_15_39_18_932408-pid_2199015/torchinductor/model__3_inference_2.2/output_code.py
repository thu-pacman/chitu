# AOT ID: ['3_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
from torch._C import _cuda_getCurrentRawStream as get_raw_stream



# kernel path: /tmp/torchinductor_wucz/7s/c7s5sqhvs6fmvu5irft7n3moed4un6ocn6mv6iafjtjnjp6eb743.py
# Topologically Sorted Source Nodes: [z], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   z => convolution
# Graph fragment:
#   %arg2_1 : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convolution : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf0
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 16384}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 1048576, 'x': 524288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 16384*y3), ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 4*x2 + 65536*y1), tmp0, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/wk/cwkfvkh22fy2fr767hlum4kcewb73tbkratayzrflmuluqewyaj2.py
# Topologically Sorted Source Nodes: [z], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   z => convolution
# Graph fragment:
#   %arg1_1 : Tensor "f16[4][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %buf0 : Tensor "f16[4, 4, 128, 128][65536, 1, 512, 4]cuda:0" = PlaceHolder[target=buf0]
#   %arg0_1 : Tensor "f16[4, 4, 1, 1][4, 1, 1, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convolution : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf1
triton_tem_fused_convolution_1 = async_compile.triton('triton_tem_fused_convolution_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'in_ptr0': '*fp16', 'arg_A': '*fp16', 'arg_B': '*fp16', 'out_ptr0': '*fp16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_1', 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': False, 'ALLOW_TF32': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16, 'GROUP_M': 8}},

)
@triton.jit
def triton_tem_fused_convolution_1(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 16
    GROUP_M : tl.constexpr = 8
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 65536
    N = 4
    K = 4
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 4
    stride_ak = 1
    stride_bk = 1
    stride_bn = 4

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 4*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 4*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 4*idx_n, xindex.shape)).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 4*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), tmp1, mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/wg/cwgbspzuk7h75wxoqzmga4qmeuzkmq6spzj23u63ufqxerbhsxll.py
# Topologically Sorted Source Nodes: [z, sample], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sample => convolution_1
#   z => convolution
# Graph fragment:
#   %arg3_1 : Tensor "f16[512, 4, 3, 3][36, 9, 3, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convolution : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf2
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 73728, 'x': 36864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 4*x2 + 36*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/2g/c2guwvj6bkq4xr6sfsh5vrcualaql4cpsjaho7cekpmie6cjtv2e.py
# Topologically Sorted Source Nodes: [z, sample], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sample => convolution_1
#   z => convolution
# Graph fragment:
#   %buf1 : Tensor "f16[65536, 4][4, 1]cuda:0" = PlaceHolder[target=buf1]
#   %buf2 : Tensor "f16[512, 4, 3, 3][36, 1, 12, 4]cuda:0" = PlaceHolder[target=buf2]
#   %convolution : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf3
triton_tem_fused_convolution_3 = async_compile.triton('triton_tem_fused_convolution_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=2,
num_warps=4,
triton_meta={'signature': {'arg_X': '*fp16', 'arg_W': '*fp16', 'out_ptr0': '*fp16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_3', 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'KERNEL_H': 3, 'KERNEL_W': 3, 'STRIDE_H': 1, 'STRIDE_W': 1, 'PADDING_H': 1, 'PADDING_W': 1, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': True, 'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}},

)
@triton.jit
def triton_tem_fused_convolution_3(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = True
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 256
    BLOCK_K : tl.constexpr = 16
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 4
    IN_C = 4
    IN_H = 128
    IN_W = 128
    OUT_C = 512
    OUT_H = 128
    OUT_W = 128

    # Strides:
    stride_xn = 65536
    stride_xc = 1
    stride_xh = 512
    stride_xw = 4
    stride_wc_out = 36
    stride_wc_in = 1
    stride_wh = 12
    stride_ww = 4

    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 128*idx_h + 16384*idx_c + 8388608*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_c + 512*idx_w + 65536*idx_h + 8388608*idx_n, acc.shape)), acc, mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/pv/cpv2iifsdkwnw7wzyljk77allkkup5omi7dsexs7syq2s6bxacp3.py
# Topologically Sorted Source Nodes: [z, sample, hidden_states], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states => convert_element_type, var_mean, view
#   sample => convolution_1
#   z => convolution
# Graph fragment:
#   %buf3 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf3]
#   %arg4_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %convolution : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %view : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf4,%buf5,%buf6
triton_per_fused_convolution_native_group_norm_4 = async_compile.triton('triton_per_fused_convolution_native_group_norm_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 5, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_4(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 256)
    x2 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0 + 512*(((r0_3 + 256*x1) % 16384)) + 8388608*x2 + ((r0_3 + 256*x1) // 16384)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (4*x0 + ((r0_3 + 256*x1) // 16384)), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp8 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp9 = tl.full([XBLOCK, 1], 256, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = (tmp8 / tmp10)
    tmp12 = tmp4 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp11, None)
    tl.store(out_ptr1 + (x4), tmp16, None)
    tl.store(out_ptr2 + (x4), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/oe/coe3p7js3hcr6foggx2kyfwk5zenqghdc22xnuzwllgb7guf4juc.py
# Topologically Sorted Source Nodes: [z, sample, hidden_states], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states => convert_element_type, var_mean, view
#   sample => convolution_1
#   z => convolution
# Graph fragment:
#   %buf4 : Tensor "f32[4, 32, 1, 1, 4, 256][32768, 4, 131072, 131072, 1, 128]cuda:0" = PlaceHolder[target=buf4]
#   %buf5 : Tensor "f32[4, 32, 1, 1, 4, 256][32768, 4, 131072, 131072, 1, 128]cuda:0" = PlaceHolder[target=buf5]
#   %buf6 : Tensor "f32[4, 32, 1, 1, 4, 256][32768, 4, 131072, 131072, 1, 128]cuda:0" = PlaceHolder[target=buf6]
#   %convolution : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %view : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf7,%buf8,%buf9
triton_per_fused_convolution_native_group_norm_5 = async_compile.triton('triton_per_fused_convolution_native_group_norm_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1597440, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 128)
    x1 = xindex // 128
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 16384*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_2 + 16384*x1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 128*r0_2 + 16384*x1), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/44/c44obyxd3xbbkqg42ld5bsvbm3buh4ne4ale2ckt3vivhrejtv35.py
# Topologically Sorted Source Nodes: [z, sample, hidden_states], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states => convert_element_type, var_mean, view
#   sample => convolution_1
#   z => convolution
# Graph fragment:
#   %buf7 : Tensor "f32[4, 32, 1, 1, 4, 2][256, 4, 1024, 1024, 1, 128]cuda:0" = PlaceHolder[target=buf7]
#   %buf8 : Tensor "f32[4, 32, 1, 1, 4, 2][256, 4, 1024, 1024, 1, 128]cuda:0" = PlaceHolder[target=buf8]
#   %buf9 : Tensor "f32[4, 32, 1, 1, 4, 2][256, 4, 1024, 1024, 1, 128]cuda:0" = PlaceHolder[target=buf9]
#   %convolution : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %view : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf10,%buf11,%buf12
triton_per_fused_convolution_native_group_norm_6 = async_compile.triton('triton_per_fused_convolution_native_group_norm_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 24576, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 128)
    x1 = xindex // 128
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 256*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_2 + 256*x1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 128*r0_2 + 256*x1), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/se/csejef3chysb3olrmr7wio3c2dh6hn4p7jwq7zlmycfzbms4sjtm.py
# Topologically Sorted Source Nodes: [z, sample, hidden_states], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states => convert_element_type, var_mean, view
#   sample => convolution_1
#   z => convolution
# Graph fragment:
#   %buf10 : Tensor "f32[4, 32, 1, 1, 4][128, 4, 512, 512, 1]cuda:0" = PlaceHolder[target=buf10]
#   %buf11 : Tensor "f32[4, 32, 1, 1, 4][128, 4, 512, 512, 1]cuda:0" = PlaceHolder[target=buf11]
#   %buf12 : Tensor "f32[4, 32, 1, 1, 4][128, 4, 512, 512, 1]cuda:0" = PlaceHolder[target=buf12]
#   %convolution : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %view : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_1,%buf14
triton_per_fused_convolution_native_group_norm_7 = async_compile.triton('triton_per_fused_convolution_native_group_norm_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 4},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2048, 'r0_': 3072}}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 4
    R0_BLOCK: tl.constexpr = 4
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 4*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 4*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 4*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/eq/ceqsbudb6yyxgxqtk7do2w6zwqxpcghtfg3rbtefv2do32g2opsq.py
# Topologically Sorted Source Nodes: [z, sample, hidden_states, hidden_states_1], Original ATen: [aten.convolution, aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states => add, add_1, convert_element_type, mul, mul_1, rsqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, var_mean, view, view_1
#   hidden_states_1 => convert_element_type_5, mul_2, sigmoid
#   sample => convolution_1
#   z => convolution
# Graph fragment:
#   %buf3 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf3]
#   %arg4_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %getitem_1 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf14 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf14]
#   %arg6_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg7_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %add_1 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=add_1]
#   %convolution : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %view : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %getitem_1), kwargs = {})
#   %add : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %view_1 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul, [4, 512, 128, 128]), kwargs = {})
#   %unsqueeze : Tensor "f16[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f16[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 2), kwargs = {})
#   %unsqueeze_2 : Tensor "f16[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 3), kwargs = {})
#   %mul_1 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_2), kwargs = {})
#   %unsqueeze_3 : Tensor "f16[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, 0), kwargs = {})
#   %unsqueeze_4 : Tensor "f16[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_3, 2), kwargs = {})
#   %unsqueeze_5 : Tensor "f16[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 3), kwargs = {})
#   %add_1 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %sigmoid : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_1,), kwargs = {})
#   %mul_2 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %sigmoid), kwargs = {})
#   %convert_element_type_5 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.float16), kwargs = {})
#   return %add_1,%convert_element_type_5
triton_poi_fused_convolution_native_group_norm_silu_8 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2097152, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr1': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 268435456, 'x': 201329664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y4 = yindex
    y0 = (yindex % 32)
    y2 = yindex // 524288
    tmp0 = tl.load(in_ptr0 + (x3 + 16*y4), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 262144.0
    tmp8 = (tmp6 / tmp7)
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = tl.sigmoid(tmp18)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr1 + (x3 + 16*y4), tmp21, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/yh/cyhabjbpiy2gnv2mpnbj3rdc7fd7j6lxrhbjsco6igtcqzw3lfdf.py
# Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_1 => convert_element_type_5, mul_2, sigmoid
#   hidden_states_2 => convolution_2
# Graph fragment:
#   %arg8_1 : Tensor "f16[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = PlaceHolder[target=arg8_1]
#   %sigmoid : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_1,), kwargs = {})
#   %mul_2 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %sigmoid), kwargs = {})
#   %convert_element_type_5 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.float16), kwargs = {})
#   %convolution_2 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_5, %arg8_1, %arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf18
triton_poi_fused_convolution_silu_9 = async_compile.triton('triton_poi_fused_convolution_silu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 9437184, 'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/nx/cnxbqf63t2lt4i5j3cxowxlzjws6gq47mkl4ejgnmvman6qucjze.py
# Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.view, aten.native_group_norm]
# Source node to ATen node mapping:
#   add => add_4
#   group_norm_2 => convert_element_type_12, var_mean_2, view_5
#   hidden_states_4 => convert_element_type_11, mul_5, sigmoid_1
#   hidden_states_6 => convolution_3
#   output_tensor => div
#   sample => convolution_1
#   view => view_4
#   z => convolution
# Graph fragment:
#   %buf3 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf3]
#   %arg4_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %buf35 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf35]
#   %arg13_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %convolution : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_1 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3,), kwargs = {})
#   %mul_5 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %sigmoid_1), kwargs = {})
#   %convert_element_type_11 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5, torch.float16), kwargs = {})
#   %convolution_3 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_11, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_4 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %convolution_3), kwargs = {})
#   %div : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_4, 1), kwargs = {})
#   %view_4 : Tensor "f16[4, 512, 16384][8388608, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%div, [4, 512, 16384]), kwargs = {})
#   %convert_element_type_12 : Tensor "f32[4, 512, 16384][8388608, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.float32), kwargs = {})
#   %view_5 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_12, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_5, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf36,%buf37,%buf38
triton_per_fused_add_convolution_div_native_group_norm_silu_view_10 = async_compile.triton('triton_per_fused_add_convolution_div_native_group_norm_silu_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_div_native_group_norm_silu_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 5, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_add_convolution_div_native_group_norm_silu_view_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 256)
    x2 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0 + 512*(((r0_3 + 256*x1) % 16384)) + 8388608*x2 + ((r0_3 + 256*x1) // 16384)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (4*x0 + ((r0_3 + 256*x1) // 16384)), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (4*x0 + 512*(((r0_3 + 256*x1) % 16384)) + 8388608*x2 + ((r0_3 + 256*x1) // 16384)), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (4*x0 + ((r0_3 + 256*x1) // 16384)), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp14 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    tmp15 = tl.full([XBLOCK, 1], 256, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = (tmp14 / tmp16)
    tmp18 = tmp10 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.sum(tmp20, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp17, None)
    tl.store(out_ptr1 + (x4), tmp22, None)
    tl.store(out_ptr2 + (x4), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/ej/cej6su6t4sjuf23qiz4eqpv2xhoctamwx5oklsfsgm7hw7c2eehh.py
# Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2, hidden_states_8, query, key, value], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.view, aten.native_group_norm, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   add => add_4
#   group_norm_2 => add_5, add_6, convert_element_type_12, convert_element_type_13, mul_6, mul_7, rsqrt_2, sub_2, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, var_mean_2, view_5, view_6
#   hidden_states_4 => convert_element_type_11, mul_5, sigmoid_1
#   hidden_states_6 => convolution_3
#   hidden_states_8 => permute_2
#   key => clone_2
#   output_tensor => div
#   query => clone_1
#   sample => convolution_1
#   value => clone_3
#   view => view_4
#   z => convolution
# Graph fragment:
#   %buf3 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf3]
#   %arg4_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %buf35 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf35]
#   %arg13_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %getitem_5 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_5]
#   %buf46 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf46]
#   %arg14_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %arg15_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg15_1]
#   %convert_element_type_13 : Tensor "f16[4, 512, 16384][8388608, 1, 512]cuda:0" = PlaceHolder[target=convert_element_type_13]
#   %convolution : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_1 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3,), kwargs = {})
#   %mul_5 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %sigmoid_1), kwargs = {})
#   %convert_element_type_11 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5, torch.float16), kwargs = {})
#   %convolution_3 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_11, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_4 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %convolution_3), kwargs = {})
#   %div : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_4, 1), kwargs = {})
#   %view_4 : Tensor "f16[4, 512, 16384][8388608, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%div, [4, 512, 16384]), kwargs = {})
#   %convert_element_type_12 : Tensor "f32[4, 512, 16384][8388608, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.float32), kwargs = {})
#   %view_5 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_12, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_5, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %getitem_5), kwargs = {})
#   %add_5 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_6 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %view_6 : Tensor "f32[4, 512, 16384][8388608, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_6, [4, 512, 16384]), kwargs = {})
#   %unsqueeze_12 : Tensor "f16[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg14_1, 0), kwargs = {})
#   %unsqueeze_13 : Tensor "f16[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, 2), kwargs = {})
#   %mul_7 : Tensor "f32[4, 512, 16384][8388608, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f16[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg15_1, 0), kwargs = {})
#   %unsqueeze_15 : Tensor "f16[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, 2), kwargs = {})
#   %add_6 : Tensor "f32[4, 512, 16384][8388608, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_15), kwargs = {})
#   %convert_element_type_13 : Tensor "f16[4, 512, 16384][8388608, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.float16), kwargs = {})
#   %permute_2 : Tensor "f16[4, 16384, 512][8388608, 1, 16384]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_13, [0, 2, 1]), kwargs = {})
#   %clone_1 : Tensor "f16[4, 16384, 512][8388608, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_2 : Tensor "f16[4, 16384, 512][8388608, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_3 : Tensor "f16[4, 16384, 512][8388608, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
#   return %convert_element_type_13,%clone_1,%clone_2,%clone_3
triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_transpose_view_11 = async_compile.triton('triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_transpose_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2097152, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp16', 'in_ptr7': '*fp16', 'out_ptr1': '*fp16', 'out_ptr2': '*fp16', 'out_ptr3': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_transpose_view_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 268435456, 'x': 536875008}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_transpose_view_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y4 = yindex
    y0 = (yindex % 32)
    y2 = yindex // 524288
    tmp0 = tl.load(in_ptr0 + (x3 + 16*y4), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x3 + 16*y4), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tl.load(in_ptr7 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = 262144.0
    tmp14 = (tmp12 / tmp13)
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp11 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr1 + (x3 + 16*y4), tmp25, xmask & ymask)
    tl.store(out_ptr2 + (x3 + 16*y4), tmp25, xmask & ymask)
    tl.store(out_ptr3 + (x3 + 16*y4), tmp25, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/7n/c7nwgkwee3pdzlxqld4pd3r63n3tvjoik5fpwmgsvppwmhqrl53b.py
# Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2, hidden_states_8, query, view_1, query_1, key, view_2, key_1, value, view_3, value_1, hidden_states_9], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.view, aten.native_group_norm, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   add => add_4
#   group_norm_2 => add_5, add_6, convert_element_type_12, convert_element_type_13, mul_6, mul_7, rsqrt_2, sub_2, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, var_mean_2, view_5, view_6
#   hidden_states_4 => convert_element_type_11, mul_5, sigmoid_1
#   hidden_states_6 => convolution_3
#   hidden_states_8 => permute_2
#   hidden_states_9 => _scaled_dot_product_efficient_attention
#   key => add_8, view_10
#   key_1 => permute_7
#   output_tensor => div
#   query => add_7, clone_1, mm, permute_3, view_7, view_8
#   query_1 => permute_6
#   sample => convolution_1
#   value => add_9, view_12
#   value_1 => permute_8
#   view => view_4
#   view_1 => view_13
#   view_2 => view_14
#   view_3 => view_15
#   z => convolution
# Graph fragment:
#   %clone_1 : Tensor "f16[4, 16384, 512][8388608, 512, 1]cuda:0" = PlaceHolder[target=clone_1]
#   %arg16_1 : Tensor "f16[512, 512][512, 1]cuda:0" = PlaceHolder[target=arg16_1]
#   %mm : Tensor "f16[65536, 512][512, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg17_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %convolution : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_1 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3,), kwargs = {})
#   %mul_5 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %sigmoid_1), kwargs = {})
#   %convert_element_type_11 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5, torch.float16), kwargs = {})
#   %convolution_3 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_11, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_4 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %convolution_3), kwargs = {})
#   %div : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_4, 1), kwargs = {})
#   %view_4 : Tensor "f16[4, 512, 16384][8388608, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%div, [4, 512, 16384]), kwargs = {})
#   %convert_element_type_12 : Tensor "f32[4, 512, 16384][8388608, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.float32), kwargs = {})
#   %view_5 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_12, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_5, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %getitem_5), kwargs = {})
#   %add_5 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_6 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %view_6 : Tensor "f32[4, 512, 16384][8388608, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_6, [4, 512, 16384]), kwargs = {})
#   %unsqueeze_12 : Tensor "f16[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg14_1, 0), kwargs = {})
#   %unsqueeze_13 : Tensor "f16[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, 2), kwargs = {})
#   %mul_7 : Tensor "f32[4, 512, 16384][8388608, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f16[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg15_1, 0), kwargs = {})
#   %unsqueeze_15 : Tensor "f16[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, 2), kwargs = {})
#   %add_6 : Tensor "f32[4, 512, 16384][8388608, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_15), kwargs = {})
#   %convert_element_type_13 : Tensor "f16[4, 512, 16384][8388608, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.float16), kwargs = {})
#   %permute_2 : Tensor "f16[4, 16384, 512][8388608, 1, 16384]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_13, [0, 2, 1]), kwargs = {})
#   %clone_1 : Tensor "f16[4, 16384, 512][8388608, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
#   %view_7 : Tensor "f16[65536, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_1, [65536, 512]), kwargs = {})
#   %permute_3 : Tensor "f16[512, 512][1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg16_1, [1, 0]), kwargs = {})
#   %mm : Tensor "f16[65536, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_7, %permute_3), kwargs = {})
#   %view_8 : Tensor "f16[4, 16384, 512][8388608, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [4, 16384, 512]), kwargs = {})
#   %add_7 : Tensor "f16[4, 16384, 512][8388608, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_8, %arg17_1), kwargs = {})
#   %view_13 : Tensor "f16[4, 16384, 1, 512][8388608, 512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_7, [4, -1, 1, 512]), kwargs = {})
#   %permute_6 : Tensor "f16[4, 1, 16384, 512][8388608, 512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_13, [0, 2, 1, 3]), kwargs = {})
#   %view_10 : Tensor "f16[4, 16384, 512][8388608, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [4, 16384, 512]), kwargs = {})
#   %add_8 : Tensor "f16[4, 16384, 512][8388608, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_10, %arg19_1), kwargs = {})
#   %view_14 : Tensor "f16[4, 16384, 1, 512][8388608, 512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_8, [4, -1, 1, 512]), kwargs = {})
#   %permute_7 : Tensor "f16[4, 1, 16384, 512][8388608, 512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_14, [0, 2, 1, 3]), kwargs = {})
#   %view_12 : Tensor "f16[4, 16384, 512][8388608, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [4, 16384, 512]), kwargs = {})
#   %add_9 : Tensor "f16[4, 16384, 512][8388608, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_12, %arg21_1), kwargs = {})
#   %view_15 : Tensor "f16[4, 16384, 1, 512][8388608, 512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_9, [4, -1, 1, 512]), kwargs = {})
#   %permute_8 : Tensor "f16[4, 1, 16384, 512][8388608, 512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_15, [0, 2, 1, 3]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_6, %permute_7, %permute_8, None, False), kwargs = {})
#   return %mm,%buf55
triton_tem_fused__scaled_dot_product_efficient_attention__unsafe_view_add_clone_convolution_div_mm_native_group_norm_silu_t_transpose_view_12 = async_compile.triton('triton_tem_fused__scaled_dot_product_efficient_attention__unsafe_view_add_clone_convolution_div_mm_native_group_norm_silu_t_transpose_view_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=4,
triton_meta={'signature': {'arg_A': '*fp16', 'arg_B': '*fp16', 'in_ptr2': '*fp16', 'out_ptr1': '*fp16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused__scaled_dot_product_efficient_attention__unsafe_view_add_clone_convolution_div_mm_native_group_norm_silu_t_transpose_view_12', 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'ALLOW_TF32': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}},

)
@triton.jit
def triton_tem_fused__scaled_dot_product_efficient_attention__unsafe_view_add_clone_convolution_div_mm_native_group_norm_silu_t_transpose_view_12(arg_A, arg_B, in_ptr2, out_ptr1):
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    GROUP_M : tl.constexpr = 8
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 65536
    N = 512
    K = 512
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 512
    stride_ak = 1
    stride_bk = 1
    stride_bn = 512

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 512*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 512*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 512*idx_n, xindex.shape)).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 512*idx_m
    tmp0 = tl.load(in_ptr2 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 512*idx_m, acc.shape)), tmp1, mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/tq/ctqfvm7to6sun7edhiwmn32tx5iqlw7bh73vbmyr6dwoqufytusu.py
# Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, , hidden_states_12, transpose_7, hidden_states_14, hidden_states_15, hidden_states_16], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.addmm, aten.view, aten.transpose]
# Source node to ATen node mapping:
#    => add_tensor
#   add => add_4
#   hidden_states_12 => view_18
#   hidden_states_14 => view_19
#   hidden_states_15 => add_10
#   hidden_states_16 => div_1
#   hidden_states_4 => convert_element_type_11, mul_5, sigmoid_1
#   hidden_states_6 => convolution_3
#   output_tensor => div
#   sample => convolution_1
#   transpose_7 => permute_11
#   z => convolution
# Graph fragment:
#   %buf63 : Tensor "f16[65536, 512][512, 1]cuda:0" = PlaceHolder[target=buf63]
#   %arg23_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg23_1]
#   %buf3 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf3]
#   %arg4_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %buf35 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf35]
#   %arg13_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %convolution : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_1 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_3,), kwargs = {})
#   %mul_5 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %sigmoid_1), kwargs = {})
#   %convert_element_type_11 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5, torch.float16), kwargs = {})
#   %convolution_3 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_11, %arg12_1, %arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_4 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_1, %convolution_3), kwargs = {})
#   %div : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_4, 1), kwargs = {})
#   %add_tensor : Tensor "f16[65536, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %arg23_1), kwargs = {})
#   %view_18 : Tensor "f16[4, 16384, 512][8388608, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [4, 16384, 512]), kwargs = {})
#   %permute_11 : Tensor "f16[4, 512, 16384][8388608, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_18, [0, 2, 1]), kwargs = {})
#   %view_19 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_11, [4, 512, 128, 128]), kwargs = {})
#   %add_10 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_19, %div), kwargs = {})
#   %div_1 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_10, 1), kwargs = {})
#   return %div_1
triton_poi_fused_add_addmm_convolution_div_silu_transpose_view_13 = async_compile.triton('triton_poi_fused_add_addmm_convolution_div_silu_transpose_view_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_div_silu_transpose_view_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 335547392}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_div_silu_transpose_view_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp2 + tmp11
    tmp13 = tmp12 * tmp10
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/3x/c3xrddqevfjymas4ws74q2cmgibtrhju43saktj3bgzts5tx3qds.py
# Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_17 => clone_5, convert_element_type_25, var_mean_3, view_20
# Graph fragment:
#   %div_1 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=div_1]
#   %clone_5 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_1,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_25 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_5, torch.float32), kwargs = {})
#   %view_20 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_25, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_20, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf65,%buf66,%buf67
triton_per_fused_clone_native_group_norm_14 = async_compile.triton('triton_per_fused_clone_native_group_norm_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_native_group_norm_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 5, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_clone_native_group_norm_14(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 256)
    x2 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0 + 512*(((r0_3 + 256*x1) % 16384)) + 8388608*x2 + ((r0_3 + 256*x1) // 16384)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp7 = tl.full([XBLOCK, 1], 256, tl.int32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = (tmp6 / tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tmp10 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp9, None)
    tl.store(out_ptr1 + (x4), tmp14, None)
    tl.store(out_ptr2 + (x4), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/vh/cvh3fo4reqldpcp7zdpmz5wj5m6jdbyhgp2kgmg6gxc6gybkeasb.py
# Topologically Sorted Source Nodes: [hidden_states_17, hidden_states_18], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_17 => add_11, add_12, clone_5, convert_element_type_25, mul_8, mul_9, rsqrt_3, sub_3, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, var_mean_3, view_20, view_21
#   hidden_states_18 => convert_element_type_30, mul_10, sigmoid_2
# Graph fragment:
#   %div_1 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=div_1]
#   %getitem_11 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_11]
#   %buf75 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf75]
#   %arg24_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg24_1]
#   %arg25_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg25_1]
#   %add_12 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=add_12]
#   %clone_5 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_1,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_25 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_5, torch.float32), kwargs = {})
#   %view_20 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_25, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_20, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_20, %getitem_11), kwargs = {})
#   %add_11 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-06), kwargs = {})
#   %rsqrt_3 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %mul_8 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %view_21 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_8, [4, 512, 128, 128]), kwargs = {})
#   %unsqueeze_16 : Tensor "f16[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg24_1, 0), kwargs = {})
#   %unsqueeze_17 : Tensor "f16[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, 2), kwargs = {})
#   %unsqueeze_18 : Tensor "f16[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_17, 3), kwargs = {})
#   %mul_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_21, %unsqueeze_18), kwargs = {})
#   %unsqueeze_19 : Tensor "f16[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg25_1, 0), kwargs = {})
#   %unsqueeze_20 : Tensor "f16[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_19, 2), kwargs = {})
#   %unsqueeze_21 : Tensor "f16[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, 3), kwargs = {})
#   %add_12 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %unsqueeze_21), kwargs = {})
#   %sigmoid_2 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_12,), kwargs = {})
#   %mul_10 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, %sigmoid_2), kwargs = {})
#   %convert_element_type_30 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10, torch.float16), kwargs = {})
#   return %add_12,%convert_element_type_30
triton_poi_fused_clone_native_group_norm_silu_15 = async_compile.triton('triton_poi_fused_clone_native_group_norm_silu_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2097152, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr1': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_native_group_norm_silu_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 268435456, 'x': 201328640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_native_group_norm_silu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y4 = yindex
    y0 = (yindex % 32)
    y2 = yindex // 524288
    tmp0 = tl.load(in_ptr0 + (x3 + 16*y4), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = 262144.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr1 + (x3 + 16*y4), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/nv/cnvzhvtqcpdnn6ovkflprnqdwxropnlqvuuixwj57aae7tvedsmm.py
# Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, sample_1, hidden_states_24], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_2 => add_15
#   hidden_states_21 => convert_element_type_36, mul_13, sigmoid_3
#   hidden_states_23 => convolution_5
#   hidden_states_24 => clone_7, var_mean_5, view_24
#   output_tensor_1 => div_2
#   sample_1 => convert_element_type_37
# Graph fragment:
#   %div_1 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=div_1]
#   %buf96 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf96]
#   %arg31_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg31_1]
#   %sigmoid_3 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_14,), kwargs = {})
#   %mul_13 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14, %sigmoid_3), kwargs = {})
#   %convert_element_type_36 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_13, torch.float16), kwargs = {})
#   %convolution_5 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_36, %arg30_1, %arg31_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_15 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_1, %convolution_5), kwargs = {})
#   %div_2 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_15, 1), kwargs = {})
#   %convert_element_type_37 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.float32), kwargs = {})
#   %clone_7 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_37,), kwargs = {memory_format: torch.contiguous_format})
#   %view_24 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_7, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_24, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf97,%buf98,%buf99
triton_per_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_16 = async_compile.triton('triton_per_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 5, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 256)
    x2 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0 + 512*(((r0_3 + 256*x1) % 16384)) + 8388608*x2 + ((r0_3 + 256*x1) // 16384)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (4*x0 + 512*(((r0_3 + 256*x1) % 16384)) + 8388608*x2 + ((r0_3 + 256*x1) // 16384)), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (4*x0 + ((r0_3 + 256*x1) // 16384)), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp12 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp13 = tl.full([XBLOCK, 1], 256, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = (tmp12 / tmp14)
    tmp16 = tmp8 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.sum(tmp18, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp15, None)
    tl.store(out_ptr1 + (x4), tmp20, None)
    tl.store(out_ptr2 + (x4), tmp14, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/pw/cpwyreajluuawykubmdvbojfxd5c7ldg7rxmif7neegtlb5opib4.py
# Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, sample_1, hidden_states_24, hidden_states_25], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_2 => add_15
#   hidden_states_21 => convert_element_type_36, mul_13, sigmoid_3
#   hidden_states_23 => convolution_5
#   hidden_states_24 => add_16, add_17, clone_7, mul_14, mul_15, rsqrt_5, sub_5, unsqueeze_28, unsqueeze_29, unsqueeze_30, unsqueeze_31, unsqueeze_32, unsqueeze_33, var_mean_5, view_24, view_25
#   hidden_states_25 => mul_16, sigmoid_4
#   output_tensor_1 => div_2
#   sample_1 => convert_element_type_37
# Graph fragment:
#   %div_1 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=div_1]
#   %buf96 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf96]
#   %arg31_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg31_1]
#   %getitem_15 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_15]
#   %buf107 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf107]
#   %arg5_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %arg32_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg32_1]
#   %add_17 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=add_17]
#   %sigmoid_3 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_14,), kwargs = {})
#   %mul_13 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14, %sigmoid_3), kwargs = {})
#   %convert_element_type_36 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_13, torch.float16), kwargs = {})
#   %convolution_5 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_36, %arg30_1, %arg31_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_15 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_1, %convolution_5), kwargs = {})
#   %div_2 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_15, 1), kwargs = {})
#   %convert_element_type_37 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.float32), kwargs = {})
#   %clone_7 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_37,), kwargs = {memory_format: torch.contiguous_format})
#   %view_24 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_7, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_24, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_24, %getitem_15), kwargs = {})
#   %add_16 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-06), kwargs = {})
#   %rsqrt_5 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_16,), kwargs = {})
#   %mul_14 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_5), kwargs = {})
#   %view_25 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_14, [4, 512, 128, 128]), kwargs = {})
#   %unsqueeze_28 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, 0), kwargs = {})
#   %unsqueeze_29 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_28, 2), kwargs = {})
#   %unsqueeze_30 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_29, 3), kwargs = {})
#   %mul_15 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_25, %unsqueeze_30), kwargs = {})
#   %unsqueeze_31 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg32_1, 0), kwargs = {})
#   %unsqueeze_32 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_31, 2), kwargs = {})
#   %unsqueeze_33 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_32, 3), kwargs = {})
#   %add_17 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %unsqueeze_33), kwargs = {})
#   %sigmoid_4 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_16 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_4), kwargs = {})
#   return %add_17,%mul_16
triton_poi_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_17 = async_compile.triton('triton_poi_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2097152, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 268435456, 'x': 402658304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y4 = yindex
    y0 = (yindex % 32)
    y2 = yindex // 524288
    tmp0 = tl.load(in_ptr0 + (x3 + 16*y4), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x3 + 16*y4), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = 262144.0
    tmp12 = (tmp10 / tmp11)
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp9 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp21 = tl.sigmoid(tmp20)
    tmp22 = tmp20 * tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + 16*y4), tmp22, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/dh/cdhod2qym23tpgr245dx6gdagzbm6a2sqnvziwybpozastik7iov.py
# Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_25 => mul_16, sigmoid_4
#   hidden_states_26 => convolution_6
# Graph fragment:
#   %arg33_1 : Tensor "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = PlaceHolder[target=arg33_1]
#   %sigmoid_4 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_16 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_4), kwargs = {})
#   %convolution_6 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_16, %arg33_1, %arg34_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf111
triton_poi_fused_convolution_silu_18 = async_compile.triton('triton_poi_fused_convolution_silu_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 18874368, 'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_18(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/mu/cmuear66hpbl34nyyeuwi2bgukwp5gvtr5pl6egiqrzqrsmkxgwz.py
# Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26, hidden_states_27], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_25 => mul_16, sigmoid_4
#   hidden_states_26 => convolution_6
#   hidden_states_27 => var_mean_6, view_26
# Graph fragment:
#   %buf112 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf112]
#   %arg34_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg34_1]
#   %sigmoid_4 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_16 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_4), kwargs = {})
#   %convolution_6 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_16, %arg33_1, %arg34_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_26 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_26, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf113,%buf114,%buf115
triton_per_fused_convolution_native_group_norm_silu_19 = async_compile.triton('triton_per_fused_convolution_native_group_norm_silu_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_silu_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 5, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_silu_19(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 256)
    x2 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0 + 512*(((r0_3 + 256*x1) % 16384)) + 8388608*x2 + ((r0_3 + 256*x1) // 16384)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x0 + ((r0_3 + 256*x1) // 16384)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([XBLOCK, 1], 256, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp10, None)
    tl.store(out_ptr1 + (x4), tmp15, None)
    tl.store(out_ptr2 + (x4), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/w5/cw5nt2eqqiht5g752uuacooejfrh6dyvzt6b3taod5trn4wvtshm.py
# Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26, hidden_states_27, hidden_states_28], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_25 => mul_16, sigmoid_4
#   hidden_states_26 => convolution_6
#   hidden_states_27 => add_18, add_19, mul_17, mul_18, rsqrt_6, sub_6, unsqueeze_34, unsqueeze_35, unsqueeze_36, unsqueeze_37, unsqueeze_38, unsqueeze_39, var_mean_6, view_26, view_27
#   hidden_states_28 => mul_19, sigmoid_5
# Graph fragment:
#   %buf112 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf112]
#   %arg34_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg34_1]
#   %getitem_17 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_17]
#   %buf123 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf123]
#   %arg35_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %arg36_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg36_1]
#   %add_19 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=add_19]
#   %sigmoid_4 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_16 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_4), kwargs = {})
#   %convolution_6 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_16, %arg33_1, %arg34_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_26 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_26, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_6 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_26, %getitem_17), kwargs = {})
#   %add_18 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-06), kwargs = {})
#   %rsqrt_6 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
#   %mul_17 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_6), kwargs = {})
#   %view_27 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_17, [4, 512, 128, 128]), kwargs = {})
#   %unsqueeze_34 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg35_1, 0), kwargs = {})
#   %unsqueeze_35 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_34, 2), kwargs = {})
#   %unsqueeze_36 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_35, 3), kwargs = {})
#   %mul_18 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_27, %unsqueeze_36), kwargs = {})
#   %unsqueeze_37 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg36_1, 0), kwargs = {})
#   %unsqueeze_38 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_37, 2), kwargs = {})
#   %unsqueeze_39 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_38, 3), kwargs = {})
#   %add_19 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %unsqueeze_39), kwargs = {})
#   %sigmoid_5 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_19,), kwargs = {})
#   %mul_19 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, %sigmoid_5), kwargs = {})
#   return %add_19,%mul_19
triton_poi_fused_convolution_native_group_norm_silu_20 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2097152, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 268435456, 'x': 402659328}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y4 = yindex
    y0 = (yindex % 32)
    y2 = yindex // 524288
    tmp0 = tl.load(in_out_ptr0 + (x3 + 16*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 262144.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + 16*y4), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/qp/cqpl7h3kgbmab5wfeghl6dinsldi5x772slctkydc4i4mryettzk.py
# Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, sample_1, hidden_states_28, hidden_states_30, add_3, output_tensor_2], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy]
# Source node to ATen node mapping:
#   add_2 => add_15
#   add_3 => add_20
#   hidden_states_21 => convert_element_type_36, mul_13, sigmoid_3
#   hidden_states_23 => convolution_5
#   hidden_states_28 => mul_19, sigmoid_5
#   hidden_states_30 => convolution_7
#   output_tensor_1 => div_2
#   output_tensor_2 => div_3
#   sample_1 => convert_element_type_37
# Graph fragment:
#   %div_1 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=div_1]
#   %buf96 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf96]
#   %arg31_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg31_1]
#   %buf128 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf128]
#   %arg38_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg38_1]
#   %sigmoid_3 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_14,), kwargs = {})
#   %mul_13 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14, %sigmoid_3), kwargs = {})
#   %convert_element_type_36 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_13, torch.float16), kwargs = {})
#   %convolution_5 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convert_element_type_36, %arg30_1, %arg31_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_15 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_1, %convolution_5), kwargs = {})
#   %div_2 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_15, 1), kwargs = {})
#   %convert_element_type_37 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.float32), kwargs = {})
#   %sigmoid_5 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_19,), kwargs = {})
#   %mul_19 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, %sigmoid_5), kwargs = {})
#   %convolution_7 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_19, %arg37_1, %arg38_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_20 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_37, %convolution_7), kwargs = {})
#   %div_3 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_20, 1.0), kwargs = {})
#   return %div_3
triton_poi_fused__to_copy_add_convolution_div_silu_21 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_div_silu_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_div_silu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 536873984}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_div_silu_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_out_ptr0 + (x2), None)
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tmp11 * tmp5
    tl.store(in_out_ptr0 + (x2), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/iq/ciqa7jbkcg5jioxxiv377ui25pdw6wpmxdlyn2jjkdc4u2ilhi55.py
# Topologically Sorted Source Nodes: [hidden_states_31], Original ATen: [aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_31 => clone_9, var_mean_7, view_28
# Graph fragment:
#   %div_3 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=div_3]
#   %clone_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_3,), kwargs = {memory_format: torch.contiguous_format})
#   %view_28 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_9, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_28, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf130,%buf131,%buf132
triton_per_fused_clone_native_group_norm_22 = async_compile.triton('triton_per_fused_clone_native_group_norm_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_native_group_norm_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 5, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_clone_native_group_norm_22(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 256)
    x2 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0 + 512*(((r0_3 + 256*x1) % 16384)) + 8388608*x2 + ((r0_3 + 256*x1) // 16384)), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp6 = tl.full([XBLOCK, 1], 256, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = (tmp5 / tmp7)
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp8, None)
    tl.store(out_ptr1 + (x4), tmp13, None)
    tl.store(out_ptr2 + (x4), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/rd/crdiih2wazzg4yjdz6die5ibivkfvnfvnjsf6demnfkkzxbcaaob.py
# Topologically Sorted Source Nodes: [hidden_states_31, hidden_states_32], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_31 => add_21, add_22, clone_9, mul_20, mul_21, rsqrt_7, sub_7, unsqueeze_40, unsqueeze_41, unsqueeze_42, unsqueeze_43, unsqueeze_44, unsqueeze_45, var_mean_7, view_28, view_29
#   hidden_states_32 => mul_22, sigmoid_6
# Graph fragment:
#   %div_3 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=div_3]
#   %getitem_19 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_19]
#   %buf140 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf140]
#   %arg39_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg39_1]
#   %arg40_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg40_1]
#   %add_22 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=add_22]
#   %clone_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_3,), kwargs = {memory_format: torch.contiguous_format})
#   %view_28 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_9, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_28, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_7 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_28, %getitem_19), kwargs = {})
#   %add_21 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_18, 1e-06), kwargs = {})
#   %rsqrt_7 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_21,), kwargs = {})
#   %mul_20 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_7), kwargs = {})
#   %view_29 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_20, [4, 512, 128, 128]), kwargs = {})
#   %unsqueeze_40 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg39_1, 0), kwargs = {})
#   %unsqueeze_41 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_40, 2), kwargs = {})
#   %unsqueeze_42 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_41, 3), kwargs = {})
#   %mul_21 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_29, %unsqueeze_42), kwargs = {})
#   %unsqueeze_43 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg40_1, 0), kwargs = {})
#   %unsqueeze_44 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_43, 2), kwargs = {})
#   %unsqueeze_45 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_44, 3), kwargs = {})
#   %add_22 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %unsqueeze_45), kwargs = {})
#   %sigmoid_6 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_22,), kwargs = {})
#   %mul_22 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %sigmoid_6), kwargs = {})
#   return %add_22,%mul_22
triton_poi_fused_clone_native_group_norm_silu_23 = async_compile.triton('triton_poi_fused_clone_native_group_norm_silu_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2097152, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_native_group_norm_silu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 268435456, 'x': 402657280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_native_group_norm_silu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y4 = yindex
    y0 = (yindex % 32)
    y2 = yindex // 524288
    tmp0 = tl.load(in_ptr0 + (x3 + 16*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 262144.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + 16*y4), tmp15, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/ax/caxwuyxrkftm6qy2bapoaugliie45ax4szk64e5zjgb3nab6oi7v.py
# Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_38], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_4 => add_25
#   hidden_states_35 => mul_25, sigmoid_7
#   hidden_states_37 => convolution_9
#   hidden_states_38 => clone_11, var_mean_9, view_32
#   output_tensor_3 => div_4
# Graph fragment:
#   %div_3 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=div_3]
#   %buf161 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf161]
#   %arg46_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg46_1]
#   %sigmoid_7 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_24,), kwargs = {})
#   %mul_25 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %sigmoid_7), kwargs = {})
#   %convolution_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_25, %arg45_1, %arg46_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_25 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %convolution_9), kwargs = {})
#   %div_4 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_25, 1.0), kwargs = {})
#   %clone_11 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_32 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_11, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_32, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf162,%buf163,%buf164
triton_per_fused_add_clone_convolution_div_native_group_norm_silu_24 = async_compile.triton('triton_per_fused_add_clone_convolution_div_native_group_norm_silu_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_convolution_div_native_group_norm_silu_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 5, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_add_clone_convolution_div_native_group_norm_silu_24(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 256
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 256)
    x2 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0 + 512*(((r0_3 + 256*x1) % 16384)) + 8388608*x2 + ((r0_3 + 256*x1) // 16384)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x0 + 512*(((r0_3 + 256*x1) % 16384)) + 8388608*x2 + ((r0_3 + 256*x1) // 16384)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (4*x0 + ((r0_3 + 256*x1) // 16384)), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([XBLOCK, 1], 256, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp7 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp14, None)
    tl.store(out_ptr1 + (x4), tmp19, None)
    tl.store(out_ptr2 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/hc/chchvddvpywdsk2ru4gzihotdlxn5pogq4f23cus2ngtlvjsqpfq.py
# Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_38, hidden_states_39], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_4 => add_25
#   hidden_states_35 => mul_25, sigmoid_7
#   hidden_states_37 => convolution_9
#   hidden_states_38 => add_26, add_27, clone_11, mul_26, mul_27, rsqrt_9, sub_9, unsqueeze_52, unsqueeze_53, unsqueeze_54, unsqueeze_55, unsqueeze_56, unsqueeze_57, var_mean_9, view_32, view_33
#   hidden_states_39 => mul_28, sigmoid_8
#   output_tensor_3 => div_4
# Graph fragment:
#   %div_3 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=div_3]
#   %buf161 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf161]
#   %arg46_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg46_1]
#   %getitem_23 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_23]
#   %buf172 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf172]
#   %arg47_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg47_1]
#   %arg48_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg48_1]
#   %add_27 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=add_27]
#   %sigmoid_7 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_24,), kwargs = {})
#   %mul_25 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %sigmoid_7), kwargs = {})
#   %convolution_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_25, %arg45_1, %arg46_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_25 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %convolution_9), kwargs = {})
#   %div_4 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_25, 1.0), kwargs = {})
#   %clone_11 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_32 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_11, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_32, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_9 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_32, %getitem_23), kwargs = {})
#   %add_26 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_22, 1e-06), kwargs = {})
#   %rsqrt_9 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_26,), kwargs = {})
#   %mul_26 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %rsqrt_9), kwargs = {})
#   %view_33 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_26, [4, 512, 128, 128]), kwargs = {})
#   %unsqueeze_52 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg47_1, 0), kwargs = {})
#   %unsqueeze_53 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_52, 2), kwargs = {})
#   %unsqueeze_54 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_53, 3), kwargs = {})
#   %mul_27 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_33, %unsqueeze_54), kwargs = {})
#   %unsqueeze_55 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg48_1, 0), kwargs = {})
#   %unsqueeze_56 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_55, 2), kwargs = {})
#   %unsqueeze_57 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_56, 3), kwargs = {})
#   %add_27 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %unsqueeze_57), kwargs = {})
#   %sigmoid_8 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_27,), kwargs = {})
#   %mul_28 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %sigmoid_8), kwargs = {})
#   return %add_27,%mul_28
triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_25 = async_compile.triton('triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2097152, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 268435456, 'x': 536877056}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y4 = yindex
    y0 = (yindex % 32)
    y2 = yindex // 524288
    tmp0 = tl.load(in_ptr0 + (x3 + 16*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + 16*y4), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 262144.0
    tmp11 = (tmp9 / tmp10)
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sigmoid(tmp19)
    tmp21 = tmp19 * tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + 16*y4), tmp21, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/pv/cpv7n2rbma4fb7kxw2jdzcq5vzsrad3mg7ip5itydzv2v2pyoku5.py
# Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
# Source node to ATen node mapping:
#   add_4 => add_25
#   add_5 => add_30
#   hidden_states_35 => mul_25, sigmoid_7
#   hidden_states_37 => convolution_9
#   hidden_states_42 => mul_31, sigmoid_9
#   hidden_states_44 => convolution_11
#   hidden_states_45 => _unsafe_index, add_31, add_32, add_33, add_34, clone_13, convert_element_type_38, convert_element_type_39, convert_element_type_40, convert_element_type_41, iota, iota_1, mul_32, mul_33, mul_34, mul_35, unsqueeze_64
#   output_tensor_3 => div_4
#   output_tensor_4 => div_5
# Graph fragment:
#   %div_3 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=div_3]
#   %buf161 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf161]
#   %arg46_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg46_1]
#   %buf193 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf193]
#   %arg54_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg54_1]
#   %sigmoid_7 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_24,), kwargs = {})
#   %mul_25 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %sigmoid_7), kwargs = {})
#   %convolution_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_25, %arg45_1, %arg46_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_25 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %convolution_9), kwargs = {})
#   %div_4 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_25, 1.0), kwargs = {})
#   %sigmoid_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_29,), kwargs = {})
#   %mul_31 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, %sigmoid_9), kwargs = {})
#   %convolution_11 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_31, %arg53_1, %arg54_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_30 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, %convolution_11), kwargs = {})
#   %div_5 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_30, 1.0), kwargs = {})
#   %iota : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_32 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_31 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, 0), kwargs = {})
#   %convert_element_type_38 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_31, torch.float32), kwargs = {})
#   %add_32 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_38, 0.0), kwargs = {})
#   %mul_33 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_32, 0.5), kwargs = {})
#   %convert_element_type_39 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_33, torch.int64), kwargs = {})
#   %unsqueeze_64 : Tensor "i64[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_39, -1), kwargs = {})
#   %iota_1 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_34 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_1, 1), kwargs = {})
#   %add_33 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, 0), kwargs = {})
#   %convert_element_type_40 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_33, torch.float32), kwargs = {})
#   %add_34 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_40, 0.0), kwargs = {})
#   %mul_35 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_34, 0.5), kwargs = {})
#   %convert_element_type_41 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_35, torch.int64), kwargs = {})
#   %_unsafe_index : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_5, [None, None, %unsqueeze_64, %convert_element_type_41]), kwargs = {})
#   %clone_13 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.channels_last})
#   return %clone_13
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_26 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 131072) % 256)
    x1 = ((xindex // 512) % 256)
    x0 = (xindex % 512)
    x3 = xindex // 33554432
    x5 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (x0 + 512*tmp8 + 65536*tmp4 + 8388608*x3), None)
    tmp10 = tl.load(in_ptr1 + (x0 + 512*tmp8 + 65536*tmp4 + 8388608*x3), None)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tl.load(in_ptr3 + (x0 + 512*tmp8 + 65536*tmp4 + 8388608*x3), None)
    tmp18 = tmp16 + tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tmp19 * tmp14
    tl.store(out_ptr0 + (x5), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/ns/cnsk6uy3to7bvje476svn4sdah54l3s5jyd7w7duiqjga5sdbfg2.py
# Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_47], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_4 => add_25
#   add_5 => add_30
#   hidden_states_35 => mul_25, sigmoid_7
#   hidden_states_37 => convolution_9
#   hidden_states_42 => mul_31, sigmoid_9
#   hidden_states_44 => convolution_11
#   hidden_states_45 => _unsafe_index, add_31, add_32, add_33, add_34, clone_13, convert_element_type_38, convert_element_type_39, convert_element_type_40, convert_element_type_41, iota, iota_1, mul_32, mul_33, mul_34, mul_35, unsqueeze_64
#   hidden_states_46 => convolution_12
#   hidden_states_47 => clone_14, var_mean_11, view_36
#   output_tensor_3 => div_4
#   output_tensor_4 => div_5
# Graph fragment:
#   %buf196 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf196]
#   %arg56_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg56_1]
#   %sigmoid_7 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_24,), kwargs = {})
#   %mul_25 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %sigmoid_7), kwargs = {})
#   %convolution_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_25, %arg45_1, %arg46_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_25 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %convolution_9), kwargs = {})
#   %div_4 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_25, 1.0), kwargs = {})
#   %sigmoid_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_29,), kwargs = {})
#   %mul_31 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, %sigmoid_9), kwargs = {})
#   %convolution_11 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_31, %arg53_1, %arg54_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_30 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, %convolution_11), kwargs = {})
#   %div_5 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_30, 1.0), kwargs = {})
#   %iota : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_32 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_31 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, 0), kwargs = {})
#   %convert_element_type_38 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_31, torch.float32), kwargs = {})
#   %add_32 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_38, 0.0), kwargs = {})
#   %mul_33 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_32, 0.5), kwargs = {})
#   %convert_element_type_39 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_33, torch.int64), kwargs = {})
#   %unsqueeze_64 : Tensor "i64[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_39, -1), kwargs = {})
#   %iota_1 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_34 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_1, 1), kwargs = {})
#   %add_33 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, 0), kwargs = {})
#   %convert_element_type_40 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_33, torch.float32), kwargs = {})
#   %add_34 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_40, 0.0), kwargs = {})
#   %mul_35 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_34, 0.5), kwargs = {})
#   %convert_element_type_41 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_35, torch.int64), kwargs = {})
#   %_unsafe_index : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_5, [None, None, %unsqueeze_64, %convert_element_type_41]), kwargs = {})
#   %clone_13 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.channels_last})
#   %convolution_12 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_13, %arg55_1, %arg56_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_14 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_12,), kwargs = {memory_format: torch.contiguous_format})
#   %view_36 : Tensor "f32[4, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_14, [4, 32, 16, 65536]), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_36, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf197,%buf198,%buf199
triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27 = async_compile.triton('triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1572864, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 64)
    x2 = xindex // 16384
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (2*x0 + 512*(((r0_3 + 2048*x1) % 65536)) + 33554432*x2 + ((r0_3 + 2048*x1) // 65536)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (2*x0 + ((r0_3 + 2048*x1) // 65536)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask, tmp4_weight_next, tmp4_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tl.store(out_ptr0 + (x4), tmp4, None)
    tl.store(out_ptr1 + (x4), tmp8, None)
    tl.store(out_ptr2 + (x4), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/ve/cvezz7qmuxinjkdjdjaqnbpaf3hmx5dgphijumceqtxzbgbiozls.py
# Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_47], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_4 => add_25
#   add_5 => add_30
#   hidden_states_35 => mul_25, sigmoid_7
#   hidden_states_37 => convolution_9
#   hidden_states_42 => mul_31, sigmoid_9
#   hidden_states_44 => convolution_11
#   hidden_states_45 => _unsafe_index, add_31, add_32, add_33, add_34, clone_13, convert_element_type_38, convert_element_type_39, convert_element_type_40, convert_element_type_41, iota, iota_1, mul_32, mul_33, mul_34, mul_35, unsqueeze_64
#   hidden_states_46 => convolution_12
#   hidden_states_47 => clone_14, var_mean_11, view_36
#   output_tensor_3 => div_4
#   output_tensor_4 => div_5
# Graph fragment:
#   %buf197 : Tensor "f32[4, 32, 1, 1, 8, 64][16384, 8, 65536, 65536, 1, 256]cuda:0" = PlaceHolder[target=buf197]
#   %buf198 : Tensor "f32[4, 32, 1, 1, 8, 64][16384, 8, 65536, 65536, 1, 256]cuda:0" = PlaceHolder[target=buf198]
#   %buf199 : Tensor "f32[4, 32, 1, 1, 8, 64][16384, 8, 65536, 65536, 1, 256]cuda:0" = PlaceHolder[target=buf199]
#   %sigmoid_7 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_24,), kwargs = {})
#   %mul_25 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %sigmoid_7), kwargs = {})
#   %convolution_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_25, %arg45_1, %arg46_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_25 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %convolution_9), kwargs = {})
#   %div_4 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_25, 1.0), kwargs = {})
#   %sigmoid_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_29,), kwargs = {})
#   %mul_31 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, %sigmoid_9), kwargs = {})
#   %convolution_11 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_31, %arg53_1, %arg54_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_30 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, %convolution_11), kwargs = {})
#   %div_5 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_30, 1.0), kwargs = {})
#   %iota : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_32 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_31 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, 0), kwargs = {})
#   %convert_element_type_38 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_31, torch.float32), kwargs = {})
#   %add_32 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_38, 0.0), kwargs = {})
#   %mul_33 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_32, 0.5), kwargs = {})
#   %convert_element_type_39 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_33, torch.int64), kwargs = {})
#   %unsqueeze_64 : Tensor "i64[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_39, -1), kwargs = {})
#   %iota_1 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_34 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_1, 1), kwargs = {})
#   %add_33 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, 0), kwargs = {})
#   %convert_element_type_40 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_33, torch.float32), kwargs = {})
#   %add_34 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_40, 0.0), kwargs = {})
#   %mul_35 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_34, 0.5), kwargs = {})
#   %convert_element_type_41 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_35, torch.int64), kwargs = {})
#   %_unsafe_index : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_5, [None, None, %unsqueeze_64, %convert_element_type_41]), kwargs = {})
#   %clone_13 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.channels_last})
#   %convolution_12 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_13, %arg55_1, %arg56_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_14 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_12,), kwargs = {memory_format: torch.contiguous_format})
#   %view_36 : Tensor "f32[4, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_14, [4, 32, 16, 65536]), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_36, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf200,%buf201,%buf202
triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28 = async_compile.triton('triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 811008, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 256)
    x1 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 16384*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 256*r0_2 + 16384*x1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 256*r0_2 + 16384*x1), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/zw/czwxakyyj5ixqisk37734fzsd6gz2y5lhzd6y4vglpvzumbqisow.py
# Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_47], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_4 => add_25
#   add_5 => add_30
#   hidden_states_35 => mul_25, sigmoid_7
#   hidden_states_37 => convolution_9
#   hidden_states_42 => mul_31, sigmoid_9
#   hidden_states_44 => convolution_11
#   hidden_states_45 => _unsafe_index, add_31, add_32, add_33, add_34, clone_13, convert_element_type_38, convert_element_type_39, convert_element_type_40, convert_element_type_41, iota, iota_1, mul_32, mul_33, mul_34, mul_35, unsqueeze_64
#   hidden_states_46 => convolution_12
#   hidden_states_47 => clone_14, var_mean_11, view_36
#   output_tensor_3 => div_4
#   output_tensor_4 => div_5
# Graph fragment:
#   %buf200 : Tensor "f32[4, 32, 1, 1, 8][256, 8, 1024, 1024, 1]cuda:0" = PlaceHolder[target=buf200]
#   %buf201 : Tensor "f32[4, 32, 1, 1, 8][256, 8, 1024, 1024, 1]cuda:0" = PlaceHolder[target=buf201]
#   %buf202 : Tensor "f32[4, 32, 1, 1, 8][256, 8, 1024, 1024, 1]cuda:0" = PlaceHolder[target=buf202]
#   %sigmoid_7 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_24,), kwargs = {})
#   %mul_25 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %sigmoid_7), kwargs = {})
#   %convolution_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_25, %arg45_1, %arg46_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_25 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %convolution_9), kwargs = {})
#   %div_4 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_25, 1.0), kwargs = {})
#   %sigmoid_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_29,), kwargs = {})
#   %mul_31 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, %sigmoid_9), kwargs = {})
#   %convolution_11 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_31, %arg53_1, %arg54_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_30 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, %convolution_11), kwargs = {})
#   %div_5 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_30, 1.0), kwargs = {})
#   %iota : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_32 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_31 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, 0), kwargs = {})
#   %convert_element_type_38 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_31, torch.float32), kwargs = {})
#   %add_32 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_38, 0.0), kwargs = {})
#   %mul_33 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_32, 0.5), kwargs = {})
#   %convert_element_type_39 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_33, torch.int64), kwargs = {})
#   %unsqueeze_64 : Tensor "i64[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_39, -1), kwargs = {})
#   %iota_1 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_34 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_1, 1), kwargs = {})
#   %add_33 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, 0), kwargs = {})
#   %convert_element_type_40 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_33, torch.float32), kwargs = {})
#   %add_34 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_40, 0.0), kwargs = {})
#   %mul_35 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_34, 0.5), kwargs = {})
#   %convert_element_type_41 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_35, torch.int64), kwargs = {})
#   %_unsafe_index : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_5, [None, None, %unsqueeze_64, %convert_element_type_41]), kwargs = {})
#   %clone_13 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.channels_last})
#   %convolution_12 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_13, %arg55_1, %arg56_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_14 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_12,), kwargs = {memory_format: torch.contiguous_format})
#   %view_36 : Tensor "f32[4, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_14, [4, 32, 16, 65536]), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_36, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_27,%buf204
triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29 = async_compile.triton('triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2048, 'r0_': 12288}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 8*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 8*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 8*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/zi/cziwo5ee5z4tgeqfezqrai2r4h6a3jk46ndkezqy5tndp2ixhqhn.py
# Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_47, hidden_states_48], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_4 => add_25
#   add_5 => add_30
#   hidden_states_35 => mul_25, sigmoid_7
#   hidden_states_37 => convolution_9
#   hidden_states_42 => mul_31, sigmoid_9
#   hidden_states_44 => convolution_11
#   hidden_states_45 => _unsafe_index, add_31, add_32, add_33, add_34, clone_13, convert_element_type_38, convert_element_type_39, convert_element_type_40, convert_element_type_41, iota, iota_1, mul_32, mul_33, mul_34, mul_35, unsqueeze_64
#   hidden_states_46 => convolution_12
#   hidden_states_47 => add_35, add_36, clone_14, mul_36, mul_37, rsqrt_11, sub_11, unsqueeze_65, unsqueeze_66, unsqueeze_67, unsqueeze_68, unsqueeze_69, unsqueeze_70, var_mean_11, view_36, view_37
#   hidden_states_48 => mul_38, sigmoid_10
#   output_tensor_3 => div_4
#   output_tensor_4 => div_5
# Graph fragment:
#   %buf196 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf196]
#   %arg56_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg56_1]
#   %getitem_27 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_27]
#   %buf204 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf204]
#   %arg57_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg57_1]
#   %arg58_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg58_1]
#   %add_36 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=add_36]
#   %sigmoid_7 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_24,), kwargs = {})
#   %mul_25 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %sigmoid_7), kwargs = {})
#   %convolution_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_25, %arg45_1, %arg46_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_25 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %convolution_9), kwargs = {})
#   %div_4 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_25, 1.0), kwargs = {})
#   %sigmoid_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_29,), kwargs = {})
#   %mul_31 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, %sigmoid_9), kwargs = {})
#   %convolution_11 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_31, %arg53_1, %arg54_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_30 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, %convolution_11), kwargs = {})
#   %div_5 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_30, 1.0), kwargs = {})
#   %iota : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_32 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_31 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, 0), kwargs = {})
#   %convert_element_type_38 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_31, torch.float32), kwargs = {})
#   %add_32 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_38, 0.0), kwargs = {})
#   %mul_33 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_32, 0.5), kwargs = {})
#   %convert_element_type_39 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_33, torch.int64), kwargs = {})
#   %unsqueeze_64 : Tensor "i64[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_39, -1), kwargs = {})
#   %iota_1 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_34 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_1, 1), kwargs = {})
#   %add_33 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, 0), kwargs = {})
#   %convert_element_type_40 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_33, torch.float32), kwargs = {})
#   %add_34 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_40, 0.0), kwargs = {})
#   %mul_35 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_34, 0.5), kwargs = {})
#   %convert_element_type_41 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_35, torch.int64), kwargs = {})
#   %_unsafe_index : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_5, [None, None, %unsqueeze_64, %convert_element_type_41]), kwargs = {})
#   %clone_13 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.channels_last})
#   %convolution_12 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_13, %arg55_1, %arg56_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_14 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_12,), kwargs = {memory_format: torch.contiguous_format})
#   %view_36 : Tensor "f32[4, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_14, [4, 32, 16, 65536]), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_36, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_11 : Tensor "f32[4, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_36, %getitem_27), kwargs = {})
#   %add_35 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_11 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_35,), kwargs = {})
#   %mul_36 : Tensor "f32[4, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %rsqrt_11), kwargs = {})
#   %view_37 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_36, [4, 512, 256, 256]), kwargs = {})
#   %unsqueeze_65 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg57_1, 0), kwargs = {})
#   %unsqueeze_66 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_65, 2), kwargs = {})
#   %unsqueeze_67 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_66, 3), kwargs = {})
#   %mul_37 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_37, %unsqueeze_67), kwargs = {})
#   %unsqueeze_68 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg58_1, 0), kwargs = {})
#   %unsqueeze_69 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_68, 2), kwargs = {})
#   %unsqueeze_70 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_69, 3), kwargs = {})
#   %add_36 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_37, %unsqueeze_70), kwargs = {})
#   %sigmoid_10 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_36,), kwargs = {})
#   %mul_38 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_36, %sigmoid_10), kwargs = {})
#   return %add_36,%mul_38
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_30 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8388608, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 1073741824, 'x': 1610618880}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8388608
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y4 = yindex
    y0 = (yindex % 32)
    y2 = yindex // 2097152
    tmp0 = tl.load(in_ptr0 + (x3 + 16*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1048576.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + 16*y4), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/z4/cz46pbp477tag2eq5jva5tk2oyuy4hqlqixxxwfqsmbhtdrylnxy.py
# Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49, hidden_states_50, hidden_states_51], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_48 => mul_38, sigmoid_10
#   hidden_states_49 => convolution_13
#   hidden_states_50 => add_37, add_38, mul_39, mul_40, rsqrt_12, sub_12, unsqueeze_71, unsqueeze_72, unsqueeze_73, unsqueeze_74, unsqueeze_75, unsqueeze_76, var_mean_12, view_38, view_39
#   hidden_states_51 => mul_41, sigmoid_11
# Graph fragment:
#   %buf209 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf209]
#   %arg60_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg60_1]
#   %getitem_29 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_29]
#   %buf217 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf217]
#   %arg61_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg61_1]
#   %arg62_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg62_1]
#   %add_38 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=add_38]
#   %sigmoid_10 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_36,), kwargs = {})
#   %mul_38 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_36, %sigmoid_10), kwargs = {})
#   %convolution_13 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_38, %arg59_1, %arg60_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_38 : Tensor "f32[4, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_13, [4, 32, 16, 65536]), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_38, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_12 : Tensor "f32[4, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_38, %getitem_29), kwargs = {})
#   %add_37 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_28, 1e-06), kwargs = {})
#   %rsqrt_12 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_37,), kwargs = {})
#   %mul_39 : Tensor "f32[4, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_12), kwargs = {})
#   %view_39 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_39, [4, 512, 256, 256]), kwargs = {})
#   %unsqueeze_71 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg61_1, 0), kwargs = {})
#   %unsqueeze_72 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_71, 2), kwargs = {})
#   %unsqueeze_73 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_72, 3), kwargs = {})
#   %mul_40 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_39, %unsqueeze_73), kwargs = {})
#   %unsqueeze_74 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg62_1, 0), kwargs = {})
#   %unsqueeze_75 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_74, 2), kwargs = {})
#   %unsqueeze_76 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_75, 3), kwargs = {})
#   %add_38 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_40, %unsqueeze_76), kwargs = {})
#   %sigmoid_11 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_38,), kwargs = {})
#   %mul_41 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %sigmoid_11), kwargs = {})
#   return %add_38,%mul_41
triton_poi_fused_convolution_native_group_norm_silu_31 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8388608, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 1073741824, 'x': 1610618880}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8388608
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y4 = yindex
    y0 = (yindex % 32)
    y2 = yindex // 2097152
    tmp0 = tl.load(in_out_ptr0 + (x3 + 16*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1048576.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + 16*y4), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/hb/chblu3iznnhmkd7jiolyoyxuvtkirf4suhxbciqgqyy27m3wfskb.py
# Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_54], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_4 => add_25
#   add_5 => add_30
#   add_6 => add_39
#   hidden_states_35 => mul_25, sigmoid_7
#   hidden_states_37 => convolution_9
#   hidden_states_42 => mul_31, sigmoid_9
#   hidden_states_44 => convolution_11
#   hidden_states_45 => _unsafe_index, add_31, add_32, add_33, add_34, clone_13, convert_element_type_38, convert_element_type_39, convert_element_type_40, convert_element_type_41, iota, iota_1, mul_32, mul_33, mul_34, mul_35, unsqueeze_64
#   hidden_states_46 => convolution_12
#   hidden_states_51 => mul_41, sigmoid_11
#   hidden_states_53 => convolution_14
#   hidden_states_54 => clone_16, var_mean_13, view_40
#   output_tensor_3 => div_4
#   output_tensor_4 => div_5
#   output_tensor_5 => div_6
# Graph fragment:
#   %buf196 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf196]
#   %arg56_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg56_1]
#   %buf222 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf222]
#   %arg64_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg64_1]
#   %sigmoid_7 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_24,), kwargs = {})
#   %mul_25 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %sigmoid_7), kwargs = {})
#   %convolution_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_25, %arg45_1, %arg46_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_25 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %convolution_9), kwargs = {})
#   %div_4 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_25, 1.0), kwargs = {})
#   %sigmoid_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_29,), kwargs = {})
#   %mul_31 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, %sigmoid_9), kwargs = {})
#   %convolution_11 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_31, %arg53_1, %arg54_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_30 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, %convolution_11), kwargs = {})
#   %div_5 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_30, 1.0), kwargs = {})
#   %iota : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_32 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_31 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, 0), kwargs = {})
#   %convert_element_type_38 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_31, torch.float32), kwargs = {})
#   %add_32 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_38, 0.0), kwargs = {})
#   %mul_33 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_32, 0.5), kwargs = {})
#   %convert_element_type_39 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_33, torch.int64), kwargs = {})
#   %unsqueeze_64 : Tensor "i64[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_39, -1), kwargs = {})
#   %iota_1 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_34 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_1, 1), kwargs = {})
#   %add_33 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, 0), kwargs = {})
#   %convert_element_type_40 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_33, torch.float32), kwargs = {})
#   %add_34 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_40, 0.0), kwargs = {})
#   %mul_35 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_34, 0.5), kwargs = {})
#   %convert_element_type_41 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_35, torch.int64), kwargs = {})
#   %_unsafe_index : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_5, [None, None, %unsqueeze_64, %convert_element_type_41]), kwargs = {})
#   %clone_13 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.channels_last})
#   %convolution_12 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_13, %arg55_1, %arg56_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_11 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_38,), kwargs = {})
#   %mul_41 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %sigmoid_11), kwargs = {})
#   %convolution_14 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_41, %arg63_1, %arg64_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_39 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_14), kwargs = {})
#   %div_6 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_39, 1.0), kwargs = {})
#   %clone_16 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_6,), kwargs = {memory_format: torch.contiguous_format})
#   %view_40 : Tensor "f32[4, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_16, [4, 32, 16, 65536]), kwargs = {})
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_40, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf223,%buf224,%buf225
triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_32 = async_compile.triton('triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1572864, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 64)
    x2 = xindex // 16384
    tmp10_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (2*x0 + 512*(((r0_3 + 2048*x1) % 65536)) + 33554432*x2 + ((r0_3 + 2048*x1) // 65536)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (2*x0 + ((r0_3 + 2048*x1) // 65536)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (2*x0 + 512*(((r0_3 + 2048*x1) % 65536)) + 33554432*x2 + ((r0_3 + 2048*x1) // 65536)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr3 + (2*x0 + ((r0_3 + 2048*x1) // 65536)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = 1.0
        tmp8 = tmp6 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(r0_mask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(r0_mask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(r0_mask, tmp10_weight_next, tmp10_weight)
    tmp11, tmp12, tmp13 = triton_helpers.welford(tmp10_mean, tmp10_m2, tmp10_weight, 1)
    tmp10 = tmp11[:, None]
    tmp14 = tmp12[:, None]
    tmp15 = tmp13[:, None]
    tl.store(out_ptr0 + (x4), tmp10, None)
    tl.store(out_ptr1 + (x4), tmp14, None)
    tl.store(out_ptr2 + (x4), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/ma/cmapodsh3eeqdoaitn7hkkkceticyqtuxdh2q3men5em2eqxfxqf.py
# Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_54, hidden_states_55], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_4 => add_25
#   add_5 => add_30
#   add_6 => add_39
#   hidden_states_35 => mul_25, sigmoid_7
#   hidden_states_37 => convolution_9
#   hidden_states_42 => mul_31, sigmoid_9
#   hidden_states_44 => convolution_11
#   hidden_states_45 => _unsafe_index, add_31, add_32, add_33, add_34, clone_13, convert_element_type_38, convert_element_type_39, convert_element_type_40, convert_element_type_41, iota, iota_1, mul_32, mul_33, mul_34, mul_35, unsqueeze_64
#   hidden_states_46 => convolution_12
#   hidden_states_51 => mul_41, sigmoid_11
#   hidden_states_53 => convolution_14
#   hidden_states_54 => add_40, add_41, clone_16, mul_42, mul_43, rsqrt_13, sub_13, unsqueeze_77, unsqueeze_78, unsqueeze_79, unsqueeze_80, unsqueeze_81, unsqueeze_82, var_mean_13, view_40, view_41
#   hidden_states_55 => mul_44, sigmoid_12
#   output_tensor_3 => div_4
#   output_tensor_4 => div_5
#   output_tensor_5 => div_6
# Graph fragment:
#   %buf196 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf196]
#   %arg56_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg56_1]
#   %buf222 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf222]
#   %arg64_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg64_1]
#   %getitem_31 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_31]
#   %buf230 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf230]
#   %arg65_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg65_1]
#   %arg66_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg66_1]
#   %add_41 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=add_41]
#   %sigmoid_7 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_24,), kwargs = {})
#   %mul_25 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %sigmoid_7), kwargs = {})
#   %convolution_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_25, %arg45_1, %arg46_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_25 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %convolution_9), kwargs = {})
#   %div_4 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_25, 1.0), kwargs = {})
#   %sigmoid_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_29,), kwargs = {})
#   %mul_31 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, %sigmoid_9), kwargs = {})
#   %convolution_11 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_31, %arg53_1, %arg54_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_30 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, %convolution_11), kwargs = {})
#   %div_5 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_30, 1.0), kwargs = {})
#   %iota : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_32 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_31 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, 0), kwargs = {})
#   %convert_element_type_38 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_31, torch.float32), kwargs = {})
#   %add_32 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_38, 0.0), kwargs = {})
#   %mul_33 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_32, 0.5), kwargs = {})
#   %convert_element_type_39 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_33, torch.int64), kwargs = {})
#   %unsqueeze_64 : Tensor "i64[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_39, -1), kwargs = {})
#   %iota_1 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_34 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_1, 1), kwargs = {})
#   %add_33 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, 0), kwargs = {})
#   %convert_element_type_40 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_33, torch.float32), kwargs = {})
#   %add_34 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_40, 0.0), kwargs = {})
#   %mul_35 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_34, 0.5), kwargs = {})
#   %convert_element_type_41 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_35, torch.int64), kwargs = {})
#   %_unsafe_index : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_5, [None, None, %unsqueeze_64, %convert_element_type_41]), kwargs = {})
#   %clone_13 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.channels_last})
#   %convolution_12 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_13, %arg55_1, %arg56_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_11 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_38,), kwargs = {})
#   %mul_41 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %sigmoid_11), kwargs = {})
#   %convolution_14 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_41, %arg63_1, %arg64_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_39 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_14), kwargs = {})
#   %div_6 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_39, 1.0), kwargs = {})
#   %clone_16 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_6,), kwargs = {memory_format: torch.contiguous_format})
#   %view_40 : Tensor "f32[4, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_16, [4, 32, 16, 65536]), kwargs = {})
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_40, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_13 : Tensor "f32[4, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_40, %getitem_31), kwargs = {})
#   %add_40 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_30, 1e-06), kwargs = {})
#   %rsqrt_13 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_40,), kwargs = {})
#   %mul_42 : Tensor "f32[4, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %rsqrt_13), kwargs = {})
#   %view_41 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_42, [4, 512, 256, 256]), kwargs = {})
#   %unsqueeze_77 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg65_1, 0), kwargs = {})
#   %unsqueeze_78 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_77, 2), kwargs = {})
#   %unsqueeze_79 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_78, 3), kwargs = {})
#   %mul_43 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_41, %unsqueeze_79), kwargs = {})
#   %unsqueeze_80 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg66_1, 0), kwargs = {})
#   %unsqueeze_81 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_80, 2), kwargs = {})
#   %unsqueeze_82 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_81, 3), kwargs = {})
#   %add_41 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %unsqueeze_82), kwargs = {})
#   %sigmoid_12 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_41,), kwargs = {})
#   %mul_44 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_41, %sigmoid_12), kwargs = {})
#   return %add_41,%mul_44
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_33 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8388608, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 1073741824, 'x': 2147491840}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8388608
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y4 = yindex
    y0 = (yindex % 32)
    y2 = yindex // 2097152
    tmp0 = tl.load(in_ptr0 + (x3 + 16*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3 + 16*y4), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1048576.0
    tmp13 = (tmp11 / tmp12)
    tmp14 = 1e-06
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp10 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = tmp21 * tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + 16*y4), tmp23, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/xs/cxsuw7imvl2i35ecvn6lsdexteqlvoa7c5fuqgjlk4rxrlfnqi7r.py
# Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_58, hidden_states_60, add_7, output_tensor_6], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
# Source node to ATen node mapping:
#   add_4 => add_25
#   add_5 => add_30
#   add_6 => add_39
#   add_7 => add_44
#   hidden_states_35 => mul_25, sigmoid_7
#   hidden_states_37 => convolution_9
#   hidden_states_42 => mul_31, sigmoid_9
#   hidden_states_44 => convolution_11
#   hidden_states_45 => _unsafe_index, add_31, add_32, add_33, add_34, clone_13, convert_element_type_38, convert_element_type_39, convert_element_type_40, convert_element_type_41, iota, iota_1, mul_32, mul_33, mul_34, mul_35, unsqueeze_64
#   hidden_states_46 => convolution_12
#   hidden_states_51 => mul_41, sigmoid_11
#   hidden_states_53 => convolution_14
#   hidden_states_58 => mul_47, sigmoid_13
#   hidden_states_60 => convolution_16
#   output_tensor_3 => div_4
#   output_tensor_4 => div_5
#   output_tensor_5 => div_6
#   output_tensor_6 => div_7
# Graph fragment:
#   %buf196 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf196]
#   %arg56_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg56_1]
#   %buf222 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf222]
#   %arg64_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg64_1]
#   %buf248 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf248]
#   %arg72_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg72_1]
#   %sigmoid_7 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_24,), kwargs = {})
#   %mul_25 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %sigmoid_7), kwargs = {})
#   %convolution_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_25, %arg45_1, %arg46_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_25 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %convolution_9), kwargs = {})
#   %div_4 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_25, 1.0), kwargs = {})
#   %sigmoid_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_29,), kwargs = {})
#   %mul_31 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, %sigmoid_9), kwargs = {})
#   %convolution_11 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_31, %arg53_1, %arg54_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_30 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_4, %convolution_11), kwargs = {})
#   %div_5 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_30, 1.0), kwargs = {})
#   %iota : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_32 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_31 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, 0), kwargs = {})
#   %convert_element_type_38 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_31, torch.float32), kwargs = {})
#   %add_32 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_38, 0.0), kwargs = {})
#   %mul_33 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_32, 0.5), kwargs = {})
#   %convert_element_type_39 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_33, torch.int64), kwargs = {})
#   %unsqueeze_64 : Tensor "i64[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_39, -1), kwargs = {})
#   %iota_1 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_34 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_1, 1), kwargs = {})
#   %add_33 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, 0), kwargs = {})
#   %convert_element_type_40 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_33, torch.float32), kwargs = {})
#   %add_34 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_40, 0.0), kwargs = {})
#   %mul_35 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_34, 0.5), kwargs = {})
#   %convert_element_type_41 : Tensor "i64[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_35, torch.int64), kwargs = {})
#   %_unsafe_index : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_5, [None, None, %unsqueeze_64, %convert_element_type_41]), kwargs = {})
#   %clone_13 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index,), kwargs = {memory_format: torch.channels_last})
#   %convolution_12 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_13, %arg55_1, %arg56_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_11 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_38,), kwargs = {})
#   %mul_41 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %sigmoid_11), kwargs = {})
#   %convolution_14 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_41, %arg63_1, %arg64_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_39 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_14), kwargs = {})
#   %div_6 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_39, 1.0), kwargs = {})
#   %sigmoid_13 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_43,), kwargs = {})
#   %mul_47 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %sigmoid_13), kwargs = {})
#   %convolution_16 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_47, %arg71_1, %arg72_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_44 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_6, %convolution_16), kwargs = {})
#   %div_7 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_44, 1.0), kwargs = {})
#   return %div_7
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_34 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2684360704}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), None)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp12 * tmp7
    tl.store(in_out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/q4/cq4jot5lr5qbevko3rmirhdq57a6broizhw3envlfpxivqylde7y.py
# Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_61 => clone_18, var_mean_15, view_44
# Graph fragment:
#   %div_7 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=div_7]
#   %clone_18 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_7,), kwargs = {memory_format: torch.contiguous_format})
#   %view_44 : Tensor "f32[4, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_18, [4, 32, 16, 65536]), kwargs = {})
#   %var_mean_15 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_44, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf250,%buf251,%buf252
triton_red_fused_clone_native_group_norm_35 = async_compile.triton('triton_red_fused_clone_native_group_norm_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clone_native_group_norm_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1572864, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_clone_native_group_norm_35(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 64)
    x2 = xindex // 16384
    tmp2_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (2*x0 + 512*(((r0_3 + 2048*x1) % 65536)) + 33554432*x2 + ((r0_3 + 2048*x1) // 65536)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(r0_mask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(r0_mask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(r0_mask, tmp2_weight_next, tmp2_weight)
    tmp3, tmp4, tmp5 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
    tmp2 = tmp3[:, None]
    tmp6 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tl.store(out_ptr0 + (x4), tmp2, None)
    tl.store(out_ptr1 + (x4), tmp6, None)
    tl.store(out_ptr2 + (x4), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/j7/cj72zfowlboos4tggze37qo5wdnrwv44xdirxaulxomd63fsleko.py
# Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_62], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_61 => add_45, add_46, clone_18, mul_48, mul_49, rsqrt_15, sub_15, unsqueeze_89, unsqueeze_90, unsqueeze_91, unsqueeze_92, unsqueeze_93, unsqueeze_94, var_mean_15, view_44, view_45
#   hidden_states_62 => mul_50, sigmoid_14
# Graph fragment:
#   %div_7 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=div_7]
#   %getitem_35 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_35]
#   %buf257 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf257]
#   %arg73_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg73_1]
#   %arg74_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg74_1]
#   %add_46 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=add_46]
#   %clone_18 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_7,), kwargs = {memory_format: torch.contiguous_format})
#   %view_44 : Tensor "f32[4, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_18, [4, 32, 16, 65536]), kwargs = {})
#   %var_mean_15 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_44, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_15 : Tensor "f32[4, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_44, %getitem_35), kwargs = {})
#   %add_45 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_34, 1e-06), kwargs = {})
#   %rsqrt_15 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_45,), kwargs = {})
#   %mul_48 : Tensor "f32[4, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %rsqrt_15), kwargs = {})
#   %view_45 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_48, [4, 512, 256, 256]), kwargs = {})
#   %unsqueeze_89 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg73_1, 0), kwargs = {})
#   %unsqueeze_90 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_89, 2), kwargs = {})
#   %unsqueeze_91 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_90, 3), kwargs = {})
#   %mul_49 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_45, %unsqueeze_91), kwargs = {})
#   %unsqueeze_92 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg74_1, 0), kwargs = {})
#   %unsqueeze_93 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_92, 2), kwargs = {})
#   %unsqueeze_94 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_93, 3), kwargs = {})
#   %add_46 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, %unsqueeze_94), kwargs = {})
#   %sigmoid_14 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_46,), kwargs = {})
#   %mul_50 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_46, %sigmoid_14), kwargs = {})
#   return %add_46,%mul_50
triton_poi_fused_clone_native_group_norm_silu_36 = async_compile.triton('triton_poi_fused_clone_native_group_norm_silu_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8388608, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_native_group_norm_silu_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 1073741824, 'x': 1610616832}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_native_group_norm_silu_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8388608
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y4 = yindex
    y0 = (yindex % 32)
    y2 = yindex // 2097152
    tmp0 = tl.load(in_ptr0 + (x3 + 16*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1048576.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + 16*y4), tmp15, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/jk/cjk23ga5vfmgjrhfr7pg3uw7px7r4olao2qid4ny3yitlcp3s3go.py
# Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
# Source node to ATen node mapping:
#   add_8 => add_49
#   hidden_states_65 => mul_53, sigmoid_15
#   hidden_states_67 => convolution_18
#   hidden_states_68 => _unsafe_index_1, add_50, add_51, add_52, add_53, clone_20, convert_element_type_42, convert_element_type_43, convert_element_type_44, convert_element_type_45, iota_2, iota_3, mul_54, mul_55, mul_56, mul_57, unsqueeze_101
#   output_tensor_7 => div_8
# Graph fragment:
#   %div_7 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=div_7]
#   %buf275 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf275]
#   %arg80_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg80_1]
#   %sigmoid_15 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_53 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_15), kwargs = {})
#   %convolution_18 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_53, %arg79_1, %arg80_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_49 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_7, %convolution_18), kwargs = {})
#   %div_8 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_49, 1.0), kwargs = {})
#   %iota_2 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_54 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_50 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_54, 0), kwargs = {})
#   %convert_element_type_42 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_50, torch.float32), kwargs = {})
#   %add_51 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_42, 0.0), kwargs = {})
#   %mul_55 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_51, 0.5), kwargs = {})
#   %convert_element_type_43 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_55, torch.int64), kwargs = {})
#   %unsqueeze_101 : Tensor "i64[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_43, -1), kwargs = {})
#   %iota_3 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_56 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_3, 1), kwargs = {})
#   %add_52 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, 0), kwargs = {})
#   %convert_element_type_44 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_52, torch.float32), kwargs = {})
#   %add_53 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_44, 0.0), kwargs = {})
#   %mul_57 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, 0.5), kwargs = {})
#   %convert_element_type_45 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_57, torch.int64), kwargs = {})
#   %_unsafe_index_1 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_8, [None, None, %unsqueeze_101, %convert_element_type_45]), kwargs = {})
#   %clone_20 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index_1,), kwargs = {memory_format: torch.channels_last})
#   return %clone_20
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_37 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_37(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 262144) % 512)
    x1 = ((xindex // 512) % 512)
    x0 = (xindex % 512)
    x3 = xindex // 134217728
    x5 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (x0 + 512*tmp8 + 131072*tmp4 + 33554432*x3), None)
    tmp10 = tl.load(in_ptr1 + (x0 + 512*tmp8 + 131072*tmp4 + 33554432*x3), None)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + (x5), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/fd/cfd4fe5k5selhu5wgcy6jgmune4jqoww7qpeo7e5imse7dc4qj5p.py
# Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, hidden_states_70], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_8 => add_49
#   hidden_states_65 => mul_53, sigmoid_15
#   hidden_states_67 => convolution_18
#   hidden_states_68 => _unsafe_index_1, add_50, add_51, add_52, add_53, clone_20, convert_element_type_42, convert_element_type_43, convert_element_type_44, convert_element_type_45, iota_2, iota_3, mul_54, mul_55, mul_56, mul_57, unsqueeze_101
#   hidden_states_69 => convolution_19
#   hidden_states_70 => clone_21, var_mean_17, view_48
#   output_tensor_7 => div_8
# Graph fragment:
#   %buf278 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0" = PlaceHolder[target=buf278]
#   %arg82_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg82_1]
#   %sigmoid_15 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_53 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_15), kwargs = {})
#   %convolution_18 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_53, %arg79_1, %arg80_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_49 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_7, %convolution_18), kwargs = {})
#   %div_8 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_49, 1.0), kwargs = {})
#   %iota_2 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_54 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_50 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_54, 0), kwargs = {})
#   %convert_element_type_42 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_50, torch.float32), kwargs = {})
#   %add_51 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_42, 0.0), kwargs = {})
#   %mul_55 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_51, 0.5), kwargs = {})
#   %convert_element_type_43 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_55, torch.int64), kwargs = {})
#   %unsqueeze_101 : Tensor "i64[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_43, -1), kwargs = {})
#   %iota_3 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_56 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_3, 1), kwargs = {})
#   %add_52 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, 0), kwargs = {})
#   %convert_element_type_44 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_52, torch.float32), kwargs = {})
#   %add_53 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_44, 0.0), kwargs = {})
#   %mul_57 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, 0.5), kwargs = {})
#   %convert_element_type_45 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_57, torch.int64), kwargs = {})
#   %_unsafe_index_1 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_8, [None, None, %unsqueeze_101, %convert_element_type_45]), kwargs = {})
#   %clone_20 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index_1,), kwargs = {memory_format: torch.channels_last})
#   %convolution_19 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_20, %arg81_1, %arg82_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_21 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_19,), kwargs = {memory_format: torch.contiguous_format})
#   %view_48 : Tensor "f32[4, 32, 16, 262144][134217728, 4194304, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_21, [4, 32, 16, 262144]), kwargs = {})
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_48, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf279,%buf280,%buf281
triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38 = async_compile.triton('triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 6291456, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 64)
    x2 = ((xindex // 2048) % 32)
    x3 = xindex // 65536
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x5 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_4 = r0_index
        tmp0 = tl.load(in_ptr0 + (16*x0 + 512*(((r0_4 + 2048*x1 + 131072*x2) % 262144)) + 134217728*x3 + ((r0_4 + 2048*x1 + 131072*x2) // 262144)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (16*x0 + ((r0_4 + 2048*x1 + 131072*x2) // 262144)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask, tmp4_weight_next, tmp4_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tl.store(out_ptr0 + (x5), tmp4, None)
    tl.store(out_ptr1 + (x5), tmp8, None)
    tl.store(out_ptr2 + (x5), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/gg/cggqjmcllz3ref37pvpzeggormv3tecvc3f2jsaabqbqp7q5cwyt.py
# Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, hidden_states_70], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_8 => add_49
#   hidden_states_65 => mul_53, sigmoid_15
#   hidden_states_67 => convolution_18
#   hidden_states_68 => _unsafe_index_1, add_50, add_51, add_52, add_53, clone_20, convert_element_type_42, convert_element_type_43, convert_element_type_44, convert_element_type_45, iota_2, iota_3, mul_54, mul_55, mul_56, mul_57, unsqueeze_101
#   hidden_states_69 => convolution_19
#   hidden_states_70 => clone_21, var_mean_17, view_48
#   output_tensor_7 => div_8
# Graph fragment:
#   %buf279 : Tensor "f32[4, 32, 1, 1, 32, 64][65536, 1, 262144, 262144, 2048, 32]cuda:0" = PlaceHolder[target=buf279]
#   %buf280 : Tensor "f32[4, 32, 1, 1, 32, 64][65536, 1, 262144, 262144, 2048, 32]cuda:0" = PlaceHolder[target=buf280]
#   %buf281 : Tensor "f32[4, 32, 1, 1, 32, 64][65536, 1, 262144, 262144, 2048, 32]cuda:0" = PlaceHolder[target=buf281]
#   %sigmoid_15 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_53 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_15), kwargs = {})
#   %convolution_18 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_53, %arg79_1, %arg80_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_49 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_7, %convolution_18), kwargs = {})
#   %div_8 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_49, 1.0), kwargs = {})
#   %iota_2 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_54 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_50 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_54, 0), kwargs = {})
#   %convert_element_type_42 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_50, torch.float32), kwargs = {})
#   %add_51 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_42, 0.0), kwargs = {})
#   %mul_55 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_51, 0.5), kwargs = {})
#   %convert_element_type_43 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_55, torch.int64), kwargs = {})
#   %unsqueeze_101 : Tensor "i64[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_43, -1), kwargs = {})
#   %iota_3 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_56 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_3, 1), kwargs = {})
#   %add_52 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, 0), kwargs = {})
#   %convert_element_type_44 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_52, torch.float32), kwargs = {})
#   %add_53 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_44, 0.0), kwargs = {})
#   %mul_57 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, 0.5), kwargs = {})
#   %convert_element_type_45 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_57, torch.int64), kwargs = {})
#   %_unsafe_index_1 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_8, [None, None, %unsqueeze_101, %convert_element_type_45]), kwargs = {})
#   %clone_20 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index_1,), kwargs = {memory_format: torch.channels_last})
#   %convolution_19 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_20, %arg81_1, %arg82_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_21 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_19,), kwargs = {memory_format: torch.contiguous_format})
#   %view_48 : Tensor "f32[4, 32, 16, 262144][134217728, 4194304, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_21, [4, 32, 16, 262144]), kwargs = {})
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_48, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf282,%buf283,%buf284
triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39 = async_compile.triton('triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3244032, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 32)
    x1 = xindex // 32
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*r0_2 + 2048*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 32*r0_2 + 2048*x1), None)
    tmp2 = tl.load(in_ptr2 + (x0 + 32*r0_2 + 2048*x1), None)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp3, tmp4, tmp5, 1)
    tmp10 = tmp7[:, None]
    tmp11 = tmp8[:, None]
    tmp12 = tmp9[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp11, None)
    tl.store(out_ptr2 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/xo/cxocyy4d2o5abb6avji6jydqrud4mbvqqmqnq5yxqtvsxxmygwxx.py
# Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, hidden_states_70], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_8 => add_49
#   hidden_states_65 => mul_53, sigmoid_15
#   hidden_states_67 => convolution_18
#   hidden_states_68 => _unsafe_index_1, add_50, add_51, add_52, add_53, clone_20, convert_element_type_42, convert_element_type_43, convert_element_type_44, convert_element_type_45, iota_2, iota_3, mul_54, mul_55, mul_56, mul_57, unsqueeze_101
#   hidden_states_69 => convolution_19
#   hidden_states_70 => clone_21, var_mean_17, view_48
#   output_tensor_7 => div_8
# Graph fragment:
#   %buf282 : Tensor "f32[4, 32, 1, 1, 32][1024, 1, 4096, 4096, 32]cuda:0" = PlaceHolder[target=buf282]
#   %buf283 : Tensor "f32[4, 32, 1, 1, 32][1024, 1, 4096, 4096, 32]cuda:0" = PlaceHolder[target=buf283]
#   %buf284 : Tensor "f32[4, 32, 1, 1, 32][1024, 1, 4096, 4096, 32]cuda:0" = PlaceHolder[target=buf284]
#   %sigmoid_15 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_53 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_15), kwargs = {})
#   %convolution_18 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_53, %arg79_1, %arg80_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_49 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_7, %convolution_18), kwargs = {})
#   %div_8 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_49, 1.0), kwargs = {})
#   %iota_2 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_54 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_50 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_54, 0), kwargs = {})
#   %convert_element_type_42 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_50, torch.float32), kwargs = {})
#   %add_51 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_42, 0.0), kwargs = {})
#   %mul_55 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_51, 0.5), kwargs = {})
#   %convert_element_type_43 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_55, torch.int64), kwargs = {})
#   %unsqueeze_101 : Tensor "i64[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_43, -1), kwargs = {})
#   %iota_3 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_56 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_3, 1), kwargs = {})
#   %add_52 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, 0), kwargs = {})
#   %convert_element_type_44 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_52, torch.float32), kwargs = {})
#   %add_53 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_44, 0.0), kwargs = {})
#   %mul_57 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, 0.5), kwargs = {})
#   %convert_element_type_45 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_57, torch.int64), kwargs = {})
#   %_unsafe_index_1 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_8, [None, None, %unsqueeze_101, %convert_element_type_45]), kwargs = {})
#   %clone_20 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index_1,), kwargs = {memory_format: torch.channels_last})
#   %convolution_19 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_20, %arg81_1, %arg82_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_21 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_19,), kwargs = {memory_format: torch.contiguous_format})
#   %view_48 : Tensor "f32[4, 32, 16, 262144][134217728, 4194304, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_21, [4, 32, 16, 262144]), kwargs = {})
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_48, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_39,%buf286
triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40 = async_compile.triton('triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 51200, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 32)
    x1 = xindex // 32
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*r0_2 + 1024*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 32*r0_2 + 1024*x1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 32*r0_2 + 1024*x1), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/dx/cdxuyhemz6f6j5qkpihrq3y332hnqsyuhkzsaqfeenmlheqh367e.py
# Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, hidden_states_70, hidden_states_71, contiguous, input_tensor], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_8 => add_49
#   contiguous => clone_23
#   hidden_states_65 => mul_53, sigmoid_15
#   hidden_states_67 => convolution_18
#   hidden_states_68 => _unsafe_index_1, add_50, add_51, add_52, add_53, clone_20, convert_element_type_42, convert_element_type_43, convert_element_type_44, convert_element_type_45, iota_2, iota_3, mul_54, mul_55, mul_56, mul_57, unsqueeze_101
#   hidden_states_69 => convolution_19
#   hidden_states_70 => add_54, add_55, clone_21, mul_58, mul_59, rsqrt_17, sub_17, unsqueeze_102, unsqueeze_103, unsqueeze_104, unsqueeze_105, unsqueeze_106, unsqueeze_107, var_mean_17, view_48, view_49
#   hidden_states_71 => mul_60, sigmoid_16
#   input_tensor => convolution_22
#   output_tensor_7 => div_8
# Graph fragment:
#   %buf278 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0" = PlaceHolder[target=buf278]
#   %arg82_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg82_1]
#   %getitem_39 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_39]
#   %buf286 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf286]
#   %arg83_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg83_1]
#   %arg84_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg84_1]
#   %add_55 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0" = PlaceHolder[target=add_55]
#   %sigmoid_15 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_53 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_15), kwargs = {})
#   %convolution_18 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_53, %arg79_1, %arg80_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_49 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_7, %convolution_18), kwargs = {})
#   %div_8 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_49, 1.0), kwargs = {})
#   %iota_2 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_54 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_50 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_54, 0), kwargs = {})
#   %convert_element_type_42 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_50, torch.float32), kwargs = {})
#   %add_51 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_42, 0.0), kwargs = {})
#   %mul_55 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_51, 0.5), kwargs = {})
#   %convert_element_type_43 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_55, torch.int64), kwargs = {})
#   %unsqueeze_101 : Tensor "i64[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_43, -1), kwargs = {})
#   %iota_3 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_56 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_3, 1), kwargs = {})
#   %add_52 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, 0), kwargs = {})
#   %convert_element_type_44 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_52, torch.float32), kwargs = {})
#   %add_53 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_44, 0.0), kwargs = {})
#   %mul_57 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, 0.5), kwargs = {})
#   %convert_element_type_45 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_57, torch.int64), kwargs = {})
#   %_unsafe_index_1 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_8, [None, None, %unsqueeze_101, %convert_element_type_45]), kwargs = {})
#   %clone_20 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index_1,), kwargs = {memory_format: torch.channels_last})
#   %convolution_19 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_20, %arg81_1, %arg82_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_21 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_19,), kwargs = {memory_format: torch.contiguous_format})
#   %view_48 : Tensor "f32[4, 32, 16, 262144][134217728, 4194304, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_21, [4, 32, 16, 262144]), kwargs = {})
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_48, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_17 : Tensor "f32[4, 32, 16, 262144][134217728, 4194304, 262144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_48, %getitem_39), kwargs = {})
#   %add_54 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_38, 1e-06), kwargs = {})
#   %rsqrt_17 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_54,), kwargs = {})
#   %mul_58 : Tensor "f32[4, 32, 16, 262144][134217728, 4194304, 262144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %rsqrt_17), kwargs = {})
#   %view_49 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_58, [4, 512, 512, 512]), kwargs = {})
#   %unsqueeze_102 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg83_1, 0), kwargs = {})
#   %unsqueeze_103 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_102, 2), kwargs = {})
#   %unsqueeze_104 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_103, 3), kwargs = {})
#   %mul_59 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_49, %unsqueeze_104), kwargs = {})
#   %unsqueeze_105 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg84_1, 0), kwargs = {})
#   %unsqueeze_106 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_105, 2), kwargs = {})
#   %unsqueeze_107 : Tensor "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_106, 3), kwargs = {})
#   %add_55 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_59, %unsqueeze_107), kwargs = {})
#   %sigmoid_16 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_55,), kwargs = {})
#   %mul_60 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, %sigmoid_16), kwargs = {})
#   %clone_23 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_19,), kwargs = {memory_format: torch.contiguous_format})
#   %convolution_22 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_23, %arg91_1, %arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_55,%buf301,%mul_60
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_41 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 33554432, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 4294967296, 'x': 12884908032}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 33554432
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y5 = yindex
    y0 = (yindex % 32)
    y2 = yindex // 8388608
    tmp0 = tl.load(in_ptr0 + (x3 + 16*y5), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + 32*y2), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3 + 16*y0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 4194304.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x3 + 16*y5), tmp2, xmask & ymask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + 16*y5), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/mv/cmvui26hw7yfyh4cjlar6fg6ol3ib7zpwb2waui3c5flol3xd3il.py
# Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_71 => mul_60, sigmoid_16
#   hidden_states_72 => convolution_20
# Graph fragment:
#   %arg85_1 : Tensor "f32[256, 512, 3, 3][4608, 9, 3, 1]cuda:0" = PlaceHolder[target=arg85_1]
#   %sigmoid_16 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_55,), kwargs = {})
#   %mul_60 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, %sigmoid_16), kwargs = {})
#   %convolution_20 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_60, %arg85_1, %arg86_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf290
triton_poi_fused_convolution_silu_42 = async_compile.triton('triton_poi_fused_convolution_silu_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 9437184, 'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_42(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/wc/cwckdt22couepc5q7hksbvqwaprlq2chakcjkwv5gtcg2qnplhan.py
# Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_71 => mul_60, sigmoid_16
#   hidden_states_72 => convolution_20
#   hidden_states_73 => var_mean_18, view_50
# Graph fragment:
#   %buf291 : Tensor "f32[4, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=buf291]
#   %arg86_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg86_1]
#   %sigmoid_16 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_55,), kwargs = {})
#   %mul_60 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, %sigmoid_16), kwargs = {})
#   %convolution_20 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_60, %arg85_1, %arg86_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_50 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_20, [4, 32, 8, 262144]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_50, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf292,%buf293,%buf294
triton_red_fused_convolution_native_group_norm_silu_43 = async_compile.triton('triton_red_fused_convolution_native_group_norm_silu_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_silu_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_silu_43(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 64)
    x2 = ((xindex // 2048) % 16)
    x3 = xindex // 32768
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x5 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_4 = r0_index
        tmp0 = tl.load(in_ptr0 + (8*x0 + 256*(((r0_4 + 2048*x1 + 131072*x2) % 262144)) + 67108864*x3 + ((r0_4 + 2048*x1 + 131072*x2) // 262144)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (8*x0 + ((r0_4 + 2048*x1 + 131072*x2) // 262144)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask, tmp4_weight_next, tmp4_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tl.store(out_ptr0 + (x5), tmp4, None)
    tl.store(out_ptr1 + (x5), tmp8, None)
    tl.store(out_ptr2 + (x5), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/wr/cwrxpihf7qtvxfemhkygltyd3c52lcugldjx33xzael4opafy6kd.py
# Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_71 => mul_60, sigmoid_16
#   hidden_states_72 => convolution_20
#   hidden_states_73 => var_mean_18, view_50
# Graph fragment:
#   %buf292 : Tensor "f32[4, 32, 1, 1, 16, 64][32768, 1, 131072, 131072, 2048, 32]cuda:0" = PlaceHolder[target=buf292]
#   %buf293 : Tensor "f32[4, 32, 1, 1, 16, 64][32768, 1, 131072, 131072, 2048, 32]cuda:0" = PlaceHolder[target=buf293]
#   %buf294 : Tensor "f32[4, 32, 1, 1, 16, 64][32768, 1, 131072, 131072, 2048, 32]cuda:0" = PlaceHolder[target=buf294]
#   %sigmoid_16 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_55,), kwargs = {})
#   %mul_60 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, %sigmoid_16), kwargs = {})
#   %convolution_20 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_60, %arg85_1, %arg86_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_50 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_20, [4, 32, 8, 262144]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_50, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf295,%buf296,%buf297
triton_per_fused_convolution_native_group_norm_silu_44 = async_compile.triton('triton_per_fused_convolution_native_group_norm_silu_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_silu_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1622016, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_silu_44(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 32)
    x1 = xindex // 32
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*r0_2 + 2048*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 32*r0_2 + 2048*x1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 32*r0_2 + 2048*x1), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/rx/crxv3tnowpzlrxyk3hw42rhur2nr6vfgyufkwsac5jfwi2kvi4oc.py
# Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_71 => mul_60, sigmoid_16
#   hidden_states_72 => convolution_20
#   hidden_states_73 => var_mean_18, view_50
# Graph fragment:
#   %buf295 : Tensor "f32[4, 32, 1, 1, 16][512, 1, 2048, 2048, 32]cuda:0" = PlaceHolder[target=buf295]
#   %buf296 : Tensor "f32[4, 32, 1, 1, 16][512, 1, 2048, 2048, 32]cuda:0" = PlaceHolder[target=buf296]
#   %buf297 : Tensor "f32[4, 32, 1, 1, 16][512, 1, 2048, 2048, 32]cuda:0" = PlaceHolder[target=buf297]
#   %sigmoid_16 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_55,), kwargs = {})
#   %mul_60 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, %sigmoid_16), kwargs = {})
#   %convolution_20 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_60, %arg85_1, %arg86_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_50 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_20, [4, 32, 8, 262144]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_50, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_41,%buf299
triton_per_fused_convolution_native_group_norm_silu_45 = async_compile.triton('triton_per_fused_convolution_native_group_norm_silu_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_silu_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 26624, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_silu_45(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 32)
    x1 = xindex // 32
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*r0_2 + 512*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 32*r0_2 + 512*x1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 32*r0_2 + 512*x1), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/te/cterf4yn4jwoh6qyfrkm6cqaychzu5xqf6ia66xtuo7onik3xbgz.py
# Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73, hidden_states_74], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_71 => mul_60, sigmoid_16
#   hidden_states_72 => convolution_20
#   hidden_states_73 => add_56, add_57, mul_61, mul_62, rsqrt_18, sub_18, unsqueeze_108, unsqueeze_109, unsqueeze_110, unsqueeze_111, unsqueeze_112, unsqueeze_113, var_mean_18, view_50, view_51
#   hidden_states_74 => mul_63, sigmoid_17
# Graph fragment:
#   %buf291 : Tensor "f32[4, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=buf291]
#   %arg86_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg86_1]
#   %getitem_41 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_41]
#   %buf299 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf299]
#   %arg87_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg87_1]
#   %arg88_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg88_1]
#   %add_57 : Tensor "f32[4, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=add_57]
#   %sigmoid_16 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_55,), kwargs = {})
#   %mul_60 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, %sigmoid_16), kwargs = {})
#   %convolution_20 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_60, %arg85_1, %arg86_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_50 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_20, [4, 32, 8, 262144]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_50, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_18 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_50, %getitem_41), kwargs = {})
#   %add_56 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_40, 1e-06), kwargs = {})
#   %rsqrt_18 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_56,), kwargs = {})
#   %mul_61 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %rsqrt_18), kwargs = {})
#   %view_51 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_61, [4, 256, 512, 512]), kwargs = {})
#   %unsqueeze_108 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg87_1, 0), kwargs = {})
#   %unsqueeze_109 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_108, 2), kwargs = {})
#   %unsqueeze_110 : Tensor "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_109, 3), kwargs = {})
#   %mul_62 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_51, %unsqueeze_110), kwargs = {})
#   %unsqueeze_111 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg88_1, 0), kwargs = {})
#   %unsqueeze_112 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_111, 2), kwargs = {})
#   %unsqueeze_113 : Tensor "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_112, 3), kwargs = {})
#   %add_57 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_113), kwargs = {})
#   %sigmoid_17 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_63 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_17), kwargs = {})
#   return %add_57,%mul_63
triton_poi_fused_convolution_native_group_norm_silu_46 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3221228544}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_46(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 256)
    x2 = xindex // 67108864
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (32*x2 + (x0 // 8)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (32*x2 + (x0 // 8)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 2097152.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/q3/cq3bzmprei7rlucdvyunshffmndq52n2mosxkbnqokkzknyphvpb.py
# Topologically Sorted Source Nodes: [hidden_states_74, hidden_states_76], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_74 => mul_63, sigmoid_17
#   hidden_states_76 => convolution_21
# Graph fragment:
#   %arg89_1 : Tensor "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg89_1]
#   %sigmoid_17 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_63 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_17), kwargs = {})
#   %convolution_21 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_63, %arg89_1, %arg90_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf305
triton_poi_fused_convolution_silu_47 = async_compile.triton('triton_poi_fused_convolution_silu_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 4718592, 'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_47(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/qd/cqd5skhwfg3me2pz4c4fxocbi4t2ospx6qs3al77qjvm5b4yvvek.py
# Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, contiguous, input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_77], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_8 => add_49
#   add_9 => add_58
#   contiguous => clone_23
#   hidden_states_65 => mul_53, sigmoid_15
#   hidden_states_67 => convolution_18
#   hidden_states_68 => _unsafe_index_1, add_50, add_51, add_52, add_53, clone_20, convert_element_type_42, convert_element_type_43, convert_element_type_44, convert_element_type_45, iota_2, iota_3, mul_54, mul_55, mul_56, mul_57, unsqueeze_101
#   hidden_states_69 => convolution_19
#   hidden_states_74 => mul_63, sigmoid_17
#   hidden_states_76 => convolution_21
#   hidden_states_77 => var_mean_19, view_52
#   input_tensor => convolution_22
#   output_tensor_7 => div_8
#   output_tensor_8 => div_9
# Graph fragment:
#   %buf302 : Tensor "f32[1048576, 256][256, 1]cuda:0" = PlaceHolder[target=buf302]
#   %buf306 : Tensor "f32[4, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=buf306]
#   %arg90_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg90_1]
#   %sigmoid_15 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_53 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_15), kwargs = {})
#   %convolution_18 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_53, %arg79_1, %arg80_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_49 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_7, %convolution_18), kwargs = {})
#   %div_8 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_49, 1.0), kwargs = {})
#   %iota_2 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_54 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_50 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_54, 0), kwargs = {})
#   %convert_element_type_42 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_50, torch.float32), kwargs = {})
#   %add_51 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_42, 0.0), kwargs = {})
#   %mul_55 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_51, 0.5), kwargs = {})
#   %convert_element_type_43 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_55, torch.int64), kwargs = {})
#   %unsqueeze_101 : Tensor "i64[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_43, -1), kwargs = {})
#   %iota_3 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_56 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_3, 1), kwargs = {})
#   %add_52 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, 0), kwargs = {})
#   %convert_element_type_44 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_52, torch.float32), kwargs = {})
#   %add_53 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_44, 0.0), kwargs = {})
#   %mul_57 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, 0.5), kwargs = {})
#   %convert_element_type_45 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_57, torch.int64), kwargs = {})
#   %_unsafe_index_1 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_8, [None, None, %unsqueeze_101, %convert_element_type_45]), kwargs = {})
#   %clone_20 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index_1,), kwargs = {memory_format: torch.channels_last})
#   %convolution_19 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_20, %arg81_1, %arg82_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_23 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_19,), kwargs = {memory_format: torch.contiguous_format})
#   %convolution_22 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_23, %arg91_1, %arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_17 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_63 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_17), kwargs = {})
#   %convolution_21 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_63, %arg89_1, %arg90_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_58 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_22, %convolution_21), kwargs = {})
#   %div_9 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_58, 1.0), kwargs = {})
#   %view_52 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_9, [4, 32, 8, 262144]), kwargs = {})
#   %var_mean_19 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_52, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf307,%buf308,%buf309
triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_48 = async_compile.triton('triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_48(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 64)
    x2 = ((xindex // 2048) % 16)
    x3 = xindex // 32768
    tmp8_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x5 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_4 = r0_index
        tmp0 = tl.load(in_ptr0 + (8*x0 + 256*(((r0_4 + 2048*x1 + 131072*x2) % 262144)) + 67108864*x3 + ((r0_4 + 2048*x1 + 131072*x2) // 262144)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (8*x0 + 256*(((r0_4 + 2048*x1 + 131072*x2) % 262144)) + 67108864*x3 + ((r0_4 + 2048*x1 + 131072*x2) // 262144)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (8*x0 + ((r0_4 + 2048*x1 + 131072*x2) // 262144)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = 1.0
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(r0_mask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(r0_mask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(r0_mask, tmp8_weight_next, tmp8_weight)
    tmp9, tmp10, tmp11 = triton_helpers.welford(tmp8_mean, tmp8_m2, tmp8_weight, 1)
    tmp8 = tmp9[:, None]
    tmp12 = tmp10[:, None]
    tmp13 = tmp11[:, None]
    tl.store(out_ptr0 + (x5), tmp8, None)
    tl.store(out_ptr1 + (x5), tmp12, None)
    tl.store(out_ptr2 + (x5), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/u3/cu3rh6qskv246uuyvxop2tm52ytto65xw4spjcr6aulsx6dogbwn.py
# Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, contiguous, input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_77, hidden_states_78], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_8 => add_49
#   add_9 => add_58
#   contiguous => clone_23
#   hidden_states_65 => mul_53, sigmoid_15
#   hidden_states_67 => convolution_18
#   hidden_states_68 => _unsafe_index_1, add_50, add_51, add_52, add_53, clone_20, convert_element_type_42, convert_element_type_43, convert_element_type_44, convert_element_type_45, iota_2, iota_3, mul_54, mul_55, mul_56, mul_57, unsqueeze_101
#   hidden_states_69 => convolution_19
#   hidden_states_74 => mul_63, sigmoid_17
#   hidden_states_76 => convolution_21
#   hidden_states_77 => add_59, add_60, mul_64, mul_65, rsqrt_19, sub_19, unsqueeze_114, unsqueeze_115, unsqueeze_116, unsqueeze_117, unsqueeze_118, unsqueeze_119, var_mean_19, view_52, view_53
#   hidden_states_78 => mul_66, sigmoid_18
#   input_tensor => convolution_22
#   output_tensor_7 => div_8
#   output_tensor_8 => div_9
# Graph fragment:
#   %buf302 : Tensor "f32[1048576, 256][256, 1]cuda:0" = PlaceHolder[target=buf302]
#   %buf306 : Tensor "f32[4, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=buf306]
#   %arg90_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg90_1]
#   %getitem_43 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_43]
#   %buf314 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf314]
#   %arg93_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg93_1]
#   %arg94_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg94_1]
#   %add_60 : Tensor "f32[4, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=add_60]
#   %sigmoid_15 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_53 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_15), kwargs = {})
#   %convolution_18 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_53, %arg79_1, %arg80_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_49 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_7, %convolution_18), kwargs = {})
#   %div_8 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_49, 1.0), kwargs = {})
#   %iota_2 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_54 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_50 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_54, 0), kwargs = {})
#   %convert_element_type_42 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_50, torch.float32), kwargs = {})
#   %add_51 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_42, 0.0), kwargs = {})
#   %mul_55 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_51, 0.5), kwargs = {})
#   %convert_element_type_43 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_55, torch.int64), kwargs = {})
#   %unsqueeze_101 : Tensor "i64[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_43, -1), kwargs = {})
#   %iota_3 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_56 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_3, 1), kwargs = {})
#   %add_52 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, 0), kwargs = {})
#   %convert_element_type_44 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_52, torch.float32), kwargs = {})
#   %add_53 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_44, 0.0), kwargs = {})
#   %mul_57 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, 0.5), kwargs = {})
#   %convert_element_type_45 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_57, torch.int64), kwargs = {})
#   %_unsafe_index_1 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_8, [None, None, %unsqueeze_101, %convert_element_type_45]), kwargs = {})
#   %clone_20 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index_1,), kwargs = {memory_format: torch.channels_last})
#   %convolution_19 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_20, %arg81_1, %arg82_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_23 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_19,), kwargs = {memory_format: torch.contiguous_format})
#   %convolution_22 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_23, %arg91_1, %arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_17 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_63 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_17), kwargs = {})
#   %convolution_21 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_63, %arg89_1, %arg90_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_58 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_22, %convolution_21), kwargs = {})
#   %div_9 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_58, 1.0), kwargs = {})
#   %view_52 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_9, [4, 32, 8, 262144]), kwargs = {})
#   %var_mean_19 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_52, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_19 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_52, %getitem_43), kwargs = {})
#   %add_59 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_42, 1e-06), kwargs = {})
#   %rsqrt_19 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_59,), kwargs = {})
#   %mul_64 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %rsqrt_19), kwargs = {})
#   %view_53 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_64, [4, 256, 512, 512]), kwargs = {})
#   %unsqueeze_114 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg93_1, 0), kwargs = {})
#   %unsqueeze_115 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_114, 2), kwargs = {})
#   %unsqueeze_116 : Tensor "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_115, 3), kwargs = {})
#   %mul_65 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_53, %unsqueeze_116), kwargs = {})
#   %unsqueeze_117 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg94_1, 0), kwargs = {})
#   %unsqueeze_118 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_117, 2), kwargs = {})
#   %unsqueeze_119 : Tensor "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_118, 3), kwargs = {})
#   %add_60 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_65, %unsqueeze_119), kwargs = {})
#   %sigmoid_18 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_60,), kwargs = {})
#   %mul_66 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_60, %sigmoid_18), kwargs = {})
#   return %add_60,%mul_66
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_49 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4294970368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 256)
    x2 = xindex // 67108864
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (32*x2 + (x0 // 8)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (32*x2 + (x0 // 8)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 2097152.0
    tmp11 = (tmp9 / tmp10)
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sigmoid(tmp19)
    tmp21 = tmp19 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/ol/coluecf2gq5iq3twotgg4hu4x6n6lgn5q5kmop3r5qagis4aehit.py
# Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, contiguous, input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_81, hidden_states_83, add_10, output_tensor_9], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
# Source node to ATen node mapping:
#   add_10 => add_63
#   add_8 => add_49
#   add_9 => add_58
#   contiguous => clone_23
#   hidden_states_65 => mul_53, sigmoid_15
#   hidden_states_67 => convolution_18
#   hidden_states_68 => _unsafe_index_1, add_50, add_51, add_52, add_53, clone_20, convert_element_type_42, convert_element_type_43, convert_element_type_44, convert_element_type_45, iota_2, iota_3, mul_54, mul_55, mul_56, mul_57, unsqueeze_101
#   hidden_states_69 => convolution_19
#   hidden_states_74 => mul_63, sigmoid_17
#   hidden_states_76 => convolution_21
#   hidden_states_81 => mul_69, sigmoid_19
#   hidden_states_83 => convolution_24
#   input_tensor => convolution_22
#   output_tensor_7 => div_8
#   output_tensor_8 => div_9
#   output_tensor_9 => div_10
# Graph fragment:
#   %buf302 : Tensor "f32[1048576, 256][256, 1]cuda:0" = PlaceHolder[target=buf302]
#   %buf306 : Tensor "f32[4, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=buf306]
#   %arg90_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg90_1]
#   %buf332 : Tensor "f32[4, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=buf332]
#   %arg100_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg100_1]
#   %sigmoid_15 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_48,), kwargs = {})
#   %mul_53 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %sigmoid_15), kwargs = {})
#   %convolution_18 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_53, %arg79_1, %arg80_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_49 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_7, %convolution_18), kwargs = {})
#   %div_8 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_49, 1.0), kwargs = {})
#   %iota_2 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_54 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_50 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_54, 0), kwargs = {})
#   %convert_element_type_42 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_50, torch.float32), kwargs = {})
#   %add_51 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_42, 0.0), kwargs = {})
#   %mul_55 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_51, 0.5), kwargs = {})
#   %convert_element_type_43 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_55, torch.int64), kwargs = {})
#   %unsqueeze_101 : Tensor "i64[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_43, -1), kwargs = {})
#   %iota_3 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_56 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_3, 1), kwargs = {})
#   %add_52 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, 0), kwargs = {})
#   %convert_element_type_44 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_52, torch.float32), kwargs = {})
#   %add_53 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_44, 0.0), kwargs = {})
#   %mul_57 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, 0.5), kwargs = {})
#   %convert_element_type_45 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_57, torch.int64), kwargs = {})
#   %_unsafe_index_1 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_8, [None, None, %unsqueeze_101, %convert_element_type_45]), kwargs = {})
#   %clone_20 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%_unsafe_index_1,), kwargs = {memory_format: torch.channels_last})
#   %convolution_19 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_20, %arg81_1, %arg82_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %clone_23 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convolution_19,), kwargs = {memory_format: torch.contiguous_format})
#   %convolution_22 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%clone_23, %arg91_1, %arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_17 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_63 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_17), kwargs = {})
#   %convolution_21 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_63, %arg89_1, %arg90_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_58 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_22, %convolution_21), kwargs = {})
#   %div_9 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_58, 1.0), kwargs = {})
#   %sigmoid_19 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_62,), kwargs = {})
#   %mul_69 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_62, %sigmoid_19), kwargs = {})
#   %convolution_24 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_69, %arg99_1, %arg100_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_63 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_9, %convolution_24), kwargs = {})
#   %div_10 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_63, 1.0), kwargs = {})
#   return %div_10
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_50 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 2147483648, 'x': 3221227520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1048576
    xnumel = 256
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 262144)
    y1 = yindex // 262144
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tmp10 * tmp5
    tl.store(out_ptr0 + (y0 + 262144*x2 + 67108864*y1), tmp11, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/vm/cvms72bcs6m3gwixhcl7nkmw4jxifbo4ssz37qckyiav6wwabg7f.py
# Topologically Sorted Source Nodes: [hidden_states_84], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_84 => var_mean_21, view_56
# Graph fragment:
#   %div_10 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0" = PlaceHolder[target=div_10]
#   %view_56 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_10, [4, 32, 8, 262144]), kwargs = {})
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_56, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf334,%buf335,%buf336
triton_red_fused_native_group_norm_51 = async_compile.triton('triton_red_fused_native_group_norm_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 49152, 'r0_': 1073741824}}
)
@triton.jit
def triton_red_fused_native_group_norm_51(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 131072
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 131072*x0), xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(xmask, tmp2_weight_next, tmp2_weight)
    tmp3, tmp4, tmp5 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
    tmp2 = tmp3[:, None]
    tmp6 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr2 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/th/cthfmejkvxqxdu7l2j7wki6fkf2dkpboyh7mm4geyvpeavj4hsyh.py
# Topologically Sorted Source Nodes: [hidden_states_84], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_84 => var_mean_21, view_56
# Graph fragment:
#   %buf334 : Tensor "f32[4, 32, 1, 1, 16][512, 16, 2048, 2048, 1]cuda:0" = PlaceHolder[target=buf334]
#   %buf335 : Tensor "f32[4, 32, 1, 1, 16][512, 16, 2048, 2048, 1]cuda:0" = PlaceHolder[target=buf335]
#   %buf336 : Tensor "f32[4, 32, 1, 1, 16][512, 16, 2048, 2048, 1]cuda:0" = PlaceHolder[target=buf336]
#   %view_56 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_10, [4, 32, 8, 262144]), kwargs = {})
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_56, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_47,%buf338
triton_per_fused_native_group_norm_52 = async_compile.triton('triton_per_fused_native_group_norm_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2048, 'r0_': 24576}}
)
@triton.jit
def triton_per_fused_native_group_norm_52(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 16*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 16*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/g5/cg5f4tl7kjdyyjuyfcd663hooakvawr2pxgq2kybyskwrbeennov.py
# Topologically Sorted Source Nodes: [hidden_states_84, hidden_states_85], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_84 => add_64, add_65, mul_70, mul_71, rsqrt_21, sub_21, unsqueeze_126, unsqueeze_127, unsqueeze_128, unsqueeze_129, unsqueeze_130, unsqueeze_131, var_mean_21, view_56, view_57
#   hidden_states_85 => mul_72, sigmoid_20
# Graph fragment:
#   %div_10 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0" = PlaceHolder[target=div_10]
#   %getitem_47 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_47]
#   %buf338 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf338]
#   %arg101_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg101_1]
#   %arg102_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg102_1]
#   %add_65 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0" = PlaceHolder[target=add_65]
#   %view_56 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_10, [4, 32, 8, 262144]), kwargs = {})
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_56, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_21 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_56, %getitem_47), kwargs = {})
#   %add_64 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_46, 1e-06), kwargs = {})
#   %rsqrt_21 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_64,), kwargs = {})
#   %mul_70 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %rsqrt_21), kwargs = {})
#   %view_57 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_70, [4, 256, 512, 512]), kwargs = {})
#   %unsqueeze_126 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg101_1, 0), kwargs = {})
#   %unsqueeze_127 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_126, 2), kwargs = {})
#   %unsqueeze_128 : Tensor "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_127, 3), kwargs = {})
#   %mul_71 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_57, %unsqueeze_128), kwargs = {})
#   %unsqueeze_129 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg102_1, 0), kwargs = {})
#   %unsqueeze_130 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_129, 2), kwargs = {})
#   %unsqueeze_131 : Tensor "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_130, 3), kwargs = {})
#   %add_65 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_71, %unsqueeze_131), kwargs = {})
#   %sigmoid_20 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_65,), kwargs = {})
#   %mul_72 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_65, %sigmoid_20), kwargs = {})
#   return %add_65,%mul_72
triton_poi_fused_native_group_norm_silu_53 = async_compile.triton('triton_poi_fused_native_group_norm_silu_53', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 262144}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 2147485696, 'x': 1073741824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 262144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 262144*y3), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3 // 8), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3 // 8), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2097152.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr1 + (y0 + 256*x2 + 67108864*y1), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/hx/chxuwa4fjdzwygooylgmlo5qlyzu2wc4gl6bqcidto2qrojkgss7.py
# Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index]
# Source node to ATen node mapping:
#   add_11 => add_68
#   hidden_states_88 => mul_75, sigmoid_21
#   hidden_states_90 => convolution_26
#   hidden_states_91 => _unsafe_index_2, add_69, add_70, add_71, add_72, convert_element_type_46, convert_element_type_47, convert_element_type_48, convert_element_type_49, iota_4, iota_5, mul_76, mul_77, mul_78, mul_79, unsqueeze_138
#   output_tensor_10 => div_11
# Graph fragment:
#   %div_10 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0" = PlaceHolder[target=div_10]
#   %buf356 : Tensor "f32[4, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=buf356]
#   %arg108_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg108_1]
#   %sigmoid_21 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_67,), kwargs = {})
#   %mul_75 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_67, %sigmoid_21), kwargs = {})
#   %convolution_26 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_75, %arg107_1, %arg108_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_68 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_10, %convolution_26), kwargs = {})
#   %div_11 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_68, 1.0), kwargs = {})
#   %iota_4 : Tensor "i64[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1024,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_76 : Tensor "i64[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_4, 1), kwargs = {})
#   %add_69 : Tensor "i64[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_76, 0), kwargs = {})
#   %convert_element_type_46 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_69, torch.float32), kwargs = {})
#   %add_70 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_46, 0.0), kwargs = {})
#   %mul_77 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_70, 0.5), kwargs = {})
#   %convert_element_type_47 : Tensor "i64[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_77, torch.int64), kwargs = {})
#   %unsqueeze_138 : Tensor "i64[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_47, -1), kwargs = {})
#   %iota_5 : Tensor "i64[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1024,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_78 : Tensor "i64[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_5, 1), kwargs = {})
#   %add_71 : Tensor "i64[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_78, 0), kwargs = {})
#   %convert_element_type_48 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_71, torch.float32), kwargs = {})
#   %add_72 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_48, 0.0), kwargs = {})
#   %mul_79 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_72, 0.5), kwargs = {})
#   %convert_element_type_49 : Tensor "i64[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_79, torch.int64), kwargs = {})
#   %_unsafe_index_2 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_11, [None, None, %unsqueeze_138, %convert_element_type_49]), kwargs = {})
#   return %_unsafe_index_2
triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_54 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_54', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1073741824}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_54(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1073741824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 262144) % 1024)
    x1 = ((xindex // 256) % 1024)
    x0 = (xindex % 256)
    x3 = xindex // 268435456
    x5 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tmp5 = x1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.load(in_ptr0 + (tmp8 + 512*tmp4 + 262144*x0 + 67108864*x3), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (x0 + 256*tmp8 + 131072*tmp4 + 67108864*x3), None)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + (x5), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/qj/cqjb5amlx3mzw4t2ahnb3bhm6iov3mk6tymvkxnj75deoul4vrpr.py
# Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91, hidden_states_92], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index]
# Source node to ATen node mapping:
#   add_11 => add_68
#   hidden_states_88 => mul_75, sigmoid_21
#   hidden_states_90 => convolution_26
#   hidden_states_91 => _unsafe_index_2, add_69, add_70, add_71, add_72, convert_element_type_46, convert_element_type_47, convert_element_type_48, convert_element_type_49, iota_4, iota_5, mul_76, mul_77, mul_78, mul_79, unsqueeze_138
#   hidden_states_92 => convolution_27
#   output_tensor_10 => div_11
# Graph fragment:
#   %buf359 : Tensor "f32[4, 256, 1024, 1024][268435456, 1, 262144, 256]cuda:0" = PlaceHolder[target=buf359]
#   %arg110_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg110_1]
#   %sigmoid_21 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_67,), kwargs = {})
#   %mul_75 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_67, %sigmoid_21), kwargs = {})
#   %convolution_26 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_75, %arg107_1, %arg108_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_68 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_10, %convolution_26), kwargs = {})
#   %div_11 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_68, 1.0), kwargs = {})
#   %iota_4 : Tensor "i64[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1024,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_76 : Tensor "i64[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_4, 1), kwargs = {})
#   %add_69 : Tensor "i64[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_76, 0), kwargs = {})
#   %convert_element_type_46 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_69, torch.float32), kwargs = {})
#   %add_70 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_46, 0.0), kwargs = {})
#   %mul_77 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_70, 0.5), kwargs = {})
#   %convert_element_type_47 : Tensor "i64[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_77, torch.int64), kwargs = {})
#   %unsqueeze_138 : Tensor "i64[1024, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_47, -1), kwargs = {})
#   %iota_5 : Tensor "i64[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1024,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_78 : Tensor "i64[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_5, 1), kwargs = {})
#   %add_71 : Tensor "i64[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_78, 0), kwargs = {})
#   %convert_element_type_48 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_71, torch.float32), kwargs = {})
#   %add_72 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_48, 0.0), kwargs = {})
#   %mul_79 : Tensor "f32[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_72, 0.5), kwargs = {})
#   %convert_element_type_49 : Tensor "i64[1024][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_79, torch.int64), kwargs = {})
#   %_unsafe_index_2 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%div_11, [None, None, %unsqueeze_138, %convert_element_type_49]), kwargs = {})
#   %convolution_27 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_2, %arg109_1, %arg110_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_27
triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_55 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_55', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 1048576}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 4294968320, 'x': 8589934592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_55(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1048576
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 268435456*y1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 1048576*y3), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/px/cpx7hw4qwtghp4o7ssycpsw2rzw5qamuilwvabluvofezzfs2p52.py
# Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_93 => var_mean_23, view_60
# Graph fragment:
#   %convolution_27 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0" = PlaceHolder[target=convolution_27]
#   %view_60 : Tensor "f32[4, 32, 8, 1048576][268435456, 8388608, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_27, [4, 32, 8, 1048576]), kwargs = {})
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_60, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf361,%buf362,%buf363
triton_red_fused_native_group_norm_56 = async_compile.triton('triton_red_fused_native_group_norm_56', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 196608, 'r0_': 4294967296}}
)
@triton.jit
def triton_red_fused_native_group_norm_56(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8192
    r0_numel = 131072
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 131072*x0), None, eviction_policy='evict_first')
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tmp2_mean_next
        tmp2_m2 = tmp2_m2_next
        tmp2_weight = tmp2_weight_next
    tmp3, tmp4, tmp5 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
    tmp2 = tmp3[:, None]
    tmp6 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tl.store(out_ptr0 + (x0), tmp2, None)
    tl.store(out_ptr1 + (x0), tmp6, None)
    tl.store(out_ptr2 + (x0), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/kx/ckxrfe5vyxjjbphrfohdqpgtw5crx6csoywjva2xw2lim6drc36a.py
# Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_93 => var_mean_23, view_60
# Graph fragment:
#   %buf361 : Tensor "f32[4, 32, 1, 1, 64][2048, 64, 8192, 8192, 1]cuda:0" = PlaceHolder[target=buf361]
#   %buf362 : Tensor "f32[4, 32, 1, 1, 64][2048, 64, 8192, 8192, 1]cuda:0" = PlaceHolder[target=buf362]
#   %buf363 : Tensor "f32[4, 32, 1, 1, 64][2048, 64, 8192, 8192, 1]cuda:0" = PlaceHolder[target=buf363]
#   %view_60 : Tensor "f32[4, 32, 8, 1048576][268435456, 8388608, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_27, [4, 32, 8, 1048576]), kwargs = {})
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_60, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_51,%buf365
triton_per_fused_native_group_norm_57 = async_compile.triton('triton_per_fused_native_group_norm_57', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2048, 'r0_': 98304}}
)
@triton.jit
def triton_per_fused_native_group_norm_57(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/it/citbhv5dakrzldgatyu2agu25ytm4h72kl3q7ynaaqkzukdyviyi.py
# Topologically Sorted Source Nodes: [hidden_states_93, hidden_states_94, input_tensor_1], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_93 => add_73, add_74, mul_80, mul_81, rsqrt_23, sub_23, unsqueeze_139, unsqueeze_140, unsqueeze_141, unsqueeze_142, unsqueeze_143, unsqueeze_144, var_mean_23, view_60, view_61
#   hidden_states_94 => mul_82, sigmoid_22
#   input_tensor_1 => convolution_30
# Graph fragment:
#   %convolution_27 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0" = PlaceHolder[target=convolution_27]
#   %getitem_51 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_51]
#   %buf365 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf365]
#   %arg111_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg111_1]
#   %arg112_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg112_1]
#   %add_74 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0" = PlaceHolder[target=add_74]
#   %view_60 : Tensor "f32[4, 32, 8, 1048576][268435456, 8388608, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_27, [4, 32, 8, 1048576]), kwargs = {})
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_60, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_23 : Tensor "f32[4, 32, 8, 1048576][268435456, 8388608, 1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_60, %getitem_51), kwargs = {})
#   %add_73 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_50, 1e-06), kwargs = {})
#   %rsqrt_23 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_73,), kwargs = {})
#   %mul_80 : Tensor "f32[4, 32, 8, 1048576][268435456, 8388608, 1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %rsqrt_23), kwargs = {})
#   %view_61 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_80, [4, 256, 1024, 1024]), kwargs = {})
#   %unsqueeze_139 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg111_1, 0), kwargs = {})
#   %unsqueeze_140 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_139, 2), kwargs = {})
#   %unsqueeze_141 : Tensor "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_140, 3), kwargs = {})
#   %mul_81 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_61, %unsqueeze_141), kwargs = {})
#   %unsqueeze_142 : Tensor "f32[1, 256][256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg112_1, 0), kwargs = {})
#   %unsqueeze_143 : Tensor "f32[1, 256, 1][256, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_142, 2), kwargs = {})
#   %unsqueeze_144 : Tensor "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_143, 3), kwargs = {})
#   %add_74 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_81, %unsqueeze_144), kwargs = {})
#   %sigmoid_22 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_74,), kwargs = {})
#   %mul_82 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_74, %sigmoid_22), kwargs = {})
#   %convolution_30 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_27, %arg119_1, %arg120_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %add_74,%mul_82,%buf380
triton_poi_fused_convolution_native_group_norm_silu_58 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_58', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 1048576}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_58', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 17179871232, 'x': 4294967296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_58(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1048576
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 1048576*y3), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3 // 8), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3 // 8), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 8388608.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr1 + (y0 + 256*x2 + 268435456*y1), tmp15, None)
    tl.store(out_ptr2 + (y0 + 256*x2 + 268435456*y1), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/oi/coiiqw6vjagqoy3d7rr6isi2bbzrqfz642xxeg2etzelqokkvwql.py
# Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_94 => mul_82, sigmoid_22
#   hidden_states_95 => convolution_28
# Graph fragment:
#   %arg113_1 : Tensor "f32[128, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg113_1]
#   %sigmoid_22 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_74,), kwargs = {})
#   %mul_82 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_74, %sigmoid_22), kwargs = {})
#   %convolution_28 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_82, %arg113_1, %arg114_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf369
triton_poi_fused_convolution_silu_59 = async_compile.triton('triton_poi_fused_convolution_silu_59', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 2359296, 'x': 1179648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_59(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/vd/cvdq57plo3vzqhah7wercpqzzxz7xko7nowarq6oig4xuciiqbaa.py
# Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95, hidden_states_96], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_94 => mul_82, sigmoid_22
#   hidden_states_95 => convolution_28
#   hidden_states_96 => var_mean_24, view_62
# Graph fragment:
#   %buf370 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=buf370]
#   %arg114_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg114_1]
#   %sigmoid_22 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_74,), kwargs = {})
#   %mul_82 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_74, %sigmoid_22), kwargs = {})
#   %convolution_28 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_82, %arg113_1, %arg114_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_62 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_28, [4, 32, 4, 1048576]), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_62, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf371,%buf372,%buf373
triton_red_fused_convolution_native_group_norm_silu_60 = async_compile.triton('triton_red_fused_convolution_native_group_norm_silu_60', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_silu_60', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 6291456, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_silu_60(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 64)
    x2 = ((xindex // 2048) % 32)
    x3 = xindex // 65536
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x5 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_4 = r0_index
        tmp0 = tl.load(in_ptr0 + (4*x0 + 128*(((r0_4 + 2048*x1 + 131072*x2) % 1048576)) + 134217728*x3 + ((r0_4 + 2048*x1 + 131072*x2) // 1048576)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (4*x0 + ((r0_4 + 2048*x1 + 131072*x2) // 1048576)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask, tmp4_weight_next, tmp4_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tl.store(out_ptr0 + (x5), tmp4, None)
    tl.store(out_ptr1 + (x5), tmp8, None)
    tl.store(out_ptr2 + (x5), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/wm/cwm7h4qbjp7afyxmptogfgo6eiaqnwigpsa4kfziapkthnl4k46r.py
# Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95, hidden_states_96, hidden_states_97], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_94 => mul_82, sigmoid_22
#   hidden_states_95 => convolution_28
#   hidden_states_96 => add_75, add_76, mul_83, mul_84, rsqrt_24, sub_24, unsqueeze_145, unsqueeze_146, unsqueeze_147, unsqueeze_148, unsqueeze_149, unsqueeze_150, var_mean_24, view_62, view_63
#   hidden_states_97 => mul_85, sigmoid_23
# Graph fragment:
#   %buf370 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=buf370]
#   %arg114_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg114_1]
#   %getitem_53 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_53]
#   %buf378 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf378]
#   %arg115_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg115_1]
#   %arg116_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg116_1]
#   %add_76 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=add_76]
#   %sigmoid_22 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_74,), kwargs = {})
#   %mul_82 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_74, %sigmoid_22), kwargs = {})
#   %convolution_28 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_82, %arg113_1, %arg114_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_62 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_28, [4, 32, 4, 1048576]), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_62, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_24 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_62, %getitem_53), kwargs = {})
#   %add_75 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_52, 1e-06), kwargs = {})
#   %rsqrt_24 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_75,), kwargs = {})
#   %mul_83 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %rsqrt_24), kwargs = {})
#   %view_63 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_83, [4, 128, 1024, 1024]), kwargs = {})
#   %unsqueeze_145 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg115_1, 0), kwargs = {})
#   %unsqueeze_146 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_145, 2), kwargs = {})
#   %unsqueeze_147 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_146, 3), kwargs = {})
#   %mul_84 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_63, %unsqueeze_147), kwargs = {})
#   %unsqueeze_148 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg116_1, 0), kwargs = {})
#   %unsqueeze_149 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_148, 2), kwargs = {})
#   %unsqueeze_150 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_149, 3), kwargs = {})
#   %add_76 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_84, %unsqueeze_150), kwargs = {})
#   %sigmoid_23 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_76,), kwargs = {})
#   %mul_85 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_76, %sigmoid_23), kwargs = {})
#   return %add_76,%mul_85
triton_poi_fused_convolution_native_group_norm_silu_61 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_61', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_61', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 6442452480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_61(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 128)
    x2 = xindex // 134217728
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 4194304.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/i4/ci4jp6dvhdx7ipbkk6ft5yrfpqmmskl3ukfogf5unzwtkcyyclsh.py
# Topologically Sorted Source Nodes: [hidden_states_97, hidden_states_99], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_97 => mul_85, sigmoid_23
#   hidden_states_99 => convolution_29
# Graph fragment:
#   %arg117_1 : Tensor "f32[128, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg117_1]
#   %sigmoid_23 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_76,), kwargs = {})
#   %mul_85 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_76, %sigmoid_23), kwargs = {})
#   %convolution_29 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_85, %arg117_1, %arg118_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf384
triton_poi_fused_convolution_silu_62 = async_compile.triton('triton_poi_fused_convolution_silu_62', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_62', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 1179648, 'x': 589824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_62(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/33/c33svbk27dnce7wdgp6oq55qrnqtkow64qusqujpsxib64mqzlsx.py
# Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_100], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_12 => add_77
#   hidden_states_100 => var_mean_25, view_64
#   hidden_states_97 => mul_85, sigmoid_23
#   hidden_states_99 => convolution_29
#   input_tensor_1 => convolution_30
#   output_tensor_11 => div_12
# Graph fragment:
#   %buf381 : Tensor "f32[4194304, 128][128, 1]cuda:0" = PlaceHolder[target=buf381]
#   %buf385 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=buf385]
#   %arg118_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg118_1]
#   %convolution_30 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_27, %arg119_1, %arg120_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_23 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_76,), kwargs = {})
#   %mul_85 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_76, %sigmoid_23), kwargs = {})
#   %convolution_29 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_85, %arg117_1, %arg118_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_77 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_30, %convolution_29), kwargs = {})
#   %div_12 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_77, 1.0), kwargs = {})
#   %view_64 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_12, [4, 32, 4, 1048576]), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_64, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf386,%buf387,%buf388
triton_red_fused_add_convolution_div_native_group_norm_silu_63 = async_compile.triton('triton_red_fused_add_convolution_div_native_group_norm_silu_63', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_div_native_group_norm_silu_63', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 6291456, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_add_convolution_div_native_group_norm_silu_63(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 64)
    x2 = ((xindex // 2048) % 32)
    x3 = xindex // 65536
    tmp8_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x5 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_4 = r0_index
        tmp0 = tl.load(in_ptr0 + (4*x0 + 128*(((r0_4 + 2048*x1 + 131072*x2) % 1048576)) + 134217728*x3 + ((r0_4 + 2048*x1 + 131072*x2) // 1048576)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (4*x0 + 128*(((r0_4 + 2048*x1 + 131072*x2) % 1048576)) + 134217728*x3 + ((r0_4 + 2048*x1 + 131072*x2) // 1048576)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (4*x0 + ((r0_4 + 2048*x1 + 131072*x2) // 1048576)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = 1.0
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(r0_mask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(r0_mask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(r0_mask, tmp8_weight_next, tmp8_weight)
    tmp9, tmp10, tmp11 = triton_helpers.welford(tmp8_mean, tmp8_m2, tmp8_weight, 1)
    tmp8 = tmp9[:, None]
    tmp12 = tmp10[:, None]
    tmp13 = tmp11[:, None]
    tl.store(out_ptr0 + (x5), tmp8, None)
    tl.store(out_ptr1 + (x5), tmp12, None)
    tl.store(out_ptr2 + (x5), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/ks/ckspu4cw7oo464tkt5s45lindf2ukekzail6lkipf66bpd5f5vn7.py
# Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_100, hidden_states_101], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_12 => add_77
#   hidden_states_100 => add_78, add_79, mul_86, mul_87, rsqrt_25, sub_25, unsqueeze_151, unsqueeze_152, unsqueeze_153, unsqueeze_154, unsqueeze_155, unsqueeze_156, var_mean_25, view_64, view_65
#   hidden_states_101 => mul_88, sigmoid_24
#   hidden_states_97 => mul_85, sigmoid_23
#   hidden_states_99 => convolution_29
#   input_tensor_1 => convolution_30
#   output_tensor_11 => div_12
# Graph fragment:
#   %buf381 : Tensor "f32[4194304, 128][128, 1]cuda:0" = PlaceHolder[target=buf381]
#   %buf385 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=buf385]
#   %arg118_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg118_1]
#   %getitem_55 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_55]
#   %buf393 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf393]
#   %arg121_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg121_1]
#   %arg122_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg122_1]
#   %add_79 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=add_79]
#   %convolution_30 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_27, %arg119_1, %arg120_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_23 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_76,), kwargs = {})
#   %mul_85 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_76, %sigmoid_23), kwargs = {})
#   %convolution_29 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_85, %arg117_1, %arg118_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_77 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_30, %convolution_29), kwargs = {})
#   %div_12 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_77, 1.0), kwargs = {})
#   %view_64 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_12, [4, 32, 4, 1048576]), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_64, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_25 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_64, %getitem_55), kwargs = {})
#   %add_78 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_54, 1e-06), kwargs = {})
#   %rsqrt_25 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_78,), kwargs = {})
#   %mul_86 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %rsqrt_25), kwargs = {})
#   %view_65 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_86, [4, 128, 1024, 1024]), kwargs = {})
#   %unsqueeze_151 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg121_1, 0), kwargs = {})
#   %unsqueeze_152 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_151, 2), kwargs = {})
#   %unsqueeze_153 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_152, 3), kwargs = {})
#   %mul_87 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_65, %unsqueeze_153), kwargs = {})
#   %unsqueeze_154 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg122_1, 0), kwargs = {})
#   %unsqueeze_155 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_154, 2), kwargs = {})
#   %unsqueeze_156 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_155, 3), kwargs = {})
#   %add_79 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_87, %unsqueeze_156), kwargs = {})
#   %sigmoid_24 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_79,), kwargs = {})
#   %mul_88 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_79, %sigmoid_24), kwargs = {})
#   return %add_79,%mul_88
triton_poi_fused_add_convolution_div_native_group_norm_silu_64 = async_compile.triton('triton_poi_fused_add_convolution_div_native_group_norm_silu_64', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_native_group_norm_silu_64', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 8589936128}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_native_group_norm_silu_64(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 128)
    x2 = xindex // 134217728
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 4194304.0
    tmp11 = (tmp9 / tmp10)
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sigmoid(tmp19)
    tmp21 = tmp19 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/fj/cfjldred3xp2aqr67clldb7q3xixjnjuwvg4uh7t43wuykwwokkw.py
# Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_104, hidden_states_106, add_13, output_tensor_12], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div]
# Source node to ATen node mapping:
#   add_12 => add_77
#   add_13 => add_82
#   hidden_states_104 => mul_91, sigmoid_25
#   hidden_states_106 => convolution_32
#   hidden_states_97 => mul_85, sigmoid_23
#   hidden_states_99 => convolution_29
#   input_tensor_1 => convolution_30
#   output_tensor_11 => div_12
#   output_tensor_12 => div_13
# Graph fragment:
#   %buf381 : Tensor "f32[4194304, 128][128, 1]cuda:0" = PlaceHolder[target=buf381]
#   %buf385 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=buf385]
#   %arg118_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg118_1]
#   %buf411 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=buf411]
#   %arg128_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg128_1]
#   %convolution_30 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_27, %arg119_1, %arg120_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_23 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_76,), kwargs = {})
#   %mul_85 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_76, %sigmoid_23), kwargs = {})
#   %convolution_29 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_85, %arg117_1, %arg118_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_77 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_30, %convolution_29), kwargs = {})
#   %div_12 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_77, 1.0), kwargs = {})
#   %sigmoid_25 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_81,), kwargs = {})
#   %mul_91 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, %sigmoid_25), kwargs = {})
#   %convolution_32 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_91, %arg127_1, %arg128_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_82 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_12, %convolution_32), kwargs = {})
#   %div_13 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_82, 1.0), kwargs = {})
#   return %div_13
triton_poi_fused_add_convolution_div_silu_65 = async_compile.triton('triton_poi_fused_add_convolution_div_silu_65', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4194304, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_silu_65', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 4294967296, 'x': 6442451968}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_silu_65(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4194304
    xnumel = 128
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1048576)
    y1 = yindex // 1048576
    tmp0 = tl.load(in_ptr0 + (x2 + 128*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + 128*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + 128*y3), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tmp10 * tmp5
    tl.store(out_ptr0 + (y0 + 1048576*x2 + 134217728*y1), tmp11, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/34/c34pas7goqeokvlrghpwidiaud5hmbrev5r5ao75um4rvc5omygr.py
# Topologically Sorted Source Nodes: [hidden_states_107], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_107 => var_mean_27, view_68
# Graph fragment:
#   %div_13 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0" = PlaceHolder[target=div_13]
#   %view_68 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_13, [4, 32, 4, 1048576]), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_68, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf413,%buf414,%buf415
triton_red_fused_native_group_norm_66 = async_compile.triton('triton_red_fused_native_group_norm_66', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_66', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 98304, 'r0_': 2147483648}}
)
@triton.jit
def triton_red_fused_native_group_norm_66(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 131072
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 131072*x0), None, eviction_policy='evict_first')
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tmp2_mean_next
        tmp2_m2 = tmp2_m2_next
        tmp2_weight = tmp2_weight_next
    tmp3, tmp4, tmp5 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
    tmp2 = tmp3[:, None]
    tmp6 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    tl.store(out_ptr0 + (x0), tmp2, None)
    tl.store(out_ptr1 + (x0), tmp6, None)
    tl.store(out_ptr2 + (x0), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/2p/c2pny55phz3z4xja37emzmne6sg73zvkpi5axel4lpftcdody77c.py
# Topologically Sorted Source Nodes: [hidden_states_107], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_107 => var_mean_27, view_68
# Graph fragment:
#   %buf413 : Tensor "f32[4, 32, 1, 1, 32][1024, 32, 4096, 4096, 1]cuda:0" = PlaceHolder[target=buf413]
#   %buf414 : Tensor "f32[4, 32, 1, 1, 32][1024, 32, 4096, 4096, 1]cuda:0" = PlaceHolder[target=buf414]
#   %buf415 : Tensor "f32[4, 32, 1, 1, 32][1024, 32, 4096, 4096, 1]cuda:0" = PlaceHolder[target=buf415]
#   %view_68 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_13, [4, 32, 4, 1048576]), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_68, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_59,%buf417
triton_per_fused_native_group_norm_67 = async_compile.triton('triton_per_fused_native_group_norm_67', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_67', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2048, 'r0_': 49152}}
)
@triton.jit
def triton_per_fused_native_group_norm_67(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/jy/cjyxvwwho7qdnvk5t7pvi22sh7l2z6bpph7gsiwohmjyhnypca6u.py
# Topologically Sorted Source Nodes: [hidden_states_107, hidden_states_108], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_107 => add_83, add_84, mul_92, mul_93, rsqrt_27, sub_27, unsqueeze_163, unsqueeze_164, unsqueeze_165, unsqueeze_166, unsqueeze_167, unsqueeze_168, var_mean_27, view_68, view_69
#   hidden_states_108 => mul_94, sigmoid_26
# Graph fragment:
#   %div_13 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0" = PlaceHolder[target=div_13]
#   %getitem_59 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_59]
#   %buf417 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf417]
#   %arg129_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg129_1]
#   %arg130_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg130_1]
#   %add_84 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0" = PlaceHolder[target=add_84]
#   %view_68 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_13, [4, 32, 4, 1048576]), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_68, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_27 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_68, %getitem_59), kwargs = {})
#   %add_83 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_58, 1e-06), kwargs = {})
#   %rsqrt_27 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_83,), kwargs = {})
#   %mul_92 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %rsqrt_27), kwargs = {})
#   %view_69 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_92, [4, 128, 1024, 1024]), kwargs = {})
#   %unsqueeze_163 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg129_1, 0), kwargs = {})
#   %unsqueeze_164 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_163, 2), kwargs = {})
#   %unsqueeze_165 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_164, 3), kwargs = {})
#   %mul_93 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_69, %unsqueeze_165), kwargs = {})
#   %unsqueeze_166 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg130_1, 0), kwargs = {})
#   %unsqueeze_167 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_166, 2), kwargs = {})
#   %unsqueeze_168 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_167, 3), kwargs = {})
#   %add_84 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_93, %unsqueeze_168), kwargs = {})
#   %sigmoid_26 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_84,), kwargs = {})
#   %mul_94 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_84, %sigmoid_26), kwargs = {})
#   return %add_84,%mul_94
triton_poi_fused_native_group_norm_silu_68 = async_compile.triton('triton_poi_fused_native_group_norm_silu_68', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 1048576}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_68', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 4294968320, 'x': 2147483648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_68(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1048576
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 1048576*y3), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3 // 4), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3 // 4), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 4194304.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr1 + (y0 + 128*x2 + 134217728*y1), tmp15, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/sy/csyib7x2p76tgfkn4fwuhqj6nrxisxo56nis7fpmjr5nzvp6c2b7.py
# Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113, add_14, output_tensor_13, sample_2], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_14 => add_87
#   hidden_states_111 => mul_97, sigmoid_27
#   hidden_states_113 => convolution_34
#   output_tensor_13 => div_14
#   sample_2 => var_mean_29, view_72
# Graph fragment:
#   %div_13 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0" = PlaceHolder[target=div_13]
#   %buf435 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=buf435]
#   %arg136_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg136_1]
#   %sigmoid_27 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_86,), kwargs = {})
#   %mul_97 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_86, %sigmoid_27), kwargs = {})
#   %convolution_34 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_97, %arg135_1, %arg136_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_87 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_13, %convolution_34), kwargs = {})
#   %div_14 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_87, 1.0), kwargs = {})
#   %view_72 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_14, [4, 32, 4, 1048576]), kwargs = {})
#   %var_mean_29 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_72, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf436,%buf437,%buf438
triton_red_fused_add_convolution_div_native_group_norm_silu_69 = async_compile.triton('triton_red_fused_add_convolution_div_native_group_norm_silu_69', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_div_native_group_norm_silu_69', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 6291456, 'r0_': 2147483648}}
)
@triton.jit
def triton_red_fused_add_convolution_div_native_group_norm_silu_69(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x5 = xindex
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 32)
    x2 = ((xindex // 2048) % 32)
    x3 = xindex // 65536
    tmp8_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_4 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_4 + 2048*x5), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (4*x2 + 128*(((r0_4 + 2048*x0 + 131072*x1) % 1048576)) + 134217728*x3 + ((r0_4 + 2048*x0 + 131072*x1) // 1048576)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (4*x2 + ((r0_4 + 2048*x0 + 131072*x1) // 1048576)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = 1.0
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(r0_mask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(r0_mask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(r0_mask, tmp8_weight_next, tmp8_weight)
    tmp9, tmp10, tmp11 = triton_helpers.welford(tmp8_mean, tmp8_m2, tmp8_weight, 1)
    tmp8 = tmp9[:, None]
    tmp12 = tmp10[:, None]
    tmp13 = tmp11[:, None]
    tl.store(out_ptr0 + (x5), tmp8, None)
    tl.store(out_ptr1 + (x5), tmp12, None)
    tl.store(out_ptr2 + (x5), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/ne/cney5r2yeeos24dd3r5gbx3phhcleslraze57of34f6j7on43odq.py
# Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113, add_14, output_tensor_13, sample_2], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_14 => add_87
#   hidden_states_111 => mul_97, sigmoid_27
#   hidden_states_113 => convolution_34
#   output_tensor_13 => div_14
#   sample_2 => var_mean_29, view_72
# Graph fragment:
#   %buf436 : Tensor "f32[4, 32, 1, 1, 32, 64][65536, 2048, 262144, 262144, 64, 1]cuda:0" = PlaceHolder[target=buf436]
#   %buf437 : Tensor "f32[4, 32, 1, 1, 32, 64][65536, 2048, 262144, 262144, 64, 1]cuda:0" = PlaceHolder[target=buf437]
#   %buf438 : Tensor "f32[4, 32, 1, 1, 32, 64][65536, 2048, 262144, 262144, 64, 1]cuda:0" = PlaceHolder[target=buf438]
#   %sigmoid_27 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_86,), kwargs = {})
#   %mul_97 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_86, %sigmoid_27), kwargs = {})
#   %convolution_34 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_97, %arg135_1, %arg136_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_87 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_13, %convolution_34), kwargs = {})
#   %div_14 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_87, 1.0), kwargs = {})
#   %view_72 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_14, [4, 32, 4, 1048576]), kwargs = {})
#   %var_mean_29 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_72, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf439,%buf440,%buf441
triton_per_fused_add_convolution_div_native_group_norm_silu_70 = async_compile.triton('triton_per_fused_add_convolution_div_native_group_norm_silu_70', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_div_native_group_norm_silu_70', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 98304, 'r0_': 3145728}}
)
@triton.jit
def triton_per_fused_add_convolution_div_native_group_norm_silu_70(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 64*x0), None)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 64*x0), None)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 64*x0), None)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp3, tmp4, tmp5, 1)
    tmp10 = tmp7[:, None]
    tmp11 = tmp8[:, None]
    tmp12 = tmp9[:, None]
    tl.store(out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr1 + (x0), tmp11, None)
    tl.store(out_ptr2 + (x0), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/ub/cubm6fqaqt47dl5lpdjbkkmfo6fvzgoiludcqc7j74rjoxupyjy6.py
# Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113, add_14, output_tensor_13, sample_2, sample_3], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_14 => add_87
#   hidden_states_111 => mul_97, sigmoid_27
#   hidden_states_113 => convolution_34
#   output_tensor_13 => div_14
#   sample_2 => add_88, add_89, mul_98, mul_99, rsqrt_29, sub_29, unsqueeze_175, unsqueeze_176, unsqueeze_177, unsqueeze_178, unsqueeze_179, unsqueeze_180, var_mean_29, view_72, view_73
#   sample_3 => mul_100, sigmoid_28
# Graph fragment:
#   %div_13 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0" = PlaceHolder[target=div_13]
#   %buf435 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=buf435]
#   %arg136_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg136_1]
#   %getitem_63 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_63]
#   %buf443 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf443]
#   %arg137_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg137_1]
#   %arg138_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg138_1]
#   %add_89 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0" = PlaceHolder[target=add_89]
#   %sigmoid_27 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_86,), kwargs = {})
#   %mul_97 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_86, %sigmoid_27), kwargs = {})
#   %convolution_34 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_97, %arg135_1, %arg136_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_87 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_13, %convolution_34), kwargs = {})
#   %div_14 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_87, 1.0), kwargs = {})
#   %view_72 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_14, [4, 32, 4, 1048576]), kwargs = {})
#   %var_mean_29 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_72, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %sub_29 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_72, %getitem_63), kwargs = {})
#   %add_88 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_62, 1e-06), kwargs = {})
#   %rsqrt_29 : Tensor "f32[4, 32, 1, 1][32, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_88,), kwargs = {})
#   %mul_98 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %rsqrt_29), kwargs = {})
#   %view_73 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_98, [4, 128, 1024, 1024]), kwargs = {})
#   %unsqueeze_175 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg137_1, 0), kwargs = {})
#   %unsqueeze_176 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_175, 2), kwargs = {})
#   %unsqueeze_177 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_176, 3), kwargs = {})
#   %mul_99 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_73, %unsqueeze_177), kwargs = {})
#   %unsqueeze_178 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg138_1, 0), kwargs = {})
#   %unsqueeze_179 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_178, 2), kwargs = {})
#   %unsqueeze_180 : Tensor "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_179, 3), kwargs = {})
#   %add_89 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_99, %unsqueeze_180), kwargs = {})
#   %sigmoid_28 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_89,), kwargs = {})
#   %mul_100 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, %sigmoid_28), kwargs = {})
#   return %add_89,%mul_100
triton_poi_fused_add_convolution_div_native_group_norm_silu_71 = async_compile.triton('triton_poi_fused_add_convolution_div_native_group_norm_silu_71', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 1048576}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_native_group_norm_silu_71', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 6442452480, 'x': 2147483648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_native_group_norm_silu_71(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1048576
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_out_ptr0 + (x2 + 1048576*y3), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + 128*x2 + 134217728*y1), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (y3 // 4), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (y3 // 4), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 4194304.0
    tmp11 = (tmp9 / tmp10)
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sigmoid(tmp19)
    tmp21 = tmp19 * tmp20
    tl.store(out_ptr0 + (y0 + 128*x2 + 134217728*y1), tmp21, ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/fx/cfxgoxzwgkmnfeqfubpery756xklrrm5ch3aywchiguc2w252e2c.py
# Topologically Sorted Source Nodes: [sample_3, sample_4], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   sample_3 => mul_100, sigmoid_28
#   sample_4 => convolution_35
# Graph fragment:
#   %arg139_1 : Tensor "f32[3, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg139_1]
#   %sigmoid_28 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_89,), kwargs = {})
#   %mul_100 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, %sigmoid_28), kwargs = {})
#   %convolution_35 : Tensor "f32[4, 3, 1024, 1024][3145728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_100, %arg139_1, %arg140_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf447
triton_poi_fused_convolution_silu_72 = async_compile.triton('triton_poi_fused_convolution_silu_72', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_72', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 27648, 'x': 13824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_72(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/bb/cbbue3wd4npqnwiudzro7c5uanfwu53wor2mp4nj2v3q5m7vomho.py
# Topologically Sorted Source Nodes: [sample_3, sample_4], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   sample_3 => mul_100, sigmoid_28
#   sample_4 => convolution_35
# Graph fragment:
#   %mul_100 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=mul_100]
#   %buf447 : Tensor "f32[3, 128, 3, 3][1152, 1, 384, 128]cuda:0" = PlaceHolder[target=buf447]
#   %sigmoid_28 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_89,), kwargs = {})
#   %mul_100 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, %sigmoid_28), kwargs = {})
#   %convolution_35 : Tensor "f32[4, 3, 1024, 1024][3145728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_100, %arg139_1, %arg140_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf448
triton_tem_fused_convolution_silu_73 = async_compile.triton('triton_tem_fused_convolution_silu_73', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=2,
num_warps=8,
triton_meta={'signature': {'arg_X': '*fp32', 'arg_W': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_convolution_silu_73', 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'KERNEL_H': 3, 'KERNEL_W': 3, 'STRIDE_H': 1, 'STRIDE_W': 1, 'PADDING_H': 1, 'PADDING_W': 1, 'GROUPS': 1, 'UNROLL': False, 'ALLOW_TF32': True, 'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 32}},

)
@triton.jit
def triton_tem_fused_convolution_silu_73(arg_X, arg_W, out_ptr0):
    KERNEL_H : tl.constexpr = 3
    KERNEL_W : tl.constexpr = 3
    STRIDE_H : tl.constexpr = 1
    STRIDE_W : tl.constexpr = 1
    PADDING_H : tl.constexpr = 1
    PADDING_W : tl.constexpr = 1
    GROUPS : tl.constexpr = 1
    UNROLL : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = True
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 32
    INDEX_DTYPE : tl.constexpr = tl.int32
    X = arg_X
    W = arg_W

    # Tensor dimensions
    BATCH = 4
    IN_C = 128
    IN_H = 1024
    IN_W = 1024
    OUT_C = 3
    OUT_H = 1024
    OUT_W = 1024

    # Strides:
    stride_xn = 134217728
    stride_xc = 1
    stride_xh = 131072
    stride_xw = 128
    stride_wc_out = 1152
    stride_wc_in = 1
    stride_wh = 384
    stride_ww = 128

    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)


    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C


    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)



    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    xindex = idx_w + 1024*idx_h + 1048576*idx_c + 3145728*idx_n
    tl.store(out_ptr0 + (tl.broadcast_to(idx_c + 3*idx_w + 3072*idx_h + 3145728*idx_n, acc.shape)), acc, mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/o4/co4gqojpgoluxziznqtzkhncxikushbfmmcvmha3ph3likuzngp5.py
# Topologically Sorted Source Nodes: [sample_3, sample_4], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   sample_3 => mul_100, sigmoid_28
#   sample_4 => convolution_35
# Graph fragment:
#   %buf448 : Tensor "f32[4, 3, 1024, 1024][3145728, 1, 3072, 3]cuda:0" = PlaceHolder[target=buf448]
#   %arg140_1 : Tensor "f32[3][1]cuda:0" = PlaceHolder[target=arg140_1]
#   %sigmoid_28 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_89,), kwargs = {})
#   %mul_100 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, %sigmoid_28), kwargs = {})
#   %convolution_35 : Tensor "f32[4, 3, 1024, 1024][3145728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_100, %arg139_1, %arg140_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_35
triton_poi_fused_convolution_silu_74 = async_compile.triton('triton_poi_fused_convolution_silu_74', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 1048576}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_74', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'y': 50331660, 'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_74(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 1048576
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    x2 = xindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 3*x2 + 3145728*y1), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 1048576*y3), tmp2, ymask)
''', device_str='cuda')

def partition_0(args):
    arg2_1, arg1_1, arg0_1, arg3_1, arg4_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg5_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg92_1, arg91_1, arg87_1, arg88_1, arg89_1, arg90_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg120_1, arg119_1, arg115_1, arg116_1, arg117_1, arg118_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1 = args
    args.clear()
    assert_size_stride(arg2_1, (4, 4, 128, 128), (65536, 16384, 128, 1))
    assert_size_stride(arg1_1, (4, ), (1, ))
    assert_size_stride(arg0_1, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(arg3_1, (512, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg4_1, (512, ), (1, ))
    assert_size_stride(arg6_1, (512, ), (1, ))
    assert_size_stride(arg7_1, (512, ), (1, ))
    assert_size_stride(arg8_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg9_1, (512, ), (1, ))
    assert_size_stride(arg10_1, (512, ), (1, ))
    assert_size_stride(arg11_1, (512, ), (1, ))
    assert_size_stride(arg12_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg13_1, (512, ), (1, ))
    assert_size_stride(arg14_1, (512, ), (1, ))
    assert_size_stride(arg15_1, (512, ), (1, ))
    assert_size_stride(arg16_1, (512, 512), (512, 1))
    assert_size_stride(arg17_1, (512, ), (1, ))
    assert_size_stride(arg18_1, (512, 512), (512, 1))
    assert_size_stride(arg19_1, (512, ), (1, ))
    assert_size_stride(arg20_1, (512, 512), (512, 1))
    assert_size_stride(arg21_1, (512, ), (1, ))
    assert_size_stride(arg22_1, (512, 512), (512, 1))
    assert_size_stride(arg23_1, (512, ), (1, ))
    assert_size_stride(arg24_1, (512, ), (1, ))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg27_1, (512, ), (1, ))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg31_1, (512, ), (1, ))
    assert_size_stride(arg5_1, (512, ), (1, ))
    assert_size_stride(arg32_1, (512, ), (1, ))
    assert_size_stride(arg33_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg34_1, (512, ), (1, ))
    assert_size_stride(arg35_1, (512, ), (1, ))
    assert_size_stride(arg36_1, (512, ), (1, ))
    assert_size_stride(arg37_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg38_1, (512, ), (1, ))
    assert_size_stride(arg39_1, (512, ), (1, ))
    assert_size_stride(arg40_1, (512, ), (1, ))
    assert_size_stride(arg41_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg42_1, (512, ), (1, ))
    assert_size_stride(arg43_1, (512, ), (1, ))
    assert_size_stride(arg44_1, (512, ), (1, ))
    assert_size_stride(arg45_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg46_1, (512, ), (1, ))
    assert_size_stride(arg47_1, (512, ), (1, ))
    assert_size_stride(arg48_1, (512, ), (1, ))
    assert_size_stride(arg49_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg54_1, (512, ), (1, ))
    assert_size_stride(arg55_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg56_1, (512, ), (1, ))
    assert_size_stride(arg57_1, (512, ), (1, ))
    assert_size_stride(arg58_1, (512, ), (1, ))
    assert_size_stride(arg59_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg60_1, (512, ), (1, ))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (512, ), (1, ))
    assert_size_stride(arg63_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg64_1, (512, ), (1, ))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (512, ), (1, ))
    assert_size_stride(arg67_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (512, ), (1, ))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg72_1, (512, ), (1, ))
    assert_size_stride(arg73_1, (512, ), (1, ))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg76_1, (512, ), (1, ))
    assert_size_stride(arg77_1, (512, ), (1, ))
    assert_size_stride(arg78_1, (512, ), (1, ))
    assert_size_stride(arg79_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg80_1, (512, ), (1, ))
    assert_size_stride(arg81_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg82_1, (512, ), (1, ))
    assert_size_stride(arg83_1, (512, ), (1, ))
    assert_size_stride(arg84_1, (512, ), (1, ))
    assert_size_stride(arg85_1, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg92_1, (256, ), (1, ))
    assert_size_stride(arg91_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg87_1, (256, ), (1, ))
    assert_size_stride(arg88_1, (256, ), (1, ))
    assert_size_stride(arg89_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg90_1, (256, ), (1, ))
    assert_size_stride(arg93_1, (256, ), (1, ))
    assert_size_stride(arg94_1, (256, ), (1, ))
    assert_size_stride(arg95_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg96_1, (256, ), (1, ))
    assert_size_stride(arg97_1, (256, ), (1, ))
    assert_size_stride(arg98_1, (256, ), (1, ))
    assert_size_stride(arg99_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg100_1, (256, ), (1, ))
    assert_size_stride(arg101_1, (256, ), (1, ))
    assert_size_stride(arg102_1, (256, ), (1, ))
    assert_size_stride(arg103_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg104_1, (256, ), (1, ))
    assert_size_stride(arg105_1, (256, ), (1, ))
    assert_size_stride(arg106_1, (256, ), (1, ))
    assert_size_stride(arg107_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg108_1, (256, ), (1, ))
    assert_size_stride(arg109_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg110_1, (256, ), (1, ))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (256, ), (1, ))
    assert_size_stride(arg113_1, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg114_1, (128, ), (1, ))
    assert_size_stride(arg120_1, (128, ), (1, ))
    assert_size_stride(arg119_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg115_1, (128, ), (1, ))
    assert_size_stride(arg116_1, (128, ), (1, ))
    assert_size_stride(arg117_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg118_1, (128, ), (1, ))
    assert_size_stride(arg121_1, (128, ), (1, ))
    assert_size_stride(arg122_1, (128, ), (1, ))
    assert_size_stride(arg123_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg124_1, (128, ), (1, ))
    assert_size_stride(arg125_1, (128, ), (1, ))
    assert_size_stride(arg126_1, (128, ), (1, ))
    assert_size_stride(arg127_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg128_1, (128, ), (1, ))
    assert_size_stride(arg129_1, (128, ), (1, ))
    assert_size_stride(arg130_1, (128, ), (1, ))
    assert_size_stride(arg131_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg132_1, (128, ), (1, ))
    assert_size_stride(arg133_1, (128, ), (1, ))
    assert_size_stride(arg134_1, (128, ), (1, ))
    assert_size_stride(arg135_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg136_1, (128, ), (1, ))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (128, ), (1, ))
    assert_size_stride(arg139_1, (3, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg140_1, (3, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 128, 128), (65536, 1, 512, 4), torch.float16)
        # Topologically Sorted Source Nodes: [z], Original ATen: [aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_0:1306
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg2_1, buf0, 16, 16384, stream=stream0)
        del arg2_1
        buf1 = empty_strided_cuda((65536, 4), (4, 1), torch.float16)
        # Topologically Sorted Source Nodes: [z], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_1.run(arg1_1, buf0, arg0_1, buf1, 512, 1, 1, stream=stream0)
        del arg0_1
        del arg1_1
        del buf0
        buf2 = empty_strided_cuda((512, 4, 3, 3), (36, 1, 12, 4), torch.float16)
        # Topologically Sorted Source Nodes: [z, sample], Original ATen: [aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_2:1308
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(arg3_1, buf2, 2048, 9, stream=stream0)
        del arg3_1
        buf3 = empty_strided_cuda((4, 512, 128, 128), (8388608, 1, 65536, 512), torch.float16)
        # Topologically Sorted Source Nodes: [z, sample], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_3.run(buf1, buf2, buf3, 1024, 2, 1, stream=stream0)
        del buf1
        del buf2
        buf4 = empty_strided_cuda((4, 32, 1, 1, 4, 256), (32768, 4, 131072, 131072, 1, 128), torch.float32)
        buf5 = empty_strided_cuda((4, 32, 1, 1, 4, 256), (32768, 4, 131072, 131072, 1, 128), torch.float32)
        buf6 = empty_strided_cuda((4, 32, 1, 1, 4, 256), (32768, 4, 131072, 131072, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [z, sample, hidden_states], Original ATen: [aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_4:1310
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_4.run(buf3, arg4_1, buf4, buf5, buf6, 131072, 256, stream=stream0)
        buf7 = empty_strided_cuda((4, 32, 1, 1, 4, 2), (256, 4, 1024, 1024, 1, 128), torch.float32)
        buf8 = empty_strided_cuda((4, 32, 1, 1, 4, 2), (256, 4, 1024, 1024, 1, 128), torch.float32)
        buf9 = empty_strided_cuda((4, 32, 1, 1, 4, 2), (256, 4, 1024, 1024, 1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [z, sample, hidden_states], Original ATen: [aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1311
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_5.run(buf4, buf5, buf6, buf7, buf8, buf9, 1024, 128, stream=stream0)
        buf10 = empty_strided_cuda((4, 32, 1, 1, 4), (128, 4, 512, 512, 1), torch.float32)
        buf11 = empty_strided_cuda((4, 32, 1, 1, 4), (128, 4, 512, 512, 1), torch.float32)
        buf12 = empty_strided_cuda((4, 32, 1, 1, 4), (128, 4, 512, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [z, sample, hidden_states], Original ATen: [aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1312
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_6.run(buf7, buf8, buf9, buf10, buf11, buf12, 512, 2, stream=stream0)
        buf13 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf14 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [z, sample, hidden_states], Original ATen: [aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_7:1313
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_7.run(buf10, buf11, buf12, buf13, buf14, 128, 4, stream=stream0)
        buf17 = empty_strided_cuda((4, 512, 128, 128), (8388608, 1, 65536, 512), torch.float16)
        # Topologically Sorted Source Nodes: [z, sample, hidden_states, hidden_states_1], Original ATen: [aten.convolution, aten.native_group_norm, aten.silu]
        # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_8:1314
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_silu_8.run(buf3, arg4_1, buf13, buf14, arg6_1, arg7_1, buf17, 2097152, 16, stream=stream0)
        del arg6_1
        del arg7_1
        buf18 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float16)
        # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_9:1315
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_9.run(arg8_1, buf18, 262144, 9, stream=stream0)
        del arg8_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf19 = extern_kernels.convolution(buf17, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'unknown_op')
        buf20 = buf6; del buf6  # reuse
        buf21 = buf5; del buf5  # reuse
        buf22 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, hidden_states_3], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_4:1316
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_4.run(buf19, arg9_1, buf20, buf21, buf22, 131072, 256, stream=stream0)
        buf23 = buf9; del buf9  # reuse
        buf24 = buf8; del buf8  # reuse
        buf25 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, hidden_states_3], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1317
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_5.run(buf20, buf21, buf22, buf23, buf24, buf25, 1024, 128, stream=stream0)
        buf26 = buf12; del buf12  # reuse
        buf27 = buf11; del buf11  # reuse
        buf28 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, hidden_states_3], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1318
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_6.run(buf23, buf24, buf25, buf26, buf27, buf28, 512, 2, stream=stream0)
        buf29 = buf14; del buf14  # reuse
        buf30 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, hidden_states_3], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_7:1319
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_7.run(buf26, buf27, buf28, buf29, buf30, 128, 4, stream=stream0)
        buf33 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, hidden_states_3, hidden_states_4], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_8:1320
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_silu_8.run(buf19, arg9_1, buf29, buf30, arg10_1, arg11_1, buf33, 2097152, 16, stream=stream0)
        del arg10_1
        del arg11_1
        del arg9_1
        buf34 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_6], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_9:1321
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_9.run(arg12_1, buf34, 262144, 9, stream=stream0)
        del arg12_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf35 = extern_kernels.convolution(buf33, buf34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'unknown_op')
        buf36 = buf22; del buf22  # reuse
        buf37 = buf21; del buf21  # reuse
        buf38 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.view, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_add_convolution_div_native_group_norm_silu_view_10:1322
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_div_native_group_norm_silu_view_10.run(buf3, arg4_1, buf35, arg13_1, buf36, buf37, buf38, 131072, 256, stream=stream0)
        buf39 = buf25; del buf25  # reuse
        buf40 = buf24; del buf24  # reuse
        buf41 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.view, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1323
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_5.run(buf36, buf37, buf38, buf39, buf40, buf41, 1024, 128, stream=stream0)
        buf42 = buf28; del buf28  # reuse
        buf43 = buf27; del buf27  # reuse
        buf44 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.view, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1324
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_6.run(buf39, buf40, buf41, buf42, buf43, buf44, 512, 2, stream=stream0)
        buf45 = buf30; del buf30  # reuse
        buf46 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.view, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_7:1325
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_7.run(buf42, buf43, buf44, buf45, buf46, 128, 4, stream=stream0)
        buf49 = reinterpret_tensor(buf33, (4, 16384, 512), (8388608, 512, 1), 0); del buf33  # reuse
        buf51 = reinterpret_tensor(buf19, (4, 16384, 512), (8388608, 512, 1), 0); del buf19  # reuse
        buf53 = empty_strided_cuda((4, 16384, 512), (8388608, 512, 1), torch.float16)
        # Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2, hidden_states_8, query, key, value], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.view, aten.native_group_norm, aten.transpose, aten.clone]
        # [Provenance debug handles] triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_transpose_view_11:1326
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_transpose_view_11.run(buf3, arg4_1, buf35, arg13_1, buf45, buf46, arg14_1, arg15_1, buf49, buf51, buf53, 2097152, 16, stream=stream0)
        del arg14_1
        del arg15_1
        buf55 = empty_strided_cuda((4, 1, 16384, 512), (8388608, 8388608, 512, 1), torch.float16)
        # Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2, hidden_states_8, query, view_1, query_1, key, view_2, key_1, value, view_3, value_1, hidden_states_9], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.view, aten.native_group_norm, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm, aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_tem_fused__scaled_dot_product_efficient_attention__unsafe_view_add_clone_convolution_div_mm_native_group_norm_silu_t_transpose_view_12.run(buf49, arg16_1, arg17_1, buf55, 2048, 1, 1, stream=stream0)
        del arg16_1
        del arg17_1
        buf56 = reinterpret_tensor(buf49, (4, 1, 16384, 512), (8388608, 8388608, 512, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2, hidden_states_8, query, view_1, query_1, key, view_2, key_1, value, view_3, value_1, hidden_states_9], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.view, aten.native_group_norm, aten.transpose, aten._unsafe_view, aten.clone, aten.t, aten.mm, aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_tem_fused__scaled_dot_product_efficient_attention__unsafe_view_add_clone_convolution_div_mm_native_group_norm_silu_t_transpose_view_12.run(buf51, arg18_1, arg19_1, buf56, 2048, 1, 1, stream=stream0)
        del arg18_1
        del arg19_1
        buf57 = reinterpret_tensor(buf51, (4, 1, 16384, 512), (8388608, 8388608, 512, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2, hidden_states_8, query, view_1, query_1, key, view_2, key_1, value, view_3, value_1, hidden_states_9], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.view, aten.native_group_norm, aten.transpose, aten._unsafe_view, aten.clone, aten.t, aten.mm, aten._scaled_dot_product_efficient_attention]
        stream0 = get_raw_stream(0)
        triton_tem_fused__scaled_dot_product_efficient_attention__unsafe_view_add_clone_convolution_div_mm_native_group_norm_silu_t_transpose_view_12.run(buf53, arg20_1, arg21_1, buf57, 2048, 1, 1, stream=stream0)
        del arg20_1
        del arg21_1
        del buf53
        # Topologically Sorted Source Nodes: [query, view_1, query_1, key, view_2, key_1, value, view_3, value_1, hidden_states_9], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
        buf58 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf55, buf56, buf57, None, False)
        del buf55
        del buf56
        buf59 = buf58[0]
        assert_size_stride(buf59, (4, 1, 16384, 512), (8388608, 512, 512, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf59, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf58
        buf63 = reinterpret_tensor(buf57, (65536, 512), (512, 1), 0); del buf57  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.mm:1484
        extern_kernels.mm(reinterpret_tensor(buf59, (65536, 512), (512, 1), 0), reinterpret_tensor(arg22_1, (512, 512), (1, 512), 0), out=buf63)
        del arg22_1
        del buf59
        buf64 = reinterpret_tensor(buf63, (4, 512, 128, 128), (8388608, 1, 65536, 512), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, , hidden_states_12, transpose_7, hidden_states_14, hidden_states_15, hidden_states_16], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.addmm, aten.view, aten.transpose]
        # [Provenance debug handles] triton_poi_fused_add_addmm_convolution_div_silu_transpose_view_13:1330
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convolution_div_silu_transpose_view_13.run(buf64, arg23_1, buf3, arg4_1, buf35, arg13_1, 33554432, stream=stream0)
        del arg13_1
        del arg23_1
        del arg4_1
        del buf3
        buf65 = buf38; del buf38  # reuse
        buf66 = buf37; del buf37  # reuse
        buf67 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_clone_native_group_norm_14:1331
        stream0 = get_raw_stream(0)
        triton_per_fused_clone_native_group_norm_14.run(buf64, buf65, buf66, buf67, 131072, 256, stream=stream0)
        buf68 = buf41; del buf41  # reuse
        buf69 = buf40; del buf40  # reuse
        buf70 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1332
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_5.run(buf65, buf66, buf67, buf68, buf69, buf70, 1024, 128, stream=stream0)
        buf71 = buf44; del buf44  # reuse
        buf72 = buf43; del buf43  # reuse
        buf73 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1333
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_6.run(buf68, buf69, buf70, buf71, buf72, buf73, 512, 2, stream=stream0)
        buf74 = buf46; del buf46  # reuse
        buf75 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_7:1334
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_7.run(buf71, buf72, buf73, buf74, buf75, 128, 4, stream=stream0)
        buf78 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_17, hidden_states_18], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
        # [Provenance debug handles] triton_poi_fused_clone_native_group_norm_silu_15:1335
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_native_group_norm_silu_15.run(buf64, buf74, buf75, arg24_1, arg25_1, buf78, 2097152, 16, stream=stream0)
        del arg24_1
        del arg25_1
        buf79 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_9:1336
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_9.run(arg26_1, buf79, 262144, 9, stream=stream0)
        del arg26_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf80 = extern_kernels.convolution(buf78, buf79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'unknown_op')
        buf81 = buf67; del buf67  # reuse
        buf82 = buf66; del buf66  # reuse
        buf83 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19, hidden_states_20], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_4:1337
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_4.run(buf80, arg27_1, buf81, buf82, buf83, 131072, 256, stream=stream0)
        buf84 = buf70; del buf70  # reuse
        buf85 = buf69; del buf69  # reuse
        buf86 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19, hidden_states_20], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1338
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_5.run(buf81, buf82, buf83, buf84, buf85, buf86, 1024, 128, stream=stream0)
        buf87 = buf73; del buf73  # reuse
        buf88 = buf72; del buf72  # reuse
        buf89 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19, hidden_states_20], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1339
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_6.run(buf84, buf85, buf86, buf87, buf88, buf89, 512, 2, stream=stream0)
        buf90 = buf75; del buf75  # reuse
        buf91 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19, hidden_states_20], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_7:1340
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_7.run(buf87, buf88, buf89, buf90, buf91, 128, 4, stream=stream0)
        buf94 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19, hidden_states_20, hidden_states_21], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_8:1341
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_silu_8.run(buf80, arg27_1, buf90, buf91, arg28_1, arg29_1, buf94, 2097152, 16, stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del buf80
        buf95 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_9:1342
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_9.run(arg30_1, buf95, 262144, 9, stream=stream0)
        del arg30_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf96 = extern_kernels.convolution(buf94, buf95, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'unknown_op')
        del buf94
        del buf95
        buf97 = buf83; del buf83  # reuse
        buf98 = buf82; del buf82  # reuse
        buf99 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, sample_1, hidden_states_24], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_16:1343
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_16.run(buf64, buf96, arg31_1, buf97, buf98, buf99, 131072, 256, stream=stream0)
        buf100 = buf86; del buf86  # reuse
        buf101 = buf85; del buf85  # reuse
        buf102 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, sample_1, hidden_states_24], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1344
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_5.run(buf97, buf98, buf99, buf100, buf101, buf102, 1024, 128, stream=stream0)
        buf103 = buf89; del buf89  # reuse
        buf104 = buf88; del buf88  # reuse
        buf105 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, sample_1, hidden_states_24], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1345
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_6.run(buf100, buf101, buf102, buf103, buf104, buf105, 512, 2, stream=stream0)
        buf106 = buf91; del buf91  # reuse
        buf107 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, sample_1, hidden_states_24], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_7:1346
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_7.run(buf103, buf104, buf105, buf106, buf107, 128, 4, stream=stream0)
        buf109 = empty_strided_cuda((4, 512, 128, 128), (8388608, 1, 65536, 512), torch.float32)
        buf110 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, sample_1, hidden_states_24, hidden_states_25], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_17:1347
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_17.run(buf110, buf64, buf96, arg31_1, buf106, buf107, arg5_1, arg32_1, 2097152, 16, stream=stream0)
        del arg32_1
        del arg5_1
        buf111 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_18:1348
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_18.run(arg33_1, buf111, 262144, 9, stream=stream0)
        del arg33_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf112 = extern_kernels.convolution(buf110, buf111, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'unknown_op')
        del buf110
        buf113 = buf99; del buf99  # reuse
        buf114 = buf98; del buf98  # reuse
        buf115 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26, hidden_states_27], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_19:1349
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_silu_19.run(buf112, arg34_1, buf113, buf114, buf115, 131072, 256, stream=stream0)
        buf116 = buf102; del buf102  # reuse
        buf117 = buf101; del buf101  # reuse
        buf118 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26, hidden_states_27], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1350
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_5.run(buf113, buf114, buf115, buf116, buf117, buf118, 1024, 128, stream=stream0)
        buf119 = buf105; del buf105  # reuse
        buf120 = buf104; del buf104  # reuse
        buf121 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26, hidden_states_27], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1351
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_6.run(buf116, buf117, buf118, buf119, buf120, buf121, 512, 2, stream=stream0)
        buf122 = buf107; del buf107  # reuse
        buf123 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26, hidden_states_27], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_7:1352
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_7.run(buf119, buf120, buf121, buf122, buf123, 128, 4, stream=stream0)
        buf125 = buf112; del buf112  # reuse
        buf126 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26, hidden_states_27, hidden_states_28], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_20:1353
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_silu_20.run(buf126, arg34_1, buf122, buf123, arg35_1, arg36_1, 2097152, 16, stream=stream0)
        del arg34_1
        del arg35_1
        del arg36_1
        buf127 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28, hidden_states_30], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_18:1354
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_18.run(arg37_1, buf127, 262144, 9, stream=stream0)
        del arg37_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf128 = extern_kernels.convolution(buf126, buf127, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'unknown_op')
        buf129 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, sample_1, hidden_states_28, hidden_states_30, add_3, output_tensor_2], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy]
        # [Provenance debug handles] triton_poi_fused__to_copy_add_convolution_div_silu_21:1355
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_convolution_div_silu_21.run(buf129, buf64, buf96, arg31_1, arg38_1, 33554432, stream=stream0)
        del arg31_1
        del arg38_1
        del buf64
        del buf96
        buf130 = buf115; del buf115  # reuse
        buf131 = buf114; del buf114  # reuse
        buf132 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31], Original ATen: [aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_clone_native_group_norm_22:1356
        stream0 = get_raw_stream(0)
        triton_per_fused_clone_native_group_norm_22.run(buf129, buf130, buf131, buf132, 131072, 256, stream=stream0)
        buf133 = buf118; del buf118  # reuse
        buf134 = buf117; del buf117  # reuse
        buf135 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31], Original ATen: [aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1357
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_5.run(buf130, buf131, buf132, buf133, buf134, buf135, 1024, 128, stream=stream0)
        buf136 = buf121; del buf121  # reuse
        buf137 = buf120; del buf120  # reuse
        buf138 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31], Original ATen: [aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1358
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_6.run(buf133, buf134, buf135, buf136, buf137, buf138, 512, 2, stream=stream0)
        buf139 = buf123; del buf123  # reuse
        buf140 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31], Original ATen: [aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_7:1359
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_7.run(buf136, buf137, buf138, buf139, buf140, 128, 4, stream=stream0)
        buf142 = buf126; del buf126  # reuse
        buf143 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31, hidden_states_32], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
        # [Provenance debug handles] triton_poi_fused_clone_native_group_norm_silu_23:1360
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_native_group_norm_silu_23.run(buf143, buf129, buf139, buf140, arg39_1, arg40_1, 2097152, 16, stream=stream0)
        del arg39_1
        del arg40_1
        buf144 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_18:1361
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_18.run(arg41_1, buf144, 262144, 9, stream=stream0)
        del arg41_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf145 = extern_kernels.convolution(buf143, buf144, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'unknown_op')
        del buf143
        buf146 = buf132; del buf132  # reuse
        buf147 = buf131; del buf131  # reuse
        buf148 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33, hidden_states_34], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_19:1362
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_silu_19.run(buf145, arg42_1, buf146, buf147, buf148, 131072, 256, stream=stream0)
        buf149 = buf135; del buf135  # reuse
        buf150 = buf134; del buf134  # reuse
        buf151 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33, hidden_states_34], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1363
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_5.run(buf146, buf147, buf148, buf149, buf150, buf151, 1024, 128, stream=stream0)
        buf152 = buf138; del buf138  # reuse
        buf153 = buf137; del buf137  # reuse
        buf154 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33, hidden_states_34], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1364
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_6.run(buf149, buf150, buf151, buf152, buf153, buf154, 512, 2, stream=stream0)
        buf155 = buf140; del buf140  # reuse
        buf156 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33, hidden_states_34], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_7:1365
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_7.run(buf152, buf153, buf154, buf155, buf156, 128, 4, stream=stream0)
        buf158 = buf145; del buf145  # reuse
        buf159 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33, hidden_states_34, hidden_states_35], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_20:1366
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_silu_20.run(buf159, arg42_1, buf155, buf156, arg43_1, arg44_1, 2097152, 16, stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        buf160 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_18:1367
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_18.run(arg45_1, buf160, 262144, 9, stream=stream0)
        del arg45_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf161 = extern_kernels.convolution(buf159, buf160, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'unknown_op')
        buf162 = buf148; del buf148  # reuse
        buf163 = buf147; del buf147  # reuse
        buf164 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_38], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_add_clone_convolution_div_native_group_norm_silu_24:1368
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_convolution_div_native_group_norm_silu_24.run(buf129, buf161, arg46_1, buf162, buf163, buf164, 131072, 256, stream=stream0)
        buf165 = buf151; del buf151  # reuse
        buf166 = buf150; del buf150  # reuse
        buf167 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_38], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1369
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_5.run(buf162, buf163, buf164, buf165, buf166, buf167, 1024, 128, stream=stream0)
        buf168 = buf154; del buf154  # reuse
        buf169 = buf153; del buf153  # reuse
        buf170 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_38], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1370
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_6.run(buf165, buf166, buf167, buf168, buf169, buf170, 512, 2, stream=stream0)
        buf171 = buf156; del buf156  # reuse
        buf172 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_38], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_7:1371
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_7.run(buf168, buf169, buf170, buf171, buf172, 128, 4, stream=stream0)
        buf174 = buf159; del buf159  # reuse
        buf175 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_38, hidden_states_39], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_25:1372
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_25.run(buf175, buf129, buf161, arg46_1, buf171, buf172, arg47_1, arg48_1, 2097152, 16, stream=stream0)
        del arg47_1
        del arg48_1
        buf176 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_18:1373
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_18.run(arg49_1, buf176, 262144, 9, stream=stream0)
        del arg49_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf177 = extern_kernels.convolution(buf175, buf176, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'unknown_op')
        del buf175
        buf178 = buf164; del buf164  # reuse
        buf179 = buf163; del buf163  # reuse
        buf180 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40, hidden_states_41], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_19:1374
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_silu_19.run(buf177, arg50_1, buf178, buf179, buf180, 131072, 256, stream=stream0)
        buf181 = buf167; del buf167  # reuse
        buf182 = buf166; del buf166  # reuse
        buf183 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40, hidden_states_41], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1375
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_5.run(buf178, buf179, buf180, buf181, buf182, buf183, 1024, 128, stream=stream0)
        buf184 = buf170; del buf170  # reuse
        buf185 = buf169; del buf169  # reuse
        buf186 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40, hidden_states_41], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1376
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_6.run(buf181, buf182, buf183, buf184, buf185, buf186, 512, 2, stream=stream0)
        buf187 = buf172; del buf172  # reuse
        buf188 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40, hidden_states_41], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_7:1377
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_7.run(buf184, buf185, buf186, buf187, buf188, 128, 4, stream=stream0)
        del buf184
        del buf185
        del buf186
        buf190 = buf177; del buf177  # reuse
        buf191 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40, hidden_states_41, hidden_states_42], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_20:1378
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_silu_20.run(buf191, arg50_1, buf187, buf188, arg51_1, arg52_1, 2097152, 16, stream=stream0)
        del arg50_1
        del arg51_1
        del arg52_1
        buf192 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_42, hidden_states_44], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_18:1379
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_18.run(arg53_1, buf192, 262144, 9, stream=stream0)
        del arg53_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf193 = extern_kernels.convolution(buf191, buf192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'unknown_op')
        del buf191
        buf194 = empty_strided_cuda((4, 512, 256, 256), (33554432, 1, 131072, 512), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
        # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_26:1380
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_26.run(buf129, buf161, arg46_1, buf193, arg54_1, buf194, 134217728, stream=stream0)
        del arg46_1
        del arg54_1
        del buf129
        del buf161
        del buf193
        buf195 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_18:1381
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_18.run(arg55_1, buf195, 262144, 9, stream=stream0)
        del arg55_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf196 = extern_kernels.convolution(buf194, buf195, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 512, 256, 256), (33554432, 1, 131072, 512), 'unknown_op')
        buf197 = empty_strided_cuda((4, 32, 1, 1, 8, 64), (16384, 8, 65536, 65536, 1, 256), torch.float32)
        buf198 = empty_strided_cuda((4, 32, 1, 1, 8, 64), (16384, 8, 65536, 65536, 1, 256), torch.float32)
        buf199 = empty_strided_cuda((4, 32, 1, 1, 8, 64), (16384, 8, 65536, 65536, 1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_47], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27:1382
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27.run(buf196, arg56_1, buf197, buf198, buf199, 65536, 2048, stream=stream0)
        buf200 = reinterpret_tensor(buf183, (4, 32, 1, 1, 8), (256, 8, 1024, 1024, 1), 0); del buf183  # reuse
        buf201 = reinterpret_tensor(buf182, (4, 32, 1, 1, 8), (256, 8, 1024, 1024, 1), 0); del buf182  # reuse
        buf202 = reinterpret_tensor(buf181, (4, 32, 1, 1, 8), (256, 8, 1024, 1024, 1), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_47], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28:1383
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28.run(buf197, buf198, buf199, buf200, buf201, buf202, 1024, 64, stream=stream0)
        buf203 = buf188; del buf188  # reuse
        buf204 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_47], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29:1384
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29.run(buf200, buf201, buf202, buf203, buf204, 128, 8, stream=stream0)
        buf206 = buf194; del buf194  # reuse
        buf207 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_47, hidden_states_48], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_30:1385
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_30.run(buf207, buf196, arg56_1, buf203, buf204, arg57_1, arg58_1, 8388608, 16, stream=stream0)
        del arg57_1
        del arg58_1
        buf208 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_18:1386
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_18.run(arg59_1, buf208, 262144, 9, stream=stream0)
        del arg59_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf209 = extern_kernels.convolution(buf207, buf208, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (4, 512, 256, 256), (33554432, 1, 131072, 512), 'unknown_op')
        del buf207
        buf210 = buf199; del buf199  # reuse
        buf211 = buf198; del buf198  # reuse
        buf212 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49, hidden_states_50], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27:1387
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27.run(buf209, arg60_1, buf210, buf211, buf212, 65536, 2048, stream=stream0)
        buf213 = buf202; del buf202  # reuse
        buf214 = buf201; del buf201  # reuse
        buf215 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49, hidden_states_50], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28:1388
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28.run(buf210, buf211, buf212, buf213, buf214, buf215, 1024, 64, stream=stream0)
        buf216 = buf204; del buf204  # reuse
        buf217 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49, hidden_states_50], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29:1389
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29.run(buf213, buf214, buf215, buf216, buf217, 128, 8, stream=stream0)
        buf219 = buf209; del buf209  # reuse
        buf220 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49, hidden_states_50, hidden_states_51], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_31:1390
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_silu_31.run(buf220, arg60_1, buf216, buf217, arg61_1, arg62_1, 8388608, 16, stream=stream0)
        del arg60_1
        del arg61_1
        del arg62_1
        buf221 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_51, hidden_states_53], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_18:1391
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_18.run(arg63_1, buf221, 262144, 9, stream=stream0)
        del arg63_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf222 = extern_kernels.convolution(buf220, buf221, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (4, 512, 256, 256), (33554432, 1, 131072, 512), 'unknown_op')
        buf223 = buf212; del buf212  # reuse
        buf224 = buf211; del buf211  # reuse
        buf225 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_54], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_32:1392
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_32.run(buf196, arg56_1, buf222, arg64_1, buf223, buf224, buf225, 65536, 2048, stream=stream0)
        buf226 = buf215; del buf215  # reuse
        buf227 = buf214; del buf214  # reuse
        buf228 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_54], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28:1393
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28.run(buf223, buf224, buf225, buf226, buf227, buf228, 1024, 64, stream=stream0)
        buf229 = buf217; del buf217  # reuse
        buf230 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_54], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29:1394
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29.run(buf226, buf227, buf228, buf229, buf230, 128, 8, stream=stream0)
        buf232 = buf220; del buf220  # reuse
        buf233 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_54, hidden_states_55], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_33:1395
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_33.run(buf233, buf196, arg56_1, buf222, arg64_1, buf229, buf230, arg65_1, arg66_1, 8388608, 16, stream=stream0)
        del arg65_1
        del arg66_1
        buf234 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_18:1396
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_18.run(arg67_1, buf234, 262144, 9, stream=stream0)
        del arg67_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf235 = extern_kernels.convolution(buf233, buf234, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (4, 512, 256, 256), (33554432, 1, 131072, 512), 'unknown_op')
        del buf233
        buf236 = buf225; del buf225  # reuse
        buf237 = buf224; del buf224  # reuse
        buf238 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56, hidden_states_57], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27:1397
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27.run(buf235, arg68_1, buf236, buf237, buf238, 65536, 2048, stream=stream0)
        buf239 = buf228; del buf228  # reuse
        buf240 = buf227; del buf227  # reuse
        buf241 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56, hidden_states_57], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28:1398
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28.run(buf236, buf237, buf238, buf239, buf240, buf241, 1024, 64, stream=stream0)
        buf242 = buf230; del buf230  # reuse
        buf243 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56, hidden_states_57], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29:1399
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29.run(buf239, buf240, buf241, buf242, buf243, 128, 8, stream=stream0)
        buf245 = buf235; del buf235  # reuse
        buf246 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56, hidden_states_57, hidden_states_58], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_31:1400
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_silu_31.run(buf246, arg68_1, buf242, buf243, arg69_1, arg70_1, 8388608, 16, stream=stream0)
        del arg68_1
        del arg69_1
        del arg70_1
        buf247 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_58, hidden_states_60], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_18:1401
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_18.run(arg71_1, buf247, 262144, 9, stream=stream0)
        del arg71_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf248 = extern_kernels.convolution(buf246, buf247, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (4, 512, 256, 256), (33554432, 1, 131072, 512), 'unknown_op')
        del buf246
        buf249 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_58, hidden_states_60, add_7, output_tensor_6], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
        # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_34:1402
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_34.run(buf249, arg56_1, buf222, arg64_1, buf248, arg72_1, 134217728, stream=stream0)
        del arg56_1
        del arg64_1
        del arg72_1
        del buf222
        buf250 = buf238; del buf238  # reuse
        buf251 = buf237; del buf237  # reuse
        buf252 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_clone_native_group_norm_35:1403
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_native_group_norm_35.run(buf249, buf250, buf251, buf252, 65536, 2048, stream=stream0)
        buf253 = buf241; del buf241  # reuse
        buf254 = buf240; del buf240  # reuse
        buf255 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28:1404
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28.run(buf250, buf251, buf252, buf253, buf254, buf255, 1024, 64, stream=stream0)
        buf256 = buf243; del buf243  # reuse
        buf257 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29:1405
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29.run(buf253, buf254, buf255, buf256, buf257, 128, 8, stream=stream0)
        buf259 = buf248; del buf248  # reuse
        buf260 = buf259; del buf259  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_62], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
        # [Provenance debug handles] triton_poi_fused_clone_native_group_norm_silu_36:1406
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_native_group_norm_silu_36.run(buf260, buf249, buf256, buf257, arg73_1, arg74_1, 8388608, 16, stream=stream0)
        del arg73_1
        del arg74_1
        buf261 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_62, hidden_states_63], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_18:1407
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_18.run(arg75_1, buf261, 262144, 9, stream=stream0)
        del arg75_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf262 = extern_kernels.convolution(buf260, buf261, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (4, 512, 256, 256), (33554432, 1, 131072, 512), 'unknown_op')
        del buf260
        buf263 = buf252; del buf252  # reuse
        buf264 = buf251; del buf251  # reuse
        buf265 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_62, hidden_states_63, hidden_states_64], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27:1408
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27.run(buf262, arg76_1, buf263, buf264, buf265, 65536, 2048, stream=stream0)
        buf266 = buf255; del buf255  # reuse
        buf267 = buf254; del buf254  # reuse
        buf268 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_62, hidden_states_63, hidden_states_64], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28:1409
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28.run(buf263, buf264, buf265, buf266, buf267, buf268, 1024, 64, stream=stream0)
        del buf263
        del buf264
        del buf265
        buf269 = buf257; del buf257  # reuse
        buf270 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_62, hidden_states_63, hidden_states_64], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29:1410
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29.run(buf266, buf267, buf268, buf269, buf270, 128, 8, stream=stream0)
        del buf266
        del buf267
        del buf268
        buf272 = buf262; del buf262  # reuse
        buf273 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_62, hidden_states_63, hidden_states_64, hidden_states_65], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_31:1411
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_silu_31.run(buf273, arg76_1, buf269, buf270, arg77_1, arg78_1, 8388608, 16, stream=stream0)
        del arg76_1
        del arg77_1
        del arg78_1
        buf274 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_18:1412
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_18.run(arg79_1, buf274, 262144, 9, stream=stream0)
        del arg79_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf275 = extern_kernels.convolution(buf273, buf274, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (4, 512, 256, 256), (33554432, 1, 131072, 512), 'unknown_op')
        del buf273
        buf276 = empty_strided_cuda((4, 512, 512, 512), (134217728, 1, 262144, 512), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
        # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_37:1413
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_37.run(buf249, buf275, arg80_1, buf276, 536870912, stream=stream0)
        del arg80_1
        del buf249
        del buf275
        buf277 = buf274; del buf274  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_18:1414
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_18.run(arg81_1, buf277, 262144, 9, stream=stream0)
        del arg81_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf278 = extern_kernels.convolution(buf276, buf277, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (4, 512, 512, 512), (134217728, 1, 262144, 512), 'unknown_op')
        del buf277
        buf279 = empty_strided_cuda((4, 32, 1, 1, 32, 64), (65536, 1, 262144, 262144, 2048, 32), torch.float32)
        buf280 = empty_strided_cuda((4, 32, 1, 1, 32, 64), (65536, 1, 262144, 262144, 2048, 32), torch.float32)
        buf281 = empty_strided_cuda((4, 32, 1, 1, 32, 64), (65536, 1, 262144, 262144, 2048, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, hidden_states_70], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38:1415
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38.run(buf278, arg82_1, buf279, buf280, buf281, 262144, 2048, stream=stream0)
        buf282 = empty_strided_cuda((4, 32, 1, 1, 32), (1024, 1, 4096, 4096, 32), torch.float32)
        buf283 = empty_strided_cuda((4, 32, 1, 1, 32), (1024, 1, 4096, 4096, 32), torch.float32)
        buf284 = empty_strided_cuda((4, 32, 1, 1, 32), (1024, 1, 4096, 4096, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, hidden_states_70], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39:1416
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39.run(buf279, buf280, buf281, buf282, buf283, buf284, 4096, 64, stream=stream0)
        del buf279
        del buf280
        del buf281
        buf285 = buf270; del buf270  # reuse
        buf286 = buf269; del buf269  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, hidden_states_70], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40:1417
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40.run(buf282, buf283, buf284, buf285, buf286, 128, 32, stream=stream0)
        del buf282
        del buf283
        del buf284
        buf288 = buf276; del buf276  # reuse
        buf301 = empty_strided_cuda((4, 512, 512, 512), (134217728, 1, 262144, 512), torch.float32)
        buf289 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, hidden_states_70, hidden_states_71, contiguous, input_tensor], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_41:1418
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_41.run(buf289, buf278, arg82_1, buf285, buf286, arg83_1, arg84_1, buf301, 33554432, 16, stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del buf278
        buf290 = empty_strided_cuda((256, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_42:1419
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_42.run(arg85_1, buf290, 131072, 9, stream=stream0)
        del arg85_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf291 = extern_kernels.convolution(buf289, buf290, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (4, 256, 512, 512), (67108864, 1, 131072, 256), 'unknown_op')
        del buf289
        del buf290
        buf292 = reinterpret_tensor(buf180, (4, 32, 1, 1, 16, 64), (32768, 1, 131072, 131072, 2048, 32), 0); del buf180  # reuse
        buf293 = reinterpret_tensor(buf179, (4, 32, 1, 1, 16, 64), (32768, 1, 131072, 131072, 2048, 32), 0); del buf179  # reuse
        buf294 = reinterpret_tensor(buf178, (4, 32, 1, 1, 16, 64), (32768, 1, 131072, 131072, 2048, 32), 0); del buf178  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_convolution_native_group_norm_silu_43:1420
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_silu_43.run(buf291, arg86_1, buf292, buf293, buf294, 131072, 2048, stream=stream0)
        buf295 = empty_strided_cuda((4, 32, 1, 1, 16), (512, 1, 2048, 2048, 32), torch.float32)
        buf296 = empty_strided_cuda((4, 32, 1, 1, 16), (512, 1, 2048, 2048, 32), torch.float32)
        buf297 = empty_strided_cuda((4, 32, 1, 1, 16), (512, 1, 2048, 2048, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_44:1421
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_silu_44.run(buf292, buf293, buf294, buf295, buf296, buf297, 2048, 64, stream=stream0)
        buf298 = buf286; del buf286  # reuse
        buf299 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_45:1422
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_silu_45.run(buf295, buf296, buf297, buf298, buf299, 128, 16, stream=stream0)
        buf302 = empty_strided_cuda((1048576, 256), (256, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.bias_addmm:1485
        extern_kernels.bias_addmm(reinterpret_tensor(arg92_1, (1048576, 256), (0, 1), 0), reinterpret_tensor(buf301, (1048576, 512), (512, 1), 0), reinterpret_tensor(arg91_1, (512, 256), (1, 512), 0), alpha=1, beta=1, out=buf302)
        del arg91_1
        del arg92_1
        del buf301
        buf303 = buf291; del buf291  # reuse
        buf304 = buf303; del buf303  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73, hidden_states_74], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_46:1423
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_silu_46.run(buf304, arg86_1, buf298, buf299, arg87_1, arg88_1, 268435456, stream=stream0)
        del arg86_1
        del arg87_1
        del arg88_1
        buf305 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_74, hidden_states_76], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_47:1424
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_47.run(arg89_1, buf305, 65536, 9, stream=stream0)
        del arg89_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf306 = extern_kernels.convolution(buf304, buf305, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (4, 256, 512, 512), (67108864, 1, 131072, 256), 'unknown_op')
        buf307 = buf294; del buf294  # reuse
        buf308 = buf293; del buf293  # reuse
        buf309 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, contiguous, input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_77], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_48:1425
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_48.run(buf302, buf306, arg90_1, buf307, buf308, buf309, 131072, 2048, stream=stream0)
        buf310 = buf297; del buf297  # reuse
        buf311 = buf296; del buf296  # reuse
        buf312 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, contiguous, input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_77], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_44:1426
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_silu_44.run(buf307, buf308, buf309, buf310, buf311, buf312, 2048, 64, stream=stream0)
        buf313 = buf299; del buf299  # reuse
        buf314 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, contiguous, input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_77], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_45:1427
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_silu_45.run(buf310, buf311, buf312, buf313, buf314, 128, 16, stream=stream0)
        buf316 = buf304; del buf304  # reuse
        buf317 = buf316; del buf316  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, contiguous, input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_77, hidden_states_78], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_49:1428
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_49.run(buf317, buf302, buf306, arg90_1, buf313, buf314, arg93_1, arg94_1, 268435456, stream=stream0)
        del arg93_1
        del arg94_1
        buf318 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_78, hidden_states_79], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_47:1429
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_47.run(arg95_1, buf318, 65536, 9, stream=stream0)
        del arg95_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf319 = extern_kernels.convolution(buf317, buf318, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (4, 256, 512, 512), (67108864, 1, 131072, 256), 'unknown_op')
        del buf317
        buf320 = buf309; del buf309  # reuse
        buf321 = buf308; del buf308  # reuse
        buf322 = buf307; del buf307  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_78, hidden_states_79, hidden_states_80], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_convolution_native_group_norm_silu_43:1430
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_silu_43.run(buf319, arg96_1, buf320, buf321, buf322, 131072, 2048, stream=stream0)
        buf323 = buf312; del buf312  # reuse
        buf324 = buf311; del buf311  # reuse
        buf325 = buf310; del buf310  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_78, hidden_states_79, hidden_states_80], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_44:1431
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_silu_44.run(buf320, buf321, buf322, buf323, buf324, buf325, 2048, 64, stream=stream0)
        buf326 = buf314; del buf314  # reuse
        buf327 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_78, hidden_states_79, hidden_states_80], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_45:1432
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_silu_45.run(buf323, buf324, buf325, buf326, buf327, 128, 16, stream=stream0)
        buf329 = buf319; del buf319  # reuse
        buf330 = buf329; del buf329  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_78, hidden_states_79, hidden_states_80, hidden_states_81], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_46:1433
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_silu_46.run(buf330, arg96_1, buf326, buf327, arg97_1, arg98_1, 268435456, stream=stream0)
        del arg96_1
        del arg97_1
        del arg98_1
        buf331 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_81, hidden_states_83], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_47:1434
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_47.run(arg99_1, buf331, 65536, 9, stream=stream0)
        del arg99_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf332 = extern_kernels.convolution(buf330, buf331, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf332, (4, 256, 512, 512), (67108864, 1, 131072, 256), 'unknown_op')
        buf333 = reinterpret_tensor(buf330, (4, 256, 512, 512), (67108864, 262144, 512, 1), 0); del buf330  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, contiguous, input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_81, hidden_states_83, add_10, output_tensor_9], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
        # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_50:1435
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_50.run(buf302, buf306, arg90_1, buf332, arg100_1, buf333, 1048576, 256, stream=stream0)
        del arg100_1
        del arg90_1
        del buf302
        del buf306
        buf334 = reinterpret_tensor(buf325, (4, 32, 1, 1, 16), (512, 16, 2048, 2048, 1), 0); del buf325  # reuse
        buf335 = reinterpret_tensor(buf324, (4, 32, 1, 1, 16), (512, 16, 2048, 2048, 1), 0); del buf324  # reuse
        buf336 = reinterpret_tensor(buf323, (4, 32, 1, 1, 16), (512, 16, 2048, 2048, 1), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_84], Original ATen: [aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_native_group_norm_51:1436
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_51.run(buf333, buf334, buf335, buf336, 2048, 131072, stream=stream0)
        buf337 = buf327; del buf327  # reuse
        buf338 = buf326; del buf326  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_84], Original ATen: [aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_native_group_norm_52:1437
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_52.run(buf334, buf335, buf336, buf337, buf338, 128, 16, stream=stream0)
        buf341 = buf332; del buf332  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_84, hidden_states_85], Original ATen: [aten.native_group_norm, aten.silu]
        # [Provenance debug handles] triton_poi_fused_native_group_norm_silu_53:1438
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_53.run(buf333, buf337, buf338, arg101_1, arg102_1, buf341, 1024, 262144, stream=stream0)
        del arg101_1
        del arg102_1
        buf342 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_47:1439
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_47.run(arg103_1, buf342, 65536, 9, stream=stream0)
        del arg103_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf343 = extern_kernels.convolution(buf341, buf342, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf343, (4, 256, 512, 512), (67108864, 1, 131072, 256), 'unknown_op')
        del buf341
        buf344 = buf322; del buf322  # reuse
        buf345 = buf321; del buf321  # reuse
        buf346 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86, hidden_states_87], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_convolution_native_group_norm_silu_43:1440
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_silu_43.run(buf343, arg104_1, buf344, buf345, buf346, 131072, 2048, stream=stream0)
        buf347 = reinterpret_tensor(buf336, (4, 32, 1, 1, 16), (512, 1, 2048, 2048, 32), 0); del buf336  # reuse
        buf348 = reinterpret_tensor(buf335, (4, 32, 1, 1, 16), (512, 1, 2048, 2048, 32), 0); del buf335  # reuse
        buf349 = reinterpret_tensor(buf334, (4, 32, 1, 1, 16), (512, 1, 2048, 2048, 32), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86, hidden_states_87], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_44:1441
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_silu_44.run(buf344, buf345, buf346, buf347, buf348, buf349, 2048, 64, stream=stream0)
        del buf344
        del buf345
        del buf346
        buf350 = buf338; del buf338  # reuse
        buf351 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86, hidden_states_87], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_45:1442
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_silu_45.run(buf347, buf348, buf349, buf350, buf351, 128, 16, stream=stream0)
        del buf347
        del buf348
        del buf349
        buf353 = buf343; del buf343  # reuse
        buf354 = buf353; del buf353  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86, hidden_states_87, hidden_states_88], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_46:1443
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_silu_46.run(buf354, arg104_1, buf350, buf351, arg105_1, arg106_1, 268435456, stream=stream0)
        del arg104_1
        del arg105_1
        del arg106_1
        buf355 = buf342; del buf342  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_47:1444
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_47.run(arg107_1, buf355, 65536, 9, stream=stream0)
        del arg107_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf356 = extern_kernels.convolution(buf354, buf355, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (4, 256, 512, 512), (67108864, 1, 131072, 256), 'unknown_op')
        del buf354
        buf357 = empty_strided_cuda((4, 256, 1024, 1024), (268435456, 1, 262144, 256), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index]
        # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_54:1445
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_54.run(buf333, buf356, arg108_1, buf357, 1073741824, stream=stream0)
        del arg108_1
        del buf333
        del buf356
        buf358 = buf355; del buf355  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91, hidden_states_92], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_47:1446
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_47.run(arg109_1, buf358, 65536, 9, stream=stream0)
        del arg109_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf359 = extern_kernels.convolution(buf357, buf358, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf359, (4, 256, 1024, 1024), (268435456, 1, 262144, 256), 'unknown_op')
        del buf358
        buf360 = reinterpret_tensor(buf357, (4, 256, 1024, 1024), (268435456, 1048576, 1024, 1), 0); del buf357  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91, hidden_states_92], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index]
        # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_55:1447
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_55.run(buf359, arg110_1, buf360, 1024, 1048576, stream=stream0)
        del arg110_1
        buf361 = empty_strided_cuda((4, 32, 1, 1, 64), (2048, 64, 8192, 8192, 1), torch.float32)
        buf362 = empty_strided_cuda((4, 32, 1, 1, 64), (2048, 64, 8192, 8192, 1), torch.float32)
        buf363 = empty_strided_cuda((4, 32, 1, 1, 64), (2048, 64, 8192, 8192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_native_group_norm_56:1448
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_56.run(buf360, buf361, buf362, buf363, 8192, 131072, stream=stream0)
        buf364 = buf351; del buf351  # reuse
        buf365 = buf350; del buf350  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_native_group_norm_57:1449
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_57.run(buf361, buf362, buf363, buf364, buf365, 128, 64, stream=stream0)
        del buf361
        del buf362
        del buf363
        buf368 = buf359; del buf359  # reuse
        buf380 = empty_strided_cuda((4, 256, 1024, 1024), (268435456, 1, 262144, 256), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_93, hidden_states_94, input_tensor_1], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_58:1450
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_silu_58.run(buf360, buf364, buf365, arg111_1, arg112_1, buf368, buf380, 1024, 1048576, stream=stream0)
        del arg111_1
        del arg112_1
        del buf360
        buf369 = empty_strided_cuda((128, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_59:1451
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_59.run(arg113_1, buf369, 32768, 9, stream=stream0)
        del arg113_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf370 = extern_kernels.convolution(buf368, buf369, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf370, (4, 128, 1024, 1024), (134217728, 1, 131072, 128), 'unknown_op')
        del buf368
        del buf369
        buf371 = empty_strided_cuda((4, 32, 1, 1, 32, 64), (65536, 1, 262144, 262144, 2048, 32), torch.float32)
        buf372 = empty_strided_cuda((4, 32, 1, 1, 32, 64), (65536, 1, 262144, 262144, 2048, 32), torch.float32)
        buf373 = empty_strided_cuda((4, 32, 1, 1, 32, 64), (65536, 1, 262144, 262144, 2048, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95, hidden_states_96], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_convolution_native_group_norm_silu_60:1452
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_silu_60.run(buf370, arg114_1, buf371, buf372, buf373, 262144, 2048, stream=stream0)
        buf374 = empty_strided_cuda((4, 32, 1, 1, 32), (1024, 1, 4096, 4096, 32), torch.float32)
        buf375 = empty_strided_cuda((4, 32, 1, 1, 32), (1024, 1, 4096, 4096, 32), torch.float32)
        buf376 = empty_strided_cuda((4, 32, 1, 1, 32), (1024, 1, 4096, 4096, 32), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95, hidden_states_96], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39:1453
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39.run(buf371, buf372, buf373, buf374, buf375, buf376, 4096, 64, stream=stream0)
        buf377 = buf365; del buf365  # reuse
        buf378 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95, hidden_states_96], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40:1454
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40.run(buf374, buf375, buf376, buf377, buf378, 128, 32, stream=stream0)
        buf381 = empty_strided_cuda((4194304, 128), (128, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.bias_addmm:1486
        extern_kernels.bias_addmm(reinterpret_tensor(arg120_1, (4194304, 128), (0, 1), 0), reinterpret_tensor(buf380, (4194304, 256), (256, 1), 0), reinterpret_tensor(arg119_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf381)
        del arg119_1
        del arg120_1
        del buf380
        buf382 = buf370; del buf370  # reuse
        buf383 = buf382; del buf382  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95, hidden_states_96, hidden_states_97], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_61:1455
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_silu_61.run(buf383, arg114_1, buf377, buf378, arg115_1, arg116_1, 536870912, stream=stream0)
        del arg114_1
        del arg115_1
        del arg116_1
        buf384 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_97, hidden_states_99], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_62:1456
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_62.run(arg117_1, buf384, 16384, 9, stream=stream0)
        del arg117_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf385 = extern_kernels.convolution(buf383, buf384, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf385, (4, 128, 1024, 1024), (134217728, 1, 131072, 128), 'unknown_op')
        buf386 = buf373; del buf373  # reuse
        buf387 = buf372; del buf372  # reuse
        buf388 = buf371; del buf371  # reuse
        # Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_100], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_add_convolution_div_native_group_norm_silu_63:1457
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_div_native_group_norm_silu_63.run(buf381, buf385, arg118_1, buf386, buf387, buf388, 262144, 2048, stream=stream0)
        buf389 = buf376; del buf376  # reuse
        buf390 = buf375; del buf375  # reuse
        buf391 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_100], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39:1458
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39.run(buf386, buf387, buf388, buf389, buf390, buf391, 4096, 64, stream=stream0)
        buf392 = buf378; del buf378  # reuse
        buf393 = buf377; del buf377  # reuse
        # Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_100], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40:1459
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40.run(buf389, buf390, buf391, buf392, buf393, 128, 32, stream=stream0)
        buf395 = buf383; del buf383  # reuse
        buf396 = buf395; del buf395  # reuse
        # Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_100, hidden_states_101], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_add_convolution_div_native_group_norm_silu_64:1460
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_native_group_norm_silu_64.run(buf396, buf381, buf385, arg118_1, buf392, buf393, arg121_1, arg122_1, 536870912, stream=stream0)
        del arg121_1
        del arg122_1
        buf397 = buf384; del buf384  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_62:1461
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_62.run(arg123_1, buf397, 16384, 9, stream=stream0)
        del arg123_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf398 = extern_kernels.convolution(buf396, buf397, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf398, (4, 128, 1024, 1024), (134217728, 1, 131072, 128), 'unknown_op')
        del buf396
        buf399 = buf388; del buf388  # reuse
        buf400 = buf387; del buf387  # reuse
        buf401 = buf386; del buf386  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102, hidden_states_103], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_convolution_native_group_norm_silu_60:1462
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_silu_60.run(buf398, arg124_1, buf399, buf400, buf401, 262144, 2048, stream=stream0)
        buf402 = buf391; del buf391  # reuse
        buf403 = buf390; del buf390  # reuse
        buf404 = buf389; del buf389  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102, hidden_states_103], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39:1463
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39.run(buf399, buf400, buf401, buf402, buf403, buf404, 4096, 64, stream=stream0)
        buf405 = buf393; del buf393  # reuse
        buf406 = buf392; del buf392  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102, hidden_states_103], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40:1464
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40.run(buf402, buf403, buf404, buf405, buf406, 128, 32, stream=stream0)
        buf408 = buf398; del buf398  # reuse
        buf409 = buf408; del buf408  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102, hidden_states_103, hidden_states_104], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_61:1465
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_silu_61.run(buf409, arg124_1, buf405, buf406, arg125_1, arg126_1, 536870912, stream=stream0)
        del arg124_1
        del arg125_1
        del arg126_1
        buf410 = buf397; del buf397  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_104, hidden_states_106], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_62:1466
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_62.run(arg127_1, buf410, 16384, 9, stream=stream0)
        del arg127_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf411 = extern_kernels.convolution(buf409, buf410, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf411, (4, 128, 1024, 1024), (134217728, 1, 131072, 128), 'unknown_op')
        buf412 = reinterpret_tensor(buf409, (4, 128, 1024, 1024), (134217728, 1048576, 1024, 1), 0); del buf409  # reuse
        # Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_104, hidden_states_106, add_13, output_tensor_12], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div]
        # [Provenance debug handles] triton_poi_fused_add_convolution_div_silu_65:1467
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_silu_65.run(buf381, buf385, arg118_1, buf411, arg128_1, buf412, 4194304, 128, stream=stream0)
        del arg118_1
        del arg128_1
        del buf381
        del buf385
        buf413 = reinterpret_tensor(buf404, (4, 32, 1, 1, 32), (1024, 32, 4096, 4096, 1), 0); del buf404  # reuse
        buf414 = reinterpret_tensor(buf403, (4, 32, 1, 1, 32), (1024, 32, 4096, 4096, 1), 0); del buf403  # reuse
        buf415 = reinterpret_tensor(buf402, (4, 32, 1, 1, 32), (1024, 32, 4096, 4096, 1), 0); del buf402  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_107], Original ATen: [aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_native_group_norm_66:1468
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_66.run(buf412, buf413, buf414, buf415, 4096, 131072, stream=stream0)
        buf416 = buf406; del buf406  # reuse
        buf417 = buf405; del buf405  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_107], Original ATen: [aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_native_group_norm_67:1469
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_67.run(buf413, buf414, buf415, buf416, buf417, 128, 32, stream=stream0)
        buf420 = buf411; del buf411  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_107, hidden_states_108], Original ATen: [aten.native_group_norm, aten.silu]
        # [Provenance debug handles] triton_poi_fused_native_group_norm_silu_68:1470
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_68.run(buf412, buf416, buf417, arg129_1, arg130_1, buf420, 512, 1048576, stream=stream0)
        del arg129_1
        del arg130_1
        buf421 = buf410; del buf410  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_62:1471
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_62.run(arg131_1, buf421, 16384, 9, stream=stream0)
        del arg131_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf422 = extern_kernels.convolution(buf420, buf421, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf422, (4, 128, 1024, 1024), (134217728, 1, 131072, 128), 'unknown_op')
        del buf420
        buf423 = buf401; del buf401  # reuse
        buf424 = buf400; del buf400  # reuse
        buf425 = buf399; del buf399  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109, hidden_states_110], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_convolution_native_group_norm_silu_60:1472
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_silu_60.run(buf422, arg132_1, buf423, buf424, buf425, 262144, 2048, stream=stream0)
        buf426 = reinterpret_tensor(buf415, (4, 32, 1, 1, 32), (1024, 1, 4096, 4096, 32), 0); del buf415  # reuse
        buf427 = reinterpret_tensor(buf414, (4, 32, 1, 1, 32), (1024, 1, 4096, 4096, 32), 0); del buf414  # reuse
        buf428 = reinterpret_tensor(buf413, (4, 32, 1, 1, 32), (1024, 1, 4096, 4096, 32), 0); del buf413  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109, hidden_states_110], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39:1473
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39.run(buf423, buf424, buf425, buf426, buf427, buf428, 4096, 64, stream=stream0)
        buf429 = buf417; del buf417  # reuse
        buf430 = buf416; del buf416  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109, hidden_states_110], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40:1474
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40.run(buf426, buf427, buf428, buf429, buf430, 128, 32, stream=stream0)
        buf432 = buf422; del buf422  # reuse
        buf433 = buf432; del buf432  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109, hidden_states_110, hidden_states_111], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_61:1475
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_native_group_norm_silu_61.run(buf433, arg132_1, buf429, buf430, arg133_1, arg134_1, 536870912, stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        buf434 = buf421; del buf421  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_62:1476
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_62.run(arg135_1, buf434, 16384, 9, stream=stream0)
        del arg135_1
        # Unsorted Source Nodes: [], Original ATen: []
        buf435 = extern_kernels.convolution(buf433, buf434, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf435, (4, 128, 1024, 1024), (134217728, 1, 131072, 128), 'unknown_op')
        del buf434
        buf436 = reinterpret_tensor(buf425, (4, 32, 1, 1, 32, 64), (65536, 2048, 262144, 262144, 64, 1), 0); del buf425  # reuse
        buf437 = reinterpret_tensor(buf424, (4, 32, 1, 1, 32, 64), (65536, 2048, 262144, 262144, 64, 1), 0); del buf424  # reuse
        buf438 = reinterpret_tensor(buf423, (4, 32, 1, 1, 32, 64), (65536, 2048, 262144, 262144, 64, 1), 0); del buf423  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113, add_14, output_tensor_13, sample_2], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.native_group_norm]
        # [Provenance debug handles] triton_red_fused_add_convolution_div_native_group_norm_silu_69:1477
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_div_native_group_norm_silu_69.run(buf412, buf435, arg136_1, buf436, buf437, buf438, 262144, 2048, stream=stream0)
        buf439 = reinterpret_tensor(buf428, (4, 32, 1, 1, 32), (1024, 32, 4096, 4096, 1), 0); del buf428  # reuse
        buf440 = reinterpret_tensor(buf427, (4, 32, 1, 1, 32), (1024, 32, 4096, 4096, 1), 0); del buf427  # reuse
        buf441 = reinterpret_tensor(buf426, (4, 32, 1, 1, 32), (1024, 32, 4096, 4096, 1), 0); del buf426  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113, add_14, output_tensor_13, sample_2], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_add_convolution_div_native_group_norm_silu_70:1478
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_div_native_group_norm_silu_70.run(buf436, buf437, buf438, buf439, buf440, buf441, 4096, 64, stream=stream0)
        del buf436
        del buf437
        del buf438
        buf442 = buf430; del buf430  # reuse
        buf443 = buf429; del buf429  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113, add_14, output_tensor_13, sample_2], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.native_group_norm]
        # [Provenance debug handles] triton_per_fused_native_group_norm_67:1479
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_67.run(buf439, buf440, buf441, buf442, buf443, 128, 32, stream=stream0)
        del buf439
        del buf440
        del buf441
        buf445 = buf412; del buf412  # reuse
        buf446 = buf433; del buf433  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113, add_14, output_tensor_13, sample_2, sample_3], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.native_group_norm]
        # [Provenance debug handles] triton_poi_fused_add_convolution_div_native_group_norm_silu_71:1480
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_native_group_norm_silu_71.run(buf445, buf435, arg136_1, buf442, buf443, arg137_1, arg138_1, buf446, 512, 1048576, stream=stream0)
        del arg136_1
        del arg137_1
        del arg138_1
        del buf435
        del buf442
        del buf443
        del buf445
        buf447 = empty_strided_cuda((3, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [sample_3, sample_4], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_72:1481
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_72.run(arg139_1, buf447, 384, 9, stream=stream0)
        del arg139_1
        buf448 = empty_strided_cuda((4, 3, 1024, 1024), (3145728, 1, 3072, 3), torch.float32)
        # Topologically Sorted Source Nodes: [sample_3, sample_4], Original ATen: [aten.silu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_tem_fused_convolution_silu_73.run(buf446, buf447, buf448, 32768, 1, 1, stream=stream0)
        del buf446
        del buf447
        buf449 = empty_strided_cuda((4, 3, 1024, 1024), (3145728, 1048576, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sample_3, sample_4], Original ATen: [aten.silu, aten.convolution]
        # [Provenance debug handles] triton_poi_fused_convolution_silu_74:1483
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_silu_74.run(buf448, arg140_1, buf449, 12, 1048576, stream=stream0)
        del arg140_1
        del buf448
    return (buf449, )


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1 = args
        args.clear()
        partition0_args = [arg2_1, arg1_1, arg0_1, arg3_1, arg4_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg5_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg92_1, arg91_1, arg87_1, arg88_1, arg89_1, arg90_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg120_1, arg119_1, arg115_1, arg116_1, arg117_1, arg118_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1]
        del arg2_1, arg1_1, arg0_1, arg3_1, arg4_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg5_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg92_1, arg91_1, arg87_1, arg88_1, arg89_1, arg90_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg120_1, arg119_1, arg115_1, arg116_1, arg117_1, arg118_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1
        (buf449,) = self.partitions[0](partition0_args)
        del partition0_args
        return (buf449, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((4, 4, 128, 128), (65536, 16384, 128, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((512, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg7_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg8_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    arg9_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg10_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg11_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg12_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    arg13_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg14_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg15_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg16_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg17_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg18_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg19_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg20_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg21_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg22_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float16)
    arg23_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg24_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg25_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg26_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    arg27_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg28_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg29_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg30_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float16)
    arg31_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg32_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((3, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
