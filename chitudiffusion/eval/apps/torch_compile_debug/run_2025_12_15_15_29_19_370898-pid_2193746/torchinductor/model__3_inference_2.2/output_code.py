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


# kernel path: /tmp/torchinductor_wucz/wi/cwin3k57tjltqxotpaedxegp4ou7ayy6g7ndv2r6l7bwxg6ljcm7.py
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1048576, 'x': 524288}},
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


# kernel path: /tmp/torchinductor_wucz/5a/c5aqp3ry5bhlqwcmzbwuy35u2kc55i3od4vwhnims3lcr2vdvdgg.py
# Topologically Sorted Source Nodes: [z], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   z => convolution
# Graph fragment:
#   %buf1 : Tensor "f16[4, 4, 128, 128][65536, 1, 512, 4]cuda:0" = PlaceHolder[target=buf1]
#   %arg1_1 : Tensor "f16[4][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1572872}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/vx/cvxpkdxl4iyyphstzcqyebualgpnwdvskk4jzxj7t6h63qecorat.py
# Topologically Sorted Source Nodes: [z, sample], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sample => convolution_1
#   z => convolution
# Graph fragment:
#   %arg3_1 : Tensor "f16[512, 4, 3, 3][36, 9, 3, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convolution : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf3
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 73728, 'x': 36864}},
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


# kernel path: /tmp/torchinductor_wucz/6n/c6netc2xsqrtlqzzkldes2ospmhtvaeigblhpeu247jqpwaibefi.py
# Topologically Sorted Source Nodes: [z, sample, hidden_states], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states => convert_element_type, var_mean, view
#   sample => convolution_1
#   z => convolution
# Graph fragment:
#   %buf4 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf4]
#   %arg4_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %convolution : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %view : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf5,%buf6,%buf7
triton_per_fused_convolution_native_group_norm_3 = async_compile.triton('triton_per_fused_convolution_native_group_norm_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 5, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/gy/cgyv2xc2kq3anu7i7vhgcalnfm22pcfdxfgmu6nr3ohar64qmkj4.py
# Topologically Sorted Source Nodes: [z, sample, hidden_states], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states => convert_element_type, var_mean, view
#   sample => convolution_1
#   z => convolution
# Graph fragment:
#   %buf5 : Tensor "f32[4, 32, 1, 1, 4, 256][32768, 4, 131072, 131072, 1, 128]cuda:0" = PlaceHolder[target=buf5]
#   %buf6 : Tensor "f32[4, 32, 1, 1, 4, 256][32768, 4, 131072, 131072, 1, 128]cuda:0" = PlaceHolder[target=buf6]
#   %buf7 : Tensor "f32[4, 32, 1, 1, 4, 256][32768, 4, 131072, 131072, 1, 128]cuda:0" = PlaceHolder[target=buf7]
#   %convolution : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %view : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf8,%buf9,%buf10
triton_per_fused_convolution_native_group_norm_4 = async_compile.triton('triton_per_fused_convolution_native_group_norm_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1597440, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/vl/cvlo4bbz7vgiqgnjcgn2q6koom6ctu5t76p443xhzcw76ul7hwrs.py
# Topologically Sorted Source Nodes: [z, sample, hidden_states], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states => convert_element_type, var_mean, view
#   sample => convolution_1
#   z => convolution
# Graph fragment:
#   %buf8 : Tensor "f32[4, 32, 1, 1, 4, 2][256, 4, 1024, 1024, 1, 128]cuda:0" = PlaceHolder[target=buf8]
#   %buf9 : Tensor "f32[4, 32, 1, 1, 4, 2][256, 4, 1024, 1024, 1, 128]cuda:0" = PlaceHolder[target=buf9]
#   %buf10 : Tensor "f32[4, 32, 1, 1, 4, 2][256, 4, 1024, 1024, 1, 128]cuda:0" = PlaceHolder[target=buf10]
#   %convolution : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %view : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf11,%buf12,%buf13
triton_per_fused_convolution_native_group_norm_5 = async_compile.triton('triton_per_fused_convolution_native_group_norm_5', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24576, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/qv/cqvnv5fwgxcapedro6ivd44fequkhxrdgln7nmrhzr6h6hrx46xf.py
# Topologically Sorted Source Nodes: [z, sample, hidden_states], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states => convert_element_type, var_mean, view
#   sample => convolution_1
#   z => convolution
# Graph fragment:
#   %buf11 : Tensor "f32[4, 32, 1, 1, 4][128, 4, 512, 512, 1]cuda:0" = PlaceHolder[target=buf11]
#   %buf12 : Tensor "f32[4, 32, 1, 1, 4][128, 4, 512, 512, 1]cuda:0" = PlaceHolder[target=buf12]
#   %buf13 : Tensor "f32[4, 32, 1, 1, 4][128, 4, 512, 512, 1]cuda:0" = PlaceHolder[target=buf13]
#   %convolution : Tensor "f16[4, 4, 128, 128][65536, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_1 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %arg3_1, %arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convert_element_type : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convolution_1, torch.float32), kwargs = {})
#   %view : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_1,%buf15
triton_per_fused_convolution_native_group_norm_6 = async_compile.triton('triton_per_fused_convolution_native_group_norm_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048, 'r0_': 3072}}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/cl/cclxdi6kponorarpttyxojgapijp5t4vdng3nzaygznmyfjdgrkm.py
# Topologically Sorted Source Nodes: [z, sample, hidden_states, hidden_states_1], Original ATen: [aten.convolution, aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states => add, add_1, convert_element_type, mul, mul_1, rsqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, var_mean, view, view_1
#   hidden_states_1 => convert_element_type_5, mul_2, sigmoid
#   sample => convolution_1
#   z => convolution
# Graph fragment:
#   %buf4 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf4]
#   %arg4_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %getitem_1 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf15 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf15]
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
triton_poi_fused_convolution_native_group_norm_silu_7 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_7', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 268435456, 'x': 201329664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/65/c656lf3fsyqgighxybgvpjg4pzebkb3726b2qwyjbwtnyon2su62.py
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
#   return %buf19
triton_poi_fused_convolution_silu_8 = async_compile.triton('triton_poi_fused_convolution_silu_8', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 9437184, 'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/6i/c6inxolsqirbmmsk4q7lp4whdgzg7np2dsnohq4uxvl7gpsjt5jg.py
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
#   %buf4 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf4]
#   %arg4_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %buf36 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf36]
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
#   return %buf37,%buf38,%buf39
triton_per_fused_add_convolution_div_native_group_norm_silu_view_9 = async_compile.triton('triton_per_fused_add_convolution_div_native_group_norm_silu_view_9', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_div_native_group_norm_silu_view_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 5, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_add_convolution_div_native_group_norm_silu_view_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/bg/cbg5x25sa375vmhvcsbfvv4t6bfkp2flp5ip7dl5ivzqdizzxznp.py
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
#   %buf4 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf4]
#   %arg4_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %buf36 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf36]
#   %arg13_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %getitem_5 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_5]
#   %buf47 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf47]
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
triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_transpose_view_10 = async_compile.triton('triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_transpose_view_10', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_transpose_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 268435456, 'x': 536875008}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_transpose_view_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/td/ctdhwhclznqedxixsfrdff3azjddjb3ulwlhnoqi6x3lr5simncx.py
# Topologically Sorted Source Nodes: [query, view_1, query_1, key, view_2, key_1, value, view_3, value_1, hidden_states_9], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   hidden_states_9 => _scaled_dot_product_efficient_attention
#   key => add_8, view_10
#   key_1 => permute_7
#   query => add_7, view_8
#   query_1 => permute_6
#   value => add_9, view_12
#   value_1 => permute_8
#   view_1 => view_13
#   view_2 => view_14
#   view_3 => view_15
# Graph fragment:
#   %mm : Tensor "f16[65536, 512][512, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg17_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg17_1]
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
#   return %buf56
triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_transpose_view_11 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_transpose_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_transpose_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 201327616}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_transpose_view_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/6n/c6nmurvx553ft7jsd5bs4q7l6uxelvbwjf2w2bb3iv4bmg54b6ev.py
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
#   %mm_default : Tensor "f16[65536, 512][512, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %arg23_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg23_1]
#   %buf4 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf4]
#   %arg4_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %buf36 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf36]
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
triton_poi_fused_add_addmm_convolution_div_silu_transpose_view_12 = async_compile.triton('triton_poi_fused_add_addmm_convolution_div_silu_transpose_view_12', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_convolution_div_silu_transpose_view_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 335547392}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_convolution_div_silu_transpose_view_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/d2/cd25l4ryozbjzpo5osnufc2zrx32zcq3dhoks7v63mqfckndvyg6.py
# Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_17 => clone_5, convert_element_type_25, var_mean_3, view_20
# Graph fragment:
#   %div_1 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=div_1]
#   %clone_5 : Tensor "f16[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_1,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_25 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_5, torch.float32), kwargs = {})
#   %view_20 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_25, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_20, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf66,%buf67,%buf68
triton_per_fused_clone_native_group_norm_13 = async_compile.triton('triton_per_fused_clone_native_group_norm_13', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_native_group_norm_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 5, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_clone_native_group_norm_13(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/6h/c6hbbj5kyw5qnvzaibuir636x45xuvtxhcbgtmimrytbi35y2rdn.py
# Topologically Sorted Source Nodes: [hidden_states_17, hidden_states_18], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_17 => add_11, add_12, clone_5, convert_element_type_25, mul_8, mul_9, rsqrt_3, sub_3, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, var_mean_3, view_20, view_21
#   hidden_states_18 => convert_element_type_30, mul_10, sigmoid_2
# Graph fragment:
#   %div_1 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=div_1]
#   %getitem_11 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_11]
#   %buf76 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf76]
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
triton_poi_fused_clone_native_group_norm_silu_14 = async_compile.triton('triton_poi_fused_clone_native_group_norm_silu_14', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_native_group_norm_silu_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 268435456, 'x': 201328640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_native_group_norm_silu_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/ym/cymfpnv7iv6nq54u5l643bxuvqmt76o4fizmjaq3w3t5hrsi27jz.py
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
#   %buf97 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf97]
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
#   return %buf98,%buf99,%buf100
triton_per_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_15 = async_compile.triton('triton_per_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_15', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 5, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_15(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/r2/cr2luqkaqyfzzeo2odkq3d7vwvv3k2m27on6gczo66bncgozsuzf.py
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
#   %buf97 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf97]
#   %arg31_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg31_1]
#   %getitem_15 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_15]
#   %buf108 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf108]
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
triton_poi_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_16 = async_compile.triton('triton_poi_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_16', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 268435456, 'x': 402658304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/5w/c5wrprwb6xt7nfwhgheyzg3pgp66ej6paxcbfmuh7vrum7yk3shi.py
# Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_25 => mul_16, sigmoid_4
#   hidden_states_26 => convolution_6
# Graph fragment:
#   %arg33_1 : Tensor "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = PlaceHolder[target=arg33_1]
#   %sigmoid_4 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_16 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_4), kwargs = {})
#   %convolution_6 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_16, %arg33_1, %arg34_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf112
triton_poi_fused_convolution_silu_17 = async_compile.triton('triton_poi_fused_convolution_silu_17', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 18874368, 'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_17(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/3f/c3f3dgxtmqr5oask5b4onzxzv2rhvo4vc6alnth2k5esj7qjvigi.py
# Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26, hidden_states_27], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_25 => mul_16, sigmoid_4
#   hidden_states_26 => convolution_6
#   hidden_states_27 => var_mean_6, view_26
# Graph fragment:
#   %buf113 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf113]
#   %arg34_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg34_1]
#   %sigmoid_4 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_16 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_4), kwargs = {})
#   %convolution_6 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_16, %arg33_1, %arg34_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_26 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_6, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_26, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf114,%buf115,%buf116
triton_per_fused_convolution_native_group_norm_silu_18 = async_compile.triton('triton_per_fused_convolution_native_group_norm_silu_18', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_silu_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 5, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_silu_18(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/jb/cjbh4fsom3aoi32sekjii4vaptnri3tzmnscb45wn6szjcsux5ko.py
# Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26, hidden_states_27, hidden_states_28], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_25 => mul_16, sigmoid_4
#   hidden_states_26 => convolution_6
#   hidden_states_27 => add_18, add_19, mul_17, mul_18, rsqrt_6, sub_6, unsqueeze_34, unsqueeze_35, unsqueeze_36, unsqueeze_37, unsqueeze_38, unsqueeze_39, var_mean_6, view_26, view_27
#   hidden_states_28 => mul_19, sigmoid_5
# Graph fragment:
#   %buf113 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf113]
#   %arg34_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg34_1]
#   %getitem_17 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_17]
#   %buf124 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf124]
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
triton_poi_fused_convolution_native_group_norm_silu_19 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_19', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 268435456, 'x': 402659328}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/y7/cy7azohxlwmlgajuvfeo3gymtrvow4daqieu6twkiumfhqpodbnb.py
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
#   %buf97 : Tensor "f16[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf97]
#   %arg31_1 : Tensor "f16[512][1]cuda:0" = PlaceHolder[target=arg31_1]
#   %buf129 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf129]
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
triton_poi_fused__to_copy_add_convolution_div_silu_20 = async_compile.triton('triton_poi_fused__to_copy_add_convolution_div_silu_20', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_convolution_div_silu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 536873984}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_convolution_div_silu_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/hh/chhpluilo4ij3oszsm4afo3kl5tactg5woo6lyn7bfijel33l2vs.py
# Topologically Sorted Source Nodes: [hidden_states_31], Original ATen: [aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_31 => clone_9, var_mean_7, view_28
# Graph fragment:
#   %div_3 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=div_3]
#   %clone_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_3,), kwargs = {memory_format: torch.contiguous_format})
#   %view_28 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_9, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_28, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf131,%buf132,%buf133
triton_per_fused_clone_native_group_norm_21 = async_compile.triton('triton_per_fused_clone_native_group_norm_21', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_native_group_norm_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 5, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_clone_native_group_norm_21(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/ha/chaeplsr2ay72mitfxtf7vyff7edselcyhn5zabuynvlrt4gxlu6.py
# Topologically Sorted Source Nodes: [hidden_states_31, hidden_states_32], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_31 => add_21, add_22, clone_9, mul_20, mul_21, rsqrt_7, sub_7, unsqueeze_40, unsqueeze_41, unsqueeze_42, unsqueeze_43, unsqueeze_44, unsqueeze_45, var_mean_7, view_28, view_29
#   hidden_states_32 => mul_22, sigmoid_6
# Graph fragment:
#   %div_3 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=div_3]
#   %getitem_19 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_19]
#   %buf141 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf141]
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
triton_poi_fused_clone_native_group_norm_silu_22 = async_compile.triton('triton_poi_fused_clone_native_group_norm_silu_22', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_native_group_norm_silu_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 268435456, 'x': 402657280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_native_group_norm_silu_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/sl/cslul4vu2urf4lrvxwpvevxfrlpqyum4y7r4xic5vrb6engull6r.py
# Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_38], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_4 => add_25
#   hidden_states_35 => mul_25, sigmoid_7
#   hidden_states_37 => convolution_9
#   hidden_states_38 => clone_11, var_mean_9, view_32
#   output_tensor_3 => div_4
# Graph fragment:
#   %div_3 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=div_3]
#   %buf162 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf162]
#   %arg46_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg46_1]
#   %sigmoid_7 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_24,), kwargs = {})
#   %mul_25 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %sigmoid_7), kwargs = {})
#   %convolution_9 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_25, %arg45_1, %arg46_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_25 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %convolution_9), kwargs = {})
#   %div_4 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_25, 1.0), kwargs = {})
#   %clone_11 : Tensor "f32[4, 512, 128, 128][8388608, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_32 : Tensor "f32[4, 32, 16, 16384][8388608, 262144, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_11, [4, 32, 16, 16384]), kwargs = {})
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_32, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf163,%buf164,%buf165
triton_per_fused_add_clone_convolution_div_native_group_norm_silu_23 = async_compile.triton('triton_per_fused_add_clone_convolution_div_native_group_norm_silu_23', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_convolution_div_native_group_norm_silu_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 5, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_add_clone_convolution_div_native_group_norm_silu_23(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/wd/cwdmpn7xrdoqypx7gvub36di6zkje525g74dzwtdtnkpgyqlikpf.py
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
#   %buf162 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf162]
#   %arg46_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg46_1]
#   %getitem_23 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_23]
#   %buf173 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf173]
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
triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_24 = async_compile.triton('triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_24', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 268435456, 'x': 536877056}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/dt/cdtpav5neeienbpseysfk53uwec6izohp7dlqoca5vrff56vkzaq.py
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
#   %buf162 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf162]
#   %arg46_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg46_1]
#   %buf194 : Tensor "f32[4, 512, 128, 128][8388608, 1, 65536, 512]cuda:0" = PlaceHolder[target=buf194]
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
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_25 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_25', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/ye/cyex37yhvnzhphk3xrwp4rgalpbwwlz36l7j6td2dvxxasjvxznp.py
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
#   %buf197 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf197]
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
#   return %buf198,%buf199,%buf200
triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_26 = async_compile.triton('triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_26', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1572864, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_26(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/sv/csv5da6734dmak4we2yil472czwa7taaypzl6h4eelthnl47xvnx.py
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
#   %buf198 : Tensor "f32[4, 32, 1, 1, 8, 64][16384, 8, 65536, 65536, 1, 256]cuda:0" = PlaceHolder[target=buf198]
#   %buf199 : Tensor "f32[4, 32, 1, 1, 8, 64][16384, 8, 65536, 65536, 1, 256]cuda:0" = PlaceHolder[target=buf199]
#   %buf200 : Tensor "f32[4, 32, 1, 1, 8, 64][16384, 8, 65536, 65536, 1, 256]cuda:0" = PlaceHolder[target=buf200]
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
#   return %buf201,%buf202,%buf203
triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27 = async_compile.triton('triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 811008, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/ct/cct5r6dandexxmdbsgtepifomgh5fpy4omqyhqnbm33f62jq3kwy.py
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
#   %buf201 : Tensor "f32[4, 32, 1, 1, 8][256, 8, 1024, 1024, 1]cuda:0" = PlaceHolder[target=buf201]
#   %buf202 : Tensor "f32[4, 32, 1, 1, 8][256, 8, 1024, 1024, 1]cuda:0" = PlaceHolder[target=buf202]
#   %buf203 : Tensor "f32[4, 32, 1, 1, 8][256, 8, 1024, 1024, 1]cuda:0" = PlaceHolder[target=buf203]
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
#   return %getitem_27,%buf205
triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28 = async_compile.triton('triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048, 'r0_': 12288}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/3x/c3x4hvtonluqlwod2gghyvwptvi7xvwklu62izha66b4vrvkyjoo.py
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
#   %buf197 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf197]
#   %arg56_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg56_1]
#   %getitem_27 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_27]
#   %buf205 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf205]
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
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1073741824, 'x': 1610618880}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/fu/cfuxhoyx4cc7yqgj4a2tcqv6diezln2z5dg5ilcrqjf3r6ujunms.py
# Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49, hidden_states_50, hidden_states_51], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_48 => mul_38, sigmoid_10
#   hidden_states_49 => convolution_13
#   hidden_states_50 => add_37, add_38, mul_39, mul_40, rsqrt_12, sub_12, unsqueeze_71, unsqueeze_72, unsqueeze_73, unsqueeze_74, unsqueeze_75, unsqueeze_76, var_mean_12, view_38, view_39
#   hidden_states_51 => mul_41, sigmoid_11
# Graph fragment:
#   %buf210 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf210]
#   %arg60_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg60_1]
#   %getitem_29 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_29]
#   %buf218 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf218]
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
triton_poi_fused_convolution_native_group_norm_silu_30 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_30', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1073741824, 'x': 1610618880}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/cc/cccxnpwqmqdmwwkynnguu7pem2zlvnom55yu36brvfkqvyn3zukh.py
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
#   %buf197 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf197]
#   %arg56_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg56_1]
#   %buf223 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf223]
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
#   return %buf224,%buf225,%buf226
triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_31 = async_compile.triton('triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_31', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1572864, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/wi/cwiokhtkw4estaonpllgtg4muqhzal33g7bru6a4dqnoncb4u73w.py
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
#   %buf197 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf197]
#   %arg56_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg56_1]
#   %buf223 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf223]
#   %arg64_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg64_1]
#   %getitem_31 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_31]
#   %buf231 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf231]
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
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_32 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_32', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1073741824, 'x': 2147491840}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/ht/chtm2yqvehcc5v5c5ytkttlsmlnpmgjaqwfrexk4iw2wucbzf3lt.py
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
#   %buf197 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf197]
#   %arg56_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg56_1]
#   %buf223 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf223]
#   %arg64_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg64_1]
#   %buf249 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf249]
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
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_33 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_33', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2684360704}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/ff/cffxi4kkh55bxzpj52ltr4bytt6wtfhtpzhgvtglvtizfkxce7jn.py
# Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_61 => clone_18, var_mean_15, view_44
# Graph fragment:
#   %div_7 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=div_7]
#   %clone_18 : Tensor "f32[4, 512, 256, 256][33554432, 65536, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_7,), kwargs = {memory_format: torch.contiguous_format})
#   %view_44 : Tensor "f32[4, 32, 16, 65536][33554432, 1048576, 65536, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_18, [4, 32, 16, 65536]), kwargs = {})
#   %var_mean_15 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_44, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf251,%buf252,%buf253
triton_red_fused_clone_native_group_norm_34 = async_compile.triton('triton_red_fused_clone_native_group_norm_34', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clone_native_group_norm_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1572864, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_clone_native_group_norm_34(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/gc/cgc6ctgjaegl47nsyi57tsrut4sat2bjrmo5gdz5426tucjyj4uv.py
# Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_62], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_61 => add_45, add_46, clone_18, mul_48, mul_49, rsqrt_15, sub_15, unsqueeze_89, unsqueeze_90, unsqueeze_91, unsqueeze_92, unsqueeze_93, unsqueeze_94, var_mean_15, view_44, view_45
#   hidden_states_62 => mul_50, sigmoid_14
# Graph fragment:
#   %div_7 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=div_7]
#   %getitem_35 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_35]
#   %buf258 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf258]
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
triton_poi_fused_clone_native_group_norm_silu_35 = async_compile.triton('triton_poi_fused_clone_native_group_norm_silu_35', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_native_group_norm_silu_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1073741824, 'x': 1610616832}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_native_group_norm_silu_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/k5/ck52wn3hadcyndr24wvg3vqhfnaglwzibqp4kvkc3fgnh2ngbdxi.py
# Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
# Source node to ATen node mapping:
#   add_8 => add_49
#   hidden_states_65 => mul_53, sigmoid_15
#   hidden_states_67 => convolution_18
#   hidden_states_68 => _unsafe_index_1, add_50, add_51, add_52, add_53, clone_20, convert_element_type_42, convert_element_type_43, convert_element_type_44, convert_element_type_45, iota_2, iota_3, mul_54, mul_55, mul_56, mul_57, unsqueeze_101
#   output_tensor_7 => div_8
# Graph fragment:
#   %div_7 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=div_7]
#   %buf276 : Tensor "f32[4, 512, 256, 256][33554432, 1, 131072, 512]cuda:0" = PlaceHolder[target=buf276]
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
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_36 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_36', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_36(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/st/cstaeqo37otkgv3xpf7wtvaltsmkt5e4w66v62iz3w7siobhetat.py
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
#   %buf279 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0" = PlaceHolder[target=buf279]
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
#   return %buf280,%buf281,%buf282
triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_37 = async_compile.triton('triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_37', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6291456, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_37(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/ly/clywd3wg4pzl35fyibkynbs5b4hkq37vy34jblnf4wbwtzqwurme.py
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
#   %buf280 : Tensor "f32[4, 32, 1, 1, 32, 64][65536, 1, 262144, 262144, 2048, 32]cuda:0" = PlaceHolder[target=buf280]
#   %buf281 : Tensor "f32[4, 32, 1, 1, 32, 64][65536, 1, 262144, 262144, 2048, 32]cuda:0" = PlaceHolder[target=buf281]
#   %buf282 : Tensor "f32[4, 32, 1, 1, 32, 64][65536, 1, 262144, 262144, 2048, 32]cuda:0" = PlaceHolder[target=buf282]
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
#   return %buf283,%buf284,%buf285
triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38 = async_compile.triton('triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3244032, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/j6/cj6pfdfgsqwvs5ypequyavzhhees7w7c25b7nj5efujtg4guucae.py
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
#   %buf283 : Tensor "f32[4, 32, 1, 1, 32][1024, 1, 4096, 4096, 32]cuda:0" = PlaceHolder[target=buf283]
#   %buf284 : Tensor "f32[4, 32, 1, 1, 32][1024, 1, 4096, 4096, 32]cuda:0" = PlaceHolder[target=buf284]
#   %buf285 : Tensor "f32[4, 32, 1, 1, 32][1024, 1, 4096, 4096, 32]cuda:0" = PlaceHolder[target=buf285]
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
#   return %getitem_39,%buf287
triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39 = async_compile.triton('triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 51200, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/bx/cbxl3sgkbhuydddvlndrlbazyvoagna7i2v4mwhz63xgc65exfpw.py
# Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, hidden_states_70, hidden_states_71, contiguous], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_8 => add_49
#   contiguous => clone_23
#   hidden_states_65 => mul_53, sigmoid_15
#   hidden_states_67 => convolution_18
#   hidden_states_68 => _unsafe_index_1, add_50, add_51, add_52, add_53, clone_20, convert_element_type_42, convert_element_type_43, convert_element_type_44, convert_element_type_45, iota_2, iota_3, mul_54, mul_55, mul_56, mul_57, unsqueeze_101
#   hidden_states_69 => convolution_19
#   hidden_states_70 => add_54, add_55, clone_21, mul_58, mul_59, rsqrt_17, sub_17, unsqueeze_102, unsqueeze_103, unsqueeze_104, unsqueeze_105, unsqueeze_106, unsqueeze_107, var_mean_17, view_48, view_49
#   hidden_states_71 => mul_60, sigmoid_16
#   output_tensor_7 => div_8
# Graph fragment:
#   %buf279 : Tensor "f32[4, 512, 512, 512][134217728, 1, 262144, 512]cuda:0" = PlaceHolder[target=buf279]
#   %arg82_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg82_1]
#   %getitem_39 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_39]
#   %buf287 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf287]
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
#   return %add_55,%clone_23,%mul_60
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 4294967296, 'x': 12884908032}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/gg/cggc2tpsptprq5fojiu2yph5av4dt4hmop636chik7usbpw5a333.py
# Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_71 => mul_60, sigmoid_16
#   hidden_states_72 => convolution_20
# Graph fragment:
#   %arg85_1 : Tensor "f32[256, 512, 3, 3][4608, 9, 3, 1]cuda:0" = PlaceHolder[target=arg85_1]
#   %sigmoid_16 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_55,), kwargs = {})
#   %mul_60 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, %sigmoid_16), kwargs = {})
#   %convolution_20 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_60, %arg85_1, %arg86_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf291
triton_poi_fused_convolution_silu_41 = async_compile.triton('triton_poi_fused_convolution_silu_41', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 9437184, 'x': 4718592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_41(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/ga/cgalu4bouutc4cf5epj3qqol2zyjrktnphtylfzkyc6ndyudcqa5.py
# Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_71 => mul_60, sigmoid_16
#   hidden_states_72 => convolution_20
#   hidden_states_73 => var_mean_18, view_50
# Graph fragment:
#   %buf292 : Tensor "f32[4, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=buf292]
#   %arg86_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg86_1]
#   %sigmoid_16 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_55,), kwargs = {})
#   %mul_60 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, %sigmoid_16), kwargs = {})
#   %convolution_20 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_60, %arg85_1, %arg86_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_50 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_20, [4, 32, 8, 262144]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_50, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf293,%buf294,%buf295
triton_red_fused_convolution_native_group_norm_silu_42 = async_compile.triton('triton_red_fused_convolution_native_group_norm_silu_42', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_silu_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_silu_42(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/3y/c3ywoa6qul6k7dveskuphbqfk6wu6dfjbbwm3ub6g47maudi5vtg.py
# Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_71 => mul_60, sigmoid_16
#   hidden_states_72 => convolution_20
#   hidden_states_73 => var_mean_18, view_50
# Graph fragment:
#   %buf293 : Tensor "f32[4, 32, 1, 1, 16, 64][32768, 1, 131072, 131072, 2048, 32]cuda:0" = PlaceHolder[target=buf293]
#   %buf294 : Tensor "f32[4, 32, 1, 1, 16, 64][32768, 1, 131072, 131072, 2048, 32]cuda:0" = PlaceHolder[target=buf294]
#   %buf295 : Tensor "f32[4, 32, 1, 1, 16, 64][32768, 1, 131072, 131072, 2048, 32]cuda:0" = PlaceHolder[target=buf295]
#   %sigmoid_16 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_55,), kwargs = {})
#   %mul_60 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, %sigmoid_16), kwargs = {})
#   %convolution_20 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_60, %arg85_1, %arg86_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_50 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_20, [4, 32, 8, 262144]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_50, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf296,%buf297,%buf298
triton_per_fused_convolution_native_group_norm_silu_43 = async_compile.triton('triton_per_fused_convolution_native_group_norm_silu_43', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_silu_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1622016, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_silu_43(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/6i/c6iselfisfb45scbktdzgdza747ykjg42rrbtwwn365obs5kcqiz.py
# Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_71 => mul_60, sigmoid_16
#   hidden_states_72 => convolution_20
#   hidden_states_73 => var_mean_18, view_50
# Graph fragment:
#   %buf296 : Tensor "f32[4, 32, 1, 1, 16][512, 1, 2048, 2048, 32]cuda:0" = PlaceHolder[target=buf296]
#   %buf297 : Tensor "f32[4, 32, 1, 1, 16][512, 1, 2048, 2048, 32]cuda:0" = PlaceHolder[target=buf297]
#   %buf298 : Tensor "f32[4, 32, 1, 1, 16][512, 1, 2048, 2048, 32]cuda:0" = PlaceHolder[target=buf298]
#   %sigmoid_16 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_55,), kwargs = {})
#   %mul_60 : Tensor "f32[4, 512, 512, 512][134217728, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, %sigmoid_16), kwargs = {})
#   %convolution_20 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_60, %arg85_1, %arg86_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_50 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_20, [4, 32, 8, 262144]), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_50, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_41,%buf300
triton_per_fused_convolution_native_group_norm_silu_44 = async_compile.triton('triton_per_fused_convolution_native_group_norm_silu_44', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_silu_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 26624, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_silu_44(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/rn/crnr4gxc3ssd4zcda2iix7oosvyjbpzfj2chk6uylttsmhd6jb4w.py
# Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73, hidden_states_74], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_71 => mul_60, sigmoid_16
#   hidden_states_72 => convolution_20
#   hidden_states_73 => add_56, add_57, mul_61, mul_62, rsqrt_18, sub_18, unsqueeze_108, unsqueeze_109, unsqueeze_110, unsqueeze_111, unsqueeze_112, unsqueeze_113, var_mean_18, view_50, view_51
#   hidden_states_74 => mul_63, sigmoid_17
# Graph fragment:
#   %buf292 : Tensor "f32[4, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=buf292]
#   %arg86_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg86_1]
#   %getitem_41 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_41]
#   %buf300 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf300]
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
triton_poi_fused_convolution_native_group_norm_silu_45 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_45', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3221228544}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/k7/ck76ycgt4vdtwyxs6hkpry3sz2q5brk5usasbvmvxptkkvs5ek7s.py
# Topologically Sorted Source Nodes: [hidden_states_74, hidden_states_76], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_74 => mul_63, sigmoid_17
#   hidden_states_76 => convolution_21
# Graph fragment:
#   %arg89_1 : Tensor "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg89_1]
#   %sigmoid_17 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_63 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_17), kwargs = {})
#   %convolution_21 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_63, %arg89_1, %arg90_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf306
triton_poi_fused_convolution_silu_46 = async_compile.triton('triton_poi_fused_convolution_silu_46', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 4718592, 'x': 2359296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_46(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/2s/c2semrp4gqxlbpcxn4tr47i5auyqwafv2hitw6fzoexiw3im4ej4.py
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
#   %buf303 : Tensor "f32[4, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=buf303]
#   %arg92_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg92_1]
#   %buf307 : Tensor "f32[4, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=buf307]
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
#   return %buf308,%buf309,%buf310
triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_47 = async_compile.triton('triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_47', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 3145728, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp10_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x5 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_4 = r0_index
        tmp0 = tl.load(in_ptr0 + (8*x0 + 256*(((r0_4 + 2048*x1 + 131072*x2) % 262144)) + 67108864*x3 + ((r0_4 + 2048*x1 + 131072*x2) // 262144)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (8*x0 + ((r0_4 + 2048*x1 + 131072*x2) // 262144)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (8*x0 + 256*(((r0_4 + 2048*x1 + 131072*x2) % 262144)) + 67108864*x3 + ((r0_4 + 2048*x1 + 131072*x2) // 262144)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr3 + (8*x0 + ((r0_4 + 2048*x1 + 131072*x2) // 262144)), r0_mask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x5), tmp10, None)
    tl.store(out_ptr1 + (x5), tmp14, None)
    tl.store(out_ptr2 + (x5), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/nb/cnbgsy3cycbs7qweirevkq2vxorpfchuxgk4jkrzdm6ufhj5pnc7.py
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
#   %buf303 : Tensor "f32[4, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=buf303]
#   %arg92_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg92_1]
#   %buf307 : Tensor "f32[4, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=buf307]
#   %arg90_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg90_1]
#   %getitem_43 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_43]
#   %buf315 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf315]
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
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_48 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4294971392}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 256)
    x2 = xindex // 67108864
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (32*x2 + (x0 // 8)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (32*x2 + (x0 // 8)), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 2097152.0
    tmp13 = (tmp11 / tmp12)
    tmp14 = 1e-06
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp10 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = tmp21 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/e7/ce75p4jnhesla5bux4oztszkwcckleburuwdrsniexvdymkrzl63.py
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
#   %buf303 : Tensor "f32[4, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=buf303]
#   %arg92_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg92_1]
#   %buf307 : Tensor "f32[4, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=buf307]
#   %arg90_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg90_1]
#   %buf333 : Tensor "f32[4, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=buf333]
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
triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_49 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2147483648, 'x': 3221228544}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp12 * tmp7
    tl.store(out_ptr0 + (y0 + 262144*x2 + 67108864*y1), tmp13, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/xs/cxs7fe2wuf6hr5t2geu2hgasdtn3oxgoolhgfspyvymfzpknuzve.py
# Topologically Sorted Source Nodes: [hidden_states_84], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_84 => var_mean_21, view_56
# Graph fragment:
#   %div_10 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0" = PlaceHolder[target=div_10]
#   %view_56 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_10, [4, 32, 8, 262144]), kwargs = {})
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_56, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf335,%buf336,%buf337
triton_red_fused_native_group_norm_50 = async_compile.triton('triton_red_fused_native_group_norm_50', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 49152, 'r0_': 1073741824}}
)
@triton.jit
def triton_red_fused_native_group_norm_50(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/wh/cwhfwjuu7jfgdteu4ajpcyks6rkzmlj42mr7t32usm5wxmmkwcwg.py
# Topologically Sorted Source Nodes: [hidden_states_84], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_84 => var_mean_21, view_56
# Graph fragment:
#   %buf335 : Tensor "f32[4, 32, 1, 1, 16][512, 16, 2048, 2048, 1]cuda:0" = PlaceHolder[target=buf335]
#   %buf336 : Tensor "f32[4, 32, 1, 1, 16][512, 16, 2048, 2048, 1]cuda:0" = PlaceHolder[target=buf336]
#   %buf337 : Tensor "f32[4, 32, 1, 1, 16][512, 16, 2048, 2048, 1]cuda:0" = PlaceHolder[target=buf337]
#   %view_56 : Tensor "f32[4, 32, 8, 262144][67108864, 2097152, 262144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_10, [4, 32, 8, 262144]), kwargs = {})
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_56, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_47,%buf339
triton_per_fused_native_group_norm_51 = async_compile.triton('triton_per_fused_native_group_norm_51', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048, 'r0_': 24576}}
)
@triton.jit
def triton_per_fused_native_group_norm_51(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/qj/cqjvkzeny6b6mxvxhcpwtjjybgindja6t5qztaur2g2nly53cjx7.py
# Topologically Sorted Source Nodes: [hidden_states_84, hidden_states_85], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_84 => add_64, add_65, mul_70, mul_71, rsqrt_21, sub_21, unsqueeze_126, unsqueeze_127, unsqueeze_128, unsqueeze_129, unsqueeze_130, unsqueeze_131, var_mean_21, view_56, view_57
#   hidden_states_85 => mul_72, sigmoid_20
# Graph fragment:
#   %div_10 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0" = PlaceHolder[target=div_10]
#   %getitem_47 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_47]
#   %buf339 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf339]
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
triton_poi_fused_native_group_norm_silu_52 = async_compile.triton('triton_poi_fused_native_group_norm_silu_52', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2147485696, 'x': 1073741824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/lt/cltlbedrynpdiymrts2epowfru47fldwnq2wvao4c2pmafjvarta.py
# Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index]
# Source node to ATen node mapping:
#   add_11 => add_68
#   hidden_states_88 => mul_75, sigmoid_21
#   hidden_states_90 => convolution_26
#   hidden_states_91 => _unsafe_index_2, add_69, add_70, add_71, add_72, convert_element_type_46, convert_element_type_47, convert_element_type_48, convert_element_type_49, iota_4, iota_5, mul_76, mul_77, mul_78, mul_79, unsqueeze_138
#   output_tensor_10 => div_11
# Graph fragment:
#   %div_10 : Tensor "f32[4, 256, 512, 512][67108864, 262144, 512, 1]cuda:0" = PlaceHolder[target=div_10]
#   %buf357 : Tensor "f32[4, 256, 512, 512][67108864, 1, 131072, 256]cuda:0" = PlaceHolder[target=buf357]
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
triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_53 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_53', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_53(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/uw/cuw4wwvc3ui6rcwp2yps5hkxthhdcccosnx7zniu5nkvj443yfpp.py
# Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91, hidden_states_92], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index]
# Source node to ATen node mapping:
#   add_11 => add_68
#   hidden_states_88 => mul_75, sigmoid_21
#   hidden_states_90 => convolution_26
#   hidden_states_91 => _unsafe_index_2, add_69, add_70, add_71, add_72, convert_element_type_46, convert_element_type_47, convert_element_type_48, convert_element_type_49, iota_4, iota_5, mul_76, mul_77, mul_78, mul_79, unsqueeze_138
#   hidden_states_92 => convolution_27
#   output_tensor_10 => div_11
# Graph fragment:
#   %buf360 : Tensor "f32[4, 256, 1024, 1024][268435456, 1, 262144, 256]cuda:0" = PlaceHolder[target=buf360]
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
triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_54 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_54', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 4294968320, 'x': 8589934592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_54(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/oc/coczz4wjuyrzfaokyotwb5r2rl3ebtvvvys7vrkl757lcxhtvn6k.py
# Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_93 => var_mean_23, view_60
# Graph fragment:
#   %convolution_27 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0" = PlaceHolder[target=convolution_27]
#   %view_60 : Tensor "f32[4, 32, 8, 1048576][268435456, 8388608, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_27, [4, 32, 8, 1048576]), kwargs = {})
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_60, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf362,%buf363,%buf364
triton_red_fused_native_group_norm_55 = async_compile.triton('triton_red_fused_native_group_norm_55', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 196608, 'r0_': 4294967296}}
)
@triton.jit
def triton_red_fused_native_group_norm_55(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/ev/cevgzrjwoti7hkmj2rey4lhog4xnaewrcypjpw4746p2rns54ofb.py
# Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_93 => var_mean_23, view_60
# Graph fragment:
#   %buf362 : Tensor "f32[4, 32, 1, 1, 64][2048, 64, 8192, 8192, 1]cuda:0" = PlaceHolder[target=buf362]
#   %buf363 : Tensor "f32[4, 32, 1, 1, 64][2048, 64, 8192, 8192, 1]cuda:0" = PlaceHolder[target=buf363]
#   %buf364 : Tensor "f32[4, 32, 1, 1, 64][2048, 64, 8192, 8192, 1]cuda:0" = PlaceHolder[target=buf364]
#   %view_60 : Tensor "f32[4, 32, 8, 1048576][268435456, 8388608, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_27, [4, 32, 8, 1048576]), kwargs = {})
#   %var_mean_23 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_60, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_51,%buf366
triton_per_fused_native_group_norm_56 = async_compile.triton('triton_per_fused_native_group_norm_56', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048, 'r0_': 98304}}
)
@triton.jit
def triton_per_fused_native_group_norm_56(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/2n/c2nodr54pbw3ccyakutvfe3boib7dsiv46vf3hxhtnwsffju4oei.py
# Topologically Sorted Source Nodes: [hidden_states_93, hidden_states_94, input_tensor_1], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_93 => add_73, add_74, mul_80, mul_81, rsqrt_23, sub_23, unsqueeze_139, unsqueeze_140, unsqueeze_141, unsqueeze_142, unsqueeze_143, unsqueeze_144, var_mean_23, view_60, view_61
#   hidden_states_94 => mul_82, sigmoid_22
#   input_tensor_1 => convolution_30
# Graph fragment:
#   %convolution_27 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0" = PlaceHolder[target=convolution_27]
#   %getitem_51 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_51]
#   %buf366 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf366]
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
#   return %add_74,%mul_82,%buf381
triton_poi_fused_convolution_native_group_norm_silu_57 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_57', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 17179871232, 'x': 4294967296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/rf/crfofg2dqrrnxwyls47k4q4itoflpvcpjufkhbrbkjzwkfr45bwa.py
# Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_94 => mul_82, sigmoid_22
#   hidden_states_95 => convolution_28
# Graph fragment:
#   %arg113_1 : Tensor "f32[128, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg113_1]
#   %sigmoid_22 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_74,), kwargs = {})
#   %mul_82 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_74, %sigmoid_22), kwargs = {})
#   %convolution_28 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_82, %arg113_1, %arg114_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf370
triton_poi_fused_convolution_silu_58 = async_compile.triton('triton_poi_fused_convolution_silu_58', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_58', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2359296, 'x': 1179648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_58(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/lr/clroovjswnttwzqudw5apiftl7ui6khpcxws7ydseagltxgyq7yo.py
# Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95, hidden_states_96], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_94 => mul_82, sigmoid_22
#   hidden_states_95 => convolution_28
#   hidden_states_96 => var_mean_24, view_62
# Graph fragment:
#   %buf371 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=buf371]
#   %arg114_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg114_1]
#   %sigmoid_22 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_74,), kwargs = {})
#   %mul_82 : Tensor "f32[4, 256, 1024, 1024][268435456, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_74, %sigmoid_22), kwargs = {})
#   %convolution_28 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_82, %arg113_1, %arg114_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %view_62 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convolution_28, [4, 32, 4, 1048576]), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_62, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf372,%buf373,%buf374
triton_red_fused_convolution_native_group_norm_silu_59 = async_compile.triton('triton_red_fused_convolution_native_group_norm_silu_59', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_silu_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6291456, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_silu_59(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/kv/ckvx26y5oiv43zmavikesvidjxwvmzmrcx22n646dymltcqr5mcy.py
# Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95, hidden_states_96, hidden_states_97], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_94 => mul_82, sigmoid_22
#   hidden_states_95 => convolution_28
#   hidden_states_96 => add_75, add_76, mul_83, mul_84, rsqrt_24, sub_24, unsqueeze_145, unsqueeze_146, unsqueeze_147, unsqueeze_148, unsqueeze_149, unsqueeze_150, var_mean_24, view_62, view_63
#   hidden_states_97 => mul_85, sigmoid_23
# Graph fragment:
#   %buf371 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=buf371]
#   %arg114_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg114_1]
#   %getitem_53 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_53]
#   %buf379 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf379]
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
triton_poi_fused_convolution_native_group_norm_silu_60 = async_compile.triton('triton_poi_fused_convolution_native_group_norm_silu_60', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_native_group_norm_silu_60', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6442452480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_native_group_norm_silu_60(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/zn/cznrlvpj6kfmwlsi7724mdeikp2fmi5dzrrcbh57o7ho7x3m4b2k.py
# Topologically Sorted Source Nodes: [hidden_states_97, hidden_states_99], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   hidden_states_97 => mul_85, sigmoid_23
#   hidden_states_99 => convolution_29
# Graph fragment:
#   %arg117_1 : Tensor "f32[128, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg117_1]
#   %sigmoid_23 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_76,), kwargs = {})
#   %mul_85 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_76, %sigmoid_23), kwargs = {})
#   %convolution_29 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_85, %arg117_1, %arg118_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf385
triton_poi_fused_convolution_silu_61 = async_compile.triton('triton_poi_fused_convolution_silu_61', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 1179648, 'x': 589824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_61(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/iq/ciqv2w6jzwurts7amw4rsz2s7befvgzygmhpvmvxks2qsytgv7ho.py
# Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_100], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_12 => add_77
#   hidden_states_100 => var_mean_25, view_64
#   hidden_states_97 => mul_85, sigmoid_23
#   hidden_states_99 => convolution_29
#   input_tensor_1 => convolution_30
#   output_tensor_11 => div_12
# Graph fragment:
#   %buf382 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=buf382]
#   %arg120_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg120_1]
#   %buf386 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=buf386]
#   %arg118_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg118_1]
#   %convolution_30 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_27, %arg119_1, %arg120_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sigmoid_23 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_76,), kwargs = {})
#   %mul_85 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_76, %sigmoid_23), kwargs = {})
#   %convolution_29 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_85, %arg117_1, %arg118_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_77 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_30, %convolution_29), kwargs = {})
#   %div_12 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_77, 1.0), kwargs = {})
#   %view_64 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_12, [4, 32, 4, 1048576]), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_64, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf387,%buf388,%buf389
triton_red_fused_add_convolution_div_native_group_norm_silu_62 = async_compile.triton('triton_red_fused_add_convolution_div_native_group_norm_silu_62', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_div_native_group_norm_silu_62', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6291456, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_add_convolution_div_native_group_norm_silu_62(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp10_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x5 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_4 = r0_index
        tmp0 = tl.load(in_ptr0 + (4*x0 + 128*(((r0_4 + 2048*x1 + 131072*x2) % 1048576)) + 134217728*x3 + ((r0_4 + 2048*x1 + 131072*x2) // 1048576)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (4*x0 + ((r0_4 + 2048*x1 + 131072*x2) // 1048576)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (4*x0 + 128*(((r0_4 + 2048*x1 + 131072*x2) % 1048576)) + 134217728*x3 + ((r0_4 + 2048*x1 + 131072*x2) // 1048576)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr3 + (4*x0 + ((r0_4 + 2048*x1 + 131072*x2) // 1048576)), r0_mask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x5), tmp10, None)
    tl.store(out_ptr1 + (x5), tmp14, None)
    tl.store(out_ptr2 + (x5), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/pe/cpednabqy6jyw2mngzhskkk56oknuulvqjzsptbiiu63hmlezksn.py
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
#   %buf382 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=buf382]
#   %arg120_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg120_1]
#   %buf386 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=buf386]
#   %arg118_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg118_1]
#   %getitem_55 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_55]
#   %buf394 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf394]
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
triton_poi_fused_add_convolution_div_native_group_norm_silu_63 = async_compile.triton('triton_poi_fused_add_convolution_div_native_group_norm_silu_63', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_native_group_norm_silu_63', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8589936640}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_native_group_norm_silu_63(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 128)
    x2 = xindex // 134217728
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3), None)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (32*x2 + (x0 // 4)), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 4194304.0
    tmp13 = (tmp11 / tmp12)
    tmp14 = 1e-06
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp10 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = tmp21 * tmp22
    tl.store(in_out_ptr0 + (x3), tmp23, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/sm/csmovjy4ah6ohyoxkov7mdwhhe5x5hno7b6jyc6ecsfftv73mg47.py
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
#   %buf382 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=buf382]
#   %arg120_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg120_1]
#   %buf386 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=buf386]
#   %arg118_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg118_1]
#   %buf412 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=buf412]
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
triton_poi_fused_add_convolution_div_silu_64 = async_compile.triton('triton_poi_fused_add_convolution_div_silu_64', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4194304, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_silu_64', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 4294967296, 'x': 6442452480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_silu_64(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + 128*y3), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2 + 128*y3), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp12 * tmp7
    tl.store(out_ptr0 + (y0 + 1048576*x2 + 134217728*y1), tmp13, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/sz/cszmrlhuifukmgjenzrisq4xmelqvrikadb6bnqonlyptt3mhhd6.py
# Topologically Sorted Source Nodes: [hidden_states_107], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_107 => var_mean_27, view_68
# Graph fragment:
#   %div_13 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0" = PlaceHolder[target=div_13]
#   %view_68 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_13, [4, 32, 4, 1048576]), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_68, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf414,%buf415,%buf416
triton_red_fused_native_group_norm_65 = async_compile.triton('triton_red_fused_native_group_norm_65', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_65', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 98304, 'r0_': 2147483648}}
)
@triton.jit
def triton_red_fused_native_group_norm_65(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/sh/cshkq2n6tkxusdfgc3uuatdpqfkkoekm6qyaqrs3g2xrplgdcqex.py
# Topologically Sorted Source Nodes: [hidden_states_107], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   hidden_states_107 => var_mean_27, view_68
# Graph fragment:
#   %buf414 : Tensor "f32[4, 32, 1, 1, 32][1024, 32, 4096, 4096, 1]cuda:0" = PlaceHolder[target=buf414]
#   %buf415 : Tensor "f32[4, 32, 1, 1, 32][1024, 32, 4096, 4096, 1]cuda:0" = PlaceHolder[target=buf415]
#   %buf416 : Tensor "f32[4, 32, 1, 1, 32][1024, 32, 4096, 4096, 1]cuda:0" = PlaceHolder[target=buf416]
#   %view_68 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_13, [4, 32, 4, 1048576]), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_68, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %getitem_59,%buf418
triton_per_fused_native_group_norm_66 = async_compile.triton('triton_per_fused_native_group_norm_66', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_66', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2048, 'r0_': 49152}}
)
@triton.jit
def triton_per_fused_native_group_norm_66(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/de/cdeugewejnka7bx67gqqembn3jk3w7dszfgogv7buwrii6tninpt.py
# Topologically Sorted Source Nodes: [hidden_states_107, hidden_states_108], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   hidden_states_107 => add_83, add_84, mul_92, mul_93, rsqrt_27, sub_27, unsqueeze_163, unsqueeze_164, unsqueeze_165, unsqueeze_166, unsqueeze_167, unsqueeze_168, var_mean_27, view_68, view_69
#   hidden_states_108 => mul_94, sigmoid_26
# Graph fragment:
#   %div_13 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0" = PlaceHolder[target=div_13]
#   %getitem_59 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_59]
#   %buf418 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf418]
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
triton_poi_fused_native_group_norm_silu_67 = async_compile.triton('triton_poi_fused_native_group_norm_silu_67', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_67', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 4294968320, 'x': 2147483648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_67(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/hw/chwmnbydsclxsp5estq2kscufscnn3u3zmxdwczl4ocxnptduouh.py
# Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113, add_14, output_tensor_13, sample_2], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_14 => add_87
#   hidden_states_111 => mul_97, sigmoid_27
#   hidden_states_113 => convolution_34
#   output_tensor_13 => div_14
#   sample_2 => var_mean_29, view_72
# Graph fragment:
#   %div_13 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0" = PlaceHolder[target=div_13]
#   %buf436 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=buf436]
#   %arg136_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg136_1]
#   %sigmoid_27 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_86,), kwargs = {})
#   %mul_97 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_86, %sigmoid_27), kwargs = {})
#   %convolution_34 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_97, %arg135_1, %arg136_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_87 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_13, %convolution_34), kwargs = {})
#   %div_14 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_87, 1.0), kwargs = {})
#   %view_72 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_14, [4, 32, 4, 1048576]), kwargs = {})
#   %var_mean_29 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_72, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf437,%buf438,%buf439
triton_red_fused_add_convolution_div_native_group_norm_silu_68 = async_compile.triton('triton_red_fused_add_convolution_div_native_group_norm_silu_68', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_div_native_group_norm_silu_68', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6291456, 'r0_': 2147483648}}
)
@triton.jit
def triton_red_fused_add_convolution_div_native_group_norm_silu_68(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/2v/c2vxag2glz2dfbgcllpr73pny3aeonp5j5sad2pvq62mea43hohu.py
# Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113, add_14, output_tensor_13, sample_2], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_14 => add_87
#   hidden_states_111 => mul_97, sigmoid_27
#   hidden_states_113 => convolution_34
#   output_tensor_13 => div_14
#   sample_2 => var_mean_29, view_72
# Graph fragment:
#   %buf437 : Tensor "f32[4, 32, 1, 1, 32, 64][65536, 2048, 262144, 262144, 64, 1]cuda:0" = PlaceHolder[target=buf437]
#   %buf438 : Tensor "f32[4, 32, 1, 1, 32, 64][65536, 2048, 262144, 262144, 64, 1]cuda:0" = PlaceHolder[target=buf438]
#   %buf439 : Tensor "f32[4, 32, 1, 1, 32, 64][65536, 2048, 262144, 262144, 64, 1]cuda:0" = PlaceHolder[target=buf439]
#   %sigmoid_27 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_86,), kwargs = {})
#   %mul_97 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_86, %sigmoid_27), kwargs = {})
#   %convolution_34 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_97, %arg135_1, %arg136_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_87 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_13, %convolution_34), kwargs = {})
#   %div_14 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_87, 1.0), kwargs = {})
#   %view_72 : Tensor "f32[4, 32, 4, 1048576][134217728, 4194304, 1048576, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%div_14, [4, 32, 4, 1048576]), kwargs = {})
#   %var_mean_29 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_72, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   return %buf440,%buf441,%buf442
triton_per_fused_add_convolution_div_native_group_norm_silu_69 = async_compile.triton('triton_per_fused_add_convolution_div_native_group_norm_silu_69', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_div_native_group_norm_silu_69', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 98304, 'r0_': 3145728}}
)
@triton.jit
def triton_per_fused_add_convolution_div_native_group_norm_silu_69(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/6m/c6mtgsb723k5q5tcx6ybq6nwlma6uhz6g7nj4twwxzxwldrmltvs.py
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
#   %buf436 : Tensor "f32[4, 128, 1024, 1024][134217728, 1, 131072, 128]cuda:0" = PlaceHolder[target=buf436]
#   %arg136_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg136_1]
#   %getitem_63 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=getitem_63]
#   %buf444 : Tensor "f32[4, 32, 1, 1][32, 1, 128, 128]cuda:0" = PlaceHolder[target=buf444]
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
triton_poi_fused_add_convolution_div_native_group_norm_silu_70 = async_compile.triton('triton_poi_fused_add_convolution_div_native_group_norm_silu_70', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_native_group_norm_silu_70', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 6442452480, 'x': 2147483648}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_native_group_norm_silu_70(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/sq/csqzsj4zdvwz63pjod3ysf2a6pmo72me2dhug2ptwkp5si4fuvh6.py
# Topologically Sorted Source Nodes: [sample_3, sample_4], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   sample_3 => mul_100, sigmoid_28
#   sample_4 => convolution_35
# Graph fragment:
#   %arg139_1 : Tensor "f32[3, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg139_1]
#   %sigmoid_28 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_89,), kwargs = {})
#   %mul_100 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, %sigmoid_28), kwargs = {})
#   %convolution_35 : Tensor "f32[4, 3, 1024, 1024][3145728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_100, %arg139_1, %arg140_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf448
triton_poi_fused_convolution_silu_71 = async_compile.triton('triton_poi_fused_convolution_silu_71', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_71', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 27648, 'x': 13824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_71(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/tw/ctwozj2vw3y6o72awhzfnlunajplt6ud6fdveaai7fzgmtsfnxrg.py
# Topologically Sorted Source Nodes: [sample_3, sample_4], Original ATen: [aten.silu, aten.convolution]
# Source node to ATen node mapping:
#   sample_3 => mul_100, sigmoid_28
#   sample_4 => convolution_35
# Graph fragment:
#   %buf449 : Tensor "f32[4, 3, 1024, 1024][3145728, 1, 3072, 3]cuda:0" = PlaceHolder[target=buf449]
#   %arg140_1 : Tensor "f32[3][1]cuda:0" = PlaceHolder[target=arg140_1]
#   %sigmoid_28 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_89,), kwargs = {})
#   %mul_100 : Tensor "f32[4, 128, 1024, 1024][134217728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, %sigmoid_28), kwargs = {})
#   %convolution_35 : Tensor "f32[4, 3, 1024, 1024][3145728, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_100, %arg139_1, %arg140_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %convolution_35
triton_poi_fused_convolution_silu_72 = async_compile.triton('triton_poi_fused_convolution_silu_72', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_72', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 50331660, 'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_silu_72(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
        assert_size_stride(arg0_1, (4, 4, 1, 1), (4, 1, 1, 1))
        assert_size_stride(arg1_1, (4, ), (1, ))
        assert_size_stride(arg2_1, (4, 4, 128, 128), (65536, 16384, 128, 1))
        assert_size_stride(arg3_1, (512, 4, 3, 3), (36, 9, 3, 1))
        assert_size_stride(arg4_1, (512, ), (1, ))
        assert_size_stride(arg5_1, (512, ), (1, ))
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
        assert_size_stride(arg87_1, (256, ), (1, ))
        assert_size_stride(arg88_1, (256, ), (1, ))
        assert_size_stride(arg89_1, (256, 256, 3, 3), (2304, 9, 3, 1))
        assert_size_stride(arg90_1, (256, ), (1, ))
        assert_size_stride(arg91_1, (256, 512, 1, 1), (512, 1, 1, 1))
        assert_size_stride(arg92_1, (256, ), (1, ))
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
        assert_size_stride(arg115_1, (128, ), (1, ))
        assert_size_stride(arg116_1, (128, ), (1, ))
        assert_size_stride(arg117_1, (128, 128, 3, 3), (1152, 9, 3, 1))
        assert_size_stride(arg118_1, (128, ), (1, ))
        assert_size_stride(arg119_1, (128, 256, 1, 1), (256, 1, 1, 1))
        assert_size_stride(arg120_1, (128, ), (1, ))
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
            # [Provenance debug handles] triton_poi_fused_convolution_0:1320
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(arg2_1, buf0, 16, 16384, stream=stream0)
            del arg2_1
            # Topologically Sorted Source Nodes: [z], Original ATen: [aten.convolution]
            buf1 = extern_kernels.convolution(buf0, arg0_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf1, (4, 4, 128, 128), (65536, 1, 512, 4), 'torch.ops.aten.convolution.default')
            del arg0_1
            del buf0
            buf2 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [z], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_1:1321
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_1.run(buf2, arg1_1, 262144, stream=stream0)
            del arg1_1
            buf3 = empty_strided_cuda((512, 4, 3, 3), (36, 1, 12, 4), torch.float16)
            # Topologically Sorted Source Nodes: [z, sample], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_2:1322
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_2.run(arg3_1, buf3, 2048, 9, stream=stream0)
            del arg3_1
            # Topologically Sorted Source Nodes: [z, sample], Original ATen: [aten.convolution]
            buf4 = extern_kernels.convolution(buf2, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf4, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
            del buf2
            del buf3
            buf5 = empty_strided_cuda((4, 32, 1, 1, 4, 256), (32768, 4, 131072, 131072, 1, 128), torch.float32)
            buf6 = empty_strided_cuda((4, 32, 1, 1, 4, 256), (32768, 4, 131072, 131072, 1, 128), torch.float32)
            buf7 = empty_strided_cuda((4, 32, 1, 1, 4, 256), (32768, 4, 131072, 131072, 1, 128), torch.float32)
            # Topologically Sorted Source Nodes: [z, sample, hidden_states], Original ATen: [aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_3:1323
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_3.run(buf4, arg4_1, buf5, buf6, buf7, 131072, 256, stream=stream0)
            buf8 = empty_strided_cuda((4, 32, 1, 1, 4, 2), (256, 4, 1024, 1024, 1, 128), torch.float32)
            buf9 = empty_strided_cuda((4, 32, 1, 1, 4, 2), (256, 4, 1024, 1024, 1, 128), torch.float32)
            buf10 = empty_strided_cuda((4, 32, 1, 1, 4, 2), (256, 4, 1024, 1024, 1, 128), torch.float32)
            # Topologically Sorted Source Nodes: [z, sample, hidden_states], Original ATen: [aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_4:1324
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_4.run(buf5, buf6, buf7, buf8, buf9, buf10, 1024, 128, stream=stream0)
            buf11 = empty_strided_cuda((4, 32, 1, 1, 4), (128, 4, 512, 512, 1), torch.float32)
            buf12 = empty_strided_cuda((4, 32, 1, 1, 4), (128, 4, 512, 512, 1), torch.float32)
            buf13 = empty_strided_cuda((4, 32, 1, 1, 4), (128, 4, 512, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [z, sample, hidden_states], Original ATen: [aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1325
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_5.run(buf8, buf9, buf10, buf11, buf12, buf13, 512, 2, stream=stream0)
            buf14 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
            buf15 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
            # Topologically Sorted Source Nodes: [z, sample, hidden_states], Original ATen: [aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1326
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_6.run(buf11, buf12, buf13, buf14, buf15, 128, 4, stream=stream0)
            buf18 = empty_strided_cuda((4, 512, 128, 128), (8388608, 1, 65536, 512), torch.float16)
            # Topologically Sorted Source Nodes: [z, sample, hidden_states, hidden_states_1], Original ATen: [aten.convolution, aten.native_group_norm, aten.silu]
            # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_7:1327
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_native_group_norm_silu_7.run(buf4, arg4_1, buf14, buf15, arg6_1, arg7_1, buf18, 2097152, 16, stream=stream0)
            del arg6_1
            del arg7_1
            buf19 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float16)
            # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_8:1328
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_8.run(arg8_1, buf19, 262144, 9, stream=stream0)
            del arg8_1
            # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2], Original ATen: [aten.silu, aten.convolution]
            buf20 = extern_kernels.convolution(buf18, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf20, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
            buf21 = buf7; del buf7  # reuse
            buf22 = buf6; del buf6  # reuse
            buf23 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, hidden_states_3], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_3:1329
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_3.run(buf20, arg9_1, buf21, buf22, buf23, 131072, 256, stream=stream0)
            buf24 = buf9; del buf9  # reuse
            buf25 = buf8; del buf8  # reuse
            buf26 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, hidden_states_3], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_4:1330
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_4.run(buf21, buf22, buf23, buf24, buf25, buf26, 1024, 128, stream=stream0)
            buf27 = buf13; del buf13  # reuse
            buf28 = buf12; del buf12  # reuse
            buf29 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, hidden_states_3], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1331
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_5.run(buf24, buf25, buf26, buf27, buf28, buf29, 512, 2, stream=stream0)
            buf30 = buf15; del buf15  # reuse
            buf31 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, hidden_states_3], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1332
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_6.run(buf27, buf28, buf29, buf30, buf31, 128, 4, stream=stream0)
            buf34 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_1, hidden_states_2, hidden_states_3, hidden_states_4], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_7:1333
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_native_group_norm_silu_7.run(buf20, arg9_1, buf30, buf31, arg10_1, arg11_1, buf34, 2097152, 16, stream=stream0)
            del arg10_1
            del arg11_1
            del arg9_1
            buf35 = buf19; del buf19  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_6], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_8:1334
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_8.run(arg12_1, buf35, 262144, 9, stream=stream0)
            del arg12_1
            # Topologically Sorted Source Nodes: [hidden_states_4, hidden_states_6], Original ATen: [aten.silu, aten.convolution]
            buf36 = extern_kernels.convolution(buf34, buf35, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf36, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
            buf37 = buf23; del buf23  # reuse
            buf38 = buf22; del buf22  # reuse
            buf39 = buf21; del buf21  # reuse
            # Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.view, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_add_convolution_div_native_group_norm_silu_view_9:1335
            stream0 = get_raw_stream(0)
            triton_per_fused_add_convolution_div_native_group_norm_silu_view_9.run(buf4, arg4_1, buf36, arg13_1, buf37, buf38, buf39, 131072, 256, stream=stream0)
            buf40 = buf26; del buf26  # reuse
            buf41 = buf25; del buf25  # reuse
            buf42 = buf24; del buf24  # reuse
            # Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.view, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_4:1336
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_4.run(buf37, buf38, buf39, buf40, buf41, buf42, 1024, 128, stream=stream0)
            buf43 = buf29; del buf29  # reuse
            buf44 = buf28; del buf28  # reuse
            buf45 = buf27; del buf27  # reuse
            # Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.view, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1337
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_5.run(buf40, buf41, buf42, buf43, buf44, buf45, 512, 2, stream=stream0)
            buf46 = buf31; del buf31  # reuse
            buf47 = buf30; del buf30  # reuse
            # Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.view, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1338
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_6.run(buf43, buf44, buf45, buf46, buf47, 128, 4, stream=stream0)
            buf50 = reinterpret_tensor(buf34, (4, 16384, 512), (8388608, 512, 1), 0); del buf34  # reuse
            buf52 = reinterpret_tensor(buf20, (4, 16384, 512), (8388608, 512, 1), 0); del buf20  # reuse
            buf54 = empty_strided_cuda((4, 16384, 512), (8388608, 512, 1), torch.float16)
            # Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2, hidden_states_8, query, key, value], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.view, aten.native_group_norm, aten.transpose, aten.clone]
            # [Provenance debug handles] triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_transpose_view_10:1339
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_transpose_view_10.run(buf4, arg4_1, buf36, arg13_1, buf46, buf47, arg14_1, arg15_1, buf50, buf52, buf54, 2097152, 16, stream=stream0)
            del arg14_1
            del arg15_1
            buf51 = empty_strided_cuda((65536, 512), (512, 1), torch.float16)
            # Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2, hidden_states_8, query], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.view, aten.native_group_norm, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:1496
            extern_kernels.mm(reinterpret_tensor(buf50, (65536, 512), (512, 1), 0), reinterpret_tensor(arg16_1, (512, 512), (1, 512), 0), out=buf51)
            del arg16_1
            buf53 = reinterpret_tensor(buf50, (65536, 512), (512, 1), 0); del buf50  # reuse
            # Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2, hidden_states_8, key], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.view, aten.native_group_norm, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:1497
            extern_kernels.mm(reinterpret_tensor(buf52, (65536, 512), (512, 1), 0), reinterpret_tensor(arg18_1, (512, 512), (1, 512), 0), out=buf53)
            del arg18_1
            buf55 = reinterpret_tensor(buf52, (65536, 512), (512, 1), 0); del buf52  # reuse
            # Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, view, group_norm_2, hidden_states_8, value], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.view, aten.native_group_norm, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:1498
            extern_kernels.mm(reinterpret_tensor(buf54, (65536, 512), (512, 1), 0), reinterpret_tensor(arg20_1, (512, 512), (1, 512), 0), out=buf55)
            del arg20_1
            del buf54
            buf56 = reinterpret_tensor(buf51, (4, 1, 16384, 512), (8388608, 8388608, 512, 1), 0); del buf51  # reuse
            # Topologically Sorted Source Nodes: [query, view_1, query_1, key, view_2, key_1, value, view_3, value_1, hidden_states_9], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_transpose_view_11:1340
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_transpose_view_11.run(buf56, arg17_1, 33554432, stream=stream0)
            del arg17_1
            buf57 = reinterpret_tensor(buf53, (4, 1, 16384, 512), (8388608, 8388608, 512, 1), 0); del buf53  # reuse
            # Topologically Sorted Source Nodes: [query, view_1, query_1, key, view_2, key_1, value, view_3, value_1, hidden_states_9], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_transpose_view_11:1341
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_transpose_view_11.run(buf57, arg19_1, 33554432, stream=stream0)
            del arg19_1
            buf58 = reinterpret_tensor(buf55, (4, 1, 16384, 512), (8388608, 8388608, 512, 1), 0); del buf55  # reuse
            # Topologically Sorted Source Nodes: [query, view_1, query_1, key, view_2, key_1, value, view_3, value_1, hidden_states_9], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_transpose_view_11:1342
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_transpose_view_11.run(buf58, arg21_1, 33554432, stream=stream0)
            del arg21_1
            # Topologically Sorted Source Nodes: [query, view_1, query_1, key, view_2, key_1, value, view_3, value_1, hidden_states_9], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf59 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf56, buf57, buf58, None, False)
            del buf56
            del buf57
            buf60 = buf59[0]
            assert_size_stride(buf60, (4, 1, 16384, 512), (8388608, 512, 512, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf60, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf59
            buf64 = reinterpret_tensor(buf58, (65536, 512), (512, 1), 0); del buf58  # reuse
            # Topologically Sorted Source Nodes: [transpose_6, hidden_states_10, hidden_states_12, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:1499
            extern_kernels.mm(reinterpret_tensor(buf60, (65536, 512), (512, 1), 0), reinterpret_tensor(arg22_1, (512, 512), (1, 512), 0), out=buf64)
            del arg22_1
            del buf60
            buf65 = reinterpret_tensor(buf64, (4, 512, 128, 128), (8388608, 1, 65536, 512), 0); del buf64  # reuse
            # Topologically Sorted Source Nodes: [z, sample, hidden_states_4, hidden_states_6, add, output_tensor, , hidden_states_12, transpose_7, hidden_states_14, hidden_states_15, hidden_states_16], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.addmm, aten.view, aten.transpose]
            # [Provenance debug handles] triton_poi_fused_add_addmm_convolution_div_silu_transpose_view_12:1343
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_convolution_div_silu_transpose_view_12.run(buf65, arg23_1, buf4, arg4_1, buf36, arg13_1, 33554432, stream=stream0)
            del arg13_1
            del arg23_1
            del arg4_1
            del buf36
            buf66 = buf39; del buf39  # reuse
            buf67 = buf38; del buf38  # reuse
            buf68 = buf37; del buf37  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_clone_native_group_norm_13:1344
            stream0 = get_raw_stream(0)
            triton_per_fused_clone_native_group_norm_13.run(buf65, buf66, buf67, buf68, 131072, 256, stream=stream0)
            buf69 = buf42; del buf42  # reuse
            buf70 = buf41; del buf41  # reuse
            buf71 = buf40; del buf40  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_4:1345
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_4.run(buf66, buf67, buf68, buf69, buf70, buf71, 1024, 128, stream=stream0)
            buf72 = buf45; del buf45  # reuse
            buf73 = buf44; del buf44  # reuse
            buf74 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1346
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_5.run(buf69, buf70, buf71, buf72, buf73, buf74, 512, 2, stream=stream0)
            buf75 = buf47; del buf47  # reuse
            buf76 = buf46; del buf46  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_17], Original ATen: [aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1347
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_6.run(buf72, buf73, buf74, buf75, buf76, 128, 4, stream=stream0)
            buf79 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_17, hidden_states_18], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
            # [Provenance debug handles] triton_poi_fused_clone_native_group_norm_silu_14:1348
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_native_group_norm_silu_14.run(buf65, buf75, buf76, arg24_1, arg25_1, buf79, 2097152, 16, stream=stream0)
            del arg24_1
            del arg25_1
            buf80 = buf35; del buf35  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_8:1349
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_8.run(arg26_1, buf80, 262144, 9, stream=stream0)
            del arg26_1
            # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19], Original ATen: [aten.silu, aten.convolution]
            buf81 = extern_kernels.convolution(buf79, buf80, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf81, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
            buf82 = buf68; del buf68  # reuse
            buf83 = buf67; del buf67  # reuse
            buf84 = buf66; del buf66  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19, hidden_states_20], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_3:1350
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_3.run(buf81, arg27_1, buf82, buf83, buf84, 131072, 256, stream=stream0)
            buf85 = buf71; del buf71  # reuse
            buf86 = buf70; del buf70  # reuse
            buf87 = buf69; del buf69  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19, hidden_states_20], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_4:1351
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_4.run(buf82, buf83, buf84, buf85, buf86, buf87, 1024, 128, stream=stream0)
            buf88 = buf74; del buf74  # reuse
            buf89 = buf73; del buf73  # reuse
            buf90 = buf72; del buf72  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19, hidden_states_20], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1352
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_5.run(buf85, buf86, buf87, buf88, buf89, buf90, 512, 2, stream=stream0)
            buf91 = buf76; del buf76  # reuse
            buf92 = buf75; del buf75  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19, hidden_states_20], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1353
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_6.run(buf88, buf89, buf90, buf91, buf92, 128, 4, stream=stream0)
            buf95 = buf79; del buf79  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19, hidden_states_20, hidden_states_21], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_7:1354
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_native_group_norm_silu_7.run(buf81, arg27_1, buf91, buf92, arg28_1, arg29_1, buf95, 2097152, 16, stream=stream0)
            del arg27_1
            del arg28_1
            del arg29_1
            del buf81
            buf96 = buf80; del buf80  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_8:1355
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_8.run(arg30_1, buf96, 262144, 9, stream=stream0)
            del arg30_1
            # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23], Original ATen: [aten.silu, aten.convolution]
            buf97 = extern_kernels.convolution(buf95, buf96, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf97, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
            del buf95
            del buf96
            buf98 = buf84; del buf84  # reuse
            buf99 = buf83; del buf83  # reuse
            buf100 = buf82; del buf82  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, sample_1, hidden_states_24], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_15:1356
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_15.run(buf65, buf97, arg31_1, buf98, buf99, buf100, 131072, 256, stream=stream0)
            buf101 = buf87; del buf87  # reuse
            buf102 = buf86; del buf86  # reuse
            buf103 = buf85; del buf85  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, sample_1, hidden_states_24], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_4:1357
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_4.run(buf98, buf99, buf100, buf101, buf102, buf103, 1024, 128, stream=stream0)
            buf104 = buf90; del buf90  # reuse
            buf105 = buf89; del buf89  # reuse
            buf106 = buf88; del buf88  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, sample_1, hidden_states_24], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1358
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_5.run(buf101, buf102, buf103, buf104, buf105, buf106, 512, 2, stream=stream0)
            buf107 = buf92; del buf92  # reuse
            buf108 = buf91; del buf91  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, sample_1, hidden_states_24], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1359
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_6.run(buf104, buf105, buf106, buf107, buf108, 128, 4, stream=stream0)
            buf110 = empty_strided_cuda((4, 512, 128, 128), (8388608, 1, 65536, 512), torch.float32)
            buf111 = buf110; del buf110  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, sample_1, hidden_states_24, hidden_states_25], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_16:1360
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_clone_convolution_div_native_group_norm_silu_16.run(buf111, buf65, buf97, arg31_1, buf107, buf108, arg5_1, arg32_1, 2097152, 16, stream=stream0)
            del arg32_1
            del arg5_1
            buf112 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_17:1361
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_17.run(arg33_1, buf112, 262144, 9, stream=stream0)
            del arg33_1
            # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26], Original ATen: [aten.silu, aten.convolution]
            buf113 = extern_kernels.convolution(buf111, buf112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf113, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
            del buf111
            buf114 = buf99; del buf99  # reuse
            buf115 = buf98; del buf98  # reuse
            buf116 = buf100; del buf100  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26, hidden_states_27], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_18:1362
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_silu_18.run(buf113, arg34_1, buf114, buf115, buf116, 131072, 256, stream=stream0)
            buf117 = buf103; del buf103  # reuse
            buf118 = buf102; del buf102  # reuse
            buf119 = buf101; del buf101  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26, hidden_states_27], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_4:1363
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_4.run(buf114, buf115, buf116, buf117, buf118, buf119, 1024, 128, stream=stream0)
            buf120 = buf106; del buf106  # reuse
            buf121 = buf105; del buf105  # reuse
            buf122 = buf104; del buf104  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26, hidden_states_27], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1364
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_5.run(buf117, buf118, buf119, buf120, buf121, buf122, 512, 2, stream=stream0)
            buf123 = buf108; del buf108  # reuse
            buf124 = buf107; del buf107  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26, hidden_states_27], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1365
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_6.run(buf120, buf121, buf122, buf123, buf124, 128, 4, stream=stream0)
            buf126 = buf113; del buf113  # reuse
            buf127 = buf126; del buf126  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_25, hidden_states_26, hidden_states_27, hidden_states_28], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_19:1366
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_native_group_norm_silu_19.run(buf127, arg34_1, buf123, buf124, arg35_1, arg36_1, 2097152, 16, stream=stream0)
            del arg34_1
            del arg35_1
            del arg36_1
            buf128 = buf112; del buf112  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_28, hidden_states_30], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_17:1367
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_17.run(arg37_1, buf128, 262144, 9, stream=stream0)
            del arg37_1
            # Topologically Sorted Source Nodes: [hidden_states_28, hidden_states_30], Original ATen: [aten.silu, aten.convolution]
            buf129 = extern_kernels.convolution(buf127, buf128, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf129, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
            buf130 = buf129; del buf129  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_21, hidden_states_23, add_2, output_tensor_1, sample_1, hidden_states_28, hidden_states_30, add_3, output_tensor_2], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_add_convolution_div_silu_20:1368
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_convolution_div_silu_20.run(buf130, buf65, buf97, arg31_1, arg38_1, 33554432, stream=stream0)
            del arg31_1
            del arg38_1
            del buf65
            del buf97
            buf131 = buf116; del buf116  # reuse
            buf132 = buf115; del buf115  # reuse
            buf133 = buf114; del buf114  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_31], Original ATen: [aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_clone_native_group_norm_21:1369
            stream0 = get_raw_stream(0)
            triton_per_fused_clone_native_group_norm_21.run(buf130, buf131, buf132, buf133, 131072, 256, stream=stream0)
            buf134 = buf119; del buf119  # reuse
            buf135 = buf118; del buf118  # reuse
            buf136 = buf117; del buf117  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_31], Original ATen: [aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_4:1370
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_4.run(buf131, buf132, buf133, buf134, buf135, buf136, 1024, 128, stream=stream0)
            buf137 = buf122; del buf122  # reuse
            buf138 = buf121; del buf121  # reuse
            buf139 = buf120; del buf120  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_31], Original ATen: [aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1371
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_5.run(buf134, buf135, buf136, buf137, buf138, buf139, 512, 2, stream=stream0)
            buf140 = buf124; del buf124  # reuse
            buf141 = buf123; del buf123  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_31], Original ATen: [aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1372
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_6.run(buf137, buf138, buf139, buf140, buf141, 128, 4, stream=stream0)
            buf143 = buf127; del buf127  # reuse
            buf144 = buf143; del buf143  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_31, hidden_states_32], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
            # [Provenance debug handles] triton_poi_fused_clone_native_group_norm_silu_22:1373
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_native_group_norm_silu_22.run(buf144, buf130, buf140, buf141, arg39_1, arg40_1, 2097152, 16, stream=stream0)
            del arg39_1
            del arg40_1
            buf145 = buf128; del buf128  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_17:1374
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_17.run(arg41_1, buf145, 262144, 9, stream=stream0)
            del arg41_1
            # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33], Original ATen: [aten.silu, aten.convolution]
            buf146 = extern_kernels.convolution(buf144, buf145, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf146, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
            del buf144
            buf147 = buf133; del buf133  # reuse
            buf148 = buf132; del buf132  # reuse
            buf149 = buf131; del buf131  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33, hidden_states_34], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_18:1375
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_silu_18.run(buf146, arg42_1, buf147, buf148, buf149, 131072, 256, stream=stream0)
            buf150 = buf136; del buf136  # reuse
            buf151 = buf135; del buf135  # reuse
            buf152 = buf134; del buf134  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33, hidden_states_34], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_4:1376
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_4.run(buf147, buf148, buf149, buf150, buf151, buf152, 1024, 128, stream=stream0)
            buf153 = buf139; del buf139  # reuse
            buf154 = buf138; del buf138  # reuse
            buf155 = buf137; del buf137  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33, hidden_states_34], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1377
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_5.run(buf150, buf151, buf152, buf153, buf154, buf155, 512, 2, stream=stream0)
            buf156 = buf141; del buf141  # reuse
            buf157 = buf140; del buf140  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33, hidden_states_34], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1378
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_6.run(buf153, buf154, buf155, buf156, buf157, 128, 4, stream=stream0)
            buf159 = buf146; del buf146  # reuse
            buf160 = buf159; del buf159  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_32, hidden_states_33, hidden_states_34, hidden_states_35], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_19:1379
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_native_group_norm_silu_19.run(buf160, arg42_1, buf156, buf157, arg43_1, arg44_1, 2097152, 16, stream=stream0)
            del arg42_1
            del arg43_1
            del arg44_1
            buf161 = buf145; del buf145  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_17:1380
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_17.run(arg45_1, buf161, 262144, 9, stream=stream0)
            del arg45_1
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37], Original ATen: [aten.silu, aten.convolution]
            buf162 = extern_kernels.convolution(buf160, buf161, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf162, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
            buf163 = buf149; del buf149  # reuse
            buf164 = buf148; del buf148  # reuse
            buf165 = buf147; del buf147  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_38], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_add_clone_convolution_div_native_group_norm_silu_23:1381
            stream0 = get_raw_stream(0)
            triton_per_fused_add_clone_convolution_div_native_group_norm_silu_23.run(buf130, buf162, arg46_1, buf163, buf164, buf165, 131072, 256, stream=stream0)
            buf166 = buf152; del buf152  # reuse
            buf167 = buf151; del buf151  # reuse
            buf168 = buf150; del buf150  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_38], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_4:1382
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_4.run(buf163, buf164, buf165, buf166, buf167, buf168, 1024, 128, stream=stream0)
            buf169 = buf155; del buf155  # reuse
            buf170 = buf154; del buf154  # reuse
            buf171 = buf153; del buf153  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_38], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1383
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_5.run(buf166, buf167, buf168, buf169, buf170, buf171, 512, 2, stream=stream0)
            buf172 = buf157; del buf157  # reuse
            buf173 = buf156; del buf156  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_38], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1384
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_6.run(buf169, buf170, buf171, buf172, buf173, 128, 4, stream=stream0)
            buf175 = buf160; del buf160  # reuse
            buf176 = buf175; del buf175  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_38, hidden_states_39], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_24:1385
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_clone_convolution_div_native_group_norm_silu_24.run(buf176, buf130, buf162, arg46_1, buf172, buf173, arg47_1, arg48_1, 2097152, 16, stream=stream0)
            del arg47_1
            del arg48_1
            buf177 = buf161; del buf161  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_17:1386
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_17.run(arg49_1, buf177, 262144, 9, stream=stream0)
            del arg49_1
            # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40], Original ATen: [aten.silu, aten.convolution]
            buf178 = extern_kernels.convolution(buf176, buf177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf178, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
            del buf176
            buf179 = buf165; del buf165  # reuse
            buf180 = buf164; del buf164  # reuse
            buf181 = buf163; del buf163  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40, hidden_states_41], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_18:1387
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_silu_18.run(buf178, arg50_1, buf179, buf180, buf181, 131072, 256, stream=stream0)
            buf182 = buf168; del buf168  # reuse
            buf183 = buf167; del buf167  # reuse
            buf184 = buf166; del buf166  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40, hidden_states_41], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_4:1388
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_4.run(buf179, buf180, buf181, buf182, buf183, buf184, 1024, 128, stream=stream0)
            buf185 = buf171; del buf171  # reuse
            buf186 = buf170; del buf170  # reuse
            buf187 = buf169; del buf169  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40, hidden_states_41], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_5:1389
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_5.run(buf182, buf183, buf184, buf185, buf186, buf187, 512, 2, stream=stream0)
            buf188 = buf173; del buf173  # reuse
            buf189 = buf172; del buf172  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40, hidden_states_41], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_6:1390
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_6.run(buf185, buf186, buf187, buf188, buf189, 128, 4, stream=stream0)
            del buf185
            del buf186
            del buf187
            buf191 = buf178; del buf178  # reuse
            buf192 = buf191; del buf191  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40, hidden_states_41, hidden_states_42], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_19:1391
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_native_group_norm_silu_19.run(buf192, arg50_1, buf188, buf189, arg51_1, arg52_1, 2097152, 16, stream=stream0)
            del arg50_1
            del arg51_1
            del arg52_1
            buf193 = buf177; del buf177  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_42, hidden_states_44], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_17:1392
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_17.run(arg53_1, buf193, 262144, 9, stream=stream0)
            del arg53_1
            # Topologically Sorted Source Nodes: [hidden_states_42, hidden_states_44], Original ATen: [aten.silu, aten.convolution]
            buf194 = extern_kernels.convolution(buf192, buf193, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf194, (4, 512, 128, 128), (8388608, 1, 65536, 512), 'torch.ops.aten.convolution.default')
            del buf192
            buf195 = empty_strided_cuda((4, 512, 256, 256), (33554432, 1, 131072, 512), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_25:1393
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_25.run(buf130, buf162, arg46_1, buf194, arg54_1, buf195, 134217728, stream=stream0)
            del arg46_1
            del arg54_1
            del buf130
            del buf162
            del buf194
            buf196 = buf193; del buf193  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_17:1394
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_17.run(arg55_1, buf196, 262144, 9, stream=stream0)
            del arg55_1
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
            buf197 = extern_kernels.convolution(buf195, buf196, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf197, (4, 512, 256, 256), (33554432, 1, 131072, 512), 'torch.ops.aten.convolution.default')
            buf198 = empty_strided_cuda((4, 32, 1, 1, 8, 64), (16384, 8, 65536, 65536, 1, 256), torch.float32)
            buf199 = empty_strided_cuda((4, 32, 1, 1, 8, 64), (16384, 8, 65536, 65536, 1, 256), torch.float32)
            buf200 = empty_strided_cuda((4, 32, 1, 1, 8, 64), (16384, 8, 65536, 65536, 1, 256), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_47], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_26:1395
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_26.run(buf197, arg56_1, buf198, buf199, buf200, 65536, 2048, stream=stream0)
            buf201 = reinterpret_tensor(buf184, (4, 32, 1, 1, 8), (256, 8, 1024, 1024, 1), 0); del buf184  # reuse
            buf202 = reinterpret_tensor(buf183, (4, 32, 1, 1, 8), (256, 8, 1024, 1024, 1), 0); del buf183  # reuse
            buf203 = reinterpret_tensor(buf182, (4, 32, 1, 1, 8), (256, 8, 1024, 1024, 1), 0); del buf182  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_47], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27:1396
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27.run(buf198, buf199, buf200, buf201, buf202, buf203, 1024, 64, stream=stream0)
            buf204 = buf189; del buf189  # reuse
            buf205 = buf188; del buf188  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_47], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28:1397
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28.run(buf201, buf202, buf203, buf204, buf205, 128, 8, stream=stream0)
            buf207 = buf195; del buf195  # reuse
            buf208 = buf207; del buf207  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_47, hidden_states_48], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29:1398
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_29.run(buf208, buf197, arg56_1, buf204, buf205, arg57_1, arg58_1, 8388608, 16, stream=stream0)
            del arg57_1
            del arg58_1
            buf209 = buf196; del buf196  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_17:1399
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_17.run(arg59_1, buf209, 262144, 9, stream=stream0)
            del arg59_1
            # Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49], Original ATen: [aten.silu, aten.convolution]
            buf210 = extern_kernels.convolution(buf208, buf209, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf210, (4, 512, 256, 256), (33554432, 1, 131072, 512), 'torch.ops.aten.convolution.default')
            del buf208
            buf211 = buf200; del buf200  # reuse
            buf212 = buf199; del buf199  # reuse
            buf213 = buf198; del buf198  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49, hidden_states_50], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_26:1400
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_26.run(buf210, arg60_1, buf211, buf212, buf213, 65536, 2048, stream=stream0)
            buf214 = buf203; del buf203  # reuse
            buf215 = buf202; del buf202  # reuse
            buf216 = buf201; del buf201  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49, hidden_states_50], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27:1401
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27.run(buf211, buf212, buf213, buf214, buf215, buf216, 1024, 64, stream=stream0)
            buf217 = buf205; del buf205  # reuse
            buf218 = buf204; del buf204  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49, hidden_states_50], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28:1402
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28.run(buf214, buf215, buf216, buf217, buf218, 128, 8, stream=stream0)
            buf220 = buf210; del buf210  # reuse
            buf221 = buf220; del buf220  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_48, hidden_states_49, hidden_states_50, hidden_states_51], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_30:1403
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_native_group_norm_silu_30.run(buf221, arg60_1, buf217, buf218, arg61_1, arg62_1, 8388608, 16, stream=stream0)
            del arg60_1
            del arg61_1
            del arg62_1
            buf222 = buf209; del buf209  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_51, hidden_states_53], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_17:1404
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_17.run(arg63_1, buf222, 262144, 9, stream=stream0)
            del arg63_1
            # Topologically Sorted Source Nodes: [hidden_states_51, hidden_states_53], Original ATen: [aten.silu, aten.convolution]
            buf223 = extern_kernels.convolution(buf221, buf222, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf223, (4, 512, 256, 256), (33554432, 1, 131072, 512), 'torch.ops.aten.convolution.default')
            buf224 = buf213; del buf213  # reuse
            buf225 = buf212; del buf212  # reuse
            buf226 = buf211; del buf211  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_54], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_31:1405
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_31.run(buf197, arg56_1, buf223, arg64_1, buf224, buf225, buf226, 65536, 2048, stream=stream0)
            buf227 = buf216; del buf216  # reuse
            buf228 = buf215; del buf215  # reuse
            buf229 = buf214; del buf214  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_54], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27:1406
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27.run(buf224, buf225, buf226, buf227, buf228, buf229, 1024, 64, stream=stream0)
            buf230 = buf218; del buf218  # reuse
            buf231 = buf217; del buf217  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_54], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28:1407
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28.run(buf227, buf228, buf229, buf230, buf231, 128, 8, stream=stream0)
            buf233 = buf221; del buf221  # reuse
            buf234 = buf233; del buf233  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_54, hidden_states_55], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_32:1408
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_32.run(buf234, buf197, arg56_1, buf223, arg64_1, buf230, buf231, arg65_1, arg66_1, 8388608, 16, stream=stream0)
            del arg65_1
            del arg66_1
            buf235 = buf222; del buf222  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_17:1409
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_17.run(arg67_1, buf235, 262144, 9, stream=stream0)
            del arg67_1
            # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56], Original ATen: [aten.silu, aten.convolution]
            buf236 = extern_kernels.convolution(buf234, buf235, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf236, (4, 512, 256, 256), (33554432, 1, 131072, 512), 'torch.ops.aten.convolution.default')
            del buf234
            buf237 = buf226; del buf226  # reuse
            buf238 = buf225; del buf225  # reuse
            buf239 = buf224; del buf224  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56, hidden_states_57], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_26:1410
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_26.run(buf236, arg68_1, buf237, buf238, buf239, 65536, 2048, stream=stream0)
            buf240 = buf229; del buf229  # reuse
            buf241 = buf228; del buf228  # reuse
            buf242 = buf227; del buf227  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56, hidden_states_57], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27:1411
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27.run(buf237, buf238, buf239, buf240, buf241, buf242, 1024, 64, stream=stream0)
            buf243 = buf231; del buf231  # reuse
            buf244 = buf230; del buf230  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56, hidden_states_57], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28:1412
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28.run(buf240, buf241, buf242, buf243, buf244, 128, 8, stream=stream0)
            buf246 = buf236; del buf236  # reuse
            buf247 = buf246; del buf246  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_55, hidden_states_56, hidden_states_57, hidden_states_58], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_30:1413
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_native_group_norm_silu_30.run(buf247, arg68_1, buf243, buf244, arg69_1, arg70_1, 8388608, 16, stream=stream0)
            del arg68_1
            del arg69_1
            del arg70_1
            buf248 = buf235; del buf235  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_58, hidden_states_60], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_17:1414
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_17.run(arg71_1, buf248, 262144, 9, stream=stream0)
            del arg71_1
            # Topologically Sorted Source Nodes: [hidden_states_58, hidden_states_60], Original ATen: [aten.silu, aten.convolution]
            buf249 = extern_kernels.convolution(buf247, buf248, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf249, (4, 512, 256, 256), (33554432, 1, 131072, 512), 'torch.ops.aten.convolution.default')
            del buf247
            buf250 = buf197; del buf197  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_37, add_4, output_tensor_3, hidden_states_42, hidden_states_44, add_5, output_tensor_4, hidden_states_45, hidden_states_46, hidden_states_51, hidden_states_53, add_6, output_tensor_5, hidden_states_58, hidden_states_60, add_7, output_tensor_6], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_33:1415
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_33.run(buf250, arg56_1, buf223, arg64_1, buf249, arg72_1, 134217728, stream=stream0)
            del arg56_1
            del arg64_1
            del arg72_1
            del buf223
            buf251 = buf239; del buf239  # reuse
            buf252 = buf238; del buf238  # reuse
            buf253 = buf237; del buf237  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused_clone_native_group_norm_34:1416
            stream0 = get_raw_stream(0)
            triton_red_fused_clone_native_group_norm_34.run(buf250, buf251, buf252, buf253, 65536, 2048, stream=stream0)
            buf254 = buf242; del buf242  # reuse
            buf255 = buf241; del buf241  # reuse
            buf256 = buf240; del buf240  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27:1417
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27.run(buf251, buf252, buf253, buf254, buf255, buf256, 1024, 64, stream=stream0)
            buf257 = buf244; del buf244  # reuse
            buf258 = buf243; del buf243  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_61], Original ATen: [aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28:1418
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28.run(buf254, buf255, buf256, buf257, buf258, 128, 8, stream=stream0)
            buf260 = buf249; del buf249  # reuse
            buf261 = buf260; del buf260  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_61, hidden_states_62], Original ATen: [aten.clone, aten.native_group_norm, aten.silu]
            # [Provenance debug handles] triton_poi_fused_clone_native_group_norm_silu_35:1419
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_native_group_norm_silu_35.run(buf261, buf250, buf257, buf258, arg73_1, arg74_1, 8388608, 16, stream=stream0)
            del arg73_1
            del arg74_1
            buf262 = buf248; del buf248  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_62, hidden_states_63], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_17:1420
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_17.run(arg75_1, buf262, 262144, 9, stream=stream0)
            del arg75_1
            # Topologically Sorted Source Nodes: [hidden_states_62, hidden_states_63], Original ATen: [aten.silu, aten.convolution]
            buf263 = extern_kernels.convolution(buf261, buf262, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf263, (4, 512, 256, 256), (33554432, 1, 131072, 512), 'torch.ops.aten.convolution.default')
            del buf261
            buf264 = buf253; del buf253  # reuse
            buf265 = buf252; del buf252  # reuse
            buf266 = buf251; del buf251  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_62, hidden_states_63, hidden_states_64], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_26:1421
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_26.run(buf263, arg76_1, buf264, buf265, buf266, 65536, 2048, stream=stream0)
            buf267 = buf256; del buf256  # reuse
            buf268 = buf255; del buf255  # reuse
            buf269 = buf254; del buf254  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_62, hidden_states_63, hidden_states_64], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27:1422
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_27.run(buf264, buf265, buf266, buf267, buf268, buf269, 1024, 64, stream=stream0)
            del buf264
            del buf265
            del buf266
            buf270 = buf258; del buf258  # reuse
            buf271 = buf257; del buf257  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_62, hidden_states_63, hidden_states_64], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28:1423
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_28.run(buf267, buf268, buf269, buf270, buf271, 128, 8, stream=stream0)
            del buf267
            del buf268
            del buf269
            buf273 = buf263; del buf263  # reuse
            buf274 = buf273; del buf273  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_62, hidden_states_63, hidden_states_64, hidden_states_65], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_30:1424
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_native_group_norm_silu_30.run(buf274, arg76_1, buf270, buf271, arg77_1, arg78_1, 8388608, 16, stream=stream0)
            del arg76_1
            del arg77_1
            del arg78_1
            buf275 = buf262; del buf262  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_17:1425
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_17.run(arg79_1, buf275, 262144, 9, stream=stream0)
            del arg79_1
            # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67], Original ATen: [aten.silu, aten.convolution]
            buf276 = extern_kernels.convolution(buf274, buf275, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf276, (4, 512, 256, 256), (33554432, 1, 131072, 512), 'torch.ops.aten.convolution.default')
            del buf274
            buf277 = empty_strided_cuda((4, 512, 512, 512), (134217728, 1, 262144, 512), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_36:1426
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_36.run(buf250, buf276, arg80_1, buf277, 536870912, stream=stream0)
            del arg80_1
            del buf250
            del buf276
            buf278 = buf275; del buf275  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_17:1427
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_17.run(arg81_1, buf278, 262144, 9, stream=stream0)
            del arg81_1
            # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
            buf279 = extern_kernels.convolution(buf277, buf278, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf279, (4, 512, 512, 512), (134217728, 1, 262144, 512), 'torch.ops.aten.convolution.default')
            del buf278
            buf280 = empty_strided_cuda((4, 32, 1, 1, 32, 64), (65536, 1, 262144, 262144, 2048, 32), torch.float32)
            buf281 = empty_strided_cuda((4, 32, 1, 1, 32, 64), (65536, 1, 262144, 262144, 2048, 32), torch.float32)
            buf282 = empty_strided_cuda((4, 32, 1, 1, 32, 64), (65536, 1, 262144, 262144, 2048, 32), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, hidden_states_70], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_37:1428
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_37.run(buf279, arg82_1, buf280, buf281, buf282, 262144, 2048, stream=stream0)
            buf283 = empty_strided_cuda((4, 32, 1, 1, 32), (1024, 1, 4096, 4096, 32), torch.float32)
            buf284 = empty_strided_cuda((4, 32, 1, 1, 32), (1024, 1, 4096, 4096, 32), torch.float32)
            buf285 = empty_strided_cuda((4, 32, 1, 1, 32), (1024, 1, 4096, 4096, 32), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, hidden_states_70], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38:1429
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38.run(buf280, buf281, buf282, buf283, buf284, buf285, 4096, 64, stream=stream0)
            del buf280
            del buf281
            del buf282
            buf286 = buf271; del buf271  # reuse
            buf287 = buf270; del buf270  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, hidden_states_70], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39:1430
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39.run(buf283, buf284, buf285, buf286, buf287, 128, 32, stream=stream0)
            del buf283
            del buf284
            del buf285
            buf289 = buf277; del buf277  # reuse
            buf302 = empty_strided_cuda((4, 512, 512, 512), (134217728, 1, 262144, 512), torch.float32)
            buf290 = buf289; del buf289  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, hidden_states_70, hidden_states_71, contiguous], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40:1431
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_40.run(buf290, buf279, arg82_1, buf286, buf287, arg83_1, arg84_1, buf302, 33554432, 16, stream=stream0)
            del arg82_1
            del arg83_1
            del arg84_1
            del buf279
            buf291 = empty_strided_cuda((256, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_41:1432
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_41.run(arg85_1, buf291, 131072, 9, stream=stream0)
            del arg85_1
            # Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72], Original ATen: [aten.silu, aten.convolution]
            buf292 = extern_kernels.convolution(buf290, buf291, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf292, (4, 256, 512, 512), (67108864, 1, 131072, 256), 'torch.ops.aten.convolution.default')
            del buf290
            del buf291
            buf293 = reinterpret_tensor(buf181, (4, 32, 1, 1, 16, 64), (32768, 1, 131072, 131072, 2048, 32), 0); del buf181  # reuse
            buf294 = reinterpret_tensor(buf180, (4, 32, 1, 1, 16, 64), (32768, 1, 131072, 131072, 2048, 32), 0); del buf180  # reuse
            buf295 = reinterpret_tensor(buf179, (4, 32, 1, 1, 16, 64), (32768, 1, 131072, 131072, 2048, 32), 0); del buf179  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused_convolution_native_group_norm_silu_42:1433
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_native_group_norm_silu_42.run(buf292, arg86_1, buf293, buf294, buf295, 131072, 2048, stream=stream0)
            buf296 = empty_strided_cuda((4, 32, 1, 1, 16), (512, 1, 2048, 2048, 32), torch.float32)
            buf297 = empty_strided_cuda((4, 32, 1, 1, 16), (512, 1, 2048, 2048, 32), torch.float32)
            buf298 = empty_strided_cuda((4, 32, 1, 1, 16), (512, 1, 2048, 2048, 32), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_43:1434
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_silu_43.run(buf293, buf294, buf295, buf296, buf297, buf298, 2048, 64, stream=stream0)
            buf299 = buf287; del buf287  # reuse
            buf300 = buf286; del buf286  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_44:1435
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_silu_44.run(buf296, buf297, buf298, buf299, buf300, 128, 16, stream=stream0)
            # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, contiguous, input_tensor], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
            buf303 = extern_kernels.convolution(buf302, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf303, (4, 256, 512, 512), (67108864, 1, 131072, 256), 'torch.ops.aten.convolution.default')
            del arg91_1
            del buf302
            buf304 = buf292; del buf292  # reuse
            buf305 = buf304; del buf304  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_71, hidden_states_72, hidden_states_73, hidden_states_74], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_45:1436
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_native_group_norm_silu_45.run(buf305, arg86_1, buf299, buf300, arg87_1, arg88_1, 268435456, stream=stream0)
            del arg86_1
            del arg87_1
            del arg88_1
            buf306 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_74, hidden_states_76], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_46:1437
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_46.run(arg89_1, buf306, 65536, 9, stream=stream0)
            del arg89_1
            # Topologically Sorted Source Nodes: [hidden_states_74, hidden_states_76], Original ATen: [aten.silu, aten.convolution]
            buf307 = extern_kernels.convolution(buf305, buf306, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf307, (4, 256, 512, 512), (67108864, 1, 131072, 256), 'torch.ops.aten.convolution.default')
            buf308 = buf295; del buf295  # reuse
            buf309 = buf294; del buf294  # reuse
            buf310 = buf293; del buf293  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, contiguous, input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_77], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_47:1438
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_47.run(buf303, arg92_1, buf307, arg90_1, buf308, buf309, buf310, 131072, 2048, stream=stream0)
            buf311 = buf298; del buf298  # reuse
            buf312 = buf297; del buf297  # reuse
            buf313 = buf296; del buf296  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, contiguous, input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_77], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_43:1439
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_silu_43.run(buf308, buf309, buf310, buf311, buf312, buf313, 2048, 64, stream=stream0)
            buf314 = buf300; del buf300  # reuse
            buf315 = buf299; del buf299  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, contiguous, input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_77], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_44:1440
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_silu_44.run(buf311, buf312, buf313, buf314, buf315, 128, 16, stream=stream0)
            buf317 = buf305; del buf305  # reuse
            buf318 = buf317; del buf317  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, contiguous, input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_77, hidden_states_78], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_48:1441
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_48.run(buf318, buf303, arg92_1, buf307, arg90_1, buf314, buf315, arg93_1, arg94_1, 268435456, stream=stream0)
            del arg93_1
            del arg94_1
            buf319 = buf306; del buf306  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_78, hidden_states_79], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_46:1442
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_46.run(arg95_1, buf319, 65536, 9, stream=stream0)
            del arg95_1
            # Topologically Sorted Source Nodes: [hidden_states_78, hidden_states_79], Original ATen: [aten.silu, aten.convolution]
            buf320 = extern_kernels.convolution(buf318, buf319, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf320, (4, 256, 512, 512), (67108864, 1, 131072, 256), 'torch.ops.aten.convolution.default')
            del buf318
            buf321 = buf310; del buf310  # reuse
            buf322 = buf309; del buf309  # reuse
            buf323 = buf308; del buf308  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_78, hidden_states_79, hidden_states_80], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused_convolution_native_group_norm_silu_42:1443
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_native_group_norm_silu_42.run(buf320, arg96_1, buf321, buf322, buf323, 131072, 2048, stream=stream0)
            buf324 = buf313; del buf313  # reuse
            buf325 = buf312; del buf312  # reuse
            buf326 = buf311; del buf311  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_78, hidden_states_79, hidden_states_80], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_43:1444
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_silu_43.run(buf321, buf322, buf323, buf324, buf325, buf326, 2048, 64, stream=stream0)
            buf327 = buf315; del buf315  # reuse
            buf328 = buf314; del buf314  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_78, hidden_states_79, hidden_states_80], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_44:1445
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_silu_44.run(buf324, buf325, buf326, buf327, buf328, 128, 16, stream=stream0)
            buf330 = buf320; del buf320  # reuse
            buf331 = buf330; del buf330  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_78, hidden_states_79, hidden_states_80, hidden_states_81], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_45:1446
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_native_group_norm_silu_45.run(buf331, arg96_1, buf327, buf328, arg97_1, arg98_1, 268435456, stream=stream0)
            del arg96_1
            del arg97_1
            del arg98_1
            buf332 = buf319; del buf319  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_81, hidden_states_83], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_46:1447
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_46.run(arg99_1, buf332, 65536, 9, stream=stream0)
            del arg99_1
            # Topologically Sorted Source Nodes: [hidden_states_81, hidden_states_83], Original ATen: [aten.silu, aten.convolution]
            buf333 = extern_kernels.convolution(buf331, buf332, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf333, (4, 256, 512, 512), (67108864, 1, 131072, 256), 'torch.ops.aten.convolution.default')
            buf334 = reinterpret_tensor(buf331, (4, 256, 512, 512), (67108864, 262144, 512, 1), 0); del buf331  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_65, hidden_states_67, add_8, output_tensor_7, hidden_states_68, hidden_states_69, contiguous, input_tensor, hidden_states_74, hidden_states_76, add_9, output_tensor_8, hidden_states_81, hidden_states_83, add_10, output_tensor_9], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index, aten.clone]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_49:1448
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_silu_unsqueeze_49.run(buf303, arg92_1, buf307, arg90_1, buf333, arg100_1, buf334, 1048576, 256, stream=stream0)
            del arg100_1
            del arg90_1
            del arg92_1
            del buf303
            del buf307
            buf335 = reinterpret_tensor(buf326, (4, 32, 1, 1, 16), (512, 16, 2048, 2048, 1), 0); del buf326  # reuse
            buf336 = reinterpret_tensor(buf325, (4, 32, 1, 1, 16), (512, 16, 2048, 2048, 1), 0); del buf325  # reuse
            buf337 = reinterpret_tensor(buf324, (4, 32, 1, 1, 16), (512, 16, 2048, 2048, 1), 0); del buf324  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_84], Original ATen: [aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused_native_group_norm_50:1449
            stream0 = get_raw_stream(0)
            triton_red_fused_native_group_norm_50.run(buf334, buf335, buf336, buf337, 2048, 131072, stream=stream0)
            buf338 = buf328; del buf328  # reuse
            buf339 = buf327; del buf327  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_84], Original ATen: [aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_native_group_norm_51:1450
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_51.run(buf335, buf336, buf337, buf338, buf339, 128, 16, stream=stream0)
            buf342 = buf333; del buf333  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_84, hidden_states_85], Original ATen: [aten.native_group_norm, aten.silu]
            # [Provenance debug handles] triton_poi_fused_native_group_norm_silu_52:1451
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_group_norm_silu_52.run(buf334, buf338, buf339, arg101_1, arg102_1, buf342, 1024, 262144, stream=stream0)
            del arg101_1
            del arg102_1
            buf343 = buf332; del buf332  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_46:1452
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_46.run(arg103_1, buf343, 65536, 9, stream=stream0)
            del arg103_1
            # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86], Original ATen: [aten.silu, aten.convolution]
            buf344 = extern_kernels.convolution(buf342, buf343, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf344, (4, 256, 512, 512), (67108864, 1, 131072, 256), 'torch.ops.aten.convolution.default')
            del buf342
            buf345 = buf323; del buf323  # reuse
            buf346 = buf322; del buf322  # reuse
            buf347 = buf321; del buf321  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86, hidden_states_87], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused_convolution_native_group_norm_silu_42:1453
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_native_group_norm_silu_42.run(buf344, arg104_1, buf345, buf346, buf347, 131072, 2048, stream=stream0)
            buf348 = reinterpret_tensor(buf337, (4, 32, 1, 1, 16), (512, 1, 2048, 2048, 32), 0); del buf337  # reuse
            buf349 = reinterpret_tensor(buf336, (4, 32, 1, 1, 16), (512, 1, 2048, 2048, 32), 0); del buf336  # reuse
            buf350 = reinterpret_tensor(buf335, (4, 32, 1, 1, 16), (512, 1, 2048, 2048, 32), 0); del buf335  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86, hidden_states_87], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_43:1454
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_silu_43.run(buf345, buf346, buf347, buf348, buf349, buf350, 2048, 64, stream=stream0)
            del buf345
            del buf346
            del buf347
            buf351 = buf339; del buf339  # reuse
            buf352 = buf338; del buf338  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86, hidden_states_87], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_convolution_native_group_norm_silu_44:1455
            stream0 = get_raw_stream(0)
            triton_per_fused_convolution_native_group_norm_silu_44.run(buf348, buf349, buf350, buf351, buf352, 128, 16, stream=stream0)
            del buf348
            del buf349
            del buf350
            buf354 = buf344; del buf344  # reuse
            buf355 = buf354; del buf354  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_85, hidden_states_86, hidden_states_87, hidden_states_88], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_45:1456
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_native_group_norm_silu_45.run(buf355, arg104_1, buf351, buf352, arg105_1, arg106_1, 268435456, stream=stream0)
            del arg104_1
            del arg105_1
            del arg106_1
            buf356 = buf343; del buf343  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_46:1457
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_46.run(arg107_1, buf356, 65536, 9, stream=stream0)
            del arg107_1
            # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90], Original ATen: [aten.silu, aten.convolution]
            buf357 = extern_kernels.convolution(buf355, buf356, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf357, (4, 256, 512, 512), (67108864, 1, 131072, 256), 'torch.ops.aten.convolution.default')
            del buf355
            buf358 = empty_strided_cuda((4, 256, 1024, 1024), (268435456, 1, 262144, 256), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_53:1458
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_53.run(buf334, buf357, arg108_1, buf358, 1073741824, stream=stream0)
            del arg108_1
            del buf334
            del buf357
            buf359 = buf356; del buf356  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91, hidden_states_92], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_46:1459
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_46.run(arg109_1, buf359, 65536, 9, stream=stream0)
            del arg109_1
            # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91, hidden_states_92], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index]
            buf360 = extern_kernels.convolution(buf358, buf359, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf360, (4, 256, 1024, 1024), (268435456, 1, 262144, 256), 'torch.ops.aten.convolution.default')
            del buf359
            buf361 = reinterpret_tensor(buf358, (4, 256, 1024, 1024), (268435456, 1048576, 1024, 1), 0); del buf358  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, add_11, output_tensor_10, hidden_states_91, hidden_states_92], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.arange, aten.mul, aten._to_copy, aten.unsqueeze, aten._unsafe_index]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_54:1460
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_index_add_arange_convolution_div_mul_silu_unsqueeze_54.run(buf360, arg110_1, buf361, 1024, 1048576, stream=stream0)
            del arg110_1
            buf362 = empty_strided_cuda((4, 32, 1, 1, 64), (2048, 64, 8192, 8192, 1), torch.float32)
            buf363 = empty_strided_cuda((4, 32, 1, 1, 64), (2048, 64, 8192, 8192, 1), torch.float32)
            buf364 = empty_strided_cuda((4, 32, 1, 1, 64), (2048, 64, 8192, 8192, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused_native_group_norm_55:1461
            stream0 = get_raw_stream(0)
            triton_red_fused_native_group_norm_55.run(buf361, buf362, buf363, buf364, 8192, 131072, stream=stream0)
            buf365 = buf352; del buf352  # reuse
            buf366 = buf351; del buf351  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_93], Original ATen: [aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_native_group_norm_56:1462
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_56.run(buf362, buf363, buf364, buf365, buf366, 128, 64, stream=stream0)
            del buf362
            del buf363
            del buf364
            buf369 = buf360; del buf360  # reuse
            buf381 = empty_strided_cuda((4, 256, 1024, 1024), (268435456, 1, 262144, 256), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_93, hidden_states_94, input_tensor_1], Original ATen: [aten.native_group_norm, aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_57:1463
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_native_group_norm_silu_57.run(buf361, buf365, buf366, arg111_1, arg112_1, buf369, buf381, 1024, 1048576, stream=stream0)
            del arg111_1
            del arg112_1
            del buf361
            buf370 = empty_strided_cuda((128, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_58:1464
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_58.run(arg113_1, buf370, 32768, 9, stream=stream0)
            del arg113_1
            # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95], Original ATen: [aten.silu, aten.convolution]
            buf371 = extern_kernels.convolution(buf369, buf370, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf371, (4, 128, 1024, 1024), (134217728, 1, 131072, 128), 'torch.ops.aten.convolution.default')
            del buf369
            del buf370
            buf372 = empty_strided_cuda((4, 32, 1, 1, 32, 64), (65536, 1, 262144, 262144, 2048, 32), torch.float32)
            buf373 = empty_strided_cuda((4, 32, 1, 1, 32, 64), (65536, 1, 262144, 262144, 2048, 32), torch.float32)
            buf374 = empty_strided_cuda((4, 32, 1, 1, 32, 64), (65536, 1, 262144, 262144, 2048, 32), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95, hidden_states_96], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused_convolution_native_group_norm_silu_59:1465
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_native_group_norm_silu_59.run(buf371, arg114_1, buf372, buf373, buf374, 262144, 2048, stream=stream0)
            buf375 = empty_strided_cuda((4, 32, 1, 1, 32), (1024, 1, 4096, 4096, 32), torch.float32)
            buf376 = empty_strided_cuda((4, 32, 1, 1, 32), (1024, 1, 4096, 4096, 32), torch.float32)
            buf377 = empty_strided_cuda((4, 32, 1, 1, 32), (1024, 1, 4096, 4096, 32), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95, hidden_states_96], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38:1466
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38.run(buf372, buf373, buf374, buf375, buf376, buf377, 4096, 64, stream=stream0)
            buf378 = buf366; del buf366  # reuse
            buf379 = buf365; del buf365  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95, hidden_states_96], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39:1467
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39.run(buf375, buf376, buf377, buf378, buf379, 128, 32, stream=stream0)
            # Topologically Sorted Source Nodes: [input_tensor_1], Original ATen: [aten.convolution]
            buf382 = extern_kernels.convolution(buf381, arg119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf382, (4, 128, 1024, 1024), (134217728, 1, 131072, 128), 'torch.ops.aten.convolution.default')
            del arg119_1
            del buf381
            buf383 = buf371; del buf371  # reuse
            buf384 = buf383; del buf383  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_94, hidden_states_95, hidden_states_96, hidden_states_97], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_60:1468
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_native_group_norm_silu_60.run(buf384, arg114_1, buf378, buf379, arg115_1, arg116_1, 536870912, stream=stream0)
            del arg114_1
            del arg115_1
            del arg116_1
            buf385 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_97, hidden_states_99], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_61:1469
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_61.run(arg117_1, buf385, 16384, 9, stream=stream0)
            del arg117_1
            # Topologically Sorted Source Nodes: [hidden_states_97, hidden_states_99], Original ATen: [aten.silu, aten.convolution]
            buf386 = extern_kernels.convolution(buf384, buf385, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf386, (4, 128, 1024, 1024), (134217728, 1, 131072, 128), 'torch.ops.aten.convolution.default')
            buf387 = buf374; del buf374  # reuse
            buf388 = buf373; del buf373  # reuse
            buf389 = buf372; del buf372  # reuse
            # Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_100], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused_add_convolution_div_native_group_norm_silu_62:1470
            stream0 = get_raw_stream(0)
            triton_red_fused_add_convolution_div_native_group_norm_silu_62.run(buf382, arg120_1, buf386, arg118_1, buf387, buf388, buf389, 262144, 2048, stream=stream0)
            buf390 = buf377; del buf377  # reuse
            buf391 = buf376; del buf376  # reuse
            buf392 = buf375; del buf375  # reuse
            # Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_100], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38:1471
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38.run(buf387, buf388, buf389, buf390, buf391, buf392, 4096, 64, stream=stream0)
            buf393 = buf379; del buf379  # reuse
            buf394 = buf378; del buf378  # reuse
            # Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_100], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39:1472
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39.run(buf390, buf391, buf392, buf393, buf394, 128, 32, stream=stream0)
            buf396 = buf384; del buf384  # reuse
            buf397 = buf396; del buf396  # reuse
            # Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_100, hidden_states_101], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused_add_convolution_div_native_group_norm_silu_63:1473
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_convolution_div_native_group_norm_silu_63.run(buf397, buf382, arg120_1, buf386, arg118_1, buf393, buf394, arg121_1, arg122_1, 536870912, stream=stream0)
            del arg121_1
            del arg122_1
            buf398 = buf385; del buf385  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_61:1474
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_61.run(arg123_1, buf398, 16384, 9, stream=stream0)
            del arg123_1
            # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102], Original ATen: [aten.silu, aten.convolution]
            buf399 = extern_kernels.convolution(buf397, buf398, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf399, (4, 128, 1024, 1024), (134217728, 1, 131072, 128), 'torch.ops.aten.convolution.default')
            del buf397
            buf400 = buf389; del buf389  # reuse
            buf401 = buf388; del buf388  # reuse
            buf402 = buf387; del buf387  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102, hidden_states_103], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused_convolution_native_group_norm_silu_59:1475
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_native_group_norm_silu_59.run(buf399, arg124_1, buf400, buf401, buf402, 262144, 2048, stream=stream0)
            buf403 = buf392; del buf392  # reuse
            buf404 = buf391; del buf391  # reuse
            buf405 = buf390; del buf390  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102, hidden_states_103], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38:1476
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38.run(buf400, buf401, buf402, buf403, buf404, buf405, 4096, 64, stream=stream0)
            buf406 = buf394; del buf394  # reuse
            buf407 = buf393; del buf393  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102, hidden_states_103], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39:1477
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39.run(buf403, buf404, buf405, buf406, buf407, 128, 32, stream=stream0)
            buf409 = buf399; del buf399  # reuse
            buf410 = buf409; del buf409  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_101, hidden_states_102, hidden_states_103, hidden_states_104], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_60:1478
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_native_group_norm_silu_60.run(buf410, arg124_1, buf406, buf407, arg125_1, arg126_1, 536870912, stream=stream0)
            del arg124_1
            del arg125_1
            del arg126_1
            buf411 = buf398; del buf398  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_104, hidden_states_106], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_61:1479
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_61.run(arg127_1, buf411, 16384, 9, stream=stream0)
            del arg127_1
            # Topologically Sorted Source Nodes: [hidden_states_104, hidden_states_106], Original ATen: [aten.silu, aten.convolution]
            buf412 = extern_kernels.convolution(buf410, buf411, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf412, (4, 128, 1024, 1024), (134217728, 1, 131072, 128), 'torch.ops.aten.convolution.default')
            buf413 = reinterpret_tensor(buf410, (4, 128, 1024, 1024), (134217728, 1048576, 1024, 1), 0); del buf410  # reuse
            # Topologically Sorted Source Nodes: [input_tensor_1, hidden_states_97, hidden_states_99, add_12, output_tensor_11, hidden_states_104, hidden_states_106, add_13, output_tensor_12], Original ATen: [aten.convolution, aten.silu, aten.add, aten.div]
            # [Provenance debug handles] triton_poi_fused_add_convolution_div_silu_64:1480
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_convolution_div_silu_64.run(buf382, arg120_1, buf386, arg118_1, buf412, arg128_1, buf413, 4194304, 128, stream=stream0)
            del arg118_1
            del arg120_1
            del arg128_1
            del buf382
            del buf386
            buf414 = reinterpret_tensor(buf405, (4, 32, 1, 1, 32), (1024, 32, 4096, 4096, 1), 0); del buf405  # reuse
            buf415 = reinterpret_tensor(buf404, (4, 32, 1, 1, 32), (1024, 32, 4096, 4096, 1), 0); del buf404  # reuse
            buf416 = reinterpret_tensor(buf403, (4, 32, 1, 1, 32), (1024, 32, 4096, 4096, 1), 0); del buf403  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_107], Original ATen: [aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused_native_group_norm_65:1481
            stream0 = get_raw_stream(0)
            triton_red_fused_native_group_norm_65.run(buf413, buf414, buf415, buf416, 4096, 131072, stream=stream0)
            buf417 = buf407; del buf407  # reuse
            buf418 = buf406; del buf406  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_107], Original ATen: [aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_native_group_norm_66:1482
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_66.run(buf414, buf415, buf416, buf417, buf418, 128, 32, stream=stream0)
            buf421 = buf412; del buf412  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_107, hidden_states_108], Original ATen: [aten.native_group_norm, aten.silu]
            # [Provenance debug handles] triton_poi_fused_native_group_norm_silu_67:1483
            stream0 = get_raw_stream(0)
            triton_poi_fused_native_group_norm_silu_67.run(buf413, buf417, buf418, arg129_1, arg130_1, buf421, 512, 1048576, stream=stream0)
            del arg129_1
            del arg130_1
            buf422 = buf411; del buf411  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_61:1484
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_61.run(arg131_1, buf422, 16384, 9, stream=stream0)
            del arg131_1
            # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109], Original ATen: [aten.silu, aten.convolution]
            buf423 = extern_kernels.convolution(buf421, buf422, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf423, (4, 128, 1024, 1024), (134217728, 1, 131072, 128), 'torch.ops.aten.convolution.default')
            del buf421
            buf424 = buf402; del buf402  # reuse
            buf425 = buf401; del buf401  # reuse
            buf426 = buf400; del buf400  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109, hidden_states_110], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused_convolution_native_group_norm_silu_59:1485
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_native_group_norm_silu_59.run(buf423, arg132_1, buf424, buf425, buf426, 262144, 2048, stream=stream0)
            buf427 = reinterpret_tensor(buf416, (4, 32, 1, 1, 32), (1024, 1, 4096, 4096, 32), 0); del buf416  # reuse
            buf428 = reinterpret_tensor(buf415, (4, 32, 1, 1, 32), (1024, 1, 4096, 4096, 32), 0); del buf415  # reuse
            buf429 = reinterpret_tensor(buf414, (4, 32, 1, 1, 32), (1024, 1, 4096, 4096, 32), 0); del buf414  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109, hidden_states_110], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38:1486
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_38.run(buf424, buf425, buf426, buf427, buf428, buf429, 4096, 64, stream=stream0)
            buf430 = buf418; del buf418  # reuse
            buf431 = buf417; del buf417  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109, hidden_states_110], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39:1487
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_index_add_arange_clone_convolution_div_mul_native_group_norm_silu_unsqueeze_39.run(buf427, buf428, buf429, buf430, buf431, 128, 32, stream=stream0)
            buf433 = buf423; del buf423  # reuse
            buf434 = buf433; del buf433  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_108, hidden_states_109, hidden_states_110, hidden_states_111], Original ATen: [aten.silu, aten.convolution, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused_convolution_native_group_norm_silu_60:1488
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_native_group_norm_silu_60.run(buf434, arg132_1, buf430, buf431, arg133_1, arg134_1, 536870912, stream=stream0)
            del arg132_1
            del arg133_1
            del arg134_1
            buf435 = buf422; del buf422  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_61:1489
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_61.run(arg135_1, buf435, 16384, 9, stream=stream0)
            del arg135_1
            # Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113], Original ATen: [aten.silu, aten.convolution]
            buf436 = extern_kernels.convolution(buf434, buf435, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf436, (4, 128, 1024, 1024), (134217728, 1, 131072, 128), 'torch.ops.aten.convolution.default')
            del buf435
            buf437 = reinterpret_tensor(buf426, (4, 32, 1, 1, 32, 64), (65536, 2048, 262144, 262144, 64, 1), 0); del buf426  # reuse
            buf438 = reinterpret_tensor(buf425, (4, 32, 1, 1, 32, 64), (65536, 2048, 262144, 262144, 64, 1), 0); del buf425  # reuse
            buf439 = reinterpret_tensor(buf424, (4, 32, 1, 1, 32, 64), (65536, 2048, 262144, 262144, 64, 1), 0); del buf424  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113, add_14, output_tensor_13, sample_2], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.native_group_norm]
            # [Provenance debug handles] triton_red_fused_add_convolution_div_native_group_norm_silu_68:1490
            stream0 = get_raw_stream(0)
            triton_red_fused_add_convolution_div_native_group_norm_silu_68.run(buf413, buf436, arg136_1, buf437, buf438, buf439, 262144, 2048, stream=stream0)
            buf440 = reinterpret_tensor(buf429, (4, 32, 1, 1, 32), (1024, 32, 4096, 4096, 1), 0); del buf429  # reuse
            buf441 = reinterpret_tensor(buf428, (4, 32, 1, 1, 32), (1024, 32, 4096, 4096, 1), 0); del buf428  # reuse
            buf442 = reinterpret_tensor(buf427, (4, 32, 1, 1, 32), (1024, 32, 4096, 4096, 1), 0); del buf427  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113, add_14, output_tensor_13, sample_2], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_add_convolution_div_native_group_norm_silu_69:1491
            stream0 = get_raw_stream(0)
            triton_per_fused_add_convolution_div_native_group_norm_silu_69.run(buf437, buf438, buf439, buf440, buf441, buf442, 4096, 64, stream=stream0)
            del buf437
            del buf438
            del buf439
            buf443 = buf431; del buf431  # reuse
            buf444 = buf430; del buf430  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113, add_14, output_tensor_13, sample_2], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.native_group_norm]
            # [Provenance debug handles] triton_per_fused_native_group_norm_66:1492
            stream0 = get_raw_stream(0)
            triton_per_fused_native_group_norm_66.run(buf440, buf441, buf442, buf443, buf444, 128, 32, stream=stream0)
            del buf440
            del buf441
            del buf442
            buf446 = buf413; del buf413  # reuse
            buf447 = buf434; del buf434  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_111, hidden_states_113, add_14, output_tensor_13, sample_2, sample_3], Original ATen: [aten.silu, aten.convolution, aten.add, aten.div, aten.native_group_norm]
            # [Provenance debug handles] triton_poi_fused_add_convolution_div_native_group_norm_silu_70:1493
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_convolution_div_native_group_norm_silu_70.run(buf446, buf436, arg136_1, buf443, buf444, arg137_1, arg138_1, buf447, 512, 1048576, stream=stream0)
            del arg136_1
            del arg137_1
            del arg138_1
            del buf436
            del buf443
            del buf444
            del buf446
            buf448 = empty_strided_cuda((3, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [sample_3, sample_4], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_71:1494
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_71.run(arg139_1, buf448, 384, 9, stream=stream0)
            del arg139_1
            # Topologically Sorted Source Nodes: [sample_3, sample_4], Original ATen: [aten.silu, aten.convolution]
            buf449 = extern_kernels.convolution(buf447, buf448, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf449, (4, 3, 1024, 1024), (3145728, 1, 3072, 3), 'torch.ops.aten.convolution.default')
            del buf447
            del buf448
            buf450 = empty_strided_cuda((4, 3, 1024, 1024), (3145728, 1048576, 1024, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sample_3, sample_4], Original ATen: [aten.silu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_silu_72:1495
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_silu_72.run(buf449, arg140_1, buf450, 12, 1048576, stream=stream0)
            del arg140_1
            del buf449
        return (buf450, )

runner = Runner(partitions=[])
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
