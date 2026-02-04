# AOT ID: ['0_inference']
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



# kernel path: /tmp/torchinductor_wucz/wc/cwctp7fkbpumr6x7hxr5m3eweu6s33smedx6zq6gtmas2ldfhint.py
# Topologically Sorted Source Nodes: [inputs_embeds, position_embeddings, embeddings, hidden_states], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   embeddings => add
#   hidden_states => add_2, add_3, convert_element_type, convert_element_type_1, mul, mul_1, rsqrt, sub, var_mean
#   inputs_embeds => embedding
#   position_embeddings => embedding_1
# Graph fragment:
#   %arg0_1 : Tensor "i64[4, 77][77, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "f16[49408, 768][768, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg2_1 : Tensor "i64[1, 77][77, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg1_1 : Tensor "f16[77, 768][768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %add : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0" = PlaceHolder[target=add]
#   %getitem_1 : Tensor "f32[4, 77, 1][77, 1, 308]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf2 : Tensor "f32[4, 77, 1][77, 1, 308]cuda:0" = PlaceHolder[target=buf2]
#   %arg4_1 : Tensor "f16[768][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f16[768][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %embedding : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg0_1), kwargs = {})
#   %embedding_1 : Tensor "f16[1, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg2_1), kwargs = {})
#   %add : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %convert_element_type : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %getitem_1), kwargs = {})
#   %add_2 : Tensor "f32[4, 77, 1][77, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[4, 77, 1][77, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg4_1), kwargs = {})
#   %add_3 : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg5_1), kwargs = {})
#   %convert_element_type_1 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.float16), kwargs = {})
#   return %add,%getitem_1,%buf2,%convert_element_type_1
triton_per_fused_add_embedding_native_layer_norm_0 = async_compile.triton('triton_per_fused_add_embedding_native_layer_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'in_ptr2': '*i64', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr0': '*fp16', 'out_ptr3': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def triton_per_fused_add_embedding_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 308
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    x3 = xindex
    r0_2 = r0_index
    x0 = (xindex % 77)
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tl.load(in_ptr5 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tl.full([XBLOCK, R0_BLOCK], 49408, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 49408)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 49408")
    tmp6 = tl.load(in_ptr1 + (r0_2 + 768*tmp4), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp8 = tl.full([XBLOCK, R0_BLOCK], 77, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert(((0 <= tmp11) & (tmp11 < 77)) | ~(xmask), "index out of bounds: 0 <= tmp11 < 77")
    tmp13 = tl.load(in_ptr3 + (r0_2 + 768*tmp11), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp14 = tmp6 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(r0_mask & xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(r0_mask & xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = tl.full([XBLOCK, 1], 768, tl.int32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = (tmp22 / tmp24)
    tmp26 = tmp16 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
    tmp30 = tl.where(r0_mask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None].to(tl.float32)
    tmp32 = tmp15 - tmp25
    tmp33 = 768.0
    tmp34 = (tmp31 / tmp33)
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.rsqrt(tmp36)
    tmp38 = tmp32 * tmp37
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp38 * tmp40
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp41 + tmp43
    tmp45 = tmp44.to(tl.float32)
    tl.store(out_ptr0 + (r0_2 + 768*x3), tmp14, r0_mask & xmask)
    tl.store(out_ptr3 + (r0_2 + 768*x3), tmp45, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/xp/cxp6p2sahwoj6tj3t24n7d73xffgdcwb4ye5ulwuzoyqzlbcc5rs.py
# Topologically Sorted Source Nodes: [queries], Original ATen: [aten.view, aten.t, aten.addmm]
# Source node to ATen node mapping:
#   queries => addmm, permute, view_2
# Graph fragment:
#   %arg7_1 : Tensor "f16[768][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %convert_element_type_1 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0" = PlaceHolder[target=convert_element_type_1]
#   %arg6_1 : Tensor "f16[768, 768][768, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %view_2 : Tensor "f16[308, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_1, [308, 768]), kwargs = {})
#   %permute : Tensor "f16[768, 768][1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg6_1, [1, 0]), kwargs = {})
#   %addmm : Tensor "f16[308, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg7_1, %view_2, %permute), kwargs = {})
#   return %addmm
triton_tem_fused_addmm_t_view_1 = async_compile.triton('triton_tem_fused_addmm_t_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=5,
num_warps=4,
triton_meta={'signature': {'in_ptr0': '*fp16', 'arg_A': '*fp16', 'arg_B': '*fp16', 'out_ptr0': '*fp16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_t_view_1', 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'ALLOW_TF32': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}},

)
@triton.jit
def triton_tem_fused_addmm_t_view_1(in_ptr0, arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 308
    N = 768
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

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
        xindex = idx_n + 768*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 768*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, xindex.shape)).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 768*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), tmp1, mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/rp/crp3nswuli5op4h6lgusxjhjn25xrwq66h7tauuonww7qu5pzazf.py
# Topologically Sorted Source Nodes: [queries, view_2, queries_1, keys, view_3, keys_1, values, view_4, values_1, mask_cond, add_1, view_1, lt, masked_fill_, mask, attn_output], Original ATen: [aten.view, aten.transpose, aten.arange, aten.add, aten.lt, aten.masked_fill, aten.full, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   add_1 => add_1
#   attn_output => _scaled_dot_product_efficient_attention, constant_pad_nd, expand_1, expand_2, slice_1, unsqueeze_2, unsqueeze_3
#   keys => view_5
#   keys_1 => permute_4
#   lt => lt
#   mask => full_default
#   mask_cond => iota
#   masked_fill_ => full_default_1, where
#   queries => view_3
#   queries_1 => permute_3
#   values => view_7
#   values_1 => permute_5
#   view_1 => view_1
#   view_2 => view_8
#   view_3 => view_9
#   view_4 => view_10
# Graph fragment:
#   %view_3 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [4, 77, 768]), kwargs = {})
#   %view_8 : Tensor "f16[4, 77, 12, 64][59136, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_3, [4, 77, -1, 64]), kwargs = {})
#   %permute_3 : Tensor "f16[4, 12, 77, 64][59136, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [0, 2, 1, 3]), kwargs = {})
#   %view_5 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [4, 77, 768]), kwargs = {})
#   %view_9 : Tensor "f16[4, 77, 12, 64][59136, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_5, [4, 77, -1, 64]), kwargs = {})
#   %permute_4 : Tensor "f16[4, 12, 77, 64][59136, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_9, [0, 2, 1, 3]), kwargs = {})
#   %view_7 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [4, 77, 768]), kwargs = {})
#   %view_10 : Tensor "f16[4, 77, 12, 64][59136, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_7, [4, 77, -1, 64]), kwargs = {})
#   %permute_5 : Tensor "f16[4, 12, 77, 64][59136, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_10, [0, 2, 1, 3]), kwargs = {})
#   %iota : Tensor "i64[77][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (77,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_1 : Tensor "i64[77][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, 1), kwargs = {})
#   %view_1 : Tensor "i64[77, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_1, [77, 1]), kwargs = {})
#   %lt : Tensor "b8[77, 77][77, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%iota, %view_1), kwargs = {})
#   %full_default_1 : Tensor "f16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default : Tensor "f16[77, 77][77, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([77, 77], -65504.0), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f16[77, 77][77, 1]cuda:0"[num_users=12] = call_function[target=torch.ops.aten.where.self](args = (%lt, %full_default_1, %full_default), kwargs = {})
#   %unsqueeze_2 : Tensor "f16[1, 77, 77][5929, 77, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where, 0), kwargs = {})
#   %unsqueeze_3 : Tensor "f16[1, 1, 77, 77][5929, 5929, 77, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 1), kwargs = {})
#   %expand_1 : Tensor "f16[4, 1, 77, 77][0, 5929, 77, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_3, [4, 1, 77, 77]), kwargs = {})
#   %constant_pad_nd : Tensor "f16[4, 1, 77, 80][6160, 6160, 80, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%expand_1, [0, 3], 0.0), kwargs = {})
#   %slice_1 : Tensor "f16[4, 1, 77, 77][6160, 6160, 80, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd, -1, 0, 77), kwargs = {})
#   %expand_2 : Tensor "f16[4, 12, 77, 77][6160, 0, 80, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_1, [4, 12, 77, 77]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_3, %permute_4, %permute_5, %expand_2, False), kwargs = {scale: 0.125})
#   return %buf8
triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 94864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 23716
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 77)
    x1 = ((xindex // 77) % 77)
    x3 = xindex // 77
    tmp0 = x0
    tmp1 = tl.full([1], 77, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = 1 + x1
    tmp5 = tmp3 < tmp4
    tmp6 = 0.0
    tmp7 = -65504.0
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp2, tmp8, tmp9)
    tl.store(out_ptr0 + (x0 + 80*x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/k3/ck3lkxz572th5mpfe52so6c56eiam7nxltb36kpj4wxwu5lo2xwh.py
# Topologically Sorted Source Nodes: [transpose_3, reshape, attn_output_3, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
# Source node to ATen node mapping:
#    => mm_default_35
#   attn_output_3 => permute_7, view_12
#   reshape => view_11
#   transpose_3 => permute_6
# Graph fragment:
#   %getitem_2 : Tensor "f16[4, 12, 77, 64][59136, 64, 768, 1]cuda:0" = PlaceHolder[target=getitem_2]
#   %arg12_1 : Tensor "f16[768, 768][768, 1]cuda:0" = PlaceHolder[target=arg12_1]
#   %permute_6 : Tensor "f16[4, 77, 12, 64][59136, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem_2, [0, 2, 1, 3]), kwargs = {})
#   %view_11 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_6, [4, 77, 768]), kwargs = {})
#   %view_12 : Tensor "f16[308, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_11, [308, 768]), kwargs = {})
#   %permute_7 : Tensor "f16[768, 768][1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg12_1, [1, 0]), kwargs = {})
#   %mm_default_35 : Tensor "f16[308, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_12, %permute_7), kwargs = {})
#   return %mm_default_35
triton_tem_fused_addmm_t_transpose_view_3 = async_compile.triton('triton_tem_fused_addmm_t_transpose_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=5,
num_warps=4,
triton_meta={'signature': {'arg_A': '*fp16', 'arg_B': '*fp16', 'out_ptr0': '*fp16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_addmm_t_transpose_view_3', 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'ALLOW_TF32': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}},

)
@triton.jit
def triton_tem_fused_addmm_t_transpose_view_3(arg_A, arg_B, out_ptr0):
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 128
    GROUP_M : tl.constexpr = 8
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 308
    N = 768
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

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
        xindex = idx_n + 768*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 768*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, xindex.shape)).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 768*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/b4/cb4fd2zagw3htycw6cxmtk2wifp74bfwl4biigz47kiedupur5db.py
# Topologically Sorted Source Nodes: [, attn_output_3, hidden_states_1, hidden_states_2], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#    => add_tensor_35
#   attn_output_3 => view_13
#   hidden_states_1 => add_4
#   hidden_states_2 => add_5, add_6, convert_element_type_14, convert_element_type_15, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %add : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0" = PlaceHolder[target=add]
#   %mm_default_35 : Tensor "f16[308, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_35]
#   %arg13_1 : Tensor "f16[768][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %getitem_7 : Tensor "f32[4, 77, 1][77, 1, 308]cuda:0" = PlaceHolder[target=getitem_7]
#   %buf16 : Tensor "f32[4, 77, 1][77, 1, 308]cuda:0" = PlaceHolder[target=buf16]
#   %arg14_1 : Tensor "f16[768][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %arg15_1 : Tensor "f16[768][1]cuda:0" = PlaceHolder[target=arg15_1]
#   %add_tensor_35 : Tensor "f16[308, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_35, %arg13_1), kwargs = {})
#   %view_13 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_35, [4, 77, 768]), kwargs = {})
#   %add_4 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_13), kwargs = {})
#   %convert_element_type_14 : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_14, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_14, %getitem_7), kwargs = {})
#   %add_5 : Tensor "f32[4, 77, 1][77, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[4, 77, 1][77, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_2 : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg14_1), kwargs = {})
#   %add_6 : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg15_1), kwargs = {})
#   %convert_element_type_15 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.float16), kwargs = {})
#   return %getitem_7,%buf16,%convert_element_type_15
triton_per_fused_add_addmm_native_layer_norm_view_4 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 1896960}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 308
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp29 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask & xmask, tmp6, 0)
    tmp9 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp13 = tl.full([XBLOCK, 1], 768, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = (tmp12 / tmp14)
    tmp16 = tmp6 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.where(r0_mask & xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None].to(tl.float32)
    tmp22 = tmp5 - tmp15
    tmp23 = 768.0
    tmp24 = (tmp21 / tmp23)
    tmp25 = 1e-05
    tmp26 = tmp24 + tmp25
    tmp27 = libdevice.rsqrt(tmp26)
    tmp28 = tmp22 * tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 * tmp30
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp35, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/jn/cjnzngpgn44dvtonmapa6uxexxvdnlvvt5z24vukseuws5qdwzmp.py
# Topologically Sorted Source Nodes: [, attn_output_3, hidden_states_1, hidden_states_2, hidden_states_3, mul, sigmoid, hidden_states_4], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t, aten.mul, aten.sigmoid]
# Source node to ATen node mapping:
#    => add_tensor_34, add_tensor_35, mm_default_34
#   attn_output_3 => view_13
#   hidden_states_1 => add_4
#   hidden_states_2 => add_5, add_6, convert_element_type_14, convert_element_type_15, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
#   hidden_states_3 => permute_8, view_14, view_15
#   hidden_states_4 => mul_5
#   mul => mul_4
#   sigmoid => sigmoid
# Graph fragment:
#   %convert_element_type_15 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0" = PlaceHolder[target=convert_element_type_15]
#   %arg16_1 : Tensor "f16[3072, 768][768, 1]cuda:0" = PlaceHolder[target=arg16_1]
#   %mm_default_34 : Tensor "f16[308, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_default_34]
#   %arg17_1 : Tensor "f16[3072][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %add_tensor_35 : Tensor "f16[308, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_35, %arg13_1), kwargs = {})
#   %view_13 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_35, [4, 77, 768]), kwargs = {})
#   %add_4 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_13), kwargs = {})
#   %convert_element_type_14 : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_14, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_14, %getitem_7), kwargs = {})
#   %add_5 : Tensor "f32[4, 77, 1][77, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[4, 77, 1][77, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_2 : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg14_1), kwargs = {})
#   %add_6 : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg15_1), kwargs = {})
#   %convert_element_type_15 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.float16), kwargs = {})
#   %view_14 : Tensor "f16[308, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_15, [308, 768]), kwargs = {})
#   %permute_8 : Tensor "f16[768, 3072][1, 768]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg16_1, [1, 0]), kwargs = {})
#   %mm_default_34 : Tensor "f16[308, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_14, %permute_8), kwargs = {})
#   %add_tensor_34 : Tensor "f16[308, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_34, %arg17_1), kwargs = {})
#   %view_15 : Tensor "f16[4, 77, 3072][236544, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [4, 77, 3072]), kwargs = {})
#   %mul_4 : Tensor "f16[4, 77, 3072][236544, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, 1.702), kwargs = {})
#   %sigmoid : Tensor "f16[4, 77, 3072][236544, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_4,), kwargs = {})
#   %mul_5 : Tensor "f16[4, 77, 3072][236544, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, %sigmoid), kwargs = {})
#   return %mm_default_34,%mul_5
triton_tem_fused_add_addmm_mul_native_layer_norm_sigmoid_t_view_5 = async_compile.triton('triton_tem_fused_add_addmm_mul_native_layer_norm_sigmoid_t_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=4,
num_warps=8,
triton_meta={'signature': {'arg_A': '*fp16', 'arg_B': '*fp16', 'in_ptr2': '*fp16', 'out_ptr1': '*fp16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_add_addmm_mul_native_layer_norm_sigmoid_t_view_5', 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'ALLOW_TF32': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}},

)
@triton.jit
def triton_tem_fused_add_addmm_mul_native_layer_norm_sigmoid_t_view_5(arg_A, arg_B, in_ptr2, out_ptr1):
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    USE_FAST_ACCUM : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    GROUP_M : tl.constexpr = 8
    INDEX_DTYPE : tl.constexpr = tl.int32
    A = arg_A
    B = arg_B

    M = 308
    N = 3072
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

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
        xindex = idx_n + 768*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 3072*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, xindex.shape)).broadcast_to(xindex.shape)))


        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 3072*idx_m
    tmp0 = tl.load(in_ptr2 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tmp2 = 1.702
    tmp3 = tmp1 * tmp2
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp1 * tmp4
    tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 3072*idx_m, acc.shape)), tmp5, mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/pf/cpf2uqjeq64atcie6whmgu4o6ogt5rgrgvt4plhkp3umjjk3hk6i.py
# Topologically Sorted Source Nodes: [, attn_output_3, hidden_states_1, hidden_states_5, hidden_states_6, hidden_states_7], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#    => add_tensor_33, add_tensor_35
#   attn_output_3 => view_13
#   hidden_states_1 => add_4
#   hidden_states_5 => view_17
#   hidden_states_6 => add_7
#   hidden_states_7 => add_8, add_9, convert_element_type_22, convert_element_type_23, mul_6, mul_7, rsqrt_2, sub_2, var_mean_2
# Graph fragment:
#   %add : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0" = PlaceHolder[target=add]
#   %mm_default_35 : Tensor "f16[308, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_35]
#   %arg13_1 : Tensor "f16[768][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %buf21 : Tensor "f16[308, 768][768, 1]cuda:0" = PlaceHolder[target=buf21]
#   %arg19_1 : Tensor "f16[768][1]cuda:0" = PlaceHolder[target=arg19_1]
#   %add_7 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0" = PlaceHolder[target=add_7]
#   %getitem_9 : Tensor "f32[4, 77, 1][77, 1, 308]cuda:0" = PlaceHolder[target=getitem_9]
#   %buf24 : Tensor "f32[4, 77, 1][77, 1, 308]cuda:0" = PlaceHolder[target=buf24]
#   %arg20_1 : Tensor "f16[768][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %arg21_1 : Tensor "f16[768][1]cuda:0" = PlaceHolder[target=arg21_1]
#   %add_tensor_35 : Tensor "f16[308, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_35, %arg13_1), kwargs = {})
#   %view_13 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_35, [4, 77, 768]), kwargs = {})
#   %add_4 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_13), kwargs = {})
#   %add_tensor_33 : Tensor "f16[308, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg19_1), kwargs = {})
#   %view_17 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_33, [4, 77, 768]), kwargs = {})
#   %add_7 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_17), kwargs = {})
#   %convert_element_type_22 : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_7, torch.float32), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_22, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_22, %getitem_9), kwargs = {})
#   %add_8 : Tensor "f32[4, 77, 1][77, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[4, 77, 1][77, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %mul_6 : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_7 : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %arg20_1), kwargs = {})
#   %add_9 : Tensor "f32[4, 77, 768][59136, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %arg21_1), kwargs = {})
#   %convert_element_type_23 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9, torch.float16), kwargs = {})
#   return %add_7,%getitem_9,%buf24,%convert_element_type_23
triton_per_fused_add_addmm_native_layer_norm_view_6 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 3317760}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 308
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask & xmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp17 = tl.full([XBLOCK, 1], 768, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.where(r0_mask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None].to(tl.float32)
    tmp26 = tmp9 - tmp19
    tmp27 = 768.0
    tmp28 = (tmp25 / tmp27)
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 * tmp34
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp8, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp39, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/qn/cqn65cvpeeavktw4flaool4azyuzval7ag6rmpp75rysbu2hxiqu.py
# Topologically Sorted Source Nodes: [to_1, argmax], Original ATen: [aten._to_copy, aten.argmax]
# Source node to ATen node mapping:
#   argmax => argmax
#   to_1 => convert_element_type_266
# Graph fragment:
#   %arg0_1 : Tensor "i64[4, 77][77, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convert_element_type_266 : Tensor "i32[4, 77][77, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.int32), kwargs = {})
#   %argmax : Tensor "i64[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.argmax.default](args = (%convert_element_type_266, -1), kwargs = {})
#   return %argmax
triton_per_fused__to_copy_argmax_7 = async_compile.triton('triton_per_fused__to_copy_argmax_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_argmax_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 32, 'r0_': 2464}}
)
@triton.jit
def triton_per_fused__to_copy_argmax_7(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4
    r0_numel = 77
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 77*x0), r0_mask & xmask, other=0.0)
    tmp1 = tmp0.to(tl.int32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask & xmask, tmp2, -2147483648)
    tmp5 = tl.broadcast_to(rindex, tmp4.shape)
    tmp3_val, tmp3_idx = triton_helpers.max_with_index(tmp4, tmp5, 1)
    tmp3 = tmp3_idx[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/7m/c7me427mtz3y34yj2clv5tb6xunccrrfodcbjwrwal7er3gmpqdv.py
# Topologically Sorted Source Nodes: [arange_1, pooled_output], Original ATen: [aten.arange, aten.index]
# Source node to ATen node mapping:
#   arange_1 => iota_1
#   pooled_output => index
# Graph fragment:
#   %argmax : Tensor "i64[4][1]cuda:0" = PlaceHolder[target=argmax]
#   %convert_element_type_265 : Tensor "f16[4, 77, 768][59136, 768, 1]cuda:0" = PlaceHolder[target=convert_element_type_265]
#   %iota_1 : Tensor "i64[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %index : Tensor "f16[4, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%convert_element_type_265, [%iota_1, %argmax]), kwargs = {})
#   return %index
triton_poi_fused_arange_index_8 = async_compile.triton('triton_poi_fused_arange_index_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_index_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_index_8(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 768
    x0 = (xindex % 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 77, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 77)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 77")
    tmp6 = tl.load(in_ptr1 + (x0 + 768*tmp4 + 59136*x1), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')

def partition_0(args):
    arg0_1, arg3_1, arg2_1, arg1_1, arg4_1, arg5_1, arg7_1, arg6_1, arg9_1, arg8_1, arg11_1, arg10_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg23_1, arg22_1, arg25_1, arg24_1, arg27_1, arg26_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg39_1, arg38_1, arg41_1, arg40_1, arg43_1, arg42_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg55_1, arg54_1, arg57_1, arg56_1, arg59_1, arg58_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg71_1, arg70_1, arg73_1, arg72_1, arg75_1, arg74_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg87_1, arg86_1, arg89_1, arg88_1, arg91_1, arg90_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg103_1, arg102_1, arg105_1, arg104_1, arg107_1, arg106_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg119_1, arg118_1, arg121_1, arg120_1, arg123_1, arg122_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg135_1, arg134_1, arg137_1, arg136_1, arg139_1, arg138_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg151_1, arg150_1, arg153_1, arg152_1, arg155_1, arg154_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg167_1, arg166_1, arg169_1, arg168_1, arg171_1, arg170_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg183_1, arg182_1, arg185_1, arg184_1, arg187_1, arg186_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 77), (77, 1))
    assert_size_stride(arg3_1, (49408, 768), (768, 1))
    assert_size_stride(arg2_1, (1, 77), (77, 1))
    assert_size_stride(arg1_1, (77, 768), (768, 1))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, 768), (768, 1))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, 768), (768, 1))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (768, 768), (768, 1))
    assert_size_stride(arg12_1, (768, 768), (768, 1))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (3072, 768), (768, 1))
    assert_size_stride(arg17_1, (3072, ), (1, ))
    assert_size_stride(arg18_1, (768, 3072), (3072, 1))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, 768), (768, 1))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, 768), (768, 1))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (768, 768), (768, 1))
    assert_size_stride(arg28_1, (768, 768), (768, 1))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (3072, 768), (768, 1))
    assert_size_stride(arg33_1, (3072, ), (1, ))
    assert_size_stride(arg34_1, (768, 3072), (3072, 1))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, 768), (768, 1))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, 768), (768, 1))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, 768), (768, 1))
    assert_size_stride(arg44_1, (768, 768), (768, 1))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (3072, 768), (768, 1))
    assert_size_stride(arg49_1, (3072, ), (1, ))
    assert_size_stride(arg50_1, (768, 3072), (3072, 1))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, 768), (768, 1))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, 768), (768, 1))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, 768), (768, 1))
    assert_size_stride(arg60_1, (768, 768), (768, 1))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (3072, 768), (768, 1))
    assert_size_stride(arg65_1, (3072, ), (1, ))
    assert_size_stride(arg66_1, (768, 3072), (3072, 1))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (768, 768), (768, 1))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg72_1, (768, 768), (768, 1))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, 768), (768, 1))
    assert_size_stride(arg76_1, (768, 768), (768, 1))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (3072, 768), (768, 1))
    assert_size_stride(arg81_1, (3072, ), (1, ))
    assert_size_stride(arg82_1, (768, 3072), (3072, 1))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (768, ), (1, ))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (768, 768), (768, 1))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, 768), (768, 1))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, 768), (768, 1))
    assert_size_stride(arg92_1, (768, 768), (768, 1))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (768, ), (1, ))
    assert_size_stride(arg96_1, (3072, 768), (768, 1))
    assert_size_stride(arg97_1, (3072, ), (1, ))
    assert_size_stride(arg98_1, (768, 3072), (3072, 1))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, 768), (768, 1))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, 768), (768, 1))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (768, 768), (768, 1))
    assert_size_stride(arg108_1, (768, 768), (768, 1))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (3072, 768), (768, 1))
    assert_size_stride(arg113_1, (3072, ), (1, ))
    assert_size_stride(arg114_1, (768, 3072), (3072, 1))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (768, ), (1, ))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, 768), (768, 1))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, 768), (768, 1))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, 768), (768, 1))
    assert_size_stride(arg124_1, (768, 768), (768, 1))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (3072, 768), (768, 1))
    assert_size_stride(arg129_1, (3072, ), (1, ))
    assert_size_stride(arg130_1, (768, 3072), (3072, 1))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (768, 768), (768, 1))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (768, 768), (768, 1))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, 768), (768, 1))
    assert_size_stride(arg140_1, (768, 768), (768, 1))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (3072, 768), (768, 1))
    assert_size_stride(arg145_1, (3072, ), (1, ))
    assert_size_stride(arg146_1, (768, 3072), (3072, 1))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg151_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (768, 768), (768, 1))
    assert_size_stride(arg153_1, (768, ), (1, ))
    assert_size_stride(arg152_1, (768, 768), (768, 1))
    assert_size_stride(arg155_1, (768, ), (1, ))
    assert_size_stride(arg154_1, (768, 768), (768, 1))
    assert_size_stride(arg156_1, (768, 768), (768, 1))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (768, ), (1, ))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (3072, 768), (768, 1))
    assert_size_stride(arg161_1, (3072, ), (1, ))
    assert_size_stride(arg162_1, (768, 3072), (3072, 1))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (768, ), (1, ))
    assert_size_stride(arg165_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (768, ), (1, ))
    assert_size_stride(arg166_1, (768, 768), (768, 1))
    assert_size_stride(arg169_1, (768, ), (1, ))
    assert_size_stride(arg168_1, (768, 768), (768, 1))
    assert_size_stride(arg171_1, (768, ), (1, ))
    assert_size_stride(arg170_1, (768, 768), (768, 1))
    assert_size_stride(arg172_1, (768, 768), (768, 1))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (768, ), (1, ))
    assert_size_stride(arg175_1, (768, ), (1, ))
    assert_size_stride(arg176_1, (3072, 768), (768, 1))
    assert_size_stride(arg177_1, (3072, ), (1, ))
    assert_size_stride(arg178_1, (768, 3072), (3072, 1))
    assert_size_stride(arg179_1, (768, ), (1, ))
    assert_size_stride(arg180_1, (768, ), (1, ))
    assert_size_stride(arg181_1, (768, ), (1, ))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg182_1, (768, 768), (768, 1))
    assert_size_stride(arg185_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, 768), (768, 1))
    assert_size_stride(arg187_1, (768, ), (1, ))
    assert_size_stride(arg186_1, (768, 768), (768, 1))
    assert_size_stride(arg188_1, (768, 768), (768, 1))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (768, ), (1, ))
    assert_size_stride(arg192_1, (3072, 768), (768, 1))
    assert_size_stride(arg193_1, (3072, ), (1, ))
    assert_size_stride(arg194_1, (768, 3072), (3072, 1))
    assert_size_stride(arg195_1, (768, ), (1, ))
    assert_size_stride(arg196_1, (768, ), (1, ))
    assert_size_stride(arg197_1, (768, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 77, 768), (59136, 768, 1), torch.float16)
        buf4 = empty_strided_cuda((4, 77, 768), (59136, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [inputs_embeds, position_embeddings, embeddings, hidden_states], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_embedding_native_layer_norm_0:1
        stream0 = get_raw_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_0.run(arg0_1, arg3_1, arg2_1, arg1_1, arg4_1, arg5_1, buf0, buf4, 308, 768, stream=stream0)
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        buf5 = empty_strided_cuda((308, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [queries], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg7_1, buf4, arg6_1, buf5, 60, 1, 1, stream=stream0)
        del arg6_1
        del arg7_1
        buf6 = empty_strided_cuda((308, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [keys], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg9_1, buf4, arg8_1, buf6, 60, 1, 1, stream=stream0)
        del arg8_1
        del arg9_1
        buf7 = empty_strided_cuda((308, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [values], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg11_1, buf4, arg10_1, buf7, 60, 1, 1, stream=stream0)
        del arg10_1
        del arg11_1
        buf8 = empty_strided_cuda((4, 1, 77, 77), (6160, 0, 80, 1), torch.float16)
        # Topologically Sorted Source Nodes: [queries, view_2, queries_1, keys, view_3, keys_1, values, view_4, values_1, mask_cond, add_1, view_1, lt, masked_fill_, mask, attn_output], Original ATen: [aten.view, aten.transpose, aten.arange, aten.add, aten.lt, aten.masked_fill, aten.full, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2:5
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2.run(buf8, 23716, stream=stream0)
        # Topologically Sorted Source Nodes: [queries, view_2, queries_1, keys, view_3, keys_1, values, view_4, values_1, mask_cond, add_1, view_1, lt, masked_fill_, mask, attn_output], Original ATen: [aten.view, aten.transpose, aten.arange, aten.add, aten.lt, aten.masked_fill, aten.full, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        buf9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf5, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf6, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf7, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf8, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
        buf10 = buf9[0]
        assert_size_stride(buf10, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf10, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf9
        buf14 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [transpose_3, reshape, attn_output_3, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_transpose_view_3.run(buf10, arg12_1, buf14, 60, 1, 1, stream=stream0)
        del arg12_1
        buf18 = reinterpret_tensor(buf10, (4, 77, 768), (59136, 768, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_3, hidden_states_1, hidden_states_2], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:7
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf0, buf14, arg13_1, arg14_1, arg15_1, buf18, 308, 768, stream=stream0)
        del arg14_1
        del arg15_1
        buf20 = empty_strided_cuda((4, 77, 3072), (236544, 3072, 1), torch.float16)
        # Topologically Sorted Source Nodes: [, attn_output_3, hidden_states_1, hidden_states_2, hidden_states_3, mul, sigmoid, hidden_states_4], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t, aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_tem_fused_add_addmm_mul_native_layer_norm_sigmoid_t_view_5.run(buf18, arg16_1, arg17_1, buf20, 120, 1, 1, stream=stream0)
        del arg16_1
        del arg17_1
        buf21 = reinterpret_tensor(buf18, (308, 768), (768, 1), 0); del buf18  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.mm:100
        extern_kernels.mm(reinterpret_tensor(buf20, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg18_1, (3072, 768), (1, 3072), 0), out=buf21)
        del arg18_1
        buf22 = reinterpret_tensor(buf14, (4, 77, 768), (59136, 768, 1), 0); del buf14  # reuse
        buf26 = reinterpret_tensor(buf6, (4, 77, 768), (59136, 768, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_3, hidden_states_1, hidden_states_5, hidden_states_6, hidden_states_7], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_6:9
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf22, buf0, arg13_1, buf21, arg19_1, arg20_1, arg21_1, buf26, 308, 768, stream=stream0)
        del arg13_1
        del arg19_1
        del arg20_1
        del arg21_1
        buf27 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [queries_2], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg23_1, buf26, arg22_1, buf27, 60, 1, 1, stream=stream0)
        del arg22_1
        del arg23_1
        buf28 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [keys_2], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg25_1, buf26, arg24_1, buf28, 60, 1, 1, stream=stream0)
        del arg24_1
        del arg25_1
        buf29 = reinterpret_tensor(buf4, (308, 768), (768, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [values_2], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg27_1, buf26, arg26_1, buf29, 60, 1, 1, stream=stream0)
        del arg26_1
        del arg27_1
        buf30 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_2, view_5, queries_3, keys_2, view_6, keys_3, values_2, view_7, values_3, attn_output_4], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2:13
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2.run(buf30, 23716, stream=stream0)
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_2, view_5, queries_3, keys_2, view_6, keys_3, values_2, view_7, values_3, attn_output_4], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        buf31 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf27, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf28, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf29, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf30, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
        buf32 = buf31[0]
        assert_size_stride(buf32, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf32, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf31
        buf36 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [transpose_7, reshape_1, attn_output_7, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_transpose_view_3.run(buf32, arg28_1, buf36, 60, 1, 1, stream=stream0)
        del arg28_1
        buf40 = reinterpret_tensor(buf32, (4, 77, 768), (59136, 768, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_7, hidden_states_8, hidden_states_9], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:15
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf22, buf36, arg29_1, arg30_1, arg31_1, buf40, 308, 768, stream=stream0)
        del arg30_1
        del arg31_1
        buf42 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_7, hidden_states_8, hidden_states_9, hidden_states_10, mul_2, sigmoid_1, hidden_states_11], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t, aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_tem_fused_add_addmm_mul_native_layer_norm_sigmoid_t_view_5.run(buf40, arg32_1, arg33_1, buf42, 120, 1, 1, stream=stream0)
        del arg32_1
        del arg33_1
        buf43 = reinterpret_tensor(buf40, (308, 768), (768, 1), 0); del buf40  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.mm:101
        extern_kernels.mm(reinterpret_tensor(buf42, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg34_1, (3072, 768), (1, 3072), 0), out=buf43)
        del arg34_1
        buf44 = reinterpret_tensor(buf36, (4, 77, 768), (59136, 768, 1), 0); del buf36  # reuse
        buf48 = reinterpret_tensor(buf28, (4, 77, 768), (59136, 768, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_7, hidden_states_8, hidden_states_12, hidden_states_13, hidden_states_14], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_6:17
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf44, buf22, arg29_1, buf43, arg35_1, arg36_1, arg37_1, buf48, 308, 768, stream=stream0)
        del arg29_1
        del arg35_1
        del arg36_1
        del arg37_1
        buf49 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [queries_4], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg39_1, buf48, arg38_1, buf49, 60, 1, 1, stream=stream0)
        del arg38_1
        del arg39_1
        buf50 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [keys_4], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg41_1, buf48, arg40_1, buf50, 60, 1, 1, stream=stream0)
        del arg40_1
        del arg41_1
        buf51 = reinterpret_tensor(buf26, (308, 768), (768, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [values_4], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg43_1, buf48, arg42_1, buf51, 60, 1, 1, stream=stream0)
        del arg42_1
        del arg43_1
        buf52 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_4, view_8, queries_5, keys_4, view_9, keys_5, values_4, view_10, values_5, attn_output_8], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2:21
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2.run(buf52, 23716, stream=stream0)
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_4, view_8, queries_5, keys_4, view_9, keys_5, values_4, view_10, values_5, attn_output_8], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        buf53 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf49, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf50, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf51, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf52, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
        buf54 = buf53[0]
        assert_size_stride(buf54, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf54, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf53
        buf58 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [transpose_11, reshape_2, attn_output_11, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_transpose_view_3.run(buf54, arg44_1, buf58, 60, 1, 1, stream=stream0)
        del arg44_1
        buf62 = reinterpret_tensor(buf54, (4, 77, 768), (59136, 768, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_11, hidden_states_15, hidden_states_16], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:23
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf44, buf58, arg45_1, arg46_1, arg47_1, buf62, 308, 768, stream=stream0)
        del arg46_1
        del arg47_1
        buf64 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_11, hidden_states_15, hidden_states_16, hidden_states_17, mul_4, sigmoid_2, hidden_states_18], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t, aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_tem_fused_add_addmm_mul_native_layer_norm_sigmoid_t_view_5.run(buf62, arg48_1, arg49_1, buf64, 120, 1, 1, stream=stream0)
        del arg48_1
        del arg49_1
        buf65 = reinterpret_tensor(buf62, (308, 768), (768, 1), 0); del buf62  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.mm:102
        extern_kernels.mm(reinterpret_tensor(buf64, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg50_1, (3072, 768), (1, 3072), 0), out=buf65)
        del arg50_1
        buf66 = reinterpret_tensor(buf58, (4, 77, 768), (59136, 768, 1), 0); del buf58  # reuse
        buf70 = reinterpret_tensor(buf50, (4, 77, 768), (59136, 768, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_11, hidden_states_15, hidden_states_19, hidden_states_20, hidden_states_21], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_6:25
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf66, buf44, arg45_1, buf65, arg51_1, arg52_1, arg53_1, buf70, 308, 768, stream=stream0)
        del arg45_1
        del arg51_1
        del arg52_1
        del arg53_1
        buf71 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [queries_6], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg55_1, buf70, arg54_1, buf71, 60, 1, 1, stream=stream0)
        del arg54_1
        del arg55_1
        buf72 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [keys_6], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg57_1, buf70, arg56_1, buf72, 60, 1, 1, stream=stream0)
        del arg56_1
        del arg57_1
        buf73 = reinterpret_tensor(buf48, (308, 768), (768, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [values_6], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg59_1, buf70, arg58_1, buf73, 60, 1, 1, stream=stream0)
        del arg58_1
        del arg59_1
        buf74 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_6, view_11, queries_7, keys_6, view_12, keys_7, values_6, view_13, values_7, attn_output_12], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2:29
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2.run(buf74, 23716, stream=stream0)
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_6, view_11, queries_7, keys_6, view_12, keys_7, values_6, view_13, values_7, attn_output_12], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        buf75 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf71, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf72, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf73, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf74, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
        buf76 = buf75[0]
        assert_size_stride(buf76, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf76, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf75
        buf80 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [transpose_15, reshape_3, attn_output_15, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_transpose_view_3.run(buf76, arg60_1, buf80, 60, 1, 1, stream=stream0)
        del arg60_1
        buf84 = reinterpret_tensor(buf76, (4, 77, 768), (59136, 768, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_15, hidden_states_22, hidden_states_23], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:31
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf66, buf80, arg61_1, arg62_1, arg63_1, buf84, 308, 768, stream=stream0)
        del arg62_1
        del arg63_1
        buf86 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_15, hidden_states_22, hidden_states_23, hidden_states_24, mul_6, sigmoid_3, hidden_states_25], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t, aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_tem_fused_add_addmm_mul_native_layer_norm_sigmoid_t_view_5.run(buf84, arg64_1, arg65_1, buf86, 120, 1, 1, stream=stream0)
        del arg64_1
        del arg65_1
        buf87 = reinterpret_tensor(buf84, (308, 768), (768, 1), 0); del buf84  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.mm:103
        extern_kernels.mm(reinterpret_tensor(buf86, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg66_1, (3072, 768), (1, 3072), 0), out=buf87)
        del arg66_1
        buf88 = reinterpret_tensor(buf80, (4, 77, 768), (59136, 768, 1), 0); del buf80  # reuse
        buf92 = reinterpret_tensor(buf72, (4, 77, 768), (59136, 768, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_15, hidden_states_22, hidden_states_26, hidden_states_27, hidden_states_28], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_6:33
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf88, buf66, arg61_1, buf87, arg67_1, arg68_1, arg69_1, buf92, 308, 768, stream=stream0)
        del arg61_1
        del arg67_1
        del arg68_1
        del arg69_1
        buf93 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [queries_8], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg71_1, buf92, arg70_1, buf93, 60, 1, 1, stream=stream0)
        del arg70_1
        del arg71_1
        buf94 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [keys_8], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg73_1, buf92, arg72_1, buf94, 60, 1, 1, stream=stream0)
        del arg72_1
        del arg73_1
        buf95 = reinterpret_tensor(buf70, (308, 768), (768, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [values_8], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg75_1, buf92, arg74_1, buf95, 60, 1, 1, stream=stream0)
        del arg74_1
        del arg75_1
        buf96 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_8, view_14, queries_9, keys_8, view_15, keys_9, values_8, view_16, values_9, attn_output_16], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2:37
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2.run(buf96, 23716, stream=stream0)
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_8, view_14, queries_9, keys_8, view_15, keys_9, values_8, view_16, values_9, attn_output_16], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        buf97 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf93, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf94, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf95, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf96, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
        buf98 = buf97[0]
        assert_size_stride(buf98, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf98, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf97
        buf102 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [transpose_19, reshape_4, attn_output_19, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_transpose_view_3.run(buf98, arg76_1, buf102, 60, 1, 1, stream=stream0)
        del arg76_1
        buf106 = reinterpret_tensor(buf98, (4, 77, 768), (59136, 768, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_19, hidden_states_29, hidden_states_30], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:39
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf88, buf102, arg77_1, arg78_1, arg79_1, buf106, 308, 768, stream=stream0)
        del arg78_1
        del arg79_1
        buf108 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_19, hidden_states_29, hidden_states_30, hidden_states_31, mul_8, sigmoid_4, hidden_states_32], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t, aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_tem_fused_add_addmm_mul_native_layer_norm_sigmoid_t_view_5.run(buf106, arg80_1, arg81_1, buf108, 120, 1, 1, stream=stream0)
        del arg80_1
        del arg81_1
        buf109 = reinterpret_tensor(buf106, (308, 768), (768, 1), 0); del buf106  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.mm:104
        extern_kernels.mm(reinterpret_tensor(buf108, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg82_1, (3072, 768), (1, 3072), 0), out=buf109)
        del arg82_1
        buf110 = reinterpret_tensor(buf102, (4, 77, 768), (59136, 768, 1), 0); del buf102  # reuse
        buf114 = reinterpret_tensor(buf94, (4, 77, 768), (59136, 768, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_19, hidden_states_29, hidden_states_33, hidden_states_34, hidden_states_35], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_6:41
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf110, buf88, arg77_1, buf109, arg83_1, arg84_1, arg85_1, buf114, 308, 768, stream=stream0)
        del arg77_1
        del arg83_1
        del arg84_1
        del arg85_1
        buf115 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [queries_10], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg87_1, buf114, arg86_1, buf115, 60, 1, 1, stream=stream0)
        del arg86_1
        del arg87_1
        buf116 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [keys_10], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg89_1, buf114, arg88_1, buf116, 60, 1, 1, stream=stream0)
        del arg88_1
        del arg89_1
        buf117 = reinterpret_tensor(buf92, (308, 768), (768, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [values_10], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg91_1, buf114, arg90_1, buf117, 60, 1, 1, stream=stream0)
        del arg90_1
        del arg91_1
        buf118 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_10, view_17, queries_11, keys_10, view_18, keys_11, values_10, view_19, values_11, attn_output_20], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2:45
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2.run(buf118, 23716, stream=stream0)
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_10, view_17, queries_11, keys_10, view_18, keys_11, values_10, view_19, values_11, attn_output_20], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        buf119 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf115, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf116, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf117, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf118, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
        buf120 = buf119[0]
        assert_size_stride(buf120, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf120, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf119
        buf124 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [transpose_23, reshape_5, attn_output_23, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_transpose_view_3.run(buf120, arg92_1, buf124, 60, 1, 1, stream=stream0)
        del arg92_1
        buf128 = reinterpret_tensor(buf120, (4, 77, 768), (59136, 768, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_23, hidden_states_36, hidden_states_37], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:47
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf110, buf124, arg93_1, arg94_1, arg95_1, buf128, 308, 768, stream=stream0)
        del arg94_1
        del arg95_1
        buf130 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_23, hidden_states_36, hidden_states_37, hidden_states_38, mul_10, sigmoid_5, hidden_states_39], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t, aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_tem_fused_add_addmm_mul_native_layer_norm_sigmoid_t_view_5.run(buf128, arg96_1, arg97_1, buf130, 120, 1, 1, stream=stream0)
        del arg96_1
        del arg97_1
        buf131 = reinterpret_tensor(buf128, (308, 768), (768, 1), 0); del buf128  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.mm:105
        extern_kernels.mm(reinterpret_tensor(buf130, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg98_1, (3072, 768), (1, 3072), 0), out=buf131)
        del arg98_1
        buf132 = reinterpret_tensor(buf124, (4, 77, 768), (59136, 768, 1), 0); del buf124  # reuse
        buf136 = reinterpret_tensor(buf116, (4, 77, 768), (59136, 768, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_23, hidden_states_36, hidden_states_40, hidden_states_41, hidden_states_42], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_6:49
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf132, buf110, arg93_1, buf131, arg99_1, arg100_1, arg101_1, buf136, 308, 768, stream=stream0)
        del arg100_1
        del arg101_1
        del arg93_1
        del arg99_1
        buf137 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [queries_12], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg103_1, buf136, arg102_1, buf137, 60, 1, 1, stream=stream0)
        del arg102_1
        del arg103_1
        buf138 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [keys_12], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg105_1, buf136, arg104_1, buf138, 60, 1, 1, stream=stream0)
        del arg104_1
        del arg105_1
        buf139 = reinterpret_tensor(buf114, (308, 768), (768, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [values_12], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg107_1, buf136, arg106_1, buf139, 60, 1, 1, stream=stream0)
        del arg106_1
        del arg107_1
        buf140 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_12, view_20, queries_13, keys_12, view_21, keys_13, values_12, view_22, values_13, attn_output_24], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2:53
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2.run(buf140, 23716, stream=stream0)
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_12, view_20, queries_13, keys_12, view_21, keys_13, values_12, view_22, values_13, attn_output_24], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        buf141 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf137, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf138, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf139, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf140, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
        buf142 = buf141[0]
        assert_size_stride(buf142, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf142, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf141
        buf146 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [transpose_27, reshape_6, attn_output_27, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_transpose_view_3.run(buf142, arg108_1, buf146, 60, 1, 1, stream=stream0)
        del arg108_1
        buf150 = reinterpret_tensor(buf142, (4, 77, 768), (59136, 768, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_27, hidden_states_43, hidden_states_44], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:55
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf132, buf146, arg109_1, arg110_1, arg111_1, buf150, 308, 768, stream=stream0)
        del arg110_1
        del arg111_1
        buf152 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_27, hidden_states_43, hidden_states_44, hidden_states_45, mul_12, sigmoid_6, hidden_states_46], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t, aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_tem_fused_add_addmm_mul_native_layer_norm_sigmoid_t_view_5.run(buf150, arg112_1, arg113_1, buf152, 120, 1, 1, stream=stream0)
        del arg112_1
        del arg113_1
        buf153 = reinterpret_tensor(buf150, (308, 768), (768, 1), 0); del buf150  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.mm:106
        extern_kernels.mm(reinterpret_tensor(buf152, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg114_1, (3072, 768), (1, 3072), 0), out=buf153)
        del arg114_1
        buf154 = reinterpret_tensor(buf146, (4, 77, 768), (59136, 768, 1), 0); del buf146  # reuse
        buf158 = reinterpret_tensor(buf138, (4, 77, 768), (59136, 768, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_27, hidden_states_43, hidden_states_47, hidden_states_48, hidden_states_49], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_6:57
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf154, buf132, arg109_1, buf153, arg115_1, arg116_1, arg117_1, buf158, 308, 768, stream=stream0)
        del arg109_1
        del arg115_1
        del arg116_1
        del arg117_1
        buf159 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [queries_14], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg119_1, buf158, arg118_1, buf159, 60, 1, 1, stream=stream0)
        del arg118_1
        del arg119_1
        buf160 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [keys_14], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg121_1, buf158, arg120_1, buf160, 60, 1, 1, stream=stream0)
        del arg120_1
        del arg121_1
        buf161 = reinterpret_tensor(buf136, (308, 768), (768, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [values_14], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg123_1, buf158, arg122_1, buf161, 60, 1, 1, stream=stream0)
        del arg122_1
        del arg123_1
        buf162 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_14, view_23, queries_15, keys_14, view_24, keys_15, values_14, view_25, values_15, attn_output_28], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2:61
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2.run(buf162, 23716, stream=stream0)
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_14, view_23, queries_15, keys_14, view_24, keys_15, values_14, view_25, values_15, attn_output_28], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        buf163 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf159, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf160, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf161, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf162, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
        buf164 = buf163[0]
        assert_size_stride(buf164, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf164, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf163
        buf168 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [transpose_31, reshape_7, attn_output_31, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_transpose_view_3.run(buf164, arg124_1, buf168, 60, 1, 1, stream=stream0)
        del arg124_1
        buf172 = reinterpret_tensor(buf164, (4, 77, 768), (59136, 768, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_31, hidden_states_50, hidden_states_51], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:63
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf154, buf168, arg125_1, arg126_1, arg127_1, buf172, 308, 768, stream=stream0)
        del arg126_1
        del arg127_1
        buf174 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_31, hidden_states_50, hidden_states_51, hidden_states_52, mul_14, sigmoid_7, hidden_states_53], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t, aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_tem_fused_add_addmm_mul_native_layer_norm_sigmoid_t_view_5.run(buf172, arg128_1, arg129_1, buf174, 120, 1, 1, stream=stream0)
        del arg128_1
        del arg129_1
        buf175 = reinterpret_tensor(buf172, (308, 768), (768, 1), 0); del buf172  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.mm:107
        extern_kernels.mm(reinterpret_tensor(buf174, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg130_1, (3072, 768), (1, 3072), 0), out=buf175)
        del arg130_1
        buf176 = reinterpret_tensor(buf168, (4, 77, 768), (59136, 768, 1), 0); del buf168  # reuse
        buf180 = reinterpret_tensor(buf160, (4, 77, 768), (59136, 768, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_31, hidden_states_50, hidden_states_54, hidden_states_55, hidden_states_56], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_6:65
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf176, buf154, arg125_1, buf175, arg131_1, arg132_1, arg133_1, buf180, 308, 768, stream=stream0)
        del arg125_1
        del arg131_1
        del arg132_1
        del arg133_1
        buf181 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [queries_16], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg135_1, buf180, arg134_1, buf181, 60, 1, 1, stream=stream0)
        del arg134_1
        del arg135_1
        buf182 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [keys_16], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg137_1, buf180, arg136_1, buf182, 60, 1, 1, stream=stream0)
        del arg136_1
        del arg137_1
        buf183 = reinterpret_tensor(buf158, (308, 768), (768, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [values_16], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg139_1, buf180, arg138_1, buf183, 60, 1, 1, stream=stream0)
        del arg138_1
        del arg139_1
        buf184 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_16, view_26, queries_17, keys_16, view_27, keys_17, values_16, view_28, values_17, attn_output_32], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2:69
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2.run(buf184, 23716, stream=stream0)
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_16, view_26, queries_17, keys_16, view_27, keys_17, values_16, view_28, values_17, attn_output_32], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        buf185 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf181, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf182, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf183, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf184, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
        del buf184
        buf186 = buf185[0]
        assert_size_stride(buf186, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf186, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf185
        buf190 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [transpose_35, reshape_8, attn_output_35, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_transpose_view_3.run(buf186, arg140_1, buf190, 60, 1, 1, stream=stream0)
        del arg140_1
        buf194 = reinterpret_tensor(buf186, (4, 77, 768), (59136, 768, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_35, hidden_states_57, hidden_states_58], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:71
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf176, buf190, arg141_1, arg142_1, arg143_1, buf194, 308, 768, stream=stream0)
        del arg142_1
        del arg143_1
        buf196 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_35, hidden_states_57, hidden_states_58, hidden_states_59, mul_16, sigmoid_8, hidden_states_60], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t, aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_tem_fused_add_addmm_mul_native_layer_norm_sigmoid_t_view_5.run(buf194, arg144_1, arg145_1, buf196, 120, 1, 1, stream=stream0)
        del arg144_1
        del arg145_1
        buf197 = reinterpret_tensor(buf194, (308, 768), (768, 1), 0); del buf194  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.mm:108
        extern_kernels.mm(reinterpret_tensor(buf196, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg146_1, (3072, 768), (1, 3072), 0), out=buf197)
        del arg146_1
        buf198 = reinterpret_tensor(buf190, (4, 77, 768), (59136, 768, 1), 0); del buf190  # reuse
        buf202 = reinterpret_tensor(buf182, (4, 77, 768), (59136, 768, 1), 0); del buf182  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_35, hidden_states_57, hidden_states_61, hidden_states_62, hidden_states_63], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_6:73
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf198, buf176, arg141_1, buf197, arg147_1, arg148_1, arg149_1, buf202, 308, 768, stream=stream0)
        del arg141_1
        del arg147_1
        del arg148_1
        del arg149_1
        buf203 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [queries_18], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg151_1, buf202, arg150_1, buf203, 60, 1, 1, stream=stream0)
        del arg150_1
        del arg151_1
        buf204 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [keys_18], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg153_1, buf202, arg152_1, buf204, 60, 1, 1, stream=stream0)
        del arg152_1
        del arg153_1
        buf205 = reinterpret_tensor(buf180, (308, 768), (768, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [values_18], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg155_1, buf202, arg154_1, buf205, 60, 1, 1, stream=stream0)
        del arg154_1
        del arg155_1
        del buf202
        buf206 = empty_strided_cuda((4, 1, 77, 77), (6160, 0, 80, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_18, view_29, queries_19, keys_18, view_30, keys_19, values_18, view_31, values_19, attn_output_36], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2:77
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2.run(buf206, 23716, stream=stream0)
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_18, view_29, queries_19, keys_18, view_30, keys_19, values_18, view_31, values_19, attn_output_36], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        buf207 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf203, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf204, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf205, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf206, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
        del buf206
        buf208 = buf207[0]
        assert_size_stride(buf208, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf208, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf207
        buf212 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [transpose_39, reshape_9, attn_output_39, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_transpose_view_3.run(buf208, arg156_1, buf212, 60, 1, 1, stream=stream0)
        del arg156_1
        buf216 = reinterpret_tensor(buf208, (4, 77, 768), (59136, 768, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_39, hidden_states_64, hidden_states_65], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:79
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf198, buf212, arg157_1, arg158_1, arg159_1, buf216, 308, 768, stream=stream0)
        del arg158_1
        del arg159_1
        buf218 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_39, hidden_states_64, hidden_states_65, hidden_states_66, mul_18, sigmoid_9, hidden_states_67], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t, aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_tem_fused_add_addmm_mul_native_layer_norm_sigmoid_t_view_5.run(buf216, arg160_1, arg161_1, buf218, 120, 1, 1, stream=stream0)
        del arg160_1
        del arg161_1
        buf219 = reinterpret_tensor(buf216, (308, 768), (768, 1), 0); del buf216  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.mm:109
        extern_kernels.mm(reinterpret_tensor(buf218, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg162_1, (3072, 768), (1, 3072), 0), out=buf219)
        del arg162_1
        buf220 = reinterpret_tensor(buf212, (4, 77, 768), (59136, 768, 1), 0); del buf212  # reuse
        buf224 = reinterpret_tensor(buf204, (4, 77, 768), (59136, 768, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_39, hidden_states_64, hidden_states_68, hidden_states_69, hidden_states_70], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_6:81
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf220, buf198, arg157_1, buf219, arg163_1, arg164_1, arg165_1, buf224, 308, 768, stream=stream0)
        del arg157_1
        del arg163_1
        del arg164_1
        del arg165_1
        buf225 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [queries_20], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg167_1, buf224, arg166_1, buf225, 60, 1, 1, stream=stream0)
        del arg166_1
        del arg167_1
        buf226 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [keys_20], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg169_1, buf224, arg168_1, buf226, 60, 1, 1, stream=stream0)
        del arg168_1
        del arg169_1
        buf227 = empty_strided_cuda((308, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [values_20], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg171_1, buf224, arg170_1, buf227, 60, 1, 1, stream=stream0)
        del arg170_1
        del arg171_1
        del buf224
        buf228 = empty_strided_cuda((4, 1, 77, 77), (6160, 0, 80, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_20, view_32, queries_21, keys_20, view_33, keys_21, values_20, view_34, values_21, attn_output_40], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2:85
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2.run(buf228, 23716, stream=stream0)
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_20, view_32, queries_21, keys_20, view_33, keys_21, values_20, view_34, values_21, attn_output_40], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        buf229 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf225, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf226, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf227, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf228, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
        del buf225
        del buf228
        buf230 = buf229[0]
        assert_size_stride(buf230, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf230, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf229
        buf234 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [transpose_43, reshape_10, attn_output_43, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_transpose_view_3.run(buf230, arg172_1, buf234, 60, 1, 1, stream=stream0)
        del arg172_1
        buf238 = reinterpret_tensor(buf230, (4, 77, 768), (59136, 768, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_43, hidden_states_71, hidden_states_72], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:87
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf220, buf234, arg173_1, arg174_1, arg175_1, buf238, 308, 768, stream=stream0)
        del arg174_1
        del arg175_1
        buf240 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_43, hidden_states_71, hidden_states_72, hidden_states_73, mul_20, sigmoid_10, hidden_states_74], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t, aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_tem_fused_add_addmm_mul_native_layer_norm_sigmoid_t_view_5.run(buf238, arg176_1, arg177_1, buf240, 120, 1, 1, stream=stream0)
        del arg176_1
        del arg177_1
        buf241 = reinterpret_tensor(buf238, (308, 768), (768, 1), 0); del buf238  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.mm:110
        extern_kernels.mm(reinterpret_tensor(buf240, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg178_1, (3072, 768), (1, 3072), 0), out=buf241)
        del arg178_1
        buf242 = reinterpret_tensor(buf234, (4, 77, 768), (59136, 768, 1), 0); del buf234  # reuse
        buf246 = reinterpret_tensor(buf226, (4, 77, 768), (59136, 768, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_43, hidden_states_71, hidden_states_75, hidden_states_76, hidden_states_77], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_6:89
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf242, buf220, arg173_1, buf241, arg179_1, arg180_1, arg181_1, buf246, 308, 768, stream=stream0)
        del arg173_1
        del arg179_1
        del arg180_1
        del arg181_1
        buf247 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [queries_22], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg183_1, buf246, arg182_1, buf247, 60, 1, 1, stream=stream0)
        del arg182_1
        del arg183_1
        buf248 = empty_strided_cuda((308, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [keys_22], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg185_1, buf246, arg184_1, buf248, 60, 1, 1, stream=stream0)
        del arg184_1
        del arg185_1
        buf249 = empty_strided_cuda((308, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [values_22], Original ATen: [aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_view_1.run(arg187_1, buf246, arg186_1, buf249, 60, 1, 1, stream=stream0)
        del arg186_1
        del arg187_1
        del buf246
        buf250 = empty_strided_cuda((4, 1, 77, 77), (6160, 0, 80, 1), torch.float16)
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_22, view_35, queries_23, keys_22, view_36, keys_23, values_22, view_37, values_23, attn_output_44], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2:93
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_2.run(buf250, 23716, stream=stream0)
        # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_22, view_35, queries_23, keys_22, view_36, keys_23, values_22, view_37, values_23, attn_output_44], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
        buf251 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf247, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf248, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf249, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf250, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
        del buf247
        del buf248
        del buf250
        buf252 = buf251[0]
        assert_size_stride(buf252, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf252, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf251
        buf256 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [transpose_47, reshape_11, attn_output_47, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_t_transpose_view_3.run(buf252, arg188_1, buf256, 60, 1, 1, stream=stream0)
        del arg188_1
        buf260 = reinterpret_tensor(buf252, (4, 77, 768), (59136, 768, 1), 0); del buf252  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_47, hidden_states_78, hidden_states_79], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:95
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf242, buf256, arg189_1, arg190_1, arg191_1, buf260, 308, 768, stream=stream0)
        del arg190_1
        del arg191_1
        buf262 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [, attn_output_47, hidden_states_78, hidden_states_79, hidden_states_80, mul_22, sigmoid_11, hidden_states_81], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t, aten.mul, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_tem_fused_add_addmm_mul_native_layer_norm_sigmoid_t_view_5.run(buf260, arg192_1, arg193_1, buf262, 120, 1, 1, stream=stream0)
        del arg192_1
        del arg193_1
        buf263 = reinterpret_tensor(buf260, (308, 768), (768, 1), 0); del buf260  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.mm:111
        extern_kernels.mm(reinterpret_tensor(buf262, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg194_1, (3072, 768), (1, 3072), 0), out=buf263)
        del arg194_1
        del buf262
        buf264 = reinterpret_tensor(buf256, (4, 77, 768), (59136, 768, 1), 0); del buf256  # reuse
        buf268 = empty_strided_cuda((4, 77, 768), (59136, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [, attn_output_47, hidden_states_78, hidden_states_82, hidden_states_83, last_hidden_state], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_6:97
        stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf264, buf242, arg189_1, buf263, arg195_1, arg196_1, arg197_1, buf268, 308, 768, stream=stream0)
        del arg189_1
        del arg195_1
        del arg196_1
        del arg197_1
        del buf263
        buf269 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [to_1, argmax], Original ATen: [aten._to_copy, aten.argmax]
        # [Provenance debug handles] triton_per_fused__to_copy_argmax_7:98
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_argmax_7.run(arg0_1, buf269, 4, 77, stream=stream0)
        del arg0_1
        buf270 = empty_strided_cuda((4, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [arange_1, pooled_output], Original ATen: [aten.arange, aten.index]
        # [Provenance debug handles] triton_poi_fused_arange_index_8:99
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_index_8.run(buf269, buf268, buf270, 3072, stream=stream0)
        del buf269
    return (buf0, buf22, buf44, buf66, buf88, buf110, buf132, buf154, buf176, buf198, buf220, buf242, buf264, buf268, buf270, )


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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1 = args
        args.clear()
        partition0_args = [arg0_1, arg3_1, arg2_1, arg1_1, arg4_1, arg5_1, arg7_1, arg6_1, arg9_1, arg8_1, arg11_1, arg10_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg23_1, arg22_1, arg25_1, arg24_1, arg27_1, arg26_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg39_1, arg38_1, arg41_1, arg40_1, arg43_1, arg42_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg55_1, arg54_1, arg57_1, arg56_1, arg59_1, arg58_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg71_1, arg70_1, arg73_1, arg72_1, arg75_1, arg74_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg87_1, arg86_1, arg89_1, arg88_1, arg91_1, arg90_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg103_1, arg102_1, arg105_1, arg104_1, arg107_1, arg106_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg119_1, arg118_1, arg121_1, arg120_1, arg123_1, arg122_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg135_1, arg134_1, arg137_1, arg136_1, arg139_1, arg138_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg151_1, arg150_1, arg153_1, arg152_1, arg155_1, arg154_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg167_1, arg166_1, arg169_1, arg168_1, arg171_1, arg170_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg183_1, arg182_1, arg185_1, arg184_1, arg187_1, arg186_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1]
        del arg0_1, arg3_1, arg2_1, arg1_1, arg4_1, arg5_1, arg7_1, arg6_1, arg9_1, arg8_1, arg11_1, arg10_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg23_1, arg22_1, arg25_1, arg24_1, arg27_1, arg26_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg39_1, arg38_1, arg41_1, arg40_1, arg43_1, arg42_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg55_1, arg54_1, arg57_1, arg56_1, arg59_1, arg58_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg71_1, arg70_1, arg73_1, arg72_1, arg75_1, arg74_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg87_1, arg86_1, arg89_1, arg88_1, arg91_1, arg90_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg103_1, arg102_1, arg105_1, arg104_1, arg107_1, arg106_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg119_1, arg118_1, arg121_1, arg120_1, arg123_1, arg122_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg135_1, arg134_1, arg137_1, arg136_1, arg139_1, arg138_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg151_1, arg150_1, arg153_1, arg152_1, arg155_1, arg154_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg167_1, arg166_1, arg169_1, arg168_1, arg171_1, arg170_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg183_1, arg182_1, arg185_1, arg184_1, arg187_1, arg186_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1
        (buf0, buf22, buf44, buf66, buf88, buf110, buf132, buf154, buf176, buf198, buf220, buf242, buf264, buf268, buf270) = self.partitions[0](partition0_args)
        del partition0_args
        return (buf268, buf270, buf0, buf22, buf44, buf66, buf88, buf110, buf132, buf154, buf176, buf198, buf220, buf242, buf264, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 77), (77, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((77, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((1, 77), (77, 1), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((49408, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg8_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg10_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg12_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg16_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg17_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg18_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg22_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg24_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg25_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg26_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg28_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg32_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg33_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg34_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg36_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg38_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg40_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg42_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg44_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg48_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg49_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg50_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg54_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg56_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg58_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg60_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg64_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg65_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg66_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg70_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg72_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg74_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg76_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg79_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg80_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg81_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg82_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg84_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg85_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg86_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg88_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg90_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg91_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg92_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg96_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg97_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg98_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg102_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg104_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg106_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg108_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg112_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg113_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg114_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg116_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg118_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg120_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg122_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg124_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg128_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg129_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg130_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg132_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg134_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg136_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg138_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg139_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg140_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg144_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg145_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg146_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg150_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg151_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg152_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg153_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg154_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg155_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg156_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg157_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg158_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg159_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg160_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg161_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg162_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    arg163_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg164_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg165_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg166_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg168_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg169_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg170_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg171_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg172_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg173_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg174_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg175_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg176_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg177_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg178_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    arg179_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg180_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg181_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg182_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg184_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg185_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg186_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg187_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg188_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg189_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg191_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg192_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg193_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg194_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    arg195_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg196_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg197_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
