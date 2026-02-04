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


# kernel path: /tmp/torchinductor_wucz/jo/cjow4e6dxix6bjudevqkawkv6k72kpfyfpdtnmvldzko2iixnim4.py
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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


# kernel path: /tmp/torchinductor_wucz/fm/cfmbn5zp4ptuyd7kzkxdpa5oy3oymflnezw3mxhzhtpyq2irr6nm.py
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
triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 94864}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/3y/c3yuyzqxyj3heryfpidj6v6hbtnx7rdjv6wnko75fa32t3z2yme2.py
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
triton_per_fused_add_addmm_native_layer_norm_view_2 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_view_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 1896960}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/7r/c7rysqb2wfnjzzjqrcv2kvqj4jcysq5dpvr652mi2kl3h466wwjc.py
# Topologically Sorted Source Nodes: [, hidden_states_3, mul, sigmoid, hidden_states_4], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid]
# Source node to ATen node mapping:
#    => add_tensor_34
#   hidden_states_3 => view_15
#   hidden_states_4 => mul_5
#   mul => mul_4
#   sigmoid => sigmoid
# Graph fragment:
#   %mm_default_34 : Tensor "f16[308, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_default_34]
#   %arg17_1 : Tensor "f16[3072][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %add_tensor_34 : Tensor "f16[308, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_34, %arg17_1), kwargs = {})
#   %view_15 : Tensor "f16[4, 77, 3072][236544, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [4, 77, 3072]), kwargs = {})
#   %mul_4 : Tensor "f16[4, 77, 3072][236544, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, 1.702), kwargs = {})
#   %sigmoid : Tensor "f16[4, 77, 3072][236544, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mul_4,), kwargs = {})
#   %mul_5 : Tensor "f16[4, 77, 3072][236544, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, %sigmoid), kwargs = {})
#   return %mul_5
triton_poi_fused_addmm_mul_sigmoid_view_3 = async_compile.triton('triton_poi_fused_addmm_mul_sigmoid_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_mul_sigmoid_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5683200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_mul_sigmoid_view_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 946176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3072)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = 1.702
    tmp4 = tmp2 * tmp3
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp2 * tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_wucz/uj/cujmlp7g4hoxd3tkyloxjvthjpspl2jzwktkzv7sheokuhjdbzbl.py
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
#   %mm_default_33 : Tensor "f16[308, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_33]
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
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 3317760}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/sw/cswynlexmrqu3b7hxdxgrpjubvlrptaoreac5lbopb6k3n6it5ru.py
# Topologically Sorted Source Nodes: [to_1, argmax], Original ATen: [aten._to_copy, aten.argmax]
# Source node to ATen node mapping:
#   argmax => argmax
#   to_1 => convert_element_type_266
# Graph fragment:
#   %arg0_1 : Tensor "i64[4, 77][77, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convert_element_type_266 : Tensor "i32[4, 77][77, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.int32), kwargs = {})
#   %argmax : Tensor "i64[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.argmax.default](args = (%convert_element_type_266, -1), kwargs = {})
#   return %argmax
triton_per_fused__to_copy_argmax_5 = async_compile.triton('triton_per_fused__to_copy_argmax_5', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_argmax_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 32, 'r0_': 2464}}
)
@triton.jit
def triton_per_fused__to_copy_argmax_5(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_wucz/kw/ckwljbfsrrfyt44bcrweclgbxf3mmdxatkzaw4jc7jvu4etf33xz.py
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
triton_poi_fused_arange_index_6 = async_compile.triton('triton_poi_fused_arange_index_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_index_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'EB2531960168FA8948CEDEB8762890B19DF9C3CEDAF023634089DCA67574673C', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_index_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        assert_size_stride(arg0_1, (4, 77), (77, 1))
        assert_size_stride(arg1_1, (77, 768), (768, 1))
        assert_size_stride(arg2_1, (1, 77), (77, 1))
        assert_size_stride(arg3_1, (49408, 768), (768, 1))
        assert_size_stride(arg4_1, (768, ), (1, ))
        assert_size_stride(arg5_1, (768, ), (1, ))
        assert_size_stride(arg6_1, (768, 768), (768, 1))
        assert_size_stride(arg7_1, (768, ), (1, ))
        assert_size_stride(arg8_1, (768, 768), (768, 1))
        assert_size_stride(arg9_1, (768, ), (1, ))
        assert_size_stride(arg10_1, (768, 768), (768, 1))
        assert_size_stride(arg11_1, (768, ), (1, ))
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
        assert_size_stride(arg22_1, (768, 768), (768, 1))
        assert_size_stride(arg23_1, (768, ), (1, ))
        assert_size_stride(arg24_1, (768, 768), (768, 1))
        assert_size_stride(arg25_1, (768, ), (1, ))
        assert_size_stride(arg26_1, (768, 768), (768, 1))
        assert_size_stride(arg27_1, (768, ), (1, ))
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
        assert_size_stride(arg38_1, (768, 768), (768, 1))
        assert_size_stride(arg39_1, (768, ), (1, ))
        assert_size_stride(arg40_1, (768, 768), (768, 1))
        assert_size_stride(arg41_1, (768, ), (1, ))
        assert_size_stride(arg42_1, (768, 768), (768, 1))
        assert_size_stride(arg43_1, (768, ), (1, ))
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
        assert_size_stride(arg54_1, (768, 768), (768, 1))
        assert_size_stride(arg55_1, (768, ), (1, ))
        assert_size_stride(arg56_1, (768, 768), (768, 1))
        assert_size_stride(arg57_1, (768, ), (1, ))
        assert_size_stride(arg58_1, (768, 768), (768, 1))
        assert_size_stride(arg59_1, (768, ), (1, ))
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
        assert_size_stride(arg70_1, (768, 768), (768, 1))
        assert_size_stride(arg71_1, (768, ), (1, ))
        assert_size_stride(arg72_1, (768, 768), (768, 1))
        assert_size_stride(arg73_1, (768, ), (1, ))
        assert_size_stride(arg74_1, (768, 768), (768, 1))
        assert_size_stride(arg75_1, (768, ), (1, ))
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
        assert_size_stride(arg86_1, (768, 768), (768, 1))
        assert_size_stride(arg87_1, (768, ), (1, ))
        assert_size_stride(arg88_1, (768, 768), (768, 1))
        assert_size_stride(arg89_1, (768, ), (1, ))
        assert_size_stride(arg90_1, (768, 768), (768, 1))
        assert_size_stride(arg91_1, (768, ), (1, ))
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
        assert_size_stride(arg102_1, (768, 768), (768, 1))
        assert_size_stride(arg103_1, (768, ), (1, ))
        assert_size_stride(arg104_1, (768, 768), (768, 1))
        assert_size_stride(arg105_1, (768, ), (1, ))
        assert_size_stride(arg106_1, (768, 768), (768, 1))
        assert_size_stride(arg107_1, (768, ), (1, ))
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
        assert_size_stride(arg118_1, (768, 768), (768, 1))
        assert_size_stride(arg119_1, (768, ), (1, ))
        assert_size_stride(arg120_1, (768, 768), (768, 1))
        assert_size_stride(arg121_1, (768, ), (1, ))
        assert_size_stride(arg122_1, (768, 768), (768, 1))
        assert_size_stride(arg123_1, (768, ), (1, ))
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
        assert_size_stride(arg134_1, (768, 768), (768, 1))
        assert_size_stride(arg135_1, (768, ), (1, ))
        assert_size_stride(arg136_1, (768, 768), (768, 1))
        assert_size_stride(arg137_1, (768, ), (1, ))
        assert_size_stride(arg138_1, (768, 768), (768, 1))
        assert_size_stride(arg139_1, (768, ), (1, ))
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
        assert_size_stride(arg150_1, (768, 768), (768, 1))
        assert_size_stride(arg151_1, (768, ), (1, ))
        assert_size_stride(arg152_1, (768, 768), (768, 1))
        assert_size_stride(arg153_1, (768, ), (1, ))
        assert_size_stride(arg154_1, (768, 768), (768, 1))
        assert_size_stride(arg155_1, (768, ), (1, ))
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
        assert_size_stride(arg166_1, (768, 768), (768, 1))
        assert_size_stride(arg167_1, (768, ), (1, ))
        assert_size_stride(arg168_1, (768, 768), (768, 1))
        assert_size_stride(arg169_1, (768, ), (1, ))
        assert_size_stride(arg170_1, (768, 768), (768, 1))
        assert_size_stride(arg171_1, (768, ), (1, ))
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
        assert_size_stride(arg182_1, (768, 768), (768, 1))
        assert_size_stride(arg183_1, (768, ), (1, ))
        assert_size_stride(arg184_1, (768, 768), (768, 1))
        assert_size_stride(arg185_1, (768, ), (1, ))
        assert_size_stride(arg186_1, (768, 768), (768, 1))
        assert_size_stride(arg187_1, (768, ), (1, ))
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
            # [Provenance debug handles] extern_kernels.addmm:52
            extern_kernels.addmm(arg7_1, reinterpret_tensor(buf4, (308, 768), (768, 1), 0), reinterpret_tensor(arg6_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf5)
            del arg6_1
            del arg7_1
            buf6 = empty_strided_cuda((308, 768), (768, 1), torch.float16)
            # Topologically Sorted Source Nodes: [keys], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:53
            extern_kernels.addmm(arg9_1, reinterpret_tensor(buf4, (308, 768), (768, 1), 0), reinterpret_tensor(arg8_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf6)
            del arg8_1
            del arg9_1
            buf7 = empty_strided_cuda((308, 768), (768, 1), torch.float16)
            # Topologically Sorted Source Nodes: [values], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:54
            extern_kernels.addmm(arg11_1, reinterpret_tensor(buf4, (308, 768), (768, 1), 0), reinterpret_tensor(arg10_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf7)
            del arg10_1
            del arg11_1
            buf8 = empty_strided_cuda((4, 1, 77, 77), (6160, 0, 80, 1), torch.float16)
            # Topologically Sorted Source Nodes: [queries, view_2, queries_1, keys, view_3, keys_1, values, view_4, values_1, mask_cond, add_1, view_1, lt, masked_fill_, mask, attn_output], Original ATen: [aten.view, aten.transpose, aten.arange, aten.add, aten.lt, aten.masked_fill, aten.full, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1:2
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1.run(buf8, 23716, stream=stream0)
            # Topologically Sorted Source Nodes: [queries, view_2, queries_1, keys, view_3, keys_1, values, view_4, values_1, mask_cond, add_1, view_1, lt, masked_fill_, mask, attn_output], Original ATen: [aten.view, aten.transpose, aten.arange, aten.add, aten.lt, aten.masked_fill, aten.full, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            buf9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf5, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf6, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf7, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf8, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
            buf10 = buf9[0]
            assert_size_stride(buf10, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf10, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf9
            buf14 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [transpose_3, reshape, attn_output_3, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:55
            extern_kernels.mm(reinterpret_tensor(buf10, (308, 768), (768, 1), 0), reinterpret_tensor(arg12_1, (768, 768), (1, 768), 0), out=buf14)
            del arg12_1
            buf18 = reinterpret_tensor(buf10, (4, 77, 768), (59136, 768, 1), 0); del buf10  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_3, hidden_states_1, hidden_states_2], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:3
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf0, buf14, arg13_1, arg14_1, arg15_1, buf18, 308, 768, stream=stream0)
            del arg14_1
            del arg15_1
            buf19 = empty_strided_cuda((308, 3072), (3072, 1), torch.float16)
            # Topologically Sorted Source Nodes: [, attn_output_3, hidden_states_1, hidden_states_2, hidden_states_3], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            # [Provenance debug handles] extern_kernels.mm:56
            extern_kernels.mm(reinterpret_tensor(buf18, (308, 768), (768, 1), 0), reinterpret_tensor(arg16_1, (768, 3072), (1, 768), 0), out=buf19)
            del arg16_1
            buf20 = reinterpret_tensor(buf19, (4, 77, 3072), (236544, 3072, 1), 0); del buf19  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_3, mul, sigmoid, hidden_states_4], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid]
            # [Provenance debug handles] triton_poi_fused_addmm_mul_sigmoid_view_3:4
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_mul_sigmoid_view_3.run(buf20, arg17_1, 946176, stream=stream0)
            del arg17_1
            buf21 = reinterpret_tensor(buf18, (308, 768), (768, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_3, mul, sigmoid, hidden_states_4, hidden_states_5], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid, aten.t]
            # [Provenance debug handles] extern_kernels.mm:57
            extern_kernels.mm(reinterpret_tensor(buf20, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg18_1, (3072, 768), (1, 3072), 0), out=buf21)
            del arg18_1
            buf22 = reinterpret_tensor(buf14, (4, 77, 768), (59136, 768, 1), 0); del buf14  # reuse
            buf26 = reinterpret_tensor(buf6, (4, 77, 768), (59136, 768, 1), 0); del buf6  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_3, hidden_states_1, hidden_states_5, hidden_states_6, hidden_states_7], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:5
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf22, buf0, arg13_1, buf21, arg19_1, arg20_1, arg21_1, buf26, 308, 768, stream=stream0)
            del arg13_1
            del arg19_1
            del arg20_1
            del arg21_1
            buf27 = buf21; del buf21  # reuse
            # Topologically Sorted Source Nodes: [queries_2], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:58
            extern_kernels.addmm(arg23_1, reinterpret_tensor(buf26, (308, 768), (768, 1), 0), reinterpret_tensor(arg22_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf27)
            del arg22_1
            del arg23_1
            buf28 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [keys_2], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:59
            extern_kernels.addmm(arg25_1, reinterpret_tensor(buf26, (308, 768), (768, 1), 0), reinterpret_tensor(arg24_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf28)
            del arg24_1
            del arg25_1
            buf29 = reinterpret_tensor(buf4, (308, 768), (768, 1), 0); del buf4  # reuse
            # Topologically Sorted Source Nodes: [values_2], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:60
            extern_kernels.addmm(arg27_1, reinterpret_tensor(buf26, (308, 768), (768, 1), 0), reinterpret_tensor(arg26_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf29)
            del arg26_1
            del arg27_1
            buf30 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_2, view_5, queries_3, keys_2, view_6, keys_3, values_2, view_7, values_3, attn_output_4], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1:6
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1.run(buf30, 23716, stream=stream0)
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_2, view_5, queries_3, keys_2, view_6, keys_3, values_2, view_7, values_3, attn_output_4], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            buf31 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf27, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf28, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf29, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf30, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
            buf32 = buf31[0]
            assert_size_stride(buf32, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf32, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf31
            buf36 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [transpose_7, reshape_1, attn_output_7, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:61
            extern_kernels.mm(reinterpret_tensor(buf32, (308, 768), (768, 1), 0), reinterpret_tensor(arg28_1, (768, 768), (1, 768), 0), out=buf36)
            del arg28_1
            buf40 = reinterpret_tensor(buf32, (4, 77, 768), (59136, 768, 1), 0); del buf32  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_7, hidden_states_8, hidden_states_9], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:7
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf22, buf36, arg29_1, arg30_1, arg31_1, buf40, 308, 768, stream=stream0)
            del arg30_1
            del arg31_1
            buf41 = reinterpret_tensor(buf20, (308, 3072), (3072, 1), 0); del buf20  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_7, hidden_states_8, hidden_states_9, hidden_states_10], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            # [Provenance debug handles] extern_kernels.mm:62
            extern_kernels.mm(reinterpret_tensor(buf40, (308, 768), (768, 1), 0), reinterpret_tensor(arg32_1, (768, 3072), (1, 768), 0), out=buf41)
            del arg32_1
            buf42 = reinterpret_tensor(buf41, (4, 77, 3072), (236544, 3072, 1), 0); del buf41  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_10, mul_2, sigmoid_1, hidden_states_11], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid]
            # [Provenance debug handles] triton_poi_fused_addmm_mul_sigmoid_view_3:8
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_mul_sigmoid_view_3.run(buf42, arg33_1, 946176, stream=stream0)
            del arg33_1
            buf43 = reinterpret_tensor(buf40, (308, 768), (768, 1), 0); del buf40  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_10, mul_2, sigmoid_1, hidden_states_11, hidden_states_12], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid, aten.t]
            # [Provenance debug handles] extern_kernels.mm:63
            extern_kernels.mm(reinterpret_tensor(buf42, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg34_1, (3072, 768), (1, 3072), 0), out=buf43)
            del arg34_1
            buf44 = reinterpret_tensor(buf36, (4, 77, 768), (59136, 768, 1), 0); del buf36  # reuse
            buf48 = reinterpret_tensor(buf28, (4, 77, 768), (59136, 768, 1), 0); del buf28  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_7, hidden_states_8, hidden_states_12, hidden_states_13, hidden_states_14], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:9
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf44, buf22, arg29_1, buf43, arg35_1, arg36_1, arg37_1, buf48, 308, 768, stream=stream0)
            del arg29_1
            del arg35_1
            del arg36_1
            del arg37_1
            buf49 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [queries_4], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:64
            extern_kernels.addmm(arg39_1, reinterpret_tensor(buf48, (308, 768), (768, 1), 0), reinterpret_tensor(arg38_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf49)
            del arg38_1
            del arg39_1
            buf50 = buf27; del buf27  # reuse
            # Topologically Sorted Source Nodes: [keys_4], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:65
            extern_kernels.addmm(arg41_1, reinterpret_tensor(buf48, (308, 768), (768, 1), 0), reinterpret_tensor(arg40_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf50)
            del arg40_1
            del arg41_1
            buf51 = reinterpret_tensor(buf26, (308, 768), (768, 1), 0); del buf26  # reuse
            # Topologically Sorted Source Nodes: [values_4], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:66
            extern_kernels.addmm(arg43_1, reinterpret_tensor(buf48, (308, 768), (768, 1), 0), reinterpret_tensor(arg42_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf51)
            del arg42_1
            del arg43_1
            buf52 = buf30; del buf30  # reuse
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_4, view_8, queries_5, keys_4, view_9, keys_5, values_4, view_10, values_5, attn_output_8], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1:10
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1.run(buf52, 23716, stream=stream0)
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_4, view_8, queries_5, keys_4, view_9, keys_5, values_4, view_10, values_5, attn_output_8], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            buf53 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf49, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf50, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf51, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf52, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
            buf54 = buf53[0]
            assert_size_stride(buf54, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf54, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf53
            buf58 = buf51; del buf51  # reuse
            # Topologically Sorted Source Nodes: [transpose_11, reshape_2, attn_output_11, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:67
            extern_kernels.mm(reinterpret_tensor(buf54, (308, 768), (768, 1), 0), reinterpret_tensor(arg44_1, (768, 768), (1, 768), 0), out=buf58)
            del arg44_1
            buf62 = reinterpret_tensor(buf54, (4, 77, 768), (59136, 768, 1), 0); del buf54  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_11, hidden_states_15, hidden_states_16], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:11
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf44, buf58, arg45_1, arg46_1, arg47_1, buf62, 308, 768, stream=stream0)
            del arg46_1
            del arg47_1
            buf63 = reinterpret_tensor(buf42, (308, 3072), (3072, 1), 0); del buf42  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_11, hidden_states_15, hidden_states_16, hidden_states_17], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            # [Provenance debug handles] extern_kernels.mm:68
            extern_kernels.mm(reinterpret_tensor(buf62, (308, 768), (768, 1), 0), reinterpret_tensor(arg48_1, (768, 3072), (1, 768), 0), out=buf63)
            del arg48_1
            buf64 = reinterpret_tensor(buf63, (4, 77, 3072), (236544, 3072, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_17, mul_4, sigmoid_2, hidden_states_18], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid]
            # [Provenance debug handles] triton_poi_fused_addmm_mul_sigmoid_view_3:12
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_mul_sigmoid_view_3.run(buf64, arg49_1, 946176, stream=stream0)
            del arg49_1
            buf65 = reinterpret_tensor(buf62, (308, 768), (768, 1), 0); del buf62  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_17, mul_4, sigmoid_2, hidden_states_18, hidden_states_19], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid, aten.t]
            # [Provenance debug handles] extern_kernels.mm:69
            extern_kernels.mm(reinterpret_tensor(buf64, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg50_1, (3072, 768), (1, 3072), 0), out=buf65)
            del arg50_1
            buf66 = reinterpret_tensor(buf58, (4, 77, 768), (59136, 768, 1), 0); del buf58  # reuse
            buf70 = reinterpret_tensor(buf50, (4, 77, 768), (59136, 768, 1), 0); del buf50  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_11, hidden_states_15, hidden_states_19, hidden_states_20, hidden_states_21], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:13
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf66, buf44, arg45_1, buf65, arg51_1, arg52_1, arg53_1, buf70, 308, 768, stream=stream0)
            del arg45_1
            del arg51_1
            del arg52_1
            del arg53_1
            buf71 = buf65; del buf65  # reuse
            # Topologically Sorted Source Nodes: [queries_6], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:70
            extern_kernels.addmm(arg55_1, reinterpret_tensor(buf70, (308, 768), (768, 1), 0), reinterpret_tensor(arg54_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf71)
            del arg54_1
            del arg55_1
            buf72 = buf49; del buf49  # reuse
            # Topologically Sorted Source Nodes: [keys_6], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:71
            extern_kernels.addmm(arg57_1, reinterpret_tensor(buf70, (308, 768), (768, 1), 0), reinterpret_tensor(arg56_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf72)
            del arg56_1
            del arg57_1
            buf73 = reinterpret_tensor(buf48, (308, 768), (768, 1), 0); del buf48  # reuse
            # Topologically Sorted Source Nodes: [values_6], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:72
            extern_kernels.addmm(arg59_1, reinterpret_tensor(buf70, (308, 768), (768, 1), 0), reinterpret_tensor(arg58_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf73)
            del arg58_1
            del arg59_1
            buf74 = buf52; del buf52  # reuse
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_6, view_11, queries_7, keys_6, view_12, keys_7, values_6, view_13, values_7, attn_output_12], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1:14
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1.run(buf74, 23716, stream=stream0)
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_6, view_11, queries_7, keys_6, view_12, keys_7, values_6, view_13, values_7, attn_output_12], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            buf75 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf71, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf72, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf73, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf74, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
            buf76 = buf75[0]
            assert_size_stride(buf76, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf76, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf75
            buf80 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [transpose_15, reshape_3, attn_output_15, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:73
            extern_kernels.mm(reinterpret_tensor(buf76, (308, 768), (768, 1), 0), reinterpret_tensor(arg60_1, (768, 768), (1, 768), 0), out=buf80)
            del arg60_1
            buf84 = reinterpret_tensor(buf76, (4, 77, 768), (59136, 768, 1), 0); del buf76  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_15, hidden_states_22, hidden_states_23], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:15
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf66, buf80, arg61_1, arg62_1, arg63_1, buf84, 308, 768, stream=stream0)
            del arg62_1
            del arg63_1
            buf85 = reinterpret_tensor(buf64, (308, 3072), (3072, 1), 0); del buf64  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_15, hidden_states_22, hidden_states_23, hidden_states_24], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            # [Provenance debug handles] extern_kernels.mm:74
            extern_kernels.mm(reinterpret_tensor(buf84, (308, 768), (768, 1), 0), reinterpret_tensor(arg64_1, (768, 3072), (1, 768), 0), out=buf85)
            del arg64_1
            buf86 = reinterpret_tensor(buf85, (4, 77, 3072), (236544, 3072, 1), 0); del buf85  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_24, mul_6, sigmoid_3, hidden_states_25], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid]
            # [Provenance debug handles] triton_poi_fused_addmm_mul_sigmoid_view_3:16
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_mul_sigmoid_view_3.run(buf86, arg65_1, 946176, stream=stream0)
            del arg65_1
            buf87 = reinterpret_tensor(buf84, (308, 768), (768, 1), 0); del buf84  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_24, mul_6, sigmoid_3, hidden_states_25, hidden_states_26], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid, aten.t]
            # [Provenance debug handles] extern_kernels.mm:75
            extern_kernels.mm(reinterpret_tensor(buf86, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg66_1, (3072, 768), (1, 3072), 0), out=buf87)
            del arg66_1
            buf88 = reinterpret_tensor(buf80, (4, 77, 768), (59136, 768, 1), 0); del buf80  # reuse
            buf92 = reinterpret_tensor(buf72, (4, 77, 768), (59136, 768, 1), 0); del buf72  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_15, hidden_states_22, hidden_states_26, hidden_states_27, hidden_states_28], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:17
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf88, buf66, arg61_1, buf87, arg67_1, arg68_1, arg69_1, buf92, 308, 768, stream=stream0)
            del arg61_1
            del arg67_1
            del arg68_1
            del arg69_1
            buf93 = buf87; del buf87  # reuse
            # Topologically Sorted Source Nodes: [queries_8], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:76
            extern_kernels.addmm(arg71_1, reinterpret_tensor(buf92, (308, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf93)
            del arg70_1
            del arg71_1
            buf94 = buf71; del buf71  # reuse
            # Topologically Sorted Source Nodes: [keys_8], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:77
            extern_kernels.addmm(arg73_1, reinterpret_tensor(buf92, (308, 768), (768, 1), 0), reinterpret_tensor(arg72_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf94)
            del arg72_1
            del arg73_1
            buf95 = reinterpret_tensor(buf70, (308, 768), (768, 1), 0); del buf70  # reuse
            # Topologically Sorted Source Nodes: [values_8], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:78
            extern_kernels.addmm(arg75_1, reinterpret_tensor(buf92, (308, 768), (768, 1), 0), reinterpret_tensor(arg74_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf95)
            del arg74_1
            del arg75_1
            buf96 = buf74; del buf74  # reuse
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_8, view_14, queries_9, keys_8, view_15, keys_9, values_8, view_16, values_9, attn_output_16], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1:18
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1.run(buf96, 23716, stream=stream0)
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_8, view_14, queries_9, keys_8, view_15, keys_9, values_8, view_16, values_9, attn_output_16], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            buf97 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf93, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf94, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf95, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf96, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
            buf98 = buf97[0]
            assert_size_stride(buf98, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf98, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf97
            buf102 = buf95; del buf95  # reuse
            # Topologically Sorted Source Nodes: [transpose_19, reshape_4, attn_output_19, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:79
            extern_kernels.mm(reinterpret_tensor(buf98, (308, 768), (768, 1), 0), reinterpret_tensor(arg76_1, (768, 768), (1, 768), 0), out=buf102)
            del arg76_1
            buf106 = reinterpret_tensor(buf98, (4, 77, 768), (59136, 768, 1), 0); del buf98  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_19, hidden_states_29, hidden_states_30], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:19
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf88, buf102, arg77_1, arg78_1, arg79_1, buf106, 308, 768, stream=stream0)
            del arg78_1
            del arg79_1
            buf107 = reinterpret_tensor(buf86, (308, 3072), (3072, 1), 0); del buf86  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_19, hidden_states_29, hidden_states_30, hidden_states_31], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            # [Provenance debug handles] extern_kernels.mm:80
            extern_kernels.mm(reinterpret_tensor(buf106, (308, 768), (768, 1), 0), reinterpret_tensor(arg80_1, (768, 3072), (1, 768), 0), out=buf107)
            del arg80_1
            buf108 = reinterpret_tensor(buf107, (4, 77, 3072), (236544, 3072, 1), 0); del buf107  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_31, mul_8, sigmoid_4, hidden_states_32], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid]
            # [Provenance debug handles] triton_poi_fused_addmm_mul_sigmoid_view_3:20
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_mul_sigmoid_view_3.run(buf108, arg81_1, 946176, stream=stream0)
            del arg81_1
            buf109 = reinterpret_tensor(buf106, (308, 768), (768, 1), 0); del buf106  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_31, mul_8, sigmoid_4, hidden_states_32, hidden_states_33], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid, aten.t]
            # [Provenance debug handles] extern_kernels.mm:81
            extern_kernels.mm(reinterpret_tensor(buf108, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg82_1, (3072, 768), (1, 3072), 0), out=buf109)
            del arg82_1
            buf110 = reinterpret_tensor(buf102, (4, 77, 768), (59136, 768, 1), 0); del buf102  # reuse
            buf114 = reinterpret_tensor(buf94, (4, 77, 768), (59136, 768, 1), 0); del buf94  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_19, hidden_states_29, hidden_states_33, hidden_states_34, hidden_states_35], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:21
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf110, buf88, arg77_1, buf109, arg83_1, arg84_1, arg85_1, buf114, 308, 768, stream=stream0)
            del arg77_1
            del arg83_1
            del arg84_1
            del arg85_1
            buf115 = buf109; del buf109  # reuse
            # Topologically Sorted Source Nodes: [queries_10], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:82
            extern_kernels.addmm(arg87_1, reinterpret_tensor(buf114, (308, 768), (768, 1), 0), reinterpret_tensor(arg86_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf115)
            del arg86_1
            del arg87_1
            buf116 = buf93; del buf93  # reuse
            # Topologically Sorted Source Nodes: [keys_10], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:83
            extern_kernels.addmm(arg89_1, reinterpret_tensor(buf114, (308, 768), (768, 1), 0), reinterpret_tensor(arg88_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf116)
            del arg88_1
            del arg89_1
            buf117 = reinterpret_tensor(buf92, (308, 768), (768, 1), 0); del buf92  # reuse
            # Topologically Sorted Source Nodes: [values_10], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:84
            extern_kernels.addmm(arg91_1, reinterpret_tensor(buf114, (308, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf117)
            del arg90_1
            del arg91_1
            buf118 = buf96; del buf96  # reuse
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_10, view_17, queries_11, keys_10, view_18, keys_11, values_10, view_19, values_11, attn_output_20], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1:22
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1.run(buf118, 23716, stream=stream0)
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_10, view_17, queries_11, keys_10, view_18, keys_11, values_10, view_19, values_11, attn_output_20], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            buf119 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf115, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf116, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf117, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf118, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
            buf120 = buf119[0]
            assert_size_stride(buf120, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf120, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf119
            buf124 = buf117; del buf117  # reuse
            # Topologically Sorted Source Nodes: [transpose_23, reshape_5, attn_output_23, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:85
            extern_kernels.mm(reinterpret_tensor(buf120, (308, 768), (768, 1), 0), reinterpret_tensor(arg92_1, (768, 768), (1, 768), 0), out=buf124)
            del arg92_1
            buf128 = reinterpret_tensor(buf120, (4, 77, 768), (59136, 768, 1), 0); del buf120  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_23, hidden_states_36, hidden_states_37], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:23
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf110, buf124, arg93_1, arg94_1, arg95_1, buf128, 308, 768, stream=stream0)
            del arg94_1
            del arg95_1
            buf129 = reinterpret_tensor(buf108, (308, 3072), (3072, 1), 0); del buf108  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_23, hidden_states_36, hidden_states_37, hidden_states_38], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            # [Provenance debug handles] extern_kernels.mm:86
            extern_kernels.mm(reinterpret_tensor(buf128, (308, 768), (768, 1), 0), reinterpret_tensor(arg96_1, (768, 3072), (1, 768), 0), out=buf129)
            del arg96_1
            buf130 = reinterpret_tensor(buf129, (4, 77, 3072), (236544, 3072, 1), 0); del buf129  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_38, mul_10, sigmoid_5, hidden_states_39], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid]
            # [Provenance debug handles] triton_poi_fused_addmm_mul_sigmoid_view_3:24
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_mul_sigmoid_view_3.run(buf130, arg97_1, 946176, stream=stream0)
            del arg97_1
            buf131 = reinterpret_tensor(buf128, (308, 768), (768, 1), 0); del buf128  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_38, mul_10, sigmoid_5, hidden_states_39, hidden_states_40], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid, aten.t]
            # [Provenance debug handles] extern_kernels.mm:87
            extern_kernels.mm(reinterpret_tensor(buf130, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg98_1, (3072, 768), (1, 3072), 0), out=buf131)
            del arg98_1
            buf132 = reinterpret_tensor(buf124, (4, 77, 768), (59136, 768, 1), 0); del buf124  # reuse
            buf136 = reinterpret_tensor(buf116, (4, 77, 768), (59136, 768, 1), 0); del buf116  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_23, hidden_states_36, hidden_states_40, hidden_states_41, hidden_states_42], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:25
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf132, buf110, arg93_1, buf131, arg99_1, arg100_1, arg101_1, buf136, 308, 768, stream=stream0)
            del arg100_1
            del arg101_1
            del arg93_1
            del arg99_1
            buf137 = buf131; del buf131  # reuse
            # Topologically Sorted Source Nodes: [queries_12], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:88
            extern_kernels.addmm(arg103_1, reinterpret_tensor(buf136, (308, 768), (768, 1), 0), reinterpret_tensor(arg102_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf137)
            del arg102_1
            del arg103_1
            buf138 = buf115; del buf115  # reuse
            # Topologically Sorted Source Nodes: [keys_12], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:89
            extern_kernels.addmm(arg105_1, reinterpret_tensor(buf136, (308, 768), (768, 1), 0), reinterpret_tensor(arg104_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf138)
            del arg104_1
            del arg105_1
            buf139 = reinterpret_tensor(buf114, (308, 768), (768, 1), 0); del buf114  # reuse
            # Topologically Sorted Source Nodes: [values_12], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:90
            extern_kernels.addmm(arg107_1, reinterpret_tensor(buf136, (308, 768), (768, 1), 0), reinterpret_tensor(arg106_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf139)
            del arg106_1
            del arg107_1
            buf140 = buf118; del buf118  # reuse
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_12, view_20, queries_13, keys_12, view_21, keys_13, values_12, view_22, values_13, attn_output_24], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1:26
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1.run(buf140, 23716, stream=stream0)
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_12, view_20, queries_13, keys_12, view_21, keys_13, values_12, view_22, values_13, attn_output_24], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            buf141 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf137, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf138, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf139, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf140, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
            buf142 = buf141[0]
            assert_size_stride(buf142, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf142, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf141
            buf146 = buf139; del buf139  # reuse
            # Topologically Sorted Source Nodes: [transpose_27, reshape_6, attn_output_27, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:91
            extern_kernels.mm(reinterpret_tensor(buf142, (308, 768), (768, 1), 0), reinterpret_tensor(arg108_1, (768, 768), (1, 768), 0), out=buf146)
            del arg108_1
            buf150 = reinterpret_tensor(buf142, (4, 77, 768), (59136, 768, 1), 0); del buf142  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_27, hidden_states_43, hidden_states_44], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:27
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf132, buf146, arg109_1, arg110_1, arg111_1, buf150, 308, 768, stream=stream0)
            del arg110_1
            del arg111_1
            buf151 = reinterpret_tensor(buf130, (308, 3072), (3072, 1), 0); del buf130  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_27, hidden_states_43, hidden_states_44, hidden_states_45], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            # [Provenance debug handles] extern_kernels.mm:92
            extern_kernels.mm(reinterpret_tensor(buf150, (308, 768), (768, 1), 0), reinterpret_tensor(arg112_1, (768, 3072), (1, 768), 0), out=buf151)
            del arg112_1
            buf152 = reinterpret_tensor(buf151, (4, 77, 3072), (236544, 3072, 1), 0); del buf151  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_45, mul_12, sigmoid_6, hidden_states_46], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid]
            # [Provenance debug handles] triton_poi_fused_addmm_mul_sigmoid_view_3:28
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_mul_sigmoid_view_3.run(buf152, arg113_1, 946176, stream=stream0)
            del arg113_1
            buf153 = reinterpret_tensor(buf150, (308, 768), (768, 1), 0); del buf150  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_45, mul_12, sigmoid_6, hidden_states_46, hidden_states_47], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid, aten.t]
            # [Provenance debug handles] extern_kernels.mm:93
            extern_kernels.mm(reinterpret_tensor(buf152, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg114_1, (3072, 768), (1, 3072), 0), out=buf153)
            del arg114_1
            buf154 = reinterpret_tensor(buf146, (4, 77, 768), (59136, 768, 1), 0); del buf146  # reuse
            buf158 = reinterpret_tensor(buf138, (4, 77, 768), (59136, 768, 1), 0); del buf138  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_27, hidden_states_43, hidden_states_47, hidden_states_48, hidden_states_49], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:29
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf154, buf132, arg109_1, buf153, arg115_1, arg116_1, arg117_1, buf158, 308, 768, stream=stream0)
            del arg109_1
            del arg115_1
            del arg116_1
            del arg117_1
            buf159 = buf153; del buf153  # reuse
            # Topologically Sorted Source Nodes: [queries_14], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:94
            extern_kernels.addmm(arg119_1, reinterpret_tensor(buf158, (308, 768), (768, 1), 0), reinterpret_tensor(arg118_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf159)
            del arg118_1
            del arg119_1
            buf160 = buf137; del buf137  # reuse
            # Topologically Sorted Source Nodes: [keys_14], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:95
            extern_kernels.addmm(arg121_1, reinterpret_tensor(buf158, (308, 768), (768, 1), 0), reinterpret_tensor(arg120_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf160)
            del arg120_1
            del arg121_1
            buf161 = reinterpret_tensor(buf136, (308, 768), (768, 1), 0); del buf136  # reuse
            # Topologically Sorted Source Nodes: [values_14], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:96
            extern_kernels.addmm(arg123_1, reinterpret_tensor(buf158, (308, 768), (768, 1), 0), reinterpret_tensor(arg122_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf161)
            del arg122_1
            del arg123_1
            buf162 = buf140; del buf140  # reuse
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_14, view_23, queries_15, keys_14, view_24, keys_15, values_14, view_25, values_15, attn_output_28], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1:30
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1.run(buf162, 23716, stream=stream0)
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_14, view_23, queries_15, keys_14, view_24, keys_15, values_14, view_25, values_15, attn_output_28], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            buf163 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf159, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf160, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf161, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf162, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
            del buf162
            buf164 = buf163[0]
            assert_size_stride(buf164, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf164, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf163
            buf168 = buf161; del buf161  # reuse
            # Topologically Sorted Source Nodes: [transpose_31, reshape_7, attn_output_31, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:97
            extern_kernels.mm(reinterpret_tensor(buf164, (308, 768), (768, 1), 0), reinterpret_tensor(arg124_1, (768, 768), (1, 768), 0), out=buf168)
            del arg124_1
            buf172 = reinterpret_tensor(buf164, (4, 77, 768), (59136, 768, 1), 0); del buf164  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_31, hidden_states_50, hidden_states_51], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:31
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf154, buf168, arg125_1, arg126_1, arg127_1, buf172, 308, 768, stream=stream0)
            del arg126_1
            del arg127_1
            buf173 = reinterpret_tensor(buf152, (308, 3072), (3072, 1), 0); del buf152  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_31, hidden_states_50, hidden_states_51, hidden_states_52], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            # [Provenance debug handles] extern_kernels.mm:98
            extern_kernels.mm(reinterpret_tensor(buf172, (308, 768), (768, 1), 0), reinterpret_tensor(arg128_1, (768, 3072), (1, 768), 0), out=buf173)
            del arg128_1
            buf174 = reinterpret_tensor(buf173, (4, 77, 3072), (236544, 3072, 1), 0); del buf173  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_52, mul_14, sigmoid_7, hidden_states_53], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid]
            # [Provenance debug handles] triton_poi_fused_addmm_mul_sigmoid_view_3:32
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_mul_sigmoid_view_3.run(buf174, arg129_1, 946176, stream=stream0)
            del arg129_1
            buf175 = reinterpret_tensor(buf172, (308, 768), (768, 1), 0); del buf172  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_52, mul_14, sigmoid_7, hidden_states_53, hidden_states_54], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid, aten.t]
            # [Provenance debug handles] extern_kernels.mm:99
            extern_kernels.mm(reinterpret_tensor(buf174, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg130_1, (3072, 768), (1, 3072), 0), out=buf175)
            del arg130_1
            buf176 = reinterpret_tensor(buf168, (4, 77, 768), (59136, 768, 1), 0); del buf168  # reuse
            buf180 = reinterpret_tensor(buf160, (4, 77, 768), (59136, 768, 1), 0); del buf160  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_31, hidden_states_50, hidden_states_54, hidden_states_55, hidden_states_56], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:33
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf176, buf154, arg125_1, buf175, arg131_1, arg132_1, arg133_1, buf180, 308, 768, stream=stream0)
            del arg125_1
            del arg131_1
            del arg132_1
            del arg133_1
            buf181 = buf175; del buf175  # reuse
            # Topologically Sorted Source Nodes: [queries_16], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:100
            extern_kernels.addmm(arg135_1, reinterpret_tensor(buf180, (308, 768), (768, 1), 0), reinterpret_tensor(arg134_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf181)
            del arg134_1
            del arg135_1
            buf182 = buf159; del buf159  # reuse
            # Topologically Sorted Source Nodes: [keys_16], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:101
            extern_kernels.addmm(arg137_1, reinterpret_tensor(buf180, (308, 768), (768, 1), 0), reinterpret_tensor(arg136_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf182)
            del arg136_1
            del arg137_1
            buf183 = reinterpret_tensor(buf158, (308, 768), (768, 1), 0); del buf158  # reuse
            # Topologically Sorted Source Nodes: [values_16], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:102
            extern_kernels.addmm(arg139_1, reinterpret_tensor(buf180, (308, 768), (768, 1), 0), reinterpret_tensor(arg138_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf183)
            del arg138_1
            del arg139_1
            del buf180
            buf184 = empty_strided_cuda((4, 1, 77, 77), (6160, 0, 80, 1), torch.float16)
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_16, view_26, queries_17, keys_16, view_27, keys_17, values_16, view_28, values_17, attn_output_32], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1:34
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1.run(buf184, 23716, stream=stream0)
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_16, view_26, queries_17, keys_16, view_27, keys_17, values_16, view_28, values_17, attn_output_32], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            buf185 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf181, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf182, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf183, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf184, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
            del buf184
            buf186 = buf185[0]
            assert_size_stride(buf186, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf186, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf185
            buf190 = buf183; del buf183  # reuse
            # Topologically Sorted Source Nodes: [transpose_35, reshape_8, attn_output_35, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:103
            extern_kernels.mm(reinterpret_tensor(buf186, (308, 768), (768, 1), 0), reinterpret_tensor(arg140_1, (768, 768), (1, 768), 0), out=buf190)
            del arg140_1
            buf194 = reinterpret_tensor(buf186, (4, 77, 768), (59136, 768, 1), 0); del buf186  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_35, hidden_states_57, hidden_states_58], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:35
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf176, buf190, arg141_1, arg142_1, arg143_1, buf194, 308, 768, stream=stream0)
            del arg142_1
            del arg143_1
            buf195 = reinterpret_tensor(buf174, (308, 3072), (3072, 1), 0); del buf174  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_35, hidden_states_57, hidden_states_58, hidden_states_59], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            # [Provenance debug handles] extern_kernels.mm:104
            extern_kernels.mm(reinterpret_tensor(buf194, (308, 768), (768, 1), 0), reinterpret_tensor(arg144_1, (768, 3072), (1, 768), 0), out=buf195)
            del arg144_1
            buf196 = reinterpret_tensor(buf195, (4, 77, 3072), (236544, 3072, 1), 0); del buf195  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_59, mul_16, sigmoid_8, hidden_states_60], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid]
            # [Provenance debug handles] triton_poi_fused_addmm_mul_sigmoid_view_3:36
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_mul_sigmoid_view_3.run(buf196, arg145_1, 946176, stream=stream0)
            del arg145_1
            buf197 = reinterpret_tensor(buf194, (308, 768), (768, 1), 0); del buf194  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_59, mul_16, sigmoid_8, hidden_states_60, hidden_states_61], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid, aten.t]
            # [Provenance debug handles] extern_kernels.mm:105
            extern_kernels.mm(reinterpret_tensor(buf196, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg146_1, (3072, 768), (1, 3072), 0), out=buf197)
            del arg146_1
            buf198 = reinterpret_tensor(buf190, (4, 77, 768), (59136, 768, 1), 0); del buf190  # reuse
            buf202 = reinterpret_tensor(buf182, (4, 77, 768), (59136, 768, 1), 0); del buf182  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_35, hidden_states_57, hidden_states_61, hidden_states_62, hidden_states_63], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:37
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf198, buf176, arg141_1, buf197, arg147_1, arg148_1, arg149_1, buf202, 308, 768, stream=stream0)
            del arg141_1
            del arg147_1
            del arg148_1
            del arg149_1
            buf203 = buf197; del buf197  # reuse
            # Topologically Sorted Source Nodes: [queries_18], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:106
            extern_kernels.addmm(arg151_1, reinterpret_tensor(buf202, (308, 768), (768, 1), 0), reinterpret_tensor(arg150_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf203)
            del arg150_1
            del arg151_1
            buf204 = buf181; del buf181  # reuse
            # Topologically Sorted Source Nodes: [keys_18], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:107
            extern_kernels.addmm(arg153_1, reinterpret_tensor(buf202, (308, 768), (768, 1), 0), reinterpret_tensor(arg152_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf204)
            del arg152_1
            del arg153_1
            buf205 = empty_strided_cuda((308, 768), (768, 1), torch.float16)
            # Topologically Sorted Source Nodes: [values_18], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:108
            extern_kernels.addmm(arg155_1, reinterpret_tensor(buf202, (308, 768), (768, 1), 0), reinterpret_tensor(arg154_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf205)
            del arg154_1
            del arg155_1
            del buf202
            buf206 = empty_strided_cuda((4, 1, 77, 77), (6160, 0, 80, 1), torch.float16)
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_18, view_29, queries_19, keys_18, view_30, keys_19, values_18, view_31, values_19, attn_output_36], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1:38
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1.run(buf206, 23716, stream=stream0)
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_18, view_29, queries_19, keys_18, view_30, keys_19, values_18, view_31, values_19, attn_output_36], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            buf207 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf203, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf204, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf205, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf206, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
            del buf203
            del buf206
            buf208 = buf207[0]
            assert_size_stride(buf208, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf208, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf207
            buf212 = buf205; del buf205  # reuse
            # Topologically Sorted Source Nodes: [transpose_39, reshape_9, attn_output_39, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:109
            extern_kernels.mm(reinterpret_tensor(buf208, (308, 768), (768, 1), 0), reinterpret_tensor(arg156_1, (768, 768), (1, 768), 0), out=buf212)
            del arg156_1
            buf216 = reinterpret_tensor(buf208, (4, 77, 768), (59136, 768, 1), 0); del buf208  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_39, hidden_states_64, hidden_states_65], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:39
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf198, buf212, arg157_1, arg158_1, arg159_1, buf216, 308, 768, stream=stream0)
            del arg158_1
            del arg159_1
            buf217 = reinterpret_tensor(buf196, (308, 3072), (3072, 1), 0); del buf196  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_39, hidden_states_64, hidden_states_65, hidden_states_66], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            # [Provenance debug handles] extern_kernels.mm:110
            extern_kernels.mm(reinterpret_tensor(buf216, (308, 768), (768, 1), 0), reinterpret_tensor(arg160_1, (768, 3072), (1, 768), 0), out=buf217)
            del arg160_1
            buf218 = reinterpret_tensor(buf217, (4, 77, 3072), (236544, 3072, 1), 0); del buf217  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_66, mul_18, sigmoid_9, hidden_states_67], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid]
            # [Provenance debug handles] triton_poi_fused_addmm_mul_sigmoid_view_3:40
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_mul_sigmoid_view_3.run(buf218, arg161_1, 946176, stream=stream0)
            del arg161_1
            buf219 = reinterpret_tensor(buf216, (308, 768), (768, 1), 0); del buf216  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_66, mul_18, sigmoid_9, hidden_states_67, hidden_states_68], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid, aten.t]
            # [Provenance debug handles] extern_kernels.mm:111
            extern_kernels.mm(reinterpret_tensor(buf218, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg162_1, (3072, 768), (1, 3072), 0), out=buf219)
            del arg162_1
            buf220 = reinterpret_tensor(buf212, (4, 77, 768), (59136, 768, 1), 0); del buf212  # reuse
            buf224 = reinterpret_tensor(buf204, (4, 77, 768), (59136, 768, 1), 0); del buf204  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_39, hidden_states_64, hidden_states_68, hidden_states_69, hidden_states_70], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:41
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf220, buf198, arg157_1, buf219, arg163_1, arg164_1, arg165_1, buf224, 308, 768, stream=stream0)
            del arg157_1
            del arg163_1
            del arg164_1
            del arg165_1
            buf225 = buf219; del buf219  # reuse
            # Topologically Sorted Source Nodes: [queries_20], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:112
            extern_kernels.addmm(arg167_1, reinterpret_tensor(buf224, (308, 768), (768, 1), 0), reinterpret_tensor(arg166_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf225)
            del arg166_1
            del arg167_1
            buf226 = empty_strided_cuda((308, 768), (768, 1), torch.float16)
            # Topologically Sorted Source Nodes: [keys_20], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:113
            extern_kernels.addmm(arg169_1, reinterpret_tensor(buf224, (308, 768), (768, 1), 0), reinterpret_tensor(arg168_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf226)
            del arg168_1
            del arg169_1
            buf227 = empty_strided_cuda((308, 768), (768, 1), torch.float16)
            # Topologically Sorted Source Nodes: [values_20], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:114
            extern_kernels.addmm(arg171_1, reinterpret_tensor(buf224, (308, 768), (768, 1), 0), reinterpret_tensor(arg170_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf227)
            del arg170_1
            del arg171_1
            del buf224
            buf228 = empty_strided_cuda((4, 1, 77, 77), (6160, 0, 80, 1), torch.float16)
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_20, view_32, queries_21, keys_20, view_33, keys_21, values_20, view_34, values_21, attn_output_40], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1:42
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1.run(buf228, 23716, stream=stream0)
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_20, view_32, queries_21, keys_20, view_33, keys_21, values_20, view_34, values_21, attn_output_40], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            buf229 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf225, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf226, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf227, (4, 12, 77, 64), (59136, 64, 768, 1), 0), reinterpret_tensor(buf228, (4, 12, 77, 77), (6160, 0, 80, 1), 0), False, scale=0.125)
            del buf225
            del buf226
            del buf228
            buf230 = buf229[0]
            assert_size_stride(buf230, (4, 12, 77, 64), (59136, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf230, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf229
            buf234 = buf227; del buf227  # reuse
            # Topologically Sorted Source Nodes: [transpose_43, reshape_10, attn_output_43, ], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:115
            extern_kernels.mm(reinterpret_tensor(buf230, (308, 768), (768, 1), 0), reinterpret_tensor(arg172_1, (768, 768), (1, 768), 0), out=buf234)
            del arg172_1
            buf238 = reinterpret_tensor(buf230, (4, 77, 768), (59136, 768, 1), 0); del buf230  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_43, hidden_states_71, hidden_states_72], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:43
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf220, buf234, arg173_1, arg174_1, arg175_1, buf238, 308, 768, stream=stream0)
            del arg174_1
            del arg175_1
            buf239 = reinterpret_tensor(buf218, (308, 3072), (3072, 1), 0); del buf218  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_43, hidden_states_71, hidden_states_72, hidden_states_73], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            # [Provenance debug handles] extern_kernels.mm:116
            extern_kernels.mm(reinterpret_tensor(buf238, (308, 768), (768, 1), 0), reinterpret_tensor(arg176_1, (768, 3072), (1, 768), 0), out=buf239)
            del arg176_1
            buf240 = reinterpret_tensor(buf239, (4, 77, 3072), (236544, 3072, 1), 0); del buf239  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_73, mul_20, sigmoid_10, hidden_states_74], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid]
            # [Provenance debug handles] triton_poi_fused_addmm_mul_sigmoid_view_3:44
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_mul_sigmoid_view_3.run(buf240, arg177_1, 946176, stream=stream0)
            del arg177_1
            buf241 = reinterpret_tensor(buf238, (308, 768), (768, 1), 0); del buf238  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_73, mul_20, sigmoid_10, hidden_states_74, hidden_states_75], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid, aten.t]
            # [Provenance debug handles] extern_kernels.mm:117
            extern_kernels.mm(reinterpret_tensor(buf240, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg178_1, (3072, 768), (1, 3072), 0), out=buf241)
            del arg178_1
            buf242 = reinterpret_tensor(buf234, (4, 77, 768), (59136, 768, 1), 0); del buf234  # reuse
            buf246 = empty_strided_cuda((4, 77, 768), (59136, 768, 1), torch.float16)
            # Topologically Sorted Source Nodes: [, attn_output_43, hidden_states_71, hidden_states_75, hidden_states_76, hidden_states_77], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:45
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf242, buf220, arg173_1, buf241, arg179_1, arg180_1, arg181_1, buf246, 308, 768, stream=stream0)
            del arg173_1
            del arg179_1
            del arg180_1
            del arg181_1
            buf247 = buf241; del buf241  # reuse
            # Topologically Sorted Source Nodes: [queries_22], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:118
            extern_kernels.addmm(arg183_1, reinterpret_tensor(buf246, (308, 768), (768, 1), 0), reinterpret_tensor(arg182_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf247)
            del arg182_1
            del arg183_1
            buf248 = empty_strided_cuda((308, 768), (768, 1), torch.float16)
            # Topologically Sorted Source Nodes: [keys_22], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:119
            extern_kernels.addmm(arg185_1, reinterpret_tensor(buf246, (308, 768), (768, 1), 0), reinterpret_tensor(arg184_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf248)
            del arg184_1
            del arg185_1
            buf249 = empty_strided_cuda((308, 768), (768, 1), torch.float16)
            # Topologically Sorted Source Nodes: [values_22], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:120
            extern_kernels.addmm(arg187_1, reinterpret_tensor(buf246, (308, 768), (768, 1), 0), reinterpret_tensor(arg186_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf249)
            del arg186_1
            del arg187_1
            del buf246
            buf250 = empty_strided_cuda((4, 1, 77, 77), (6160, 0, 80, 1), torch.float16)
            # Topologically Sorted Source Nodes: [mask_cond, add_1, view_1, lt, masked_fill_, mask, queries_22, view_35, queries_23, keys_22, view_36, keys_23, values_22, view_37, values_23, attn_output_44], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full, aten.transpose, aten.unsqueeze, aten.expand, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1:46
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_constant_pad_nd_expand_full_lt_masked_fill_slice_transpose_unsqueeze_view_1.run(buf250, 23716, stream=stream0)
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
            # [Provenance debug handles] extern_kernels.mm:121
            extern_kernels.mm(reinterpret_tensor(buf252, (308, 768), (768, 1), 0), reinterpret_tensor(arg188_1, (768, 768), (1, 768), 0), out=buf256)
            del arg188_1
            buf260 = reinterpret_tensor(buf252, (4, 77, 768), (59136, 768, 1), 0); del buf252  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_47, hidden_states_78, hidden_states_79], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_2:47
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf242, buf256, arg189_1, arg190_1, arg191_1, buf260, 308, 768, stream=stream0)
            del arg190_1
            del arg191_1
            buf261 = reinterpret_tensor(buf240, (308, 3072), (3072, 1), 0); del buf240  # reuse
            # Topologically Sorted Source Nodes: [, attn_output_47, hidden_states_78, hidden_states_79, hidden_states_80], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.t]
            # [Provenance debug handles] extern_kernels.mm:122
            extern_kernels.mm(reinterpret_tensor(buf260, (308, 768), (768, 1), 0), reinterpret_tensor(arg192_1, (768, 3072), (1, 768), 0), out=buf261)
            del arg192_1
            del buf260
            buf262 = reinterpret_tensor(buf261, (4, 77, 3072), (236544, 3072, 1), 0); del buf261  # reuse
            # Topologically Sorted Source Nodes: [, hidden_states_80, mul_22, sigmoid_11, hidden_states_81], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid]
            # [Provenance debug handles] triton_poi_fused_addmm_mul_sigmoid_view_3:48
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_mul_sigmoid_view_3.run(buf262, arg193_1, 946176, stream=stream0)
            del arg193_1
            buf263 = empty_strided_cuda((308, 768), (768, 1), torch.float16)
            # Topologically Sorted Source Nodes: [, hidden_states_80, mul_22, sigmoid_11, hidden_states_81, hidden_states_82], Original ATen: [aten.addmm, aten.view, aten.mul, aten.sigmoid, aten.t]
            # [Provenance debug handles] extern_kernels.mm:123
            extern_kernels.mm(reinterpret_tensor(buf262, (308, 3072), (3072, 1), 0), reinterpret_tensor(arg194_1, (3072, 768), (1, 3072), 0), out=buf263)
            del arg194_1
            del buf262
            buf264 = reinterpret_tensor(buf256, (4, 77, 768), (59136, 768, 1), 0); del buf256  # reuse
            buf268 = empty_strided_cuda((4, 77, 768), (59136, 768, 1), torch.float16)
            # Topologically Sorted Source Nodes: [, attn_output_47, hidden_states_78, hidden_states_82, hidden_states_83, last_hidden_state], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_view_4:49
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf264, buf242, arg189_1, buf263, arg195_1, arg196_1, arg197_1, buf268, 308, 768, stream=stream0)
            del arg189_1
            del arg195_1
            del arg196_1
            del arg197_1
            del buf263
            buf269 = empty_strided_cuda((4, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [to_1, argmax], Original ATen: [aten._to_copy, aten.argmax]
            # [Provenance debug handles] triton_per_fused__to_copy_argmax_5:50
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_argmax_5.run(arg0_1, buf269, 4, 77, stream=stream0)
            del arg0_1
            buf270 = empty_strided_cuda((4, 768), (768, 1), torch.float16)
            # Topologically Sorted Source Nodes: [arange_1, pooled_output], Original ATen: [aten.arange, aten.index]
            # [Provenance debug handles] triton_poi_fused_arange_index_6:51
            stream0 = get_raw_stream(0)
            triton_poi_fused_arange_index_6.run(buf269, buf268, buf270, 3072, stream=stream0)
            del buf269
        return (buf268, buf270, buf0, buf22, buf44, buf66, buf88, buf110, buf132, buf154, buf176, buf198, buf220, buf242, buf264, )

runner = Runner(partitions=[])
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
