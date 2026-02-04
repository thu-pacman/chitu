
import os
os.environ['TORCHDYNAMO_VERBOSE'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0 9.0+PTX'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_LOGS'] = 'dynamo'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_wucz'
os.environ['TRITON_CACHE_DIR'] = '/tmp/torchinductor_wucz/triton/0'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.triton.store_cubin = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.test_configs.runtime_triton_dtype_assert = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.9.1+cu128
# torch cuda version: 12.8
# torch git version: 5811a8d7da873dd699ff6687092c225caffcf1bb


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Thu_Mar_28_02:18:24_PDT_2024 
# Cuda compilation tools, release 12.4, V12.4.131 
# Build cuda_12.4.r12.4/compiler.34097967_0 

# GPU Hardware Info: 
# NVIDIA H100 PCIe : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1):
        convolution = torch.ops.aten.convolution.default(arg2_1, arg0_1, arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg2_1 = arg0_1 = arg1_1 = None
        convolution_1 = torch.ops.aten.convolution.default(convolution, arg3_1, arg4_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convolution = arg3_1 = arg4_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32)
        view = torch.ops.aten.view.default(convert_element_type, [4, 32, 16, 16384]);  convert_element_type = None
        var_mean = torch.ops.aten.var_mean.correction(view, [2, 3], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        sub = torch.ops.aten.sub.Tensor(view, getitem_1);  view = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        view_1 = torch.ops.aten.view.default(mul, [4, 512, 128, 128]);  mul = None
        unsqueeze = torch.ops.aten.unsqueeze.default(arg6_1, 0);  arg6_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(view_1, unsqueeze_2);  view_1 = unsqueeze_2 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(arg7_1, 0);  arg7_1 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        sigmoid = torch.ops.aten.sigmoid.default(convert_element_type_default_3)
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, sigmoid);  convert_element_type_default_3 = sigmoid = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(mul_2, torch.float16);  mul_2 = None
        convolution_2 = torch.ops.aten.convolution.default(convert_element_type_5, arg8_1, arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_5 = arg8_1 = arg9_1 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        view_2 = torch.ops.aten.view.default(convert_element_type_6, [4, 32, 16, 16384]);  convert_element_type_6 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(view_2, [2, 3], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_2 = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1 = torch.ops.aten.sub.Tensor(view_2, getitem_3);  view_2 = getitem_3 = None
        mul_3 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        view_3 = torch.ops.aten.view.default(mul_3, [4, 512, 128, 128]);  mul_3 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(arg10_1, 0);  arg10_1 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        mul_4 = torch.ops.aten.mul.Tensor(view_3, unsqueeze_8);  view_3 = unsqueeze_8 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(arg11_1, 0);  arg11_1 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(unsqueeze_9, 2);  unsqueeze_9 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None
        add_3 = torch.ops.aten.add.Tensor(mul_4, unsqueeze_11);  mul_4 = unsqueeze_11 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_3, torch.float32);  add_3 = None
        sigmoid_1 = torch.ops.aten.sigmoid.default(convert_element_type_default_2)
        mul_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, sigmoid_1);  convert_element_type_default_2 = sigmoid_1 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(mul_5, torch.float16);  mul_5 = None
        convolution_3 = torch.ops.aten.convolution.default(convert_element_type_11, arg12_1, arg13_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_11 = arg12_1 = arg13_1 = None
        add_4 = torch.ops.aten.add.Tensor(convolution_1, convolution_3);  convolution_1 = convolution_3 = None
        div = torch.ops.aten.div.Tensor(add_4, 1);  add_4 = None
        view_4 = torch.ops.aten.view.default(div, [4, 512, 16384])
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        view_5 = torch.ops.aten.view.default(convert_element_type_12, [4, 32, 16, 16384]);  convert_element_type_12 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(view_5, [2, 3], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_2 = torch.ops.aten.sub.Tensor(view_5, getitem_5);  view_5 = getitem_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        view_6 = torch.ops.aten.view.default(mul_6, [4, 512, 16384]);  mul_6 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(arg14_1, 0);  arg14_1 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
        mul_7 = torch.ops.aten.mul.Tensor(view_6, unsqueeze_13);  view_6 = unsqueeze_13 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(arg15_1, 0);  arg15_1 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, 2);  unsqueeze_14 = None
        add_6 = torch.ops.aten.add.Tensor(mul_7, unsqueeze_15);  mul_7 = unsqueeze_15 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(add_6, torch.float16);  add_6 = None
        permute_2 = torch.ops.aten.permute.default(convert_element_type_13, [0, 2, 1]);  convert_element_type_13 = None
        permute_3 = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        clone_1 = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format)
        view_7 = torch.ops.aten.view.default(clone_1, [65536, 512]);  clone_1 = None
        mm = torch.ops.aten.mm.default(view_7, permute_3);  view_7 = permute_3 = None
        view_8 = torch.ops.aten.view.default(mm, [4, 16384, 512]);  mm = None
        add_7 = torch.ops.aten.add.Tensor(view_8, arg17_1);  view_8 = arg17_1 = None
        permute_4 = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        clone_2 = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format)
        view_9 = torch.ops.aten.view.default(clone_2, [65536, 512]);  clone_2 = None
        mm_1 = torch.ops.aten.mm.default(view_9, permute_4);  view_9 = permute_4 = None
        view_10 = torch.ops.aten.view.default(mm_1, [4, 16384, 512]);  mm_1 = None
        add_8 = torch.ops.aten.add.Tensor(view_10, arg19_1);  view_10 = arg19_1 = None
        permute_5 = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        clone_3 = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_11 = torch.ops.aten.view.default(clone_3, [65536, 512]);  clone_3 = None
        mm_2 = torch.ops.aten.mm.default(view_11, permute_5);  view_11 = permute_5 = None
        view_12 = torch.ops.aten.view.default(mm_2, [4, 16384, 512]);  mm_2 = None
        add_9 = torch.ops.aten.add.Tensor(view_12, arg21_1);  view_12 = arg21_1 = None
        view_13 = torch.ops.aten.view.default(add_7, [4, -1, 1, 512]);  add_7 = None
        permute_6 = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
        view_14 = torch.ops.aten.view.default(add_8, [4, -1, 1, 512]);  add_8 = None
        permute_7 = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        view_15 = torch.ops.aten.view.default(add_9, [4, -1, 1, 512]);  add_9 = None
        permute_8 = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_6, permute_7, permute_8, None, False);  permute_6 = permute_7 = permute_8 = None
        getitem_6 = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        permute_9 = torch.ops.aten.permute.default(getitem_6, [0, 2, 1, 3]);  getitem_6 = None
        view_16 = torch.ops.aten.view.default(permute_9, [4, -1, 512]);  permute_9 = None
        view_17 = torch.ops.aten.view.default(view_16, [65536, 512]);  view_16 = None
        permute_10 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm = torch.ops.aten.addmm.default(arg23_1, view_17, permute_10);  arg23_1 = view_17 = permute_10 = None
        view_18 = torch.ops.aten.view.default(addmm, [4, 16384, 512]);  addmm = None
        permute_11 = torch.ops.aten.permute.default(view_18, [0, 2, 1]);  view_18 = None
        view_19 = torch.ops.aten.view.default(permute_11, [4, 512, 128, 128]);  permute_11 = None
        add_10 = torch.ops.aten.add.Tensor(view_19, div);  view_19 = div = None
        div_1 = torch.ops.aten.div.Tensor(add_10, 1);  add_10 = None
        clone_5 = torch.ops.aten.clone.default(div_1, memory_format = torch.contiguous_format)
        convert_element_type_25 = torch.ops.prims.convert_element_type.default(clone_5, torch.float32);  clone_5 = None
        view_20 = torch.ops.aten.view.default(convert_element_type_25, [4, 32, 16, 16384]);  convert_element_type_25 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(view_20, [2, 3], correction = 0, keepdim = True)
        getitem_10 = var_mean_3[0]
        getitem_11 = var_mean_3[1];  var_mean_3 = None
        add_11 = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_3 = torch.ops.aten.sub.Tensor(view_20, getitem_11);  view_20 = getitem_11 = None
        mul_8 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        view_21 = torch.ops.aten.view.default(mul_8, [4, 512, 128, 128]);  mul_8 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(arg24_1, 0);  arg24_1 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(unsqueeze_16, 2);  unsqueeze_16 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(unsqueeze_17, 3);  unsqueeze_17 = None
        mul_9 = torch.ops.aten.mul.Tensor(view_21, unsqueeze_18);  view_21 = unsqueeze_18 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(arg25_1, 0);  arg25_1 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(unsqueeze_19, 2);  unsqueeze_19 = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(unsqueeze_20, 3);  unsqueeze_20 = None
        add_12 = torch.ops.aten.add.Tensor(mul_9, unsqueeze_21);  mul_9 = unsqueeze_21 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_12, torch.float32);  add_12 = None
        sigmoid_2 = torch.ops.aten.sigmoid.default(convert_element_type_default_1)
        mul_10 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, sigmoid_2);  convert_element_type_default_1 = sigmoid_2 = None
        convert_element_type_30 = torch.ops.prims.convert_element_type.default(mul_10, torch.float16);  mul_10 = None
        convolution_4 = torch.ops.aten.convolution.default(convert_element_type_30, arg26_1, arg27_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_30 = arg26_1 = arg27_1 = None
        convert_element_type_31 = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32);  convolution_4 = None
        view_22 = torch.ops.aten.view.default(convert_element_type_31, [4, 32, 16, 16384]);  convert_element_type_31 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(view_22, [2, 3], correction = 0, keepdim = True)
        getitem_12 = var_mean_4[0]
        getitem_13 = var_mean_4[1];  var_mean_4 = None
        add_13 = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_4 = torch.ops.aten.sub.Tensor(view_22, getitem_13);  view_22 = getitem_13 = None
        mul_11 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        view_23 = torch.ops.aten.view.default(mul_11, [4, 512, 128, 128]);  mul_11 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(arg28_1, 0);  arg28_1 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(unsqueeze_22, 2);  unsqueeze_22 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(unsqueeze_23, 3);  unsqueeze_23 = None
        mul_12 = torch.ops.aten.mul.Tensor(view_23, unsqueeze_24);  view_23 = unsqueeze_24 = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(arg29_1, 0);  arg29_1 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(unsqueeze_25, 2);  unsqueeze_25 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(unsqueeze_26, 3);  unsqueeze_26 = None
        add_14 = torch.ops.aten.add.Tensor(mul_12, unsqueeze_27);  mul_12 = unsqueeze_27 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_14, torch.float32);  add_14 = None
        sigmoid_3 = torch.ops.aten.sigmoid.default(convert_element_type_default)
        mul_13 = torch.ops.aten.mul.Tensor(convert_element_type_default, sigmoid_3);  convert_element_type_default = sigmoid_3 = None
        convert_element_type_36 = torch.ops.prims.convert_element_type.default(mul_13, torch.float16);  mul_13 = None
        convolution_5 = torch.ops.aten.convolution.default(convert_element_type_36, arg30_1, arg31_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  convert_element_type_36 = arg30_1 = arg31_1 = None
        add_15 = torch.ops.aten.add.Tensor(div_1, convolution_5);  div_1 = convolution_5 = None
        div_2 = torch.ops.aten.div.Tensor(add_15, 1);  add_15 = None
        convert_element_type_37 = torch.ops.prims.convert_element_type.default(div_2, torch.float32);  div_2 = None
        clone_7 = torch.ops.aten.clone.default(convert_element_type_37, memory_format = torch.contiguous_format)
        view_24 = torch.ops.aten.view.default(clone_7, [4, 32, 16, 16384]);  clone_7 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(view_24, [2, 3], correction = 0, keepdim = True)
        getitem_14 = var_mean_5[0]
        getitem_15 = var_mean_5[1];  var_mean_5 = None
        add_16 = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_5 = torch.ops.aten.sub.Tensor(view_24, getitem_15);  view_24 = getitem_15 = None
        mul_14 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        view_25 = torch.ops.aten.view.default(mul_14, [4, 512, 128, 128]);  mul_14 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(arg5_1, 0);  arg5_1 = None
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(unsqueeze_28, 2);  unsqueeze_28 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(unsqueeze_29, 3);  unsqueeze_29 = None
        mul_15 = torch.ops.aten.mul.Tensor(view_25, unsqueeze_30);  view_25 = unsqueeze_30 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(arg32_1, 0);  arg32_1 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(unsqueeze_31, 2);  unsqueeze_31 = None
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(unsqueeze_32, 3);  unsqueeze_32 = None
        add_17 = torch.ops.aten.add.Tensor(mul_15, unsqueeze_33);  mul_15 = unsqueeze_33 = None
        sigmoid_4 = torch.ops.aten.sigmoid.default(add_17)
        mul_16 = torch.ops.aten.mul.Tensor(add_17, sigmoid_4);  add_17 = sigmoid_4 = None
        convolution_6 = torch.ops.aten.convolution.default(mul_16, arg33_1, arg34_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_16 = arg33_1 = arg34_1 = None
        view_26 = torch.ops.aten.view.default(convolution_6, [4, 32, 16, 16384]);  convolution_6 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(view_26, [2, 3], correction = 0, keepdim = True)
        getitem_16 = var_mean_6[0]
        getitem_17 = var_mean_6[1];  var_mean_6 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_6 = torch.ops.aten.sub.Tensor(view_26, getitem_17);  view_26 = getitem_17 = None
        mul_17 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        view_27 = torch.ops.aten.view.default(mul_17, [4, 512, 128, 128]);  mul_17 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(arg35_1, 0);  arg35_1 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(unsqueeze_34, 2);  unsqueeze_34 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(unsqueeze_35, 3);  unsqueeze_35 = None
        mul_18 = torch.ops.aten.mul.Tensor(view_27, unsqueeze_36);  view_27 = unsqueeze_36 = None
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(arg36_1, 0);  arg36_1 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(unsqueeze_37, 2);  unsqueeze_37 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(unsqueeze_38, 3);  unsqueeze_38 = None
        add_19 = torch.ops.aten.add.Tensor(mul_18, unsqueeze_39);  mul_18 = unsqueeze_39 = None
        sigmoid_5 = torch.ops.aten.sigmoid.default(add_19)
        mul_19 = torch.ops.aten.mul.Tensor(add_19, sigmoid_5);  add_19 = sigmoid_5 = None
        convolution_7 = torch.ops.aten.convolution.default(mul_19, arg37_1, arg38_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_19 = arg37_1 = arg38_1 = None
        add_20 = torch.ops.aten.add.Tensor(convert_element_type_37, convolution_7);  convert_element_type_37 = convolution_7 = None
        div_3 = torch.ops.aten.div.Tensor(add_20, 1.0);  add_20 = None
        clone_9 = torch.ops.aten.clone.default(div_3, memory_format = torch.contiguous_format)
        view_28 = torch.ops.aten.view.default(clone_9, [4, 32, 16, 16384]);  clone_9 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(view_28, [2, 3], correction = 0, keepdim = True)
        getitem_18 = var_mean_7[0]
        getitem_19 = var_mean_7[1];  var_mean_7 = None
        add_21 = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_7 = torch.ops.aten.sub.Tensor(view_28, getitem_19);  view_28 = getitem_19 = None
        mul_20 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        view_29 = torch.ops.aten.view.default(mul_20, [4, 512, 128, 128]);  mul_20 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(arg39_1, 0);  arg39_1 = None
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(unsqueeze_40, 2);  unsqueeze_40 = None
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(unsqueeze_41, 3);  unsqueeze_41 = None
        mul_21 = torch.ops.aten.mul.Tensor(view_29, unsqueeze_42);  view_29 = unsqueeze_42 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(arg40_1, 0);  arg40_1 = None
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(unsqueeze_43, 2);  unsqueeze_43 = None
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(unsqueeze_44, 3);  unsqueeze_44 = None
        add_22 = torch.ops.aten.add.Tensor(mul_21, unsqueeze_45);  mul_21 = unsqueeze_45 = None
        sigmoid_6 = torch.ops.aten.sigmoid.default(add_22)
        mul_22 = torch.ops.aten.mul.Tensor(add_22, sigmoid_6);  add_22 = sigmoid_6 = None
        convolution_8 = torch.ops.aten.convolution.default(mul_22, arg41_1, arg42_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_22 = arg41_1 = arg42_1 = None
        view_30 = torch.ops.aten.view.default(convolution_8, [4, 32, 16, 16384]);  convolution_8 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(view_30, [2, 3], correction = 0, keepdim = True)
        getitem_20 = var_mean_8[0]
        getitem_21 = var_mean_8[1];  var_mean_8 = None
        add_23 = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_8 = torch.ops.aten.sub.Tensor(view_30, getitem_21);  view_30 = getitem_21 = None
        mul_23 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        view_31 = torch.ops.aten.view.default(mul_23, [4, 512, 128, 128]);  mul_23 = None
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(arg43_1, 0);  arg43_1 = None
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(unsqueeze_46, 2);  unsqueeze_46 = None
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(unsqueeze_47, 3);  unsqueeze_47 = None
        mul_24 = torch.ops.aten.mul.Tensor(view_31, unsqueeze_48);  view_31 = unsqueeze_48 = None
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(arg44_1, 0);  arg44_1 = None
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(unsqueeze_49, 2);  unsqueeze_49 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(unsqueeze_50, 3);  unsqueeze_50 = None
        add_24 = torch.ops.aten.add.Tensor(mul_24, unsqueeze_51);  mul_24 = unsqueeze_51 = None
        sigmoid_7 = torch.ops.aten.sigmoid.default(add_24)
        mul_25 = torch.ops.aten.mul.Tensor(add_24, sigmoid_7);  add_24 = sigmoid_7 = None
        convolution_9 = torch.ops.aten.convolution.default(mul_25, arg45_1, arg46_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_25 = arg45_1 = arg46_1 = None
        add_25 = torch.ops.aten.add.Tensor(div_3, convolution_9);  div_3 = convolution_9 = None
        div_4 = torch.ops.aten.div.Tensor(add_25, 1.0);  add_25 = None
        clone_11 = torch.ops.aten.clone.default(div_4, memory_format = torch.contiguous_format)
        view_32 = torch.ops.aten.view.default(clone_11, [4, 32, 16, 16384]);  clone_11 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(view_32, [2, 3], correction = 0, keepdim = True)
        getitem_22 = var_mean_9[0]
        getitem_23 = var_mean_9[1];  var_mean_9 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_9 = torch.ops.aten.sub.Tensor(view_32, getitem_23);  view_32 = getitem_23 = None
        mul_26 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        view_33 = torch.ops.aten.view.default(mul_26, [4, 512, 128, 128]);  mul_26 = None
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(arg47_1, 0);  arg47_1 = None
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(unsqueeze_52, 2);  unsqueeze_52 = None
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(unsqueeze_53, 3);  unsqueeze_53 = None
        mul_27 = torch.ops.aten.mul.Tensor(view_33, unsqueeze_54);  view_33 = unsqueeze_54 = None
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(arg48_1, 0);  arg48_1 = None
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(unsqueeze_55, 2);  unsqueeze_55 = None
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(unsqueeze_56, 3);  unsqueeze_56 = None
        add_27 = torch.ops.aten.add.Tensor(mul_27, unsqueeze_57);  mul_27 = unsqueeze_57 = None
        sigmoid_8 = torch.ops.aten.sigmoid.default(add_27)
        mul_28 = torch.ops.aten.mul.Tensor(add_27, sigmoid_8);  add_27 = sigmoid_8 = None
        convolution_10 = torch.ops.aten.convolution.default(mul_28, arg49_1, arg50_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_28 = arg49_1 = arg50_1 = None
        view_34 = torch.ops.aten.view.default(convolution_10, [4, 32, 16, 16384]);  convolution_10 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(view_34, [2, 3], correction = 0, keepdim = True)
        getitem_24 = var_mean_10[0]
        getitem_25 = var_mean_10[1];  var_mean_10 = None
        add_28 = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_10 = torch.ops.aten.sub.Tensor(view_34, getitem_25);  view_34 = getitem_25 = None
        mul_29 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        view_35 = torch.ops.aten.view.default(mul_29, [4, 512, 128, 128]);  mul_29 = None
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(arg51_1, 0);  arg51_1 = None
        unsqueeze_59 = torch.ops.aten.unsqueeze.default(unsqueeze_58, 2);  unsqueeze_58 = None
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(unsqueeze_59, 3);  unsqueeze_59 = None
        mul_30 = torch.ops.aten.mul.Tensor(view_35, unsqueeze_60);  view_35 = unsqueeze_60 = None
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(arg52_1, 0);  arg52_1 = None
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(unsqueeze_61, 2);  unsqueeze_61 = None
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(unsqueeze_62, 3);  unsqueeze_62 = None
        add_29 = torch.ops.aten.add.Tensor(mul_30, unsqueeze_63);  mul_30 = unsqueeze_63 = None
        sigmoid_9 = torch.ops.aten.sigmoid.default(add_29)
        mul_31 = torch.ops.aten.mul.Tensor(add_29, sigmoid_9);  add_29 = sigmoid_9 = None
        convolution_11 = torch.ops.aten.convolution.default(mul_31, arg53_1, arg54_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_31 = arg53_1 = arg54_1 = None
        add_30 = torch.ops.aten.add.Tensor(div_4, convolution_11);  div_4 = convolution_11 = None
        div_5 = torch.ops.aten.div.Tensor(add_30, 1.0);  add_30 = None
        iota = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_32 = torch.ops.aten.mul.Tensor(iota, 1);  iota = None
        add_31 = torch.ops.aten.add.Tensor(mul_32, 0);  mul_32 = None
        convert_element_type_38 = torch.ops.prims.convert_element_type.default(add_31, torch.float32);  add_31 = None
        add_32 = torch.ops.aten.add.Tensor(convert_element_type_38, 0.0);  convert_element_type_38 = None
        mul_33 = torch.ops.aten.mul.Tensor(add_32, 0.5);  add_32 = None
        convert_element_type_39 = torch.ops.prims.convert_element_type.default(mul_33, torch.int64);  mul_33 = None
        unsqueeze_64 = torch.ops.aten.unsqueeze.default(convert_element_type_39, -1);  convert_element_type_39 = None
        iota_1 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_34 = torch.ops.aten.mul.Tensor(iota_1, 1);  iota_1 = None
        add_33 = torch.ops.aten.add.Tensor(mul_34, 0);  mul_34 = None
        convert_element_type_40 = torch.ops.prims.convert_element_type.default(add_33, torch.float32);  add_33 = None
        add_34 = torch.ops.aten.add.Tensor(convert_element_type_40, 0.0);  convert_element_type_40 = None
        mul_35 = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
        convert_element_type_41 = torch.ops.prims.convert_element_type.default(mul_35, torch.int64);  mul_35 = None
        _unsafe_index = torch.ops.aten._unsafe_index.Tensor(div_5, [None, None, unsqueeze_64, convert_element_type_41]);  div_5 = unsqueeze_64 = convert_element_type_41 = None
        clone_13 = torch.ops.aten.clone.default(_unsafe_index, memory_format = torch.channels_last);  _unsafe_index = None
        convolution_12 = torch.ops.aten.convolution.default(clone_13, arg55_1, arg56_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  clone_13 = arg55_1 = arg56_1 = None
        clone_14 = torch.ops.aten.clone.default(convolution_12, memory_format = torch.contiguous_format)
        view_36 = torch.ops.aten.view.default(clone_14, [4, 32, 16, 65536]);  clone_14 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(view_36, [2, 3], correction = 0, keepdim = True)
        getitem_26 = var_mean_11[0]
        getitem_27 = var_mean_11[1];  var_mean_11 = None
        add_35 = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        sub_11 = torch.ops.aten.sub.Tensor(view_36, getitem_27);  view_36 = getitem_27 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        view_37 = torch.ops.aten.view.default(mul_36, [4, 512, 256, 256]);  mul_36 = None
        unsqueeze_65 = torch.ops.aten.unsqueeze.default(arg57_1, 0);  arg57_1 = None
        unsqueeze_66 = torch.ops.aten.unsqueeze.default(unsqueeze_65, 2);  unsqueeze_65 = None
        unsqueeze_67 = torch.ops.aten.unsqueeze.default(unsqueeze_66, 3);  unsqueeze_66 = None
        mul_37 = torch.ops.aten.mul.Tensor(view_37, unsqueeze_67);  view_37 = unsqueeze_67 = None
        unsqueeze_68 = torch.ops.aten.unsqueeze.default(arg58_1, 0);  arg58_1 = None
        unsqueeze_69 = torch.ops.aten.unsqueeze.default(unsqueeze_68, 2);  unsqueeze_68 = None
        unsqueeze_70 = torch.ops.aten.unsqueeze.default(unsqueeze_69, 3);  unsqueeze_69 = None
        add_36 = torch.ops.aten.add.Tensor(mul_37, unsqueeze_70);  mul_37 = unsqueeze_70 = None
        sigmoid_10 = torch.ops.aten.sigmoid.default(add_36)
        mul_38 = torch.ops.aten.mul.Tensor(add_36, sigmoid_10);  add_36 = sigmoid_10 = None
        convolution_13 = torch.ops.aten.convolution.default(mul_38, arg59_1, arg60_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_38 = arg59_1 = arg60_1 = None
        view_38 = torch.ops.aten.view.default(convolution_13, [4, 32, 16, 65536]);  convolution_13 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(view_38, [2, 3], correction = 0, keepdim = True)
        getitem_28 = var_mean_12[0]
        getitem_29 = var_mean_12[1];  var_mean_12 = None
        add_37 = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_12 = torch.ops.aten.sub.Tensor(view_38, getitem_29);  view_38 = getitem_29 = None
        mul_39 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        view_39 = torch.ops.aten.view.default(mul_39, [4, 512, 256, 256]);  mul_39 = None
        unsqueeze_71 = torch.ops.aten.unsqueeze.default(arg61_1, 0);  arg61_1 = None
        unsqueeze_72 = torch.ops.aten.unsqueeze.default(unsqueeze_71, 2);  unsqueeze_71 = None
        unsqueeze_73 = torch.ops.aten.unsqueeze.default(unsqueeze_72, 3);  unsqueeze_72 = None
        mul_40 = torch.ops.aten.mul.Tensor(view_39, unsqueeze_73);  view_39 = unsqueeze_73 = None
        unsqueeze_74 = torch.ops.aten.unsqueeze.default(arg62_1, 0);  arg62_1 = None
        unsqueeze_75 = torch.ops.aten.unsqueeze.default(unsqueeze_74, 2);  unsqueeze_74 = None
        unsqueeze_76 = torch.ops.aten.unsqueeze.default(unsqueeze_75, 3);  unsqueeze_75 = None
        add_38 = torch.ops.aten.add.Tensor(mul_40, unsqueeze_76);  mul_40 = unsqueeze_76 = None
        sigmoid_11 = torch.ops.aten.sigmoid.default(add_38)
        mul_41 = torch.ops.aten.mul.Tensor(add_38, sigmoid_11);  add_38 = sigmoid_11 = None
        convolution_14 = torch.ops.aten.convolution.default(mul_41, arg63_1, arg64_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_41 = arg63_1 = arg64_1 = None
        add_39 = torch.ops.aten.add.Tensor(convolution_12, convolution_14);  convolution_12 = convolution_14 = None
        div_6 = torch.ops.aten.div.Tensor(add_39, 1.0);  add_39 = None
        clone_16 = torch.ops.aten.clone.default(div_6, memory_format = torch.contiguous_format)
        view_40 = torch.ops.aten.view.default(clone_16, [4, 32, 16, 65536]);  clone_16 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(view_40, [2, 3], correction = 0, keepdim = True)
        getitem_30 = var_mean_13[0]
        getitem_31 = var_mean_13[1];  var_mean_13 = None
        add_40 = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_13 = torch.ops.aten.sub.Tensor(view_40, getitem_31);  view_40 = getitem_31 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        view_41 = torch.ops.aten.view.default(mul_42, [4, 512, 256, 256]);  mul_42 = None
        unsqueeze_77 = torch.ops.aten.unsqueeze.default(arg65_1, 0);  arg65_1 = None
        unsqueeze_78 = torch.ops.aten.unsqueeze.default(unsqueeze_77, 2);  unsqueeze_77 = None
        unsqueeze_79 = torch.ops.aten.unsqueeze.default(unsqueeze_78, 3);  unsqueeze_78 = None
        mul_43 = torch.ops.aten.mul.Tensor(view_41, unsqueeze_79);  view_41 = unsqueeze_79 = None
        unsqueeze_80 = torch.ops.aten.unsqueeze.default(arg66_1, 0);  arg66_1 = None
        unsqueeze_81 = torch.ops.aten.unsqueeze.default(unsqueeze_80, 2);  unsqueeze_80 = None
        unsqueeze_82 = torch.ops.aten.unsqueeze.default(unsqueeze_81, 3);  unsqueeze_81 = None
        add_41 = torch.ops.aten.add.Tensor(mul_43, unsqueeze_82);  mul_43 = unsqueeze_82 = None
        sigmoid_12 = torch.ops.aten.sigmoid.default(add_41)
        mul_44 = torch.ops.aten.mul.Tensor(add_41, sigmoid_12);  add_41 = sigmoid_12 = None
        convolution_15 = torch.ops.aten.convolution.default(mul_44, arg67_1, arg68_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_44 = arg67_1 = arg68_1 = None
        view_42 = torch.ops.aten.view.default(convolution_15, [4, 32, 16, 65536]);  convolution_15 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(view_42, [2, 3], correction = 0, keepdim = True)
        getitem_32 = var_mean_14[0]
        getitem_33 = var_mean_14[1];  var_mean_14 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_14 = torch.ops.aten.sub.Tensor(view_42, getitem_33);  view_42 = getitem_33 = None
        mul_45 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        view_43 = torch.ops.aten.view.default(mul_45, [4, 512, 256, 256]);  mul_45 = None
        unsqueeze_83 = torch.ops.aten.unsqueeze.default(arg69_1, 0);  arg69_1 = None
        unsqueeze_84 = torch.ops.aten.unsqueeze.default(unsqueeze_83, 2);  unsqueeze_83 = None
        unsqueeze_85 = torch.ops.aten.unsqueeze.default(unsqueeze_84, 3);  unsqueeze_84 = None
        mul_46 = torch.ops.aten.mul.Tensor(view_43, unsqueeze_85);  view_43 = unsqueeze_85 = None
        unsqueeze_86 = torch.ops.aten.unsqueeze.default(arg70_1, 0);  arg70_1 = None
        unsqueeze_87 = torch.ops.aten.unsqueeze.default(unsqueeze_86, 2);  unsqueeze_86 = None
        unsqueeze_88 = torch.ops.aten.unsqueeze.default(unsqueeze_87, 3);  unsqueeze_87 = None
        add_43 = torch.ops.aten.add.Tensor(mul_46, unsqueeze_88);  mul_46 = unsqueeze_88 = None
        sigmoid_13 = torch.ops.aten.sigmoid.default(add_43)
        mul_47 = torch.ops.aten.mul.Tensor(add_43, sigmoid_13);  add_43 = sigmoid_13 = None
        convolution_16 = torch.ops.aten.convolution.default(mul_47, arg71_1, arg72_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_47 = arg71_1 = arg72_1 = None
        add_44 = torch.ops.aten.add.Tensor(div_6, convolution_16);  div_6 = convolution_16 = None
        div_7 = torch.ops.aten.div.Tensor(add_44, 1.0);  add_44 = None
        clone_18 = torch.ops.aten.clone.default(div_7, memory_format = torch.contiguous_format)
        view_44 = torch.ops.aten.view.default(clone_18, [4, 32, 16, 65536]);  clone_18 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(view_44, [2, 3], correction = 0, keepdim = True)
        getitem_34 = var_mean_15[0]
        getitem_35 = var_mean_15[1];  var_mean_15 = None
        add_45 = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_15 = torch.ops.aten.sub.Tensor(view_44, getitem_35);  view_44 = getitem_35 = None
        mul_48 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        view_45 = torch.ops.aten.view.default(mul_48, [4, 512, 256, 256]);  mul_48 = None
        unsqueeze_89 = torch.ops.aten.unsqueeze.default(arg73_1, 0);  arg73_1 = None
        unsqueeze_90 = torch.ops.aten.unsqueeze.default(unsqueeze_89, 2);  unsqueeze_89 = None
        unsqueeze_91 = torch.ops.aten.unsqueeze.default(unsqueeze_90, 3);  unsqueeze_90 = None
        mul_49 = torch.ops.aten.mul.Tensor(view_45, unsqueeze_91);  view_45 = unsqueeze_91 = None
        unsqueeze_92 = torch.ops.aten.unsqueeze.default(arg74_1, 0);  arg74_1 = None
        unsqueeze_93 = torch.ops.aten.unsqueeze.default(unsqueeze_92, 2);  unsqueeze_92 = None
        unsqueeze_94 = torch.ops.aten.unsqueeze.default(unsqueeze_93, 3);  unsqueeze_93 = None
        add_46 = torch.ops.aten.add.Tensor(mul_49, unsqueeze_94);  mul_49 = unsqueeze_94 = None
        sigmoid_14 = torch.ops.aten.sigmoid.default(add_46)
        mul_50 = torch.ops.aten.mul.Tensor(add_46, sigmoid_14);  add_46 = sigmoid_14 = None
        convolution_17 = torch.ops.aten.convolution.default(mul_50, arg75_1, arg76_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_50 = arg75_1 = arg76_1 = None
        view_46 = torch.ops.aten.view.default(convolution_17, [4, 32, 16, 65536]);  convolution_17 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(view_46, [2, 3], correction = 0, keepdim = True)
        getitem_36 = var_mean_16[0]
        getitem_37 = var_mean_16[1];  var_mean_16 = None
        add_47 = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        sub_16 = torch.ops.aten.sub.Tensor(view_46, getitem_37);  view_46 = getitem_37 = None
        mul_51 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        view_47 = torch.ops.aten.view.default(mul_51, [4, 512, 256, 256]);  mul_51 = None
        unsqueeze_95 = torch.ops.aten.unsqueeze.default(arg77_1, 0);  arg77_1 = None
        unsqueeze_96 = torch.ops.aten.unsqueeze.default(unsqueeze_95, 2);  unsqueeze_95 = None
        unsqueeze_97 = torch.ops.aten.unsqueeze.default(unsqueeze_96, 3);  unsqueeze_96 = None
        mul_52 = torch.ops.aten.mul.Tensor(view_47, unsqueeze_97);  view_47 = unsqueeze_97 = None
        unsqueeze_98 = torch.ops.aten.unsqueeze.default(arg78_1, 0);  arg78_1 = None
        unsqueeze_99 = torch.ops.aten.unsqueeze.default(unsqueeze_98, 2);  unsqueeze_98 = None
        unsqueeze_100 = torch.ops.aten.unsqueeze.default(unsqueeze_99, 3);  unsqueeze_99 = None
        add_48 = torch.ops.aten.add.Tensor(mul_52, unsqueeze_100);  mul_52 = unsqueeze_100 = None
        sigmoid_15 = torch.ops.aten.sigmoid.default(add_48)
        mul_53 = torch.ops.aten.mul.Tensor(add_48, sigmoid_15);  add_48 = sigmoid_15 = None
        convolution_18 = torch.ops.aten.convolution.default(mul_53, arg79_1, arg80_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_53 = arg79_1 = arg80_1 = None
        add_49 = torch.ops.aten.add.Tensor(div_7, convolution_18);  div_7 = convolution_18 = None
        div_8 = torch.ops.aten.div.Tensor(add_49, 1.0);  add_49 = None
        iota_2 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_54 = torch.ops.aten.mul.Tensor(iota_2, 1);  iota_2 = None
        add_50 = torch.ops.aten.add.Tensor(mul_54, 0);  mul_54 = None
        convert_element_type_42 = torch.ops.prims.convert_element_type.default(add_50, torch.float32);  add_50 = None
        add_51 = torch.ops.aten.add.Tensor(convert_element_type_42, 0.0);  convert_element_type_42 = None
        mul_55 = torch.ops.aten.mul.Tensor(add_51, 0.5);  add_51 = None
        convert_element_type_43 = torch.ops.prims.convert_element_type.default(mul_55, torch.int64);  mul_55 = None
        unsqueeze_101 = torch.ops.aten.unsqueeze.default(convert_element_type_43, -1);  convert_element_type_43 = None
        iota_3 = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_56 = torch.ops.aten.mul.Tensor(iota_3, 1);  iota_3 = None
        add_52 = torch.ops.aten.add.Tensor(mul_56, 0);  mul_56 = None
        convert_element_type_44 = torch.ops.prims.convert_element_type.default(add_52, torch.float32);  add_52 = None
        add_53 = torch.ops.aten.add.Tensor(convert_element_type_44, 0.0);  convert_element_type_44 = None
        mul_57 = torch.ops.aten.mul.Tensor(add_53, 0.5);  add_53 = None
        convert_element_type_45 = torch.ops.prims.convert_element_type.default(mul_57, torch.int64);  mul_57 = None
        _unsafe_index_1 = torch.ops.aten._unsafe_index.Tensor(div_8, [None, None, unsqueeze_101, convert_element_type_45]);  div_8 = unsqueeze_101 = convert_element_type_45 = None
        clone_20 = torch.ops.aten.clone.default(_unsafe_index_1, memory_format = torch.channels_last);  _unsafe_index_1 = None
        convolution_19 = torch.ops.aten.convolution.default(clone_20, arg81_1, arg82_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  clone_20 = arg81_1 = arg82_1 = None
        clone_21 = torch.ops.aten.clone.default(convolution_19, memory_format = torch.contiguous_format)
        view_48 = torch.ops.aten.view.default(clone_21, [4, 32, 16, 262144]);  clone_21 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(view_48, [2, 3], correction = 0, keepdim = True)
        getitem_38 = var_mean_17[0]
        getitem_39 = var_mean_17[1];  var_mean_17 = None
        add_54 = torch.ops.aten.add.Tensor(getitem_38, 1e-06);  getitem_38 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_17 = torch.ops.aten.sub.Tensor(view_48, getitem_39);  view_48 = getitem_39 = None
        mul_58 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        view_49 = torch.ops.aten.view.default(mul_58, [4, 512, 512, 512]);  mul_58 = None
        unsqueeze_102 = torch.ops.aten.unsqueeze.default(arg83_1, 0);  arg83_1 = None
        unsqueeze_103 = torch.ops.aten.unsqueeze.default(unsqueeze_102, 2);  unsqueeze_102 = None
        unsqueeze_104 = torch.ops.aten.unsqueeze.default(unsqueeze_103, 3);  unsqueeze_103 = None
        mul_59 = torch.ops.aten.mul.Tensor(view_49, unsqueeze_104);  view_49 = unsqueeze_104 = None
        unsqueeze_105 = torch.ops.aten.unsqueeze.default(arg84_1, 0);  arg84_1 = None
        unsqueeze_106 = torch.ops.aten.unsqueeze.default(unsqueeze_105, 2);  unsqueeze_105 = None
        unsqueeze_107 = torch.ops.aten.unsqueeze.default(unsqueeze_106, 3);  unsqueeze_106 = None
        add_55 = torch.ops.aten.add.Tensor(mul_59, unsqueeze_107);  mul_59 = unsqueeze_107 = None
        sigmoid_16 = torch.ops.aten.sigmoid.default(add_55)
        mul_60 = torch.ops.aten.mul.Tensor(add_55, sigmoid_16);  add_55 = sigmoid_16 = None
        convolution_20 = torch.ops.aten.convolution.default(mul_60, arg85_1, arg86_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_60 = arg85_1 = arg86_1 = None
        view_50 = torch.ops.aten.view.default(convolution_20, [4, 32, 8, 262144]);  convolution_20 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(view_50, [2, 3], correction = 0, keepdim = True)
        getitem_40 = var_mean_18[0]
        getitem_41 = var_mean_18[1];  var_mean_18 = None
        add_56 = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_18 = torch.ops.aten.sub.Tensor(view_50, getitem_41);  view_50 = getitem_41 = None
        mul_61 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        view_51 = torch.ops.aten.view.default(mul_61, [4, 256, 512, 512]);  mul_61 = None
        unsqueeze_108 = torch.ops.aten.unsqueeze.default(arg87_1, 0);  arg87_1 = None
        unsqueeze_109 = torch.ops.aten.unsqueeze.default(unsqueeze_108, 2);  unsqueeze_108 = None
        unsqueeze_110 = torch.ops.aten.unsqueeze.default(unsqueeze_109, 3);  unsqueeze_109 = None
        mul_62 = torch.ops.aten.mul.Tensor(view_51, unsqueeze_110);  view_51 = unsqueeze_110 = None
        unsqueeze_111 = torch.ops.aten.unsqueeze.default(arg88_1, 0);  arg88_1 = None
        unsqueeze_112 = torch.ops.aten.unsqueeze.default(unsqueeze_111, 2);  unsqueeze_111 = None
        unsqueeze_113 = torch.ops.aten.unsqueeze.default(unsqueeze_112, 3);  unsqueeze_112 = None
        add_57 = torch.ops.aten.add.Tensor(mul_62, unsqueeze_113);  mul_62 = unsqueeze_113 = None
        sigmoid_17 = torch.ops.aten.sigmoid.default(add_57)
        mul_63 = torch.ops.aten.mul.Tensor(add_57, sigmoid_17);  add_57 = sigmoid_17 = None
        convolution_21 = torch.ops.aten.convolution.default(mul_63, arg89_1, arg90_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_63 = arg89_1 = arg90_1 = None
        clone_23 = torch.ops.aten.clone.default(convolution_19, memory_format = torch.contiguous_format);  convolution_19 = None
        convolution_22 = torch.ops.aten.convolution.default(clone_23, arg91_1, arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clone_23 = arg91_1 = arg92_1 = None
        add_58 = torch.ops.aten.add.Tensor(convolution_22, convolution_21);  convolution_22 = convolution_21 = None
        div_9 = torch.ops.aten.div.Tensor(add_58, 1.0);  add_58 = None
        view_52 = torch.ops.aten.view.default(div_9, [4, 32, 8, 262144])
        var_mean_19 = torch.ops.aten.var_mean.correction(view_52, [2, 3], correction = 0, keepdim = True)
        getitem_42 = var_mean_19[0]
        getitem_43 = var_mean_19[1];  var_mean_19 = None
        add_59 = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        sub_19 = torch.ops.aten.sub.Tensor(view_52, getitem_43);  view_52 = getitem_43 = None
        mul_64 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        view_53 = torch.ops.aten.view.default(mul_64, [4, 256, 512, 512]);  mul_64 = None
        unsqueeze_114 = torch.ops.aten.unsqueeze.default(arg93_1, 0);  arg93_1 = None
        unsqueeze_115 = torch.ops.aten.unsqueeze.default(unsqueeze_114, 2);  unsqueeze_114 = None
        unsqueeze_116 = torch.ops.aten.unsqueeze.default(unsqueeze_115, 3);  unsqueeze_115 = None
        mul_65 = torch.ops.aten.mul.Tensor(view_53, unsqueeze_116);  view_53 = unsqueeze_116 = None
        unsqueeze_117 = torch.ops.aten.unsqueeze.default(arg94_1, 0);  arg94_1 = None
        unsqueeze_118 = torch.ops.aten.unsqueeze.default(unsqueeze_117, 2);  unsqueeze_117 = None
        unsqueeze_119 = torch.ops.aten.unsqueeze.default(unsqueeze_118, 3);  unsqueeze_118 = None
        add_60 = torch.ops.aten.add.Tensor(mul_65, unsqueeze_119);  mul_65 = unsqueeze_119 = None
        sigmoid_18 = torch.ops.aten.sigmoid.default(add_60)
        mul_66 = torch.ops.aten.mul.Tensor(add_60, sigmoid_18);  add_60 = sigmoid_18 = None
        convolution_23 = torch.ops.aten.convolution.default(mul_66, arg95_1, arg96_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_66 = arg95_1 = arg96_1 = None
        view_54 = torch.ops.aten.view.default(convolution_23, [4, 32, 8, 262144]);  convolution_23 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(view_54, [2, 3], correction = 0, keepdim = True)
        getitem_44 = var_mean_20[0]
        getitem_45 = var_mean_20[1];  var_mean_20 = None
        add_61 = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_20 = torch.ops.aten.sub.Tensor(view_54, getitem_45);  view_54 = getitem_45 = None
        mul_67 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        view_55 = torch.ops.aten.view.default(mul_67, [4, 256, 512, 512]);  mul_67 = None
        unsqueeze_120 = torch.ops.aten.unsqueeze.default(arg97_1, 0);  arg97_1 = None
        unsqueeze_121 = torch.ops.aten.unsqueeze.default(unsqueeze_120, 2);  unsqueeze_120 = None
        unsqueeze_122 = torch.ops.aten.unsqueeze.default(unsqueeze_121, 3);  unsqueeze_121 = None
        mul_68 = torch.ops.aten.mul.Tensor(view_55, unsqueeze_122);  view_55 = unsqueeze_122 = None
        unsqueeze_123 = torch.ops.aten.unsqueeze.default(arg98_1, 0);  arg98_1 = None
        unsqueeze_124 = torch.ops.aten.unsqueeze.default(unsqueeze_123, 2);  unsqueeze_123 = None
        unsqueeze_125 = torch.ops.aten.unsqueeze.default(unsqueeze_124, 3);  unsqueeze_124 = None
        add_62 = torch.ops.aten.add.Tensor(mul_68, unsqueeze_125);  mul_68 = unsqueeze_125 = None
        sigmoid_19 = torch.ops.aten.sigmoid.default(add_62)
        mul_69 = torch.ops.aten.mul.Tensor(add_62, sigmoid_19);  add_62 = sigmoid_19 = None
        convolution_24 = torch.ops.aten.convolution.default(mul_69, arg99_1, arg100_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_69 = arg99_1 = arg100_1 = None
        add_63 = torch.ops.aten.add.Tensor(div_9, convolution_24);  div_9 = convolution_24 = None
        div_10 = torch.ops.aten.div.Tensor(add_63, 1.0);  add_63 = None
        view_56 = torch.ops.aten.view.default(div_10, [4, 32, 8, 262144])
        var_mean_21 = torch.ops.aten.var_mean.correction(view_56, [2, 3], correction = 0, keepdim = True)
        getitem_46 = var_mean_21[0]
        getitem_47 = var_mean_21[1];  var_mean_21 = None
        add_64 = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_21 = torch.ops.aten.sub.Tensor(view_56, getitem_47);  view_56 = getitem_47 = None
        mul_70 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        view_57 = torch.ops.aten.view.default(mul_70, [4, 256, 512, 512]);  mul_70 = None
        unsqueeze_126 = torch.ops.aten.unsqueeze.default(arg101_1, 0);  arg101_1 = None
        unsqueeze_127 = torch.ops.aten.unsqueeze.default(unsqueeze_126, 2);  unsqueeze_126 = None
        unsqueeze_128 = torch.ops.aten.unsqueeze.default(unsqueeze_127, 3);  unsqueeze_127 = None
        mul_71 = torch.ops.aten.mul.Tensor(view_57, unsqueeze_128);  view_57 = unsqueeze_128 = None
        unsqueeze_129 = torch.ops.aten.unsqueeze.default(arg102_1, 0);  arg102_1 = None
        unsqueeze_130 = torch.ops.aten.unsqueeze.default(unsqueeze_129, 2);  unsqueeze_129 = None
        unsqueeze_131 = torch.ops.aten.unsqueeze.default(unsqueeze_130, 3);  unsqueeze_130 = None
        add_65 = torch.ops.aten.add.Tensor(mul_71, unsqueeze_131);  mul_71 = unsqueeze_131 = None
        sigmoid_20 = torch.ops.aten.sigmoid.default(add_65)
        mul_72 = torch.ops.aten.mul.Tensor(add_65, sigmoid_20);  add_65 = sigmoid_20 = None
        convolution_25 = torch.ops.aten.convolution.default(mul_72, arg103_1, arg104_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_72 = arg103_1 = arg104_1 = None
        view_58 = torch.ops.aten.view.default(convolution_25, [4, 32, 8, 262144]);  convolution_25 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(view_58, [2, 3], correction = 0, keepdim = True)
        getitem_48 = var_mean_22[0]
        getitem_49 = var_mean_22[1];  var_mean_22 = None
        add_66 = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_22 = torch.ops.aten.sub.Tensor(view_58, getitem_49);  view_58 = getitem_49 = None
        mul_73 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        view_59 = torch.ops.aten.view.default(mul_73, [4, 256, 512, 512]);  mul_73 = None
        unsqueeze_132 = torch.ops.aten.unsqueeze.default(arg105_1, 0);  arg105_1 = None
        unsqueeze_133 = torch.ops.aten.unsqueeze.default(unsqueeze_132, 2);  unsqueeze_132 = None
        unsqueeze_134 = torch.ops.aten.unsqueeze.default(unsqueeze_133, 3);  unsqueeze_133 = None
        mul_74 = torch.ops.aten.mul.Tensor(view_59, unsqueeze_134);  view_59 = unsqueeze_134 = None
        unsqueeze_135 = torch.ops.aten.unsqueeze.default(arg106_1, 0);  arg106_1 = None
        unsqueeze_136 = torch.ops.aten.unsqueeze.default(unsqueeze_135, 2);  unsqueeze_135 = None
        unsqueeze_137 = torch.ops.aten.unsqueeze.default(unsqueeze_136, 3);  unsqueeze_136 = None
        add_67 = torch.ops.aten.add.Tensor(mul_74, unsqueeze_137);  mul_74 = unsqueeze_137 = None
        sigmoid_21 = torch.ops.aten.sigmoid.default(add_67)
        mul_75 = torch.ops.aten.mul.Tensor(add_67, sigmoid_21);  add_67 = sigmoid_21 = None
        convolution_26 = torch.ops.aten.convolution.default(mul_75, arg107_1, arg108_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_75 = arg107_1 = arg108_1 = None
        add_68 = torch.ops.aten.add.Tensor(div_10, convolution_26);  div_10 = convolution_26 = None
        div_11 = torch.ops.aten.div.Tensor(add_68, 1.0);  add_68 = None
        iota_4 = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_76 = torch.ops.aten.mul.Tensor(iota_4, 1);  iota_4 = None
        add_69 = torch.ops.aten.add.Tensor(mul_76, 0);  mul_76 = None
        convert_element_type_46 = torch.ops.prims.convert_element_type.default(add_69, torch.float32);  add_69 = None
        add_70 = torch.ops.aten.add.Tensor(convert_element_type_46, 0.0);  convert_element_type_46 = None
        mul_77 = torch.ops.aten.mul.Tensor(add_70, 0.5);  add_70 = None
        convert_element_type_47 = torch.ops.prims.convert_element_type.default(mul_77, torch.int64);  mul_77 = None
        unsqueeze_138 = torch.ops.aten.unsqueeze.default(convert_element_type_47, -1);  convert_element_type_47 = None
        iota_5 = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_78 = torch.ops.aten.mul.Tensor(iota_5, 1);  iota_5 = None
        add_71 = torch.ops.aten.add.Tensor(mul_78, 0);  mul_78 = None
        convert_element_type_48 = torch.ops.prims.convert_element_type.default(add_71, torch.float32);  add_71 = None
        add_72 = torch.ops.aten.add.Tensor(convert_element_type_48, 0.0);  convert_element_type_48 = None
        mul_79 = torch.ops.aten.mul.Tensor(add_72, 0.5);  add_72 = None
        convert_element_type_49 = torch.ops.prims.convert_element_type.default(mul_79, torch.int64);  mul_79 = None
        _unsafe_index_2 = torch.ops.aten._unsafe_index.Tensor(div_11, [None, None, unsqueeze_138, convert_element_type_49]);  div_11 = unsqueeze_138 = convert_element_type_49 = None
        convolution_27 = torch.ops.aten.convolution.default(_unsafe_index_2, arg109_1, arg110_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  _unsafe_index_2 = arg109_1 = arg110_1 = None
        view_60 = torch.ops.aten.view.default(convolution_27, [4, 32, 8, 1048576])
        var_mean_23 = torch.ops.aten.var_mean.correction(view_60, [2, 3], correction = 0, keepdim = True)
        getitem_50 = var_mean_23[0]
        getitem_51 = var_mean_23[1];  var_mean_23 = None
        add_73 = torch.ops.aten.add.Tensor(getitem_50, 1e-06);  getitem_50 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        sub_23 = torch.ops.aten.sub.Tensor(view_60, getitem_51);  view_60 = getitem_51 = None
        mul_80 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        view_61 = torch.ops.aten.view.default(mul_80, [4, 256, 1024, 1024]);  mul_80 = None
        unsqueeze_139 = torch.ops.aten.unsqueeze.default(arg111_1, 0);  arg111_1 = None
        unsqueeze_140 = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
        unsqueeze_141 = torch.ops.aten.unsqueeze.default(unsqueeze_140, 3);  unsqueeze_140 = None
        mul_81 = torch.ops.aten.mul.Tensor(view_61, unsqueeze_141);  view_61 = unsqueeze_141 = None
        unsqueeze_142 = torch.ops.aten.unsqueeze.default(arg112_1, 0);  arg112_1 = None
        unsqueeze_143 = torch.ops.aten.unsqueeze.default(unsqueeze_142, 2);  unsqueeze_142 = None
        unsqueeze_144 = torch.ops.aten.unsqueeze.default(unsqueeze_143, 3);  unsqueeze_143 = None
        add_74 = torch.ops.aten.add.Tensor(mul_81, unsqueeze_144);  mul_81 = unsqueeze_144 = None
        sigmoid_22 = torch.ops.aten.sigmoid.default(add_74)
        mul_82 = torch.ops.aten.mul.Tensor(add_74, sigmoid_22);  add_74 = sigmoid_22 = None
        convolution_28 = torch.ops.aten.convolution.default(mul_82, arg113_1, arg114_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_82 = arg113_1 = arg114_1 = None
        view_62 = torch.ops.aten.view.default(convolution_28, [4, 32, 4, 1048576]);  convolution_28 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(view_62, [2, 3], correction = 0, keepdim = True)
        getitem_52 = var_mean_24[0]
        getitem_53 = var_mean_24[1];  var_mean_24 = None
        add_75 = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        sub_24 = torch.ops.aten.sub.Tensor(view_62, getitem_53);  view_62 = getitem_53 = None
        mul_83 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        view_63 = torch.ops.aten.view.default(mul_83, [4, 128, 1024, 1024]);  mul_83 = None
        unsqueeze_145 = torch.ops.aten.unsqueeze.default(arg115_1, 0);  arg115_1 = None
        unsqueeze_146 = torch.ops.aten.unsqueeze.default(unsqueeze_145, 2);  unsqueeze_145 = None
        unsqueeze_147 = torch.ops.aten.unsqueeze.default(unsqueeze_146, 3);  unsqueeze_146 = None
        mul_84 = torch.ops.aten.mul.Tensor(view_63, unsqueeze_147);  view_63 = unsqueeze_147 = None
        unsqueeze_148 = torch.ops.aten.unsqueeze.default(arg116_1, 0);  arg116_1 = None
        unsqueeze_149 = torch.ops.aten.unsqueeze.default(unsqueeze_148, 2);  unsqueeze_148 = None
        unsqueeze_150 = torch.ops.aten.unsqueeze.default(unsqueeze_149, 3);  unsqueeze_149 = None
        add_76 = torch.ops.aten.add.Tensor(mul_84, unsqueeze_150);  mul_84 = unsqueeze_150 = None
        sigmoid_23 = torch.ops.aten.sigmoid.default(add_76)
        mul_85 = torch.ops.aten.mul.Tensor(add_76, sigmoid_23);  add_76 = sigmoid_23 = None
        convolution_29 = torch.ops.aten.convolution.default(mul_85, arg117_1, arg118_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_85 = arg117_1 = arg118_1 = None
        convolution_30 = torch.ops.aten.convolution.default(convolution_27, arg119_1, arg120_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convolution_27 = arg119_1 = arg120_1 = None
        add_77 = torch.ops.aten.add.Tensor(convolution_30, convolution_29);  convolution_30 = convolution_29 = None
        div_12 = torch.ops.aten.div.Tensor(add_77, 1.0);  add_77 = None
        view_64 = torch.ops.aten.view.default(div_12, [4, 32, 4, 1048576])
        var_mean_25 = torch.ops.aten.var_mean.correction(view_64, [2, 3], correction = 0, keepdim = True)
        getitem_54 = var_mean_25[0]
        getitem_55 = var_mean_25[1];  var_mean_25 = None
        add_78 = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_25 = torch.ops.aten.sub.Tensor(view_64, getitem_55);  view_64 = getitem_55 = None
        mul_86 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None
        view_65 = torch.ops.aten.view.default(mul_86, [4, 128, 1024, 1024]);  mul_86 = None
        unsqueeze_151 = torch.ops.aten.unsqueeze.default(arg121_1, 0);  arg121_1 = None
        unsqueeze_152 = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
        unsqueeze_153 = torch.ops.aten.unsqueeze.default(unsqueeze_152, 3);  unsqueeze_152 = None
        mul_87 = torch.ops.aten.mul.Tensor(view_65, unsqueeze_153);  view_65 = unsqueeze_153 = None
        unsqueeze_154 = torch.ops.aten.unsqueeze.default(arg122_1, 0);  arg122_1 = None
        unsqueeze_155 = torch.ops.aten.unsqueeze.default(unsqueeze_154, 2);  unsqueeze_154 = None
        unsqueeze_156 = torch.ops.aten.unsqueeze.default(unsqueeze_155, 3);  unsqueeze_155 = None
        add_79 = torch.ops.aten.add.Tensor(mul_87, unsqueeze_156);  mul_87 = unsqueeze_156 = None
        sigmoid_24 = torch.ops.aten.sigmoid.default(add_79)
        mul_88 = torch.ops.aten.mul.Tensor(add_79, sigmoid_24);  add_79 = sigmoid_24 = None
        convolution_31 = torch.ops.aten.convolution.default(mul_88, arg123_1, arg124_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_88 = arg123_1 = arg124_1 = None
        view_66 = torch.ops.aten.view.default(convolution_31, [4, 32, 4, 1048576]);  convolution_31 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(view_66, [2, 3], correction = 0, keepdim = True)
        getitem_56 = var_mean_26[0]
        getitem_57 = var_mean_26[1];  var_mean_26 = None
        add_80 = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_26 = torch.ops.aten.sub.Tensor(view_66, getitem_57);  view_66 = getitem_57 = None
        mul_89 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = rsqrt_26 = None
        view_67 = torch.ops.aten.view.default(mul_89, [4, 128, 1024, 1024]);  mul_89 = None
        unsqueeze_157 = torch.ops.aten.unsqueeze.default(arg125_1, 0);  arg125_1 = None
        unsqueeze_158 = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
        unsqueeze_159 = torch.ops.aten.unsqueeze.default(unsqueeze_158, 3);  unsqueeze_158 = None
        mul_90 = torch.ops.aten.mul.Tensor(view_67, unsqueeze_159);  view_67 = unsqueeze_159 = None
        unsqueeze_160 = torch.ops.aten.unsqueeze.default(arg126_1, 0);  arg126_1 = None
        unsqueeze_161 = torch.ops.aten.unsqueeze.default(unsqueeze_160, 2);  unsqueeze_160 = None
        unsqueeze_162 = torch.ops.aten.unsqueeze.default(unsqueeze_161, 3);  unsqueeze_161 = None
        add_81 = torch.ops.aten.add.Tensor(mul_90, unsqueeze_162);  mul_90 = unsqueeze_162 = None
        sigmoid_25 = torch.ops.aten.sigmoid.default(add_81)
        mul_91 = torch.ops.aten.mul.Tensor(add_81, sigmoid_25);  add_81 = sigmoid_25 = None
        convolution_32 = torch.ops.aten.convolution.default(mul_91, arg127_1, arg128_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_91 = arg127_1 = arg128_1 = None
        add_82 = torch.ops.aten.add.Tensor(div_12, convolution_32);  div_12 = convolution_32 = None
        div_13 = torch.ops.aten.div.Tensor(add_82, 1.0);  add_82 = None
        view_68 = torch.ops.aten.view.default(div_13, [4, 32, 4, 1048576])
        var_mean_27 = torch.ops.aten.var_mean.correction(view_68, [2, 3], correction = 0, keepdim = True)
        getitem_58 = var_mean_27[0]
        getitem_59 = var_mean_27[1];  var_mean_27 = None
        add_83 = torch.ops.aten.add.Tensor(getitem_58, 1e-06);  getitem_58 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        sub_27 = torch.ops.aten.sub.Tensor(view_68, getitem_59);  view_68 = getitem_59 = None
        mul_92 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = rsqrt_27 = None
        view_69 = torch.ops.aten.view.default(mul_92, [4, 128, 1024, 1024]);  mul_92 = None
        unsqueeze_163 = torch.ops.aten.unsqueeze.default(arg129_1, 0);  arg129_1 = None
        unsqueeze_164 = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
        unsqueeze_165 = torch.ops.aten.unsqueeze.default(unsqueeze_164, 3);  unsqueeze_164 = None
        mul_93 = torch.ops.aten.mul.Tensor(view_69, unsqueeze_165);  view_69 = unsqueeze_165 = None
        unsqueeze_166 = torch.ops.aten.unsqueeze.default(arg130_1, 0);  arg130_1 = None
        unsqueeze_167 = torch.ops.aten.unsqueeze.default(unsqueeze_166, 2);  unsqueeze_166 = None
        unsqueeze_168 = torch.ops.aten.unsqueeze.default(unsqueeze_167, 3);  unsqueeze_167 = None
        add_84 = torch.ops.aten.add.Tensor(mul_93, unsqueeze_168);  mul_93 = unsqueeze_168 = None
        sigmoid_26 = torch.ops.aten.sigmoid.default(add_84)
        mul_94 = torch.ops.aten.mul.Tensor(add_84, sigmoid_26);  add_84 = sigmoid_26 = None
        convolution_33 = torch.ops.aten.convolution.default(mul_94, arg131_1, arg132_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_94 = arg131_1 = arg132_1 = None
        view_70 = torch.ops.aten.view.default(convolution_33, [4, 32, 4, 1048576]);  convolution_33 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(view_70, [2, 3], correction = 0, keepdim = True)
        getitem_60 = var_mean_28[0]
        getitem_61 = var_mean_28[1];  var_mean_28 = None
        add_85 = torch.ops.aten.add.Tensor(getitem_60, 1e-06);  getitem_60 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_28 = torch.ops.aten.sub.Tensor(view_70, getitem_61);  view_70 = getitem_61 = None
        mul_95 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = rsqrt_28 = None
        view_71 = torch.ops.aten.view.default(mul_95, [4, 128, 1024, 1024]);  mul_95 = None
        unsqueeze_169 = torch.ops.aten.unsqueeze.default(arg133_1, 0);  arg133_1 = None
        unsqueeze_170 = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
        unsqueeze_171 = torch.ops.aten.unsqueeze.default(unsqueeze_170, 3);  unsqueeze_170 = None
        mul_96 = torch.ops.aten.mul.Tensor(view_71, unsqueeze_171);  view_71 = unsqueeze_171 = None
        unsqueeze_172 = torch.ops.aten.unsqueeze.default(arg134_1, 0);  arg134_1 = None
        unsqueeze_173 = torch.ops.aten.unsqueeze.default(unsqueeze_172, 2);  unsqueeze_172 = None
        unsqueeze_174 = torch.ops.aten.unsqueeze.default(unsqueeze_173, 3);  unsqueeze_173 = None
        add_86 = torch.ops.aten.add.Tensor(mul_96, unsqueeze_174);  mul_96 = unsqueeze_174 = None
        sigmoid_27 = torch.ops.aten.sigmoid.default(add_86)
        mul_97 = torch.ops.aten.mul.Tensor(add_86, sigmoid_27);  add_86 = sigmoid_27 = None
        convolution_34 = torch.ops.aten.convolution.default(mul_97, arg135_1, arg136_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_97 = arg135_1 = arg136_1 = None
        add_87 = torch.ops.aten.add.Tensor(div_13, convolution_34);  div_13 = convolution_34 = None
        div_14 = torch.ops.aten.div.Tensor(add_87, 1.0);  add_87 = None
        view_72 = torch.ops.aten.view.default(div_14, [4, 32, 4, 1048576]);  div_14 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(view_72, [2, 3], correction = 0, keepdim = True)
        getitem_62 = var_mean_29[0]
        getitem_63 = var_mean_29[1];  var_mean_29 = None
        add_88 = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_29 = torch.ops.aten.sub.Tensor(view_72, getitem_63);  view_72 = getitem_63 = None
        mul_98 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = rsqrt_29 = None
        view_73 = torch.ops.aten.view.default(mul_98, [4, 128, 1024, 1024]);  mul_98 = None
        unsqueeze_175 = torch.ops.aten.unsqueeze.default(arg137_1, 0);  arg137_1 = None
        unsqueeze_176 = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None
        unsqueeze_177 = torch.ops.aten.unsqueeze.default(unsqueeze_176, 3);  unsqueeze_176 = None
        mul_99 = torch.ops.aten.mul.Tensor(view_73, unsqueeze_177);  view_73 = unsqueeze_177 = None
        unsqueeze_178 = torch.ops.aten.unsqueeze.default(arg138_1, 0);  arg138_1 = None
        unsqueeze_179 = torch.ops.aten.unsqueeze.default(unsqueeze_178, 2);  unsqueeze_178 = None
        unsqueeze_180 = torch.ops.aten.unsqueeze.default(unsqueeze_179, 3);  unsqueeze_179 = None
        add_89 = torch.ops.aten.add.Tensor(mul_99, unsqueeze_180);  mul_99 = unsqueeze_180 = None
        sigmoid_28 = torch.ops.aten.sigmoid.default(add_89)
        mul_100 = torch.ops.aten.mul.Tensor(add_89, sigmoid_28);  add_89 = sigmoid_28 = None
        convolution_35 = torch.ops.aten.convolution.default(mul_100, arg139_1, arg140_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_100 = arg139_1 = arg140_1 = None
        return (convolution_35,)
        
def load_args(reader):
    buf0 = reader.storage(None, 32, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf0, (4, 4, 1, 1), dtype=torch.float16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1, (4,), dtype=torch.float16, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf2, (4, 4, 128, 128), dtype=torch.float16, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 36864, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf3, (512, 4, 3, 3), dtype=torch.float16, is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf4, (512,), dtype=torch.float16, is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf5, (512,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf6, (512,), dtype=torch.float16, is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf7, (512,), dtype=torch.float16, is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf8, (512, 512, 3, 3), dtype=torch.float16, is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf9, (512,), dtype=torch.float16, is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf10, (512,), dtype=torch.float16, is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf11, (512,), dtype=torch.float16, is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf12, (512, 512, 3, 3), dtype=torch.float16, is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf13, (512,), dtype=torch.float16, is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf14, (512,), dtype=torch.float16, is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf15, (512,), dtype=torch.float16, is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf16, (512, 512), dtype=torch.float16, is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf17, (512,), dtype=torch.float16, is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf18, (512, 512), dtype=torch.float16, is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf19, (512,), dtype=torch.float16, is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf20, (512, 512), dtype=torch.float16, is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf21, (512,), dtype=torch.float16, is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf22, (512, 512), dtype=torch.float16, is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf23, (512,), dtype=torch.float16, is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf24, (512,), dtype=torch.float16, is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf25, (512,), dtype=torch.float16, is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf26, (512, 512, 3, 3), dtype=torch.float16, is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf27, (512,), dtype=torch.float16, is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf28, (512,), dtype=torch.float16, is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf29, (512,), dtype=torch.float16, is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf30, (512, 512, 3, 3), dtype=torch.float16, is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf31, (512,), dtype=torch.float16, is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf32, (512,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf33, (512, 512, 3, 3), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf34, (512,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf35, (512,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf36, (512,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf37, (512, 512, 3, 3), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf38, (512,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf39, (512,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf40, (512,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf41, (512, 512, 3, 3), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf42, (512,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf43, (512,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf44, (512,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf45, (512, 512, 3, 3), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf46, (512,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf47, (512,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf48, (512,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf49, (512, 512, 3, 3), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf50, (512,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf51, (512,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf52, (512,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf53, (512, 512, 3, 3), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf54, (512,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf55, (512, 512, 3, 3), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf56, (512,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf57, (512,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf58, (512,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf59, (512, 512, 3, 3), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf60, (512,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf61, (512,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf62, (512,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf63, (512, 512, 3, 3), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf64, (512,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf65, (512,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf66, (512,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf67, (512, 512, 3, 3), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf68, (512,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf69, (512,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf70, (512,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf71, (512, 512, 3, 3), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf72, (512,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf73, (512,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf74, (512,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf75, (512, 512, 3, 3), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf76, (512,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf77, (512,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf78, (512,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf79, (512, 512, 3, 3), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf80, (512,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf81, (512, 512, 3, 3), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf82, (512,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf83, (512,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf84, (512,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf85, (256, 512, 3, 3), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf86, (256,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf87, (256,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf88, (256,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf89, (256, 256, 3, 3), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf90, (256,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf91, (256, 512, 1, 1), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf92, (256,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf93, (256,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf94, (256,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf95, (256, 256, 3, 3), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf96, (256,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf97, (256,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf98, (256,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf99, (256, 256, 3, 3), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf100, (256,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf101, (256,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf102, (256,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf103, (256, 256, 3, 3), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf104, (256,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf105, (256,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf106, (256,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf107, (256, 256, 3, 3), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf108, (256,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf109, (256, 256, 3, 3), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf110, (256,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf111, (256,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf112, (256,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf113, (128, 256, 3, 3), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf114, (128,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf115, (128,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf116, (128,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf117, (128, 128, 3, 3), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf118, (128,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf119, (128, 256, 1, 1), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf120, (128,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf121, (128,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf122, (128,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf123, (128, 128, 3, 3), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf124, (128,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf125, (128,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf126, (128,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf127, (128, 128, 3, 3), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf128, (128,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf129, (128,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf130, (128,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf131, (128, 128, 3, 3), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf132, (128,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf133, (128,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf134, (128,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf135, (128, 128, 3, 3), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf136, (128,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf137, (128,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf138, (128,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 13824, device=device(type='cuda', index=0))
    reader.tensor(buf139, (3, 128, 3, 3), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 12, device=device(type='cuda', index=0))
    reader.tensor(buf140, (3,), is_leaf=True)  # arg140_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)