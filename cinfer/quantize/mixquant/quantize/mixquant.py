import torch
from typing import Dict, List, Optional
import logging
import functools
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict
from mixquant.utils.utils import clear_memory
from mixquant.utils.calib_data import get_calib_dataset
from mixquant.modules.linear import MixLinear_GEMM

from mixquant.utils.module import get_named_linears, set_op_by_name, weight_only_map, eightbit_only_name


class MixQuantizer:
    def __init__(self, f16_model, model, tokenizer, w_bit, group_size, version, weight_only_map_str = None) -> None:
        self.f16_model = f16_model
        self.model = model
        self.tokenizer = tokenizer

        self.group_size = group_size
        self.version = version
        self.w_bit = w_bit

        self.modules, self.module_kwargs= self.init_quant()
        self.weight_only_map = weight_only_map_str
    def init_quant(self, n_samples=128, seqlen=512):
        modules = self.f16_model.get_model_layers(self.model)


        inps = []
        layer_kwargs = {}

        modules[0] = modules[0].cuda()
        self.f16_model.move_embed(self.model, "cuda")
        
        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, hijacked_inputs, **kwargs):
                inps.append(hijacked_inputs)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        modules[0] = Catcher(modules[0])

        modules[0] = modules[0].module  # restore

        modules[0] = modules[0].cpu()
        self.f16_model.move_embed(self.model, "cpu")
        
        clear_memory()
        
        if "attention_mask" in layer_kwargs.keys():
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to("cuda")

        return modules, layer_kwargs
    

    def quantize(self,weight_only = False):
        for i in tqdm(range(len(self.modules)), desc="Mix quant"):

            self.modules[i] = self.modules[i].cuda()
            named_linears = get_named_linears(self.modules[i])

            clear_memory()

            # Quantize weights
            self._apply_quant(self.modules[i], named_linears, weight_only, layer = i)
            clear_memory()
 

    def quantize_QUIK(self,weight_only = False):
        for i in tqdm(range(len(self.modules)), desc="QUIK quant"):

            self.modules[i] = self.modules[i].cuda()
            named_linears = get_named_linears(self.modules[i])

            clear_memory()

            # Quantize weights
            self._apply_quant_quik(self.modules[i], named_linears, weight_only, layer = i)
            clear_memory()

    def _apply_quant_quik(self, module, named_linears: Dict[str, nn.Linear], weight_only_, layer):

        
        if isinstance(self.model.config.architectures,list):
            name = self.model.config.architectures[0]
        else:
            name = self.model.config.architectures
        weight_only_name = weight_only_map[ name ]
 
        for name, linear_layer in named_linears.items():


            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            linear_layer = linear_layer.cuda().half()

            if self.version == 'QUIK':
                from mixquant.modules.qlinear import MixedQLinear
                q_linear_module = MixedQLinear

            else:
                raise NotImplementedError
            
            # for same small blocks we do not need the mixquant, we only use the weight only quant

            bit_8_only = False


                
            w_bit = self.w_bit
            assert w_bit == 4
            for key in eightbit_only_name:
                if key in  name:
                    w_bit = 8
                    bit_8_only = False 

   
            relative_path = "act_scales/%s.pt"%(self.model.config._name_or_path.split("/")[-1])
            act_scales = torch.load(relative_path)
            if 'opt' in self.model.config._name_or_path.split("/")[-1]:
                layer_scales = act_scales['model.decoder.layers.{}.{}'.format(layer, name)]
            else:

                layer_scales = act_scales['model.layers.{}.{}'.format(layer, name)]

            fp_features = 256
            fp_indices = torch.sort(layer_scales)[1][-fp_features:]  
            if bit_8_only is True:
                  
                w_bit = 8

                scales =  (torch.max(torch.abs(linear_layer.weight.data), dim=1)[0].unsqueeze(1) / (
                                127)).to(torch.float16).reshape((linear_layer.out_features, 1))
                q_linear = q_linear_module.from_linear(linear_layer,
                    weight_matrix=linear_layer.weight.data,
                    weights_scales = scales,
                    bits = 8,
                    fp_indices = fp_indices
                )
            else:
                scales =  (torch.max(torch.abs(linear_layer.weight.data), dim=1)[0].unsqueeze(1) / (
                                7)).to(torch.float16).reshape((linear_layer.out_features, 1))
                q_linear = q_linear_module.from_linear(linear_layer,
                    weight_matrix = linear_layer.weight.data,
                    weights_scales = scales,
                    bits = 4,
                    fp_indices = fp_indices
                )

            linear_layer.cpu()

            set_op_by_name(module, name, q_linear)
            clear_memory()
            
    def _apply_quant(self, module, named_linears: Dict[str, nn.Linear], weight_only_, layer):

        
        if isinstance(self.model.config.architectures,list):
            name = self.model.config.architectures[0]
        else:
            name = self.model.config.architectures
        weight_only_name = weight_only_map[ name ]
 
        for name, linear_layer in named_linears.items():


            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            linear_layer = linear_layer.cuda().half()

            if self.version == 'MIX':
                q_linear_module = MixLinear_GEMM

            else:
                raise NotImplementedError
            
            # for same small blocks we do not need the mixquant, we only use the weight only quant

            weight_only = False

            for key in weight_only_name:
                if key in  name:
                    weight_only = True
                    break
                
            w_bit = self.w_bit
            if w_bit == 4:
                for key in eightbit_only_name:
                    if key in  name:
                        w_bit = 8
                        weight_only = False 

            if w_bit == 4:
                relative_path = "act_scales/%s.pt"%(self.model.config._name_or_path.split("/")[-1])
                act_scales = torch.load(relative_path)
                if 'opt' in self.model.config._name_or_path.split("/")[-1]:
                    layer_scales = act_scales['model.decoder.layers.{}.{}'.format(layer, name)]
                else:

                    layer_scales = act_scales['model.layers.{}.{}'.format(layer, name)]
            else:
                layer_scales = None

            if self.weight_only_map is not None:
                for i in self.weight_only_map.split(","):
                    if i in name:
                        weight_only = True
                        #print(name + "use weight only forward!!")
            if weight_only is True:


                q_linear = q_linear_module.from_linear(
                    linear=linear_layer,
                    weight_only = weight_only,
                    init_only=False,
                    bit = w_bit,
                    layer_scales = layer_scales
                )

                # # weight_only awq
                # import sys
                # sys.path.append('/home/chenyidong/quant/AutoAWQ1.0')
                # from awq.modules.linear import (
                #     WQLinear_GEMM,
                # )
                # #from awq.utils.utils import  get_best_device
                # linear_layer = linear_layer.half()
                # linear_layer.weight.data, scales, zeros = self.pseudo_quantize_tensor(
                # linear_layer.weight.data )
                # scales = scales.t().contiguous()
                # zeros = zeros.t().contiguous()

                # print("awq start")

                # q_linear_module = WQLinear_GEMM
                # q_linear = q_linear_module.from_linear(
                #     linear=linear_layer,
                #     w_bit=4,
                #     group_size=128,
                #     init_only=False,
                #     scales=scales,
                #     zeros=zeros,
                # )
                # print("awq done")

            else:
                q_linear = q_linear_module.from_linear(
                    linear=linear_layer,
                    weight_only = weight_only,
                    init_only = False,
                    bit = w_bit,
                    layer_scales = layer_scales
                )

            linear_layer.cpu()
            q_linear = q_linear.cpu()

            set_op_by_name(module, name, q_linear)
            clear_memory()

    def pseudo_dequantize_tensor(
        self, w: nn.Linear, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None
    ):
        # get repeated count
        repeat_count = w.weight.data.shape[-1] // scales.shape[-1]
        scales = scales.repeat(1, repeat_count).reshape(w.weight.data.shape)

        # dequantize
        if self.zero_point:
            zeros = zeros.repeat(1, repeat_count).reshape(w.weight.data.shape)
            w = (w.weight.data - zeros) * scales
        else:
            w = w.weight.data * scales

        return w
    def pseudo_quantize_tensor(self, w: torch.Tensor):
        org_w_shape = w.shape
        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0
            w = w.reshape(-1, self.group_size)
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0
        self.zero_point = True
        # zero point quantization
        if self.zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2**self.w_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            w = (
                torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
            ) * scales
            zeros = zeros.view(org_w_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (self.w_bit - 1) - 1
            min_int = -(2 ** (self.w_bit - 1))
            scales = max_val / max_int
            zeros = None
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        scales = scales.view(org_w_shape[0], -1)
        w = w.reshape(org_w_shape)

        return w, scales, zeros