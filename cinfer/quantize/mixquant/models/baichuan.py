from .base import BaseForCausalLM
from typing import Dict
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM

class BaichuanMixQForCausalLM(BaseForCausalLM):
    layer_type = "LlamaDecoderLayer"
    max_new_tokens_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model: LlamaForCausalLM, quant_config: Dict, mix = False, cache = None):
 
        fuser = LlamaFuser(model, quant_config)
        
        fuser.fuse_attention(MixGemmCache = cache)
        
        fuser.fuse_mlp(mix, MixGemmCache = cache)
        fuser.fuse_rmsnorm(MixGemmCache = cache)


        for layer in model.model.layers:
            layer.input_layernorm.next_layer = layer.self_attn.W_pack
            layer.post_attention_layernorm.next_layer = layer.mlp.up_proj_ 

    @staticmethod
    def get_model_layers(model: LlamaForCausalLM):
        return model.model.layers
    
 
    
    @staticmethod
    def move_embed(model: LlamaForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
  

import torch
from typing import List, Tuple, Union
from mixquant.utils.utils import set_module_name
from mixquant.modules.fused.mlp import  MixLlamaMLP
from mixquant.modules.fused.attn import QuantAttentionFused, QuantAttentionFusedBaichuan13B
from mixquant.modules.fused.norm import FasterTransformerRMSNorm
from mixquant.modules.linear import  MixLinear_GEMM

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm, LlamaMLP
import sys


#from  modeling_baichuan import Attention
class LlamaFuser:
    def __init__(self, model, quant_config):
        self.model = model
        self.quant_config = quant_config

        #print(model.model.layers[0].self_attn.o_proj) # 确认一下模型的权重的格式
 
        #需要加入百川的 Attention
        self.attention_modules: List[Tuple[str, LlamaAttention]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, LlamaAttention) or  "Attention" in str(module.__class__)
        ]
        #print(self.attention_modules)

        self.rmsnorm_modules: List[Tuple[str, LlamaRMSNorm]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, LlamaRMSNorm)   or  "RMSNorm" in str(module.__class__)
        ]
        
        self.mlp_modules: List[Tuple[str, LlamaMLP]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, LlamaMLP)   or  "MLP" in str(module.__class__)
        ]
    
    def fuse_attention(self, MixGemmCache):
        for name, module in self.attention_modules:
            qkv_layer  = self._fuse_qkv(module, MixGemmCache)
            try:
                num_key_value_heads = module.num_key_value_heads
            except:
                # 为了处理百川的模型
                num_key_value_heads = 32

            if self.model.config.num_hidden_layers == 40:
                attn = QuantAttentionFusedBaichuan13B(
                    module.hidden_size,
                    module.num_heads,
                    num_key_value_heads,
                    qkv_layer, 
                    module.o_proj,
                    next(iter(qkv_layer.state_dict().values())).device,
                    self.model.config.max_new_tokens,
                    MixGemmCache = MixGemmCache
                )

            else:
                attn = QuantAttentionFused(
                    module.hidden_size,
                    module.num_heads,
                    num_key_value_heads,
                    qkv_layer, 
                    module.o_proj,
                    next(iter(qkv_layer.state_dict().values())).device,
                    self.model.config.max_new_tokens,
                    MixGemmCache = MixGemmCache
                )
            set_module_name(self.model, name, attn)
    
    def fuse_attention(self, MixGemmCache):
        
        for name, module in self.attention_modules:

            layer_idx = int(name.split('.')[2])
            qkv_layer  = self._fuse_qkv(module, MixGemmCache)
            try:
                num_key_value_heads = module.num_key_value_heads
            except:
                # 为了处理百川的模型
                print("do not find the attr module.num_key_value_heads")
                num_key_value_heads = 32
            attn = QuantAttentionFused(
                module.hidden_size,
                module.num_heads,
                num_key_value_heads,
                qkv_layer, 
                module.o_proj,
                next(iter(qkv_layer.state_dict().values())).device,
                self.model.config.max_new_tokens,
                MixGemmCache = MixGemmCache,
                layer_idx = layer_idx
            )
            set_module_name(self.model, name, attn)
    
    def _fuse_qkv(self, module: LlamaAttention,cache):
        try:
            q_proj, k_proj, v_proj = module.q_proj, module.k_proj, module.v_proj
        except:
            qkv_layer = module.W_pack
            return qkv_layer
 
 
        
        if not  isinstance(q_proj, MixLinear_GEMM) :
            raise "no implement error"
 
        if isinstance(q_proj, MixLinear_GEMM):
            qkv_layer = MixLinear_GEMM(q_proj.in_features,q_proj.out_features + k_proj.out_features + v_proj.out_features,
                                        q_proj.bias is not None,
                                        next(iter(module.state_dict().values())).device,
                                        bit = self.quant_config['w_bit'],
                                        weight_only=False,
                                        cache=cache)


        
        if isinstance(qkv_layer, MixLinear_GEMM):
            shapew = qkv_layer.q_weight.shape
            
            if qkv_layer.weight_only:
                qkv_layer.q_weight = torch.cat([q_proj.q_weight, k_proj.q_weight, v_proj.q_weight], dim=1)
                qkv_layer.scale_col = torch.cat([q_proj.scale_col, k_proj.scale_col, v_proj.scale_col], dim=0)
                
            else:
                qkv_layer.q_weight = torch.cat([q_proj.q_weight, k_proj.q_weight, v_proj.q_weight], dim=0)
                qkv_layer.scale_col = torch.cat([q_proj.scale_col, k_proj.scale_col, v_proj.scale_col], dim=1)
                assert shapew[0] == qkv_layer.q_weight.shape[0]
                assert shapew[1] == qkv_layer.q_weight.shape[1]
                assert shapew[0] == qkv_layer.scale_col.shape[1]
                assert 1 == qkv_layer.scale_col.shape[0]
            if self.quant_config['w_bit'] == 4:


                
                qkv_layer.weight_cache.copy_(torch.cat([q_proj.weight_cache, 
                                                        k_proj.weight_cache, 
                                                        v_proj.weight_cache], dim=0))
      
 

                qkv_layer.ind.copy_(q_proj.ind)


  



            if q_proj.bias is not None:
                raise NotImplementedError
            else:
                qkv_layer.bias = None

        else:
            raise "no implement"
        
        q_proj.q_weight = q_proj.q_weight.to('cpu')
        k_proj.q_weight = k_proj.q_weight.to('cpu')
        v_proj.q_weight = v_proj.q_weight.to('cpu')
        q_proj.scale_col = q_proj.scale_col.to('cpu')
        k_proj.scale_col = k_proj.scale_col.to('cpu')
        v_proj.scale_col = v_proj.scale_col.to('cpu')
        torch.cuda.empty_cache()
        return qkv_layer

    def fuse_rmsnorm(self, MixGemmCache):
        for name, module in self.rmsnorm_modules:
            norm = FasterTransformerRMSNorm(module.weight, module.variance_epsilon, MixGemmCache)
            set_module_name(self.model, name, norm)

    def fuse_mlp(self,mix, MixGemmCache = None):
        for name, module in self.mlp_modules:
            if  mix:
                assert MixGemmCache is not None
                mlp = MixLlamaMLP(module.gate_proj, module.down_proj, module.up_proj , MixGemmCache)
            set_module_name(self.model, name, mlp)