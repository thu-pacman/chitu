from typing import Dict
from .base import BaseForCausalLM
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralForCausalLM

class MistralMixForCausalLM(BaseForCausalLM):
    layer_type = "MistralDecoderLayer"
    max_new_tokens_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model: MistralForCausalLM, quant_config: Dict, mix, cache):
        fuser = MistralFuser(model, quant_config)
        fuser.fuse_attention(cache)
        fuser.fuse_rmsnorm()
        fuser.fuse_mlp()
    
    @staticmethod
    def get_model_layers(model: MistralForCausalLM):
        return model.model.layers
    
    @staticmethod
    def get_act_for_scaling(module: MistralDecoderLayer):
        return dict(
            is_scalable=False
        )
    
    @staticmethod
    def move_embed(model: MistralForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    


import torch
from typing import List, Tuple, Union
from mixquant.utils.utils import set_module_name
from mixquant.modules.fused.mlp import MixLlamaMLP
from mixquant.modules.fused.mistral_attn import MistralQuantAttentionFused
from mixquant.modules.fused.norm import FasterTransformerRMSNorm
from mixquant.modules.linear import  MixLinear_GEMM

from transformers.models.mistral.modeling_mistral import MistralAttention, MistralRMSNorm, MistralMLP

class MistralFuser:
    def __init__(self, model, quant_config):
        self.model = model
        self.quant_config = quant_config

        self.attention_modules: List[Tuple[str, MistralAttention]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, MistralAttention)
        ]

        self.rmsnorm_modules: List[Tuple[str, MistralRMSNorm]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, MistralRMSNorm)
        ]
        
        self.mlp_modules: List[Tuple[str, MistralMLP]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, MistralMLP)
        ]
    
    def fuse_attention(self,cache):
        for name, module in self.attention_modules:
            qkv_layer  = self._fuse_qkv(module, cache)
            attn = MistralQuantAttentionFused(
                module.hidden_size,
                module.num_heads,
                module.num_key_value_heads,
                qkv_layer, 
                module.o_proj,
                next(iter(qkv_layer.state_dict().values())).device,
                self.model.config.max_new_tokens,
                MixGemmCache = cache,
                config = self.model.config
            )
            set_module_name(self.model, name, attn)
    
    def _fuse_qkv(self, module: MistralAttention, cache):
        try:
            q_proj, k_proj, v_proj = module.q_proj, module.k_proj, module.v_proj
        except:
            qkv_layer = module.W_pack
            return qkv_layer
 
 
        
        if not  isinstance(q_proj, MixLinear_GEMM) :
            raise NotImplementedError
 
        if isinstance(q_proj, MixLinear_GEMM):
            qkv_layer = MixLinear_GEMM(q_proj.in_features,q_proj.out_features + k_proj.out_features + v_proj.out_features,
                                        q_proj.bias is not None,
                                        next(iter(module.state_dict().values())).device,
                                        False,
                                        cache)


        
        if isinstance(qkv_layer, MixLinear_GEMM):
            shapew = qkv_layer.q_weight.shape
            qkv_layer.q_weight = torch.cat([q_proj.q_weight, k_proj.q_weight, v_proj.q_weight], dim=0)
            qkv_layer.scale_col = torch.cat([q_proj.scale_col, k_proj.scale_col, v_proj.scale_col], dim=1)

            assert shapew[0] == qkv_layer.q_weight.shape[0]
            assert shapew[1] == qkv_layer.q_weight.shape[1]
            assert shapew[0] == qkv_layer.scale_col.shape[1]
            assert 1 == qkv_layer.scale_col.shape[0]

            if q_proj.bias is not None:
                raise NotImplementedError
            else:
                qkv_layer.bias = None

        else:
            raise NotImplementedError
        
        q_proj.q_weight = q_proj.q_weight.to('cpu')
        k_proj.q_weight = k_proj.q_weight.to('cpu')
        v_proj.q_weight = v_proj.q_weight.to('cpu')
        q_proj.scale_col = q_proj.scale_col.to('cpu')
        k_proj.scale_col = k_proj.scale_col.to('cpu')
        v_proj.scale_col = v_proj.scale_col.to('cpu')
        torch.cuda.empty_cache()
        return qkv_layer

    def fuse_rmsnorm(self):
        for name, module in self.rmsnorm_modules:
            norm = FasterTransformerRMSNorm(module.weight, module.variance_epsilon)
            set_module_name(self.model, name, norm)

    def fuse_mlp(self):
        for name, module in self.mlp_modules:
            mlp = MixLlamaMLP(module.gate_proj, module.down_proj, module.up_proj)
            set_module_name(self.model, name, mlp)
