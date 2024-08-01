from .base import BaseForCausalLM
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM, GPTJBlock, GPTJAttention, GPTJMLP
from typing import List, Tuple, Union
from mixquant.modules.linear import  MixLinear_GEMM
import torch
from mixquant.modules.fused.gptj_attn import QuantGPTJAttentionFused
from mixquant.modules.fused.mlp import  MixGPTJMLP
from mixquant.utils.utils import set_module_name


class GPTJMixForCausalLM(BaseForCausalLM):
    layer_type = "GPTJBlock"
    max_new_tokens_key = "n_positions"

    @staticmethod
    def get_model_layers(model: GPTJForCausalLM):
        return model.transformer.h
    

    @staticmethod
    def move_embed(model: GPTJForCausalLM, device: str):
        model.transformer.wte = model.transformer.wte.to(device)
    




    @staticmethod
    def fuse_layers(model: GPTJForCausalLM, quant_config, mix, cache):
        fuser = GPTJFuser(model)


        fuser.fuse_mlp(mix, cache)
        fuser.fuse_attention(MixGemmCache = cache)





class GPTJFuser:
    def __init__(self, model: GPTJForCausalLM):
        self.model = model

        self.attention_modules: List[Tuple[str, GPTJAttention]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, GPTJAttention) or  "Attention" in str(module.__class__)
        ]
        self.mlp_modules: List[Tuple[str, GPTJMLP]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, GPTJMLP)   or  "MLP" in str(module.__class__)
        ]
    def fuse_mlp(self,mix, MixGemmCache = None):
        for name, module in self.mlp_modules:
            if  mix:
                assert MixGemmCache is not None
                mlp = MixGPTJMLP(module, self.model.config, MixGemmCache)
            set_module_name(self.model, name, mlp)


    def _fuse_qkv(self, module,cache):
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
            raise "no implement"
        
        q_proj.q_weight = q_proj.q_weight.to('cpu')
        k_proj.q_weight = k_proj.q_weight.to('cpu')
        v_proj.q_weight = v_proj.q_weight.to('cpu')
        q_proj.scale_col = q_proj.scale_col.to('cpu')
        k_proj.scale_col = k_proj.scale_col.to('cpu')
        v_proj.scale_col = v_proj.scale_col.to('cpu')
        torch.cuda.empty_cache()
        return qkv_layer


    def fuse_attention(self, MixGemmCache):
        for name, module in self.attention_modules:
            qkv_layer  = self._fuse_qkv(module)

            attn = QuantGPTJAttentionFused(
                self.model.config,
                module,
                qkv_layer, 
                module.out_proj,
                MixGemmCache = MixGemmCache
            )
            set_module_name(self.model, name, attn)