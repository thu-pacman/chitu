## Reference from llama.py
from .base import BaseAWQForCausalLM
from typing import Dict
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer as AquilaDecoderLayer,
    LlamaForCausalLM as AquilaForCausalLM,
    LlamaAttention as AquilaAttention,
    LlamaRMSNorm as AquilaRMSNorm,
    LlamaMLP as AquilaMLP
)

class AquilaAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "AquilaDecoderLayer"
    max_new_tokens_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model: AquilaForCausalLM, quant_config: Dict):
        fuser = AquilaFuser(model, quant_config)
        fuser.fuse_attention()
        fuser.fuse_rmsnorm()
        fuser.fuse_mlp()

    @staticmethod
    def get_model_layers(model: AquilaForCausalLM):
        return model.model.layers
    
    @staticmethod
    def get_act_for_scaling(module: AquilaDecoderLayer):
        return dict(
            is_scalable=False
        )
    
    @staticmethod
    def move_embed(model: AquilaForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    
    @staticmethod
    def get_layers_for_scaling(module: AquilaDecoderLayer, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(dict(
            prev_op=module.input_layernorm,
            layers=[module.self_attn.q_proj,
                    module.self_attn.k_proj, module.self_attn.v_proj],
            inp=input_feat['self_attn.q_proj'],
            module2inspect=module.self_attn, kwargs=module_kwargs,
        ))

        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat['self_attn.o_proj'],
            ))
        
        # linear 1
        layers.append(dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
            inp=input_feat['mlp.gate_proj'],
            module2inspect=module.mlp,
        ))

        # linear 2
        layers.append(dict(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat['mlp.down_proj'],
        ))

        return layers

