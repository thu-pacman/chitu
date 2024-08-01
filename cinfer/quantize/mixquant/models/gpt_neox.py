from .base import BaseAWQForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer, GPTNeoXForCausalLM

class GPTNeoXAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "GPTNeoXDecoderLayer"
    max_new_tokens_key = "max_position_embeddings"

    @staticmethod
    def get_model_layers(model: GPTNeoXForCausalLM):
        return model.gpt_neox.layers
    
    @staticmethod
    def get_act_for_scaling(module: GPTNeoXLayer):
        return dict(
            is_scalable=True,
            scale_name="mlp.act",
            scale_layer=module.mlp.act,
            scale_shape=module.mlp.dense_h_to_4h.out_features,
        )
    
    @staticmethod
    def move_embed(model: GPTNeoXForCausalLM, device: str):
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
    
    @staticmethod
    def get_layers_for_scaling(module: GPTNeoXLayer, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(dict(
            prev_op=module.input_layernorm,
            layers=[module.attention.query_key_value],
            inp=input_feat['attention.query_key_value'],
        ))

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/issues/2#issuecomment-1606297469
        """
        layers.append(dict(
            prev_op=module.attention.query_key_value,
            layers=[module.attention.dense],
            inp=input_feat['attention.dense'],
        ))
        """

        # linear 1
        layers.append(dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.dense_h_to_4h],
            inp=input_feat['mlp.dense_h_to_4h'],
        ))

        # linear 2
        layers.append(dict(
            prev_op=module.mlp.act,
            layers=[module.mlp.dense_4h_to_h],
            inp=input_feat['mlp.dense_4h_to_h'],
        ))

        return layers
