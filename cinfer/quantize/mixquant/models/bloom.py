from .base import BaseAWQForCausalLM
from transformers.models.bloom.modeling_bloom import BloomForCausalLM, BloomBlock

class BloomMixForCausalLM(BaseAWQForCausalLM):
    layer_type = "BloomBlock"

    @staticmethod
    def get_model_layers(model: BloomForCausalLM):
        return model.transformer.h
    

    @staticmethod
    def move_embed(model: BloomForCausalLM, device: str):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(device)
    
