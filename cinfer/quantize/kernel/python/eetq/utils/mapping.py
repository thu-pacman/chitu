Mappings = {
    "llama": "transformers_llama_mapping",
}

transformers_llama_mapping = {
    "decoder": "layers",
    "decoder_layer": "LlamaDecoderLayer",
    "attention": "LlamaAttention",
    "mlp": "LlamaMLP",
    "layernorm": "LlamaRMSNorm",
    "embedding": "embed_tokens",
}


def structure_mapping(module, name):
    if name == "":
        return name
    mapping = eval(Mappings[name])
    return mapping
