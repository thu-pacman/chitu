import os
from transformers import AutoConfig
from mixquant.models import *
from mixquant.models.base import BaseForCausalLM

CAUSAL_LM_MODEL_MAP = {

    "llama": LlamaMixQForCausalLM,
    "baichuan": BaichuanMixQForCausalLM,
    "aquila": LlamaMixQForCausalLM,
    #"mistral": MistralMixForCausalLM,
    "gptj" : GPTJMixForCausalLM,
    "falcon": FalconMixForCausalLM,
    "opt": OptMixForCausalLM
}

def check_and_get_model_type(model_dir, trust_remote_code=True):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    if config.model_type not in CAUSAL_LM_MODEL_MAP.keys():
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    if config.architectures[0]=="BaichuanForCausalLM":
        model_type="baichuan"
    return model_type

class AutoForCausalLM:
    def __init__(self):
        raise EnvironmentError('You must instantiate AutoAWQForCausalLM with\n'
                               'AutoAWQForCausalLM.from_quantized or AutoAWQForCausalLM.from_pretrained')
    
    @classmethod
    def from_pretrained(self, model_path, trust_remote_code=True, safetensors=False,
                              device_map=None, mix = False, **model_init_kwargs) -> BaseForCausalLM:
        model_type = check_and_get_model_type(model_path, trust_remote_code)
        
        return CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
            model_path, model_type, trust_remote_code=trust_remote_code, safetensors=safetensors,
            device_map=device_map, mix = mix, **model_init_kwargs
        )

    @classmethod
    def from_quantized(self, quant_path, quant_filename='', max_new_tokens=None,
                       trust_remote_code=True, fuse_layers=True,
                       batch_size=1, safetensors=False,
                       max_memory=None, offload_folder=None, mix = False, cache = None) -> BaseForCausalLM:

        model_type = check_and_get_model_type(quant_path, trust_remote_code)
        os.environ["BATCH_SIZE"] = str(batch_size)
        return CAUSAL_LM_MODEL_MAP[model_type].from_quantized(
            quant_path, model_type, quant_filename, max_new_tokens, trust_remote_code=trust_remote_code, 
            fuse_layers=fuse_layers, safetensors=safetensors, 
            max_memory=max_memory, offload_folder=offload_folder, mix = mix, cache = cache
        )
