from .base import BaseForCausalLM
from typing import Dict
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer as OldFalconDecoderLayer, FalconForCausalLM, FalconAttention
from transformers.models.falcon.configuration_falcon import FalconConfig 
from transformers.models.falcon.modeling_falcon import FalconMLP


from mixquant.modules.fused.mlp import  MixFalconMLP
from mixquant.utils.utils import set_module_name
import torch
from typing import Optional, Tuple, Union, List
from torch import nn
from torch.nn import functional as F

class FalconMixForCausalLM(BaseForCausalLM):
    layer_type = "FalconDecoderLayer"

    @staticmethod
    def fuse_layers(model: FalconForCausalLM, quant_config: Dict, mix, cache):
        fuser = FalconFuser(model)


        fuser.fuse_mlp(mix, cache)

    @staticmethod
    def get_model_layers(model: FalconForCausalLM):
        return model.transformer.h
    
    @staticmethod
    def move_embed(model: FalconForCausalLM, device):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    



class FalconFuser:
    def __init__(self, model: FalconForCausalLM):
        self.model = model

        self.attention_modules = [
            (name, module) for name, module in self.model.named_modules()
            if  "Attention" in str(module.__class__)
        ]
        self.mlp_modules: List[Tuple[str, FalconMLP]] = [
            (name, module) for name, module in self.model.named_modules()
            if isinstance(module, FalconMLP)   or  "MLP" in str(module.__class__)
        ]
   
    def fuse_mlp(self,mix, MixGemmCache = None):
        for name, module in self.mlp_modules:
            if  mix:
                assert MixGemmCache is not None
                mlp = MixFalconMLP(module.dense_h_to_4h, module.dense_4h_to_h, MixGemmCache)
            set_module_name(self.model, name, mlp)