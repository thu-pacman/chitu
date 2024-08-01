import torch.nn as nn

import torch
from mixquant.Cache import MixLibCache, MLPCache
import mixlib


class MixFalconMLP(nn.Module):

    def __init__(
        self,
        dense_h_to_4h,
        dense_4h_to_h,
        MixGemmCache = None
    ):
        super().__init__()



        self.dense_h_to_4h = dense_h_to_4h
        self.dense_4h_to_h = dense_4h_to_h
        self.act = nn.GELU()
        self.MixGemmCache = MixGemmCache   

 
    def forward(self, x):
 
        x = self.act(self.dense_h_to_4h(x, self.MixGemmCache))
        
        
        x = self.dense_4h_to_h(x, self.MixGemmCache,True)
        
        return x
    

import time
class MixLlamaMLP(nn.Module):

    def __init__(
        self,
        gate_proj,
        down_proj,
        up_proj,
        MixGemmCache = None
    ):
        super().__init__()



        self.down_proj_ = down_proj
        self.gate_proj_ = gate_proj
        self.up_proj_ = up_proj
        self.out_features = down_proj.out_features
        self.MLPCache = MixGemmCache
        
 
    def forward(self, x):
 
        
        
        up_output = self.up_proj_(x, self.MLPCache)
        gate_output = self.gate_proj_.forward_without_preconditionFusedSilu(x, self.MLPCache)
        
        gate_output *= up_output


        y = self.down_proj_(gate_output,None,True)

 
        return y
    
 
from transformers.activations import ACT2FN
class MixGPTJMLP(nn.Module):
    def __init__(self, module, config, MixGemmCache = None):  # in MLP: intermediate_size= 4 * embed_dim
        super().__init__()


        self.fc_in = module.fc_in
        self.fc_out = module.fc_out

        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)


        self.MLPCache = MLPCache()

    def forward(self, hidden_states) -> torch.FloatTensor:

        hidden_states = self.fc_in(hidden_states, self.MLPCache)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states, self.MLPCache)
        hidden_states = self.dropout(hidden_states)
        return hidden_states