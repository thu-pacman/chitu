import torch

import bitsandbytes as bnb


from cinfer.model import *

from cinfer.tokenizer import Tokenizer, ChatFormat
import cinfer.awq as awq
#import cinfer.evaluator as eval

__all__ = ["quant"]

def replace_with_bnb(model, current_key_name=None):
     
    has_been_replaced = False
    for name, module in model.named_children():
        #print(name)
        if current_key_name is None:
            current_key_name = []
            
        current_key_name.append(name)
        current_key_name_str = ".".join(current_key_name)
        print(current_key_name_str)
        #if current_key_name_str == "model.model":
        #    continue
        #if (name == "model"):
        #    continue
        if isinstance(module, torch.nn.Linear):
            bnb_module = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                module.bias is not None,
                has_fp16_weights=False,
                threshold=6.0,
            )
            bnb_module.weight.data = module.weight.data
            if module.bias is not None:
                bnb_module.bias.data = module.bias.data
            bnb_module.requires_grad_(False)
            setattr(model, name, bnb_module)
            has_been_replaced = True
        if len(list(module.children())) > 0:
            _, _has_been_replaced = replace_with_bnb(module, current_key_name)
            has_been_replaced = has_been_replaced | _has_been_replaced
        
        current_key_name.pop(-1)
    return model, has_been_replaced




def quantize_llmint8(model):
    model, has_been_replaced = replace_with_bnb(model)

    if not has_been_replaced:
        print("no linear modules were found.")

    print(model)
    return model

def quantize_awq(model):
    q_config = {
        "zero_point": True,
        "q_group_size": 128
    }

    awq.real_quantize_model_weight(
        model, w_bit=4, q_config=q_config, init_only=True
    )
    #print(model.device)

    sd = torch.load("/home/tanyijun/cinfer/quant_cache/Llama3-8B-4bit.pth", map_location="cpu")
    model.load_state_dict(sd)

    
    class FP16Trans(torch.nn.Module):
        def __init__(self, toke):
            super().__init__()
            self.tok_embeddings = toke

        def forward(self, x):
            return self.tok_embeddings(x).to(torch.float16)
        

    model.tok_embeddings = FP16Trans(model.tok_embeddings)

    print(model)

    return model

def quant(model, method=None):

    if method == "llmint8":
        return quantize_llmint8(model)
    elif method == "awq":
        return quantize_awq(model)
        
    return model