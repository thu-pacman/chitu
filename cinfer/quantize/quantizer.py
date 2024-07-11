import torch

import bitsandbytes as bnb

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


def quant(model, quantconfig):
    if quantconfig['method'] == "llmint8":
        quantize_llmint8(model, quantconfig)

def quantize_llmint8(model, quantconfig):
    model, has_been_replaced = replace_with_bnb(model)

    if not has_been_replaced:
        print("no linear modules were found.")

    print(model)
    return model