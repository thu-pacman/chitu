import gc
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from ..modules.qlinear import W8A16Linear
from .base import set_op_by_name, get_named_linears, get_named_layers, find_submodule, find_layers
from .mapping import structure_mapping


def replace_with_eet_qlinear(model, init_only=False, target_model="llama", device="cuda:0"):
    sub_name = structure_mapping(model, target_model)['decoder']
    layers = find_submodule(model, sub_name)
    for i in tqdm(range(len(layers)), desc="[EET][INFO] replace with eet weight quantize only linear..." + ("(init only)" if init_only else "")):
        layer = layers[i]
        named_linears = get_named_linears(layer)            # linear dict
        for name, linear in named_linears.items():
            # print("[EET][DEBUG] name:{}, type(linear):{}".format(name, type(linear)))
            if linear.weight.dtype == torch.float16:        # nn.Linear
                q_linear = W8A16Linear.from_torch(linear, scales=None, init_only=init_only)
            elif linear.weight.dtype == torch.int8:         # bitsandbytes.nn.Linear8bitLt
                scales = torch.div(linear.state_dict()["SCB"], 127.0)
                q_linear = W8A16Linear.from_torch(linear, scales=scales, init_only=init_only)
            else:
                raise ValueError("Unsupported data type: {}".format(linear.weight.dtype))
            set_op_by_name(layer, name, q_linear)

            if not init_only:
                # del linear
                linear.cpu()
                del linear
                torch.cuda.empty_cache()
                gc.collect()
            else:
                del linear
                gc.collect()

def eet_quantize(model, init_only=False, include=[nn.Linear], exclude=["lm_head"], device="cuda:0"):
    named_linears = find_layers(model, include=include, exclude=exclude)
    for name, linear in tqdm(named_linears.items(), desc="[EET][INFO] quantization preprocessing..." + ("(init only)" if init_only else "")):
        # print("[EET][DEBUG] name:{}, type(linear):{}".format(name, type(linear)))
        if linear.weight.dtype == torch.float16:        # nn.Linear
            q_linear = W8A16Linear.from_torch(linear, scales=None, init_only=init_only)
        elif linear.weight.dtype == torch.int8:         # bitsandbytes.nn.Linear8bitLt
            scales = torch.div(linear.state_dict()["SCB"], 127.0)
            q_linear = W8A16Linear.from_torch(linear, scales=scales, init_only=init_only)
        else:
            raise ValueError("Unsupported data type: {}".format(linear.weight.dtype))
        set_op_by_name(model, name, q_linear)

        if not init_only:
            # del linear
            linear.cpu()
            del linear
            torch.cuda.empty_cache()
            gc.collect()
        else:
            del linear
            gc.collect()
