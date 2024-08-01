import torch
import torch.nn as nn


def find_submodule(module, sub_name):
    res = None
    if hasattr(module, sub_name):
        return getattr(module, sub_name)
    else:
        for name, m in module.named_children():
            res = find_submodule(m, sub_name)
            if res is not None:
                return res
    raise ValueError(f"Cannot find submodule {sub_name} in module {module}")


def get_op_by_name(module, op_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")


def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        delattr(mod_, levels[-1])
        setattr(mod_, levels[-1], new_module)
    else:
        delattr(layer, name)
        setattr(layer, name, new_module)


def get_op_name(module, op):
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")


def append_str_prefix(x, prefix):
    if isinstance(x, str):
        return prefix + x
    elif isinstance(x, tuple):
        return tuple([append_str_prefix(y, prefix) for y in x])
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear) and "lm_head" not in name}


def get_named_layers(module, layers=[nn.Linear]):
    return {name: m for name, m in module.named_modules() if type(m) in layers}

def find_layers(module, include=[nn.Linear], exclude=["lm_head"]):
    res = {}
    for name, m in module.named_modules():
        if type(m) in include and not any([e in name for e in exclude]):
            res.update({name: m})
    return res
