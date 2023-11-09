import torch


def create_index(hs, ws):
    assert len(hs) == len(ws)
    HxWs = [h * w for h, w in zip(hs, ws)]
    idx_cuda = torch.tensor([hs, ws, HxWs], dtype=torch.int64)
    idx_cpu = idx_cuda.to("cpu")
    return idx_cuda, idx_cpu


def create_index_from_regular(n, h, w):
    hs = [h] * n
    ws = [w] * n
    return create_index(hs, ws)


# def replace_layer(module, name, old_layer, new_layer):
#     """
#     Recursively put desired batch norm in nn.module module.

#     set module = net to start code.
#     """
#     # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
#     for attr_str in dir(module):
#         target_attr = getattr(module, attr_str)
#         if isinstance(target_attr, old_layer):
#             print("replaced: ", name, attr_str)
#             new_bn = new_layer(target_attr)
#             setattr(module, attr_str, new_bn)

#     # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
#     for name, immediate_child_module in module.named_children():
#         replace_layer(immediate_child_module, name)


# replace_layer(model, "model")
