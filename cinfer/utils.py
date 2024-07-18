import torch
from logging import getLogger


logger = getLogger(__name__)


def load_tensor_parallel(checkpoint, model, num_layers, rank, world_size, type):
    keys = checkpoint.keys()
    if type == "llama":
        cpl_str = ["wq", "wk", "wv", "w1", "w3", "output", "embed"]
        rpl_str = ["wo", "w2"]
    elif type == "qwen":
        cpl_str = [
            "q_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "lm_head",
            "embed",
        ]
        rpl_str = ["down_proj", "o_proj"]
    else:
        assert False, f"Unknown model type {type}"
    partial_checkpoint = {}
    # 遍历模型参数并进行切分
    for name, param in checkpoint.items():
        # 检查参数是否至少有一维
        if any(s in name for s in cpl_str):
            print(f"{name}, handle cpl")
            chunks = torch.chunk(param, world_size, dim=0)
            partial_checkpoint[name] = chunks[rank]
        elif any(s in name for s in rpl_str):
            chunks = torch.chunk(param, world_size, dim=1)
            print(f"{param.shape}, {chunks[0].shape}, {chunks[1].shape}")
            partial_checkpoint[name] = chunks[rank]
        else:
            print(f"{name}, handle nothing")
            # 如果参数是标量或空张量，直接复制到两个部分中
            partial_checkpoint[name] = param
    model.load_state_dict(partial_checkpoint)


def load_pipe(checkpoint, model, num_layers, rank, world_size, type):
    keys = checkpoint.keys()
    # logger.warning(f"Loading checkpoint {keys}")
    partial_checkpoint = {}
    if rank == 0:
        if type == "llama":
            partial_checkpoint["tok_embeddings.weight"] = checkpoint[
                "tok_embeddings.weight"
            ]
        elif type == "qwen":
            partial_checkpoint["embed_tokens.weight"] = checkpoint[
                "embed_tokens.weight"
            ]
    base = num_layers // world_size * rank
    for i in range(base, num_layers // world_size * (rank + 1)):
        for key in keys:
            if f"layers.{i}." in key:
                partial_checkpoint[key.replace(str(i), str(i - base), 1)] = checkpoint[
                    key
                ]
    if rank == world_size - 1:
        if type == "llama":
            partial_checkpoint["output.weight"] = checkpoint["output.weight"]
        elif type == "qwen":
            partial_checkpoint["lm_head.weight"] = checkpoint["lm_head.weight"]
        partial_checkpoint["norm.weight"] = checkpoint["norm.weight"]
    model.load_state_dict(partial_checkpoint)


class VarLens:
    def __init__(self, tokens, device) -> None:
        self.lens = torch.tensor(
            [len(t) for t in tokens], device=device, dtype=torch.int32
        )
        self.cpu_prefix_lens = [0]
        for t in tokens:
            self.cpu_prefix_lens.append(self.cpu_prefix_lens[-1] + len(t))
        self.prefix_lens = torch.tensor(
            self.cpu_prefix_lens, device=device, dtype=torch.int32
        )
        self.cpu_lens = [len(t) for t in tokens]
        self.max_len = int(torch.max(self.lens))
        self.total_len = int(torch.sum(self.lens))
