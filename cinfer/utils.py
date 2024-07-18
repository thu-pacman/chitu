import torch
from logging import getLogger


logger = getLogger(__name__)


def load_pipe(checkpoint, model, num_layers, rank, world_size, type):
    keys = checkpoint.keys()
    logger.warning(f"Loading checkpoint {keys}")
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
