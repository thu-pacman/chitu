import torch
from logging import getLogger
import numpy as np


logger = getLogger(__name__)


def load_tensor_parallel(checkpoint, model, num_layers, rank, world_size, type):
    keys = checkpoint.keys()
    if type == "llama":
        cpl_str = ["wq", "wk", "wv", "w1", "w3", "output", "embed"]
        rpl_str = ["wo", "w2"]
    elif type == "qwen":
        cpl_str = [
            "qkv_proj",  # new after fusion
            "q_proj",  # unused after fusion, for compatibility
            "k_proj",  # unused after fusion, for compatibility
            "v_proj",  # unused after fusion, for compatibility
            "gate_proj",
            "up_proj",
            "lm_head",
            "embed",
        ]
        rpl_str = ["down_proj", "o_proj"]
    else:
        assert False, f"Unknown model type {type}"
    partial_checkpoint = {}
    for name, param in checkpoint.items():
        if any(s in name for s in cpl_str):
            chunks = torch.chunk(param, world_size, dim=0)
            partial_checkpoint[name] = chunks[rank]
        elif any(s in name for s in rpl_str):
            chunks = torch.chunk(param, world_size, dim=1)
            partial_checkpoint[name] = chunks[rank]
        else:
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


def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor = None,  # TODO support min_ps
):
    """A top-k, top-p and min-p sampling implementation with native pytorch operations."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # min_p_thresholds = probs_sort[:, 0] * min_ps
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    # probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.max(dim=-1, keepdim=True)[0])
    sampled_index = torch.multinomial(probs_sort, num_samples=1)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids


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
        self.position_ids = torch.from_numpy(
            np.concatenate([np.arange(l) for l in self.cpu_lens])
        ).to(device)
