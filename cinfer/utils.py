import torch
from logging import getLogger


logger = getLogger(__name__)


def load(checkpoint, model, num_layers, rank, world_size):
    keys = checkpoint.keys()
    logger.warning(f"Loading checkpoint {keys}")
    partial_checkpoint = {}
    if rank == 0:
        partial_checkpoint["tok_embeddings.weight"] = checkpoint[
            "tok_embeddings.weight"
        ]
    base = num_layers // world_size * rank
    for i in range(base, num_layers // world_size * (rank + 1)):
        for key in keys:
            if f"layers.{i}." in key:
                partial_checkpoint[key.replace(str(i), str(i - base), 1)] = checkpoint[
                    key
                ]
    if rank == world_size - 1:
        partial_checkpoint["output.weight"] = checkpoint["output.weight"]
        partial_checkpoint["norm.weight"] = checkpoint["norm.weight"]
    model.load_state_dict(partial_checkpoint)
