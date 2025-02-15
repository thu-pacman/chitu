import torch
from logging import getLogger
import numpy as np


logger = getLogger(__name__)


def is_layer(layer_name, full_name):
    return (
        f".{layer_name}." in full_name
        or full_name.startswith(layer_name + ".")
        or full_name.endswith("." + layer_name)
    )


def compute_layer_dist_in_pipe(num_layers, world_size):
    num_layers_of_each_rank = [
        num_layers // world_size + (1 if i < num_layers % world_size else 0)
        for i in range(world_size)
    ]
    # If non-divisible, make the fisrst and the last rank to have fewer layers, because they have pre-layers and post-layers
    if world_size > 2 and num_layers_of_each_rank[0] > num_layers_of_each_rank[-2]:
        num_layers_of_each_rank[0] -= 1
        num_layers_of_each_rank[-2] += 1
    return num_layers_of_each_rank


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


def merge_column_parallel_weights(weights, model_parallel_size):
    """
    For example, fuse weight_Q, weight_K, weight_V into one tensor.

    This function can handle any (output_hidden, input_hidden) shaped tensor.

    - Column parallel means the fairsacale-style ColumnPararllelLinear layer. The merged
    dimension is actually the FIRST dimension instead of the last.
    - The fused projected shape should be concatenated after the model parallel dimension.
    See model_hf_llama.py for details.
    """

    new_weights = []
    for weight in weights:
        assert weight.shape[0] % model_parallel_size == 0
        new_weights.append(weight.reshape(model_parallel_size, -1, weight.shape[-1]))
    ret_weight = torch.cat(new_weights, dim=1)
    ret_weight = ret_weight.reshape(-1, ret_weight.shape[-1])
    return ret_weight


def merge_column_parallel_biases(biases, model_parallel_size):
    """
    For example, fuse bias_Q, bias_K, bias_V into one tensor.

    This function can handle any (output_hidden,) shaped tensor.
    """
    new_biases = []
    for bias in biases:
        assert bias.shape[0] % model_parallel_size == 0
        new_biases.append(bias.reshape(model_parallel_size, -1))
    ret_bias = torch.cat(new_biases, dim=1)
    ret_bias = ret_bias.reshape(-1)
    return ret_bias


def print_d(*args, **kargs):
    # return
    print(f">>> rank{torch.distributed.get_rank()}:", *args, **kargs)
