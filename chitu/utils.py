"""
This file has adaption of open-source code from the following sources:
- The sampling implementation is originally from SGLang
  (https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/sampler.py),
  licensed under Apache 2.0.
"""

from logging import getLogger
from typing import Any, Tuple

import numpy as np
import torch

from chitu.global_vars import get_global_args

logger = getLogger(__name__)


def try_import_opt_dep(pkg_name: str, opt_dep_name: str) -> Tuple[Any, bool]:
    try:
        return __import__(pkg_name), True
    except ImportError:

        class ReportErrorWhenUsed:
            def __getattr__(self, item):
                raise ImportError(
                    f"Optional dependency '{opt_dep_name}' is not installed. "
                    f"Please refer to README.md for installation instructions."
                )

        return ReportErrorWhenUsed(), False


def is_layer(layer_name, full_name):
    return (
        f".{layer_name}." in full_name
        or full_name.startswith(layer_name + ".")
        or full_name.endswith("." + layer_name)
    )


def compute_layer_dist_in_pipe(num_layers, world_size):
    args = get_global_args()
    if args.infer.pp_layer_partition is not None:
        assert (
            len(args.infer.pp_layer_partition) == world_size
            and sum(args.infer.pp_layer_partition) == args.models.n_layers
        ), f"pp_layer_partition must be a list of length {world_size} and sum up to {args.models.n_layers}"
        num_layers_of_each_rank = args.infer.pp_layer_partition
    else:
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


def get_config_dir_path():
    # Deprecated, but we need to support Python 3.8. importlib.resources is preferred in the future.
    import pkg_resources

    return pkg_resources.resource_filename("chitu", "config")
