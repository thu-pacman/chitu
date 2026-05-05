# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from chitu.moe.impl import MoEImplBase, MoEImplEP
from chitu.moe.load_balancer import get_moe_load_planner
from chitu.global_vars import get_global_args


def compute_local_batch_size_dist_in_dp(
    global_batch_size: int, dp_size: int
) -> list[int]:
    return [
        global_batch_size // dp_size + int(i < global_batch_size % dp_size)
        for i in range(dp_size)
    ]


def compute_layer_dist_in_pp(num_layers: int, pp_size: int):
    args = get_global_args()
    if args.infer.pp_layer_partition is not None:
        assert (
            len(args.infer.pp_layer_partition) == pp_size
            and sum(args.infer.pp_layer_partition) == args.models.n_layers
        ), f"pp_layer_partition must be a list of length {pp_size} and sum up to {args.models.n_layers}"
        num_layers_of_each_rank = args.infer.pp_layer_partition
    else:
        num_layers_of_each_rank = [
            num_layers // pp_size + (1 if i < num_layers % pp_size else 0)
            for i in range(pp_size)
        ]
        # If non-divisible, make the fisrst and the last rank to have fewer layers, because they have pre-layers and post-layers
        if pp_size > 2 and num_layers_of_each_rank[0] > num_layers_of_each_rank[-2]:
            num_layers_of_each_rank[0] -= 1
            num_layers_of_each_rank[-2] += 1
    return num_layers_of_each_rank


def compute_expert_dist_in_ep(
    num_moe_layers: int, ep_size: int, num_experts: int, moe_impl: Optional[MoEImplBase]
) -> list[list[list[int]]]:  # expert ids for each layer for each ep rank
    if isinstance(moe_impl, MoEImplEP):
        enable_dynamic_load_balance = get_global_args().infer.moe_lb_trigger > 0
        if enable_dynamic_load_balance:
            planner = get_moe_load_planner()
            return [
                [
                    planner.get_current_mapping(layer_id)[ep_rank]
                    for layer_id in moe_impl.moe_layer_id_list
                ]
                for ep_rank in range(ep_size)
            ]
        else:
            return [
                [
                    moe_impl.load_balancer[layer_id].get_local_experts(ep_rank)
                    for layer_id in moe_impl.moe_layer_id_list
                ]
                for ep_rank in range(ep_size)
            ]
    else:
        return [[list(range(num_experts))] * num_moe_layers] * ep_size
