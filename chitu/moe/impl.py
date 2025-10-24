# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from chitu.task_type import TaskType
from chitu.utils import try_import_opt_dep, try_import_and_setup_torch_npu
from chitu.moe.token_dispatchers import (
    MoETokenDispatcher,
    MoEEmptyTokenDispatcher,
    MoEAllGatherTokenDispatcher,
)
from chitu.device_type import is_ascend_910b

import torch
from .load_balancer import (
    MoELargeScaleNaiveLoadBalancer,
    MoENaiveLoadBalancer,
)
from chitu.distributed.parallel_state import get_ep_group


deep_ep, has_deep_ep = try_import_opt_dep("deep_ep", "deep_ep")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()

if has_deep_ep:
    from .token_dispatchers import MoELowLatencyTokenDispatcher
    from .token_dispatchers import MoENormalTokenDispatcher

MOE_IMPL_INSTANCE: Optional["MoEImpl"] = None


def init_moe_impl(args) -> None:
    """Initialize MoEImpl instance."""
    global MOE_IMPL_INSTANCE
    assert MOE_IMPL_INSTANCE is None, "moe impl already initialized"

    if args.infer.ep_size > 1:
        MOE_IMPL_INSTANCE = MoEImpl(args)
    else:
        MOE_IMPL_INSTANCE = None


def get_moe_impl() -> Optional["MoEImpl"]:
    """Get MoEImpl instance."""
    return MOE_IMPL_INSTANCE


class MoEImpl:
    """MoEImpl is a base class for MoE implementation."""

    def __init__(self, args) -> None:
        self.tp_size = args.infer.tp_size
        self.dp_size = args.infer.dp_size
        self.ep_size = args.infer.ep_size
        self.hidden_dim = args.models.dim
        self.ep_rank = get_ep_group().rank_in_group

        self.num_experts = getattr(args.models, "n_routed_experts", None) or getattr(
            args.models, "num_experts", None
        )
        if self.num_experts is None:
            raise ValueError(
                "n_routed_experts or num_experts must be specified in model args"
            )

        self.task_type: Optional[TaskType] = None

        self.prefill_experts_impl = "auto"
        self.decode_experts_impl = "auto"
        self.prefill_token_dispatcher_impl = args.infer.moe.prefill_token_dispatcher
        self.decode_token_dispatcher_impl = args.infer.moe.decode_token_dispatcher
        self.use_cuda_graph = args.infer.use_cuda_graph
        self.n_layers = args.models.n_layers
        self.n_dense_layers = (
            args.models.n_dense_layers if hasattr(args.models, "n_dense_layers") else 0
        )
        self.moe_layer_id_list = [x for x in range(self.n_dense_layers, self.n_layers)]
        # self.n_global_experts_slots = args.model.n_global_experts_slots
        self.n_global_experts_slots = (
            (self.num_experts + self.ep_size - 1) // self.ep_size
        ) * self.ep_size

        self._init_token_dispatcher()
        self._init_experts_impl()
        expert_stats_path = (
            args.infer.expert_stats_path
            if hasattr(args.infer, "expert_stats_path")
            else None
        )
        self._init_load_balancer(expert_stats_path)

    def _init_token_dispatcher(self):
        # impl selection
        if self.prefill_token_dispatcher_impl == "auto":
            if self.dp_size > 1 and has_deep_ep:
                self.prefill_token_dispatcher_impl = "deepep-nl"
            elif self.dp_size > 1 and has_torch_npu:
                self.prefill_token_dispatcher_impl = (
                    "fused_experts_with_a2a_communication"
                )
            else:
                self.prefill_token_dispatcher_impl = "allgather"

        if self.decode_token_dispatcher_impl == "auto":
            if self.dp_size > 1 and has_deep_ep:
                self.decode_token_dispatcher_impl = "deepep-ll"
            elif (
                self.dp_size > 1
                and has_torch_npu
                and not (is_ascend_910b() and self.tp_size > 1)
            ):
                self.decode_token_dispatcher_impl = "fused_experts_with_communication"
            elif self.dp_size > 1 and has_torch_npu:
                self.decode_token_dispatcher_impl = (
                    "fused_experts_with_a2a_communication"
                )
            else:
                self.decode_token_dispatcher_impl = "allgather"

        # impl initialization
        if self.prefill_token_dispatcher_impl == "deepep-nl":
            self.prefill_token_dispatcher = MoENormalTokenDispatcher(
                self.num_experts,
                self.hidden_dim,
                mode=(
                    "auto"
                    if self.decode_token_dispatcher_impl == "deepep-ll"
                    else "deepep-normal"
                ),
            )
            self.prefill_experts_impl = "ep_group_gemm_contiguous"
        elif (
            self.prefill_token_dispatcher_impl == "fused_experts_with_a2a_communication"
        ):
            self.prefill_token_dispatcher = MoEEmptyTokenDispatcher()
            self.prefill_experts_impl = "fused_experts_with_a2a_communication"
        elif self.prefill_token_dispatcher_impl == "allgather":
            self.prefill_token_dispatcher = MoEAllGatherTokenDispatcher()
        else:
            raise ValueError(
                f"Invalid prefill token dispatcher: {self.prefill_token_dispatcher_impl}"
            )

        if self.decode_token_dispatcher_impl == "deepep-ll":
            self.decode_token_dispatcher = MoELowLatencyTokenDispatcher(
                self.num_experts, self.hidden_dim
            )
            self.decode_experts_impl = "ep_group_gemm_masked"
        elif (
            self.decode_token_dispatcher_impl == "fused_experts_with_a2a_communication"
        ):
            self.decode_token_dispatcher = MoEEmptyTokenDispatcher()
            self.decode_experts_impl = "fused_experts_with_a2a_communication"
        elif self.decode_token_dispatcher_impl == "fused_experts_with_communication":
            self.decode_token_dispatcher = MoEEmptyTokenDispatcher()
            self.decode_experts_impl = "fused_experts_with_communication"
        elif self.decode_token_dispatcher_impl == "allgather":
            self.decode_token_dispatcher = MoEAllGatherTokenDispatcher()
        else:
            raise ValueError(
                f"Invalid decode token dispatcher: {self.decode_token_dispatcher_impl}"
            )

    def _get_current_token_dispatcher(self) -> MoETokenDispatcher:
        assert self.task_type is not None
        if self.task_type in [TaskType.Prefill, TaskType.EmptyPrefill]:
            return self.prefill_token_dispatcher
        elif self.task_type in [TaskType.Decode, TaskType.EmptyDecode]:
            return self.decode_token_dispatcher
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")

    def _init_experts_impl(self):
        self.impl_map = {
            TaskType.Prefill: self.prefill_experts_impl,
            TaskType.EmptyPrefill: self.prefill_experts_impl,
            TaskType.Decode: self.decode_experts_impl,
            TaskType.EmptyDecode: self.decode_experts_impl,
        }

    def get_experts_impl(self) -> str:
        return self.impl_map[self.task_type]

    def prepare(self, task_type: TaskType, num_tokens: int) -> None:
        self.task_type = task_type
        self._get_current_token_dispatcher().prepare(num_tokens)

    def token_permutation(self, *args, **kwargs):
        return self._get_current_token_dispatcher().token_permutation(*args, **kwargs)

    def token_unpermutation(self, *args, **kwargs):
        return self._get_current_token_dispatcher().token_unpermutation(*args, **kwargs)

    def _load_expert_stats(self, file_path):
        expert_stats = torch.load(file_path)
        assert expert_stats.shape == (self.n_layers, self.num_experts)
        return expert_stats

    def _init_load_balancer(self, expert_stats_path: str = None):
        if expert_stats_path is not None:
            expert_stats = self._load_expert_stats(expert_stats_path)
        else:
            expert_stats = [None for _ in range(self.n_layers)]

        self.load_balancer = {}
        for layer_id in self.moe_layer_id_list:
            cur_load_balancer = MoELargeScaleNaiveLoadBalancer(
                self.num_experts,
                self.n_global_experts_slots,
                self.ep_size,
            )
            cur_load_balancer.update_expert_mapping(
                expert_stats=expert_stats[layer_id],
            )
            self.load_balancer[layer_id] = cur_load_balancer

    def get_expert_mapping(
        self,
        layer_id: int,
    ):
        return self.load_balancer[layer_id].get_expert_mapping(self.ep_rank)
