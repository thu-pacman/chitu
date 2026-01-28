# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import functools

import torch

from chitu.distributed.comm_group import CommGroup
from chitu.moe.batched_expert_result import (
    BatchedExpertResult,
    ConcatPermutedBatchedExpertResultMinimal,
)
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    ConcatPermutedBatchedRoutedActivationMinimal,
    ConcatPermutedBatchedRoutedActivationMinimalAscendInt8,
)
from chitu.moe.token_dispatchers.base import MoETokenDispatcher
from chitu.moe.load_balancer import get_moe_load_planner
from chitu.npu_utils import (
    fused_experts_npu_tp_split,
    fused_experts_npu_tp_all_gather,
    get_hcomm_info,
)
from chitu.utils import try_import_and_setup_torch_npu, ceil_div
from chitu.device_type import is_ascend_910b
from chitu.global_vars import get_global_args

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


class MoENpuDistributeTokenDispatcher(MoETokenDispatcher):
    def __init__(
        self,
        num_experts: int,
        *,
        tp_group: CommGroup,
        dp_group: CommGroup,
        etp_group: CommGroup,
        ep_group: CommGroup,
    ):
        super().__init__(
            tp_group=tp_group, dp_group=dp_group, etp_group=etp_group, ep_group=ep_group
        )
        self.num_global_experts = num_experts
        assert self.num_global_experts % self.ep_group.group_size == 0
        self.num_local_experts = self.num_global_experts // self.ep_group.group_size

        global_rank = torch.distributed.get_rank()
        self.ep_hcomm_info = get_hcomm_info(global_rank, self.ep_group.gpu_group)
        if self.etp_group.group_size > 1:
            self.etp_hcomm_info = get_hcomm_info(global_rank, self.etp_group.gpu_group)
        else:
            self.etp_hcomm_info = ""

    @override
    def prepare(self, num_tokens):
        pass

    @override
    @functools.singledispatchmethod
    def enter_moe(
        self,
        x: BatchedRoutedActivation,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[BatchedRoutedActivation, Optional[torch.Tensor]]:
        raise NotImplementedError(
            f"{type(x)} not supported for MoENpuDistributeTokenDispatcher.enter_moe"
        )

    @enter_moe.register
    def _(
        self,
        x: IndexedBatchedRoutedActivation,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[ConcatPermutedBatchedRoutedActivationMinimal, Optional[torch.Tensor]]:
        hidden_states = x.activation
        topk_ids = x.token_to_expert_indices

        use_int8_w8a8 = may_fuse_quant in ["ascend_w8a8", "ascend_w8a8_dynamic"]

        origin_bs = hidden_states.shape[0]

        if is_ascend_910b():
            if self.ep_group.group_size % 16 != 0:
                raise NotImplementedError(
                    "torch_npu.npu_moe_distribute_dispatch_v2 has an additional limit of ep_size % 16 == 0 on 910B devices"
                )

        if self.etp_group.group_size > 1:
            raise NotImplementedError
        if self.tp_group.group_size > 1 or origin_bs == 0:
            # split inputs from tp group into ep rank
            hidden_states = fused_experts_npu_tp_split(hidden_states)
            topk_weights = fused_experts_npu_tp_split(topk_weights)
            topk_ids = fused_experts_npu_tp_split(
                topk_ids, self.num_local_experts, is_expert_ids=True
            )

        global_bs_for_distpatch_combine = (
            ceil_div(get_global_args().infer.max_reqs, self.ep_group.group_size)
            * self.ep_group.group_size
            * get_global_args().infer.mtp_size
        )

        (
            expand_x,
            dynamic_scales,
            expand_idx,
            expert_token_nums,
            ep_recv_counts,
            tp_recv_counts,
            expand_scales,
        ) = torch_npu.npu_moe_distribute_dispatch_v2(
            x=hidden_states,
            expert_ids=topk_ids,
            group_ep=self.ep_hcomm_info,
            ep_world_size=self.ep_group.group_size,
            ep_rank_id=self.ep_group.rank_in_group,
            # NOTE: `tp` in this API means `etp` in chitu
            group_tp=self.etp_hcomm_info,
            tp_world_size=self.etp_group.group_size,
            tp_rank_id=self.etp_group.rank_in_group,
            shared_expert_rank_num=0,
            moe_expert_num=self.num_global_experts,
            quant_mode=0 if not use_int8_w8a8 else 2,
            global_bs=global_bs_for_distpatch_combine,
        )

        if get_global_args().infer.moe_lb_trigger > 0:
            if (planner := get_moe_load_planner()) is not None:
                planner.record_global_slot_activations(
                    layer_id=layer_id, local_slot_stats=expert_token_nums
                )

        self.topk_ids = topk_ids
        self.topk_weights = topk_weights
        self.expand_idx = expand_idx
        self.expand_scales = expand_scales
        self.ep_recv_counts = ep_recv_counts
        self.tp_recv_counts = tp_recv_counts
        self.origin_bs = origin_bs
        self.global_bs_for_distpatch_combine = global_bs_for_distpatch_combine

        if use_int8_w8a8:
            return (
                ConcatPermutedBatchedRoutedActivationMinimalAscendInt8(
                    concat_activation=expand_x,
                    concat_activation_scale=dynamic_scales,
                    n_tokens_per_expert=expert_token_nums,
                    expert_ids_are_local=True,
                ),
                None,
            )
        else:
            return (
                ConcatPermutedBatchedRoutedActivationMinimal(
                    concat_activation=expand_x,
                    n_tokens_per_expert=expert_token_nums,
                    expert_ids_are_local=True,
                ),
                None,
            )

    @override
    def exit_moe_prefer_before_local_sum(self) -> bool:
        return True

    @override
    @functools.singledispatchmethod
    def exit_moe_before_local_sum(
        self, expert_result: BatchedExpertResult
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(expert_result)} not supported for MoENpuDistributeTokenDispatcher.exit_moe_before_local_sum"
        )

    @exit_moe_before_local_sum.register
    def _(
        self, expert_result: ConcatPermutedBatchedExpertResultMinimal
    ) -> torch.Tensor:
        hidden_states = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=expert_result.concat_activation,
            expert_ids=self.topk_ids,
            assist_info_for_combine=self.expand_idx,
            ep_send_counts=self.ep_recv_counts,
            expert_scales=self.topk_weights.to(torch.float),
            tp_send_counts=self.tp_recv_counts,
            expand_scales=self.expand_scales,
            group_ep=self.ep_hcomm_info,
            ep_world_size=self.ep_group.group_size,
            ep_rank_id=self.ep_group.rank_in_group,
            # NOTE: `tp` in this API means `etp` in chitu
            group_tp=self.etp_hcomm_info,
            tp_world_size=self.etp_group.group_size,
            tp_rank_id=self.etp_group.rank_in_group,
            moe_expert_num=self.num_global_experts,
            global_bs=self.global_bs_for_distpatch_combine,
        )

        if self.origin_bs == 0:
            hidden_states = torch.empty(
                0,
                hidden_states.shape[1],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            return hidden_states

        if self.tp_group.group_size > 1:
            hidden_states = fused_experts_npu_tp_all_gather(
                hidden_states, self.origin_bs
            )

        self.topk_ids = None
        self.topk_weights = None
        self.expand_idx = None
        self.expand_scales = None
        self.ep_recv_counts = None
        self.tp_recv_counts = None
        self.origin_bs = None
        self.global_bs_for_distpatch_combine = None

        return hidden_states
