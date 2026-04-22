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
from chitu.npu_utils import fused_experts_npu_tp_split, fused_experts_npu_tp_all_gather
from chitu.utils import try_import_and_setup_torch_npu

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


class MoENpuAllToAllTokenDispatcher(MoETokenDispatcher):
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
            f"{type(x)} not supported for MoENpuAllToAllTokenDispatcher.enter_moe"
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

        if self.etp_group.group_size > 1:
            raise NotImplementedError
        origin_bs = hidden_states.shape[0]
        if self.tp_group.group_size > 1:
            # split inputs from tp group into ep rank
            hidden_states = fused_experts_npu_tp_split(
                hidden_states,
                tp_group=self.tp_group,
                ep_group=self.ep_group,
                is_zero_batch_ok=True,
            )
            topk_weights = fused_experts_npu_tp_split(
                topk_weights,
                tp_group=self.tp_group,
                ep_group=self.ep_group,
                is_zero_batch_ok=True,
            )
            topk_ids = fused_experts_npu_tp_split(
                topk_ids,
                tp_group=self.tp_group,
                ep_group=self.ep_group,
                n_local_experts=self.num_local_experts,
                is_expert_ids=True,
            )

        bs, hidden_dim = hidden_states.shape
        _, topk = topk_weights.shape
        assert (
            hidden_states.dtype == topk_weights.dtype
        ), "hidden_states and topk_weights must have the same dtype"
        topk_ids = topk_ids.int()
        max_num_deployed_expert = self.num_local_experts * self.ep_group.group_size

        if self.etp_group.group_size == 1 or self.etp_group.is_first_rank:
            if not hidden_states.shape[0] == 0:
                (
                    expanded_x,
                    expanded_row_idx,
                    n_tokens_per_expert_local_dp_rank,
                    pertoken_scale,
                ) = torch_npu.npu_moe_init_routing_v2(
                    hidden_states,
                    expert_idx=topk_ids,
                    scale=None,
                    expert_num=max_num_deployed_expert,
                    active_expert_range=[0, max_num_deployed_expert],
                    expert_tokens_num_type=1,
                    expert_tokens_num_flag=True,
                    active_num=topk_ids.numel(),
                    drop_pad_mode=0,
                    row_idx_type=0,
                    quant_mode=1 if use_int8_w8a8 else -1,
                )
            else:
                expanded_x = torch.empty(
                    0,
                    hidden_dim,
                    device=hidden_states.device,
                    dtype=torch.int8 if use_int8_w8a8 else hidden_states.dtype,
                )
                expanded_row_idx = torch.empty(
                    0, device=hidden_states.device, dtype=torch.int32
                )
                n_tokens_per_expert_local_dp_rank = torch.zeros(
                    max_num_deployed_expert,
                    device=hidden_states.device,
                    dtype=torch.int64,
                )
                if use_int8_w8a8:
                    pertoken_scale = torch.empty(
                        0, device=hidden_states.device, dtype=torch.float32
                    )

            assert tuple(expanded_x.shape) == (bs * topk, hidden_dim)
            assert tuple(expanded_row_idx.shape) == (bs * topk,)
            assert expanded_row_idx.dtype == torch.int32
            assert tuple(n_tokens_per_expert_local_dp_rank.shape) == (
                max_num_deployed_expert,
            )
            assert n_tokens_per_expert_local_dp_rank.dtype == torch.int64
            if use_int8_w8a8:
                assert tuple(pertoken_scale.shape) == (bs * topk,)
                assert pertoken_scale.dtype == torch.float32
        else:
            raise NotImplementedError

        n_tokens_per_expert_local_ep_rank = n_tokens_per_expert_local_dp_rank.new_empty(
            n_tokens_per_expert_local_dp_rank.shape[0]
        )
        torch.distributed.all_to_all_single(
            n_tokens_per_expert_local_ep_rank, n_tokens_per_expert_local_dp_rank
        )  # (total_experts,) --> (total_ranks * n_routed_experts_per_rank)
        combine_tokens = torch.stack(
            [n_tokens_per_expert_local_ep_rank, n_tokens_per_expert_local_dp_rank],
            dim=0,
        )

        combine_tokens = combine_tokens.view(2, self.ep_group.group_size, -1).sum(2)
        all_tokens = combine_tokens[0].sum()
        combine_tokens_cpu = combine_tokens.cpu().tolist()
        # alltoall input splits, the total number of tokens routed from the current rank to other ranks
        input_splits = combine_tokens_cpu[1]
        # alltoall output splits, the number of tokens each rank receives from other cards
        output_splits = combine_tokens_cpu[0]
        # alltoall output, unfolded into one dimension, the size is the sum of the number of tokens routed from other cards to the current rank.
        gathered_tokens = expanded_x.new_empty(all_tokens.item(), expanded_x.shape[1])
        torch.distributed.all_to_all_single(
            gathered_tokens, expanded_x, output_splits, input_splits
        )
        if use_int8_w8a8:
            gathered_scales = pertoken_scale.new_empty(all_tokens.item(), 1)
            torch.distributed.all_to_all_single(
                gathered_scales, pertoken_scale, output_splits, input_splits
            )
        (
            hidden_states,
            permute_per_token_scales,
            gathered_idxs_unsort,
            tokens_per_local_expert,
        ) = torch_npu.npu_moe_re_routing(
            gathered_tokens,
            n_tokens_per_expert_local_ep_rank.view(self.ep_group.group_size, -1),
            per_token_scales=None if not use_int8_w8a8 else gathered_scales,
        )

        self.expanded_x_shape = expanded_x.shape
        self.input_splits = input_splits
        self.output_splits = output_splits
        self.topk_weights = topk_weights
        self.origin_bs = origin_bs
        self.gathered_idxs_unsort = gathered_idxs_unsort
        self.expanded_row_idx = expanded_row_idx

        if use_int8_w8a8:
            return (
                ConcatPermutedBatchedRoutedActivationMinimalAscendInt8(
                    concat_activation=hidden_states,
                    concat_activation_scale=permute_per_token_scales,
                    n_tokens_per_expert=tokens_per_local_expert,
                    expert_ids_are_local=True,
                ),
                None,
            )
        else:
            return (
                ConcatPermutedBatchedRoutedActivationMinimal(
                    concat_activation=hidden_states,
                    n_tokens_per_expert=tokens_per_local_expert,
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
            f"{type(expert_result)} not supported for MoENpuAllToAllTokenDispatcher.exit_moe_before_local_sum"
        )

    @exit_moe_before_local_sum.register
    def _(
        self, expert_result: ConcatPermutedBatchedExpertResultMinimal
    ) -> torch.Tensor:
        hidden_states = torch.index_select(
            expert_result.concat_activation,
            0,
            self.gathered_idxs_unsort.to(torch.float32).argsort().to(torch.int32),
        )

        gathered_tokens = hidden_states.new_empty(*self.expanded_x_shape)
        torch.distributed.all_to_all_single(
            gathered_tokens,
            hidden_states,
            self.input_splits,
            self.output_splits,
        )

        if self.etp_group.group_size == 1 or self.etp_group.is_first_rank:
            final_hidden_states = torch_npu.npu_moe_finalize_routing(
                gathered_tokens,
                skip1=None,
                skip2=None,
                bias=None,
                scales=self.topk_weights.to(gathered_tokens.dtype),
                expanded_src_to_dst_row=self.expanded_row_idx,
                export_for_source_row=None,
                drop_pad_mode=2,
            )
            assert final_hidden_states.dtype == hidden_states.dtype
        else:
            raise NotImplementedError

        if self.etp_group.group_size > 1:
            raise NotImplementedError

        if self.tp_group.group_size > 1:
            final_hidden_states = fused_experts_npu_tp_all_gather(
                final_hidden_states, self.tp_group, self.origin_bs
            )

        self.expanded_x_shape = None
        self.input_splits = None
        self.output_splits = None
        self.topk_weights = None
        self.origin_bs = None
        self.gathered_idxs_unsort = None
        self.expanded_row_idx = None

        return final_hidden_states
