# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from typing import Optional, Sequence
from typing_extensions import override
import functools

import torch

from chitu.device_type import is_blackwell
from chitu.global_vars import get_global_args
from chitu.utils import parse_dtype, ceil_div
from chitu.import_utils import try_import_opt_dep
from chitu.distributed.comm_group import CommGroup
from chitu.moe.token_dispatchers.base import MoETokenDispatcher
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    IndexedBatchedRoutedActivationWithPaddedPerExpertCnt,
    IndexedBatchedRoutedActivationBlockfp8,
    IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt,
)

# replace the buffer setting with DeepEP to concurrently enbale ll mode and normal mode, need more test to verify.
from chitu.moe.token_dispatchers.buffercontroller import DeepEPBuffer

deep_ep, has_deep_ep = try_import_opt_dep("deep_ep", "deep_ep")

logger = getLogger(__name__)


class MoENormalTokenDispatcher(MoETokenDispatcher):

    def __init__(
        self,
        num_experts: int,
        hidden: int,
        max_bs_per_dp_rank: int,
        profile: bool = False,
        mode: str = "deepep-normal",
        *,
        tp_group: CommGroup,
        dp_group: CommGroup,
        etp_group: CommGroup,
        ep_group: CommGroup,
    ):
        super().__init__(
            tp_group=tp_group, dp_group=dp_group, etp_group=etp_group, ep_group=ep_group
        )
        self.ep_etp_group = self.ep_group.cartesian_product(self.etp_group)
        self.num_global_experts = num_experts
        self._buffer = None
        self.hidden = hidden
        self.max_bs_per_dp_rank = max_bs_per_dp_rank
        self.profile = profile
        self.mode = mode
        # Set the number of SMs to use
        # NOTES: this is a static variable, so it will be shared by all the instances of the class
        deep_ep.Buffer.set_num_sms(24)
        assert self.num_global_experts % self.ep_group.group_size == 0
        self.num_local_experts = self.num_global_experts // self.ep_group.group_size

    @override
    def prepare(self, num_tokens):
        # NOTES: you may also replace `get_*_config` with your auto-tuned results via all the tests

        self._buffer = DeepEPBuffer.get_and_cache_deepep_buffer(
            self.ep_etp_group.gpu_group,
            self.hidden,
            self.max_bs_per_dp_rank,
            2,
            self.mode,
            self.num_global_experts * self.etp_group.group_size,
        )
        DeepEPBuffer.set_dispatch_mode_as_normal()

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
            f"{type(x)} not supported for MoENormalTokenDispatcher.enter_moe"
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
    ) -> tuple[
        IndexedBatchedRoutedActivationWithPaddedPerExpertCnt, Optional[torch.Tensor]
    ]:
        dispatch_use_fp8 = False
        round_scale_to_pow2 = False
        if (
            may_fuse_quant == "blockfp8"
            and may_fuse_quant_kwargs.get("block_size", 128) == 128
            and parse_dtype(get_global_args().infer.raise_lower_bit_float_to).itemsize
            <= 1
        ):
            dispatch_use_fp8 = True
            round_scale_to_pow2 = may_fuse_quant_kwargs.get(
                "round_scale_to_pow2", False
            )
        if may_fuse_quant == "blockfp4" and not is_blackwell():
            dispatch_use_fp8 = True
        if dispatch_use_fp8:
            from chitu.ops.quant.blockfp8 import blockfp8_act_quant

            hidden_states_fp8, scale = blockfp8_act_quant(
                x.activation, block_size=128, round_scale_to_pow2=round_scale_to_pow2
            )
            return self.enter_moe(
                IndexedBatchedRoutedActivationBlockfp8(
                    activation=hidden_states_fp8,
                    token_to_expert_indices=x.token_to_expert_indices,
                    activation_scale=scale,
                    expected_n_tokens_per_expert=x.expected_n_tokens_per_expert,
                    expert_ids_are_local=True,
                ),
                topk_weights,
                may_fuse_quant=may_fuse_quant,
                may_fuse_quant_kwargs=may_fuse_quant_kwargs,
                layer_id=layer_id,
            )

        dp_local_bs = topk_weights.shape[0]
        (
            recv_activation,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = self.dispatch_forward(
            x.activation,
            x.token_to_expert_indices.to(torch.int64),
            topk_weights.to(torch.float32),
        )

        self.dispatch_ctx = (handle, recv_topk_idx, recv_topk_weights, dp_local_bs)
        # `num_recv_tokens_per_expert_list` is host-side (std::vector<int>) with each
        # entry already aligned to `expert_alignment=128`, so `sum()` is the exact
        # padded row count and costs no extra device sync.
        n_tokens_padded = sum(num_recv_tokens_per_expert_list)
        return (
            IndexedBatchedRoutedActivationWithPaddedPerExpertCnt(
                recv_activation,
                recv_topk_idx.to(torch.int32),
                torch.tensor(
                    num_recv_tokens_per_expert_list,
                    dtype=torch.int32,
                    device=recv_topk_idx.device,
                ),
                pad_block_size=128,
                n_tokens_padded=n_tokens_padded,
                # NOTE on expected_n_tokens_per_expert: Recompute using info local to EP,
                # because DP ranks may be inbalance, and cannot reflect EP reality.
                expected_n_tokens_per_expert=ceil_div(
                    n_tokens_padded, len(num_recv_tokens_per_expert_list)
                ),
                expert_ids_are_local=True,
            ),
            recv_topk_weights,
        )

    @enter_moe.register
    def _(
        self,
        x: IndexedBatchedRoutedActivationBlockfp8,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[
        IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt,
        Optional[torch.Tensor],
    ]:
        dp_local_bs = topk_weights.shape[0]
        (
            (recv_activation, recv_activation_scale),
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = self.dispatch_forward(
            (x.activation, x.activation_scale),
            x.token_to_expert_indices.to(torch.int64),
            topk_weights.to(torch.float32),
        )

        self.dispatch_ctx = (handle, recv_topk_idx, recv_topk_weights, dp_local_bs)
        n_tokens_padded = sum(num_recv_tokens_per_expert_list)
        return (
            IndexedBatchedRoutedActivationBlockfp8WithPaddedPerExpertCnt(
                activation=recv_activation,
                activation_scale=recv_activation_scale,
                token_to_expert_indices=recv_topk_idx.to(torch.int32),
                n_tokens_per_expert_padded=torch.tensor(
                    num_recv_tokens_per_expert_list,
                    dtype=torch.int32,
                    device=recv_topk_idx.device,
                ),
                pad_block_size=128,
                n_tokens_padded=n_tokens_padded,
                # NOTE on expected_n_tokens_per_expert: Recompute using info local to EP,
                # because DP ranks may be inbalance, and cannot reflect EP reality.
                expected_n_tokens_per_expert=ceil_div(
                    n_tokens_padded, len(num_recv_tokens_per_expert_list)
                ),
                expert_ids_are_local=True,
            ),
            recv_topk_weights,
        )

    @override
    def exit_moe_prefer_before_local_sum(self) -> bool:
        return False

    @override
    def exit_moe_after_local_sum(
        self, local_sum_result, previous_event: Optional["deep_ep.EventOverlap"] = None
    ):
        handle, topk_ids, topk_weights, dp_local_bs = self.dispatch_ctx
        combined_x, event = self.combine_forward(
            local_sum_result, topk_weights, handle, dp_local_bs
        )
        # TODO: may set `previous_event`,
        return combined_x

    # SPDX-SnippetBegin
    # SPDX-License-Identifier: MIT
    # SPDX-SnippetCopyrightText: 2025 DeepSeek
    # SDPX—SnippetName: dispatch_forward from DeepEP README
    #
    # From https://github.com/deepseek-ai/DeepEP/blob/main/README.md
    def dispatch_forward(
        self,
        hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        async_finish: bool = False,
        previous_event: Optional["deep_ep.EventOverlap"] = None,
    ):
        if self.tp_group.group_size > 1:
            if self.etp_group.group_size == 1:
                if not self.tp_group.is_first_rank:
                    # Don't dispatch from this rank. It's the same as TP rank 0.
                    topk_idx = torch.full_like(topk_idx, -1)
            elif self.etp_group.group_size == self.tp_group.group_size:
                # Although undocumented, DeepEP does not support non-contiguous ranks in EP
                # group, which means we can't put ETP groups inside EP groups. Therefore,
                # we have to use virtual expert IDs to mimick the ETP group here.
                topk_idx = (
                    topk_idx
                    // self.num_local_experts
                    * (self.num_local_experts * self.etp_group.group_size)
                    + self.etp_group.rank_in_group * self.num_local_experts
                    + topk_idx % self.num_local_experts
                )
            else:
                raise NotImplementedError(
                    "Only TP=1, TP=ETP, (TP>1 and ETP=1) are supported in MoENormalTokenDispatcher"
                )

        # NOTES: an optional `previous_event` means a CUDA event captured that you want to make it as a dependency
        # of the dispatch kernel, it may be useful with communication-computation overlap. For more information, please
        # refer to the docs of `Buffer.dispatch`
        # Calculate layout before actual dispatch
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = self._buffer.get_dispatch_layout(
            topk_idx,
            self.num_global_experts * self.etp_group.group_size,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=previous_event is not None,
        )
        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive, so this is not compatible with CUDA graph
        # Unless you specify `num_worst_tokens`, but this flag is for intranode only
        # For more advanced usages, please refer to the docs of the `dispatch` function
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = self._buffer.dispatch(
            hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=(previous_event is not None) and async_finish,
            expert_alignment=128,
        )
        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        )

    # SPDX-SnippetEnd

    # SPDX-SnippetBegin
    # SPDX-License-Identifier: MIT
    # SPDX-SnippetCopyrightText: 2025 DeepSeek
    # SDPX—SnippetName: combine_forward from DeepEP README
    #
    # From https://github.com/deepseek-ai/DeepEP/blob/main/README.md
    def combine_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        dp_local_bs: int,
        async_finish: bool = False,
        previous_event: Optional["deep_ep.EventOverlap"] = None,
    ):
        dtype = hidden_states.dtype
        device = hidden_states.device

        combined_x, _, event = self._buffer.combine(
            hidden_states,
            handle,
            topk_weights=topk_weights.to(torch.float32),
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None,
        )
        if self.tp_group.group_size > 1 and self.etp_group.group_size == 1:
            if self.tp_group.is_first_rank:
                assert tuple(combined_x.shape) == (
                    dp_local_bs,
                    self.hidden,
                ), f"combined_x.shape ({combined_x.shape}) should be ({dp_local_bs}, {self.hidden})"
                assert combined_x.dtype == dtype
                assert combined_x.device == device
            else:
                combined_x = torch.empty(
                    (dp_local_bs, self.hidden), dtype=dtype, device=device
                )
        else:
            assert tuple(combined_x.shape) == (
                dp_local_bs,
                self.hidden,
            ), f"combined_x.shape ({combined_x.shape}) should be ({dp_local_bs}, {self.hidden})"
            assert combined_x.dtype == dtype
            assert combined_x.device == device

        if self.tp_group.group_size > 1:
            if self.etp_group.group_size == 1:
                torch.distributed.broadcast(
                    combined_x,
                    src=self.tp_group.rank_list[0],
                    group=self.tp_group.gpu_group,
                )
            elif self.etp_group.group_size == self.tp_group.group_size:
                self.etp_group.all_reduce(combined_x)
            else:
                raise NotImplementedError(
                    "Only TP=1, TP=ETP, (TP>1 and ETP=1) are supported in MoENormalTokenDispatcher"
                )

        return combined_x, event

    # SPDX-SnippetEnd

    def dump_and_reset_profile(self):
        if self.profile:
            logger.warning("Normal dispatcher cannot profile yet.")
