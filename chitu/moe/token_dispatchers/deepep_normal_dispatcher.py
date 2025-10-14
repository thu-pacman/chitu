# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from typing import Optional
from typing_extensions import override
import functools

import torch

from chitu.distributed.parallel_state import get_ep_group, get_tp_size, get_tp_group
from chitu.utils import try_import_opt_dep
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
        profile: bool = False,
        mode: str = "deepep-normal",
    ):
        self.num_experts = num_experts
        self._buffer = None
        self.group = get_ep_group().gpu_group
        self.hidden = hidden
        self.profile = profile
        self.mode = mode
        # Set the number of SMs to use
        # NOTES: this is a static variable, so it will be shared by all the instances of the class
        deep_ep.Buffer.set_num_sms(24)
        assert self.num_experts % self.group.size() == 0

    @override
    def prepare(self, num_tokens):
        # NOTES: you may also replace `get_*_config` with your auto-tuned results via all the tests

        self._buffer = DeepEPBuffer.get_deepep_buffer(
            self.group, self.hidden, 2, self.mode, 256, self.num_experts
        )  # FIXME 256 is hard code
        DeepEPBuffer.set_dispatch_mode_as_normal()

    @override
    @functools.singledispatchmethod
    def token_permutation(
        self,
        x: BatchedRoutedActivation,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[BatchedRoutedActivation, Optional[torch.Tensor]]:
        raise NotImplementedError(
            f"{type(x)} not supported for MoENormalTokenDispatcher.token_permutation"
        )

    @token_permutation.register
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
        return (
            IndexedBatchedRoutedActivationWithPaddedPerExpertCnt(
                recv_activation,
                recv_topk_idx.to(torch.int32),
                torch.tensor(
                    num_recv_tokens_per_expert_list,
                    dtype=torch.int32,
                    device=recv_topk_idx.device,
                ),
            ),
            recv_topk_weights,
        )

    @token_permutation.register
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
            ),
            recv_topk_weights,
        )

    @override
    def token_unpermutation(
        self, expert_outputs, previous_event: Optional["deep_ep.EventOverlap"] = None
    ):
        handle, topk_ids, topk_weights, dp_local_bs = self.dispatch_ctx
        combined_x, event = self.combine_forward(
            expert_outputs,
            topk_weights,
            handle,
            dp_local_bs,
            previous_event=previous_event,
        )
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
        if get_tp_size() > 1 and not get_tp_group().is_first_rank:
            # Don't dispatch from this rank. It's the same as TP rank 0.
            topk_idx = torch.full_like(topk_idx, -1)

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
            self.num_experts,
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
        if get_tp_size() == 1 or get_tp_group().is_first_rank:
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

        if get_tp_size() > 1:
            torch.distributed.broadcast(
                combined_x,
                src=get_tp_group().rank_list[0],
                group=get_tp_group().gpu_group,
            )

        return combined_x, event

    # SPDX-SnippetEnd

    def dump_and_reset_profile(self):
        if self.profile:
            logger.warning("Normal dispatcher cannot profile yet.")
