# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# Modified from DeepSeek's DeepEP project
# https://github.com/deepseek-ai/DeepEP

from logging import getLogger
from typing import Optional, Tuple, Union

import torch

from chitu.distributed.parallel_state import get_ep_group
from chitu.utils import try_import_opt_dep

deep_ep, has_deep_ep = try_import_opt_dep("deep_ep", "deep_ep")

from .base import MoETokenDispatcher

# replace the buffer setting with DeepEP to concurrently enbale ll mode and normal mode, need more test to verify.
from .buffercontroller import DeepEPBuffer

logger = getLogger(__name__)


class MoENormalTokenDispatcher(MoETokenDispatcher):

    def __init__(
        self,
        num_experts: int,
        hidden: int,
        deepep_use_fp8: bool = False,
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

    def prepare(self, num_tokens):
        # NOTES: you may also replace `get_*_config` with your auto-tuned results via all the tests

        self._buffer = DeepEPBuffer.get_deepep_buffer(
            self.group, self.hidden, 2, self.mode, 256, self.num_experts
        )  # FIXME 256 is hard code
        DeepEPBuffer.set_dispatch_mode_as_normal()

    def token_permutation(self, tokens, topk_ids, topk_weights, layer_id: int = 0):
        topk_ids = topk_ids.to(torch.int64)
        topk_weights = topk_weights.to(torch.float32)
        (
            recv_hidden_states,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = self.dispatch_forward(tokens, topk_ids, topk_weights)

        self.dispatch_ctx = (handle, recv_topk_idx, recv_topk_weights)
        recv_topk_idx = recv_topk_idx.to(torch.int32)
        return (
            recv_hidden_states,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
        )

    def token_unpermutation(
        self, expert_outputs, previous_event: Optional["deep_ep.EventOverlap"] = None
    ):
        handle, topk_ids, topk_weights = self.dispatch_ctx
        combined_x, event = self.combine_forward(
            expert_outputs, topk_weights, handle, previous_event=previous_event
        )
        return combined_x

    def dispatch_forward(
        self,
        hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        async_finish: bool = False,
        previous_event: Optional["deep_ep.EventOverlap"] = None,
    ):
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

    def combine_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: Tuple,
        async_finish: bool = False,
        previous_event: Optional["deep_ep.EventOverlap"] = None,
    ):
        if topk_weights.dtype != torch.float32:
            topk_weights = topk_weights.to(torch.float32)

        combined_x, _, event = self._buffer.combine(
            hidden_states,
            handle,
            topk_weights=topk_weights,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None,
        )

        return combined_x, event

    def dump_and_reset_profile(self):
        if self.profile:
            logger.warning("Normal dispatcher cannot profile yet.")
