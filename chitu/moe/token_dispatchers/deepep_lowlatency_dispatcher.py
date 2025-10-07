# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from typing import Optional
from typing_extensions import override
import functools
import math

import torch

from chitu.distributed.parallel_state import get_ep_group
from chitu.utils import try_import_opt_dep
from chitu.moe.token_dispatchers.base import MoETokenDispatcher
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    PerExpertDenseBatchedRoutedActivation,
    PerExpertDenseBatchedRoutedActivationBlockfp8,
)

# replace the buffer setting with DeepEP to concurrently enbale ll mode and normal mode, need more test to verify.
from chitu.moe.token_dispatchers.buffercontroller import DeepEPBuffer

deep_ep, has_deep_ep = try_import_opt_dep("deep_ep", "deep_ep")

logger = getLogger(__name__)


def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


class MoELowLatencyTokenDispatcher(MoETokenDispatcher):

    def __init__(
        self,
        num_experts: int,
        hidden: int,
        deepep_use_fp8: bool = False,
        profile: bool = False,
        mode: str = "deepep-ll",
    ):
        self.num_experts = num_experts
        self._buffer = None
        self.ep_rank = get_ep_group().rank_in_group
        self.group = get_ep_group().gpu_group
        self.hidden = hidden
        self.use_fp8 = deepep_use_fp8
        self.mode = mode

        # NOTES: for the best performance, the QP number **must** be equal to the number of the local experts
        assert self.num_experts % self.group.size() == 0
        self.num_local_experts = self.num_experts // self.group.size()

        self.profile = profile
        self.prepare_profile = False

    def prepare_decode_profile(self):
        if self.prepare_profile:
            return
        self.prepare_profile = True

        # TODO(zms): remove num_layers hard coding
        if self.profile:
            self.cumulative_local_expert_recv_stats = [
                torch.zeros((self.num_local_experts,), dtype=torch.int, device="cuda")
                for _ in range(61)
            ]
        else:
            self.cumulative_local_expert_recv_stats = [None for _ in range(61)]

    def dump_and_reset_profile(self):
        if self.profile:
            # TODO(zms): remove moe layer range hard coding
            for layer_id in range(3, 61):
                expert_stats = torch.zeros(
                    (self.num_experts,), dtype=torch.int, device="cuda"
                )
                get_ep_group().all_gather_into_tensor(
                    expert_stats, self.cumulative_local_expert_recv_stats[layer_id]
                )
                self.cumulative_local_expert_recv_stats[layer_id].zero_()
                if self.ep_rank == 0:
                    logger.warning(f"{layer_id=} {expert_stats=}")

    @override
    def prepare(self, num_tokens):
        self.prepare_deepep_buffer(num_tokens)
        self.prepare_decode_profile()

    def prepare_deepep_buffer(self, num_tokens):
        # NOTES from DeepEP: the low-latency mode will consume much more space than the normal mode
        # So we recommend that `num_max_dispatch_tokens_per_rank` (the actual batch size in the decoding engine) should be less than 256

        ep_size = self.group.size()
        min_tokens = ep_size * num_tokens
        base_lcm = lcm(ep_size, 256)
        num_max_dispatch_tokens_per_rank = base_lcm
        while num_max_dispatch_tokens_per_rank < min_tokens:
            num_max_dispatch_tokens_per_rank += base_lcm
        num_tokens_per_rank = num_max_dispatch_tokens_per_rank // self.group.size()

        # hard code here
        num_tokens_per_rank = 256
        DeepEPBuffer.set_dispatch_mode_as_low_latency()
        self._buffer = DeepEPBuffer.get_deepep_buffer(
            self.group, self.hidden, 2, self.mode, num_tokens_per_rank, self.num_experts
        )

        self.num_max_dispatch_tokens_per_rank = num_tokens_per_rank

    @override
    @functools.singledispatchmethod
    def token_permutation(
        self,
        x: BatchedRoutedActivation,
        topk_weights: torch.Tensor,
        layer_id: Optional[int] = None,
    ) -> tuple[BatchedRoutedActivation, Optional[torch.Tensor]]:
        raise NotImplementedError(
            f"{type(x)} not supported for MoELowLatencyTokenDispatcher.token_permutation"
        )

    @token_permutation.register
    def _(
        self,
        x: IndexedBatchedRoutedActivation,
        topk_weights: torch.Tensor,
        layer_id: Optional[int] = None,
    ) -> tuple[PerExpertDenseBatchedRoutedActivation, Optional[torch.Tensor]]:
        topk_ids = x.token_to_expert_indices.to(torch.int64)
        recv_activation, recv_expert_count, deepep_handle, event, hook = (
            self.deepep_token_dispatch(
                x.activation,
                topk_ids,
                return_recv_hook=True,
                dispatch_use_fp8=self.use_fp8,
                cumulative_local_expert_recv_stats=(
                    self.cumulative_local_expert_recv_stats[layer_id]
                    if layer_id is not None
                    else None
                ),
            )
        )
        hook()

        # TODO(zms): A more flexible context management.
        # Currently, we should call permutation + unpermutation contiguously.
        self.dispatcher_ctx = (deepep_handle, topk_ids, topk_weights)

        if not self.use_fp8:
            return (
                PerExpertDenseBatchedRoutedActivation(
                    activation_per_expert=recv_activation,
                    n_tokens_per_expert=recv_expert_count,
                ),
                None,
            )
        else:
            recv_activation, recv_activation_scale = recv_activation
            return (
                PerExpertDenseBatchedRoutedActivationBlockfp8(
                    activation_per_expert=recv_activation,
                    activation_scale_per_expert=recv_activation_scale,
                    n_tokens_per_expert=recv_expert_count,
                ),
                None,
            )

    @override
    def token_unpermutation(self, expert_outputs):
        handle, topk_ids, topk_weights = self.dispatcher_ctx
        # Now we disable any type of overlap.
        outputs, _, _ = self.deepep_token_combine(
            expert_outputs, topk_ids, topk_weights, handle
        )
        return outputs

    # SPDX-SnippetBegin
    # SPDX-License-Identifier: MIT
    # SPDX-SnippetCopyrightText: 2025 DeepSeek
    # SDPX—SnippetName: low_latency_dispatch from DeepEP README
    #
    # From https://github.com/deepseek-ai/DeepEP/blob/main/README.md
    def deepep_token_dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        dispatch_use_fp8: bool = False,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        assert not (async_finish and return_recv_hook)
        # Do MoE dispatch, compatible with CUDA graph (but you may restore some buffer status once you replay)
        recv_hidden_states, recv_expert_count, handle, event, hook = (
            self._buffer.low_latency_dispatch(
                hidden_states,
                topk_idx,
                self.num_max_dispatch_tokens_per_rank,
                self.num_experts,
                use_fp8=dispatch_use_fp8,
                cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                async_finish=async_finish,
                return_recv_hook=return_recv_hook,
            )
        )
        # NOTES: the actual tensor will not be received only if you call `hook()`,
        # it is useful for double-batch overlapping, but **without any SM occupation**
        # If you don't want to overlap, please set `return_recv_hook=False`
        # Later, you can use our GEMM library to do the computation with this specific format
        return recv_hidden_states, recv_expert_count, handle, event, hook

    # SPDX-SnippetEnd

    # SPDX-SnippetBegin
    # SPDX-License-Identifier: MIT
    # SPDX-SnippetCopyrightText: 2025 DeepSeek
    # SDPX—SnippetName: low_latency_combine from DeepEP README
    #
    # From https://github.com/deepseek-ai/DeepEP/blob/main/README.md
    def deepep_token_combine(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        assert not (async_finish and return_recv_hook)
        if topk_weights.dtype != torch.float32:
            topk_weights = topk_weights.to(torch.float32)
        if zero_copy:
            self._buffer.get_next_low_latency_combine_buffer(handle)[
                :, :, :
            ] = hidden_states
        # Do MoE combine, compatible with CUDA graph (but you may restore some buffer status once you replay)
        combined_hidden_states, event, hook = self._buffer.low_latency_combine(
            hidden_states,
            topk_idx,
            topk_weights,
            handle,
            zero_copy=zero_copy,
            async_finish=async_finish,
            return_recv_hook=return_recv_hook,
        )

        # NOTES: the same behavior as described in the dispatch kernel
        return combined_hidden_states, event, hook

    # SPDX-SnippetEnd
