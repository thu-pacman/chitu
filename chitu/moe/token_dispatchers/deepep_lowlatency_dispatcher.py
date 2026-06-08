# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from typing import Optional, Sequence
from typing_extensions import override
import functools
import os

import torch

from chitu.distributed.comm_group import CommGroup
from chitu.utils import parse_dtype
from chitu.import_utils import try_import_opt_dep
from chitu.moe.token_dispatchers.base import MoETokenDispatcher
from chitu.moe.batched_expert_result import (
    BatchedExpertResult,
    PerExpertDenseBatchedExpertResultMinimal,
)
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    PerExpertDenseBatchedRoutedActivationMinimal,
    PerExpertDenseBatchedRoutedActivationWithScaleMinimal,
)
from chitu.moe.load_balancer import get_moe_load_planner
from chitu.global_vars import get_global_args
from chitu.device_type import is_blackwell, is_muxi, is_hygon
from contextlib import nullcontext
from chitu.device_type import is_hygon

# replace the buffer setting with DeepEP to concurrently enbale ll mode and normal mode, need more test to verify.
from chitu.moe.token_dispatchers.buffercontroller import DeepEPBuffer

deep_ep, has_deep_ep = try_import_opt_dep("deep_ep", "deep_ep")

logger = getLogger(__name__)


class MoELowLatencyTokenDispatcher(MoETokenDispatcher):

    def __init__(
        self,
        num_experts: int,
        hidden: int,
        max_bs_per_dp_rank: int,
        profile: bool = False,
        mode: str = "deepep-ll",
        *,
        tp_group: CommGroup,
        dp_group: CommGroup,
        etp_group: CommGroup,
        ep_group: CommGroup,
        moe_layer_id_list: list[int],
    ):
        super().__init__(
            tp_group=tp_group, dp_group=dp_group, etp_group=etp_group, ep_group=ep_group
        )
        self.ep_etp_group = self.ep_group.cartesian_product(self.etp_group)
        self.num_global_experts = num_experts
        os.environ["DEEPEP_DISABLE_LL_DISPATCH_OPT"] = (
            "0" if self.ep_etp_group.group_size % 8 == 0 else "1"
        )
        self._buffer = None
        self.hidden = hidden
        self.max_bs_per_dp_rank = max_bs_per_dp_rank
        self.mode = mode

        # NOTES: for the best performance, the QP number **must** be equal to the number of the local experts
        assert self.num_global_experts % self.ep_group.group_size == 0
        self.num_local_experts = self.num_global_experts // self.ep_group.group_size

        self.profile = profile
        self.prepare_profile = False

        self.moe_layer_id_list = moe_layer_id_list
        self.dispatch_stream = None
        if not is_muxi():
            self.dispatch_stream = torch.cuda.Stream()

    def prepare_decode_profile(self):
        if self.prepare_profile:
            return
        self.prepare_profile = True

        # layer_id -> stat
        self.cumulative_local_expert_recv_stats: dict[int, torch.Tensor] = {}
        if self.profile:
            self.cumulative_local_expert_recv_stats = {
                i: torch.zeros(
                    (self.num_local_experts,), dtype=torch.int, device="cuda"
                )
                for i in self.moe_layer_id_list
            }

    def dump_and_reset_profile(self):
        if self.profile:
            # TODO(zms): remove moe layer range hard coding
            for layer_id in self.moe_layer_id_list:
                expert_stats = torch.zeros(
                    (self.num_global_experts * self.etp_group.group_size,),
                    dtype=torch.int,
                    device="cuda",
                )
                self.ep_group.all_gather_into_tensor(
                    expert_stats, self.cumulative_local_expert_recv_stats[layer_id]
                )
                self.cumulative_local_expert_recv_stats[layer_id].zero_()
                if self.ep_group.rank_in_group == 0:
                    logger.info(f"{layer_id=} {expert_stats=}")

    @override
    def prepare(self, num_tokens):
        self.prepare_deepep_buffer()
        self.prepare_decode_profile()

    def prepare_deepep_buffer(self):
        DeepEPBuffer.set_dispatch_mode_as_low_latency()
        self._buffer = DeepEPBuffer.get_and_cache_deepep_buffer(
            self.ep_etp_group.gpu_group,
            self.hidden,
            self.max_bs_per_dp_rank,
            2,
            self.mode,
            self.num_global_experts * self.etp_group.group_size,
        )

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
            f"{type(x)} not supported for MoELowLatencyTokenDispatcher.enter_moe"
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
    ) -> tuple[PerExpertDenseBatchedRoutedActivationMinimal, Optional[torch.Tensor]]:
        routed_x, weights, dispatch_stream = self.enter_moe_dispatch_streaming(
            x,
            topk_weights,
            may_fuse_quant=may_fuse_quant,
            may_fuse_quant_kwargs=may_fuse_quant_kwargs,
            layer_id=layer_id,
        )
        if dispatch_stream is not None:
            torch.cuda.current_stream().wait_stream(dispatch_stream)
        return routed_x, weights

    def enter_moe_dispatch_streaming(
        self,
        x: IndexedBatchedRoutedActivation,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[
        BatchedRoutedActivation, Optional[torch.Tensor], Optional[torch.cuda.Stream]
    ]:
        dp_local_bs = topk_weights.shape[0]

        dispatch_use_fp8 = False
        dispatch_use_int8 = may_fuse_quant == "w8a8_dynamic"
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
            # FIXME: Add fp4 option to infer.raise_lower_bit_float_to and use it here
            dispatch_use_fp8 = True

        ctx = nullcontext()
        if self.dispatch_stream is not None:
            self.dispatch_stream.wait_stream(torch.cuda.current_stream())
            ctx = torch.cuda.stream(self.dispatch_stream)
        with ctx:
            # Here we may modify `topk_ids`. Please note that `dispatcher_ctx` must save
            # the MODIFIED `topk_ids`.
            topk_ids = x.token_to_expert_indices.to(torch.int64)
            if self.tp_group.group_size > 1:
                if self.etp_group.group_size == 1:
                    if not self.tp_group.is_first_rank:
                        # Don't dispatch from this rank. It's the same as TP rank 0.
                        topk_ids = torch.full_like(topk_ids, -1)
                elif self.etp_group.group_size == self.tp_group.group_size:
                    # Although undocumented, DeepEP does not support non-contiguous ranks in EP
                    # group, which means we can't put ETP groups inside EP groups. Therefore,
                    # we have to use virtual expert IDs to mimick the ETP group here.
                    topk_ids = (
                        topk_ids
                        // self.num_local_experts
                        * (self.num_local_experts * self.etp_group.group_size)
                        + self.etp_group.rank_in_group * self.num_local_experts
                        + topk_ids % self.num_local_experts
                    )
                else:
                    raise NotImplementedError(
                        "Only TP=1, TP=ETP, (TP>1 and ETP=1) are supported in MoELowLatencyTokenDispatcher"
                    )

            recv_activation, recv_expert_count, deepep_handle, event, hook = (
                self.deepep_token_dispatch(
                    x.activation,
                    topk_ids,
                    return_recv_hook=True,
                    dispatch_use_fp8=dispatch_use_fp8,
                    dispatch_use_int8=dispatch_use_int8,
                    round_scale_to_pow2=round_scale_to_pow2,
                    cumulative_local_expert_recv_stats=(
                        self.cumulative_local_expert_recv_stats.get(layer_id, None)
                    ),
                )
            )
        hook()

        # TODO(zms): A more flexible context management.
        # Currently, we should call permutation + unpermutation contiguously.
        self.dispatcher_ctx = (deepep_handle, topk_ids, topk_weights, dp_local_bs)
        planner = get_moe_load_planner()
        if planner is not None:
            planner.record_global_slot_activations(
                layer_id=layer_id, local_slot_stats=recv_expert_count
            )

        if dispatch_use_int8:
            recv_activation, recv_activation_scale = recv_activation
            return (
                PerExpertDenseBatchedRoutedActivationWithScaleMinimal(
                    activation_per_expert=recv_activation,
                    activation_scale_per_expert=recv_activation_scale,
                    quant_method="w8a8_dynamic",
                    output_dtype=x.activation.dtype,
                    # NOTE on expected_n_tokens_per_expert: Although the estimation here is based
                    # on information per DP rank and not global, we have to use it because our
                    # CUDA graph is also captured per DP rank. It should be close enough. But a
                    # DP rank may have 0 tokens, so we have to max with 1 here.
                    expected_n_tokens_per_expert=max(x.expected_n_tokens_per_expert, 1),
                    n_tokens_per_expert=recv_expert_count,
                    expert_ids_are_local=True,
                ),
                None,
                self.dispatch_stream,
            )
        elif not dispatch_use_fp8:
            return (
                PerExpertDenseBatchedRoutedActivationMinimal(
                    activation_per_expert=recv_activation,
                    # NOTE on expected_n_tokens_per_expert: Although the estimation here is based
                    # on information per DP rank and not global, we have to use it because our
                    # CUDA graph is also captured per DP rank. It should be close enough. But a
                    # DP rank may have 0 tokens, so we have to max with 1 here.
                    expected_n_tokens_per_expert=max(x.expected_n_tokens_per_expert, 1),
                    n_tokens_per_expert=recv_expert_count,
                    expert_ids_are_local=True,
                ),
                None,
                self.dispatch_stream,
            )
        else:
            recv_activation, recv_activation_scale = recv_activation
            return (
                PerExpertDenseBatchedRoutedActivationWithScaleMinimal(
                    activation_per_expert=recv_activation,
                    activation_scale_per_expert=recv_activation_scale,
                    quant_method="blockfp8",
                    output_dtype=None,
                    # NOTE on expected_n_tokens_per_expert: Although the estimation here is based
                    # on information per DP rank and not global, we have to use it because our
                    # CUDA graph is also captured per DP rank. It should be close enough. But a
                    # DP rank may have 0 tokens, so we have to max with 1 here.
                    expected_n_tokens_per_expert=max(x.expected_n_tokens_per_expert, 1),
                    n_tokens_per_expert=recv_expert_count,
                    expert_ids_are_local=True,
                ),
                None,
                self.dispatch_stream,
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
            f"{type(expert_result)} not supported for MoELowLatencyTokenDispatcher.exit_moe_before_local_sum"
        )

    @exit_moe_before_local_sum.register
    def _(
        self, expert_result: PerExpertDenseBatchedExpertResultMinimal
    ) -> torch.Tensor:
        handle, topk_ids, topk_weights, dp_local_bs = self.dispatcher_ctx
        # Now we disable any type of overlap.
        outputs, _, _ = self.deepep_token_combine(
            expert_result.activation_per_expert,
            topk_ids,
            topk_weights,
            handle,
            dp_local_bs,
        )

        if self.tp_group.group_size > 1:
            if self.etp_group.group_size == 1:
                torch.distributed.broadcast(
                    outputs,
                    src=self.tp_group.rank_list[0],
                    group=self.tp_group.gpu_group,
                )
            elif self.etp_group.group_size == self.tp_group.group_size:
                self.etp_group.all_reduce(outputs)
            else:
                raise NotImplementedError(
                    "Only TP=1, TP=ETP, (TP>1 and ETP=1) are supported in MoELowLatencyTokenDispatcher"
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
        dispatch_use_int8: bool = False,
        round_scale_to_pow2: bool = False,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        if dispatch_use_fp8 and dispatch_use_int8:
            raise ValueError(
                "DeepEP low-latency dispatch cannot use FP8 and INT8 together"
            )

        assert not (async_finish and return_recv_hook)
        # Do MoE dispatch, compatible with CUDA graph (but you may restore some buffer status once you replay)
        # ---- _chitu_hygon_lowlatency_dispatch_marker_ ----
        if is_hygon():
            # Hygon DCU DeepEP build replaced the (use_fp8, use_ue8m0, round_scale) trio
            # with an integer quant_type enum (0=none, 1=int8, 2=fp8_e4m3, 3=fp8_ue8m0,
            # 4=fp8_e5m2) and a renamed fp8_round_scale arg.
            # cumulative_local_expert_recv_stats is not implemented in this build
            # (profiling-only on the upstream path).
            quant_type = 1 if dispatch_use_int8 else (2 if dispatch_use_fp8 else 0)
            quant_group_size = (
                0 if dispatch_use_int8 else (128 if dispatch_use_fp8 else 0)
            )
            recv_hidden_states, recv_expert_count, handle, event, hook = (
                self._buffer.low_latency_dispatch(
                    hidden_states,
                    topk_idx,
                    DeepEPBuffer._lowlatency_num_max_dispatch_tokens_per_rank,
                    self.num_global_experts * self.etp_group.group_size,
                    quant_type=quant_type,
                    quant_group_size=quant_group_size,
                    fp8_round_scale=round_scale_to_pow2,
                    async_finish=async_finish,
                    return_recv_hook=return_recv_hook,
                )
            )
        else:
            if dispatch_use_int8:
                raise NotImplementedError(
                    "DeepEP low-latency INT8 dispatch is only implemented for Hygon"
                )
            recv_hidden_states, recv_expert_count, handle, event, hook = (
                self._buffer.low_latency_dispatch(
                    hidden_states,
                    topk_idx,
                    DeepEPBuffer._lowlatency_num_max_dispatch_tokens_per_rank,
                    self.num_global_experts * self.etp_group.group_size,
                    use_fp8=dispatch_use_fp8,
                    round_scale=round_scale_to_pow2,
                    use_ue8m0=False,  # Not using 8bit storage for now
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
        dp_local_bs: int,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        dtype = hidden_states.dtype
        device = hidden_states.device

        assert not (async_finish and return_recv_hook)
        if zero_copy:
            self._buffer.get_next_low_latency_combine_buffer(handle)[
                :, :, :
            ] = hidden_states
        # Do MoE combine, compatible with CUDA graph (but you may restore some buffer status once you replay)
        combined_hidden_states, event, hook = self._buffer.low_latency_combine(
            hidden_states,
            topk_idx,
            topk_weights.to(torch.float32),
            handle,
            zero_copy=zero_copy,
            async_finish=async_finish,
            return_recv_hook=return_recv_hook,
        )
        if self.tp_group.group_size > 1 and self.etp_group.group_size == 1:
            if self.tp_group.is_first_rank:
                assert tuple(combined_hidden_states.shape) == (
                    dp_local_bs,
                    self.hidden,
                ), f"combined_hidden_states.shape ({combined_hidden_states.shape}) should be ({dp_local_bs}, {self.hidden})"
                assert combined_hidden_states.dtype == dtype
                assert combined_hidden_states.device == device
            else:
                combined_hidden_states = torch.empty(
                    (dp_local_bs, self.hidden), dtype=dtype, device=device
                )
        else:
            assert tuple(combined_hidden_states.shape) == (
                dp_local_bs,
                self.hidden,
            ), f"combined_hidden_states.shape ({combined_hidden_states.shape}) should be ({dp_local_bs}, {self.hidden})"
            assert combined_hidden_states.dtype == dtype
            assert combined_hidden_states.device == device

        # NOTES: the same behavior as described in the dispatch kernel
        return combined_hidden_states, event, hook

    # SPDX-SnippetEnd
