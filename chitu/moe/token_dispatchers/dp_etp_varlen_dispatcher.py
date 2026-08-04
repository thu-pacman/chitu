# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Graph-safe variable-length DP <-> ETP token dispatch for Hygon.

For EP=1, TP=1, DP=ETP=8, all ETP ranks execute the same statically-shaped
expert batch. Device-side counts control how many rows each DP rank actually
contributes and receives, without assuming an even token distribution.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from typing_extensions import override

from chitu.distributed.comm_group import CommGroup
from chitu.lazy import eval_lazy
from chitu.native_layout import NativeLayoutTensor
from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
)
from chitu.moe.token_dispatchers.base import MoETokenDispatcher
from chitu.utils import ceil_div


class MoEDPETPVarlenTokenDispatcher(MoETokenDispatcher):
    """Compact DP all-gather, ETP compute, then compact reduce-scatter.

    Activation, routing weights, and routing ids are packed into one BF16 row,
    so every MoE layer needs one custom all-gather and one custom
    reduce-scatter. The fixed global row capacity makes the calls CUDA/HIP-
    graph capturable; ``local_count`` remains a device scalar and may differ
    across ranks, including zero or all global rows on one rank.
    """

    def __init__(
        self,
        num_experts: int,
        global_capacity: int,
        *,
        tp_group: CommGroup,
        dp_group: CommGroup,
        etp_group: CommGroup,
        ep_group: CommGroup,
    ):
        super().__init__(
            tp_group=tp_group,
            dp_group=dp_group,
            etp_group=etp_group,
            ep_group=ep_group,
        )
        if global_capacity <= 0:
            raise ValueError(
                "DP+ETP global token capacity must be positive, got "
                f"{global_capacity}"
            )
        if dp_group.group_size != 8 or etp_group.group_size != 8:
            raise ValueError(
                "Hygon variable-length DP+ETP dispatch currently requires "
                f"DP=ETP=8, got DP={dp_group.group_size}, ETP={etp_group.group_size}"
            )
        if (
            dp_group.rank_list != etp_group.rank_list
            or dp_group.rank_in_group != etp_group.rank_in_group
        ):
            raise ValueError(
                "DP and ETP groups must contain the same ranks in the same order"
            )
        if ep_group.group_size != 1 or tp_group.group_size != 1:
            raise ValueError("Hygon variable-length DP+ETP dispatch requires EP=TP=1")

        manager = etp_group.get_custom_ar_manager
        if (
            manager is None
            or manager.disabled
            or not getattr(manager, "supports_varlen_collectives", False)
        ):
            raise RuntimeError(
                "Hygon variable-length DP+ETP dispatch requires the custom "
                "collective manager with varlen AG/RS support"
            )

        self.num_global_experts = int(num_experts)
        self.global_capacity = int(global_capacity)
        self.group_size = etp_group.group_size
        self.manager = manager
        self.local_count = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.local_num_tokens = 0
        self.enabled_for_step = False

    @override
    def prepare(self, num_tokens: int, enabled: bool = True) -> None:
        num_tokens = int(num_tokens)
        if enabled and not 0 <= num_tokens <= self.global_capacity:
            raise ValueError(
                f"local DP token count {num_tokens} exceeds fixed global "
                f"capacity {self.global_capacity}"
            )
        self.enabled_for_step = bool(enabled)
        self.local_num_tokens = num_tokens if enabled else 0
        # prepare() is called by the executor before graph replay.  Keeping the
        # count in a stable device allocation lets one captured collective read
        # a different count on every rank without a host-side shape exchange.
        self.local_count.fill_(self.local_num_tokens)

    @override
    def enter_moe(
        self,
        x: BatchedRoutedActivation,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[BatchedRoutedActivation, Optional[torch.Tensor]]:
        if not isinstance(x, IndexedBatchedRoutedActivation):
            raise NotImplementedError(
                f"{type(x)} not supported for {type(self).__name__}.enter_moe"
            )
        return self._enter_moe_indexed(
            x,
            topk_weights,
            may_fuse_quant=may_fuse_quant,
            may_fuse_quant_kwargs=may_fuse_quant_kwargs,
            layer_id=layer_id,
        )

    def _enter_moe_indexed(
        self,
        x: IndexedBatchedRoutedActivation,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[IndexedBatchedRoutedActivation, Optional[torch.Tensor]]:
        if not self.enabled_for_step:
            raise RuntimeError("DP+ETP dispatcher entered while disabled")
        if x.activation.dtype != torch.bfloat16:
            raise TypeError(
                f"DP+ETP packed dispatch requires BF16 activation, got {x.activation.dtype}"
            )
        if topk_weights.dtype != torch.bfloat16:
            raise TypeError(
                f"DP+ETP packed dispatch requires BF16 weights, got {topk_weights.dtype}"
            )
        if x.token_to_expert_indices.dtype != torch.int32:
            raise TypeError(
                "DP+ETP packed dispatch requires int32 routing ids, got "
                f"{x.token_to_expert_indices.dtype}"
            )

        n = x.activation.shape[0]
        if n != self.local_num_tokens:
            raise ValueError(
                f"prepared token count {self.local_num_tokens} does not match "
                f"activation rows {n}"
            )
        if topk_weights.ndim != 2 or x.token_to_expert_indices.ndim != 2:
            raise ValueError("routing ids and weights must both be 2-D")
        if topk_weights.shape != x.token_to_expert_indices.shape:
            raise ValueError("routing ids and weights must have identical shapes")
        if topk_weights.shape[0] != n:
            raise ValueError("routing metadata row count must match activation")

        hidden = x.activation.shape[1]
        topk = topk_weights.shape[1]
        # int32 routing ids are reinterpreted as two BF16 values each.  No
        # numerical conversion is performed, so the exact id bits survive AG.
        ids_bf16_width = topk * 2
        packed_width = hidden + ids_bf16_width + topk
        if packed_width % 8 != 0:
            raise ValueError(
                "packed DP+ETP rows must be a multiple of 16 bytes, got "
                f"{packed_width * 2} bytes"
            )
        required_staging_bytes = (
            self.global_capacity * packed_width * x.activation.element_size()
        )
        if required_staging_bytes > self.manager.max_size:
            raise ValueError(
                "DP+ETP global capacity does not fit the registered staging "
                f"buffer: need {required_staging_bytes} bytes for "
                f"{self.global_capacity} rows, have {self.manager.max_size}"
            )
        local_packed = torch.zeros(
            (self.global_capacity, packed_width),
            dtype=torch.bfloat16,
            device=x.activation.device,
        )
        if n:
            local_packed[:n, :hidden].copy_(x.activation)
            ids_bits = x.token_to_expert_indices.contiguous().view(torch.bfloat16)
            local_packed[:n, hidden : hidden + ids_bf16_width].copy_(ids_bits)
            local_packed[:n, hidden + ids_bf16_width :].copy_(topk_weights)

        global_packed = torch.empty(
            (self.global_capacity, packed_width),
            dtype=torch.bfloat16,
            device=x.activation.device,
        )
        # The manager copies this fixed-global-capacity tensor into its persistent
        # uncached staging allocation before peers pull only the valid rows.
        self.manager.varlen_all_gather(
            local_packed,
            self.local_count,
            out=global_packed,
        )

        global_activation = global_packed[:, :hidden].contiguous()
        global_ids = (
            global_packed[:, hidden : hidden + ids_bf16_width]
            .contiguous()
            .view(torch.int32)
        )
        global_weights = global_packed[:, hidden + ids_bf16_width :].contiguous()

        return (
            IndexedBatchedRoutedActivation(
                global_activation,
                global_ids,
                expected_n_tokens_per_expert=ceil_div(
                    global_ids.numel(), self.num_global_experts
                ),
                expert_ids_are_local=True,
            ),
            global_weights,
        )

    @override
    def exit_moe_prefer_before_local_sum(self) -> bool:
        return False

    @override
    def exit_moe_after_local_sum(self, local_sum_result: torch.Tensor) -> torch.Tensor:
        if not self.enabled_for_step:
            raise RuntimeError("DP+ETP dispatcher exited while disabled")
        evaluated_result = eval_lazy(local_sum_result)
        if isinstance(evaluated_result, NativeLayoutTensor):
            evaluated_result = evaluated_result.convert_to(torch.Tensor)
        assert isinstance(evaluated_result, torch.Tensor)
        local_sum_result = evaluated_result.contiguous()
        expected_rows = self.global_capacity
        if local_sum_result.ndim != 2 or local_sum_result.shape[0] != expected_rows:
            raise ValueError(
                "ETP expert output must have fixed global capacity "
                f"{expected_rows}, got {tuple(local_sum_result.shape)}"
            )
        out = torch.empty(
            (self.global_capacity, local_sum_result.shape[1]),
            dtype=local_sum_result.dtype,
            device=local_sum_result.device,
        )
        self.manager.varlen_reduce_scatter(
            local_sum_result,
            self.local_count,
            out=out,
        )
        # This slice is static within each model graph key.  The collective
        # itself always retains the same global capacity and launch geometry.
        return out[: self.local_num_tokens]

    @override
    def exit_moe_reduce_rank_lists(self) -> Optional[Sequence[Sequence[int]]]:
        return self.etp_group.rank_lists
