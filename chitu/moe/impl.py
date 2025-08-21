# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Optional
from chitu.utils import try_import_opt_dep

from .token_dispatchers import (
    MoETokenDispatcher,
    MoETPTokenDispatcher,
    MoEAllGatherTokenDispatcher,
)

deep_ep, has_deep_ep = try_import_opt_dep("deep_ep", "deep_ep")

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
        self.args = args
        self.ep_size = args.infer.ep_size
        self.tp_size = args.infer.tp_size
        self.dp_size = args.infer.dp_size
        self.hidden_dim = args.models.dim

        self.num_experts = getattr(args.models, "n_routed_experts", None) or getattr(
            args.models, "num_experts", None
        )
        if self.num_experts is None:
            raise ValueError(
                "n_routed_experts or num_experts must be specified in model args"
            )

        self.task_type: str = None

        self.prefill_experts_impl = "auto"
        self.decode_experts_impl = "auto"
        self.use_fp8 = args.infer.moe.deepep_use_fp8
        self.prefill_token_dispatcher_impl = args.infer.moe.prefill_token_dispatcher
        self.decode_token_dispatcher_impl = args.infer.moe.decode_token_dispatcher
        self.use_cuda_graph = args.infer.use_cuda_graph

        self._init_token_dispatcher()
        self._init_experts_impl()

    def _init_token_dispatcher(self):
        # impl selection
        if self.prefill_token_dispatcher_impl == "auto":
            if self.tp_size > 1:
                self.prefill_token_dispatcher_impl = "tp"
            elif has_deep_ep:
                self.prefill_token_dispatcher_impl = "deepep-nl"
            else:
                self.prefill_token_dispatcher_impl = "allgather"

        if self.decode_token_dispatcher_impl == "auto":
            if self.tp_size > 1:
                self.decode_token_dispatcher_impl = "tp"
            elif has_deep_ep:
                self.decode_token_dispatcher_impl = "deepep-ll"
            else:
                self.decode_token_dispatcher_impl = "allgather"
                assert (
                    not self.use_cuda_graph
                ), "allgather is not supported with cuda graph"

        # impl initialization
        if self.prefill_token_dispatcher_impl == "tp":
            self.prefill_token_dispatcher = MoETPTokenDispatcher()
        elif self.prefill_token_dispatcher_impl == "deepep-nl":
            self.prefill_token_dispatcher = MoENormalTokenDispatcher(
                self.num_experts,
                self.hidden_dim,
                mode=(
                    "auto"
                    if self.decode_token_dispatcher_impl == "deepep-ll"
                    else "deepep-normal"
                ),
            )
            self.prefill_experts_impl = "deepgemm-contiguous"
        elif self.prefill_token_dispatcher_impl == "allgather":
            self.prefill_token_dispatcher = MoEAllGatherTokenDispatcher()
        else:
            raise ValueError(
                f"Invalid prefill token dispatcher: {self.prefill_token_dispatcher_impl}"
            )

        if self.decode_token_dispatcher_impl == "tp":
            self.decode_token_dispatcher = MoETPTokenDispatcher()
        elif self.decode_token_dispatcher_impl == "deepep-ll":
            self.decode_token_dispatcher = MoELowLatencyTokenDispatcher(
                self.num_experts,
                self.hidden_dim,
                deepep_use_fp8=self.use_fp8,
            )
            self.decode_experts_impl = "deepgemm-masked"
        elif self.decode_token_dispatcher_impl == "allgather":
            self.decode_token_dispatcher = MoEAllGatherTokenDispatcher()
        else:
            raise ValueError(
                f"Invalid decode token dispatcher: {self.decode_token_dispatcher_impl}"
            )

    def _get_current_token_dispatcher(self) -> MoETokenDispatcher:
        if self.task_type == "prefill":
            return self.prefill_token_dispatcher
        elif self.task_type == "decode":
            return self.decode_token_dispatcher
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")

    def _init_experts_impl(self):
        self.impl_map = {
            "prefill": self.prefill_experts_impl,
            "decode": self.decode_experts_impl,
        }

    def get_experts_impl(self) -> str:
        return self.impl_map[self.task_type]

    def prepare(self, task_type: str, num_tokens: int) -> None:
        self.task_type = task_type
        self._get_current_token_dispatcher().prepare(num_tokens)

    def token_permutation(self, *args, **kwargs):
        return self._get_current_token_dispatcher().token_permutation(*args, **kwargs)

    def token_unpermutation(self, *args, **kwargs):
        return self._get_current_token_dispatcher().token_unpermutation(*args, **kwargs)
