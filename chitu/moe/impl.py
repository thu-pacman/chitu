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

        self._init_token_dispatcher()
        self._init_experts_impl()

    def _init_token_dispatcher(self):
        # [TODO] use args to select token dispatcher
        if self.tp_size > 1:
            _token_dispatcher = MoETPTokenDispatcher()
            self.prefill_token_dispatcher = _token_dispatcher
            self.decode_token_dispatcher = _token_dispatcher
        else:
            self.prefill_token_dispatcher = MoEAllGatherTokenDispatcher()
            if (
                self.args.infer.moe.decode_token_dispatcher == "lowlatency"
                and has_deep_ep
            ):
                self.decode_token_dispatcher = MoELowLatencyTokenDispatcher(
                    self.num_experts,
                    self.hidden_dim,
                    deepep_use_fp8=True,
                )
            else:
                self.decode_token_dispatcher = MoEAllGatherTokenDispatcher()

    def _get_current_token_dispatcher(self) -> MoETokenDispatcher:
        if self.task_type == "prefill":
            return self.prefill_token_dispatcher
        elif self.task_type == "decode":
            return self.decode_token_dispatcher
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")

    def _init_experts_impl(self):
        # [TODO] check something, should we use enum? no need
        self.impl_map = {
            "prefill": self.args.infer.moe.prefill_experts_impl,
            "decode": self.args.infer.moe.decode_experts_impl,
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
