# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.import_utils import try_import_platform_dep
from chitu.ops.utils import make_op_dispatcher

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")


@make_op_dispatcher
def add_shared_experts(
    weights: torch.Tensor,
    indices: torch.Tensor,
    n_routed_experts: int,
    n_shared_experts: int,
    *,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Add one or more shared experts as always-routed routed experts

    The added routed experts always have indices `n_routed_experts` to
    `n_routed_experts + n_shared_experts - 1`, and weights equal to 1.
    """
    raise NotImplementedError


@add_shared_experts.register_auto
def _auto_add_shared_experts():
    if has_chitu_backend:
        return "cuda"
    return "torch"


@add_shared_experts.register("torch")
def add_shared_experts_torch(
    weights: torch.Tensor,
    indices: torch.Tensor,
    n_routed_experts: int,
    n_shared_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    new_weights = torch.nn.functional.pad(
        weights, (0, n_shared_experts), mode="constant", value=1.0
    )

    assert indices.ndim == 2
    new_indices = torch.cat(
        [
            indices,
            torch.arange(
                n_routed_experts,
                n_routed_experts + n_shared_experts,
                device=indices.device,
                dtype=indices.dtype,
            ).repeat(indices.shape[0], 1),
        ],
        dim=-1,
    )

    return new_weights, new_indices


@add_shared_experts.register("cuda", available=has_chitu_backend)
def _add_shared_experts_cuda(
    weights: torch.Tensor,
    indices: torch.Tensor,
    n_routed_experts: int,
    n_shared_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    new_indices = torch.empty(
        *indices.shape[:-1],
        indices.shape[1] + n_shared_experts,
        dtype=indices.dtype,
        device=indices.device,
    )
    new_weights = torch.empty(
        *weights.shape[:-1],
        weights.shape[1] + n_shared_experts,
        dtype=weights.dtype,
        device=weights.device,
    )
    if indices.numel() > 0:
        chitu_backend.cuda_add_shared_experts(
            new_weights,
            new_indices,
            weights,
            indices,
            n_routed_experts,
            n_shared_experts,
        )
    return new_weights, new_indices
