# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from chitu.import_utils import try_import_platform_dep

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")


def topk_indices_cuda(
    logits: torch.Tensor,
    k: int,
    *,
    lengths: Optional[torch.Tensor] = None,
    out_dtype=torch.int32,
) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.empty(0, k, dtype=out_dtype, device=logits.device)
    if k == logits.shape[-1]:
        return torch.arange(logits.shape[-1], device=logits.device).repeat(
            *logits.shape[:-1], 1
        )

    assert len(logits.shape) == 2
    assert k == 2048, "fast_topk only supports k=2048"

    topk_indices = logits.new_empty((logits.shape[0], k), dtype=out_dtype)

    chitu_backend.fast_topk(logits, topk_indices, lengths)

    return topk_indices


def topk_indices(
    logits: torch.Tensor,
    k: int,
    *,
    lengths: Optional[torch.Tensor] = None,
    out_dtype=torch.int32,
    impl: str = "auto",
) -> torch.Tensor:
    """
    Return the largest k indices along the last dimension.

    Just like `torch.topk`, except that it returns indices instead of values.

    An optional `lengths` can be set, which means only the first `lengths[i...]`
    items are valid for each `logits[i...]`. Mathematically, this is equivalent
    to mask the other items as `-inf`.
    """

    if impl == "auto":
        if (
            has_chitu_backend
            and (k == 2048 or k == logits.shape[-1])
            and out_dtype == torch.int32
        ):
            impl = "cuda"
        else:
            impl = "torch"

    if impl == "cuda":
        return topk_indices_cuda(logits, k, lengths=lengths, out_dtype=out_dtype)

    elif impl == "torch":
        if lengths is not None:
            logits = logits.masked_fill(
                torch.arange(logits.shape[-1], device=logits.device)
                >= lengths.unsqueeze(-1),
                float("-inf"),
            )
        values, indices = logits.topk(k, dim=-1)
        return indices.to(out_dtype)

    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")
