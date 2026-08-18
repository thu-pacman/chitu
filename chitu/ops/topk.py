# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from typing import Optional, cast

import torch

from chitu.import_utils import try_import_platform_dep
from chitu.logging_utils import ChituLogger
from chitu.ops.utils import make_op_dispatcher

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
logger = cast(ChituLogger, getLogger(__name__))


has_hygon_indexer_topk = False


def topk_page_table_decode_cuda(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    source_page_table: torch.Tensor,
) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.empty(
            logits.shape[0], 2048, dtype=torch.int32, device=logits.device
        )

    assert has_chitu_backend
    assert logits.dim() == 2
    assert logits.shape[0] == lengths.shape[0]
    assert logits.shape[0] == source_page_table.shape[0]
    assert source_page_table.dtype == torch.int32

    query_cu_seqlens = torch.arange(
        logits.shape[0] + 1,
        dtype=torch.int32,
        device=logits.device,
    )
    page_table = torch.empty(
        logits.shape[0], 2048, dtype=torch.int32, device=logits.device
    )
    chitu_backend.fast_topk_transform(
        logits,
        lengths,
        page_table,
        source_page_table,
        query_cu_seqlens,
    )
    return page_table


@make_op_dispatcher
def topk_indices(
    logits: torch.Tensor,
    k: int,
    *,
    lengths: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    out_dtype=torch.int32,
    impl: str = "auto",
) -> torch.Tensor:
    """
    Return the largest k indices along the last dimension.

    Just like `torch.topk`, except that it returns indices instead of values.

    An optional `lengths` can be set, which means only `lengths[i...]` items are
    valid for each `logits[i...]`. By default they start at column zero. When
    `row_starts` is set, the valid window starts at `row_starts[i...]`, and the
    returned indices are relative to that start. `row_starts` requires
    `lengths`.
    """
    raise NotImplementedError


@topk_indices.register_auto
def _auto_topk_indices(
    logits: torch.Tensor,
    k: int,
    *,
    lengths: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    out_dtype=torch.int32,
):
    if (
        has_chitu_backend
        and (k == 2048 or k == logits.shape[-1])
        and out_dtype == torch.int32
    ):
        return "cuda"
    return "torch"


@topk_indices.register("cuda", available=has_chitu_backend)
def topk_indices_cuda(
    logits: torch.Tensor,
    k: int,
    *,
    lengths: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    out_dtype=torch.int32,
) -> torch.Tensor:
    if row_starts is not None and lengths is None:
        raise ValueError("row_starts requires lengths")
    if logits.numel() == 0:
        return torch.empty(0, k, dtype=out_dtype, device=logits.device)
    if k == logits.shape[-1]:
        return torch.arange(
            logits.shape[-1], device=logits.device, dtype=out_dtype
        ).repeat(*logits.shape[:-1], 1)

    assert len(logits.shape) == 2
    assert k == 2048, "fast_topk only supports k=2048"

    topk_indices = logits.new_empty((logits.shape[0], k), dtype=out_dtype)

    chitu_backend.fast_topk(logits, topk_indices, lengths, row_starts)

    return topk_indices


@topk_indices.register("torch")
def _topk_indices_torch(
    logits: torch.Tensor,
    k: int,
    *,
    lengths: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    out_dtype=torch.int32,
) -> torch.Tensor:
    if row_starts is not None and lengths is None:
        raise ValueError("row_starts requires lengths")
    if lengths is not None:
        columns = torch.arange(logits.shape[-1], device=logits.device)
        if row_starts is None:
            valid = columns < lengths.unsqueeze(-1)
        else:
            valid = (columns >= row_starts.unsqueeze(-1)) & (
                columns < (row_starts + lengths).unsqueeze(-1)
            )
        logits = logits.masked_fill(
            ~valid,
            float("-inf"),
        )
    values, indices = logits.topk(k, dim=-1)
    if row_starts is not None:
        indices = indices - row_starts.unsqueeze(-1)
    return indices.to(out_dtype)
