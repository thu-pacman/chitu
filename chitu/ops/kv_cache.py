# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch

from chitu.utils import try_import_platform_dep
from chitu.global_vars import get_global_args

triton, has_triton = try_import_platform_dep("triton")

if has_triton and torch.cuda.is_available():
    from chitu.ops.triton_ops import (
        append_to_paged_kv_cache_triton,
        append_to_dense_kv_cache_triton,
    )

torch_npu, has_torch_npu = try_import_platform_dep("torch_npu")


def append_to_paged_kv_cache(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    this_kv: torch.Tensor,
    delta_position_ids: torch.Tensor,
    delta_seq_ids: Optional[torch.Tensor] = None,
    impl: str = "auto",
):
    """
    Read from ragged K/V, append to paged K/V cache.

    Args:
        kv_cache: (num_pages, page_size, other contiguous dims...). Data of the paged K/V cache.
        page_table: (batch_size, num_pages_per_sample). Page table of the paged K/V cache.
        this_kv: (num_tokens, other contiguous dims...). Ragged K/V.
        delta_position_ids: (num_tokens,). Position IDs of the incremented tokens. E.g, if
            appending the 8th, 9th token of the 1st sequence, and the 7th token of the 2nd
            sequence, delta_position_ids = [8, 9, 7].
        delta_seq_ids: (num_tokens,). Sequence IDs of the incremented tokens. E.g, if appending
            the 8th, 9th token of the 1st sequence, and the 7th token of the 2nd sequence,
            delta_seq_ids = [1, 1, 2]. This parameter can be ignored if the number of incremented
            tokens of every sequence is 1.
    """

    if impl == "auto":
        if get_global_args().infer.op_impl == "cpu":
            impl = "cpu"
        elif has_triton:
            impl = "triton"
        else:
            impl = "torch"

    if get_global_args().infer.dp_size > 1:
        this_kv = this_kv[: delta_position_ids.shape[0]]

    if impl == "triton" and has_triton:
        append_to_paged_kv_cache_triton(
            kv_cache, page_table, this_kv, delta_position_ids, delta_seq_ids
        )
    else:
        append_to_paged_kv_cache_torch(
            kv_cache, page_table, this_kv, delta_position_ids, delta_seq_ids
        )


def append_to_dense_kv_cache(
    kv_cache: torch.Tensor,
    this_kv: torch.Tensor,
    delta_position_ids: torch.Tensor,
    delta_seq_ids: Optional[torch.Tensor] = None,
    impl: str = "auto",
):
    """
    Read from ragged K/V, append to dense K/V cache.

    Args:
        kv_cache: (batch_size, seq_len, other contiguous dims...). Dense K/V cache.
        this_kv: (num_tokens, other contiguous dims...). Ragged K/V.
        delta_position_ids: (num_tokens,). Position IDs of the incremented tokens. E.g, if
            appending the 8th, 9th token of the 1st sequence, and the 7th token of the 2nd
            sequence, delta_position_ids = [8, 9, 7].
        delta_seq_ids: (num_tokens,). Sequence IDs of the incremented tokens. E.g, if appending
            the 8th, 9th token of the 1st sequence, and the 7th token of the 2nd sequence,
            delta_seq_ids = [1, 1, 2]. This parameter can be ignored if the number of incremented
            tokens of every sequence is 1.
    """

    if impl == "auto":
        if has_triton:
            impl = "triton"
        else:
            impl = "torch"

    if get_global_args().infer.dp_size > 1:
        this_kv = this_kv[: delta_position_ids.shape[0]]

    if impl == "triton" and has_triton:
        append_to_dense_kv_cache_triton(
            kv_cache, this_kv, delta_position_ids, delta_seq_ids
        )
    elif impl == "torch":
        append_to_dense_kv_cache_torch(
            kv_cache, this_kv, delta_position_ids, delta_seq_ids
        )
    elif impl == "torch_npu" and has_torch_npu:
        torch_npu.scatter_update_(kv_cache, delta_position_ids, this_kv, 1)


def append_to_paged_kv_cache_torch(
    kv_cache: torch.Tensor,  # (num_pages, page_size, other contiguous dims...)
    page_table: torch.Tensor,  # (batch_size, num_pages_per_sample)
    this_kv: torch.Tensor,  # (num_tokens, other contiguous dims...)
    delta_position_ids: torch.Tensor,  # (num_tokens,)
    delta_seq_ids: Optional[torch.Tensor] = None,  # (num_tokens,)
):
    if delta_seq_ids is None:
        if page_table.shape[0] != delta_position_ids.shape[0]:
            raise ValueError(
                f"batch_size ({page_table.shape[0]}) must be equal to num_tokens "
                f"({delta_position_ids.shape[0]}) if ignoring delta_seq_ids"
            )
        delta_seq_ids = torch.arange(
            delta_position_ids.shape[0],
            dtype=delta_position_ids.dtype,
            device=delta_position_ids.device,
        )

    page_size = kv_cache.shape[1]
    kv_cache[
        page_table[delta_seq_ids, delta_position_ids // page_size],
        delta_position_ids % page_size,
    ] = this_kv.view(this_kv.shape[0], *kv_cache.shape[2:])


def append_to_dense_kv_cache_torch(
    kv_cache: torch.Tensor,  # (batch_size, seq_len, other contiguous dims...)
    this_kv: torch.Tensor,  # (num_tokens, other contiguous dims...)
    delta_position_ids: torch.Tensor,  # (num_tokens,)
    delta_seq_ids: Optional[torch.Tensor] = None,  # (num_tokens,)
):
    if delta_seq_ids is None:
        if kv_cache.shape[0] != delta_position_ids.shape[0]:
            raise ValueError(
                f"batch_size ({kv_cache.shape[0]}) must be equal to num_tokens "
                f"({delta_position_ids.shape[0]}) if ignoring delta_seq_ids"
            )
        delta_seq_ids = torch.arange(
            delta_position_ids.shape[0],
            dtype=delta_position_ids.dtype,
            device=delta_position_ids.device,
        )

    kv_cache[delta_seq_ids, delta_position_ids] = this_kv.view(
        this_kv.shape[0], *kv_cache.shape[2:]
    )
