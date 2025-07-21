import torch

from chitu.utils import try_import_opt_dep

triton, has_triton = try_import_opt_dep("triton", "triton")

if has_triton:
    from chitu.ops.triton_ops import (
        append_to_paged_kv_cache_triton,
        append_to_non_paged_kv_cache_triton,
    )


def append_to_paged_kv_cache(
    kv_cache,  # (num_pages, page_size, other contiguous dims...)
    page_table,  # (batch_size, num_pages_per_sample)
    this_kv,  # (batch_size, other contiguous dims...)
    old_seq_lens,  # (batch_size,)
    impl: str = "auto",
):
    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        append_to_paged_kv_cache_triton(kv_cache, page_table, this_kv, old_seq_lens)
    else:
        append_to_paged_kv_cache_torch(kv_cache, page_table, this_kv, old_seq_lens)


def append_to_non_paged_kv_cache(
    kv_cache,  # (batch_size, seq_len, other contiguous dims...)
    this_kv,  # (batch_size, other contiguous dims...)
    old_seq_lens,  # (batch_size,)
    impl: str = "auto",
):
    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        append_to_non_paged_kv_cache_triton(kv_cache, this_kv, old_seq_lens)
    else:
        append_to_non_paged_kv_cache_torch(kv_cache, this_kv, old_seq_lens)


def append_to_paged_kv_cache_torch(
    kv_cache,  # (num_pages, page_size, other contiguous dims...)
    page_table,  # (batch_size, num_pages_per_sample)
    this_kv,  # (batch_size, other contiguous dims...)
    old_seq_lens,  # (batch_size,)
):
    page_size = kv_cache.shape[1]
    for i in range(old_seq_lens.shape[0]):
        kv_cache[
            page_table[i, old_seq_lens[i] // page_size], old_seq_lens[i] % page_size
        ] = this_kv[i].clone()


def append_to_non_paged_kv_cache_torch(
    kv_cache,  # (batch_size, seq_len, other contiguous dims...)
    this_kv,  # (batch_size, other contiguous dims...)
    old_seq_lens,  # (batch_size,)
):
    for i in range(old_seq_lens.shape[0]):
        kv_cache[i, old_seq_lens[i]] = this_kv[i]
