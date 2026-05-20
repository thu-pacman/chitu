# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Callable
import torch

from chitu.device_type import has_accelerator
from chitu.ops.utils import make_op_dispatcher
from chitu.utils import try_import_platform_dep, try_import_and_setup_torch_npu
from chitu.global_vars import get_global_args

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
has_triton_impl = has_triton and has_accelerator()

if has_triton_impl:
    from chitu.ops.triton_ops import (
        append_to_paged_kv_cache_triton,
        append_to_dense_kv_cache_triton,
        fp8_e4m3fn_quant_per_tensor_triton,
        quant_pertoken_kvcache_dsa,  # only for DSV32 fp8 cache
        append_to_paged_kv_cache_blockfp8_deepgemm_triton,
        read_from_paged_indexer_kv_cache_deepgemm_triton,
    )


@make_op_dispatcher
def append_to_paged_kv_cache(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    this_kv: torch.Tensor,
    delta_position_ids: torch.Tensor,
    delta_seq_ids: Optional[torch.Tensor] = None,
    get_page_ids: Optional[Callable[[], torch.Tensor]] = None,
    get_offs_in_page: Optional[Callable[[], torch.Tensor]] = None,
    use_i64_offsets: bool = False,
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
    raise NotImplementedError


@append_to_paged_kv_cache.register_auto
def _auto_append_to_paged_kv_cache(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    this_kv: torch.Tensor,
    delta_position_ids: torch.Tensor,
    delta_seq_ids: Optional[torch.Tensor] = None,
    get_page_ids: Optional[Callable[[], torch.Tensor]] = None,
    get_offs_in_page: Optional[Callable[[], torch.Tensor]] = None,
    use_i64_offsets: bool = False,
):
    if has_triton_impl and get_global_args().infer.op_impl != "cpu":
        return "triton"
    return "torch"


@append_to_paged_kv_cache.register("torch")
def append_to_paged_kv_cache_torch(
    kv_cache: torch.Tensor,  # (num_pages, page_size, other contiguous dims...)
    page_table: torch.Tensor,  # (batch_size, num_pages_per_sample)
    this_kv: torch.Tensor,  # (num_tokens, other contiguous dims...)
    delta_position_ids: torch.Tensor,  # (num_tokens,)
    delta_seq_ids: Optional[torch.Tensor] = None,  # (num_tokens,)
    get_page_ids: Optional[Callable[[], torch.Tensor]] = None,  # fn -> (num_tokens,)
    get_offs_in_page: Optional[
        Callable[[], torch.Tensor]
    ] = None,  # fn -> (num_tokens,)
    use_i64_offsets: bool = False,
):
    # NOTE: PyTorch always use i64 offsets in its ops. So we can safely ignore use_i64_offsets

    page_size = kv_cache.shape[1]

    if get_page_ids is None:
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
        page_ids = page_table[delta_seq_ids, delta_position_ids // page_size]
    else:
        page_ids = get_page_ids()

    if get_offs_in_page is None:
        offs_in_page = delta_position_ids % page_size
    else:
        offs_in_page = get_offs_in_page()

    kv_cache[page_ids, offs_in_page] = this_kv.view(
        this_kv.shape[0], *kv_cache.shape[2:]
    )


append_to_paged_kv_cache.register_candidate("triton")
if has_triton_impl:
    append_to_paged_kv_cache.register("triton")(append_to_paged_kv_cache_triton)


@make_op_dispatcher
def append_to_paged_kv_cache_blockfp8_deepgemm(
    kv_cache: torch.Tensor,  # (num_pages, page_size, other contiguous dims...)
    page_table: torch.Tensor,  # (batch_size, num_pages_per_sample)
    k_fp8: torch.Tensor,  # (num_tokens, other contiguous dims...)
    k_scale: torch.Tensor,  # (num_tokens, other contiguous dims...)
    delta_position_ids: torch.Tensor,  # (num_tokens,)
    delta_seq_ids: Optional[torch.Tensor] = None,  # (num_tokens,)
    use_i64_offsets: bool = False,
    impl: str = "auto",
):
    raise NotImplementedError


@append_to_paged_kv_cache_blockfp8_deepgemm.register_auto
def _auto_append_to_paged_kv_cache_blockfp8_deepgemm(
    kv_cache: torch.Tensor,  # (num_pages, page_size, other contiguous dims...)
    page_table: torch.Tensor,  # (batch_size, num_pages_per_sample)
    k_fp8: torch.Tensor,  # (num_tokens, other contiguous dims...)
    k_scale: torch.Tensor,  # (num_tokens, other contiguous dims...)
    delta_position_ids: torch.Tensor,  # (num_tokens,)
    delta_seq_ids: Optional[torch.Tensor] = None,  # (num_tokens,)
    use_i64_offsets: bool = False,
):
    if has_triton_impl and get_global_args().infer.op_impl != "cpu":
        return "triton"
    return "torch"


@append_to_paged_kv_cache_blockfp8_deepgemm.register("torch")
def append_to_paged_kv_cache_blockfp8_deepgemm_torch(
    kv_cache: torch.Tensor,  # (num_pages, page_size, other contiguous dims...)
    page_table: torch.Tensor,  # (batch_size, num_pages_per_sample)
    k_fp8: torch.Tensor,  # (num_tokens, other contiguous dims...)
    k_scale: torch.Tensor,  # (num_tokens, other contiguous dims...)
    delta_position_ids: torch.Tensor,  # (num_tokens,)
    delta_seq_ids: Optional[torch.Tensor] = None,  # (num_tokens,)
    use_i64_offsets: bool = False,
):
    num_pages = kv_cache.shape[0]
    page_size = kv_cache.shape[1]
    kv_cache = kv_cache.view(num_pages, -1)
    assert kv_cache.shape[1] == page_size * 132

    scale_page_offs = page_size * 128
    k_fp8_paged = kv_cache[:, :scale_page_offs].view(num_pages, page_size, -1)
    k_scale_paged = kv_cache[:, scale_page_offs:].view(num_pages, page_size, -1)

    append_to_paged_kv_cache_torch(
        k_fp8_paged,
        page_table,
        k_fp8,
        delta_position_ids,
        delta_seq_ids,
    )
    append_to_paged_kv_cache_torch(
        k_scale_paged,
        page_table,
        k_scale.view(torch.float8_e4m3fn).view(k_scale.shape[0], -1),
        delta_position_ids,
        delta_seq_ids,
    )


append_to_paged_kv_cache_blockfp8_deepgemm.register_candidate("triton")
if has_triton_impl:
    append_to_paged_kv_cache_blockfp8_deepgemm.register("triton")(
        append_to_paged_kv_cache_blockfp8_deepgemm_triton
    )


@make_op_dispatcher
def update_singleton_paged_kv_cache(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    this_kv: torch.Tensor,
    mtp_size: int = 1,
    impl: str = "auto",
):
    """
    Update singleton paged K/V cache.

    Args:
        kv_cache: (num_pages, 1, other contiguous dims...). Data of the paged K/V cache.
        page_table: (batch_size, 1). Page table of the paged K/V cache.
        this_kv: (num_tokens, other contiguous dims...). New K/V value.
    """
    raise NotImplementedError


@update_singleton_paged_kv_cache.register_auto
def _auto_update_singleton_paged_kv_cache():
    return "torch"


@update_singleton_paged_kv_cache.register("torch")
def update_singleton_paged_kv_cache_torch(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    this_kv: torch.Tensor,
    mtp_size: int = 1,
):
    # Page size is always 1
    assert kv_cache.shape[1] == mtp_size
    assert page_table.shape[1] == 1

    kv_cache[page_table.squeeze(1)] = this_kv.view(
        this_kv.shape[0], mtp_size, *kv_cache.shape[2:]
    )


@make_op_dispatcher
def append_to_dense_kv_cache(
    kv_cache: torch.Tensor,
    this_kv: torch.Tensor,
    delta_position_ids: torch.Tensor,
    delta_seq_ids: Optional[torch.Tensor] = None,
    use_i64_offsets: bool = False,
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
    raise NotImplementedError


@append_to_dense_kv_cache.register_auto
def _auto_append_to_dense_kv_cache(
    kv_cache: torch.Tensor,
    this_kv: torch.Tensor,
    delta_position_ids: torch.Tensor,
    delta_seq_ids: Optional[torch.Tensor] = None,
    use_i64_offsets: bool = False,
):
    if has_triton_impl:
        return "triton"
    return "torch"


@append_to_dense_kv_cache.register("torch")
def append_to_dense_kv_cache_torch(
    kv_cache: torch.Tensor,  # (batch_size, seq_len, other contiguous dims...)
    this_kv: torch.Tensor,  # (num_tokens, other contiguous dims...)
    delta_position_ids: torch.Tensor,  # (num_tokens,)
    delta_seq_ids: Optional[torch.Tensor] = None,  # (num_tokens,)
    use_i64_offsets: bool = False,
):
    # NOTE: PyTorch always use i64 offsets in its ops. So we can safely ignore use_i64_offsets

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


@append_to_dense_kv_cache.register("torch_npu", available=has_torch_npu)
def append_to_dense_kv_cache_torch_npu(
    kv_cache: torch.Tensor,  # (batch_size, seq_len, other contiguous dims...)
    this_kv: torch.Tensor,  # (num_tokens, other contiguous dims...)
    delta_position_ids: torch.Tensor,  # (num_tokens,)
    delta_seq_ids: Optional[torch.Tensor] = None,  # (num_tokens,)
    use_i64_offsets: bool = False,
):
    # NOTE: We can safely igonre use_i64_offsets. This is guaranteed by test:
    # test_dense_offset_overflow_case_matches_torch_with_i64

    if delta_seq_ids is None and kv_cache.shape[0] != delta_position_ids.shape[0]:
        raise ValueError(
            f"batch_size ({kv_cache.shape[0]}) must be equal to num_tokens "
            f"({delta_position_ids.shape[0]}) if ignoring delta_seq_ids"
        )
    if this_kv.numel() == 0:
        return

    if this_kv.shape[0] == kv_cache.shape[0]:
        torch_npu.scatter_update_(kv_cache, delta_position_ids, this_kv.unsqueeze(1), 1)
    else:
        if delta_seq_ids is None:
            raise ValueError(
                f"batch_size ({kv_cache.shape[0]}) must be equal to num_tokens "
                f"({delta_position_ids.shape[0]}) if ignoring delta_seq_ids"
            )
        absolute_position_ids = delta_position_ids + delta_seq_ids * kv_cache.shape[1]
        indices = (
            absolute_position_ids
            - torch.arange(this_kv.shape[0], dtype=torch.int32, device=this_kv.device)
        ) * this_kv.shape[1]
        torch_npu.scatter_update_(
            kv_cache.view(-1, *kv_cache.shape[2:]), indices, this_kv, 1
        )


append_to_dense_kv_cache.register_candidate("triton")
if has_triton_impl:
    append_to_dense_kv_cache.register("triton")(append_to_dense_kv_cache_triton)


@make_op_dispatcher
def read_from_paged_kv_cache(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    position_ids: torch.Tensor,
    seq_ids: torch.Tensor,
    impl: str = "auto",
) -> torch.Tensor:
    """
    Read from paged K/V cache, write to ragged K/V.

    Args:
        kv_cache: (num_pages, page_size, other contiguous dims...). Data of the paged K/V cache.
        page_table: (batch_size, num_pages_per_sample). Page table of the paged K/V cache.
        position_ids: (num_tokens,). Position IDs of the incremented tokens. E.g, if
            reading the 0th, 1st token of the 1st sequence, and the 0th token of the 2nd
            sequence, position_ids = [0, 1, 0].
        seq_ids: (num_tokens,). Sequence IDs of the incremented tokens. E.g, if reading
            the 0th, 1st token of the 1st sequence, and the 0th token of the 2nd sequence,
            seq_ids = [1, 1, 2].
    """
    raise NotImplementedError


@read_from_paged_kv_cache.register_auto
def _auto_read_from_paged_kv_cache():
    return "torch"


@read_from_paged_kv_cache.register("torch")
def read_from_paged_kv_cache_torch(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    position_ids: torch.Tensor,
    seq_ids: torch.Tensor,
) -> torch.Tensor:
    return kv_cache[
        page_table[seq_ids, position_ids // kv_cache.shape[1]],
        position_ids % kv_cache.shape[1],
    ]


@make_op_dispatcher
def read_from_paged_indexer_kv_cache_deepgemm(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    position_ids: torch.Tensor,
    seq_ids: torch.Tensor,
    use_i64_offsets: bool = False,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError


@read_from_paged_indexer_kv_cache_deepgemm.register_auto
def _auto_read_from_paged_indexer_kv_cache_deepgemm(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    position_ids: torch.Tensor,
    seq_ids: torch.Tensor,
    use_i64_offsets: bool = False,
):
    if has_triton_impl and get_global_args().infer.op_impl != "cpu":
        return "triton"
    return "torch"


@read_from_paged_indexer_kv_cache_deepgemm.register("torch")
def read_from_paged_indexer_kv_cache_deepgemm_torch(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    position_ids: torch.Tensor,
    seq_ids: torch.Tensor,
    use_i64_offsets: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_pages = kv_cache.shape[0]
    page_size = kv_cache.shape[1]
    kv_cache = kv_cache.view(num_pages, -1)
    assert kv_cache.shape[1] == page_size * 132

    scale_page_offs = page_size * 128
    k_fp8_paged = kv_cache[:, :scale_page_offs].view(num_pages, page_size, -1)
    k_scale_paged = kv_cache[:, scale_page_offs:].view(num_pages, page_size, -1)

    k_fp8_ragged = read_from_paged_kv_cache_torch(
        k_fp8_paged,
        page_table,
        position_ids,
        seq_ids,
    )
    k_scale_ragged = read_from_paged_kv_cache_torch(
        k_scale_paged,
        page_table,
        position_ids,
        seq_ids,
    )

    return k_fp8_ragged.contiguous(), k_scale_ragged.contiguous().view(torch.float32)


read_from_paged_indexer_kv_cache_deepgemm.register_candidate("triton")
if has_triton_impl:
    read_from_paged_indexer_kv_cache_deepgemm.register("triton")(
        read_from_paged_indexer_kv_cache_deepgemm_triton
    )


@make_op_dispatcher
def read_from_singleton_paged_kv_cache(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    mtp_accept_indices: torch.Tensor | None = None,
    impl: str = "auto",
) -> torch.Tensor:
    """
    Read from singleton paged K/V cache.

    Args:
        kv_cache: (num_pages, page_size, other contiguous dims...). Data of the paged K/V cache.
        page_table: (batch_size, num_pages_per_sample). Page table of the paged K/V cache.
    """
    raise NotImplementedError


@read_from_singleton_paged_kv_cache.register_auto
def _auto_read_from_singleton_paged_kv_cache():
    return "torch"


@read_from_singleton_paged_kv_cache.register("torch")
def read_from_singleton_paged_kv_cache_torch(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    mtp_accept_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if mtp_accept_indices is None:
        return kv_cache[page_table.squeeze(1), 0]
    else:
        return kv_cache[page_table.squeeze(1), mtp_accept_indices]


@make_op_dispatcher
def read_from_dense_kv_cache(
    kv_cache: torch.Tensor,
    position_ids: torch.Tensor,
    seq_ids: torch.Tensor,
    impl: str = "auto",
) -> torch.Tensor:
    """
    Read from dense K/V cache, write to ragged K/V.

    Args:
        kv_cache: (batch_size, seq_len, other contiguous dims...). Dense K/V cache.
        position_ids: (num_tokens,). Position IDs of the incremented tokens. E.g, if
            reading the 0th, 1st token of the 1st sequence, and the 0th token of the 2nd
            sequence, position_ids = [0, 1, 0].
        seq_ids: (num_tokens,). Sequence IDs of the incremented tokens. E.g, if reading
            the 0th, 1st token of the 1st sequence, and the 0th token of the 2nd sequence,
            seq_ids = [1, 1, 2].

    Returns:
        (num_tokens, other contiguous dims...). Ragged K/V.
    """
    raise NotImplementedError


@read_from_dense_kv_cache.register_auto
def _auto_read_from_dense_kv_cache():
    return "torch"


@read_from_dense_kv_cache.register("torch")
def read_from_dense_kv_cache_torch(
    kv_cache: torch.Tensor, position_ids: torch.Tensor, seq_ids: torch.Tensor
) -> torch.Tensor:
    return kv_cache[seq_ids, position_ids]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def fp8_pertensor_kvcache_quant(xq, xk, xv, k_scale, v_scale, batch_size, head_num):
    q_scale = (xq.abs().amax() / 448).to(torch.float32)
    xq = fp8_e4m3fn_quant_per_tensor_triton(xq, q_scale)
    xk = fp8_e4m3fn_quant_per_tensor_triton(xk, k_scale)
    xv = fp8_e4m3fn_quant_per_tensor_triton(xv, v_scale)
    descales = {
        "q_descale": q_scale.view(1, 1).expand(batch_size, head_num),
        "k_descale": k_scale.view(1, 1).expand(batch_size, head_num),
        "v_descale": v_scale.view(1, 1).expand(batch_size, head_num),
    }
    return xq, xk, xv, descales


def fp8_pertoken_kvcache_quant_dsa(kv_lora_k_pe, kv_lora_rank):
    return quant_pertoken_kvcache_dsa(kv_lora_k_pe, kv_lora_rank)
