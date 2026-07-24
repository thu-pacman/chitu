# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Torch fallback ops for the DeepSeek-V4 HCA/CSA compressor.

The functions in this module intentionally mirror the Triton wrapper signatures
in ``chitu.ops.triton_ops.deepseek_compressor``. Model code can therefore share
one flat/ragged prefill flow and select only the backend implementation.
"""

import itertools
from typing import Optional

import torch

from chitu.ops.deepseek_v4_decode_plan import (
    DeepSeekV4DecodeCompressPlan,
    build_decode_compress_plan,
)
from chitu.ops.utils import make_op_dispatcher
from chitu.utils import try_import_platform_dep


def _to_int_list(tensor: torch.Tensor) -> list[int]:
    return [int(v) for v in tensor.detach().cpu().tolist()]


def build_compress_metadata(
    start_positions: list[int],
    seqlens: list[int],
    ratio: int,
    device: torch.device,
) -> dict:
    """Precompute ragged compressor metadata on CPU and return tensor fields on device."""
    # FIXME: Metadata tensors are currently int32 to match the Triton compressor
    # wrappers. The torch fallback converts them back to Python ints for slicing,
    # but the shared metadata construction itself still needs an int64 path when
    # flat ragged offsets can exceed the int32-safe range.
    pending_lens = [sp % ratio for sp in start_positions]
    effective_lens = [p + s for p, s in zip(pending_lens, seqlens)]
    remainders = [e % ratio for e in effective_lens]
    cutoffs = [e - r for e, r in zip(effective_lens, remainders)]
    n_groups_list = [c // ratio for c in cutoffs]

    cu_eff = list(itertools.accumulate([0] + effective_lens))
    cu_new = list(itertools.accumulate([0] + seqlens))
    cu_groups = list(itertools.accumulate([0] + n_groups_list))

    group_to_req = []
    for i, ng in enumerate(n_groups_list):
        group_to_req.extend([i] * ng)

    start_pos_div_ratio = [sp // ratio for sp in start_positions]

    def t32(lst: list[int]) -> torch.Tensor:
        return torch.tensor(lst, dtype=torch.int32, device=device)

    return dict(
        pending_lens=t32(pending_lens),
        effective_lens=effective_lens,
        remainders=t32(remainders),
        cutoffs=t32(cutoffs),
        n_groups_list=n_groups_list,
        cu_eff=t32(cu_eff),
        cu_new=t32(cu_new),
        cu_groups=t32(cu_groups),
        group_to_req=(
            t32(group_to_req)
            if group_to_req
            else torch.zeros(0, dtype=torch.int32, device=device)
        ),
        start_pos_div_ratio=t32(start_pos_div_ratio),
        total_groups=cu_groups[-1],
        total_eff=cu_eff[-1],
    )


def build_compress_metadata_from_tensors(
    start_positions: torch.Tensor,
    seqlens: torch.Tensor,
    ratio: int,
    device: torch.device,
) -> dict:
    """Build ragged compressor metadata with tensor ops.

    Shape fields stay int32 to match the Triton compressor wrappers.
    ``total_eff`` and ``total_groups`` are Python ints because callers use them
    for tensor allocation and Triton grid sizes.
    """
    start_positions = start_positions.to(device=device, dtype=torch.long)
    seqlens = seqlens.to(device=device, dtype=torch.long)
    n = int(seqlens.numel())

    pending_lens_l = start_positions.remainder(ratio)
    effective_lens_l = pending_lens_l + seqlens
    remainders_l = effective_lens_l.remainder(ratio)
    cutoffs_l = effective_lens_l - remainders_l
    n_groups_l = torch.div(cutoffs_l, ratio, rounding_mode="floor")

    cu_eff_l = torch.empty(n + 1, dtype=torch.long, device=device)
    cu_new_l = torch.empty(n + 1, dtype=torch.long, device=device)
    cu_groups_l = torch.empty(n + 1, dtype=torch.long, device=device)
    cu_eff_l[0] = 0
    cu_new_l[0] = 0
    cu_groups_l[0] = 0
    if n > 0:
        cu_eff_l[1:] = torch.cumsum(effective_lens_l, dim=0)
        cu_new_l[1:] = torch.cumsum(seqlens, dim=0)
        cu_groups_l[1:] = torch.cumsum(n_groups_l, dim=0)

    total_eff = int(cu_eff_l[-1].item()) if n > 0 else 0
    total_groups = int(cu_groups_l[-1].item()) if n > 0 else 0
    if total_groups > 0:
        group_to_req = torch.repeat_interleave(
            torch.arange(n, dtype=torch.int32, device=device),
            n_groups_l.to(torch.long),
            output_size=total_groups,
        )
    else:
        group_to_req = torch.zeros(0, dtype=torch.int32, device=device)

    return dict(
        pending_lens=pending_lens_l.to(torch.int32),
        effective_lens=effective_lens_l,
        remainders=remainders_l.to(torch.int32),
        cutoffs=cutoffs_l.to(torch.int32),
        n_groups_list=n_groups_l.to(torch.int32),
        cu_eff=cu_eff_l.to(torch.int32),
        cu_new=cu_new_l.to(torch.int32),
        cu_groups=cu_groups_l.to(torch.int32),
        group_to_req=group_to_req,
        start_pos_div_ratio=torch.div(start_positions, ratio, rounding_mode="floor").to(
            torch.int32
        ),
        total_groups=total_groups,
        total_eff=total_eff,
    )


def gather_pending_and_new(
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    kv_cat: torch.Tensor,
    score_cat: torch.Tensor,
    ape: torch.Tensor,
    flat_kv: torch.Tensor,
    flat_score: torch.Tensor,
    cu_eff: torch.Tensor,
    cu_new: torch.Tensor,
    pending_lens: torch.Tensor,
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    ratio: int,
    pending_offset: int = 0,
    max_eff: int | None = None,
) -> None:
    """Gather each request's pending state and new tokens into flat ragged buffers."""
    del pending_offset, max_eff
    n = int(pending_lens.numel())
    if n == 0 or flat_kv.numel() == 0:
        return

    cu_eff_list = _to_int_list(cu_eff)
    cu_new_list = _to_int_list(cu_new)
    pending_lens_list = _to_int_list(pending_lens)
    start_positions_list = _to_int_list(start_positions)
    cache_slots_list = _to_int_list(cache_slots)
    state_rows = kv_state.shape[1]

    for i in range(n):
        eff_start = cu_eff_list[i]
        new_start = cu_new_list[i]
        new_len = cu_new_list[i + 1] - new_start
        pending = pending_lens_list[i]
        abs_start = start_positions_list[i] - pending
        slot = cache_slots_list[i]

        if pending > 0:
            dst = slice(eff_start, eff_start + pending)
            positions = torch.arange(
                abs_start,
                abs_start + pending,
                device=kv_state.device,
                dtype=torch.long,
            )
            rows = positions.remainder(state_rows)
            flat_kv[dst] = kv_state[slot, rows]
            flat_score[dst] = score_state[slot, rows] - ape[positions.remainder(ratio)]

        if new_len > 0:
            src = slice(new_start, new_start + new_len)
            dst = slice(eff_start + pending, eff_start + pending + new_len)
            flat_kv[dst] = kv_cat[src]
            flat_score[dst] = score_cat[src]


def compress_hca(
    flat_kv: torch.Tensor,
    flat_score: torch.Tensor,
    ape: torch.Tensor,
    out_kv: torch.Tensor,
    cu_eff: torch.Tensor,
    cu_groups: torch.Tensor,
    group_to_req: torch.Tensor,
    ratio: int,
) -> None:
    """Compress HCA groups from flat ragged buffers into ``out_kv``."""
    del group_to_req
    total_groups = out_kv.shape[0]
    if total_groups == 0:
        return

    cu_eff_list = _to_int_list(cu_eff)
    cu_groups_list = _to_int_list(cu_groups)
    D = flat_kv.shape[1]

    for req in range(len(cu_groups_list) - 1):
        group_start = cu_groups_list[req]
        group_end = cu_groups_list[req + 1]
        n_groups = group_end - group_start
        if n_groups == 0:
            continue

        eff_start = cu_eff_list[req]
        cutoff = n_groups * ratio
        kv = flat_kv[eff_start : eff_start + cutoff].reshape(n_groups, ratio, D)
        score = flat_score[eff_start : eff_start + cutoff].reshape(n_groups, ratio, D)
        score = score + ape
        out_kv[group_start:group_end] = (kv * score.softmax(dim=1)).sum(dim=1)


def compress_csa(
    flat_kv: torch.Tensor,
    flat_score: torch.Tensor,
    ape: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    out_kv: torch.Tensor,
    cu_eff: torch.Tensor,
    cu_groups: torch.Tensor,
    group_to_req: torch.Tensor,
    pending_lens: torch.Tensor,
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    n_groups_list_t: torch.Tensor,
    ratio: int,
    head_dim: int,
) -> None:
    """Compress CSA groups and update the per-slot prev-group state in-place.

    CSA group 0 overlaps with the previous complete group's first head. The
    pending state is a logical-position ring, so that reference is loaded from
    the previous group's absolute positions instead of a mutable compact prefix.
    """
    del group_to_req
    n = int(pending_lens.numel())
    if n == 0:
        return

    cu_eff_list = _to_int_list(cu_eff)
    cu_groups_list = _to_int_list(cu_groups)
    pending_lens_list = _to_int_list(pending_lens)
    start_positions_list = _to_int_list(start_positions)
    cache_slots_list = _to_int_list(cache_slots)
    n_groups_list = _to_int_list(n_groups_list_t)

    device = flat_kv.device
    dtype = flat_kv.dtype
    full_d = flat_kv.shape[1]
    state_rows = kv_state.shape[1]

    if out_kv.shape[0] > 0:
        for req, n_groups in enumerate(n_groups_list):
            if n_groups == 0:
                continue

            eff_start = cu_eff_list[req]
            group_start = cu_groups_list[req]
            group_end = cu_groups_list[req + 1]
            cutoff = n_groups * ratio
            kv = flat_kv[eff_start : eff_start + cutoff].reshape(
                n_groups, ratio, full_d
            )
            score = flat_score[eff_start : eff_start + cutoff].reshape(
                n_groups, ratio, full_d
            )

            ape_prev = ape[:, :head_dim]
            ape_cur = ape[:, head_dim:]
            cur_kv = kv[:, :, head_dim:]
            cur_score = score[:, :, head_dim:] + ape_cur

            prev_kv = torch.zeros(n_groups, ratio, head_dim, device=device, dtype=dtype)
            prev_score = torch.full_like(prev_kv, float("-inf"))
            group_abs_start = start_positions_list[req] - pending_lens_list[req]
            prev_abs_start = group_abs_start - ratio
            if prev_abs_start >= 0:
                rows = torch.arange(
                    prev_abs_start,
                    prev_abs_start + ratio,
                    device=device,
                    dtype=torch.long,
                ).remainder(state_rows)
                slot = cache_slots_list[req]
                prev_kv[0] = kv_state[slot, rows, :head_dim]
                prev_score[0] = score_state[slot, rows, :head_dim]
            if n_groups > 1:
                prev_kv[1:] = kv[:-1, :, :head_dim]
                prev_score[1:] = score[:-1, :, :head_dim] + ape_prev

            kv_all = torch.cat([prev_kv, cur_kv], dim=1)
            score_all = torch.cat([prev_score, cur_score], dim=1)
            out_kv[group_start:group_end] = (kv_all * score_all.softmax(dim=1)).sum(
                dim=1
            )


def writeback_pending(
    flat_kv: torch.Tensor,
    flat_score: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    ape: torch.Tensor,
    cu_eff: torch.Tensor,
    pending_lens: torch.Tensor,
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    pending_offset: int,
    ratio: int,
) -> None:
    """Write the live suffix back to the logical-position pending-state ring."""
    del pending_offset
    n = int(pending_lens.numel())
    if n == 0:
        return

    cu_eff_list = _to_int_list(cu_eff)
    pending_lens_list = _to_int_list(pending_lens)
    start_positions_list = _to_int_list(start_positions)
    cache_slots_list = _to_int_list(cache_slots)
    state_rows = kv_state.shape[1]

    for req in range(n):
        eff_start = cu_eff_list[req]
        eff_len = cu_eff_list[req + 1] - eff_start
        if eff_len == 0:
            continue
        slot = cache_slots_list[req]
        suffix_len = min(eff_len, state_rows)
        src_start = eff_start + eff_len - suffix_len
        abs_start = start_positions_list[req] - pending_lens_list[req]
        positions = torch.arange(
            abs_start + eff_len - suffix_len,
            abs_start + eff_len,
            device=kv_state.device,
            dtype=torch.long,
        )
        rows = positions.remainder(state_rows)
        src = slice(src_start, src_start + suffix_len)
        kv_state[slot, rows] = flat_kv[src]
        score_state[slot, rows] = flat_score[src] + ape[positions.remainder(ratio)]


_, _has_triton = try_import_platform_dep("triton")
if _has_triton:
    from chitu.ops.triton_ops.deepseek_compressor import (
        decode_hca as _decode_hca_triton,
        decode_mtp_hca as _decode_mtp_hca_triton,
        decode_mtp_csa as _decode_mtp_csa_triton,
        postprocess_write_flashmla as _postprocess_write_flashmla_triton,
        postprocess_write_kv_cache as _postprocess_write_kv_cache_triton,
        postprocess_write_kv_cache_hadamard as _postprocess_write_kv_cache_hadamard_triton,
    )

_HAS_TRITON_DECODE = _has_triton
_HAS_TRITON_FLASHMLA_POSTPROCESS = _has_triton
_HAS_TRITON_KV_CACHE_POSTPROCESS = _has_triton


def _ensure_decode_compress_plan(
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    *,
    ratio: int,
    q_len: int,
    head_dim: int,
    is_csa: bool,
    use_cuda_graph: bool,
    plan: Optional[DeepSeekV4DecodeCompressPlan] = None,
) -> DeepSeekV4DecodeCompressPlan:
    if plan is not None:
        return plan
    return build_decode_compress_plan(
        start_positions,
        cache_slots,
        ratio=ratio,
        q_len=q_len,
        head_dim=head_dim,
        is_csa=is_csa,
        use_cuda_graph=use_cuda_graph,
    )


def _select_decode_compress_rows(
    plan: DeepSeekV4DecodeCompressPlan,
    *,
    is_csa: bool,
    use_cuda_graph: bool,
    request_rows: Optional[torch.Tensor] = None,
    row_is_valid: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if request_rows is not None:
        return request_rows, row_is_valid
    if use_cuda_graph or is_csa:
        return plan.full_request_rows, plan.full_row_is_valid
    return plan.compact_request_rows, None


def supports_decode_fastpath(
    kv: torch.Tensor,
    *,
    q_len: int,
    ratio: int,
) -> bool:
    return q_len <= ratio and _HAS_TRITON_DECODE and kv.is_cuda


@make_op_dispatcher
def decode_compressor(
    kv: torch.Tensor,
    score: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    ape: torch.Tensor,
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    *,
    ratio: int,
    q_len: int,
    head_dim: int,
    is_csa: bool,
    use_cuda_graph: bool,
    plan: Optional[DeepSeekV4DecodeCompressPlan] = None,
    request_rows: Optional[torch.Tensor] = None,
    row_is_valid: Optional[torch.Tensor] = None,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    raise NotImplementedError


@decode_compressor.register_auto
def _auto_decode_compressor(
    kv: torch.Tensor,
    score: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    ape: torch.Tensor,
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    *,
    ratio: int,
    q_len: int,
    head_dim: int,
    is_csa: bool,
    use_cuda_graph: bool,
    plan: Optional[DeepSeekV4DecodeCompressPlan] = None,
    request_rows: Optional[torch.Tensor] = None,
    row_is_valid: Optional[torch.Tensor] = None,
):
    del plan, request_rows, row_is_valid
    if _HAS_TRITON_DECODE and kv.is_cuda:
        return "triton"
    if q_len == 1:
        return "torch"
    raise RuntimeError(
        "DeepSeek-V4 compressor decode dispatch does not support this shape"
    )


@decode_compressor.register("triton", available=_HAS_TRITON_DECODE)
def _decode_compressor_triton(
    kv: torch.Tensor,
    score: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    ape: torch.Tensor,
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    *,
    ratio: int,
    q_len: int,
    head_dim: int,
    is_csa: bool,
    use_cuda_graph: bool,
    plan: Optional[DeepSeekV4DecodeCompressPlan] = None,
    request_rows: Optional[torch.Tensor] = None,
    row_is_valid: Optional[torch.Tensor] = None,
):
    plan = _ensure_decode_compress_plan(
        start_positions,
        cache_slots,
        ratio=ratio,
        q_len=q_len,
        head_dim=head_dim,
        is_csa=is_csa,
        use_cuda_graph=use_cuda_graph,
        plan=plan,
    )
    compressed_idx, valid_rows = _select_decode_compress_rows(
        plan,
        is_csa=is_csa,
        use_cuda_graph=use_cuda_graph,
        request_rows=request_rows,
        row_is_valid=row_is_valid,
    )
    compressed_mask = plan.should_compress
    mask_required = valid_rows is not None
    cache_slots_i32 = cache_slots.to(device=kv.device, dtype=torch.int32)
    if q_len == 1 and not is_csa:
        out_kv = torch.empty(
            start_positions.numel(), head_dim, device=kv.device, dtype=torch.float32
        )
        _decode_hca_triton(
            kv,
            score,
            kv_state,
            score_state,
            ape,
            out_kv,
            start_positions,
            cache_slots_i32,
            ratio,
        )
        if not mask_required:
            out_kv = out_kv.index_select(0, compressed_idx)
        return out_kv, compressed_mask, compressed_idx, mask_required

    out_rows = start_positions.numel() if mask_required else int(compressed_idx.numel())
    out_kv = torch.empty(out_rows, head_dim, device=kv.device, dtype=torch.float32)
    if is_csa:
        kv = kv.reshape(start_positions.numel() * q_len, kv.shape[-1])
        score = score.reshape(start_positions.numel() * q_len, score.shape[-1])
        _decode_mtp_csa_triton(
            kv,
            score,
            kv_state,
            score_state,
            ape,
            out_kv,
            start_positions,
            cache_slots_i32,
            ratio,
            q_len,
            head_dim,
        )
    else:
        _decode_mtp_hca_triton(
            kv,
            score,
            kv_state,
            score_state,
            ape,
            out_kv,
            compressed_idx,
            start_positions,
            cache_slots_i32,
            ratio,
            q_len,
        )
    return out_kv, compressed_mask, compressed_idx, mask_required


@decode_compressor.register("torch")
def _decode_compressor_torch(
    kv: torch.Tensor,
    score: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    ape: torch.Tensor,
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    *,
    ratio: int,
    q_len: int,
    head_dim: int,
    is_csa: bool,
    use_cuda_graph: bool,
    plan: Optional[DeepSeekV4DecodeCompressPlan] = None,
    request_rows: Optional[torch.Tensor] = None,
    row_is_valid: Optional[torch.Tensor] = None,
):
    if q_len != 1:
        raise RuntimeError("Torch decode fallback only supports q_len == 1")
    score = score + ape[start_positions % ratio].unsqueeze(1)
    state_rows = kv_state.size(1)
    write_pos = start_positions % state_rows
    kv_state[cache_slots, write_pos] = kv.squeeze(1)
    score_state[cache_slots, write_pos] = score.squeeze(1)

    if is_csa:
        gather_positions = (
            start_positions[:, None]
            - (2 * ratio - 1)
            + torch.arange(2 * ratio, device=kv.device, dtype=torch.long)
        )
        gather_rows = gather_positions.remainder(state_rows)
        gather_mask = gather_positions >= 0
        kv_prev = kv_state[cache_slots[:, None], gather_rows[:, :ratio], :head_dim]
        kv_cur = kv_state[cache_slots[:, None], gather_rows[:, ratio:], head_dim:]
        score_prev = score_state[
            cache_slots[:, None], gather_rows[:, :ratio], :head_dim
        ]
        score_cur = score_state[cache_slots[:, None], gather_rows[:, ratio:], head_dim:]
        kv_src = torch.cat([kv_prev, kv_cur], dim=1)
        score_src = torch.cat([score_prev, score_cur], dim=1)
        score_src = score_src.masked_fill(~gather_mask.unsqueeze(-1), float("-inf"))
    else:
        gather_positions = (
            start_positions[:, None]
            - (ratio - 1)
            + torch.arange(ratio, device=kv.device, dtype=torch.long)
        )
        gather_rows = gather_positions.remainder(state_rows)
        gather_mask = gather_positions >= 0
        kv_src = kv_state[cache_slots[:, None], gather_rows]
        score_src = score_state[cache_slots[:, None], gather_rows].masked_fill(
            ~gather_mask.unsqueeze(-1), float("-inf")
        )
    out_kv = (kv_src * score_src.softmax(dim=1)).sum(dim=1)
    plan = _ensure_decode_compress_plan(
        start_positions,
        cache_slots,
        ratio=ratio,
        q_len=q_len,
        head_dim=head_dim,
        is_csa=is_csa,
        use_cuda_graph=use_cuda_graph,
        plan=plan,
    )
    compressed_idx, valid_rows = _select_decode_compress_rows(
        plan,
        is_csa=is_csa,
        use_cuda_graph=use_cuda_graph,
        request_rows=request_rows,
        row_is_valid=row_is_valid,
    )
    compressed_mask = plan.should_compress
    mask_required = valid_rows is not None
    out_kv = out_kv.index_select(0, compressed_idx)
    if valid_rows is not None:
        out_kv = torch.where(valid_rows.unsqueeze(-1), out_kv, torch.zeros_like(out_kv))
    return out_kv, compressed_mask, compressed_idx, mask_required


def _is_flashmla_packed_cache(kv_cache: Optional[torch.Tensor]) -> bool:
    return (
        kv_cache is not None
        and kv_cache.dtype == torch.uint8
        and kv_cache.ndim >= 3
        and kv_cache.shape[-1] == 584
    )


@make_op_dispatcher
def try_fused_decode_postprocess_write_cache(
    kv_compress: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    kv_cache: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    cache_seq_ids: torch.Tensor,
    *,
    ratio: int,
    norm_eps: float,
    q_len: int = 1,
    rope_dim: int = 64,
    use_cuda_graph: bool = False,
    use_hadamard: bool = False,
    kv_cache_is_paged: bool = False,
    impl: str = "auto",
) -> bool:
    raise NotImplementedError


@try_fused_decode_postprocess_write_cache.register_auto
def _auto_try_fused_decode_postprocess_write_cache(
    kv_compress: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    kv_cache: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    cache_seq_ids: torch.Tensor,
    *,
    ratio: int,
    norm_eps: float,
    q_len: int = 1,
    rope_dim: int = 64,
    use_cuda_graph: bool = False,
    use_hadamard: bool = False,
    kv_cache_is_paged: bool = False,
):
    if not (
        use_cuda_graph
        and kv_cache is not None
        and kv_cache.is_cuda
        and kv_compress.is_cuda
    ):
        return "skip"

    if (
        _HAS_TRITON_FLASHMLA_POSTPROCESS
        and not use_hadamard
        and kv_cache_is_paged
        and block_table is not None
        and _is_flashmla_packed_cache(kv_cache)
    ):
        return "flashmla"

    if (
        _HAS_TRITON_KV_CACHE_POSTPROCESS
        and use_hadamard
        and kv_compress.shape[-1] == 128
        and kv_cache.ndim == 3
        and not _is_flashmla_packed_cache(kv_cache)
        and kv_cache.shape[-1] == kv_compress.shape[-1]
        and (not kv_cache_is_paged or block_table is not None)
    ):
        return "kv_cache_hadamard"

    if (
        _HAS_TRITON_KV_CACHE_POSTPROCESS
        and not use_hadamard
        and kv_cache.ndim == 3
        and not _is_flashmla_packed_cache(kv_cache)
        and kv_cache.shape[-1] == kv_compress.shape[-1]
        and (not kv_cache_is_paged or block_table is not None)
    ):
        return "kv_cache"
    return "skip"


@try_fused_decode_postprocess_write_cache.register("skip")
def _try_fused_decode_postprocess_write_cache_skip(
    kv_compress: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    kv_cache: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    cache_seq_ids: torch.Tensor,
    *,
    ratio: int,
    norm_eps: float,
    q_len: int = 1,
    rope_dim: int = 64,
    use_cuda_graph: bool = False,
    use_hadamard: bool = False,
    kv_cache_is_paged: bool = False,
) -> bool:
    return False


@try_fused_decode_postprocess_write_cache.register(
    "flashmla", available=_HAS_TRITON_FLASHMLA_POSTPROCESS
)
def _try_fused_decode_postprocess_write_cache_flashmla_impl(
    kv_compress: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    kv_cache: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    cache_seq_ids: torch.Tensor,
    *,
    ratio: int,
    norm_eps: float,
    q_len: int = 1,
    rope_dim: int = 64,
    use_cuda_graph: bool = False,
    use_hadamard: bool = False,
    kv_cache_is_paged: bool = False,
) -> bool:
    assert kv_cache is not None and block_table is not None
    _postprocess_write_flashmla_triton(
        kv_compress,
        norm_weight,
        freqs_cis,
        kv_cache,
        block_table,
        start_positions,
        cache_seq_ids,
        ratio,
        norm_eps,
        q_len=q_len,
    )
    return True


def _call_postprocess_kv_cache_writer(
    writer,
    kv_compress: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    kv_cache: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    cache_seq_ids: torch.Tensor,
    *,
    ratio: int,
    norm_eps: float,
    q_len: int,
    rope_dim: int,
    kv_cache_is_paged: bool,
) -> bool:
    assert kv_cache is not None
    writer(
        kv_compress,
        norm_weight,
        freqs_cis,
        kv_cache,
        block_table if kv_cache_is_paged else None,
        start_positions,
        cache_slots,
        cache_seq_ids,
        ratio,
        norm_eps,
        q_len=q_len,
        rope_dim=rope_dim,
    )
    return True


@try_fused_decode_postprocess_write_cache.register(
    "kv_cache", available=_HAS_TRITON_KV_CACHE_POSTPROCESS
)
def _try_fused_decode_postprocess_write_cache_kv_impl(
    kv_compress: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    kv_cache: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    cache_seq_ids: torch.Tensor,
    *,
    ratio: int,
    norm_eps: float,
    q_len: int = 1,
    rope_dim: int = 64,
    use_cuda_graph: bool = False,
    use_hadamard: bool = False,
    kv_cache_is_paged: bool = False,
) -> bool:
    return _call_postprocess_kv_cache_writer(
        _postprocess_write_kv_cache_triton,
        kv_compress,
        norm_weight,
        freqs_cis,
        kv_cache,
        block_table,
        start_positions,
        cache_slots,
        cache_seq_ids,
        ratio=ratio,
        norm_eps=norm_eps,
        q_len=q_len,
        rope_dim=rope_dim,
        kv_cache_is_paged=kv_cache_is_paged,
    )


@try_fused_decode_postprocess_write_cache.register(
    "kv_cache_hadamard", available=_HAS_TRITON_KV_CACHE_POSTPROCESS
)
def _try_fused_decode_postprocess_write_cache_kv_hadamard_impl(
    kv_compress: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    kv_cache: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    cache_seq_ids: torch.Tensor,
    *,
    ratio: int,
    norm_eps: float,
    q_len: int = 1,
    rope_dim: int = 64,
    use_cuda_graph: bool = False,
    use_hadamard: bool = False,
    kv_cache_is_paged: bool = False,
) -> bool:
    return _call_postprocess_kv_cache_writer(
        _postprocess_write_kv_cache_hadamard_triton,
        kv_compress,
        norm_weight,
        freqs_cis,
        kv_cache,
        block_table,
        start_positions,
        cache_slots,
        cache_seq_ids,
        ratio=ratio,
        norm_eps=norm_eps,
        q_len=q_len,
        rope_dim=rope_dim,
        kv_cache_is_paged=kv_cache_is_paged,
    )
