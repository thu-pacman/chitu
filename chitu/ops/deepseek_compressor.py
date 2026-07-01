# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Torch fallback ops for the DeepSeek-V4 HCA/CSA compressor.

The functions in this module intentionally mirror the Triton wrapper signatures
in ``chitu.ops.triton_ops.deepseek_compressor``. Model code can therefore share
one flat/ragged prefill flow and select only the backend implementation.
"""

import itertools

import torch


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
