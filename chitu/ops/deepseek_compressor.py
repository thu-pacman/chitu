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
    cache_slots: torch.Tensor,
    ratio: int,
    pending_offset: int = 0,
) -> None:
    """Gather each request's pending state and new tokens into flat ragged buffers."""
    del ratio
    n = int(pending_lens.numel())
    if n == 0 or flat_kv.numel() == 0:
        return

    cu_eff_list = _to_int_list(cu_eff)
    cu_new_list = _to_int_list(cu_new)
    pending_lens_list = _to_int_list(pending_lens)
    cache_slots_list = _to_int_list(cache_slots)

    for i in range(n):
        eff_start = cu_eff_list[i]
        new_start = cu_new_list[i]
        new_len = cu_new_list[i + 1] - new_start
        pending = pending_lens_list[i]
        slot = cache_slots_list[i]

        if pending > 0:
            src = slice(pending_offset, pending_offset + pending)
            dst = slice(eff_start, eff_start + pending)
            flat_kv[dst] = kv_state[slot, src]
            flat_score[dst] = score_state[slot, src] - ape[:pending]

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
    cache_slots: torch.Tensor,
    cutoffs: torch.Tensor,
    n_groups_list_t: torch.Tensor,
    ratio: int,
    head_dim: int,
) -> None:
    """Compress CSA groups and update the per-slot prev-group state in-place.

    CSA group 0 overlaps with the previous complete group's first head. That
    reference lives in ``kv_state[slot, :ratio, :head_dim]`` and
    ``score_state[slot, :ratio, :head_dim]``. This function snapshots that
    reference before compression and, after producing ``out_kv``, writes the
    last complete group's first head back to the same state area for the next
    prefill/decode chunk. This mirrors the Triton wrapper's state update.
    """
    del group_to_req
    n = int(pending_lens.numel())
    if n == 0:
        return

    cu_eff_list = _to_int_list(cu_eff)
    cu_groups_list = _to_int_list(cu_groups)
    pending_lens_list = _to_int_list(pending_lens)
    cache_slots_list = _to_int_list(cache_slots)
    cutoffs_list = _to_int_list(cutoffs)
    n_groups_list = _to_int_list(n_groups_list_t)

    device = flat_kv.device
    dtype = flat_kv.dtype
    full_d = flat_kv.shape[1]
    prev_ref_kv = torch.zeros(n, ratio, head_dim, device=device, dtype=dtype)
    prev_ref_score = torch.full(
        (n, ratio, head_dim), float("-inf"), device=device, dtype=dtype
    )

    for req in range(n):
        if pending_lens_list[req] == 0:
            continue
        slot = cache_slots_list[req]
        prev_ref_kv[req] = kv_state[slot, :ratio, :head_dim]
        prev_ref_score[req] = score_state[slot, :ratio, :head_dim] - ape

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

            cur_kv = kv[:, :, head_dim:]
            cur_score = score[:, :, head_dim:] + ape

            prev_kv = torch.empty(n_groups, ratio, head_dim, device=device, dtype=dtype)
            prev_score = torch.empty_like(prev_kv)
            prev_kv[0] = prev_ref_kv[req]
            prev_score[0] = prev_ref_score[req] + ape
            if n_groups > 1:
                prev_kv[1:] = kv[:-1, :, :head_dim]
                prev_score[1:] = score[:-1, :, :head_dim] + ape

            kv_all = torch.cat([prev_kv, cur_kv], dim=1)
            score_all = torch.cat([prev_score, cur_score], dim=1)
            out_kv[group_start:group_end] = (kv_all * score_all.softmax(dim=1)).sum(
                dim=1
            )

    # Keep this state update inside compress_csa: the external compressor flow
    # treats HCA/CSA compression as one backend op, while CSA still needs this
    # prev-group reference for the next chunk's group-0 overlap.
    for req, n_groups in enumerate(n_groups_list):
        if n_groups == 0:
            continue
        slot = cache_slots_list[req]
        eff_start = cu_eff_list[req]
        last_group_start = eff_start + cutoffs_list[req] - ratio
        src = slice(last_group_start, last_group_start + ratio)
        kv_state[slot, :ratio, :head_dim] = flat_kv[src, :head_dim]
        score_state[slot, :ratio, :head_dim] = flat_score[src, :head_dim] + ape


def writeback_pending(
    flat_kv: torch.Tensor,
    flat_score: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    ape: torch.Tensor,
    cu_eff: torch.Tensor,
    cutoffs: torch.Tensor,
    remainders: torch.Tensor,
    cache_slots: torch.Tensor,
    pending_offset: int,
    ratio: int,
) -> None:
    """Write each request's incomplete trailing group back to pending state."""
    del ratio
    n = int(remainders.numel())
    if n == 0:
        return

    cu_eff_list = _to_int_list(cu_eff)
    cutoffs_list = _to_int_list(cutoffs)
    remainders_list = _to_int_list(remainders)
    cache_slots_list = _to_int_list(cache_slots)

    for req in range(n):
        remainder = remainders_list[req]
        if remainder == 0:
            continue
        slot = cache_slots_list[req]
        src_start = cu_eff_list[req] + cutoffs_list[req]
        src = slice(src_start, src_start + remainder)
        dst = slice(pending_offset, pending_offset + remainder)
        kv_state[slot, dst] = flat_kv[src]
        score_state[slot, dst] = flat_score[src] + ape[:remainder]
