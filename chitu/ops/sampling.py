# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import xgrammar

from chitu.ops.utils import make_op_dispatcher
from chitu.utils import (
    try_import_platform_dep,
    try_import_and_setup_torch_npu,
    create_tensor,
)
from chitu.device_type import has_accelerator, is_muxi, is_ascend

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
has_triton_impl = has_triton and has_accelerator()

if has_triton_impl:
    from chitu.ops.triton_ops import apply_frequency_penalty_triton


@make_op_dispatcher
def apply_frequency_penalty(
    logits: torch.Tensor,
    indices: list[int],
    blocks: list[torch.Tensor],
    sizes: list[int],
    penalties: list[float],
    impl="auto",
):
    raise NotImplementedError


@apply_frequency_penalty.register_auto
def _auto_apply_frequency_penalty(
    logits: torch.Tensor,
    indices: list[int],
    blocks: list[torch.Tensor],
    sizes: list[int],
    penalties: list[float],
):
    bs = len(blocks)
    device = logits.device
    if device.type == "cpu":
        return "torch"
    if has_triton_impl and bs > 8 and bs <= 16:
        # NOTE: This is a temporary solution based tests on h20.
        return "triton"
    if bs < 16 or has_torch_npu:
        return "torch"
    return "cuda"


@apply_frequency_penalty.register("torch")
def _apply_frequency_penalty_torch(
    logits: torch.Tensor,
    indices: list[int],
    blocks: list[torch.Tensor],
    sizes: list[int],
    penalties: list[float],
):
    if len(blocks) == 0:
        return
    bs = len(blocks)
    assert len(blocks) == len(indices) == len(sizes) == len(penalties) == bs

    device = logits.device
    for block, idx, size, penalty in zip(blocks, indices, sizes, penalties):
        logits[idx].index_add_(
            -1,
            block[:size],
            torch.ones(size, dtype=logits.dtype, device=device) * -penalty,
        )


@apply_frequency_penalty.register("triton", available=has_triton_impl)
def _apply_frequency_penalty_triton(
    logits: torch.Tensor,
    indices: list[int],
    blocks: list[torch.Tensor],
    sizes: list[int],
    penalties: list[float],
):
    if len(blocks) == 0:
        return
    device = logits.device
    indices_ = create_tensor(indices, device=device, dtype=torch.int64)
    sizes_ = create_tensor(sizes, device=device, dtype=torch.int64)
    penalties_ = create_tensor(penalties, device=device, dtype=torch.float32)
    stacked_blocks = torch.stack(blocks)
    apply_frequency_penalty_triton(
        logits,
        indices_,
        stacked_blocks,
        sizes_,
        penalties_,
    )


@apply_frequency_penalty.register("cuda", available=has_chitu_backend)
def _apply_frequency_penalty_cuda(
    logits: torch.Tensor,
    indices: list[int],
    blocks: list[torch.Tensor],
    sizes: list[int],
    penalties: list[float],
):
    if len(blocks) == 0:
        return
    bs = len(blocks)
    device = logits.device
    indices_ = create_tensor(indices, device=device, dtype=torch.int64)
    sizes_ = create_tensor(sizes, device=device, dtype=torch.int64)
    penalties_ = create_tensor(penalties, device=device, dtype=torch.float32)
    assert logits.dtype == torch.float32
    block_ptrs = create_tensor(
        [block.data_ptr() for block in blocks], device=device, dtype=torch.int64
    )
    chitu_backend.cuda_frequency_penalty(
        logits,
        indices_,
        block_ptrs,
        penalties_,
        sizes_,
        bs,
        logits.shape[-1],
        logits.stride(0),
        logits.stride(1),
    )


@make_op_dispatcher
def batch_append_tokens(
    blocks: list[torch.Tensor], indices: list[int], tokens: list[int], impl="auto"
):
    raise NotImplementedError


@batch_append_tokens.register_auto
def _auto_batch_append_tokens(
    blocks: list[torch.Tensor], indices: list[int], tokens: list[int]
):
    device = blocks[0].device
    if has_chitu_backend and device.type == "cuda" and len(blocks) > 8:
        return "cuda"
    return "torch"


@batch_append_tokens.register("torch")
def _batch_append_tokens_torch(
    blocks: list[torch.Tensor],
    indices: list[int],
    tokens: list[int],
):
    for block, index, token in zip(blocks, indices, tokens):
        block[index] = token


@batch_append_tokens.register("cuda", available=has_chitu_backend)
def _batch_append_tokens_cuda(
    blocks: list[torch.Tensor],
    indices: list[int],
    tokens: list[int],
):
    device = blocks[0].device
    block_ptrs = create_tensor(
        [block.data_ptr() for block in blocks], device=device, dtype=torch.int64
    )
    tokens = create_tensor(tokens, device=device, dtype=torch.int64)
    indices = create_tensor(indices, device=device, dtype=torch.int32)
    _need_expand = torch.zeros(len(blocks), device=device, dtype=torch.bool)
    chitu_backend.cuda_response_append(
        block_ptrs, block_ptrs, tokens, indices, _need_expand
    )


def apply_bitmask_torch(
    logits: torch.Tensor, bitmask: torch.Tensor, indices: list[int]
):
    _, H = logits.shape
    _, M = bitmask.shape
    B = len(indices)
    bitmask = bitmask[indices].view(B, M, 1)
    # shift left fallback to cpu on npu, thus we use pow
    bits = torch.arange(32, device=logits.device, dtype=torch.int32)
    bits = torch.pow(2, bits).view(1, 1, 32)
    mask = bits & bitmask
    mask = mask.view(B, M * 32)[:, :H]
    logits[indices] = logits[indices].masked_fill_(mask == 0, float("-inf"))
    return logits


def apply_bitmask(logits: torch.Tensor, bitmask: torch.Tensor, indices: list[int]):
    """
    apply bitmask to logits

    for idx in indices:
        for token in range(vocab_size):
            if bitmask[idx, token // 32] & (1 << (token % 32)) == 0:
                logits[idx, token] = -inf
    """
    if is_ascend() or is_muxi():
        return apply_bitmask_torch(logits, bitmask, indices)
    return xgrammar.apply_token_bitmask_inplace(logits, bitmask, indices=indices)


def filter_logits_top_k_top_p(
    logits: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    max_top_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply top-k truncation + top-p filtering, return renormalized reduced-space probs.

    Steps:
      1. torch.topk(logits, k=max_top_k) — restrict to top-k candidates
      2. Mask positions >= top_k[b] with -inf
      3. softmax over reduced space
      4. Mask cumulative mass > top_p[b] to 0
      5. Renormalize over the surviving support

    Renormalization is required for correctness: Gumbel-max sampling is
    scale-invariant, so drawing from the filtered (unnormalized) probs is
    equivalent to drawing from ``probs / sum`` — the effective proposal/target
    distribution is the conditional distribution over the surviving support.
    compute_mtp_acceptance / resample_mtp_rejected compare q and p by value, so
    they MUST receive normalized distributions; otherwise the acceptance ratio
    and the residual are off by ``sum(p) / sum(q)`` and the output drifts from
    the target.

    Returns:
        filtered_probs: (N, max_top_k)  renormalized filtered probabilities
        token_ids:      (N, max_top_k)  original vocab indices from topk
    """
    if max_top_k is None:
        max_top_k = logits.shape[-1]

    logits, token_ids = torch.topk(logits, k=max_top_k, dim=-1)  # reduce calculation
    topk_mask = (
        torch.arange(max_top_k, device=logits.device)[None, :] >= top_ks[:, None]
    )
    logits[topk_mask] = float("-inf")
    # topp is applied on filtered probs by topk

    probs = torch.softmax(logits, dim=-1)
    topp_mask = (torch.cumsum(probs, dim=-1) - probs) > top_ps[:, None]
    probs[topp_mask] = 0

    # Renormalize over the surviving support (per row). Row sums are positive
    # (top-k >= 1 always keeps the top token), the clamp is purely defensive.
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(probs.dtype).tiny
    )

    return probs, token_ids


def gumbel_max_sample(
    probs: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Sample one token per row via Gumbel-max trick.

    probs need not be normalized; argmax(probs / exponential_noise) is
    equivalent to multinomial sampling from the normalized distribution.
    """
    noise = torch.empty_like(probs).exponential_()
    sample_idx = (probs / noise).argmax(dim=-1)
    return torch.gather(token_ids, index=sample_idx[:, None], dim=1).squeeze(1)


def top_k_top_p_min_p_sampling_from_logits(
    logits: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    max_top_k: int | None = None,
):
    filtered_probs, token_ids = filter_logits_top_k_top_p(
        logits, top_ks, top_ps, max_top_k=max_top_k
    )
    return gumbel_max_sample(filtered_probs, token_ids)


def _sparse_lookup(
    probs: torch.Tensor,
    token_ids: torch.Tensor,
    query_ids: torch.Tensor,
) -> torch.Tensor:
    """Look up probability mass of a sparse (probs, token_ids) row at query ids.

    ``token_ids`` rows may contain padding ids whose probability is 0, so the
    mask-sum is exact: a query id matches at most one active position per row.

    Args:
        probs:      (N, K)  sparse probabilities (0 where masked/padded)
        token_ids:  (N, K)  matching vocab ids from top-k
        query_ids:  (N, Q)  vocab ids to look up

    Returns:
        (N, Q) probability at each query id (0 where the id is absent).
    """
    matches = token_ids.unsqueeze(1) == query_ids.unsqueeze(2)  # (N, Q, K)
    return (probs.unsqueeze(1) * matches).sum(-1)


def compute_mtp_acceptance(
    q_probs: torch.Tensor,
    q_token_ids: torch.Tensor,
    p_probs: torch.Tensor,
    p_token_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    exact_match: torch.Tensor,
    greedy_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-depth acceptance for MTP rejection sampling.

    For greedy requests (greedy_mask[b] == True):
        accepted = exact_match  (sampled_token == draft)
    For non-greedy requests:
        accepted = (rand < min(1, q(d)/p(d)))

    q and p are passed as sparse top-k ``(probs, token_ids)`` pairs -- the only
    values consumed are those at the draft token positions, so materializing
    full-vocab tensors is unnecessary. Both must be normalized over their
    surviving support (filter_logits_top_k_top_p returns such rows); comparing
    raw masses would bias acceptance by ``sum(p) / sum(q)``.

    Args:
        q_probs:      (bs*n_drafts, K)  target top-k probs at draft positions
        q_token_ids:  (bs*n_drafts, K)  target top-k vocab ids
        p_probs:      (bs, n_drafts, K) draft proposal top-k probs
        p_token_ids:  (bs, n_drafts, K) draft proposal top-k vocab ids
        draft_tokens: (bs, n_drafts)    draft token ids (from MTP layers)
        exact_match:  (bs, n_drafts) bool  sampled_token == draft token
        greedy_mask:  (bs,) bool        which requests are greedy

    Returns:
        accepted:       (bs, n_drafts) bool  per-depth acceptance
        accept_indices: (bs,) long            first rejected depth, or n_drafts if all accepted
    """
    bs, n_drafts = draft_tokens.shape
    mtp_size = n_drafts + 1

    # Probabilistic acceptance for non-greedy:
    #   q(d): target probability of the draft token
    #   p(d): draft probability of the draft token
    draft_flat = draft_tokens.reshape(bs * n_drafts, 1)
    q_d = _sparse_lookup(q_probs, q_token_ids, draft_flat).view(bs, n_drafts)
    p_d = _sparse_lookup(
        p_probs.reshape(bs * n_drafts, -1),
        p_token_ids.reshape(bs * n_drafts, -1),
        draft_flat,
    ).view(bs, n_drafts)
    accept_prob = torch.clamp(q_d / p_d, max=1.0)  # min(1, q(d)/p(d))
    rand = torch.rand(bs, n_drafts, device=q_probs.device)
    prob_accepted = rand < accept_prob  # (bs, n_drafts)

    # Per-request dispatch: exact_match for greedy, prob_accepted for non-greedy
    greedy_exp = greedy_mask.unsqueeze(-1).expand(bs, n_drafts)
    accepted = torch.where(greedy_exp, exact_match, prob_accepted)

    # First rejected depth → accept_indices
    accept_indices = torch.argmin(accepted.int(), dim=1)
    accept_indices[accepted.all(dim=1)] = mtp_size - 1

    return accepted, accept_indices


def resample_mtp_rejected(
    tokens: torch.Tensor,
    q_all_probs: torch.Tensor,
    q_all_token_ids: torch.Tensor,
    p_probs: torch.Tensor,
    p_token_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    accept_indices: torch.Tensor,
    greedy_mask: torch.Tensor,
    mtp_size: int,
) -> torch.Tensor:
    """Resample rejected positions and finalize accepted positions for MTP.

    tokens[:, 0] is the target prediction at the same position as draft[0];
    tokens[:, n_drafts-1] is the target at draft[n_drafts-1];
    tokens[:, n_drafts] (the last position, the bonus token) has no draft — it
    is kept as-is from _sample_mtp.

    Accepted draft positions (depth d < accept_indices[b]) emit the DRAFT token
    (target's KV cache and MTP hidden states advance on drafts).  Rejected
    positions are resampled from norm(max(0, q - p)).

    q and p must be normalized over their surviving support (as returned by
    filter_logits_top_k_top_p); otherwise the residual is off by the scale
    mismatch and the output drifts from the target.

    p is passed as a sparse top-k ``(probs, token_ids)`` pair -- only the values
    at the target's top-k positions are consumed, so no full-vocab tensor is
    materialized.

    Args:
        tokens:         (bs, mtp_size)          in/out — sampled token ids
        q_all_probs:    (bs*mtp_size, K)        filtered target probs (all positions)
        q_all_token_ids:(bs*mtp_size, K)        token indices from top-k
        p_probs:        (bs, n_drafts, K)       draft proposal top-k probs
        p_token_ids:    (bs, n_drafts, K)       draft proposal top-k vocab ids
        draft_tokens:   (bs, n_drafts)          draft token ids (what was proposed)
        accept_indices: (bs,) long              first rejected draft depth, or n_drafts if all accepted
        greedy_mask:    (bs,) bool
        mtp_size:       int

    Returns:
        tokens: (bs, mtp_size)  updated token ids
    """
    bs = tokens.shape[0]
    n_drafts = mtp_size - 1
    K = q_all_probs.shape[-1]
    device = q_all_probs.device

    q_probs_3d = q_all_probs.view(bs, mtp_size, K)
    q_token_ids_3d = q_all_token_ids.view(bs, mtp_size, K)

    # ---- q for positions 0..n_drafts-1 (= positions with drafts) ----
    q_probs = q_probs_3d[:, :n_drafts].reshape(bs * n_drafts, K)
    q_token_ids = q_token_ids_3d[:, :n_drafts].reshape(bs * n_drafts, K)

    # ---- lookup p at q_token_ids positions (sparse, no full-vocab tensor) ----
    p_reduced = _sparse_lookup(
        p_probs.reshape(bs * n_drafts, -1),
        p_token_ids.reshape(bs * n_drafts, -1),
        q_token_ids,
    ).view(bs, n_drafts, K)

    # ---- residual = max(0, q - p) ----
    residual = torch.clamp(q_probs.view(bs, n_drafts, K) - p_reduced, min=0)
    residual_sum = residual.sum(dim=-1)  # (bs, n_drafts)

    # ---- Gumbel-max samples ----
    res_sample = gumbel_max_sample(
        residual.reshape(bs * n_drafts, K),
        q_token_ids,
    ).view(bs, n_drafts)

    q_fallback = gumbel_max_sample(
        q_probs.view(bs, n_drafts, K).reshape(bs * n_drafts, K),
        q_token_ids,
    ).view(bs, n_drafts)

    # ---- select residual or q-fallback ----
    rejection_sample = torch.where(
        residual_sum > 0,
        res_sample,
        q_fallback,
    )  # (bs, n_drafts)

    # ---- masks ----
    depths = torch.arange(n_drafts, device=device)[None, :]  # (1, n_drafts)
    # Accepted depth d: d < accept_indices[b] (all depths before the first rejection)
    accept_mask = depths < accept_indices[:, None]  # (bs, n_drafts)
    reject_mask = ~greedy_mask[:, None] & (
        accept_indices[:, None] == depths
    )  # (bs, n_drafts)

    # ---- apply updates ----
    # 1. Accepted draft positions → emit the DRAFT token
    # 2. Rejected position → resampled token
    # 3. tokens[:, n_drafts] is the bonus token already sampled from q by _sample_mtp
    #    — kept as-is.
    tokens_out = tokens.clone()
    tokens_out[:, :n_drafts] = torch.where(
        accept_mask, draft_tokens, tokens[:, :n_drafts]
    )
    tokens_out[:, :n_drafts] = torch.where(
        reject_mask, rejection_sample, tokens_out[:, :n_drafts]
    )

    return tokens_out
