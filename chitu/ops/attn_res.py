# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Attention Residuals (AttnRes) mixing used by Kimi-K3.

Kimi-K3 replaces the plain residual connection with a per-token softmax mixture
of the running residual stream ("prefix sum") and periodic snapshots of it
("blocks", appended every ``attn_res_block_size`` layers). At each sublayer
boundary the layer input is

    out = sum_i softmax_i(s) * v_i,   s_i = RMSNorm(v_i) . (w_norm ⊙ w_proj)

where the sources ``v_i`` are {all blocks so far, the current prefix sum} and
the scores are computed on the RMS-normalized sources while the mixture output
uses the **raw** (un-normalized) sources. All score math runs in fp32.

This state is per-token and intra-forward only: the blocks are re-derived each
forward pass as the tokens flow through the layers, so nothing needs to be
cached across decode steps.
"""

import torch


def attn_res_mix(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    norm_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Mix the residual sources with a per-token softmax.

    Args:
        prefix_sum:     (num_tokens, hidden_size) running residual stream.
        block_residual: (num_tokens, num_blocks, hidden_size) snapshots; must
                        contain at least one block.
        norm_weight:    (hidden_size,) RMSNorm weight of the res-norm.
        proj_weight:    (hidden_size,) the res-proj linear weight with its
                        singleton output dim squeezed.
        eps:            RMSNorm variance epsilon.

    Returns:
        (num_tokens, hidden_size) mixed residual, in ``prefix_sum.dtype``.
    """
    num_tokens = prefix_sum.shape[0]
    hidden_size = prefix_sum.shape[-1]
    num_blocks = block_residual.shape[1]
    assert num_blocks >= 1, "attn_res_mix requires at least one block"
    assert block_residual.shape[0] == num_tokens
    assert block_residual.shape[2] == hidden_size

    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)  # [T, B+1, H]
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + eps)
    # The RMSNorm weight is folded into the score projection (rank-1 per source).
    score_weight = norm_weight.float() * proj_weight.float()  # [H]
    scores = (k * score_weight).sum(-1)  # [T, B+1]
    probs = scores.softmax(-1).unsqueeze(-1)  # [T, B+1, 1]
    # Weighted sum of the raw sources. Deliberately an elementwise mul + sum
    # (not torch.matmul): matmul may run in TF32 depending on the global backend
    # setting, while the reference computes this in full fp32. B+1 is tiny (≤ 9
    # for Kimi-K3), so the elementwise form costs nothing.
    mixed = (probs * v_float).sum(dim=1)  # [T, H]
    return mixed.to(prefix_sum.dtype)
