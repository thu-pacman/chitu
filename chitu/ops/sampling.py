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
def multinomial(
    probs: torch.Tensor,
    num_samples: int,
    seq_groups: Optional[list] = None,
    impl: str = "auto",
) -> torch.Tensor:
    raise NotImplementedError


@multinomial.register_auto
def _auto_multinomial():
    return "torch"


@multinomial.register("torch")
def _multinomial_torch(
    probs: torch.Tensor, num_samples: int, seq_groups: Optional[list] = None
) -> torch.Tensor:
    return torch.multinomial(probs, num_samples)


@multinomial.register("sync-free")
def _multinomial_sync_free(
    probs: torch.Tensor,
    num_samples: int,
    seq_groups: Optional[list] = None,
) -> torch.Tensor:
    # Adapted from
    # https://github.com/vllm-project/vllm/blob/4577fc9abb064d74b2082ffc5005cbb82ca91766/vllm/model_executor/layers/sampler.py#L527
    # SPDX-SnippetBegin
    # SPDX-License-Identifier: Apache-2.0
    # SPDX-SnippetCopyrightText: 2025 vLLM Team
    # SDPX—SnippetName: _multinomial from vllm
    if num_samples > 1:
        probs = probs.repeat_interleave(num_samples, dim=0)
    q = torch.empty_like(probs)
    if seq_groups is None:
        q.exponential_()
    else:
        sample_idx = 0
        for seq_group in seq_groups:
            seq_ids = seq_group.seq_ids
            stride = len(seq_ids) * num_samples
            assert seq_group.generator is not None
            q[sample_idx : sample_idx + stride].exponential_(
                generator=seq_group.generator
            )
            sample_idx += stride
    # SPDX-SnippetEnd
    return probs.div_(q).argmax(dim=1).view(-1, num_samples)


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


def top_k_top_p_min_p_sampling_from_logits(
    logits: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    # TODO: Support min_ps
):
    """A top-k, top-p and min-p sampling implementation."""

    if is_ascend() and has_torch_npu:
        assert logits.dim() == 2
        assert (
            top_ps.shape[0] == logits.shape[0]
        ), f"top_ps.shape[0]={top_ps.shape[0]} didn't match logits.shape[0]={logits.shape[0]}"
        assert (
            top_ks.shape[0] == logits.shape[0]
        ), f"top_ks.shape[0]={top_ks.shape[0]} didn't match logits.shape[0]={logits.shape[0]}"
        top_ps = top_ps.to(torch.float)
        top_ks = top_ks.to(torch.int32)
        probs = torch.softmax(logits, dim=-1)
        probs = torch_npu.npu_top_k_top_p(probs, top_ps, top_ks)
        sampled_index = multinomial(probs, num_samples=1, impl="sync-free").view(-1)
        return sampled_index

    # SPDX-SnippetBegin
    # SPDX-License-Identifier: Apache-2.0
    # SPDX-SnippetCopyrightText: 2025 SGLang Team
    # SPDX—SnippetName: top_k_top_p_min_p_sampling_from_logits_torch
    #
    # This sampling implementation is originally from SGLang
    # (https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/sampler.py),
    # licensed under Apache 2.0.
    probs = torch.softmax(logits, dim=-1)
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # TODO: Support min_ps like: min_p_thresholds = probs_sort[:, 0] * min_ps

    top_p_mask = (probs_sum - probs_sort) > top_ps.view(-1, 1)
    top_k_mask = torch.arange(0, probs.shape[-1], device=probs.device).view(
        1, -1
    ) >= top_ks.view(-1, 1)
    if is_ascend():
        probs_sort *= ~(top_p_mask | top_k_mask)
    else:
        probs_sort[top_p_mask | top_k_mask] = 0.0
    # TODO: Support min_ps like:  probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.max(dim=-1, keepdim=True)[0])
    sampled_index = multinomial(probs_sort, num_samples=1, impl="sync-free")
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids
    # SPDX-SnippetEnd
