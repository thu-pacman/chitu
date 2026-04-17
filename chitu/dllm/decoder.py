# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""dLLM Parallel Decoder with confidence-based token selection."""

import math
from typing import Optional

import torch
import torch.nn.functional as F


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Add Gumbel noise to logits for sampling diversity."""
    if math.isclose(temperature, 0.0):
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


class DLLMDecoder:
    """dLLM Parallel Decoder with confidence-based token selection.

    Tokens with prediction confidence >= threshold are decoded in parallel.
    """

    def __init__(
        self,
        temperature: float = 0.0,
        threshold: float = 0.9,
        mask_id: int = 126336,
        eos_id: int = 126081,
    ):
        self.temperature = temperature
        self.threshold = threshold
        self.mask_id = mask_id
        self.eos_id = eos_id

    def _get_transfer_mask(
        self,
        logits: torch.Tensor,
        block_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Core decoding logic: compute which mask tokens to decode.

        Returns:
            (transfer_index, predicted_tokens): Boolean mask of positions to decode,
            and the predicted token values.
        """
        mask_index = block_tokens == self.mask_id
        if not mask_index.any():
            return mask_index, block_tokens

        noisy_logits = add_gumbel_noise(logits, self.temperature)
        predicted_tokens = torch.argmax(noisy_logits, dim=-1)

        probs = F.softmax(logits.to(torch.float32), dim=-1)
        confidence = probs.gather(dim=-1, index=predicted_tokens.unsqueeze(-1)).squeeze(
            -1
        )
        confidence = torch.where(
            mask_index, confidence, torch.full_like(confidence, float("-inf"))
        )

        max_confidence = confidence.max(dim=-1, keepdim=True)[0]
        actual_threshold = torch.clamp(max_confidence - 1e-5, max=self.threshold)
        transfer_index = (confidence >= actual_threshold) & mask_index

        return transfer_index, predicted_tokens

    def decode(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        block_start: int,
        block_end: int,
    ) -> torch.Tensor:
        """Decode mask tokens within a single block."""
        block_tokens = tokens[:, block_start:block_end]
        transfer_index, predicted_tokens = self._get_transfer_mask(logits, block_tokens)
        tokens[:, block_start:block_end] = torch.where(
            transfer_index, predicted_tokens, block_tokens
        )
        return tokens

    def batch_decode(
        self,
        logits: torch.Tensor,
        block_starts: torch.Tensor,
        token_array,
        block_length: int,
    ) -> None:
        """Batch decode with different starting positions. Modifies token_array in-place."""
        tokens = token_array.data if hasattr(token_array, "data") else token_array
        batch_size, total_length = tokens.shape
        device = tokens.device

        offsets = torch.arange(block_length, device=device).unsqueeze(0)
        indices = block_starts.unsqueeze(1) + offsets

        block_tokens = torch.gather(
            tokens, dim=1, index=indices.clamp(max=total_length - 1)
        )
        transfer_index, predicted_tokens = self._get_transfer_mask(logits, block_tokens)
        new_block_tokens = torch.where(transfer_index, predicted_tokens, block_tokens)
        tokens.scatter_(dim=1, index=indices, src=new_block_tokens)

    def has_mask(self, tokens: torch.Tensor, block_start: int, block_end: int) -> bool:
        return (tokens[:, block_start:block_end] == self.mask_id).any().item()

    def batch_has_mask(
        self,
        tokens: torch.Tensor,
        block_starts: torch.Tensor,
        block_length: int,
    ) -> torch.Tensor:
        batch_size, total_length = tokens.shape
        device = tokens.device

        offsets = torch.arange(block_length, device=device).unsqueeze(0)
        indices = block_starts.unsqueeze(1) + offsets
        block_tokens = torch.gather(
            tokens, dim=1, index=indices.clamp(max=total_length - 1)
        )
        return (block_tokens == self.mask_id).any(dim=-1)

    def count_masks(
        self, tokens: torch.Tensor, block_start: int, block_end: int
    ) -> int:
        return (tokens[:, block_start:block_end] == self.mask_id).sum().item()

    def get_mask_positions(
        self, tokens: torch.Tensor, block_start: int, block_end: int
    ) -> torch.Tensor:
        return tokens[:, block_start:block_end] == self.mask_id
