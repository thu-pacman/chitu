# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Transfer buffer abstraction for PD disaggregation.

``TransferBuffers`` carries physical block IDs and optional sequence page indices.
All GPU address computation is handled by ``StaticTransferPlan`` +
``create_transfer_plan``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TransferBuffers:
    """Per-request block ID lists for KV cache transfer.

    ``cache_block_ids`` maps cache_name to a 1-D int32 array of
    **physical** block IDs.

    ``sequence_page_indices`` is used in dcp for decode to tell
    prefill which logical pages is current physical page belongs to
    so that prefill can related the prefill physical page id to the decode physical page id

    """

    cache_block_ids: dict[str, np.ndarray] = field(default_factory=dict)
    sequence_page_indices: dict[str, np.ndarray] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return any(len(v) > 0 for v in self.cache_block_ids.values())

    # ---- serialization -----------------------------------------------------

    def to_msgpackable(self) -> dict[str, Any]:
        """Serialize nested cache_block_ids and optional logical_block_ids mappings."""

        result: dict[str, Any] = {}
        physical = {k: v.tobytes() for k, v in self.cache_block_ids.items()}
        result["cache_block_ids"] = physical
        if self.sequence_page_indices:
            result["logical_block_ids"] = {
                k: v.tobytes() for k, v in self.sequence_page_indices.items()
            }

        return result

    @classmethod
    def from_msgpackable(cls, data: dict[str, Any]) -> "TransferBuffers":
        """Deserialize physical block IDs and optional sequence page indices."""
        return cls(
            cache_block_ids={
                k: np.frombuffer(v, dtype=np.int32)
                for k, v in data["cache_block_ids"].items()
            },
            sequence_page_indices={
                k: np.frombuffer(v, dtype=np.int32)
                for k, v in data.get("logical_block_ids", {}).items()
            },
        )
