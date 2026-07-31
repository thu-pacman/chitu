# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Transfer buffer abstraction for PD disaggregation.

``TransferBuffers`` carries only per-cache physical block ID lists.
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
    """

    cache_block_ids: dict[str, np.ndarray] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return any(len(v) > 0 for v in self.cache_block_ids.values())

    # ---- serialization -----------------------------------------------------

    def to_msgpackable(self) -> dict[str, Any]:
        """Serialize as ``{cache_name: int32_bytes}``."""
        return {k: v.tobytes() for k, v in self.cache_block_ids.items()}

    @classmethod
    def from_msgpackable(cls, data: dict[str, Any]) -> "TransferBuffers":
        """Reconstruct from msgpack data."""
        return cls(
            cache_block_ids={
                k: np.frombuffer(v, dtype=np.int32) for k, v in data.items()
            },
        )
