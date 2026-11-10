# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Unmerged MXFP4 routed experts on Enflame GCU.

The implementation lives in :mod:`chitu.quantization.mxfp4` (shared with the
portable path); this module only registers it for GCU. GCU's DSV4 paged and
skew paths need the unmerged expert layout, hence ``merge_gate_up=False``.
"""

from chitu.device_type import is_enflame
from chitu.quantization.mxfp4 import MXFP4MoeExpertsUnmergedBase
from chitu.quantization.registry import QuantizationRegistry


@QuantizationRegistry.register_moe_experts(
    "mxfp4", merge_gate_up=False, when=lambda _: is_enflame()
)
class Mxfp4MoeExpertsUnmergedGCU(MXFP4MoeExpertsUnmergedBase):
    """MXFP4 routed experts (unmerged gate/up) on Enflame GCU."""
