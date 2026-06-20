# SPDX-FileCopyrightText: 2025 Qingcheng.AI
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional


def _env_flag(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    val = raw.strip().lower()
    if val == "1":
        return True
    if val == "0":
        return False
    return None


def pd_trace_enabled() -> bool:
    """Enable per-request PD trace logs.

    This flag is intentionally independent from the normal logger level used for
    verbose PD disaggregation logs.
    """
    env = _env_flag("CHITU_PD_TRACE")
    if env is not None:
        return env
    return False
