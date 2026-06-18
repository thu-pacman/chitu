# SPDX-FileCopyrightText: 2025 Qingcheng.AI
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

from chitu.global_vars import get_global_args


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


def _get_pd_cfg():
    try:
        args = get_global_args()
    except Exception:
        return None
    multi_inst = getattr(args, "multi_inst", None)
    router = getattr(multi_inst, "router", None)
    return getattr(router, "pd_disaggregation", None)


def pd_verbose_enabled() -> bool:
    """Enable high-frequency PD logs without global DEBUG."""
    env = _env_flag("CHITU_PD_LOG_VERBOSE")
    if env is not None:
        return env
    pd_cfg = _get_pd_cfg()
    if pd_cfg is None:
        return False
    return bool(getattr(pd_cfg, "log_verbose", False))


def pd_trace_enabled() -> bool:
    """Enable per-request PD trace logs."""
    env = _env_flag("CHITU_PD_TRACE")
    if env is not None:
        return env
    return pd_verbose_enabled()
