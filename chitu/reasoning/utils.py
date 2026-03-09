# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
from chitu.global_vars import get_global_args


@functools.lru_cache(maxsize=1)
def get_initial_reasoning_state():
    # Some models include <think> in the prompt when reasoning is enabled, so we need
    # the initial reasoning state to decide whether output belongs to content or reasoning_content.
    args = get_global_args()
    flag = getattr(args.models, "reasoning_without_begin", False)
    return flag
