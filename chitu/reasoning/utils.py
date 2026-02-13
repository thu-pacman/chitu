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
    if args.models.name in [
        "DeepSeek-V3.1",
        "GLM-4.7",
        "GLM-5",
        "GLM-5-FP8",
        "DeepSeek-V3.2",
    ]:
        return True
    else:
        return False
