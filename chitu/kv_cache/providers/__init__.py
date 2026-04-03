# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0


def register_all_providers():
    from . import default  # noqa: F401
    from . import deepseek_v3  # noqa: F401
    from . import llada  # noqa: F401
    from . import qwen  # noqa: F401
