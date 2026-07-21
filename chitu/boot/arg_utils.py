# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Sequence, List


def args_as_list(args) -> List[str]:
    if isinstance(args, str):
        return args.split()
    elif isinstance(args, Sequence):
        return [str(arg) for arg in args]
    else:
        raise ValueError(f"Unsupported argument type: {type(args)}")


def suffixed_name(name: Optional[str], suffix: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    if suffix is None or suffix == "":
        return name
    return f"{name}-{suffix}"
