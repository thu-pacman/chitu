# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import shutil

_PLATFORM_PROBES = (
    ("nvidia", "nvidia-smi"),
    ("ascend", "npu-smi"),
    ("hygon", "hy-smi"),
    ("metax", "mx-smi"),
)


def resolve_platform(configured_platform, supported_platforms, runtime_name):
    if configured_platform != "auto":
        if configured_platform not in supported_platforms:
            supported_values = ", ".join(("auto", *supported_platforms))
            raise ValueError(
                f"boot.platform={configured_platform!r} is not supported by "
                f"{runtime_name}; supported values: {supported_values}"
            )
        return configured_platform

    for platform, probe in _PLATFORM_PROBES:
        if platform in supported_platforms and shutil.which(probe):
            return platform

    raise RuntimeError(
        f"No supported type of devices detected for {runtime_name}; "
        f"supported platforms: {', '.join(supported_platforms)}"
    )
