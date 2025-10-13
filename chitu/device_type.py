# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

_device_name = None


def get_device_name():
    global _device_name
    if _device_name is None:
        if torch.cuda.is_available():
            _device_name = torch.cuda.get_device_name()
        else:
            _device_name = "CPU"
    return _device_name


def is_nvidia():
    return "NVIDIA" in get_device_name()


def is_muxi():
    MUXI_DEVICE_PATTERNS = ["4000", "4001", "MetaX"]
    device_name = get_device_name()
    return any(
        pattern.lower() in device_name.lower() for pattern in MUXI_DEVICE_PATTERNS
    )


def is_ascend():
    return "Ascend" in get_device_name()


def is_ascend_910b():
    return "910B" in get_device_name()


def has_native_fp8():
    return is_nvidia() and torch.cuda.get_device_capability() >= (8, 9)


def is_hopper():
    HOPPER_DEVICE_PATTERNS = ["H20", "H100"]
    device_name = get_device_name()
    return any(pattern in device_name for pattern in HOPPER_DEVICE_PATTERNS)


def is_blackwell():
    BLACKWELL_DEVICE_PATTERNS = ["5090", "B200", "B100"]
    device_name = get_device_name()
    return any(pattern in device_name for pattern in BLACKWELL_DEVICE_PATTERNS)


def is_hygon():
    HYGON_DEVICE_PATTERNS = ["BW"]
    device_name = get_device_name()
    return any(pattern in device_name for pattern in HYGON_DEVICE_PATTERNS)
