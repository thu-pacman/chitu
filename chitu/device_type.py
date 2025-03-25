import torch

_device_name = None


def get_device_name():
    global _device_name
    if _device_name is None:
        _device_name = torch.cuda.get_device_name()
    return _device_name


def is_nvidia():
    return "NVIDIA" in get_device_name()


def is_muxi():
    MUXI_DEVICE_PATTERNS = ["4000", "4001"]
    device_name = get_device_name()
    return any(pattern in device_name for pattern in MUXI_DEVICE_PATTERNS)


def has_native_fp8():
    return is_nvidia() and torch.cuda.get_device_capability() >= (8, 9)
