# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Tuple
import importlib
import site
import os
import torch


def try_import_platform_dep(pkg_name: str) -> Tuple[Any, bool]:
    """
    Import a dependency that may not be available on all platforms.

    DO NOT use this function to import optional dependencies that users can pick.
    Use `try_import_opt_dep` instead.

    Args:
        pkg_name (str): The name of the Python package to import.

    Returns:
        [0]: The imported module if successful, or a dummy object that raises an ImportError.
        [1]: A boolean indicating whether the import was successful.
    """
    try:
        return importlib.import_module(pkg_name), True
    except ImportError as e:

        class ReportErrorWhenUsed:
            def __init__(self, e):
                self.root_cause = e

            def __getattr__(self, item):
                raise ImportError(
                    f"Chitu does not support this case because '{pkg_name}' is not present on this platform. "
                    f"This is likely a bug of Chitu."
                ) from self.root_cause

        return ReportErrorWhenUsed(e), False


def try_import_opt_dep(pkg_name: str, opt_dep_name: str) -> tuple[Any, bool]:
    """
    Import an optional dependency.

    The package name and optional dependency name should be consistent with the listing
    in `setup.py`. For example, you can list a Python package `my_quant_wxax` in the
    `quant` extra of `setup.py`, then you can use this function like `try_import_opt_dep('my_quant_wxax', 'quant')`,
    and the user may install the optional dependency like `pip install chitu[quant]`.

    DO NOT use this function to import platform-specific dependencies that users are unable
    to install at their will. Use `try_import_platform_dep` instead.

    Args:
        pkg_name (str): The name of the Python package to import.
        opt_dep_name (str): The name of the optional dependency category in `setup.py`.

    Returns:
        [0]: The imported module if successful, or a dummy object that raises an ImportError.
        [1]: A boolean indicating whether the import was successful.
    """

    # Keep this sync with get_requires.py
    opt_deps = {
        "quant",
        "muxi_layout_kernels",
        "muxi_w8a8_kernels",
        "ascend_kernels",
        "flash_attn",
        "flashinfer",
        "fla",
        "flash_mla",
        "deep_gemm",
        "deep_ep",
        "cpu",
        "hard_fp4_kernels",
        "scipy",
        "fast_hadamard_transform",
        "flash_attn_interface",
        "torchada",
        "metax_soft_fp4_kernels",
        "sugon_mixq4_kernels",
        "sugon_w4a8_kernels",
        "mooncake",
    }
    assert (
        opt_dep_name in opt_deps
    ), f"To chitu developers: Please don't use {opt_dep_name} as an optional dependency name, it is not listed in get_requires.py."

    try:
        return importlib.import_module(pkg_name), True
    except ImportError as e:

        class ReportErrorWhenUsed:
            def __init__(self, e):
                self.root_cause = e

            def __getattr__(self, item):
                raise ImportError(
                    f"Optional dependency '{opt_dep_name}' is not installed. "
                    f"Please refer to README.md for installation instructions."
                ) from self.root_cause

        return ReportErrorWhenUsed(e), False


def get_ascend_custom_opp_path():
    site_packages_path = os.path.join(site.getsitepackages()[0], "vendors", "customize")
    return site_packages_path


_torch_npu_has_set_up = False


def try_import_and_setup_torch_npu():
    """
    Try importing `torch_npu`. If successful, also do some setup.
    """

    global _torch_npu_has_set_up

    torch_npu, has_torch_npu = try_import_platform_dep("torch_npu")

    if has_torch_npu and not _torch_npu_has_set_up:
        # Make "torch.cuda" point to NPU devices
        from torch_npu.contrib import transfer_to_npu

        torch.cuda.CUDAGraph = torch.npu.NPUGraph

        # Allow using NpuFractalNzTensor and NpuFractalZnTensor
        torch_npu.npu.config.allow_internal_format = True

        # Setup paths to op libraries
        site_packages_path = get_ascend_custom_opp_path()
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = site_packages_path

        _torch_npu_has_set_up = True

    return torch_npu, has_torch_npu
