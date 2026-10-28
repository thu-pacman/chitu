# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Tuple
import importlib
import site
import os


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
        "grammar",
        "muxi_layout_kernels",
        "muxi_w8a8_kernels",
        "ascend_kernels",
        "flash_attn",
        "flashinfer",
        "fla",
        "flash_mla",
        "deep_gemm",
        "deep_ep",
        "tilelang",
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
        "hpc_ops",
        "nvidia-ml-py",
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
        import torch
        from torch_npu.contrib import transfer_to_npu

        torch.cuda.CUDAGraph = torch.npu.NPUGraph

        # Allow using NpuFractalNzTensor and NpuFractalZnTensor
        torch_npu.npu.config.allow_internal_format = True

        # Setup paths to op libraries
        site_packages_path = get_ascend_custom_opp_path()
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = site_packages_path

        _torch_npu_has_set_up = True

    return torch_npu, has_torch_npu


_torch_gcu_has_set_up = False


def try_import_and_setup_torch_gcu():
    """
    Try importing `torch_gcu`. If successful, install its torch and Triton shims.

    This must run before any other chitu submodule is imported: `device_type`
    caches its device name, and the stock `triton` package resolves its active
    driver at import time.
    """

    global _torch_gcu_has_set_up

    torch_gcu, has_torch_gcu = try_import_platform_dep("torch_gcu")

    if has_torch_gcu and not _torch_gcu_has_set_up:
        try:
            # The shim maps ".cuda()" to ".gcu()" and the "nccl" process-group
            # backend to "eccl".
            importlib.import_module("torch_gcu.transfer_to_gcu")
        except Exception:
            # torch_gcu exists but its shim is unusable; leave the stock torch
            # behavior intact.
            return torch_gcu, has_torch_gcu

        try:
            # Activate the GCU backend for the stock `triton` package.
            importlib.import_module("triton_gcu.triton")
        except Exception:
            # triton_gcu is optional; without it the stock Triton backend is used.
            pass

        _torch_gcu_has_set_up = True

    return torch_gcu, has_torch_gcu


_gcu_c_ops_loaded = False


def ensure_gcu_c_ops():
    """Load the compiled `vllm_gcu._C` extension exactly once, which registers the native
    GCU kernels under `torch.ops._C.*` (rms_norm, rotary_embedding, silu_and_mul,
    linear_quant, dynamic_per_token_group_fp8_quant, ...). We load only the .so, NOT the
    vllm / vllm_gcu Python stack (importing those inside chitu re-activates vLLM's platform
    and hangs the GCU). Idempotent; safe to call from any GCU op impl."""
    global _gcu_c_ops_loaded
    if _gcu_c_ops_loaded:
        return
    import importlib.util
    import os
    import glob
    import sys

    spec = importlib.util.find_spec("vllm_gcu")
    if spec is not None:
        # Support both a regular package (spec.origin = __init__.py) and a namespace
        # package (origin=None, dir in submodule_search_locations).
        search_dirs = []
        if spec.origin and os.path.isfile(spec.origin):
            search_dirs.append(os.path.dirname(spec.origin))
        search_dirs += list(spec.submodule_search_locations or [])
        for d in dict.fromkeys(search_dirs):
            so_files = glob.glob(os.path.join(d, "_C*.so"))
            if not so_files:
                continue
            # Load with LAZY + GLOBAL binding. The vendor `_C.abi3.so` may carry a skewed
            # unused overload vs the installed libtopsaten; eager binding would reject the
            # whole module at load, but chitu never calls that overload -- lazy binding
            # resolves only the ops we use at call time. GLOBAL so its symbols are visible
            # to later dlopens. Name MUST end in "._C" so CPython looks up PyInit__C in the
            # abi3 extension.
            old_flags = sys.getdlopenflags()
            sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_GLOBAL)
            try:
                cspec = importlib.util.spec_from_file_location(
                    "vllm_gcu._C", so_files[0]
                )
                mod = importlib.util.module_from_spec(cspec)
                cspec.loader.exec_module(mod)
            finally:
                sys.setdlopenflags(old_flags)
            break
    _gcu_c_ops_loaded = True
