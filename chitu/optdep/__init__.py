# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""The single entry point for optional third-party dependencies.

Usage::

    from chitu.optdep import torch_npu
    if torch_npu:                 # falsy _MissingModule when not installed
        torch_npu.npu_grouped_matmul(...)

A missing dependency is reported as a falsy ``_MissingModule``, not as ``None``.
Write ``if x:`` and do NOT write ``if x is not None``, which is always true.

Each dependency has one ``_load_<name>`` function whose body is the real import
statement, so a typo in a package name is caught by pyflakes. The "is it
installed" decision is centralized in ``__getattr__``.
"""

import logging
import os
import site
from typing import Any

# Only the dependencies whose call-site name differs from the import path
# (submodule probes) need an entry here.
_PATHS: dict[str, str] = {
    "aiter_moe": "aiter.moe",
    "lmslim_quantize_quant_ops": "lmslim.quantize.quant_ops",
    "boltops_kda": "boltops.fla.kda",
    "boltops_kda_gate": "boltops.fla.kda.triton.fused_recurrent",
    "boltops_conv": "boltops.generic.triton.causal_conv1d",
    "hygon_w8a8_kernel_dense": "hygon_w8a8_kernel.dense",
}

# Dependencies installable with `pip install chitu[extra]`. Everything else is
# provided by the platform, so a missing one is a Chitu bug. The value must
# match the extra name in get_requires.py, case included.
_EXTRAS: dict[str, str] = {
    "cpuinfer": "cpu",
    "numa": "cpu",
    "hpc": "hpc_ops",
    "pynvml": "nvidia-ml-py",
    "tbsgemm": "muxi_w8a8_kernels",
    "cinfer_ascendc": "ascend_kernels",
    "bitsandbytes": "quant",
    "gptqmodel_marlin_kernels": "quant",
    "sugon_mixQ4_kernels": "sugon_mixq4_kernels",
    "fla": "flash_linear_attention",
}

# Export nothing: `from chitu.optdep import *` must not import every dependency.
__all__: list[str] = []

_cache: dict[str, Any] = {}

logger = logging.getLogger(__name__)

# `hpc` loads a native library at import time, so on a host without the driver it
# fails with OSError rather than being absent. Warn once when that happens,
# otherwise every call site would repeat the same line.
_hpc_load_error_reported = False


class _MissingModule:
    """Stand-in for a dependency that is not installed.

    It is falsy on purpose, so call sites can write ``if x:``. Attribute access
    raises ImportError with installation instructions, so forgetting the check
    still yields a readable error instead of
    ``AttributeError: 'NoneType' object has no attribute ...``.
    """

    def __init__(self, name: str, extra: str | None, cause: BaseException) -> None:
        # Use __dict__ so that reading a field from __getattr__ cannot recurse
        # when __init__ has not finished yet.
        self.__dict__["_name"] = name
        self.__dict__["_extra"] = extra
        self.__dict__["_cause"] = cause

    def __bool__(self) -> bool:
        return False

    def __getattr__(self, item: str) -> Any:
        name = self.__dict__["_name"]
        extra = self.__dict__["_extra"]
        if extra is None:
            message = (
                f"Chitu does not support this case because '{name}' is not present "
                f"on this platform. This is likely a bug of Chitu."
            )
        else:
            message = (
                f"Optional dependency '{name}' is not installed. "
                f"Install it with `pip install chitu[{extra}]`."
            )
        raise ImportError(message) from self.__dict__["_cause"]

    def __repr__(self) -> str:
        return f"<missing module {self.__dict__['_name']!r}>"


# ---------------------------------------------------------------------------
# Plain dependencies: importing them is enough.
# ---------------------------------------------------------------------------


def _load_triton() -> Any:
    import triton

    return triton


def _load_chitu_backend() -> Any:
    import chitu_backend

    return chitu_backend


def _load_cpuinfer() -> Any:
    import cpuinfer

    return cpuinfer


def _load_deep_gemm() -> Any:
    import deep_gemm

    return deep_gemm


def _load_deepgemm() -> Any:
    import deepgemm

    return deepgemm


def _load_deep_ep() -> Any:
    import deep_ep

    return deep_ep


def _load_flash_attn() -> Any:
    import flash_attn

    return flash_attn


def _load_flash_attn_interface() -> Any:
    import flash_attn_interface

    return flash_attn_interface


def _load_flashinfer() -> Any:
    import flashinfer

    return flashinfer


def _load_flash_mla() -> Any:
    import flash_mla

    return flash_mla


def _load_fla() -> Any:
    import fla

    return fla


def _load_hpc() -> Any:
    global _hpc_load_error_reported

    try:
        import hpc
    except ModuleNotFoundError as e:
        # A missing inner dependency means the environment is broken, not that
        # `hpc` is absent.
        if not (e.name == "hpc" or "hpc".startswith(f"{e.name}.")):
            raise
        return _MissingModule("hpc", "hpc_ops", e)
    except (OSError, RuntimeError) as e:
        # `hpc` calls `torch.ops.load_library` in its `__init__`, which raises
        # OSError when a shared library it links against is missing (typically
        # libcuda.so.1 on a host without an NVIDIA driver). The package is
        # installed but unusable here, so treat it as unavailable.
        if not _hpc_load_error_reported:
            _hpc_load_error_reported = True
            logger.warning(
                "Optional dependency 'hpc' (package 'hpc') is installed but could not be "
                "loaded in this environment, and is treated as unavailable: %s: %s",
                type(e).__name__,
                e,
            )
        return _MissingModule("hpc", "hpc_ops", e)
    return hpc


def _load_scipy() -> Any:
    import scipy

    return scipy


def _load_fast_hadamard_transform() -> Any:
    import fast_hadamard_transform

    return fast_hadamard_transform


def _load_numa() -> Any:
    import numa

    return numa


def _load_pynvml() -> Any:
    import pynvml

    return pynvml


def _load_amdsmi() -> Any:
    import amdsmi

    return amdsmi


def _load_bitsandbytes() -> Any:
    import bitsandbytes

    return bitsandbytes


def _load_gptqmodel_marlin_kernels() -> Any:
    import gptqmodel_marlin_kernels

    return gptqmodel_marlin_kernels


def _load_muxi_layout_kernels() -> Any:
    import muxi_layout_kernels

    return muxi_layout_kernels


def _load_metax_soft_fp4_kernels() -> Any:
    import metax_soft_fp4_kernels

    return metax_soft_fp4_kernels


def _load_hard_fp4_kernels() -> Any:
    import hard_fp4_kernels

    return hard_fp4_kernels


def _load_cinfer_ascendc() -> Any:
    import cinfer_ascendc

    return cinfer_ascendc


def _load_sugon_mixQ4_kernels() -> Any:
    import sugon_mixQ4_kernels

    return sugon_mixQ4_kernels


def _load_sugon_w4a8_kernels() -> Any:
    import sugon_w4a8_kernels

    return sugon_w4a8_kernels


def _load_hygon_w4a16_kernel() -> Any:
    import hygon_w4a16_kernel

    return hygon_w4a16_kernel


def _load_hygon_w8a8_kernel() -> Any:
    import hygon_w8a8_kernel

    return hygon_w8a8_kernel


def _load_hygon_w8a8_kernel_dense() -> Any:
    import hygon_w8a8_kernel.dense

    return hygon_w8a8_kernel.dense


def _load_tbsgemm() -> Any:
    import tbsgemm

    return tbsgemm


def _load_lightop() -> Any:
    import lightop

    return lightop


def _load_lmslim() -> Any:
    import lmslim

    return lmslim


def _load_lmslim_quantize_quant_ops() -> Any:
    import lmslim.quantize.quant_ops

    return lmslim.quantize.quant_ops


def _load_boltops_kda() -> Any:
    import boltops.fla.kda

    return boltops.fla.kda


def _load_boltops_kda_gate() -> Any:
    import boltops.fla.kda.triton.fused_recurrent

    return boltops.fla.kda.triton.fused_recurrent


def _load_boltops_conv() -> Any:
    import boltops.generic.triton.causal_conv1d

    return boltops.generic.triton.causal_conv1d


def _load_aiter() -> Any:
    import aiter

    return aiter


def _load_aiter_moe() -> Any:
    import aiter.moe

    return aiter.moe


def _load_mooncake() -> Any:
    import mooncake

    return mooncake


def _load_tilelang() -> Any:
    import tilelang

    return tilelang


def _load_torchada() -> Any:
    import torchada

    return torchada


# ---------------------------------------------------------------------------
# Dependencies that need extra process setup after the import.
# ---------------------------------------------------------------------------


def _load_torch_npu() -> Any:
    import torch
    import torch_npu
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    # Make "torch.cuda" point to NPU devices
    torch.cuda.CUDAGraph = torch.npu.NPUGraph  # type: ignore[misc, attr-defined]
    # Allow using NpuFractalNzTensor and NpuFractalZnTensor
    torch_npu.npu.config.allow_internal_format = True
    # Setup paths to op libraries
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.path.join(
        site.getsitepackages()[0], "vendors", "customize"
    )
    return torch_npu


def _load_torch_gcu() -> Any:
    import torch_gcu
    import torch_gcu.transfer_to_gcu  # noqa: F401

    # The GCU Triton backend is optional; only its absence may be tolerated.
    try:
        import triton_gcu.triton  # noqa: F401
    except ImportError:
        pass
    return torch_gcu


# ---------------------------------------------------------------------------
# Dependencies that cannot be handled by a plain import.
# ---------------------------------------------------------------------------


def _load_hygon_mxfp4_decode() -> Any:
    import importlib.util

    # We only need to know whether it is installed; do not load the .so here.
    if importlib.util.find_spec("hygon_mxfp4_decode") is None:
        return _MissingModule(
            "hygon_mxfp4_decode",
            "hygon_mxfp4_decode",
            ImportError("find_spec returned None"),
        )
    import hygon_mxfp4_decode

    return hygon_mxfp4_decode


# ---------------------------------------------------------------------------
# Vendor extensions that must be loaded by hand rather than imported.
# ---------------------------------------------------------------------------

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
    import glob
    import importlib.util
    import os
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
                assert cspec is not None and cspec.loader is not None
                mod = importlib.util.module_from_spec(cspec)
                cspec.loader.exec_module(mod)
            finally:
                sys.setdlopenflags(old_flags)
            break
    _gcu_c_ops_loaded = True


def __getattr__(name: str) -> Any:
    if name not in _cache:
        loader = globals().get(f"_load_{name}")
        if loader is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        try:
            _cache[name] = loader()
        except ModuleNotFoundError as e:
            expected = _PATHS.get(name, name)
            # A dependency is missing only if the target module itself (or one
            # of its parents) does not exist. If the package exists but one of
            # its own imports fails (e.name is some other package), the
            # environment is broken; propagate instead of pretending the
            # dependency is not installed.
            if not (e.name == expected or expected.startswith(f"{e.name}.")):
                raise
            _cache[name] = _MissingModule(expected, _EXTRAS.get(name), e)
    return _cache[name]


def __dir__() -> list[str]:
    return sorted(n[len("_load_") :] for n in globals() if n.startswith("_load_"))
