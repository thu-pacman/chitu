# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Checked binding for the optional exact GCU raw-beta delta kernel."""

from __future__ import annotations

import ctypes
import os
import threading

import torch


_ABI = 0x4744424101023080
_INT32_MAX = (1 << 31) - 1
_launch = None
_library_path = None
_load_lock = threading.Lock()


def _ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _get_launch():
    """Load and freeze the opt-in native launcher for the process lifetime."""
    global _launch, _library_path
    configured_path = os.environ.get("QWEN38_RAW_BETA_DELTA_BATCH_LIB")
    if not configured_path:
        raise RuntimeError("QWEN38_RAW_BETA_DELTA_BATCH_LIB must name the GCU library")
    path = os.path.realpath(configured_path)
    if _launch is not None:
        if _library_path != path:
            raise RuntimeError(
                "raw-beta delta library cannot change after loading; restart the "
                f"process: loaded={_library_path} requested={path}"
            )
        return _launch

    with _load_lock:
        if _launch is not None:
            if _library_path != path:
                raise RuntimeError(
                    "raw-beta delta library changed during initialization: "
                    f"loaded={_library_path} requested={path}"
                )
            return _launch
        library = ctypes.CDLL(path)
        abi = library.raw_beta_delta_batch_abi
        abi.argtypes = []
        abi.restype = ctypes.c_ulonglong
        actual_abi = int(abi())
        if actual_abi != _ABI:
            raise RuntimeError(
                f"raw-beta delta ABI mismatch: 0x{actual_abi:016x} != 0x{_ABI:016x}"
            )
        launch = library.launch_raw_beta_delta_batch
        launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        launch.restype = ctypes.c_int
        # A captured Graph can retain native code after Python returns. Keep the
        # CDLL and its canonical path alive with the callable for the process.
        launch._chitu_library = library
        launch._chitu_path = path
        _launch = launch
        _library_path = path
    return _launch


def _aligned_rows(tensor: torch.Tensor, alignment: int) -> bool:
    return (
        tensor.data_ptr() % alignment == 0
        and tensor.stride(0) * tensor.element_size() % alignment == 0
    )


def raw_beta_delta(
    value: torch.Tensor, kv: torch.Tensor, raw_beta: torch.Tensor
) -> torch.Tensor:
    """Return exact FP32 ``(value.float() - kv) * sigmoid(raw_beta).float()``.

    This is deliberately a narrow Qwen3.8 decode contract. Unsupported shapes
    fail before a native launch so callers can keep the generic implementation.
    """
    batch = value.shape[0] if value.ndim == 3 else 0
    strides = (
        value.stride(0) if value.ndim == 3 else 0,
        kv.stride(0) if kv.ndim == 3 else 0,
        raw_beta.stride(0) if raw_beta.ndim == 2 else 0,
    )
    if (
        batch not in (2, 4, 8)
        or tuple(value.shape) != (batch, 48, 128)
        or tuple(kv.shape) != (batch, 48, 128)
        or tuple(raw_beta.shape) != (batch, 48)
        or value.dtype != torch.bfloat16
        or kv.dtype != torch.float32
        or raw_beta.dtype != torch.bfloat16
        or value.stride(-1) != 1
        or value.stride(-2) != 128
        or kv.stride(-1) != 1
        or kv.stride(-2) != 128
        or raw_beta.stride(-1) != 1
        or strides[0] < 48 * 128
        or strides[1] < 48 * 128
        or strides[2] < 48
        or not all(0 < stride <= _INT32_MAX for stride in strides)
        # BF16 vector loads cover 128 lanes (256 B); FP32 vector loads/stores
        # cover 128 lanes (512 B). Check every batch-row origin, not only base.
        or not _aligned_rows(value, 256)
        or not _aligned_rows(kv, 512)
        or len({value.device, kv.device, raw_beta.device}) != 1
        or value.device.type != "gcu"
    ):
        raise ValueError("unsupported raw-beta delta batch contract")

    delta = torch.empty(kv.shape, dtype=kv.dtype, device=kv.device)
    if not _aligned_rows(delta, 512):
        raise RuntimeError("GCU allocator returned an unaligned raw-beta delta output")
    launch = _get_launch()
    with torch.gcu.device(value.device):
        stream = torch.gcu.current_stream(value.device)
        status = launch(
            _ptr(value),
            _ptr(kv),
            _ptr(raw_beta),
            _ptr(delta),
            batch,
            strides[0],
            strides[1],
            strides[2],
            delta.stride(0),
            ctypes.c_void_p(stream.gcu_stream),
        )
    if status:
        raise RuntimeError(f"raw-beta delta batch launch failed: status={status}")
    return delta
