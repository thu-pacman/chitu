# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Checked binding for the experimental TCLE weighted-row-sum kernel."""

from __future__ import annotations

import ctypes
import os
import threading

import torch


_ABI = 0x5752530100308010
_launch = None
_library_path = None
_load_lock = threading.Lock()


def _ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _get_launch():
    global _launch, _library_path
    configured_path = os.environ.get("CHITU_GCU_FUSED_WEIGHTED_ROW_SUM_LIB")
    if not configured_path:
        raise RuntimeError("CHITU_GCU_FUSED_WEIGHTED_ROW_SUM_LIB must name the GCU library")
    path = os.path.realpath(configured_path)
    if _launch is not None:
        if _library_path != path:
            raise RuntimeError(
                "weighted-row-sum library cannot change after loading; restart the process: "
                f"loaded={_library_path} requested={path}"
            )
        return _launch

    with _load_lock:
        if _launch is not None:
            if _library_path != path:
                raise RuntimeError(
                    "weighted-row-sum library changed during initialization: "
                    f"loaded={_library_path} requested={path}"
                )
            return _launch
        library = ctypes.CDLL(path)
        abi = library.fused_weighted_row_sum_abi
        abi.argtypes = []
        abi.restype = ctypes.c_ulonglong
        actual_abi = int(abi())
        if actual_abi != _ABI:
            raise RuntimeError(
                f"weighted-row-sum ABI mismatch: 0x{actual_abi:016x} != 0x{_ABI:016x}"
            )
        launch = library.launch_fused_weighted_row_sum
        launch.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        launch.restype = ctypes.c_int
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


def fused_weighted_row_sum(state: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return ``sum_K(state * repeat3(weight)[..., None])`` for decode."""
    batch = state.shape[0] if state.ndim == 4 else 0
    if (
        batch not in (2, 4, 8)
        or tuple(state.shape) != (batch, 48, 128, 128)
        or tuple(weight.shape) != (batch, 16, 128)
        or state.dtype != torch.float32
        or weight.dtype != torch.float32
        or state.device.type != "gcu"
        or state.device != weight.device
        or not state.is_contiguous()
        or not weight.is_contiguous()
        or not _aligned_rows(state, 512)
        or not _aligned_rows(weight, 512)
    ):
        raise ValueError("unsupported fused weighted-row-sum contract")

    output = torch.empty((batch, 48, 128), dtype=torch.float32, device=state.device)
    if not _aligned_rows(output, 512):
        raise RuntimeError("GCU allocator returned an unaligned weighted-row-sum output")
    launch = _get_launch()
    with torch.gcu.device(state.device):
        stream = torch.gcu.current_stream(state.device)
        status = launch(
            _ptr(state),
            _ptr(weight),
            _ptr(output),
            batch,
            48,
            16,
            ctypes.c_void_p(stream.gcu_stream),
        )
    if status:
        raise RuntimeError(f"weighted-row-sum launch failed: status={status}")
    return output
