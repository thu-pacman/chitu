# SPDX-License-Identifier: Apache-2.0
"""Checked binding for the exact fused one-step recurrent TCLE kernel."""

from __future__ import annotations

import ctypes
import os
import threading

import torch


_ABI = 0x4652575501308011
_launch = None
_library_path = None
_load_lock = threading.Lock()


def _ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _get_launch():
    global _launch, _library_path
    configured = os.environ.get("CHITU_GCU_FUSED_RECURRENT_WRS_UPDATE_LIB")
    if not configured:
        raise RuntimeError(
            "CHITU_GCU_FUSED_RECURRENT_WRS_UPDATE_LIB must name the GCU library"
        )
    path = os.path.realpath(configured)
    if _launch is not None:
        if _library_path != path:
            raise RuntimeError("fused recurrent library cannot change after loading")
        return _launch
    with _load_lock:
        if _launch is not None:
            return _launch
        library = ctypes.CDLL(path)
        abi = library.fused_recurrent_wrs_update_abi
        abi.argtypes = []
        abi.restype = ctypes.c_ulonglong
        actual = int(abi())
        if actual != _ABI:
            raise RuntimeError(f"fused recurrent ABI mismatch: 0x{actual:016x}")
        launch = library.launch_fused_recurrent_wrs_update
        launch.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        launch.restype = ctypes.c_int
        launch._chitu_library = library
        _launch = launch
        _library_path = path
    return _launch


def _aligned_rows(tensor: torch.Tensor, alignment: int = 512) -> bool:
    return (
        tensor.data_ptr() % alignment == 0
        and tensor.stride(0) * tensor.element_size() % alignment == 0
    )


def fused_recurrent_wrs_update(
    state: torch.Tensor,
    key: torch.Tensor,
    query: torch.Tensor,
    value: torch.Tensor,
    raw_beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = state.shape[0] if state.ndim == 4 else 0
    tensors = (state, key, query, value, raw_beta)
    if (
        batch not in (2, 4, 8)
        or tuple(state.shape) != (batch, 48, 128, 128)
        or tuple(key.shape) != (batch, 16, 128)
        or tuple(query.shape) != (batch, 16, 128)
        or tuple(value.shape) != (batch, 48, 128)
        or tuple(raw_beta.shape) != (batch, 48)
        or state.dtype != torch.float32
        or key.dtype != torch.float32
        or query.dtype != torch.float32
        or value.dtype != torch.bfloat16
        or raw_beta.dtype != torch.bfloat16
        or any(t.device.type != "gcu" or t.device != state.device for t in tensors)
        or not state.is_contiguous()
        or not key.is_contiguous()
        or not query.is_contiguous()
        or value.stride(-1) != 1
        or value.stride(-2) != 128
        or raw_beta.stride(-1) != 1
        or any(not _aligned_rows(t) for t in (state, key, query, value))
    ):
        raise ValueError("unsupported fused recurrent WRS/update contract")

    output = torch.empty((batch, 48, 128), dtype=torch.float32, device=state.device)
    next_state = torch.empty_like(state)
    if not _aligned_rows(output) or not _aligned_rows(next_state):
        raise RuntimeError("GCU allocator returned an unaligned fused recurrent output")
    launch = _get_launch()
    with torch.gcu.device(state.device):
        stream = torch.gcu.current_stream(state.device)
        status = launch(
            *(_ptr(t) for t in tensors),
            _ptr(output),
            _ptr(next_state),
            batch,
            48,
            16,
            value.stride(0),
            raw_beta.stride(0),
            ctypes.c_void_p(stream.gcu_stream),
        )
    if status:
        raise RuntimeError(f"fused recurrent WRS/update launch failed: status={status}")
    return output, next_state
