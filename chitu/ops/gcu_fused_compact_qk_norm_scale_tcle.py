# SPDX-License-Identifier: Apache-2.0
"""Checked binding for exact strided compact Q/K norm-scale on GCU400."""

from __future__ import annotations

import ctypes
import os
import threading

import torch


_ABI = 0x46514B4E01308011
_launch = None
_library_path = None
_load_lock = threading.Lock()


def _ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _get_launch():
    global _launch, _library_path
    configured = os.environ.get("QWEN38_FUSED_COMPACT_QK_NORM_SCALE_LIB")
    if not configured:
        raise RuntimeError(
            "QWEN38_FUSED_COMPACT_QK_NORM_SCALE_LIB must name the GCU library"
        )
    path = os.path.realpath(configured)
    if _launch is not None:
        if _library_path != path:
            raise RuntimeError("fused compact Q/K library cannot change after loading")
        return _launch
    with _load_lock:
        if _launch is not None:
            return _launch
        library = ctypes.CDLL(path)
        abi = library.fused_compact_qk_norm_scale_abi
        abi.argtypes = []
        abi.restype = ctypes.c_ulonglong
        actual = int(abi())
        if actual != _ABI:
            raise RuntimeError(f"fused compact Q/K ABI mismatch: 0x{actual:016x}")
        launch = library.launch_fused_compact_qk_norm_scale
        launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_void_p]
        launch.restype = ctypes.c_int
        launch._chitu_library = library
        _launch = launch
        _library_path = path
    return _launch


def fused_compact_qk_norm_scale(
    qkv: torch.Tensor,
    query_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = qkv.shape[0] if qkv.ndim == 2 else 0
    if (
        batch not in (2, 4, 8)
        or tuple(qkv.shape) != (batch, 10240)
        or tuple(qkv.stride()) != (10240, 1)
        or tuple(query_scale.shape) != (1,)
        or qkv.dtype != torch.bfloat16
        or query_scale.dtype != torch.float32
        or qkv.device.type != "gcu"
        or query_scale.device != qkv.device
        or not qkv.is_contiguous()
        or not query_scale.is_contiguous()
        or qkv.data_ptr() % 512 != 0
    ):
        raise ValueError("unsupported fused compact Q/K norm-scale contract")

    query_out = torch.empty((batch, 16, 128), dtype=torch.float32, device=qkv.device)
    key_out = torch.empty_like(query_out)
    if query_out.data_ptr() % 512 or key_out.data_ptr() % 512:
        raise RuntimeError("GCU allocator returned an unaligned compact Q/K output")
    launch = _get_launch()
    with torch.gcu.device(qkv.device):
        stream = torch.gcu.current_stream(qkv.device)
        status = launch(
            _ptr(qkv),
            _ptr(query_scale),
            _ptr(query_out),
            _ptr(key_out),
            batch,
            ctypes.c_void_p(stream.gcu_stream),
        )
    if status:
        raise RuntimeError(f"fused compact Q/K launch failed: status={status}")
    return query_out, key_out
