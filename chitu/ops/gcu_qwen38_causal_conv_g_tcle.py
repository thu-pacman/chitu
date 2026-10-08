# SPDX-FileCopyrightText: 2025 Qingcheng.AI
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed binding for fixed-shape Qwen3.8 TCLE conv + fused-g."""

from __future__ import annotations

import ctypes
import os
import threading

import torch


_ABI = 0x434F4E5600280402
_launch = None
_library_path = None
_load_lock = threading.Lock()


def _ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _get_launch():
    global _launch, _library_path
    configured_path = os.environ.get("QWEN38_CAUSAL_CONV_G_TCLE_LIB")
    if not configured_path:
        raise RuntimeError("QWEN38_CAUSAL_CONV_G_TCLE_LIB must name the TCLE library")
    path = os.path.realpath(configured_path)
    if _launch is not None:
        if _library_path != path:
            raise RuntimeError(
                "TCLE conv+g library cannot change after loading; restart the process: "
                f"loaded={_library_path} requested={path}"
            )
        return _launch
    with _load_lock:
        if _launch is not None:
            return _launch
        library = ctypes.CDLL(path)
        abi = library.qwen38_causal_conv_update_abi
        abi.argtypes = []
        abi.restype = ctypes.c_ulonglong
        actual_abi = int(abi())
        if actual_abi != _ABI:
            raise RuntimeError(
                f"TCLE conv+g ABI mismatch: 0x{actual_abi:016x} != 0x{_ABI:016x}"
            )
        launch = library.launch_qwen38_causal_conv_g_update
        launch.argtypes = (
            [ctypes.c_void_p] * 8
            + [ctypes.c_int]
            + [ctypes.c_longlong] * 3
            + [ctypes.c_void_p]
        )
        launch.restype = ctypes.c_int
        launch._chitu_library = library
        launch._chitu_path = path
        _launch = launch
        _library_path = path
    return _launch


def _aligned_rows(tensor: torch.Tensor, alignment: int = 512) -> bool:
    return (
        tensor.data_ptr() % alignment == 0
        and tensor.stride(0) * tensor.element_size() % alignment == 0
    )


def qwen38_causal_conv_g_update_tcle(
    this_hidden_states: torch.Tensor,
    old_hidden_states: torch.Tensor,
    weight: torch.Tensor,
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
):
    batch = this_hidden_states.shape[0] if this_hidden_states.ndim == 2 else 0
    tensors = (
        this_hidden_states,
        old_hidden_states,
        weight,
        a,
        A_log,
        dt_bias,
    )
    if (
        batch not in (2, 4, 8)
        or tuple(this_hidden_states.shape) != (batch, 10240)
        or tuple(old_hidden_states.shape) != (batch, 10240, 4)
        or tuple(weight.shape) != (10240, 1, 4)
        or tuple(a.shape) != (batch, 48)
        or tuple(A_log.shape) != (48,)
        or tuple(dt_bias.shape) != (48,)
        or this_hidden_states.dtype != torch.bfloat16
        or old_hidden_states.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or a.dtype != torch.bfloat16
        or A_log.dtype != torch.bfloat16
        or dt_bias.dtype != torch.bfloat16
        or any(tensor.device.type != "gcu" for tensor in tensors)
        or len({tensor.device for tensor in tensors}) != 1
        or this_hidden_states.stride(1) != 1
        or a.stride(1) != 1
        or not old_hidden_states.is_contiguous()
        or not weight.is_contiguous()
        or not A_log.is_contiguous()
        or not dt_bias.is_contiguous()
        or not _aligned_rows(this_hidden_states)
        or not _aligned_rows(old_hidden_states)
        or weight.data_ptr() % 512 != 0
    ):
        raise ValueError("unsupported Qwen3.8 TCLE causal-conv+g contract")

    output = torch.empty_like(this_hidden_states, memory_format=torch.contiguous_format)
    g = torch.empty_like(a, dtype=torch.float32, memory_format=torch.contiguous_format)
    if not _aligned_rows(output):
        raise RuntimeError("GCU allocator returned an unaligned TCLE conv output")
    launch = _get_launch()
    with torch.gcu.device(this_hidden_states.device):
        current_stream = torch.gcu.current_stream(this_hidden_states.device)
        status = launch(
            _ptr(this_hidden_states),
            _ptr(old_hidden_states),
            _ptr(weight),
            _ptr(output),
            _ptr(a),
            _ptr(A_log),
            _ptr(dt_bias),
            _ptr(g),
            batch,
            this_hidden_states.stride(0),
            a.stride(0),
            g.stride(0),
            ctypes.c_void_p(current_stream.gcu_stream),
        )
    if status:
        raise RuntimeError(f"TCLE conv+g launch failed: status={status}")
    return output, old_hidden_states, g
