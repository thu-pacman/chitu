# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch


def _require_cuda_bf16() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Qwen3-VL patch embed test")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("This CUDA device does not support bfloat16")


@torch.inference_mode()
def test_qwen3_vl_conv3d_small_shape() -> None:
    _require_cuda_bf16()

    conv = torch.nn.Conv3d(
        in_channels=3,
        out_channels=1152,
        kernel_size=(2, 16, 16),
        stride=(2, 16, 16),
        bias=True,
    ).to(device="cuda", dtype=torch.bfloat16)

    # [N, 1536] where 1536 = 3 * 2 * 16 * 16
    print("Testing conv3d with Small shape input...")
    x = torch.randn((128, 1536), device="cuda", dtype=torch.bfloat16)
    y = conv(x.view(-1, 3, 2, 16, 16).to(dtype=conv.weight.dtype)).view(-1, 1152)

    assert y.shape == (128, 1152)
    assert y.dtype == conv.weight.dtype
    assert torch.isfinite(y).all().item()


@torch.inference_mode()
def test_qwen3_vl_conv3d_large_repro_shape() -> None:
    _require_cuda_bf16()

    conv = torch.nn.Conv3d(
        in_channels=3,
        out_channels=1152,
        kernel_size=(2, 16, 16),
        stride=(2, 16, 16),
        bias=True,
    ).to(device="cuda", dtype=torch.bfloat16)

    # Reproduce the reported runtime shape exactly:
    # pixel_values: [36720, 1536] -> conv3d input [36720, 3, 2, 16, 16]
    print("Testing conv3d with Large shape input for reproduction...")
    x = torch.randn((36720, 1536), device="cuda", dtype=torch.bfloat16)
    y = conv(x.view(-1, 3, 2, 16, 16).to(dtype=conv.weight.dtype)).view(-1, 1152)

    assert y.shape == (36720, 1152)
    assert y.dtype == conv.weight.dtype
    assert torch.isfinite(y).all().item()
