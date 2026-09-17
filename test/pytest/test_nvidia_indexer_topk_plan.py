# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Compile the production host planner without CUDA to test simulated SM counts."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from chitu.device_type import is_nvidia


@pytest.mark.skipif(
    not is_nvidia(),
    reason="NVIDIA planner header is shipped only in NVIDIA test images",
)
def test_nvidia_topk_sm_scaled_grid_matches_h20_and_other_sm_counts(tmp_path):
    # CXX and PATH may select a device compiler wrapper (for example, mxcc).
    # This test needs only the system host C++ compiler, not the GPU toolchain.
    compiler = shutil.which("c++", path=os.defpath)
    if compiler is None:
        pytest.skip("host planner test requires a C++ compiler")
    source = Path(__file__).with_name("nvidia_indexer_topk_plan_test.cc")
    include = Path(__file__).resolve().parents[2] / "csrc" / "cuda" / "topk"
    binary = tmp_path / "nvidia_topk_plan_test"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-D__host__=",
            "-D__device__=",
            "-D__constant__=",
            "-I",
            str(include),
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "H20 equivalence and 8 SM counts" in result.stdout
