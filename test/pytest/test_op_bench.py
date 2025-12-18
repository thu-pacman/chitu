# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Op benchmark tests integrated with pytest.

These tests verify that op benchmarks can run without errors. By default they
also print benchmark timing tables for quick perf sanity checks.
"""

import importlib.util
import os
import sys

# Add project root to path for benchmarks module
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
import torch

from chitu.device_type import has_native_fp8, is_hopper, is_nvidia, is_ascend
from chitu.utils import try_import_opt_dep, try_import_and_setup_torch_npu

# Check optional dependencies
cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")
flashinfer, has_flashinfer = try_import_opt_dep("flashinfer", "flashinfer")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


# Provider availability checks (used for line_available in benchmarks)
def _torch_available() -> bool:
    """Torch provider is always available."""
    return True


def _triton_available() -> bool:
    return importlib.util.find_spec("triton") is not None


def _flashinfer_available() -> bool:
    return has_flashinfer


def _cuda_backend_available() -> bool:
    """CUDA backend provider requires chitu_backend and CUDA."""
    return (
        importlib.util.find_spec("chitu_backend") is not None
        and torch.cuda.is_available()
    )


def _has_supported_device() -> bool:
    """Check if CUDA or NPU is available."""
    if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)):
        if torch.npu.is_available():
            return True
    return torch.cuda.is_available()


def _is_nvidia_gpu() -> bool:
    """Check if running on NVIDIA GPU."""
    if not torch.cuda.is_available():
        return False
    return is_nvidia()


def _has_fp8_support() -> bool:
    """Check if device supports fp8 (Hopper+)."""
    if not torch.cuda.is_available():
        return False
    return has_native_fp8() or is_hopper()


def _has_graph_bench_support() -> bool:
    """Check if do_bench_graph can run on the current CUDA/NPU backend."""
    use_npu = (
        hasattr(torch, "npu")
        and callable(getattr(torch.npu, "is_available", None))
        and torch.npu.is_available()
    )
    if use_npu:
        backend = torch.npu
        graph_cls = getattr(backend, "NPUGraph", None)
        graph_ctx = getattr(backend, "graph", None)
        if graph_ctx is None:
            graph_ctx = getattr(torch.cuda, "graph", None)
            graph_cls = getattr(torch.cuda, "CUDAGraph", None) or graph_cls
    else:
        if not torch.cuda.is_available():
            return False
        backend = torch.cuda
        graph_cls = getattr(backend, "CUDAGraph", None)
        graph_ctx = getattr(backend, "graph", None)

    if graph_cls is None or graph_ctx is None:
        return False

    # Minimal APIs used by bench_util.do_bench_graph
    required_backend_attrs = ["Event", "current_stream", "synchronize"]
    return all(
        getattr(backend, name, None) is not None for name in required_backend_attrs
    )


@pytest.mark.skipif(
    not _has_graph_bench_support(),
    reason="do_bench_graph requires CUDA/NPU with Graph + Event support",
)
def test_op_bench_do_bench_graph(record_property):
    """Run a real Graph timed benchmark and validate timing/output on CUDA or NPU."""
    from benchmarks.op_bench.bench_util import do_bench_graph

    import math

    use_npu = (
        hasattr(torch, "npu")
        and callable(getattr(torch.npu, "is_available", None))
        and torch.npu.is_available()
    )
    device = torch.device("npu") if use_npu else torch.device("cuda")
    backend = torch.npu if use_npu else torch.cuda
    backend_name = "npu" if use_npu else "cuda"

    # Use a simple, graph-capturable op with fixed shapes to support both CUDA and NPU.
    a = torch.randn((2048, 2048), device=device, dtype=torch.float32)
    b = torch.randn((2048, 2048), device=device, dtype=torch.float32)
    out = torch.empty_like(a)

    def fn():
        torch.add(a, b, out=out)

    try:
        times_ms = do_bench_graph(fn, rep=1, return_mode="all")
    except RuntimeError as e:
        msg = str(e)
        if (
            "do_bench_graph requires" in msg
            or "Graph support" in msg
            or "does not support replay" in msg
        ):
            pytest.skip(msg)
        raise

    assert isinstance(times_ms, list)
    assert len(times_ms) == 10  # n_retries in bench_util.do_bench_graph
    assert all(isinstance(t, float) and math.isfinite(t) and t >= 0.0 for t in times_ms)

    t_min = min(times_ms)
    t_max = max(times_ms)
    t_mean = sum(times_ms) / len(times_ms)
    record_property(f"do_bench_graph_{backend_name}_torch_add_mean_ms", t_mean)
    record_property(f"do_bench_graph_{backend_name}_torch_add_min_ms", t_min)
    record_property(f"do_bench_graph_{backend_name}_torch_add_max_ms", t_max)
    print(
        f"[do_bench_graph/{backend_name}] torch.add: "
        f"mean={t_mean:.6f} ms, min={t_min:.6f} ms, max={t_max:.6f} ms, samples={times_ms}"
    )

    # Validate correctness of the op result after Graph capture + replay.
    backend.synchronize()
    torch.testing.assert_close(out, a + b)


# FP4 Benchmarks - require Hopper+ for fp8 support


@pytest.mark.skipif(not _has_fp8_support(), reason="FP4 requires Hopper+ GPU with fp8")
def test_op_bench_fp4_bf16():
    from benchmarks.op_bench.bench_fp4 import benchmark_fp4_raise_to_bf16_gemm

    benchmark_fp4_raise_to_bf16_gemm.run(
        show_plots=False, print_data=True, save_path=""
    )


@pytest.mark.skipif(not _has_fp8_support(), reason="FP4 requires Hopper+ GPU with fp8")
def test_op_bench_fp4_fp8():
    from benchmarks.op_bench.bench_fp4 import benchmark_fp4_raise_to_fp8_gemm

    benchmark_fp4_raise_to_fp8_gemm.run(show_plots=False, print_data=True, save_path="")


# FP8 Benchmarks - require Hopper+ for fp8 support


@pytest.mark.skipif(not _has_fp8_support(), reason="FP8 requires Hopper+ GPU")
def test_op_bench_fp8_gemm():
    from benchmarks.op_bench.bench_fp8 import benchmark_fp8_gemm

    benchmark_fp8_gemm.run(show_plots=False, print_data=True, save_path="")


@pytest.mark.skipif(not _has_fp8_support(), reason="FP8 requires Hopper+ GPU")
def test_op_bench_soft_fp8_gemm():
    from benchmarks.op_bench.bench_fp8 import benchmark_soft_fp8_gemm

    benchmark_soft_fp8_gemm.run(show_plots=False, print_data=True, save_path="")


@pytest.mark.skipif(not _has_fp8_support(), reason="FP8 requires Hopper+ GPU")
def test_op_bench_fp8_group_gemm():
    from benchmarks.op_bench.bench_fp8_group_gemm import (
        benchmark_blockfp8_einsum_shc_hdc_shd,
    )

    benchmark_blockfp8_einsum_shc_hdc_shd.run(
        show_plots=False, print_data=True, save_path=""
    )


@pytest.mark.skipif(
    not _has_supported_device(), reason="No CUDA/NPU available for op_bench"
)
def test_op_bench_frequency_penalty():
    from benchmarks.op_bench.bench_frequency_penalty import bench_frequency_penalty

    # line_vals=["torch", "triton", "cuda"]
    bench_frequency_penalty.run(
        show_plots=False,
        print_data=True,
        save_path="",
        line_available=[_torch_available, _triton_available, _cuda_backend_available],
    )


@pytest.mark.skipif(
    not _has_supported_device(), reason="No CUDA/NPU available for op_bench"
)
def test_op_bench_moe_fuse_gate():
    from benchmarks.op_bench.bench_moe_fuse_gate import benchmark as bench_moe_fuse_gate

    bench_moe_fuse_gate.run(show_plots=False, print_data=True, save_path="")


@pytest.mark.skipif(
    not _has_supported_device(), reason="No CUDA/NPU available for op_bench"
)
def test_op_bench_moe_sum():
    from benchmarks.op_bench.bench_moe_sum import benchmark as bench_moe_sum

    # line_vals=["torch", "triton"]
    bench_moe_sum.run(
        show_plots=False,
        print_data=True,
        save_path="",
        line_available=[_torch_available, _triton_available],
    )


@pytest.mark.skipif(
    not _has_supported_device(), reason="No CUDA/NPU available for op_bench"
)
def test_op_bench_rms_norm():
    from benchmarks.op_bench.bench_rms_norm import benchmark as bench_rms_norm

    # line_vals=["triton", "torch"]
    bench_rms_norm.run(
        show_plots=False,
        print_data=True,
        save_path="",
        line_available=[_triton_available, _torch_available],
    )


@pytest.mark.skipif(
    not _has_supported_device(), reason="No CUDA/NPU available for op_bench"
)
def test_op_bench_rotary():
    from benchmarks.op_bench.bench_rotary import benchmark as bench_rotary

    # line_vals=["torch", "triton", "cuda"]
    bench_rotary.run(
        show_plots=False,
        print_data=True,
        save_path="",
        line_available=[_torch_available, _triton_available, _cuda_backend_available],
    )


@pytest.mark.skipif(
    not _has_supported_device(), reason="No CUDA/NPU available for op_bench"
)
def test_op_bench_silu_and_mul():
    from benchmarks.op_bench.bench_silu_and_mul import benchmark as bench_silu_and_mul

    # line_vals=["triton", "torch"]
    bench_silu_and_mul.run(
        show_plots=False,
        print_data=True,
        save_path="",
        line_available=[_triton_available, _torch_available],
    )


@pytest.mark.skipif(
    not _is_nvidia_gpu(),
    reason="Attention benchmark requires NVIDIA GPU with triton/flashinfer",
)
def test_op_bench_attn():
    from benchmarks.op_bench.bench_attn import (
        benchmark_mla_decode_paged_kv,
        benchmark_prefill_ragged_qkvo,
    )

    line_available = [_triton_available, _flashinfer_available]

    benchmark_mla_decode_paged_kv.run(
        show_plots=False,
        print_data=True,
        save_path="",
        line_available=line_available,
    )
    benchmark_prefill_ragged_qkvo.run(
        show_plots=False,
        print_data=True,
        save_path="",
        line_available=line_available,
    )


@pytest.mark.skipif(not has_cpuinfer, reason="cpuinfer module not available")
def test_op_bench_cpuinfer_linear():
    from benchmarks.op_bench.bench_cpuinfer_linear import benchmark as bench_linear

    bench_linear.run(show_plots=False, print_data=True, save_path="")


@pytest.mark.skipif(not has_cpuinfer, reason="cpuinfer module not available")
def test_op_bench_cpuinfer_moe_gate():
    from benchmarks.op_bench.bench_cpuinfer_moe_gate import benchmark as bench_moe_gate

    bench_moe_gate.run(show_plots=False, print_data=True, save_path="")


@pytest.mark.skipif(not has_cpuinfer, reason="cpuinfer module not available")
def test_op_bench_cpuinfer_rmsnorm():
    from benchmarks.op_bench.bench_cpuinfer_rmsnorm import benchmark as bench_rmsnorm

    bench_rmsnorm.run(show_plots=False, print_data=True, save_path="")


@pytest.mark.skipif(not has_cpuinfer, reason="cpuinfer module not available")
def test_op_bench_cpuinfer_silu_mul():
    from benchmarks.op_bench.bench_cpuinfer_silu_mul import benchmark as bench_silu_mul

    bench_silu_mul.run(show_plots=False, print_data=True, save_path="")
