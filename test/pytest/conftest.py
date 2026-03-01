# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for cinfer tests.

Provides automatic benchmark capability for any test:
- Benchmarks are always enabled
- Use `record_benchmark` fixture in any test to record performance
- Control timing precision with `--warmup-round` and `--timing-round`
  (default: 0 and 1 for CI, increase for accurate performance testing)

Usage in tests (three methods):

Method A1 - Single impl benchmark (separate correctness and benchmark):
    def test_my_op(M, N, impl, record_benchmark):
        x = torch.randn(M, N, device="cuda")
        result = my_op(x, impl=impl)
        assert result == expected
        record_benchmark(lambda: my_op(x, impl=impl), N=N, impl=impl)

Method A2 - Compare multiple impls (recommended for multi-impl comparison):
    def test_my_op(M, N, record_benchmark):
        x = torch.randn(M, N, device="cuda")
        result = my_op(x, impl="triton")
        assert result == expected

        # Compare all implementations at once
        record_benchmark(
            N=N,
            impls={
                "triton": lambda: my_op(x, impl="triton"),
                "torch": lambda: my_op(x, impl="torch"),
                "cuda": lambda: my_op(x, impl="cuda"),
            }
        )

Method B - Combined correctness and benchmark (runs once, returns result):
    def test_my_op(M, N, impl, record_benchmark):
        x = torch.randn(M, N, device="cuda")
        # Run once, get result AND record timing
        result = record_benchmark.run(
            lambda: my_op(x, impl=impl),
            N=N,
            impl=impl,
        )
        assert result == expected

Run with more rounds for accurate performance testing:
    pytest test.py --warmup-round=5 --timing-round=20 -s
"""

import asyncio
import os
import sys
import math
import atexit
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import pytest
import torch
from omegaconf import OmegaConf

import chitu.global_vars as global_vars
from chitu.global_vars import set_global_args
from chitu.distributed.tcp_ip import get_free_port
from chitu.distributed.parallel_state import (
    initialize_parallel_groups,
    parallel_groups_initialized,
    destroy_parallel_groups,
)
from chitu.distributed.pd_disaggregation.pd_coordination import PDCoordinationService
from chitu.distributed.pd_disaggregation.kv_transfer.mooncake.transfer_engine import (
    MooncakeBootstrapServer,
)

# Add project root to path (required for correct module imports in pytest)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--warmup-round",
        type=int,
        default=0,
        help="Number of warmup rounds for benchmarks (default: 0 for CI)",
    )
    parser.addoption(
        "--timing-round",
        type=int,
        default=1,
        help="Number of timing rounds for benchmarks (default: 1 for CI)",
    )


# Global storage for benchmark results
_benchmark_data: Dict[str, Dict[str, List[tuple]]] = defaultdict(
    lambda: defaultdict(list)
)
_warmup_round = 0
_timing_round = 1


def _format_table(columns: List[str], rows: List[List[Any]], float_precision: int = 6):
    """Format and print a table."""

    def _format_cell(val: Any) -> str:
        if val is None or val == "":
            return ""
        if isinstance(val, str):
            return val
        if isinstance(val, (int, float)):
            if isinstance(val, float) and math.isfinite(val):
                if abs(val - round(val)) < 1e-9:
                    return f"{val:.1f}"
                return f"{val:.{float_precision}f}"
            return str(val)
        return str(val)

    cell_rows = [[_format_cell(v) for v in row] for row in rows]
    col_widths = []
    for j, col in enumerate(columns):
        max_cell = max((len(r[j]) for r in cell_rows), default=0)
        col_widths.append(max(len(col), max_cell))

    idx_width = len(str(len(rows) - 1)) if rows else 0
    header_prefix = " " * (idx_width + 1) if idx_width else ""
    header_line = header_prefix + " ".join(
        col.rjust(w) for col, w in zip(columns, col_widths)
    )
    print(header_line)

    for i, r in enumerate(cell_rows):
        idx = str(i).rjust(idx_width) + " "
        print(idx + " ".join(cell.rjust(w) for cell, w in zip(r, col_widths)))


def _print_benchmark_results():
    """Print collected benchmark results at the end of the session."""
    if not _benchmark_data:
        return

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    for test_name, impl_data in sorted(_benchmark_data.items()):
        print(f"\n.{test_name}-performance:")

        # Collect all unique x values and impls
        all_x_vals = set()
        all_impls = set()
        x_param_name = None

        for impl, measurements in impl_data.items():
            all_impls.add(impl)
            for x_val, x_name, _ in measurements:
                all_x_vals.add(x_val)
                if x_param_name is None:
                    x_param_name = x_name

        if x_param_name is None:
            x_param_name = "x"

        # Sort values
        try:
            sorted_x_vals = sorted(
                all_x_vals, key=lambda x: float(x) if isinstance(x, (int, float)) else x
            )
        except (TypeError, ValueError):
            sorted_x_vals = sorted(all_x_vals, key=str)
        sorted_impls = sorted(all_impls)

        # Build lookup table
        lookup = {}
        for impl, measurements in impl_data.items():
            for x_val, _, time_us in measurements:
                lookup[(x_val, impl)] = time_us

        # Build table
        columns = [x_param_name] + sorted_impls
        rows = []
        for x_val in sorted_x_vals:
            row = [float(x_val) if isinstance(x_val, (int, float)) else x_val]
            for impl in sorted_impls:
                time_us = lookup.get((x_val, impl), "")
                row.append(time_us)
            rows.append(row)

        _format_table(columns, rows)


# Register cleanup function
atexit.register(_print_benchmark_results)


def _check_impl_available(impl: str) -> bool:
    """Check if an implementation is available."""
    try:
        if impl == "triton":
            from chitu.utils import try_import_platform_dep

            _, has = try_import_platform_dep("triton")
            return has
        elif impl == "cuda":
            from chitu.utils import try_import_platform_dep

            _, has = try_import_platform_dep("chitu_backend")
            return has
        elif impl == "torch_npu":
            from chitu.utils import try_import_and_setup_torch_npu

            _, has = try_import_and_setup_torch_npu()
            return has
        elif impl in ["torch", "ref"]:
            return True
        elif impl == "cpuinfer":
            from chitu.utils import try_import_opt_dep

            _, has = try_import_opt_dep("cpuinfer", "cpu")
            return has
        else:
            return True  # Unknown impl, assume available
    except Exception:
        return False


class BenchmarkRecorder:
    """Records benchmark measurements for a test."""

    def __init__(self, test_name: str, warmup_round: int, timing_round: int):
        self.test_name = test_name
        self.warmup_round = warmup_round
        self.timing_round = timing_round
        self._backend = None

    def _get_backend(self):
        """Lazy import torch.cuda backend."""
        if self._backend is None:
            import torch

            self._backend = torch.cuda
        return self._backend

    def _do_bench_rounds(self, fn: Callable) -> float:
        """
        Simple benchmark using fixed number of rounds.

        Args:
            fn: Callable to benchmark.

        Returns:
            Average execution time in milliseconds.
        """
        import gc

        backend = self._get_backend()

        # Warmup rounds
        for _ in range(self.warmup_round):
            fn()
        backend.synchronize()

        if self.timing_round <= 0:
            return 0.0

        # Timing rounds
        start_event = backend.Event(enable_timing=True)
        end_event = backend.Event(enable_timing=True)
        stream = backend.current_stream()

        start_event.record(stream=stream)
        for _ in range(self.timing_round):
            fn()
        end_event.record(stream=stream)
        backend.synchronize()

        total_ms = start_event.elapsed_time(end_event)

        # Clean up GPU memory to avoid OOM in subsequent tests
        gc.collect()
        backend.empty_cache()

        return total_ms / self.timing_round

    def __call__(
        self,
        fn: Optional[Callable] = None,
        x_val: Any = None,
        x_name: str = "x",
        impl: str = "default",
        impls: Optional[Dict[str, Callable]] = None,
        **kwargs,
    ):
        """
        Record benchmark measurement(s).

        Two usage modes:

        Mode 1 - Single impl:
            record_benchmark(lambda: op(x), N=N, impl="triton")

        Mode 2 - Multiple impls (recommended):
            record_benchmark(
                N=N,
                impls={
                    "triton": lambda: op(x, impl="triton"),
                    "torch": lambda: op(x, impl="torch"),
                }
            )

        Args:
            fn: Callable to benchmark (single impl mode).
            x_val: The x-axis value (e.g., size, batch_size).
            x_name: Name of the x-axis parameter.
            impl: Implementation name for single impl mode.
            impls: Dict of {impl_name: callable} for multi-impl mode.
            **kwargs: Additional parameters (first numeric one used as x_val if not specified).
        """
        # Auto-detect x_val from kwargs if not specified
        if x_val is None:
            for k, v in kwargs.items():
                if isinstance(v, (int, float)) and k != "impl":
                    x_val = v
                    x_name = k
                    break

        if x_val is None:
            x_val = 0

        # Mode 2: Multiple implementations
        if impls is not None:
            for impl_name, impl_fn in impls.items():
                if not _check_impl_available(impl_name):
                    continue
                try:
                    ms = self._do_bench_rounds(impl_fn)
                    time_us = ms * 1000
                    _benchmark_data[self.test_name][impl_name].append(
                        (x_val, x_name, time_us)
                    )
                except Exception as e:
                    print(f"Benchmark failed for {self.test_name}[{impl_name}]: {e}")
            return

        # Mode 1: Single implementation
        if fn is None:
            return

        # Auto-detect impl from kwargs
        if impl == "default" and "impl" in kwargs:
            impl = str(kwargs["impl"])

        try:
            ms = self._do_bench_rounds(fn)
            time_us = ms * 1000  # Convert to microseconds
            _benchmark_data[self.test_name][impl].append((x_val, x_name, time_us))
        except Exception as e:
            # Don't fail the test if benchmarking fails
            print(f"Benchmark failed for {self.test_name}: {e}")

    def run(
        self,
        fn: Callable,
        x_val: Any = None,
        x_name: str = "x",
        impl: str = "default",
        **kwargs,
    ) -> Any:
        """
        Execute function, record benchmark, and return result.

        This combines correctness test and benchmark into one call.
        Only runs once for correctness, timing is measured during that run.

        Usage:
            out = record_benchmark.run(
                lambda: my_op(x, impl="triton"),
                x_val=N,
                impl="triton",
            )
            chitu.testing.assert_close(out, ref_out)

        Args:
            fn: Callable to benchmark and get result from.
            x_val: The x-axis value (e.g., size, batch_size).
            x_name: Name of the x-axis parameter.
            impl: Implementation name.
            **kwargs: Additional parameters (first numeric one used as x_val if not specified).

        Returns:
            The result of fn() (from the last timing round).
        """
        import gc

        backend = self._get_backend()

        # Auto-detect x_val from kwargs if not specified
        if x_val is None:
            for k, v in kwargs.items():
                if isinstance(v, (int, float)) and k != "impl":
                    x_val = v
                    x_name = k
                    break
        if x_val is None:
            x_val = 0

        # Auto-detect impl from kwargs
        if impl == "default" and "impl" in kwargs:
            impl = str(kwargs["impl"])

        result = None
        try:
            # Warmup rounds (discard results)
            for _ in range(self.warmup_round):
                fn()
            backend.synchronize()

            if self.timing_round <= 0:
                # No timing, just run once for result
                result = fn()
                backend.synchronize()
                return result

            # Timing rounds
            start_event = backend.Event(enable_timing=True)
            end_event = backend.Event(enable_timing=True)
            stream = backend.current_stream()

            start_event.record(stream=stream)
            for _ in range(self.timing_round):
                result = fn()  # Keep last result for correctness check
            end_event.record(stream=stream)
            backend.synchronize()

            total_ms = start_event.elapsed_time(end_event)
            time_us = (total_ms / self.timing_round) * 1000
            _benchmark_data[self.test_name][impl].append((x_val, x_name, time_us))

            # Clean up GPU memory
            gc.collect()
            backend.empty_cache()

        except Exception as e:
            print(f"Benchmark failed for {self.test_name}: {e}")
            # Still try to return a result if benchmark failed
            if result is None:
                result = fn()

        return result


@pytest.fixture
def record_benchmark(request):
    """
    Fixture to record benchmark measurements.

    Benchmarks are always enabled. Control precision with:
        --warmup-round=N  (default: 0 for CI)
        --timing-round=N  (default: 1 for CI)

    For accurate performance testing, use:
        pytest test.py --warmup-round=5 --timing-round=20 -s

    Usage Mode 1 - Single impl (when test already iterates over impls):
        def test_my_op(M, N, impl, record_benchmark):
            x = torch.randn(M, N, device="cuda")
            result = my_op(x, impl=impl)
            record_benchmark(lambda: my_op(x, impl=impl), N=N, impl=impl)
            assert result == expected

    Usage Mode 2 - Multiple impls (recommended, compare all at once):
        def test_my_op(M, N, record_benchmark):
            x = torch.randn(M, N, device="cuda")
            result = my_op(x, impl="triton")

            record_benchmark(
                N=N,
                impls={
                    "triton": lambda: my_op(x, impl="triton"),
                    "torch": lambda: my_op(x, impl="torch"),
                    "cuda": lambda: my_op(x, impl="cuda"),
                }
            )
            assert result == expected
    """
    global _warmup_round, _timing_round
    _warmup_round = request.config.getoption("--warmup-round")
    _timing_round = request.config.getoption("--timing-round")

    # Extract test name without parameters
    test_name = request.node.originalname or request.node.name
    # Remove 'test_' prefix for cleaner output
    if test_name.startswith("test_"):
        test_name = test_name[5:]

    return BenchmarkRecorder(test_name, _warmup_round, _timing_round)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "benchmark: mark test as having benchmark capability",
    )
    config.addinivalue_line(
        "markers",
        "pd_unit: PD disaggregation unit tests",
    )
    config.addinivalue_line(
        "markers",
        "pd_deadlock: PD deadlock reproduction tests",
    )
    config.addinivalue_line(
        "markers",
        "pd_dist: PD distributed tests",
    )


# ---------------------------------------------------------------------------
# PD disaggregation fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _set_mooncake_env():
    # If mooncake is unavailable, fall back to mock.
    if "MOONCAKE_MOCK_MODE" not in os.environ:
        try:
            import mooncake  # noqa: F401

            os.environ["MOONCAKE_MOCK_MODE"] = "0"
        except Exception:
            os.environ["MOONCAKE_MOCK_MODE"] = "1"
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("PD_MASTER_ADDR", master_addr)
    yield


@pytest.fixture(scope="session")
def cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for PD KV tests.")


@pytest.fixture(scope="session")
def pd_ports():
    coordination_port = int(os.environ.get("PD_COORDINATION_PORT", "29800"))
    metadata_port = int(os.environ.get("PD_METADATA_PORT", "29801"))
    bootstrap_port = int(os.environ.get("PD_BOOTSTRAP_PORT", "8080"))
    return {
        "coordination_port": coordination_port,
        "metadata_port": metadata_port,
        "bootstrap_port": bootstrap_port,
    }


@pytest.fixture(scope="session")
def coordination_service(pd_ports):
    if os.environ.get("PD_COORDINATION_EXTERNAL", "0") == "1":
        yield None
        return
    service = PDCoordinationService(
        coordination_port=pd_ports["coordination_port"],
        metadata_sync_port=pd_ports["metadata_port"],
    )
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(service.start())
        started.set()
        loop.run_forever()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    started.wait(timeout=5)
    if not started.is_set():
        raise RuntimeError("failed to start PDCoordinationService")
    yield service

    fut = asyncio.run_coroutine_threadsafe(service.stop(), loop)
    fut.result(timeout=5)
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=5)


@pytest.fixture(scope="session")
def bootstrap_server(pd_ports):
    if os.environ.get("PD_BOOTSTRAP_EXTERNAL", "0") == "1":
        yield None
        return
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("PD_MASTER_ADDR", master_addr)
    server = MooncakeBootstrapServer(port=pd_ports["bootstrap_port"])
    server.start_in_background()
    # Give server a moment to bind.
    time.sleep(0.2)
    yield server


@pytest.fixture(scope="session")
def global_args(pd_ports):
    ib_device = os.environ.get("PD_IB_DEVICE", "mlx5_0")
    cfg = OmegaConf.create(
        {
            "models": {
                "name": "unit-test",
                "vocab_size": 128,
                "n_kv_heads": 2,
                "head_dim": 8,
            },
            "infer": {
                "tp_size": 1,
                "pp_size": 1,
                "dp_size": 1,
                "ep_size": 1,
                "mtp_size": 1,
                "max_seq_len": 512,
                "max_reqs": 16,
                "prefill_chunk_size": 128,
                "use_cuda_graph": False,
            },
            "scheduler": {
                "type": "prefill_only",
                "pp_config": {
                    "prefill_num_tasks_divided_by_pp": True,
                    "prefill_num_tasks": None,
                    "enforce_decode_num_tasks_max": True,
                    "decode_num_tasks": None,
                },
            },
            "dp_config": {
                "dp_id": 0,
                "router": {
                    "host": "127.0.0.1",
                    "pd_disaggregation": {
                        "enabled": True,
                        "coordination_port": pd_ports["coordination_port"],
                        "metadata_sync_port": pd_ports["metadata_port"],
                        "bootstrap_port": pd_ports["bootstrap_port"],
                        "ib_device": ib_device,
                        "kv_transfer": {
                            "decode_wait_timeout_s": 5.0,
                            "decode_resend_interval_s": 0.2,
                        },
                    },
                },
            },
        }
    )
    set_global_args(cfg, need_ensure=False)
    if global_vars._GLOBAL_TIMERS is None:
        global_vars._set_timers()
    return cfg


@pytest.fixture(scope="session")
def init_distributed(global_args):
    initialized = False
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(get_free_port()))
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
        initialized = True

    if not parallel_groups_initialized():
        initialize_parallel_groups(
            tp_size=global_args.infer.tp_size,
            pp_size=global_args.infer.pp_size,
            dp_size=global_args.infer.dp_size,
            ep_size=global_args.infer.ep_size,
            etp_size=1,
        )

    yield

    if initialized:
        if torch.distributed.is_initialized():
            if torch.distributed.get_world_size() > 1 and parallel_groups_initialized():
                destroy_parallel_groups()
            torch.distributed.destroy_process_group()
