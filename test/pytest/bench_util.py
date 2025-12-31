# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark utilities using torch.cuda.Event for GPU timing.

This module provides a triton.testing-compatible API for benchmarking GPU kernels
without depending on triton.
"""

import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
from chitu.utils import try_import_and_setup_torch_npu


def _get_device_backend():
    """
    Return the backend module and default device used by op benchmarks.
    """
    try_import_and_setup_torch_npu()
    if torch.cuda.is_available():
        return torch.cuda, torch.device("cuda")
    raise RuntimeError("No supported device backend found (cuda required).")


def get_default_device() -> torch.device:
    """Return the default device (use "cuda"; on Ascend it is redirected to NPU)."""
    _, device = _get_device_backend()
    return device


# SPDX-SnippetBegin
# SPDX-License-Identifier: MIT
# SPDX-SnippetCopyrightText: OpenAI
# SDPX—SnippetName: triton.testing.do_bench compatible utilities


def _quantile(a: List[float], q: List[float]) -> List[float]:
    """np.quantile/torch.quantile."""
    n = len(a)
    a = sorted(a)

    def get_quantile(q_val: float) -> float:
        if not (0 <= q_val <= 1):
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = q_val * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * a[lower] + t * a[upper]

    return [get_quantile(q_val) for q_val in q]


def _summarize_statistics(
    times: List[float],
    quantiles: Optional[List[float]],
    return_mode: str,
) -> Union[float, List[float]]:
    """Summarize benchmark statistics."""
    if quantiles is not None:
        ret = _quantile(times, quantiles)
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times
    elif return_mode == "min":
        return min(times)
    elif return_mode == "max":
        return max(times)
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)
    raise ValueError(f"Unknown return_mode: {return_mode}")


def do_bench(
    fn: Callable,
    warmup: int = 25,
    rep: int = 100,
    grad_to_none: Optional[List[torch.Tensor]] = None,
    quantiles: Optional[List[float]] = None,
    return_mode: str = "mean",
) -> Union[float, List[float]]:
    """
    Benchmark the runtime of the provided function using CUDA Events.

    :param fn: Function to benchmark
    :param warmup: Warmup time (in ms)
    :param rep: Repetition time (in ms)
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :param quantiles: Performance percentile to return
    :param return_mode: "min", "max", "mean", "median", or "all"
    :return: Execution time in milliseconds
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]
    backend, _ = _get_device_backend()

    fn()
    backend.synchronize()

    # Estimate execution time with 5 iterations
    start_event = backend.Event(enable_timing=True)
    end_event = backend.Event(enable_timing=True)
    stream = backend.current_stream()
    start_event.record(stream=stream)
    for _ in range(5):
        fn()
    end_event.record(stream=stream)
    backend.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5
    # Guard extremely small/zero estimates to avoid huge iteration counts
    estimate_ms = max(estimate_ms, 1e-3)

    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    # Using separate events allows continuous kernel submission without sync
    start_events = [backend.Event(enable_timing=True) for _ in range(n_repeat)]
    end_events = [backend.Event(enable_timing=True) for _ in range(n_repeat)]

    for _ in range(n_warmup):
        fn()

    for i in range(n_repeat):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None

        # Record start time on GPU
        stream = backend.current_stream()
        start_events[i].record(stream=stream)
        # Execute the function
        fn()
        # Record end time on GPU
        end_events[i].record(stream=stream)

    backend.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    return _summarize_statistics(times, quantiles, return_mode)


def do_bench_graph(
    fn: Callable,
    rep: int = 20,
    grad_to_none: Optional[List[torch.Tensor]] = None,
    quantiles: Optional[List[float]] = None,
    return_mode: str = "mean",
) -> Union[float, List[float]]:
    """
    Benchmark `fn` using device Graph replay to reduce host overhead.

    The function returns the per-call execution time in milliseconds.
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]
    from contextlib import nullcontext

    # Setup NPU compatibility (if present) so `torch.cuda.*` can be used uniformly.
    try_import_and_setup_torch_npu()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "do_bench_graph requires CUDA (or Ascend redirected as CUDA)."
        )

    backend = torch.cuda
    graph_cls = getattr(torch.cuda, "CUDAGraph", None)
    graph_ctx = getattr(torch.cuda, "graph", None)
    stream_ctx = getattr(torch.cuda, "stream", None)
    stream_cls = getattr(torch.cuda, "Stream", None)

    if graph_cls is None or graph_ctx is None:
        raise RuntimeError("do_bench_graph requires Graph support.")

    stream_cm = nullcontext()
    if stream_ctx is not None and stream_cls is not None:
        stream_cm = stream_ctx(stream_cls())

    with stream_cm:
        # Warmup
        fn()
        if grad_to_none is not None:
            for x in grad_to_none:
                x.detach_()
                x.requires_grad_(True)
                x.grad = None

        # Step 1: estimate runtime (without graph capture)
        start_event = backend.Event(enable_timing=True)
        end_event = backend.Event(enable_timing=True)
        stream = backend.current_stream()
        start_event.record(stream=stream)
        for _ in range(5):
            fn()
        end_event.record(stream=stream)
        backend.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5

        # Compute number of unrolled iterations in the graph
        if estimate_ms == 0:
            n_repeat = 1000
        else:
            n_repeat = max(1, int(rep / estimate_ms))

        # Step 2: capture `n_repeat` unrolled calls into a Graph
        g = graph_cls()
        if not hasattr(g, "replay"):
            raise RuntimeError("Graph object does not support replay().")
        try:
            # Some backends accept extra kwargs for graph capture (e.g. Ascend auto-dispatch).
            graph_cm = graph_ctx(g, auto_dispatch_capture=True)
        except TypeError:
            graph_cm = graph_ctx(g)

        with graph_cm:
            for _ in range(n_repeat):
                if grad_to_none is not None:
                    for x in grad_to_none:
                        x.grad = None
                fn()
        backend.synchronize()

        # Step 3: replay and time
        ret: List[float] = []
        n_retries = 10
        for _ in range(n_retries):
            start_event = backend.Event(enable_timing=True)
            end_event = backend.Event(enable_timing=True)
            stream = backend.current_stream()
            start_event.record(stream=stream)
            g.replay()
            end_event.record(stream=stream)
            backend.synchronize()
            ret.append(start_event.elapsed_time(end_event) / n_repeat)

        return _summarize_statistics(ret, quantiles, return_mode)


@dataclass
class Benchmark:
    """Configuration for a benchmark test."""

    x_names: List[str]
    x_vals: List[Any]
    line_arg: str
    line_vals: List[Any]
    line_names: List[str]
    plot_name: str
    args: Dict[str, Any]
    xlabel: str = ""
    ylabel: str = ""
    x_log: bool = False
    y_log: bool = False
    styles: Optional[List[Tuple[str, str]]] = None
    # Optional list of callables, one per line_val, returning True if provider is available.
    # If None, all providers are assumed available.
    line_available: Optional[List[Callable[[], bool]]] = None


class Mark:
    """Wrapper for benchmark functions."""

    def __init__(self, fn: Callable, benchmarks: Union[Benchmark, List[Benchmark]]):
        self.fn = fn
        self.benchmarks = benchmarks

    @staticmethod
    def _format_plain_table(
        columns: List[str],
        rows: List[List[Any]],
        *,
        float_precision: int = 6,
        index: bool = True,
    ) -> str:
        """Format a simple, pandas-like aligned table without external deps."""

        def _is_int_like(x: float) -> bool:
            try:
                return math.isfinite(x) and abs(x - round(x)) < 1e-9
            except Exception:
                return False

        def _format_cell(val: Any, col_idx: int) -> str:
            if val is None:
                return ""
            if isinstance(val, bool):
                return "True" if val else "False"
            if isinstance(val, str):
                return val
            if isinstance(val, int):
                return f"{float(val):.1f}" if col_idx >= 0 else str(val)
            if isinstance(val, float):
                if _is_int_like(val):
                    return f"{val:.1f}"
                return f"{val:.{float_precision}f}"
            return str(val)

        cell_rows: List[List[str]] = [
            [_format_cell(v, j) for j, v in enumerate(row)] for row in rows
        ]
        col_widths: List[int] = []
        for j, col in enumerate(columns):
            max_cell = max((len(r[j]) for r in cell_rows), default=0)
            col_widths.append(max(len(col), max_cell))

        idx_width = len(str(len(rows) - 1)) if index and rows else 0
        header_prefix = (" " * (idx_width + 1)) if index else ""
        header = header_prefix + " ".join(
            col.rjust(w) for col, w in zip(columns, col_widths)
        )

        lines = [header]
        for i, r in enumerate(cell_rows):
            idx = str(i).rjust(idx_width) + " " if index else ""
            lines.append(
                idx + " ".join(cell.rjust(w) for cell, w in zip(r, col_widths))
            )
        return "\n".join(lines)

    def _run(
        self,
        bench: Benchmark,
        save_path: str,
        show_plots: bool,
        print_data: bool,
        diff_col: bool = False,
        save_precision: int = 6,
        line_available: Optional[List[Callable[[], bool]]] = None,
        **kwargs,
    ):
        import os
        import importlib.util
        import importlib

        pd_spec = importlib.util.find_spec("pandas")
        pd = importlib.import_module("pandas") if pd_spec is not None else None

        y_mean = bench.line_names
        y_min = [f"{x}-min" for x in bench.line_names]
        y_max = [f"{x}-max" for x in bench.line_names]
        x_names = list(bench.x_names)
        if pd is not None:
            df = pd.DataFrame(columns=x_names + y_mean + y_min + y_max)
        else:
            df = None
            if show_plots or save_path:
                print("pandas not installed, skip plotting and saving images")

        # Precompute provider availability (once, not per x_val).
        # Prefer runtime line_available over benchmark-configured one.
        effective_line_available = line_available or bench.line_available
        provider_available: Dict[Any, bool] = {}
        if effective_line_available is not None:
            for i, y in enumerate(bench.line_vals):
                provider_available[y] = effective_line_available[i]()
        else:
            for y in bench.line_vals:
                provider_available[y] = True

        plain_rows: List[List[Any]] = []
        for x in bench.x_vals:
            # x can be a single value or a sequence of values
            if not isinstance(x, (list, tuple)):
                x = [x for _ in x_names]

            if len(x) != len(x_names):
                raise ValueError(f"Expected {len(x_names)} values, got {x}")
            x_args = dict(zip(x_names, x))

            row_mean, row_min, row_max = [], [], []
            for y in bench.line_vals:
                if not provider_available[y]:
                    # Provider not available: mark as "unsupported".
                    row_mean.append("unsupported")
                    row_min.append(None)
                    row_max.append(None)
                    continue
                ret = self.fn(**x_args, **{bench.line_arg: y}, **bench.args, **kwargs)
                if isinstance(ret, (list, tuple)):
                    if len(ret) == 3:
                        y_mean_val, y_min_val, y_max_val = ret
                    else:
                        y_mean_val, y_min_val, y_max_val = ret, None, None
                else:
                    y_mean_val, y_min_val, y_max_val = ret, None, None
                row_mean.append(y_mean_val)
                row_min.append(y_min_val)
                row_max.append(y_max_val)
            if df is not None:
                df.loc[len(df)] = list(x) + row_mean + row_min + row_max
            plain_rows.append(list(x) + row_mean)

        if df is not None and bench.plot_name and (show_plots or save_path):
            plt_spec = importlib.util.find_spec("matplotlib.pyplot")
            if plt_spec is None:
                print("matplotlib not installed, skip plotting and saving images")
                plt = None
            else:
                plt = importlib.import_module("matplotlib.pyplot")

            if plt is not None:
                plt.figure()
                ax = plt.subplot()
                first_x = x_names[0]
                for i, y in enumerate(bench.line_names):
                    y_min_col, y_max_col = df[y + "-min"], df[y + "-max"]
                    col = bench.styles[i][0] if bench.styles else None
                    sty = bench.styles[i][1] if bench.styles else None
                    ax.plot(df[first_x], df[y], label=y, color=col, ls=sty)
                    if not y_min_col.isnull().all() and not y_max_col.isnull().all():
                        y_min_col = y_min_col.astype(float)
                        y_max_col = y_max_col.astype(float)
                        ax.fill_between(
                            df[first_x], y_min_col, y_max_col, alpha=0.15, color=col
                        )
                ax.legend()
                ax.set_xlabel(bench.xlabel or first_x)
                ax.set_ylabel(bench.ylabel)
                ax.set_xscale("log" if bench.x_log else "linear")
                ax.set_yscale("log" if bench.y_log else "linear")
                if show_plots:
                    plt.show()
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
                    plt.savefig(os.path.join(save_path, f"{bench.plot_name}.png"))
                plt.close()

        if df is not None:
            df = df[x_names + bench.line_names]
            if diff_col and len(bench.line_names) == 2:
                col0, col1 = bench.line_names
                df["Diff"] = df[col1] - df[col0]

            if print_data:
                print(bench.plot_name + ":")
                print(df.to_string())
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                df.to_csv(
                    os.path.join(save_path, f"{bench.plot_name}.csv"),
                    float_format=f"%.{save_precision}f",
                    index=False,
                )
            return df

        # pandas not available: print aligned table without pandas.
        columns = x_names + bench.line_names
        if diff_col and len(bench.line_names) == 2:
            col0_idx = len(x_names) + 0
            col1_idx = len(x_names) + 1
            columns = columns + ["Diff"]
            for r in plain_rows:
                v0, v1 = r[col0_idx], r[col1_idx]
                if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                    r.append(v1 - v0)
                else:
                    r.append(None)

        if print_data:
            print(bench.plot_name + ":")
            print(
                self._format_plain_table(
                    columns,
                    plain_rows,
                    float_precision=save_precision,
                    index=True,
                )
            )
        return {"columns": columns, "rows": plain_rows}

    def run(
        self,
        show_plots: bool = False,
        print_data: bool = False,
        save_path: str = "",
        return_df: bool = False,
        line_available: Optional[List[Callable[[], bool]]] = None,
        **kwargs,
    ):
        has_single_bench = isinstance(self.benchmarks, Benchmark)
        benchmarks_list: List[Benchmark] = (
            [cast(Benchmark, self.benchmarks)]
            if has_single_bench
            else list(cast(List[Benchmark], self.benchmarks))
        )
        result_dfs: List[Any] = []

        html = None
        if save_path:
            import os

            os.makedirs(save_path, exist_ok=True)
            html = open(os.path.join(save_path, "results.html"), "w")
            html.write("<html><body>\n")

        for bench in benchmarks_list:
            result_dfs.append(
                self._run(
                    bench,
                    save_path,
                    show_plots,
                    print_data,
                    line_available=line_available,
                    **kwargs,
                )
            )
            if html is not None:
                html.write(f'<image src="{bench.plot_name}.png"/>\n')

        if html is not None:
            html.write("</body></html>\n")
            html.close()

        if return_df:
            return result_dfs[0] if has_single_bench else result_dfs
        return None


def perf_report(benchmarks: Union[Benchmark, List[Benchmark]]) -> Callable:

    def wrapper(fn: Callable) -> Mark:
        return Mark(fn, benchmarks)

    return wrapper


# SPDX-SnippetEnd
