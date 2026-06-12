# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import logging
import json
import os
import statistics
import tempfile
import time
import traceback
import gc
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Hashable, Optional, Sequence, TypeVar

import torch

from chitu.device_type import has_accelerator
from chitu.ops.utils import add_op_callback, remove_op_callback
from chitu.utils import try_import_and_setup_torch_npu

TConfig = TypeVar("TConfig")
logger = logging.getLogger(__name__)


def _summarize_statistics(times: list[float], return_mode: str) -> float:
    if return_mode == "min":
        return min(times)
    if return_mode == "max":
        return max(times)
    if return_mode == "mean":
        return statistics.mean(times)
    if return_mode == "median":
        return statistics.median(times)
    raise ValueError(f"Unknown return_mode: {return_mode}")


def do_bench_graph(
    fn: Callable[[], Any],
    *,
    rep: int = 20,
    min_repeat: int = 100,
    grad_to_none: Optional[list[torch.Tensor]] = None,
    return_mode: str = "mean",
) -> float:
    assert return_mode in ("min", "max", "mean", "median")
    if min_repeat < 1:
        raise ValueError(f"min_repeat must be >= 1, got {min_repeat}")
    try_import_and_setup_torch_npu()
    if not has_accelerator():
        raise RuntimeError(
            "do_bench_graph requires CUDA (or Ascend redirected as CUDA)."
        )

    with torch.cuda.stream(torch.cuda.Stream()):
        fn()
        if grad_to_none is not None:
            for x in grad_to_none:
                x.detach_()
                x.requires_grad_(True)
                x.grad = None

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream()
        start_event.record(stream=stream)
        for _ in range(5):
            fn()
        end_event.record(stream=stream)
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5
        n_repeat = 100 if estimate_ms == 0 else max(min_repeat, int(rep / estimate_ms))

        g = torch.cuda.CUDAGraph()
        gc.disable()  # Disable GC to prevent mid-capture tensor destruction
        try:
            with torch.cuda.graph(g):
                for _ in range(n_repeat):
                    if grad_to_none is not None:
                        for x in grad_to_none:
                            x.grad = None
                    fn()
        finally:
            gc.enable()  # Always re-enable GC
            gc.collect()  # Clean up anything that was delayed
        torch.cuda.synchronize()

        ret: list[float] = []
        for _ in range(1):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream()
            start_event.record(stream=stream)
            g.replay()
            end_event.record(stream=stream)
            torch.cuda.synchronize()
            ret.append(start_event.elapsed_time(end_event) / n_repeat)
        return _summarize_statistics(ret, return_mode)


def _default_autotune_timer(fn: Callable[[], Any]) -> float:
    return float(do_bench_graph(fn, rep=20, return_mode="mean"))


def _is_rank0() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0"))) == 0


class AutotuneGraphTimer:
    def __init__(
        self,
        timer_fn: Optional[Callable[[Callable[[], Any]], float]] = None,
    ):
        self._timer_fn = timer_fn or _default_autotune_timer

    def __call__(self, fn: Callable[[], Any]) -> float:
        return float(self._timer_fn(fn))


class Autotuner:
    def __init__(
        self,
        *,
        timer: Optional[Callable[[Callable[[], Any]], float]] = None,
        name: str = "autotuner",
        cache_dir: Optional[str] = None,
    ):
        self._timer = timer or AutotuneGraphTimer()
        self._cache: dict[Hashable, Any] = {}
        self._name = name
        cache_root = cache_dir or os.getenv(
            "CHITU_AUTOTUNE_CACHE_DIR", "~/.cache/chitu"
        )
        self._cache_path = Path(cache_root).expanduser() / f"{self._name}.json"
        self._disk_cache: dict[str, Any] = {}
        self._load_disk_cache()

    def clear(self) -> None:
        self._cache.clear()
        self._disk_cache.clear()
        try:
            self._cache_path.unlink(missing_ok=True)
        except Exception:
            logger.exception(
                "%s failed to clear cache file: %s", self._name, self._cache_path
            )

    def _normalize_key_for_json(self, obj: Any) -> Any:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, (tuple, list)):
            return [self._normalize_key_for_json(item) for item in obj]
        if isinstance(obj, dict):
            return {
                str(k): self._normalize_key_for_json(v)
                for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
            }
        return repr(obj)

    def _serialize_cache_key(self, key: Hashable) -> str:
        normalized = self._normalize_key_for_json(key)
        return json.dumps(
            normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )

    def _load_disk_cache(self) -> None:
        try:
            if not self._cache_path.exists():
                return
            content = json.loads(self._cache_path.read_text())
            if isinstance(content, dict):
                self._disk_cache = content
            else:
                logger.warning(
                    "%s cache file format is invalid, ignored: %s",
                    self._name,
                    self._cache_path,
                )
        except Exception:
            logger.exception(
                "%s failed to load cache file: %s", self._name, self._cache_path
            )
            self._disk_cache = {}

    def _flush_disk_cache(self) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self._cache_path.parent,
                prefix=f".{self._name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp_f:
                json.dump(self._disk_cache, tmp_f, ensure_ascii=True, sort_keys=True)
                tmp_name = tmp_f.name
            os.replace(tmp_name, self._cache_path)
        except Exception:
            logger.exception(
                "%s failed to flush cache file: %s", self._name, self._cache_path
            )

    def _is_json_serializable(self, value: Any) -> bool:
        try:
            json.dumps(value, ensure_ascii=True, sort_keys=True)
            return True
        except Exception:
            return False

    def try_get_cached(
        self,
        *,
        key: Hashable,
        valid_configs: Optional[Sequence[TConfig]] = None,
    ) -> Optional[TConfig]:
        cached = self._cache.get(key)
        if cached is not None:
            if valid_configs is None or cached in valid_configs:
                logger.debug_once("%s hit in-memory cache for key=%s", self._name, key)
                return cached

        cache_key = self._serialize_cache_key(key)
        cached_from_disk = self._disk_cache.get(cache_key)
        if cached_from_disk is not None:
            if valid_configs is None or cached_from_disk in valid_configs:
                self._cache[key] = cached_from_disk
                logger.debug_once("%s hit disk cache for key=%s", self._name, key)
                return cached_from_disk
        return None

    def tune(
        self,
        *,
        key: Hashable,
        configs: Sequence[TConfig],
        make_bench_fn: Callable[[TConfig], Callable[[], Any]],
    ) -> TConfig:
        if not configs:
            raise ValueError("Autotuner requires non-empty configs.")
        cached = self.try_get_cached(key=key, valid_configs=configs)
        if cached is not None:
            return cached
        cache_key = self._serialize_cache_key(key)

        logger.debug_once(
            "%s autotune miss for key=%s, trying %d configs",
            self._name,
            key,
            len(configs),
        )
        tune_start_t = time.perf_counter()
        best_config: Optional[TConfig] = None
        best_time_ms: Optional[float] = None
        errors: list[tuple[TConfig, Exception]] = []

        for config in configs:
            try:
                elapsed_ms = float(self._timer(make_bench_fn(config)))
            except Exception as exc:
                errors.append((config, exc))
                continue
            if best_time_ms is None or elapsed_ms < best_time_ms:
                best_time_ms = elapsed_ms
                best_config = config

        if best_config is None:
            details = ", ".join(
                [
                    f"{cfg}: {exc} at {''.join(traceback.format_exception(exc))}"
                    for cfg, exc in errors[:3]
                ]
            )
            raise RuntimeError(f"{self._name} failed for all configs: {details}")

        self._cache[key] = best_config
        if self._is_json_serializable(best_config):
            self._disk_cache[cache_key] = best_config
            self._flush_disk_cache()
            logger.debug_once(
                "%s saved best config to disk cache path=%s key=%s config=%s",
                self._name,
                self._cache_path,
                key,
                best_config,
            )
        else:
            logger.debug(
                "%s selected config=%s for key=%s but skipped disk cache (not JSON serializable).",
                self._name,
                best_config,
                key,
            )
        if best_time_ms is not None:
            logger.debug(
                "%s selected config=%s for key=%s (%.3f ms)",
                self._name,
                best_config,
                key,
                best_time_ms,
            )
        tune_total_ms = (time.perf_counter() - tune_start_t) * 1000.0
        if _is_rank0():
            logger.debug_once(
                "%s autotune duration for key=%s: %.3f ms (configs=%d)",
                self._name,
                key,
                tune_total_ms,
                len(configs),
            )
        return best_config


def assert_close(
    actual,
    expected,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    cos_sim_tol: float = 0.0,
):
    """
    Assert that `actual` is close to `expected`.

    Since tensors often only bearly close to each other in quantized operations, this
    function supports multiple ways to check. If any of them passes, the test succeeds.
    These ways are:
    - All elememts in the two tensors absolutely differ no more than `atol`. If `atol`
      is not specified, it will be decided according to the data type as in
      `torch.testing.assert_close`.
    - All elememts in the two tensors relatively differ no more than `rtol`. If `rtol`
      is not specified, it will be decided according to the data type as in
      `torch.testing.assert_close`.
    - The two tensors as a whole are close to each other in cosine similarity. The
      similarity should be no less than `1 - cos_sim_tol`.
    """

    if actual.numel() == 0:
        assert (
            actual.shape == expected.shape
        ), f"Tensor shape of actual({actual.shape}) does not match expected({expected.shape})."
    else:
        if cos_sim_tol > 0:
            x, y = actual.double(), expected.double()
            denominator = (x * x + y * y).sum()
            sim = 2 * (x * y).sum() / denominator
            if 1 - sim <= cos_sim_tol:
                return
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


class AssertOpCalled:
    """
    A context manager to help assert a specific op is called for a specific times.

    Usage:

    ```
    with AssertOpCalled("op_name", "impl_name", expected_call_cnt=1):
        ... # Do some computation
    ```
    """

    def __init__(
        self, op_name: str, impl_name: Optional[str] = None, expected_call_cnt: int = 1
    ):
        self.op_name = op_name
        self.impl_name = impl_name
        self.expected_call_cnt = expected_call_cnt
        self.actual_call_cnt = 0

        def callback(actual_op_name: str, actual_impl_name: str):
            if actual_op_name == self.op_name:
                if self.impl_name is None or actual_impl_name == self.impl_name:
                    self.actual_call_cnt += 1

        self.callback = callback

    def __enter__(self):
        add_op_callback(self.callback)

    def __exit__(self, exc_type, exc_value, traceback):
        remove_op_callback(self.callback)
        if exc_type is None:
            if self.impl_name is not None:
                err_msg = f"Expected {self.expected_call_cnt} calls on op {self.op_name}'s impl {self.impl_name}, but got {self.actual_call_cnt}"
            else:
                err_msg = f"Expected {self.expected_call_cnt} calls on op {self.op_name}'s any impl, but got {self.actual_call_cnt}"
            assert self.actual_call_cnt == self.expected_call_cnt, err_msg


def gen_token_to_expert_indices(
    num_tokens: int, num_experts: int, topk: int, distribution: str
):
    """
    Generate fake data for expert selection in MoE models.
    """

    if distribution == "imbalance":
        return torch.arange(topk, dtype=torch.int32, device="cuda").repeat(
            num_tokens, 1
        )
    elif distribution == "uniform":
        return torch.multinomial(
            torch.ones(num_tokens, num_experts, device="cuda"), topk, replacement=False
        ).to(torch.int32)
