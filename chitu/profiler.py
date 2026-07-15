# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
import os
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import Callable
from chitu.global_vars import get_global_args


import torch

logger = getLogger(__name__)


# 参考自 https://github.com/sgl-project/sglang/blob/main/python/sglang/profiler.py
class MemoryRecorder:
    """Process-global singleton

    All memory recording / dumping / OOM-hook operations go through this class.
    """

    _instance: MemoryRecorder | None = None

    @classmethod
    def get(cls) -> MemoryRecorder:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._recording: bool = False
        self._snapshot_dir: str = ""
        self._oom_hook_installed: bool = False

    @property
    def enabled(self) -> bool:
        """True when early-start mode was activated (env CHITU_MEM_TRACK=1)."""
        return bool(self._snapshot_dir)

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def snapshot_dir(self) -> str:
        return self._snapshot_dir or os.path.join("trace", "chitu", "mem_track")

    def start_recording(
        self,
        *,
        max_entries: int = 100_000,
        stacks: str = "python",
    ) -> None:
        if self._recording:
            return
        torch.cuda.memory._record_memory_history(max_entries=max_entries, stacks=stacks)
        self._recording = True
        logger.warning(
            "MemoryRecorder: recording started (max_entries=%s, stacks=%s)",
            max_entries,
            stacks,
        )

    def stop_recording(self) -> None:
        if not self._recording:
            return
        torch.cuda.memory._record_memory_history(enabled=None)
        self._recording = False
        logger.warning("MemoryRecorder: recording stopped")

    def dump_snapshot(
        self,
        output_dir: str,
        *,
        suffix: str = "",
        tag: str = "",
    ) -> str | None:
        """Dump the current memory snapshot. Returns the file path on success."""
        path = _build_memory_snapshot_path(output_dir, suffix=suffix, tag=tag)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            torch.cuda.memory._dump_snapshot(path)
        except Exception:
            logger.exception("MemoryRecorder: failed to dump snapshot to %s", path)
            return None
        logger.warning("MemoryRecorder: snapshot dumped -> %s", path)
        return path

    def install_oom_hook(self, snapshot_dir: str) -> None:
        if self._oom_hook_installed:
            return

        try:
            torch._C._cuda_attach_out_of_memory_observer(
                lambda device, alloc, device_alloc, device_free: (
                    logger.warning(
                        f"MemoryRecorder: OOM on device {device} "
                        f"(requested={alloc} B, allocated={device_alloc} B, free={device_free} B)",
                    ),
                    self.dump_snapshot(snapshot_dir, tag="OOM"),
                )
            )
        except AttributeError:
            logger.warning(
                "MemoryRecorder: _cuda_attach_out_of_memory_observer not available, "
                "OOM hook skipped"
            )
            return

        self._oom_hook_installed = True
        logger.warning("MemoryRecorder: OOM observer hook installed")

    @staticmethod
    def checkpoint(label: str) -> None:
        """Log a memory-usage checkpoint. No-op when early-start is off."""
        rec = MemoryRecorder.get()
        if not rec.enabled:
            return

        rank = _get_rank()
        dev = torch.cuda.current_device()
        torch.cuda.synchronize(dev)

        free, total = torch.cuda.mem_get_info(dev)
        used = total - free
        stats = torch.cuda.memory_stats(dev)

        logger.warning(
            "[mem_track][rank %d] %-40s | "
            "used=%7.2f GB  alloc=%7.2f GB  reserved=%7.2f GB  "
            "non_torch=%7.2f GB  peak_alloc=%7.2f GB  "
            "retries=%d  ooms=%d",
            rank,
            label,
            used / 1e9,
            stats.get("allocated_bytes.all.current", 0) / 1e9,
            stats.get("reserved_bytes.all.current", 0) / 1e9,
            max(0, used - stats.get("reserved_bytes.all.current", 0)) / 1e9,
            stats.get("allocated_bytes.all.peak", 0) / 1e9,
            stats.get("num_alloc_retries", 0),
            stats.get("num_ooms", 0),
        )

    @classmethod
    def init(cls) -> None:
        """Call before any CUDA allocation
        Activated by CHITU_MEM_TRACK=1.
        """
        if os.getenv("CHITU_MEM_TRACK") != "1":
            return
        if not torch.cuda.is_available():
            return

        rec = cls.get()
        max_entries_env = os.getenv("CHITU_MEM_TRACK_MAX_ENTRIES", "").strip()
        max_entries = int(max_entries_env) if max_entries_env else 1_000_000
        rec._snapshot_dir = os.getenv(
            "CHITU_MEM_TRACK_SNAPSHOT_DIR",
            os.path.join(os.getcwd(), "trace", "chitu", "mem_track"),
        )
        os.makedirs(rec._snapshot_dir, exist_ok=True)

        rec.start_recording(max_entries=max_entries, stacks="python")
        logger.warning(
            "MemoryRecorder: early-start activated (snapshot_dir=%s)",
            rec._snapshot_dir,
        )
        atexit.register(lambda: rec.dump_snapshot(rec._snapshot_dir, tag="atexit"))


class ProfilerBase(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    def finish_collection(self) -> None:
        pass

    def step(self) -> None:
        pass


class ProfilerList(ProfilerBase):
    def __init__(self, inners: list[ProfilerBase]):
        self.inners = inners

    def start(self) -> None:
        for p in self.inners:
            p.start()

    def stop(self) -> None:
        for p in self.inners:
            p.stop()

    def finish_collection(self) -> None:
        for p in self.inners:
            p.finish_collection()

    def step(self) -> None:
        for p in self.inners:
            p.step()


class ProfilerTorch(ProfilerBase):
    def __init__(
        self,
        *,
        output_dir: str,
        output_suffix: str,
        with_stack: bool,
    ):
        self.output_dir = output_dir
        self.output_suffix = output_suffix
        self.with_stack = with_stack
        self._profiler: torch.profiler.profile | None = None
        self._collection_finished = False

    def start(self) -> None:
        self._profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=self.with_stack,
            with_modules=False,
            with_flops=False,
        )
        self._profiler.start()
        self._collection_finished = False

    def finish_collection(self) -> None:
        if self._profiler is None or self._collection_finished:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._profiler.stop()
        self._collection_finished = True

    def stop(self) -> None:
        if self._profiler is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        trace_path = _build_trace_path(self.output_dir, self.output_suffix)
        self.finish_collection()
        self._profiler.export_chrome_trace(trace_path)
        logger.warning("Profiler trace saved: %s", trace_path)
        self._profiler = None
        self._collection_finished = False

    def step(self) -> None:
        # ProfileManager owns the real-step window; torch profiler runs continuously.
        return


class ProfilerMemory(ProfilerBase):
    """Records CUDA memory history and dumps a snapshot on stop.
    Delegates to MemoryRecorder.
    """

    def __init__(
        self,
        *,
        output_dir: str,
        output_suffix: str,
        max_entries: int,
    ):
        self.output_dir = output_dir
        self.output_suffix = output_suffix
        self.max_entries = max_entries
        self._started_by_us = False

    def start(self) -> None:
        rec = MemoryRecorder.get()
        if rec.recording:
            self._started_by_us = False
            logger.warning(
                "ProfilerMemory: recording already active, will only dump on stop"
            )
        else:
            rec.start_recording(max_entries=self.max_entries, stacks="python")
            self._started_by_us = True

    def stop(self) -> None:
        rec = MemoryRecorder.get()
        rec.dump_snapshot(self.output_dir, suffix=self.output_suffix)
        if self._started_by_us:
            rec.stop_recording()
            self._started_by_us = False


def _get_rank() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")))


def _build_trace_path(output_dir: str, suffix: str = "") -> str:
    # PID is included so distinct PD scheduler processes that share the
    # same node + rank id don't overwrite each other's trace.
    rank = _get_rank()
    hostname = socket.gethostname()
    pid = os.getpid()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(
        output_dir,
        f"{timestamp}.rank_{rank}.{hostname}.pid_{pid}{suffix}.pt.trace.json.gz",
    )


def _build_memory_snapshot_path(
    output_dir: str,
    suffix: str = "",
    tag: str = "",
) -> str:
    rank = _get_rank()
    hostname = socket.gethostname()
    pid = os.getpid()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tag_part = f"-{tag}" if tag else ""
    return os.path.join(
        output_dir,
        f"{timestamp}.rank_{rank}.{hostname}.pid_{pid}{suffix}{tag_part}"
        f".memory_snapshot.pickle",
    )


def create_profiler(
    *,
    activities: list[str],
    output_dir: str,
    output_suffix: str,
    with_stack: bool,
    memory_max_entries: int,
) -> ProfilerBase:
    """Build profiler from activities list.

    Recognised activities: "CPU", "GPU", "MEM".
    """
    inners: list[ProfilerBase] = []
    if "CPU" in activities or "GPU" in activities:
        inners.append(
            ProfilerTorch(
                output_dir=output_dir,
                output_suffix=output_suffix,
                with_stack=with_stack,
            )
        )
    if "MEM" in activities:
        inners.append(
            ProfilerMemory(
                output_dir=output_dir,
                output_suffix=output_suffix,
                max_entries=memory_max_entries,
            )
        )
    if not inners:
        raise ValueError(f"No recognised profiler activities in {activities}")
    if len(inners) == 1:
        return inners[0]
    return ProfilerList(inners)


class StageBasedTrigger:
    """Counts steps per stage and fires start/stop callbacks."""

    @dataclass
    class _StageConfig:
        target_count: int

    @dataclass
    class _RunningState:
        curr_stage: str
        curr_count: int

    def __init__(self, on_start: Callable, on_stop: Callable):
        self.on_start = on_start
        self.on_stop = on_stop
        self.running_state: StageBasedTrigger._RunningState | None = None
        self.stage_configs: dict[str, StageBasedTrigger._StageConfig] = {}

    def configure(self, num_steps: int, interesting_stages: list[str]):
        self.stage_configs = {
            stage: self._StageConfig(target_count=num_steps)
            for stage in interesting_stages
        }

    def step(self, stage: str):
        if (s := self.running_state) is not None:
            s.curr_count += 1

        if (s := self.running_state) is not None and (
            s.curr_count >= self.stage_configs[s.curr_stage].target_count
            or stage != s.curr_stage
        ):
            del self.stage_configs[s.curr_stage]
            self.running_state = None
            self.on_stop()

        if self.running_state is None and stage in self.stage_configs:
            self.running_state = self._RunningState(
                curr_stage=stage,
                curr_count=0,
            )
            self.on_start(stage=stage)

    @property
    def done(self) -> bool:
        return not self.stage_configs and self.running_state is None


@dataclass
class _ProfileConfig:
    output_dir: str
    activities: list[str]
    start_step: int
    num_steps: int
    with_stack: bool
    profile_by_stage: bool
    memory_max_entries: int


class ProfileManager:
    """Central profiling controller.

    Usage::

        mgr = ProfileManager()
        mgr.configure(...)
        for each_inference_step:
            mgr.step(stage)
        mgr.stop()
    """

    def __init__(self):
        self._config: _ProfileConfig | None = None
        self._trigger: StageBasedTrigger | None = None
        self._current_profiler: ProfilerBase | None = None
        self._steps_seen: int = 0
        self._recorded_steps: int = 0
        self._completed: bool = False
        self._collection_finished: bool = False

    @property
    def active(self) -> bool:
        return self._config is not None and not self._completed

    @property
    def completed(self) -> bool:
        return self._completed

    def configure(
        self,
        *,
        output_dir: str,
        activities: list[str],
        start_step: int = 0,
        num_steps: int = 10,
        with_stack: bool = False,
        profile_by_stage: bool = False,
        memory_max_entries: int = 100000,
    ):
        self._config = _ProfileConfig(
            output_dir=output_dir,
            activities=list(activities),
            start_step=max(start_step, 0),
            num_steps=max(num_steps, 1),
            with_stack=with_stack,
            profile_by_stage=profile_by_stage,
            memory_max_entries=max(memory_max_entries, 1),
        )
        self._steps_seen = 0
        self._recorded_steps = 0
        self._completed = False
        self._collection_finished = False
        self._trigger = None
        self._current_profiler = None

        if profile_by_stage:
            self._trigger = StageBasedTrigger(
                on_start=self._do_start,
                on_stop=self._do_stop,
            )
            self._trigger.configure(
                num_steps=num_steps,
                interesting_stages=["Prefill", "Decode"],
            )

        logger.warning(
            "ProfileManager configured: output_dir=%s, activities=%s, "
            "start_step=%d, num_steps=%d, profile_by_stage=%s",
            output_dir,
            activities,
            self._config.start_step,
            self._config.num_steps,
            profile_by_stage,
        )

    def step(self, stage: str | None):
        """Call once per inference step. stage is "Prefill" or "Decode" (or None)."""
        if not self.active or self._collection_finished:
            return

        cfg = self._config
        if cfg.profile_by_stage and stage not in ("Prefill", "Decode"):
            return

        self._steps_seen += 1
        if self._steps_seen <= cfg.start_step:
            return

        try:
            if cfg.profile_by_stage:
                self._trigger.step(stage)
                if self._trigger.done:
                    self._completed = True
                    logger.warning("Stage profiling complete.")
            else:
                if self._current_profiler is None:
                    self._do_start()
                self._current_profiler.step()
                self._recorded_steps += 1
                if self._recorded_steps == cfg.num_steps:
                    self._finish_collection()
                    logger.warning(
                        "Profiling collected requested steps; call /profile/stop to export."
                    )
        except Exception:
            logger.exception("ProfileManager: error during step, disabling profiler")
            if self._current_profiler is not None:
                try:
                    self._current_profiler.stop()
                except Exception:
                    pass
                self._current_profiler = None
            self._completed = True

    def stop(self):
        """Force-stop profiling (e.g. via API /profile/stop)."""
        self._do_stop()
        self._completed = True

    def begin_step(self, stage: str | None, num_tasks: int) -> bool:
        """Start or resume collection for a real local inference step.

        Empty DP/EP padding steps must not consume num_steps. Once collection
        starts, the profiler stays active until the requested real steps finish.
        """
        if not self.active or self._collection_finished:
            return False

        cfg = self._config
        if cfg.profile_by_stage and stage not in ("Prefill", "Decode"):
            return False
        if num_tasks <= 0:
            return False

        self._steps_seen += 1
        if self._steps_seen <= cfg.start_step:
            return False

        try:
            if self._current_profiler is None:
                self._do_start()
            return True
        except Exception:
            logger.exception("ProfileManager: error starting step, disabling profiler")
            self._completed = True
            return False

    def end_step(self, stage: str | None, num_tasks: int) -> None:
        if not self.active or self._current_profiler is None or num_tasks <= 0:
            return

        cfg = self._config
        if cfg.profile_by_stage and stage not in ("Prefill", "Decode"):
            return

        try:
            self._current_profiler.step()
            self._recorded_steps += 1
            if self._recorded_steps == cfg.num_steps:
                self._finish_collection()
                logger.warning(
                    "Profiling collected requested steps; call /profile/stop to export."
                )
        except Exception:
            logger.exception("ProfileManager: error ending step, disabling profiler")
            if self._current_profiler is not None:
                try:
                    self._current_profiler.stop()
                except Exception:
                    pass
                self._current_profiler = None
            self._completed = True

    def _do_start(self, stage: str | None = None):
        cfg = self._config
        role = getattr(getattr(get_global_args(), "multi_inst", None), "role", None)
        suffix = f"-{role}" if role and role != "prefill_and_decode" else ""
        self._current_profiler = create_profiler(
            activities=cfg.activities,
            output_dir=cfg.output_dir,
            output_suffix=suffix,
            with_stack=cfg.with_stack,
            memory_max_entries=cfg.memory_max_entries,
        )
        self._current_profiler.start()
        logger.warning(
            "Profiling started%s.  Traces -> %s",
            f" for {stage}" if stage else "",
            cfg.output_dir,
        )

    def _finish_collection(self):
        if self._current_profiler is not None:
            self._current_profiler.finish_collection()
        self._collection_finished = True

    def _do_stop(self):
        if self._current_profiler is None:
            return
        self._current_profiler.stop()
        self._current_profiler = None
