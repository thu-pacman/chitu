# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from logging import getLogger
from dataclasses import dataclass, field
from typing import Any, Optional, Iterable, Literal
import time

from chitu.global_vars import get_global_args, get_instance_id, get_rank

logger = getLogger(__name__)


class TraceLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    NOTRACE = 1000


@dataclass
class TraceData:
    name: str
    timestamp: int
    instance_id: int
    rank: int
    data: dict[str, Any]

    @staticmethod
    def from_data(
        log_data: str | dict[str, Any], name: Optional[str] = None
    ) -> Optional["TraceData"]:
        if isinstance(log_data, str):
            log_data = dict(text=log_data)
            name = name or "unknown_log"
        elif isinstance(log_data, dict):
            name = log_data.pop("name", None) or name or "unknown_data"
        else:
            return None
        timestamp = log_data.pop("time", None) or time.time()
        instance_id = log_data.pop("instance_id", None) or get_instance_id()
        rank = log_data.pop("rank", None) or get_rank()
        return TraceData(
            name=name,
            timestamp=timestamp,
            instance_id=instance_id,
            rank=rank,
            data=log_data,
        )

    def to_dict(self) -> dict[str, Any]:
        data = self.data.copy()
        data["name"] = self.name
        data["time"] = self.timestamp
        data["instance_id"] = self.instance_id
        data["rank"] = self.rank
        return data


@dataclass
class Trace:
    """Trace data class

    Tracing the request, recording runtime infomation by level and reporting to users.
    """

    request_id: str = "NULL"
    trace_level: TraceLevel = TraceLevel.NOTRACE
    events: list[TraceData] = field(default_factory=list)

    def _trace(
        self, level: TraceLevel, log: str | dict[str, Any], name: Optional[str] = None
    ):
        if level < self.trace_level:
            return
        if isinstance(log, str):
            logger.log(level, log)
        trace_data = TraceData.from_data(log, name)
        if trace_data is not None:
            self.events.append(trace_data)

    def __bool__(self):
        return len(self.events) > 0

    def dump(self):
        return dict(
            request_id=self.request_id,
            trace_level=self.trace_level.value,
            events=[trace_data.to_dict() for trace_data in self.events],
        )

    @staticmethod
    def load(data: dict[str, Any]):
        request_id = data.get("request_id", "")
        trace_level = data.get("trace_level", TraceLevel.NOTRACE)
        event_dicts = data.get("events", [])
        return Trace(
            request_id=request_id,
            trace_level=trace_level,
            events=[TraceData.from_data(data) for data in event_dicts],
        )

    def merge(self, other: Optional["Trace" | Iterable["Trace"]]):
        """Merge two or more traces"""
        if other is None:
            return
        if isinstance(other, Trace):
            other = [other]
        all_trace = [self] + [trace for trace in other]
        self.events = [data for trace in all_trace for data in trace.events]

    def sort(self, key: Literal["time", "rank", "mixed"] = "mixed"):
        """Sort trace data by timestamp"""

        def trace_data_time(trace_data: TraceData):
            return (trace_data.timestamp, trace_data.instance_id, trace_data.rank)

        def trace_data_rank(trace_data: TraceData):
            return (trace_data.instance_id, trace_data.rank, trace_data.timestamp)

        def trace_data_mixed(trace_data: TraceData):
            return (trace_data.instance_id, trace_data.timestamp, trace_data.rank)

        key_func = None
        if key == "time":
            key_func = trace_data_time
        elif key == "rank":
            key_func = trace_data_rank
        elif key == "mixed":
            key_func = trace_data_mixed
        self.events.sort(key=key_func)

    def debug(self, *args, **kwargs):
        return self._trace(TraceLevel.DEBUG, *args, **kwargs)

    def info(self, *args, **kwargs):
        return self._trace(TraceLevel.INFO, *args, **kwargs)

    def warning(self, *args, **kwargs):
        return self._trace(TraceLevel.WARNING, *args, **kwargs)

    def error(self, *args, **kwargs):
        return self._trace(TraceLevel.ERROR, *args, **kwargs)

    def critical(self, *args, **kwargs):
        return self._trace(TraceLevel.CRITICAL, *args, **kwargs)
