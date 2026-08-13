# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any


class CheckpointPrefixError(ValueError):
    """Raised when checkpoint-prefix paths cannot be resolved consistently."""


class CheckpointPrefix:
    """One or more real checkpoint paths represented as a single prefix value."""

    __slots__ = ("_paths",)

    def __init__(self, path: str | Iterable[str] | "CheckpointPrefix"):
        paths: Iterable[str]
        if isinstance(path, CheckpointPrefix):
            paths = path.paths
        elif isinstance(path, str):
            paths = {path}
        else:
            paths = set(path)

        normalized_paths: set[str] = set()
        for item in paths:
            if not isinstance(item, str):
                raise TypeError(
                    f"checkpoint prefix paths must be strings, got {type(item).__name__}"
                )
            normalized_paths.add(self._normalize_path(item))

        if not normalized_paths:
            raise ValueError("CheckpointPrefix requires at least one path")

        self._paths = frozenset(normalized_paths)

    @property
    def paths(self) -> frozenset[str]:
        return self._paths

    @classmethod
    def merged(
        cls, *prefixes: str | Iterable[str] | "CheckpointPrefix"
    ) -> "CheckpointPrefix":
        paths: set[str] = set()
        for prefix in prefixes:
            paths.update(as_checkpoint_prefix(prefix).paths)
        return cls(paths)

    @staticmethod
    def _normalize_path(path: str) -> str:
        return path.strip(".")

    @staticmethod
    def _join_path(prefix: str, suffix: str) -> str:
        suffix = suffix.strip(".")
        if not prefix:
            return suffix
        if not suffix:
            return prefix
        return f"{prefix}.{suffix}"

    @classmethod
    def _join_prefixes(
        cls,
        left: str | Iterable[str] | "CheckpointPrefix",
        right: str | Iterable[str] | "CheckpointPrefix",
    ) -> "CheckpointPrefix":
        left_prefix = as_checkpoint_prefix(left)
        right_prefix = as_checkpoint_prefix(right)
        return cls(
            cls._join_path(left_path, right_path)
            for left_path in left_prefix.paths
            for right_path in right_prefix.paths
        )

    def is_single(self) -> bool:
        return len(self._paths) == 1

    def single(self) -> str:
        if not self.is_single():
            raise CheckpointPrefixError(
                "A single checkpoint path is required, but this prefix represents "
                f"multiple paths: {', '.join(self._paths)}"
            )
        return next(iter(self._paths))

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __truediv__(
        self, suffix: str | Iterable[str] | "CheckpointPrefix"
    ) -> "CheckpointPrefix":
        return self._join_prefixes(self, suffix)

    def __rtruediv__(
        self, prefix: str | Iterable[str] | "CheckpointPrefix"
    ) -> "CheckpointPrefix":
        return self._join_prefixes(prefix, self)

    def __str__(self) -> str:
        return self.single()

    def __repr__(self) -> str:
        if self.is_single():
            return f"CheckpointPrefix({self.single()!r})"
        return f"CheckpointPrefix({list(self._paths)!r})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CheckpointPrefix):
            return False
        return self.paths == other.paths

    def __hash__(self) -> int:
        return hash(self._paths)


def as_checkpoint_prefix(
    prefix: str | Iterable[str] | CheckpointPrefix,
) -> CheckpointPrefix:
    if isinstance(prefix, CheckpointPrefix):
        return prefix
    return CheckpointPrefix(prefix)
