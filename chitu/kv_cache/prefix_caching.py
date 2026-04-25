# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Optional, Any, Callable
import pickle

# NONE_BLK_HASH must be deterministic across processes/ranks.
NONE_BLK_HASH = "0" * 64


def _hash_fn(inputs: tuple[Any, ...]) -> str:
    byte_inputs = pickle.dumps(inputs)
    return sha256(byte_inputs).hexdigest()


class KVBlockState(str, Enum):
    MISSING = "missing"
    ACTIVE = "active"
    IDLE = "idle"


@dataclass(frozen=True)
class BlockIdentity:
    """Immutable identity for a token block."""

    tokens: tuple[int, ...] = field(default_factory=tuple)
    blk_size: int = 0
    pre_blk_hash: str = NONE_BLK_HASH
    blk_hash: Optional[str] = None

    @classmethod
    def from_tokens(
        cls,
        tokens: list[int],
        *,
        blk_size: int,
        pre_blk_hash: str,
        blk_hash: Optional[str] = None,
        auto_hash: bool = True,
    ) -> "BlockIdentity":
        if auto_hash and blk_hash is None and len(tokens) == blk_size:
            blk_hash = _hash_fn((pre_blk_hash, tuple(tokens)))
        return cls(
            tokens=tuple(tokens),
            blk_size=blk_size,
            pre_blk_hash=pre_blk_hash,
            blk_hash=blk_hash,
        )

    @property
    def is_full(self) -> bool:
        return len(self.tokens) == self.blk_size


@dataclass
class BlockRuntime:
    """Runtime occupancy for a KV physical block."""

    cache_idx: Optional[int] = None
    active_cnt: int = 0

    @property
    def state(self) -> KVBlockState:
        if self.cache_idx is None:
            return KVBlockState.MISSING
        if self.active_cnt > 0:
            return KVBlockState.ACTIVE
        return KVBlockState.IDLE


class TokenBlock:
    """
    Task side block placeholder.
    包含identity(用于prefix cache)和runtime(用于kv cache 索引)两部分信息
    """

    def __init__(
        self,
        tokens: Optional[list[int]] = None,
        blk_hash: Optional[str] = None,
        pre_blk_hash: str = NONE_BLK_HASH,
        blk_size: int = 0,
        cache_idx: Optional[int] = None,
        active_cnt: int = 0,
        *,
        identity: Optional[BlockIdentity] = None,
        runtime: Optional[BlockRuntime] = None,
    ):
        self.identity = identity or BlockIdentity.from_tokens(
            list(tokens or []),
            blk_size=blk_size,
            pre_blk_hash=pre_blk_hash,
            blk_hash=blk_hash,
            auto_hash=False,
        )
        self.runtime = runtime or BlockRuntime(
            cache_idx=cache_idx,
            active_cnt=active_cnt,
        )
        assert self.identity is not None and self.runtime is not None

    @property
    def tokens(self) -> list[int]:
        return list(self.identity.tokens)

    @tokens.setter
    def tokens(self, value: list[int]) -> None:
        self.update_identity(tokens=value, blk_hash=None)

    @property
    def blk_hash(self) -> Optional[str]:
        return self.identity.blk_hash

    @blk_hash.setter
    def blk_hash(self, value: Optional[str]) -> None:
        self.update_identity(blk_hash=value)

    @property
    def pre_blk_hash(self) -> str:
        return self.identity.pre_blk_hash

    @pre_blk_hash.setter
    def pre_blk_hash(self, value: str) -> None:
        self.update_identity(pre_blk_hash=value, blk_hash=None)

    @property
    def blk_size(self) -> int:
        return self.identity.blk_size

    @blk_size.setter
    def blk_size(self, value: int) -> None:
        self.update_identity(blk_size=value, blk_hash=None)

    @property
    def cache_idx(self) -> Optional[int]:
        return self.runtime.cache_idx

    @cache_idx.setter
    def cache_idx(self, value: Optional[int]) -> None:
        self.runtime.cache_idx = value

    @property
    def active_cnt(self) -> int:
        return self.runtime.active_cnt

    @active_cnt.setter
    def active_cnt(self, value: int) -> None:
        self.runtime.active_cnt = value

    def update_identity(
        self,
        *,
        tokens: Optional[list[int]] = None,
        pre_blk_hash: Optional[str] = None,
        blk_hash: Optional[str] = None,
        blk_size: Optional[int] = None,
    ) -> None:
        """Update block identity when identity info changed."""
        assert self.identity is not None
        changed = tokens is not None or pre_blk_hash is not None or blk_size is not None
        tokens = list(self.identity.tokens) if tokens is None else list(tokens)
        blk_size = self.identity.blk_size if blk_size is None else blk_size
        pre_blk_hash = (
            self.identity.pre_blk_hash if pre_blk_hash is None else pre_blk_hash
        )
        if blk_hash is None and changed:
            blk_hash = None
        else:
            blk_hash = self.identity.blk_hash if blk_hash is None else blk_hash
        self.identity = BlockIdentity.from_tokens(
            tokens,
            blk_size=blk_size,
            pre_blk_hash=pre_blk_hash,
            blk_hash=blk_hash,
        )

    def set_identity(self, identity: BlockIdentity) -> None:
        self.identity = identity

    def set_runtime(self, runtime: BlockRuntime) -> None:
        self.runtime = runtime

    @classmethod
    def hash_fn(cls, inputs: tuple[Any, ...]) -> str:
        return _hash_fn(inputs)

    def generate_blk_hash(self) -> None:
        assert len(self.tokens) == self.blk_size, (
            "blk_hash can only be generated when the block is full, "
            f"got tokens length ({len(self.tokens)}) but block size is ({self.blk_size}) "
        )
        self.update_identity(
            blk_hash=self.hash_fn((self.pre_blk_hash, tuple(self.tokens)))
        )

    @property
    def state(self) -> KVBlockState:
        assert self.runtime is not None
        return self.runtime.state

    def __len__(self):
        """TokenBlock中实际存储的token数量"""
        assert self.identity is not None
        return len(self.identity.tokens)

    def __repr__(self):
        return (
            f"TokenBlock(length={self.__len__()}, "
            f"blk_hash={self.blk_hash}, "
            f"pre_blk_hash={self.pre_blk_hash}, "
            f"blk_size={self.blk_size}, "
            f"cache_idx={self.cache_idx}, "
            f"active_cnt={self.active_cnt})"
        )


class BlockIdentityChainBuilder:
    """
    将 token 序列按 block_size 切分并构建带前缀哈希链的 BlockIdentity 列表。

    existing_identities 可选，用于做 identity 复用和哈希冲突检查。
    """

    def __init__(
        self,
        block_size: int,
        existing_identities: Any = None,
    ):
        self.block_size = block_size
        self.existing_identities: Any = (
            existing_identities if existing_identities is not None else {}
        )

    def _resolve_hash(self, identity: BlockIdentity) -> str:
        """解决哈希冲突，返回 identity 的唯一哈希值。"""
        blk_hash = identity.blk_hash or TokenBlock.hash_fn(
            (identity.pre_blk_hash, identity.tokens)
        )
        collision_count = 0
        while blk_hash in self.existing_identities:
            existing = self.existing_identities[blk_hash]
            if (
                existing.tokens == identity.tokens
                and existing.pre_blk_hash == identity.pre_blk_hash
            ):
                break  # 同一个 block，无冲突
            collision_count += 1
            blk_hash = TokenBlock.hash_fn(
                (identity.pre_blk_hash, identity.tokens, collision_count)
            )
        return blk_hash

    def make_identity(
        self,
        token_chunk: list[int],
        pre_blk_hash: str,
        auto_register: bool = False,
    ) -> BlockIdentity:
        """将一个 token块（长度 ≤ block_size）封装为 BlockIdentity。"""
        if len(token_chunk) > self.block_size:
            raise ValueError(
                f"Input tokens length ({len(token_chunk)}) shouldn't bigger than block size ({self.block_size})"
            )

        identity = BlockIdentity.from_tokens(
            list(token_chunk),
            blk_hash=None,
            blk_size=self.block_size,
            pre_blk_hash=pre_blk_hash,
        )
        if identity.blk_hash is None:
            return identity

        blk_hash = self._resolve_hash(identity)
        identity = BlockIdentity.from_tokens(
            list(identity.tokens),
            blk_size=identity.blk_size,
            pre_blk_hash=identity.pre_blk_hash,
            blk_hash=blk_hash,
        )

        if blk_hash in self.existing_identities:
            return self.existing_identities[blk_hash]

        if auto_register:
            self.existing_identities[blk_hash] = identity

        return identity

    def build(
        self,
        tokens: list[int],
        initial_pre_hash: str = NONE_BLK_HASH,
        auto_register: bool = False,
    ) -> list[BlockIdentity]:
        """
        Build block identities from a token sequence.
        Args:
            tokens: Token sequence to partition into blocks
            initial_pre_hash: Starting hash for prefix chain (default: NONE_BLK_HASH)
            auto_register: whether to insert new generated identity into store
        Returns:
            list[BlockIdentity]: List of identities with established prefix hash chain
        """
        identities: list[BlockIdentity] = []
        pre_blk_hash = initial_pre_hash

        for i in range(0, len(tokens), self.block_size):
            chunk = tokens[i : i + self.block_size]
            identity = self.make_identity(chunk, pre_blk_hash, auto_register)
            identities.append(identity)

            # Only full blocks have valid hash for chaining
            if len(chunk) == self.block_size and identity.blk_hash:
                pre_blk_hash = identity.blk_hash

        return identities
