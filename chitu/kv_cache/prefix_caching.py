# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Optional
import pickle
from chitu.global_vars import get_global_args
import functools

if TYPE_CHECKING:
    from task import Task

# NONE_BLK_HASH must be deterministic across processes/ranks.
NONE_BLK_HASH = "0" * 64
_HASH_FN_CACHE_MAX = 2048


@functools.lru_cache(maxsize=_HASH_FN_CACHE_MAX)
def hash_fn_cached(inputs: tuple[Any, ...]) -> str:
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
        identity: Optional[BlockIdentity] = None,
        runtime: Optional[BlockRuntime] = None,
    ):
        self.identity = identity if identity is not None else BlockIdentity()
        self.runtime = runtime if runtime is not None else BlockRuntime()
        assert self.identity is not None and self.runtime is not None

    @property
    def tokens(self) -> tuple[int]:
        return self.identity.tokens

    @property
    def blk_hash(self) -> Optional[str]:
        return self.identity.blk_hash

    @property
    def pre_blk_hash(self) -> str:
        return self.identity.pre_blk_hash

    @property
    def blk_size(self) -> int:
        return self.identity.blk_size

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

    def set_identity(self, identity: BlockIdentity) -> None:
        self.identity = identity

    def set_runtime(self, runtime: BlockRuntime) -> None:
        self.runtime = runtime

    @property
    def state(self) -> KVBlockState:
        assert self.runtime is not None
        return self.runtime.state

    def __len__(self):
        """TokenBlock中实际存储的token数量"""
        assert self.identity is not None
        return len(self.identity.tokens)

    def __repr__(self):
        bh = self.blk_hash
        ph = self.pre_blk_hash
        return (
            f"TokenBlock(length={self.__len__()}, "
            f"blk_hash={(bh[:6] if bh else None)}, "
            f"pre_blk_hash={ph[:6]}, "
            f"blk_size={self.blk_size}, "
            f"cache_idx={self.cache_idx}, "
            f"active_cnt={self.active_cnt})"
        )


class BlockIdentityChainBuilder:
    """
    将 token 序列按 block_size 切分并构建带前缀哈希链的 BlockIdentity 列表。
    """

    _registry: dict[tuple[str, int], "BlockIdentityChainBuilder"] = {}

    def __init__(self, manager_name: str, block_size: int):
        self.manager_name = manager_name
        self.block_size = block_size
        self.hashed_block_pool: dict[str, BlockIdentity] = dict()
        self.tid_to_identities: dict[str, list[BlockIdentity]] = dict()

        # 优化不开启前缀缓存/不计算块哈希时的耗时
        max_seq_len = getattr(get_global_args().infer, "max_seq_len", 8192)

        self._placeholder_identity = BlockIdentity(blk_size=block_size)
        self._placeholder_identity_chain = [
            self._placeholder_identity
            for _ in range((max_seq_len + self.block_size - 1) // self.block_size)
        ]

    @classmethod
    def acquire(cls, manager_name: str, block_size: int) -> "BlockIdentityChainBuilder":
        """同一进程内按 (manager_name, block_size) 共享实例；hashed_block_pool 维护满块
        canonical identity，供哈希冲突消解与复用。
        """
        key = (manager_name, block_size)
        if key not in cls._registry:
            cls._registry[key] = cls(manager_name, block_size)
        return cls._registry[key]

    @classmethod
    def clear_registry(cls) -> None:
        """清空按 acquire 缓存的 builder 实例。"""
        cls._registry.clear()

    def forget_hash(self, blk_hash: Optional[str]):
        if blk_hash is None:
            return
        self.hashed_block_pool.pop(blk_hash, None)

    def make_identity(
        self,
        token_chunk: list[int],
        pre_blk_hash: Optional[str] = None,
        *,
        canonical_prefix_hashes: bool = True,
    ) -> BlockIdentity:
        """将一个长度不超过 ``block_size`` 的 token 块封装为 ``BlockIdentity``。

        Args:
            token_chunk: 当前块的 token id；长度可为小于 ``block_size`` 的尾部块。
            pre_blk_hash: 上一满块的 ``blk_hash``；为首块时可使用 ``NONE_BLK_HASH``。
            canonical_prefix_hashes: 为 False，或当前块未满（``len(token_chunk) < block_size``）时，
                不计算 ``blk_hash``、不访问 ``hashed_block_pool``，且 ``identity.tokens`` 为空元组；
                为 True 且块满时，计算哈希、做碰撞消解并写入或复用 ``hashed_block_pool``。

        Returns:
            对应块的 ``BlockIdentity``。
        """
        if len(token_chunk) > self.block_size:
            raise ValueError(
                f"Input tokens length ({len(token_chunk)}) shouldn't bigger than block size ({self.block_size})"
            )
        if pre_blk_hash is None:
            pre_blk_hash = NONE_BLK_HASH

        want_hash = canonical_prefix_hashes and len(token_chunk) == self.block_size

        if not want_hash:
            return (
                self._placeholder_identity
            )  # 不计算块哈希时返回占位identity，减少耗时

        tokens = tuple(token_chunk)
        blk_hash = hash_fn_cached((pre_blk_hash, tokens))
        collision_count = 0
        while True:
            existing = self.hashed_block_pool.get(blk_hash)
            if existing is None:
                break
            if existing.tokens == tokens and existing.pre_blk_hash == pre_blk_hash:
                return existing
            collision_count += 1
            blk_hash = hash_fn_cached((pre_blk_hash, tokens, collision_count))

        identity = BlockIdentity(
            tokens=tokens,
            blk_size=self.block_size,
            pre_blk_hash=pre_blk_hash,
            blk_hash=blk_hash,
        )
        self.hashed_block_pool[blk_hash] = identity
        return identity

    def make_identity_chain_from_tokens(
        self,
        tokens: list[int],
        *,
        initial_pre_hash: str = NONE_BLK_HASH,
        canonical_prefix_hashes: bool = True,
    ) -> list[BlockIdentity]:
        """按 block_size 切分 token 序列并构建前缀哈希链（无 Task / tid 缓存）。

        供 Router 等仅有 prompt token 列表、不与 Task 生命周期绑定的调用方使用；
        与同参数的 ``make_identity_chain`` 在重建链条时的语义一致。

        Args:
            tokens: prompt 的 token id 序列。
            initial_pre_hash: 前缀哈希链起始值，默认 ``NONE_BLK_HASH``。
            canonical_prefix_hashes: 为 False 时跳过满块的哈希计算及 ``hashed_block_pool`` 读写。

        Returns:
            各块的 ``BlockIdentity`` 列表（满块上的前缀哈希链已建立）。
        """
        identities: list[BlockIdentity] = []
        pre_blk_hash: Optional[str] = initial_pre_hash
        for i in range(0, len(tokens), self.block_size):
            chunk = tokens[i : i + self.block_size]
            identity = self.make_identity(
                chunk,
                pre_blk_hash,
                canonical_prefix_hashes=canonical_prefix_hashes,
            )
            identities.append(identity)
            pre_blk_hash = identity.blk_hash
        return identities

    def forget_task(self, task: "Task"):
        self.tid_to_identities.pop(task.task_id, None)

    def make_identity_chain(
        self,
        task: "Task",
        *,
        initial_pre_hash: str = NONE_BLK_HASH,
        canonical_prefix_hashes: bool = True,
    ) -> list[BlockIdentity]:
        """根据 ``task.prefix_tokens`` 构建块身份链。

        结果按 ``task.task_id`` 缓存在 ``tid_to_identities``；若当前缓存块数少于
        ``task.prefix_tokens`` 所需块数，则丢弃旧缓存并整条重建。

        Args:
            task: 推理任务，切分所用序列为 ``task.prefix_tokens``。
            initial_pre_hash: 前缀哈希链起始值，默认 ``NONE_BLK_HASH``。
            canonical_prefix_hashes: 为 False 时跳过满块的哈希计算及 ``hashed_block_pool`` 读写。

        Returns:
            ``task.prefix_tokens`` 对应的 ``BlockIdentity`` 列表（满块上的前缀哈希链已建立）。
        """
        if not canonical_prefix_hashes:
            return self._placeholder_identity_chain[
                : (len(task.prefix_tokens) + self.block_size - 1) // self.block_size
            ]

        identities: list[BlockIdentity] = self.tid_to_identities.get(task.task_id, [])

        if (
            len(identities)
            < (len(task.prefix_tokens) + self.block_size - 1) // self.block_size
        ):
            identities = self.make_identity_chain_from_tokens(
                task.prefix_tokens,
                initial_pre_hash=initial_pre_hash,
                canonical_prefix_hashes=canonical_prefix_hashes,
            )
            self.tid_to_identities[task.task_id] = identities
        return identities
