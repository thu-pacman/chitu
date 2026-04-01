# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Optional, Any
import pickle


# NONE_BLK_HASH must be deterministic across processes/ranks.
NONE_BLK_HASH = "0" * 64


@dataclass
class TokenBlock:

    # Instance variables:
    tokens: list[int] = field(
        default_factory=list
    )  # 本block包含的token序列， token sequence of this block contains
    blk_hash: Optional[str] = (
        None  # 本block的哈希值，len(tokens)==blk_size有效，否则为None。 ths hash value of this block
    )
    pre_blk_hash: str = (
        NONE_BLK_HASH  # 上一个block的哈希值， hash value of the previous block
    )
    blk_size: int = 0  # TokenBlock的block size，是tokens的最大容量
    cache_idx: Optional[int] = (
        None  # 为kvcache的块索引，只有当该block被分配了kvcache索引时有效，否则为None
    )
    active_cnt: int = (
        0  # 表示该TokenBlock击中的task数量，只有当该block被分配了kvcache索引时有效
    )

    @classmethod
    def hash_fn(cls, inputs: tuple[Any, ...]) -> str:
        byte_inputs = pickle.dumps(inputs)
        return sha256(byte_inputs).hexdigest()

    def generate_blk_hash(self) -> None:
        assert len(self.tokens) == self.blk_size, (
            "blk_hash can only be generated when the block is full, "
            f"got tokens length ({len(self.tokens)}) but block size is ({self.blk_size}) "
        )
        self.blk_hash = self.hash_fn((self.pre_blk_hash, tuple(self.tokens)))

    def __len__(self):
        """TokenBlock中实际存储的token数量"""
        return len(self.tokens)

    def __repr__(self):
        return (
            f"TokenBlock(length={self.__len__()}, "
            f"blk_hash={self.blk_hash}, "
            f"pre_blk_hash={self.pre_blk_hash}, "
            f"blk_size={self.blk_size}, "
            f"cache_idx={self.cache_idx}, "
            f"active_cnt={self.active_cnt})"
        )
