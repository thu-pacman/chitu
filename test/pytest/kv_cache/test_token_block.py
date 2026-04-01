# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from chitu.kv_cache import TokenBlock, NONE_BLK_HASH

"""Tests for TokenBlock class."""


def test_token_block_creation():
    """创建token block"""
    tokens = [1, 2, 3, 4]
    block = TokenBlock(tokens=tokens, blk_size=4, pre_blk_hash=NONE_BLK_HASH)

    assert block.tokens == tokens
    assert block.blk_size == 4
    assert block.blk_hash is None
    assert block.cache_idx is None
    assert block.active_cnt == 0


def test_token_block_hash_generation():
    """测试生成blk_hash"""
    tokens = [1, 2, 3, 4]
    block = TokenBlock(tokens=tokens, blk_size=4, pre_blk_hash=NONE_BLK_HASH)
    block.generate_blk_hash()

    assert block.blk_hash is not None
    assert len(block.blk_hash) == 64  # SHA-256 produces 64 hex characters


def test_token_block_hash_deterministic():
    """测试相同的tokens和pre_blk_hash会生成相同的blk_hash"""
    tokens = [1, 2, 3, 4]

    block1 = TokenBlock(tokens=tokens, blk_size=4, pre_blk_hash=NONE_BLK_HASH)
    block1.generate_blk_hash()

    block2 = TokenBlock(tokens=tokens, blk_size=4, pre_blk_hash=NONE_BLK_HASH)
    block2.generate_blk_hash()

    assert block1.blk_hash == block2.blk_hash


def test_token_block_hash_different_for_different_tokens():
    """测试不同的tokens生成不同的blk_hash"""
    block1 = TokenBlock(tokens=[1, 2, 3, 4], blk_size=4, pre_blk_hash=NONE_BLK_HASH)
    block1.generate_blk_hash()

    block2 = TokenBlock(tokens=[5, 6, 7, 8], blk_size=4, pre_blk_hash=NONE_BLK_HASH)
    block2.generate_blk_hash()

    assert block1.blk_hash != block2.blk_hash


def test_token_block_hash_different_for_different_pre_blk_hash():
    """测试不同的pre_blk_hash生成不同的blk_hash"""
    tokens = [1, 2, 3, 4]

    block1 = TokenBlock(tokens=tokens, blk_size=4, pre_blk_hash=NONE_BLK_HASH)
    block1.generate_blk_hash()

    block2 = TokenBlock(tokens=tokens, blk_size=4, pre_blk_hash="some_hash")
    block2.generate_blk_hash()

    assert block1.blk_hash != block2.blk_hash


def test_token_block_len():
    """测试TokenBlock的len功能"""
    tokens = [1, 2, 3, 4, 5]
    block = TokenBlock(tokens=tokens, blk_size=10, pre_blk_hash=NONE_BLK_HASH)

    assert len(block) == 5


def test_token_block_hash_only_for_full_block():
    """测试仅当tokens长度为blk_size时可以生成blk_hash"""
    tokens = [1, 2, 3]  # Not full
    block = TokenBlock(tokens=tokens, blk_size=4, pre_blk_hash=NONE_BLK_HASH)

    with pytest.raises(AssertionError):
        block.generate_blk_hash()
