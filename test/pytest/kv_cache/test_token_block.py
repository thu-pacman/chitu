# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any


import pytest

from chitu.kv_cache import (
    BlockIdentity,
    BlockIdentityChainBuilder,
    BlockRuntime,
    KVBlockState,
    NONE_BLK_HASH,
    TokenBlock,
)

_BUILDER_NAME = "__pytest_token_block__"
_BLOCK_SIZE = 4


@pytest.fixture(autouse=True)
def _clear_block_identity_builder_registry():
    BlockIdentityChainBuilder.clear_registry()
    yield
    BlockIdentityChainBuilder.clear_registry()


def test_token_block_default_identity_and_runtime():
    block = TokenBlock()
    assert block.tokens == tuple[int]()
    assert block.blk_size == 0
    assert block.blk_hash is None
    assert block.pre_blk_hash == NONE_BLK_HASH
    assert block.cache_idx is None
    assert block.active_cnt == 0
    assert block.state == KVBlockState.MISSING


def test_token_block_wraps_identity_and_runtime():
    identity = BlockIdentity(
        tokens=(1, 2),
        blk_size=_BLOCK_SIZE,
        pre_blk_hash=NONE_BLK_HASH,
        blk_hash="a" * 64,
    )
    runtime = BlockRuntime(cache_idx=7, active_cnt=1)
    block = TokenBlock(identity=identity, runtime=runtime)
    assert block.tokens == (1, 2)
    assert block.blk_size == _BLOCK_SIZE
    assert block.blk_hash == "a" * 64
    assert block.cache_idx == 7
    assert block.active_cnt == 1
    assert block.state == KVBlockState.ACTIVE


def test_full_block_blk_hash_via_make_identity():
    builder = BlockIdentityChainBuilder.acquire(_BUILDER_NAME, _BLOCK_SIZE)
    identity = builder.make_identity([1, 2, 3, 4], NONE_BLK_HASH)
    block = TokenBlock(
        identity=identity, runtime=BlockRuntime(cache_idx=0, active_cnt=1)
    )

    assert block.blk_hash is not None
    assert len(block.blk_hash) == 64
    assert block.tokens == (1, 2, 3, 4)
    assert len(block) == _BLOCK_SIZE


def test_make_identity_reuses_canonical_identity_same_pool():
    """同一 (pre_blk_hash, tokens) 在 hashed_block_pool 中复用同一 BlockIdentity。"""
    builder = BlockIdentityChainBuilder.acquire(_BUILDER_NAME, _BLOCK_SIZE)
    a = builder.make_identity([1, 2, 3, 4], NONE_BLK_HASH)
    b = builder.make_identity([1, 2, 3, 4], NONE_BLK_HASH)
    assert a is b
    assert a.blk_hash == b.blk_hash


def test_make_identity_different_tokens_different_blk_hash():
    builder = BlockIdentityChainBuilder.acquire(_BUILDER_NAME, _BLOCK_SIZE)
    i1 = builder.make_identity([1, 2, 3, 4], NONE_BLK_HASH)
    i2 = builder.make_identity([5, 6, 7, 8], NONE_BLK_HASH)
    assert i1.blk_hash != i2.blk_hash


def test_make_identity_different_pre_blk_hash_different_blk_hash():
    builder = BlockIdentityChainBuilder.acquire(_BUILDER_NAME, _BLOCK_SIZE)
    i1 = builder.make_identity([1, 2, 3, 4], NONE_BLK_HASH)
    i2 = builder.make_identity([1, 2, 3, 4], "b" * 64)
    assert i1.blk_hash != i2.blk_hash


def test_partial_block_no_blk_hash_and_empty_token_view():
    """未满块：make_identity 不填 blk_hash，且为省拷贝 identity.tokens 为空。"""
    builder = BlockIdentityChainBuilder.acquire(_BUILDER_NAME, _BLOCK_SIZE)
    identity = builder.make_identity([1, 2, 3], NONE_BLK_HASH)
    block = TokenBlock(identity=identity, runtime=BlockRuntime())

    assert identity.blk_hash is None
    assert block.blk_hash is None
    assert block.blk_size == _BLOCK_SIZE
    assert len(block) == 0
    assert block.tokens == tuple[int]()


def test_token_block_len_full_block():
    builder = BlockIdentityChainBuilder.acquire(_BUILDER_NAME, _BLOCK_SIZE)
    identity = builder.make_identity([1, 2, 3, 4], NONE_BLK_HASH)
    block = TokenBlock(identity=identity, runtime=BlockRuntime())
    assert len(block) == _BLOCK_SIZE
