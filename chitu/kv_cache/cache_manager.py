# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, TYPE_CHECKING
from logging import getLogger
from collections import deque, OrderedDict

from chitu.global_vars import get_global_args
from chitu.task_type import TaskType
from chitu.utils import ceil_div
from chitu.kv_cache.prefix_caching import (
    TokenBlock,
    BlockIdentityChainBuilder,
    BlockRuntime,
    KVBlockState,
    NONE_BLK_HASH,
)
from weakref import WeakValueDictionary
from collections import defaultdict

if TYPE_CHECKING:
    from chitu.task import Task

logger = getLogger(__name__)


class KVCacheManagerBase:
    pass


class PagedKVCacheManager(KVCacheManagerBase):

    def __init__(
        self,
        num_blocks: int,
        *,
        num_hot_req,
        max_seq_len,
        dp_rank: int,
        mtp_size: int = 1,
        enable_prefix_caching=False,
        block_size: int = 512,  # must be a multiple of 256 for FlashAttention
        manager_name: str = "main",
    ):

        self.max_blocks_per_req = ceil_div(max_seq_len, block_size)
        self.page_table_max_num_blocks = self.max_blocks_per_req * num_hot_req
        self.max_num_blocks = self.page_table_max_num_blocks

        if enable_prefix_caching:
            self.allocatable_max_num_blocks = 1 << 60
        else:
            self.allocatable_max_num_blocks = self.page_table_max_num_blocks

        self.num_blocks = num_blocks
        self.dp_rank = dp_rank
        self.manager_name = manager_name

        self.block_size = block_size
        self.task_to_cache_ids: defaultdict[str, set[int]] = defaultdict(
            set
        )  # {task_id: {block_idx,...}}
        self.task_to_token_blocks: dict[str, list[TokenBlock]] = defaultdict(list)
        self.mtp_size = mtp_size
        self.free_cache_ids: deque[int] = deque(
            range(self.num_blocks)
        )  # list of cache_idx, 保存所有未分配给TokenBlock的cache索引

        # 保存活跃度大于0的运行时块状态, 数据含义: {cache_idx: BlockRuntime}
        # 元素生命周期: TokenBlock在prepare_metadata_before_prefill/decode中，初次分配cache_idx或active_cnt从0变为1时，进入，
        #             在finalize_metadata_all_decode中active_cnt为0时移除
        self.active_blocks: dict[int, BlockRuntime] = {}

        # 保存活跃度为0，但分配了cache_idx的运行时块状态, 数据含义: {cache_idx: BlockRuntime}
        # 元素生命周期: 在调用finalize_metadata_all_decode时，若元素的active_cnt为0，则被放入self.cached_idle_blocks链表最右边
        #             在调用get_free_cache_idx时，若self.free_cache_ids为空，self.cached_idle_blocks链表最左边元素被最先移除
        self.cached_idle_blocks: OrderedDict[int, BlockRuntime] = OrderedDict()

        # prefix_caching related
        self.enable_prefix_caching: bool = enable_prefix_caching
        self.identity_builder = BlockIdentityChainBuilder.acquire(
            self.manager_name,
            self.block_size,
        )
        # 保存块身份到运行时状态的映射, 数据含义: {blk_hash: BlockRuntime}
        self.identity_runtime_pool: dict[str, BlockRuntime] = {}

        # prefix cache aware routing policy related
        # 保存cache_idx到blk_hash的映射
        self.cache_idx_to_hash: dict[int, str] = {}
        # Incremental evicted block hashes, consumed by scheduler stats reporter.
        self.evicted_blk_hashes: deque[str] = deque()

    def get_allocatable_max_num_blocks(self) -> int:
        return int(getattr(self, "allocatable_max_num_blocks", self.max_num_blocks))

    @property
    def num_active_blocks(self):
        return len(self.active_blocks)

    def realloc(self, num_blocks):
        requested_num_blocks = int(num_blocks)
        allocatable_cap = self.get_allocatable_max_num_blocks()

        logger.info(
            f"Requested realloc to {requested_num_blocks} KV blocks. "
            f"page_table_max_num_blocks={self.page_table_max_num_blocks}, "
            f"allocatable_max_num_blocks={allocatable_cap}, "
            f"infer.max_batch_size={get_global_args().infer.max_batch_size}, "
            f"infer.max_seq_len={get_global_args().infer.max_seq_len}, "
            f"prefix_caching={self.enable_prefix_caching}"
        )

        if requested_num_blocks < 0:
            raise ValueError(f"num_blocks must be >= 0, got {requested_num_blocks}")

        self.num_blocks = min(requested_num_blocks, allocatable_cap)

        logger.info(
            f"Reallocating KV cache manager to {self.num_blocks} blocks, "
            f"each of size {self.block_size}"
        )

        self.free_cache_ids = deque(range(self.num_blocks))
        self.active_blocks.clear()
        self.cached_idle_blocks.clear()

        # after warmup, clear status
        self.task_to_cache_ids.clear()
        self.task_to_token_blocks.clear()
        self.identity_runtime_pool.clear()
        self.cache_idx_to_hash.clear()

    def num_cached_blocks(self, task: "Task") -> int:
        """Number of contiguous cached blocks hit from prompt start."""
        num_computed_blocks = len(self.task_to_cache_ids[task.task_id])
        if not self.enable_prefix_caching or task.task_type == TaskType.Decode:
            return num_computed_blocks

        # 二分查找第一个cache_idx为None的block序号（LRU逐出策略确保cached blocks连续）
        task_identities = [
            identity
            for identity in self.identity_builder.make_identity_chain(
                task, canonical_prefix_hashes=self.enable_prefix_caching
            )
        ]
        left = num_computed_blocks
        right = len(task_identities)
        while left < right:
            mid = (left + right) // 2
            if task_identities[mid].blk_hash not in self.identity_runtime_pool:
                right = mid
            else:
                left = mid + 1
        return left

    def num_cached_idle_blocks(
        self, task: "Task", *, max_cached_token_len: Optional[int] = None
    ) -> int:
        """Number of idle cached blocks inside the contiguous cached prefix."""
        if not self.enable_prefix_caching:
            return 0
        if max_cached_token_len is None:
            max_cached_token_len = task.prefix_tokens_len
        if max_cached_token_len <= 0:
            return 0

        num_computed_blocks = len(self.task_to_cache_ids[task.task_id])
        num_cached_blocks = min(
            ceil_div(max_cached_token_len, self.block_size),
            self.num_cached_blocks(task),
        )

        if num_computed_blocks >= num_cached_blocks:
            return 0

        # 二分查找第一个idle block序号（同一任务前缀中的active/idle块连续）
        task_identities = [
            identity
            for identity in self.identity_builder.make_identity_chain(
                task, canonical_prefix_hashes=self.enable_prefix_caching
            )
        ]

        left = num_computed_blocks
        right = num_cached_blocks
        while left < right:
            mid = (left + right) // 2
            blk_hash = task_identities[mid].blk_hash
            if (
                blk_hash in self.identity_runtime_pool
                and self.identity_runtime_pool[blk_hash].cache_idx
                in self.cached_idle_blocks
            ):
                right = mid
            else:
                left = mid + 1
        return num_cached_blocks - left

    def upload_to_identity_runtime_pool(self, block: TokenBlock) -> None:
        """将TokenBlock中的[identity,runtime]信息不覆盖的同步到self.identity_runtime_pool"""
        blk_hash = block.blk_hash
        if (
            self.enable_prefix_caching
            and blk_hash is not None
            and len(block.tokens) == block.blk_size
            and blk_hash not in self.identity_runtime_pool
            and block.cache_idx is not None
        ):
            # 不覆盖同步[identity,runtime]到self.identity_runtime_pool
            self.identity_runtime_pool[blk_hash] = block.runtime
            self.cache_idx_to_hash[block.cache_idx] = blk_hash

    def get_free_cache_idx(self):
        if self.free_cache_ids:
            return self.free_cache_ids.popleft()
        if not self.cached_idle_blocks:
            raise Exception(
                f"No more free KVCache blocks: KVCache has total {self.num_blocks} blocks, {len(self.active_blocks)} blocks has been used."
            )
        cache_idx, runtime_evicted = self.cached_idle_blocks.popitem(last=False)
        runtime_evicted.cache_idx = None
        blk_hash = self.cache_idx_to_hash.pop(cache_idx, None)
        self.identity_runtime_pool.pop(blk_hash, None)
        self.identity_builder.forget_hash(blk_hash)
        dp_config = getattr(get_global_args(), "dp_config", None)
        if (
            blk_hash is not None
            and dp_config
            and getattr(dp_config, "enabled", False)
            and not getattr(
                getattr(getattr(dp_config, "router", None), "pd_disaggregation", None),
                "enabled",
                True,
            )
        ):
            self.evicted_blk_hashes.append(blk_hash)
        if (
            blk_hash is not None
            and self.identity_runtime_pool.get(blk_hash) is runtime_evicted
        ):
            self.identity_runtime_pool.pop(blk_hash, None)
        return cache_idx

    def pop_evicted_blk_hashes(self, max_items: int = 512) -> list[str]:
        # Bounded pop to avoid oversized stats payload.
        if max_items <= 0 or not self.evicted_blk_hashes:
            return []
        out = []
        while self.evicted_blk_hashes and len(out) < max_items:
            out.append(self.evicted_blk_hashes.popleft())
        return out

    def prepare_metadata_before_prefill(self, task: "Task") -> list[int]:
        """Prepare cache-manager metadata for one prefill step.

        Preconditions (set by the scheduler before calling):
            - ``task.consumed_req_tokens`` is the scheduler watermark for how many
              prefix tokens are already covered by prefix hits and/or prior Prefill
              steps (after any full-prompt clamp).

        Returns:
            New cache indices allocated or reactivated for this step.
        """
        new_cache_ids: list[int] = []
        task_identities = self.identity_builder.make_identity_chain(
            task, canonical_prefix_hashes=self.enable_prefix_caching
        )
        task_num_cached_blocks = ceil_div(task.consumed_req_tokens, self.block_size)
        assert task_num_cached_blocks <= len(
            task_identities
        ), f"{task_num_cached_blocks} vs {len(task_identities)}, task_id={task.task_id}"

        if self.enable_prefix_caching:
            # 被prefix caching击中block不占chunk prefill size的容量，也不增加额外的kv cache block需求
            for idx in range(
                len(self.task_to_cache_ids[task.task_id]), task_num_cached_blocks
            ):
                identity = task_identities[idx]
                assert (
                    identity.blk_hash in self.identity_runtime_pool
                ), f"identity should be in cache_manager.identity_runtime_pool in scheduler's view"

                # 从identity_runtime_pool获取runtime，并更新runtime信息
                runtime = self.identity_runtime_pool[identity.blk_hash]
                block = TokenBlock(identity=identity, runtime=runtime)
                block.runtime.active_cnt += 1
                assert (
                    block.runtime.state == KVBlockState.ACTIVE
                ), f"{block.runtime.state.value} vs {KVBlockState.ACTIVE.value}"
                self.cached_idle_blocks.pop(block.cache_idx, None)
                self.active_blocks[block.cache_idx] = block.runtime

                # 更新任务状态
                new_cache_ids.append(block.cache_idx)
                self.task_to_cache_ids[task.task_id].add(block.cache_idx)
                self.task_to_token_blocks[task.task_id].append(block)

        assert (
            len(self.task_to_cache_ids.get(task.task_id, set()))
            == task_num_cached_blocks
        ), f"task_id={task.task_id}: {len(self.task_to_cache_ids.get(task.task_id,set()))} vs {task_num_cached_blocks}"

        # [0,num_target_blocks)区间内的block均需要分配cache块索引
        target_seq_len = task.kv_cache_len_used_in_completed_steps_and_next_step
        num_target_blocks = ceil_div(target_seq_len, self.block_size)

        for idx in range(
            len(self.task_to_cache_ids[task.task_id]), num_target_blocks, 1
        ):
            identity = task_identities[idx]
            cache_idx = self.get_free_cache_idx()
            assert (
                cache_idx not in self.task_to_cache_ids[task.task_id]
            ), f"{cache_idx} is already in {self.task_to_cache_ids[task.task_id]}"
            new_cache_ids.append(cache_idx)
            block = TokenBlock(identity, runtime=BlockRuntime(cache_idx, active_cnt=1))
            self.active_blocks[block.cache_idx] = block.runtime
            self.upload_to_identity_runtime_pool(block)
            self.task_to_cache_ids[task.task_id].add(cache_idx)
            self.task_to_token_blocks[task.task_id].append(block)
        return new_cache_ids

    def prepare_metadata_before_decode(self, task: "Task") -> list[int]:
        """prepare and update metadata before the task begin a decode step"""
        new_cache_ids: list[int] = []

        target_seq_len = task.kv_cache_len_used_in_completed_steps_and_next_step
        num_target_blocks = ceil_div(target_seq_len, self.block_size)
        token_blocks = self.task_to_token_blocks.get(task.task_id, [])
        assert len(token_blocks) == len(
            self.task_to_cache_ids[task.task_id]
        ), f"{len(token_blocks)} vs {len(self.task_to_cache_ids[task.task_id])}"

        # Prefix-caching 满块 identity / 与池同步在任务结束时于 finalize_metadata_all_decode 统一做，
        # 避免 decode 过程中把与其它请求可能碰撞的 blk_hash 过早挂进全局 pool
        for idx in range(
            len(self.task_to_cache_ids[task.task_id]), num_target_blocks, 1
        ):
            cache_idx = self.get_free_cache_idx()
            new_cache_ids.append(cache_idx)
            # New generated TokenBlock in next decode step
            block = TokenBlock(
                identity=self.identity_builder.make_identity(
                    [], canonical_prefix_hashes=False
                ),  # 仅占位，在任务结束时再维护identity_runtime_pool
                runtime=BlockRuntime(cache_idx, active_cnt=1),
            )
            self.active_blocks[block.cache_idx] = block.runtime
            self.task_to_cache_ids[task.task_id].add(cache_idx)
            self.task_to_token_blocks[task.task_id].append(block)
        return new_cache_ids

    def finalize_metadata_all_decode(self, task: "Task"):
        if task.task_id not in self.task_to_token_blocks:
            return

        # 任务结束时用已同步的 prefix 一次性补全满块 identity，
        # 便于后续请求的 prefix cache 命中 identity_builder.hashed_block_pool / identity_runtime_pool。

        # 由于decode阶段未维护哈希链，因此需要重建
        identities = self.identity_builder.make_identity_chain(
            task, canonical_prefix_hashes=self.enable_prefix_caching
        )
        cached_blocks = self.task_to_token_blocks[task.task_id]
        num_cached_blocks = len(cached_blocks)
        assert num_cached_blocks <= len(
            identities
        ), f"{num_cached_blocks} vs {len(identities)}"

        # 更新blockruntime、self.active_blocks、self.cached_idle_blocks以及self.identity_runtime_pool
        for i in range(num_cached_blocks - 1, -1, -1):
            # 倒序遍历cached_blocks，确保lru逐出顺序为: 先逐出后缀、再逐出前缀
            blk_hash = identities[i].blk_hash
            runtime = cached_blocks[i].runtime

            runtime.active_cnt -= 1
            assert (
                runtime.active_cnt >= 0
            ), f"TokenBlock.activte_cnt ({runtime.active_cnt}) shouldn't smaller than 0."

            if runtime.state == KVBlockState.IDLE:
                self.active_blocks.pop(runtime.cache_idx)
                self.cached_idle_blocks[runtime.cache_idx] = runtime
                self.cached_idle_blocks.move_to_end(runtime.cache_idx, last=True)

            if blk_hash is not None:
                if (
                    blk_hash not in self.identity_runtime_pool
                    and self.enable_prefix_caching
                ):
                    self.identity_runtime_pool[blk_hash] = runtime
                    self.cache_idx_to_hash[runtime.cache_idx] = blk_hash
                elif (
                    blk_hash in self.identity_runtime_pool
                    and self.identity_runtime_pool[blk_hash] is not runtime
                ):
                    # task侧的runtime与identity_runtime_pool侧不一致时，task侧的runtime永远不会被引用到，因此移至左侧最先被逐出
                    self.cached_idle_blocks.move_to_end(runtime.cache_idx, last=False)
                    blk_visible_runtime = self.identity_runtime_pool[blk_hash]
                    if blk_visible_runtime.cache_idx in self.cached_idle_blocks:
                        self.cached_idle_blocks.move_to_end(
                            blk_visible_runtime.cache_idx, last=True
                        )
                    else:
                        assert blk_visible_runtime.cache_idx in self.active_blocks

        self.task_to_cache_ids.pop(task.task_id)
        self.task_to_token_blocks.pop(task.task_id)
        self.identity_builder.forget_task(task)
