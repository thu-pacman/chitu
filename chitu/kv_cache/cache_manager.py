# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, TYPE_CHECKING
from logging import getLogger
from collections import deque, OrderedDict

from chitu.global_vars import get_global_args, is_independent_multi_inst
from chitu.task_type import TaskType
from chitu.utils import ceil_div
from chitu.kv_cache.prefix_caching import (
    TokenBlock,
    BlockIdentityChainBuilder,
    BlockRuntime,
    KVBlockState,
)
from collections import defaultdict

if TYPE_CHECKING:
    from chitu.task import Task

logger = getLogger(__name__)


class KVCacheManagerBase:
    """Base class for KV cache managers — the scheduler-side counterpart of KVCacheBase.

    While ``KVCacheBase`` owns the GPU tensor storage, ``KVCacheManagerBase``
    owns the allocation metadata: which blocks are assigned to which task,
    which blocks are free, and (with prefix caching) which blocks are shared.
    The scheduler uses the manager for admission control and to populate the
    block metadata that the executor later consumes in ``prepare_cache_*``.
    """

    def publish_prefix_cache_blocks(self, *args, **kwargs) -> None:
        pass


class PagedKVCacheManager(KVCacheManagerBase):
    """Block allocator for paged KV caches with optional prefix caching.

    Maintains:
    - ``task_to_cache_ids``: maps each task to its allocated physical block IDs.
    - Free block pool (reclaimed when tasks finish or blocks are evicted).
    - Optional prefix-cache metadata: tracks block hashes and reference counts
      to enable block sharing across requests with shared prefixes.
    """

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

        # PD分离Decode-Only: 记录KV传输+reorder尚未就绪的prefix block，
        # 数据含义: {task_id: target_seq_len}。在该任务首个decode step
        # (prepare_metadata_before_decode)发布，届时recv_kv_cache_and_insert
        # (KV插入 + kv_recv_reorder)必然已完成。
        self._deferred_prefix_publish: dict[str, int] = {}

    def has_active_blocks(self):
        if len(self.active_blocks) > 0:
            return True
        assert (
            not self.task_to_cache_ids
        ), f"something wrong here, task_to_cache_ids should be empty, found: {self.task_to_cache_ids}"
        return False

    def clear_prefix_cache(self):
        """Drop idle prefix-cache metadata"""
        assert (
            not self.has_active_blocks()
        ), f"cache_manager is busy, can't clear prefix cache metadata."
        self.free_cache_ids = deque(range(self.num_blocks))

        self.active_blocks.clear()
        self.cached_idle_blocks.clear()
        self.evicted_blk_hashes.clear()
        self.task_to_cache_ids.clear()
        self.task_to_token_blocks.clear()
        self.identity_runtime_pool.clear()
        self.cache_idx_to_hash.clear()
        self._deferred_prefix_publish.clear()
        self.identity_builder.hashed_block_pool.clear()
        self.identity_builder.tid_to_identities.clear()

    def get_allocatable_max_num_blocks(self) -> int:
        return int(getattr(self, "allocatable_max_num_blocks", self.max_num_blocks))

    def num_blocks_for_seq_len(self, seq_len: int) -> int:
        return ceil_div(int(seq_len), self.block_size)

    def _make_task_identities(
        self,
        task: "Task",
        *,
        required_identity_blocks: Optional[int] = None,
    ):
        """Build logical KV block identities for a task.

        The default manager derives identities from ``task.prefix_tokens``. Some
        logical managers, such as DeepSeek-V4 sliding/compressed caches, may need
        placeholder identities even when the task token list is shorter than the
        number of metadata blocks being prepared; those managers honor
        ``required_identity_blocks`` in their override.
        """
        return self.identity_builder.make_identity_chain(
            task, canonical_prefix_hashes=self.enable_prefix_caching
        )

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

    def _num_computed_blocks(self, task: "Task") -> int:
        """Prefill only: fully-written block count (excludes a partial last block).

        Decode callers should use ``len(task_to_cache_ids[...])`` directly; that
        count may include a non-full trailing block and must not go through here.
        """
        n = len(self.task_to_cache_ids.get(task.task_id, set()))
        return min(n, task.kv_cache_len_used_in_completed_steps // self.block_size)

    def num_cached_blocks(self, task: "Task") -> int:
        """Number of contiguous cached blocks hit from prompt start."""
        if not self.enable_prefix_caching or task.task_type == TaskType.Decode:
            return len(self.task_to_cache_ids.get(task.task_id, set()))

        # Prefill: start binary search from fully-computed blocks only.
        left = self._num_computed_blocks(task)

        # 二分查找第一个cache_idx为None的block序号（LRU逐出策略确保cached blocks连续）
        num_needed_blocks = self.num_blocks_for_seq_len(task.prefix_tokens_len)
        task_identities = self._make_task_identities(
            task, required_identity_blocks=num_needed_blocks
        )
        right = min(num_needed_blocks, len(task_identities))
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

        if task.task_type == TaskType.Decode:
            num_computed_blocks = len(self.task_to_cache_ids.get(task.task_id, set()))
        else:
            num_computed_blocks = self._num_computed_blocks(task)
        num_cached_blocks = min(
            self.num_blocks_for_seq_len(max_cached_token_len),
            self.num_cached_blocks(task),
        )

        if num_computed_blocks >= num_cached_blocks:
            return 0

        # 二分查找第一个idle block序号（同一任务前缀中的active/idle块连续）
        task_identities = self._make_task_identities(
            task, required_identity_blocks=num_cached_blocks
        )

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
        multi_inst = getattr(get_global_args(), "multi_inst", None)
        role = getattr(multi_inst, "role", None)
        # Update evicted_blk_hashes for routers to sync shadow cache in cache-aware routing policy.
        # Independent multi-inst uses a shared policy namespace; classic PD has separate
        # prefill/decode policy instances, so both roles must report their own evictions.
        if (
            blk_hash is not None
            and multi_inst
            and getattr(multi_inst, "n_insts", 1) > 1
            and (is_independent_multi_inst() or role in ("prefill", "decode"))
        ):
            # Only record where a consumer pops: enhanced-scheduler stats loop
            # (independent multi-inst) or PD get_pd_stats.
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

    def prepare_metadata_before_prefill(
        self, task: "Task", *, eager_prefix_cache_insert: bool = True
    ) -> list[int]:
        """Prepare cache-manager metadata for one prefill step.

        Preconditions (set by the scheduler before calling):
            - ``task.consumed_req_tokens`` is the scheduler watermark for how many
              prefix tokens are already covered by prefix hits and/or prior Prefill
              steps (after any full-prompt clamp).

        Returns:
            New cache indices allocated or reactivated for this step.
        """
        new_cache_ids: list[int] = []
        task_num_cached_blocks = self.num_blocks_for_seq_len(task.consumed_req_tokens)
        target_seq_len = task.kv_cache_len_used_in_completed_steps_and_next_step
        num_target_blocks = self.num_blocks_for_seq_len(target_seq_len)
        num_identity_blocks = max(task_num_cached_blocks, num_target_blocks)
        task_identities = self._make_task_identities(
            task, required_identity_blocks=num_identity_blocks
        )
        assert num_identity_blocks <= len(task_identities), (
            f"{num_identity_blocks} vs {len(task_identities)}, "
            f"task_id={task.task_id}"
        )

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
            self.task_to_cache_ids[task.task_id].add(cache_idx)
            self.task_to_token_blocks[task.task_id].append(block)

        if eager_prefix_cache_insert:
            # 非PD-Decode-Only节点调用prepare_metadata_before_prefill后立刻进入model_run，可视为cache就绪可复用
            self.publish_prefix_cache_blocks(task, target_seq_len)
        else:
            # PD分离Decode-Only节点：此时KV cache还未从prefill传输并reorder就绪，
            # 不能视为可复用的cache blocks。推迟到该任务首个decode step
            # (prepare_metadata_before_decode)再发布。
            self._deferred_prefix_publish[task.task_id] = target_seq_len
        return new_cache_ids

    def publish_prefix_cache_blocks(self, task: "Task", target_seq_len: int) -> None:
        for idx, block in enumerate(self.task_to_token_blocks[task.task_id]):
            if (idx + 1) * self.block_size <= target_seq_len:
                self.upload_to_identity_runtime_pool(block)

    def prepare_metadata_before_decode(self, task: "Task") -> list[int]:
        """prepare and update metadata before the task begin a decode step"""
        # PD分离Decode-Only：任务被调度进入首个decode step时，其KV传输与reorder
        # (在同一step稍后的before_decode_step->recv_kv_cache_and_insert中执行)
        # 即将在本step完成，且其它请求最早只能在下一个step的调度阶段
        # (_do_pd_scheduler)观察到这些block，故在此发布prefix block可复用是安全的。
        deferred_seq_len = self._deferred_prefix_publish.pop(task.task_id, None)
        if deferred_seq_len is not None:
            self.publish_prefix_cache_blocks(task, deferred_seq_len)

        new_cache_ids: list[int] = []

        target_seq_len = task.kv_cache_len_used_in_completed_steps_and_next_step
        num_target_blocks = self.num_blocks_for_seq_len(target_seq_len)
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
        cached_blocks = self.task_to_token_blocks[task.task_id]
        num_cached_blocks = len(cached_blocks)
        identities = self._make_task_identities(task)

        # 更新blockruntime、self.active_blocks、self.cached_idle_blocks以及self.identity_runtime_pool
        for i in range(num_cached_blocks - 1, -1, -1):
            # 倒序遍历cached_blocks，确保lru逐出顺序为: 先逐出后缀、再逐出前缀
            # decode 按 mtp 可能多预分配块，超出 prefix identity 链的块不入 prefix pool
            blk_hash = identities[i].blk_hash if i < len(identities) else None
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

        self._deferred_prefix_publish.pop(task.task_id, None)
        self.task_to_cache_ids.pop(task.task_id)
        self.task_to_token_blocks.pop(task.task_id)
        self.identity_builder.forget_task(task)


class _DeepSeekV4LogicalBlockMetadataMixin:
    def _make_task_identities(
        self,
        task: "Task",
        *,
        required_identity_blocks: Optional[int] = None,
    ):
        if self.enable_prefix_caching:
            return super()._make_task_identities(
                task, required_identity_blocks=required_identity_blocks
            )
        if required_identity_blocks is None:
            required_identity_blocks = self.num_blocks_for_seq_len(
                task.prefix_tokens_len
            )
        placeholder = self.identity_builder.make_identity(
            [], canonical_prefix_hashes=False
        )
        return [placeholder for _ in range(max(0, int(required_identity_blocks)))]


class DeepSeekV4SlidingKVCacheManager(
    _DeepSeekV4LogicalBlockMetadataMixin, PagedKVCacheManager
):
    """Scheduler-side allocator for DeepSeek-V4 sliding-window KV pages.

    V4 sliding-window cache uses one physical page per active request. The page
    size is the paged-cache block size, and model code writes tokens into that
    page as a ring buffer with ``position % window_size`` offsets. The model's
    logical visible SWA window may be smaller than this physical page.
    """

    def __init__(
        self,
        num_blocks: int,
        *,
        window_size: int,
        **kwargs,
    ):
        self.window_size = int(window_size)
        if int(kwargs["block_size"]) != self.window_size:
            raise ValueError(
                "DeepSeek-V4 sliding-window manager requires "
                f"block_size == window_size, got block_size={kwargs['block_size']}, "
                f"window_size={self.window_size}"
            )
        fixed_num_blocks = int(kwargs["num_hot_req"])
        if int(num_blocks) != fixed_num_blocks:
            logger.info(
                "DeepSeek-V4 sliding-window manager uses one page per hot "
                "request; ignoring num_blocks=%s and using num_hot_req=%s",
                int(num_blocks),
                fixed_num_blocks,
            )
        super().__init__(num_blocks, **kwargs)
        self.max_blocks_per_req = 1
        self.page_table_max_num_blocks = self.max_blocks_per_req * int(
            kwargs["num_hot_req"]
        )
        self.max_num_blocks = self.page_table_max_num_blocks
        self.num_blocks = fixed_num_blocks
        self.free_cache_ids = deque(range(self.num_blocks))
        self.fixed_num_blocks = True
        if not self.enable_prefix_caching:
            self.allocatable_max_num_blocks = self.page_table_max_num_blocks

    def num_blocks_for_seq_len(self, seq_len: int) -> int:
        seq_len = max(0, int(seq_len))
        if seq_len == 0:
            return 0
        return 1

    def realloc(self, num_blocks):
        if int(num_blocks) != self.page_table_max_num_blocks:
            logger.info(
                "DeepSeek-V4 sliding-window manager keeps one page per hot "
                "request; requested num_blocks=%s, using %s",
                int(num_blocks),
                self.page_table_max_num_blocks,
            )
        super().realloc(self.page_table_max_num_blocks)


class DeepSeekV4CompressedKVCacheManager(
    _DeepSeekV4LogicalBlockMetadataMixin, PagedKVCacheManager
):
    """Scheduler-side allocator for one DeepSeek-V4 compressed KV stream."""

    def __init__(
        self,
        num_blocks: int,
        *,
        compress_ratio: int,
        **kwargs,
    ):
        self.compress_ratio = int(compress_ratio)
        super().__init__(num_blocks, **kwargs)
        self.max_blocks_per_req = max(
            1,
            ceil_div(
                int(kwargs["max_seq_len"]) // self.compress_ratio,
                self.block_size,
            ),
        )
        self.page_table_max_num_blocks = self.max_blocks_per_req * int(
            kwargs["num_hot_req"]
        )
        self.max_num_blocks = self.page_table_max_num_blocks
        if not self.enable_prefix_caching:
            self.allocatable_max_num_blocks = self.page_table_max_num_blocks

    def num_blocks_for_seq_len(self, seq_len: int) -> int:
        seq_len = max(0, int(seq_len))
        compressed_len = seq_len // self.compress_ratio
        if compressed_len == 0:
            return 0
        return ceil_div(compressed_len, self.block_size)


class SingletonPagedKVCacheManager(KVCacheManagerBase):
    """Block allocator for SingletonPagedKVCache.

    Each active request owns exactly one block. Allocation happens once in
    ``prepare_metadata_before_prefill``; decode phase reuses the same block
    and ``prepare_metadata_before_decode`` returns an empty list (no
    incremental allocation).

    The pool size is ``num_hot_req``, which exactly matches the physical
    capacity of ``SingletonPagedKVCache`` (one block per hot request).
    """

    def __init__(
        self,
        *,
        num_hot_req: int,
        manager_name: str,
    ):
        self.num_blocks = int(num_hot_req)
        self.manager_name = manager_name
        self.block_size = 1
        self.max_blocks_per_req = 1

        self.free_cache_ids: deque[int] = deque(range(self.num_blocks))
        self.task_to_cache_ids: defaultdict[str, set[int]] = defaultdict(set)
        """task_id -> set of allocated cache block ids.

        Stored as ``defaultdict[str, set[int]]`` for interface compatibility
        with ``PagedKVCacheManager.task_to_cache_ids`` which is consumed
        directly by ``scheduler.py``.
        """

        # —— interface alignment with PagedKVCacheManager ——
        self.max_num_blocks = self.num_blocks
        self.page_table_max_num_blocks = self.num_blocks
        self.allocatable_max_num_blocks = self.num_blocks
        self.enable_prefix_caching = False

    # ========================
    #   Capacity / queries
    # ========================

    @property
    def num_active_blocks(self) -> int:
        return len(self.task_to_cache_ids)

    def num_blocks_for_seq_len(self, seq_len: int) -> int:
        return 1 if seq_len > 0 else 0

    def num_cached_blocks(self, task) -> int:
        return 0  # linear / mtp caches do not participate in prefix caching

    def num_cached_idle_blocks(self, task, *, max_cached_token_len=None) -> int:
        return 0

    # ========================
    #   Allocation / release
    # ========================

    def prepare_metadata_before_prefill(self, task, *args, **kwargs) -> list[int]:
        """Allocate the single block for the full request lifetime."""
        tid = task.task_id
        if tid not in self.task_to_cache_ids:
            cache_id = self._get_free_cache_id()
            self.task_to_cache_ids[tid] = {cache_id}
            return [cache_id]
        return []

    def prepare_metadata_before_decode(self, task) -> list[int]:
        """Decode phase reuses the block allocated during prefill.

        Returns an empty list — no incremental allocation.  Raises
        ``RuntimeError`` if the task was never seen by
        ``prepare_metadata_before_prefill`` (programmer error).
        """
        tid = task.task_id
        if tid not in self.task_to_cache_ids:
            raise RuntimeError(
                f"SingletonPagedKVCacheManager '{self.manager_name}': "
                f"task '{tid}' reached decode without prefill allocation"
            )
        return []

    def finalize_metadata_all_decode(self, task):
        tid = task.task_id
        cache_ids = self.task_to_cache_ids.pop(tid, None)
        if cache_ids is not None:
            for cache_id in cache_ids:
                self.free_cache_ids.append(cache_id)

    def _get_free_cache_id(self) -> int:
        if not self.free_cache_ids:
            raise RuntimeError(
                f"SingletonPagedKVCacheManager '{self.manager_name}': "
                f"no free blocks (total={self.num_blocks}, "
                f"active={len(self.task_to_cache_ids)})"
            )
        return self.free_cache_ids.popleft()

    def realloc(self, num_blocks: int):
        self.num_blocks = int(num_blocks)
        self.free_cache_ids = deque(range(self.num_blocks))
        self.task_to_cache_ids.clear()
        self.max_num_blocks = self.num_blocks
