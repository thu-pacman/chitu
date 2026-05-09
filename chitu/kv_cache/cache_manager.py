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
    BlockIdentity,
    BlockIdentityChainBuilder,
    BlockRuntime,
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

        self.tid_to_cached_len: dict[str, int] = (
            {}
        )  # {task_id: task_consumed_req_tokens}
        self.block_size = block_size
        self.task_to_cache_ids: defaultdict[str, set[int]] = defaultdict(
            set
        )  # {task_id: {block_idx,...}}
        self.task_to_token_blocks: dict[str, list[TokenBlock]] = {}
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
        # 保存满容量块的不可变身份对象, 数据含义: {blk_hash: BlockIdentity}
        self.hashed_block_pool: WeakValueDictionary[str, BlockIdentity] = (
            WeakValueDictionary()
        )
        self.identity_builder = BlockIdentityChainBuilder(
            block_size=self.block_size,
            existing_identities=self.hashed_block_pool,
        )
        # 保存块身份到运行时状态的映射, 数据含义: {blk_hash: BlockRuntime}
        self.identity_runtime_pool: WeakValueDictionary[str, BlockRuntime] = (
            WeakValueDictionary()
        )
        self.cache_idx_to_hash: dict[int, str] = {}
        # 保存该任务已生成blk_hash的TokenBLock数量（该任务在self.hashed_block_pool中的TokenBlock数量），仅需在enable_prefix_caching时维护
        # 数据含义: {task_id: cache_hashed_block_cnt}
        self.task_hashed_block_cnt: defaultdict[str, int] = defaultdict(int)
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
        self.tid_to_cached_len.clear()
        self.task_to_cache_ids.clear()
        self.task_to_token_blocks.clear()
        self.task_hashed_block_cnt.clear()
        self.hashed_block_pool = WeakValueDictionary()
        self.identity_builder.existing_identities = self.hashed_block_pool
        self.identity_runtime_pool = WeakValueDictionary()
        self.cache_idx_to_hash.clear()

    def get_task_token_blocks(self, task: "Task") -> list[TokenBlock]:
        return self.task_to_token_blocks.get(task.task_id, [])

    def drop_task_token_blocks(self, task: "Task") -> None:
        self.task_to_token_blocks.pop(task.task_id, None)

    def prompt_to_token_block(self, task: "Task") -> list[TokenBlock]:
        """
        将prompt转化为TokenBlock列表，如：
        block_size: 4
        prompt: [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4]
        chunk: [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4]]
        返回4个TokenBlock实例组成的列表，最后一个实例未满，因此其blk_hash为None
        """
        identities = self.identity_builder.build(
            tokens=task.prefix_tokens,
            auto_register=self.enable_prefix_caching,
        )
        token_blocks = [
            TokenBlock(identity=identity, runtime=BlockRuntime())
            for identity in identities
        ]
        if self.enable_prefix_caching:
            for block in token_blocks:
                self.refresh_prefix_cache_block(block)
        self.task_to_token_blocks[task.task_id] = token_blocks
        return token_blocks

    def ensure_task_token_blocks(self, task: "Task") -> list[TokenBlock]:
        """Ensure the task's token blocks are generated by this cache_manager and that all block runtime information is up to date."""
        token_blocks = self.get_task_token_blocks(task)
        if len(token_blocks) < ceil_div(len(task.prefix_tokens), self.block_size):
            return self.prompt_to_token_block(task)

        if self.enable_prefix_caching:
            for block in token_blocks[len(self.task_to_cache_ids[task.task_id]) :]:
                self.sync_prefix_cache_runtime(block)
        return token_blocks

    def num_cached_blocks(self, task: "Task") -> int:
        """Number of contiguous cached blocks hit from prompt start."""
        num_computed_blocks = len(self.task_to_cache_ids[task.task_id])
        if not self.enable_prefix_caching:
            return num_computed_blocks

        # 二分查找第一个cache_idx为None的block序号（LRU逐出策略确保cached blocks连续）
        task_token_blocks = self.get_task_token_blocks(task)
        left = num_computed_blocks
        right = len(task_token_blocks)
        while left < right:
            mid = (left + right) // 2
            if task_token_blocks[mid].cache_idx is None:
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
        task_token_blocks = self.get_task_token_blocks(task)
        left = num_computed_blocks
        right = num_cached_blocks
        while left < right:
            mid = (left + right) // 2
            if task_token_blocks[mid].cache_idx in self.cached_idle_blocks:
                right = mid
            else:
                left = mid + 1
        return num_cached_blocks - left

    def canonicalize_prefix_identity(self, block: TokenBlock) -> BlockIdentity:
        """Use the shared prefix identity object for full cacheable blocks."""
        if not self.enable_prefix_caching or len(block.tokens) != block.blk_size:
            return block.identity

        identity = self.identity_builder.make_identity(
            token_chunk=list(block.identity.tokens),
            pre_blk_hash=block.identity.pre_blk_hash,
            auto_register=True,
        )
        block.set_identity(identity)

    def sync_prefix_cache_runtime(self, block: TokenBlock) -> None:
        """Attach or register runtime state for a cacheable prefix block."""
        blk_hash = block.blk_hash
        if (
            not self.enable_prefix_caching
            or blk_hash is None
            or len(block.tokens) != block.blk_size
        ):
            return
        runtime = self.identity_runtime_pool.get(blk_hash)
        if runtime is not None:
            if block.cache_idx is not None:
                # 若当前block已绑定cache_idx，则与复用runtime的关键状态必须一致，否则状态已漂移。
                assert block.runtime is runtime or (
                    block.cache_idx == runtime.cache_idx
                    and block.active_cnt == runtime.active_cnt
                ), f"Inconsistent runtime object: {block.runtime} vs {runtime}"
            block.set_runtime(runtime)
        elif block.cache_idx is not None:
            self.identity_runtime_pool[blk_hash] = block.runtime
            self.cache_idx_to_hash[block.cache_idx] = blk_hash

    def refresh_prefix_cache_block(self, block: TokenBlock) -> BlockIdentity:
        """Refresh prefix identity and runtime sharing for a task-side block."""
        self.canonicalize_prefix_identity(block)
        self.sync_prefix_cache_runtime(block)
        return block.identity

    def _bind_block_to_cache_idx(self, block: TokenBlock, cache_idx: int) -> None:
        runtime = block.runtime
        runtime.cache_idx = cache_idx
        runtime.active_cnt += 1
        self.active_blocks[cache_idx] = runtime
        if block.blk_hash is not None and len(block.tokens) == block.blk_size:
            self.identity_runtime_pool[block.blk_hash] = runtime
            self.cache_idx_to_hash[cache_idx] = block.blk_hash

    def _build_task_block(
        self, tokens: list[int], pre_blk_hash: Optional[str]
    ) -> TokenBlock:
        """Build task block via builder abstractions."""
        if pre_blk_hash is None:
            pre_blk_hash = NONE_BLK_HASH
        tokens = list(tokens)
        identity = self.identity_builder.make_identity(
            token_chunk=tokens,
            pre_blk_hash=pre_blk_hash,
            auto_register=self.enable_prefix_caching,
        )
        block = TokenBlock(identity=identity, runtime=BlockRuntime())
        if self.enable_prefix_caching:
            self.refresh_prefix_cache_block(block)
        return block

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
        dp_config = getattr(get_global_args(), "dp_config", None)
        if (
            dp_config
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

    def prepare_metadata_before_prefill(
        self,
        task: "Task",
        *,
        max_cached_token_len: Optional[int] = None,
    ) -> list[int]:
        """Prepare cache-manager metadata for one prefill step.

        Returns:
            list[int]:
                - new cache ids allocated/taken by this manager
        """
        consumed_req_tokens = task.consumed_req_tokens
        new_cache_ids: list[int] = []
        if max_cached_token_len is None:
            max_cached_token_len = task.prefix_tokens_len
        max_cached_token_len = min(max_cached_token_len, task.prefix_tokens_len)

        if self.enable_prefix_caching:
            # 被prefix caching击中block不占chunk prefill size的容量，也不增加额外的kv cache block需求
            token_blocks = self.get_task_token_blocks(task)
            for idx in range(
                len(self.task_to_cache_ids[task.task_id]), len(token_blocks)
            ):
                if consumed_req_tokens >= max_cached_token_len:
                    break
                block = token_blocks[idx]

                # 如果block未被prefix caching块击中，停止遍历
                if (
                    block.cache_idx not in self.cached_idle_blocks
                    and block.cache_idx not in self.active_blocks
                ):
                    break

                # block被空闲的prefix caching块击中，空闲块变为活跃块
                if block.cache_idx in self.cached_idle_blocks:
                    assert (
                        block.active_cnt == 0
                    ), f"The active count of cached idle block ({block.active_cnt}) should 0. "
                    self.active_blocks[block.cache_idx] = block.runtime
                    self.cached_idle_blocks.pop(block.cache_idx)

                # 更新块的活跃状态、当前step新分配给任务的kvcache块索引，以及任务的所有kvcache块索引
                self.task_to_cache_ids[task.task_id].add(block.cache_idx)
                new_cache_ids.append(block.cache_idx)
                block.runtime.active_cnt += 1
                consumed_req_tokens += min(
                    self.block_size,
                    max_cached_token_len - consumed_req_tokens,
                )

            if consumed_req_tokens == task.prefix_tokens_len:
                # prompt被完全击中，此时只需对最后一个token做prefill step获取logits, 但无需额外的kv cache block
                assert task.prefill_chunk_size == 1
                consumed_req_tokens -= 1

        num_full_blocks = consumed_req_tokens // self.block_size
        self.update_prefix_caching_metadata(task, num_full_blocks)

        # [0,num_target_blocks)区间内的block均需要分配cache块索引
        self.tid_to_cached_len[task.task_id] = (
            consumed_req_tokens + task.next_req_tokens_len
        )
        num_target_blocks = ceil_div(
            self.tid_to_cached_len[task.task_id], self.block_size
        )
        token_blocks = self.get_task_token_blocks(task)
        for idx in range(
            len(self.task_to_cache_ids[task.task_id]), num_target_blocks, 1
        ):
            block = token_blocks[idx]
            assert (
                block.cache_idx is None
            ), f"[DP {self.dp_rank}] task_id: {task.task_id}, idx:{idx}, tasks.cached_blocks:{[blk.cache_idx for blk in token_blocks]}, cache_manager.task_to_cache_ids:{self.task_to_cache_ids[task.task_id]}"
            cache_idx = self.get_free_cache_idx()
            assert (
                cache_idx not in self.task_to_cache_ids[task.task_id]
            ), f"{cache_idx} is already in {self.task_to_cache_ids[task.task_id]}"
            new_cache_ids.append(cache_idx)
            self._bind_block_to_cache_idx(block, cache_idx)
            self.task_to_cache_ids[task.task_id].add(cache_idx)
        return new_cache_ids

    def prepare_metadata_before_decode(self, task: "Task") -> list[int]:
        """prepare and update metadata before the task begin a decode step"""
        new_cache_ids: list[int] = []

        # DLLM decode: use decoding_start + block_length
        from chitu.models.registry import ModelType

        args = get_global_args()
        if hasattr(args, "models") and args.models.type == ModelType.LLADA2:
            return self._prepare_metadata_before_decode_dllm(task)

        # NOTE:
        # For prefix caching, hashes must be derived from tokens that are already synced into
        # `task.prefix_tokens`. Under schedule_overlap and/or PP, `task.prefix_tokens_len`
        # may temporarily include pending unsynced tokens , while `task.prefix_tokens` has not
        # been extended yet.
        num_full_blocks = max(0, (len(task.prefix_tokens) - 1) // self.block_size)
        self.update_prefix_caching_metadata(task, num_full_blocks)

        # task.prefix_tokens中的最后一个token(下一个step输入的token)还未缓存，因此需要减1
        self.tid_to_cached_len[task.task_id] = (
            task.prefix_tokens_len - 1 + self.mtp_size
        )
        num_target_blocks = ceil_div(
            self.tid_to_cached_len[task.task_id], self.block_size
        )
        token_blocks = self.get_task_token_blocks(task)

        for idx in range(
            len(self.task_to_cache_ids[task.task_id]), num_target_blocks, 1
        ):
            if idx < len(token_blocks):
                # TokenBlock already exists (created during prefill), but may not have cache idx yet.
                block = token_blocks[idx]
                assert (
                    block.cache_idx is None
                ), f"idx:{idx}, block.cache_idx:{block.cache_idx}, token_blocks:{token_blocks}"
            else:
                # New generated TokenBlock in next decode step
                tokens = task.prefix_tokens[
                    idx * self.block_size : (idx + 1) * self.block_size
                ]
                pre_blk_hash = token_blocks[-1].blk_hash
                block = self._build_task_block(tokens=tokens, pre_blk_hash=pre_blk_hash)
                assert (
                    block.cache_idx is None
                ), f"idx:{idx}, token_blocks:{token_blocks}"
                token_blocks.append(block)

            cache_idx = self.get_free_cache_idx()
            assert (
                cache_idx not in self.task_to_cache_ids[task.task_id]
            ), f"{cache_idx} is already in {self.task_to_cache_ids[task.task_id]}"
            self._bind_block_to_cache_idx(block, cache_idx)
            self.task_to_cache_ids[task.task_id].add(cache_idx)
            new_cache_ids.append(cache_idx)
        return new_cache_ids

    def _prepare_metadata_before_decode_dllm(self, task: "Task") -> list[int]:
        """Prepare metadata for DLLM decode step.

        DLLM decode uses decoding_start + block_length to determine cache requirements.
        Unlike normal decode, DLLM decode processes a full block of tokens per step.
        """
        new_cache_ids: list[int] = []

        block_length = task.block_length
        decoding_start = task.decoding_start
        target_seq_len = decoding_start + block_length

        self.tid_to_cached_len[task.task_id] = target_seq_len
        num_target_blocks = ceil_div(target_seq_len, self.block_size)
        token_blocks = self.get_task_token_blocks(task)

        for idx in range(
            len(self.task_to_cache_ids[task.task_id]), num_target_blocks, 1
        ):
            cache_idx = self.get_free_cache_idx()
            self.task_to_cache_ids[task.task_id].add(cache_idx)
            new_cache_ids.append(cache_idx)

            # Create a placeholder TokenBlock if needed
            # DLLM decode doesn't use prefix_tokens for token tracking,
            # so we create placeholder blocks for metadata management only.
            if idx < len(token_blocks):
                block = token_blocks[idx]
            else:
                # Create a placeholder block for DLLM decode
                block = TokenBlock(
                    tokens=[0]
                    * self.block_size,  # Placeholder tokens (not used for DLLM)
                    blk_hash=None,
                    blk_size=self.block_size,
                    pre_blk_hash=(
                        token_blocks[-1].blk_hash
                        if (token_blocks and token_blocks[-1].blk_hash is not None)
                        else NONE_BLK_HASH
                    ),
                )
                token_blocks.append(block)

            self._bind_block_to_cache_idx(block, cache_idx)
        return new_cache_ids

    def update_prefix_caching_metadata(self, task: "Task", num_full_blocks):
        """Update metadata in task_hashed_block_cnt, token blocks and hashed_block_pool"""
        if not self.enable_prefix_caching:
            return
        num_hashed_blocks = self.task_hashed_block_cnt[task.task_id]
        token_blocks = self.get_task_token_blocks(task)
        for idx in range(num_hashed_blocks, num_full_blocks, 1):
            block = token_blocks[idx]
            if idx == 0:
                # block是task的第一个tokenblock
                pre_blk_hash = NONE_BLK_HASH
            elif block.pre_blk_hash != NONE_BLK_HASH:
                # block被prefix caching击中，原本已生成pre_blk_hash和blk_hash
                pre_blk_hash = block.pre_blk_hash
            else:
                # block未被prefix caching击中
                pre_blk_hash = token_blocks[idx - 1].blk_hash
                assert (
                    pre_blk_hash is not None
                ), f"idx:{idx}, pre_block: {token_blocks[idx-1]} , token_blocks:{token_blocks}"
            block.update_identity(pre_blk_hash=pre_blk_hash, blk_hash=None)
            if len(block.tokens) != self.block_size:
                # decode
                block.update_identity(
                    tokens=task.prefix_tokens[
                        idx * self.block_size : (idx + 1) * self.block_size
                    ],
                    blk_hash=None,
                )
            self.refresh_prefix_cache_block(block)
            if idx > 0:
                assert block.pre_blk_hash == token_blocks[idx - 1].blk_hash, (
                    f"hash chain broken for task {task.task_id} idx={idx}: "
                    f"pre={block.pre_blk_hash}, prev={token_blocks[idx - 1].blk_hash}"
                )
        self.task_hashed_block_cnt[task.task_id] = num_full_blocks

    def finalize_metadata_all_decode(self, task: "Task"):
        if task.task_id not in self.tid_to_cached_len:
            self.drop_task_token_blocks(task)
            self.task_hashed_block_cnt.pop(task.task_id, None)
            return
        for block in reversed(self.get_task_token_blocks(task)):
            if block.cache_idx is None:
                continue
            block.runtime.active_cnt -= 1
            assert (
                block.active_cnt >= 0
            ), f"TokenBlock.activte_cnt ({block.active_cnt}) shouldn't smaller than 0."
            if block.active_cnt == 0:
                # 当block的活跃度为0时，将block从active_blocks中移除，并放入cached_idle_blocks的最右端
                self.active_blocks.pop(block.cache_idx)
                self.cached_idle_blocks[block.cache_idx] = block.runtime
        self.tid_to_cached_len.pop(task.task_id)
        self.task_to_cache_ids.pop(task.task_id)
        self.drop_task_token_blocks(task)
        if task.task_id in self.task_hashed_block_cnt:
            self.task_hashed_block_cnt.pop(task.task_id)
