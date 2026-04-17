# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, TYPE_CHECKING
from logging import getLogger
from collections import deque, OrderedDict

from chitu.global_vars import get_global_args
from chitu.task_type import TaskType
from chitu.utils import ceil_div
from chitu.kv_cache import TokenBlock, NONE_BLK_HASH
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

        self.tid_to_cached_len: dict[str, int] = (
            {}
        )  # {task_id: task_consumed_req_tokens}
        self.block_size = block_size
        self.task_to_cache_ids: defaultdict[str, set[int]] = defaultdict(
            set
        )  # {task_id: {block_idx,...}}
        self.mtp_size = mtp_size
        self.free_cache_ids: deque[int] = deque(
            range(self.num_blocks)
        )  # list of cache_idx, 保存所有未分配给TokenBlock的cache索引

        # 保存活跃度大于0的TokenBlock, 数据含义: {cache_idx: TokenBlock}
        # 元素生命周期: TokenBlock在prepare_metadata_before_prefill/decode中，初次分配cache_idx或active_cnt从0变为1时，进入，
        #             在finalize_metadata_all_decode中active_cnt为0时移除
        self.active_blocks: dict[int, TokenBlock] = {}

        # 保存活跃度为0，但分配了cache_idx的TokenBlock, 数据含义: {cache_idx: TokenBlock}
        # 元素生命周期: 在调用finalize_metadata_all_decode时，若元素的active_cnt为0，则被放入self.cached_idle_blocks链表最右边
        #             在调用get_free_cache_idx时，若self.free_cache_ids为空，self.cached_idle_blocks链表最左边元素被最先移除
        self.cached_idle_blocks: OrderedDict[int, TokenBlock] = OrderedDict()

        # prefix_caching related
        self.enable_prefix_caching: bool = enable_prefix_caching
        # 保存满容量且已分配cache_idx的TokenBlock块，仅需在enable_prefix_caching时维护, 数据含义: {blk_hash: TokenBlock}
        # 元素生命周期: 在TokenBlock的满容量且cache_idx不为None时移入，在TokenBlock的引用计数为0时移除（python自动支持）
        self.hashed_block_pool: WeakValueDictionary[str, TokenBlock] = (
            WeakValueDictionary()
        )
        # 保存该任务已生成blk_hash的TokenBLock数量（该任务在self.hashed_block_pool中的TokenBlock数量），仅需在enable_prefix_caching时维护
        # 数据含义: {task_id: cache_hashed_block_cnt}
        self.task_hashed_block_cnt: defaultdict[str, int] = defaultdict(int)

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
        self.task_hashed_block_cnt.clear()
        self.hashed_block_pool = WeakValueDictionary()

    def add_to_hashed_block_pool(self, block: TokenBlock) -> TokenBlock:
        """将block添加到hashed_block_pool中
        检查block是否与self.hash_to_block中的block哈希碰撞，若是，修改block的blk_hash值为非碰撞的哈希值, 并返回新的block
        """
        if not self.enable_prefix_caching or len(block.tokens) != block.blk_size:
            return block

        blk_hash = block.blk_hash
        if blk_hash is None:
            blk_hash = TokenBlock.hash_fn((block.pre_blk_hash, tuple(block.tokens)))

        # 哈希碰撞检查
        collision_count = 0
        while blk_hash in self.hashed_block_pool:
            cached_block = self.hashed_block_pool[blk_hash]
            if (
                cached_block.tokens != block.tokens
                or cached_block.pre_blk_hash != block.pre_blk_hash
            ):
                logger.warning(
                    f"检测到哈希碰撞!: \n"
                    f" - blk_hash: {blk_hash} \n"
                    f" - cached_block.tokens: {cached_block.tokens} \n"
                    f" - block.tokens: {block.tokens} \n"
                    f" - cached_block.pre_blk_hash: {cached_block.pre_blk_hash} \n"
                    f" - block.pre_blk_hash: {block.pre_blk_hash} \n"
                )
                collision_count += 1
                blk_hash = TokenBlock.hash_fn(
                    (block.pre_blk_hash, tuple(block.tokens), collision_count)
                )
                continue
            else:
                break

        block.blk_hash = blk_hash
        if blk_hash in self.hashed_block_pool:
            block = self.hashed_block_pool[blk_hash]
        else:
            self.hashed_block_pool[blk_hash] = block

        return block

    def tokens_to_block(
        self, tokens: list[int], pre_blk_hash: Optional[str]
    ) -> TokenBlock:
        """将tokens转化为一个TokenBlock"""
        if pre_blk_hash is None:
            pre_blk_hash = NONE_BLK_HASH
        tokens = list(tokens)

        if len(tokens) <= self.block_size:
            block = TokenBlock(
                tokens=tokens,
                blk_hash=None,
                blk_size=self.block_size,
                pre_blk_hash=pre_blk_hash,
            )
            block = self.add_to_hashed_block_pool(block)
            return block

        raise ValueError(
            f"Input tokens length ({len(tokens)}) shouldn't bigger than block size ({self.block_size})"
        )

    def get_free_cache_idx(self):
        if self.free_cache_ids:
            return self.free_cache_ids.popleft()
        if not self.cached_idle_blocks:
            raise Exception(
                f"No more free KVCache blocks: KVCache has total {self.num_blocks} blocks, {len(self.active_blocks)} blocks has been used."
            )
        cache_idx, block_evicted = self.cached_idle_blocks.popitem(last=False)
        block_evicted.cache_idx = None
        return cache_idx

    def prepare_metadata_before_prefill(self, task: "Task"):
        """prepare and update metadata before the task begin a prefill step"""
        task.new_cache_ids = []
        task.hit_token_len = 0  # must be reset every step
        consumed_before_prefill = task.consumed_req_tokens

        if self.enable_prefix_caching:
            # 被prefix caching击中block不占chunk prefill size的容量，也不增加额外的kv cache block需求
            for idx in range(
                len(self.task_to_cache_ids[task.task_id]), len(task.token_blocks)
            ):
                block = task.token_blocks[idx]

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
                    self.active_blocks[block.cache_idx] = block
                    self.cached_idle_blocks.pop(block.cache_idx)

                # 更新块的活跃状态、当前step新分配给任务的kvcache块索引，以及任务的所有kvcache块索引
                self.task_to_cache_ids[task.task_id].add(block.cache_idx)
                task.new_cache_ids.append(block.cache_idx)
                block.active_cnt += 1
                task.consumed_req_tokens += self.block_size

            if task.consumed_req_tokens == task.prefix_tokens_len:
                # prompt被完全击中，此时只需对最后一个token做prefill step获取logits, 但无需额外的kv cache block
                assert task.prefill_chunk_size == 1
                task.consumed_req_tokens -= 1

            if task.consumed_req_tokens > consumed_before_prefill:
                # incremental prefix-caching hits in this step.
                task.hit_token_len = task.consumed_req_tokens - consumed_before_prefill

        num_full_blocks = task.consumed_req_tokens // self.block_size
        self.update_prefix_caching_metadata(task, num_full_blocks)

        # [0,num_target_blocks)区间内的block均需要分配cache块索引
        self.tid_to_cached_len[task.task_id] = (
            task.kv_cache_len_used_in_completed_steps_and_next_step
        )
        num_target_blocks = ceil_div(
            self.tid_to_cached_len[task.task_id], self.block_size
        )
        for idx in range(
            len(self.task_to_cache_ids[task.task_id]), num_target_blocks, 1
        ):
            block = task.token_blocks[idx]
            assert (
                block.cache_idx is None
            ), f"[DP {self.dp_rank}] task_id: {task.task_id}, idx:{idx}, tasks.cached_blocks:{[blk.cache_idx for blk in task.token_blocks]}, cache_manager.task_to_cache_ids:{self.task_to_cache_ids[task.task_id]}"
            cache_idx = self.get_free_cache_idx()
            assert (
                cache_idx not in self.task_to_cache_ids[task.task_id]
            ), f"{cache_idx} is already in {self.task_to_cache_ids[task.task_id]}"
            task.new_cache_ids.append(cache_idx)
            block.cache_idx = cache_idx
            self.task_to_cache_ids[task.task_id].add(block.cache_idx)
            block.active_cnt += 1
            self.active_blocks[block.cache_idx] = block

    def prepare_metadata_before_decode(self, task: "Task"):
        """prepare and update metadata before the task begin a decode step"""
        task.new_cache_ids = []

        # DLLM decode: use decoding_start + block_length
        from chitu.models.registry import ModelType

        args = get_global_args()
        if hasattr(args, "models") and args.models.type == ModelType.LLADA2:
            self._prepare_metadata_before_decode_dllm(task)
            return

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

        for idx in range(
            len(self.task_to_cache_ids[task.task_id]), num_target_blocks, 1
        ):
            if idx < len(task.token_blocks):
                # TokenBlock already exists (created during prefill), but may not have cache idx yet.
                block = task.token_blocks[idx]
                assert (
                    block.cache_idx is None
                ), f"idx:{idx}, block.cache_idx:{block.cache_idx}, task.token_blocks:{task.token_blocks}"
            else:
                # New generated TokenBlock in next decode step
                tokens = task.prefix_tokens[
                    idx * self.block_size : (idx + 1) * self.block_size
                ]
                pre_blk_hash = task.token_blocks[-1].blk_hash
                block = self.tokens_to_block(tokens=tokens, pre_blk_hash=pre_blk_hash)
                assert (
                    block.cache_idx is None
                ), f"idx:{idx}, task.token_blocks:{task.token_blocks}"
                task.token_blocks.append(block)

            cache_idx = self.get_free_cache_idx()
            assert (
                cache_idx not in self.task_to_cache_ids[task.task_id]
            ), f"{cache_idx} is already in {self.task_to_cache_ids[task.task_id]}"
            block.cache_idx = cache_idx
            self.task_to_cache_ids[task.task_id].add(block.cache_idx)
            task.new_cache_ids.append(cache_idx)
            block.active_cnt += 1
            self.active_blocks[block.cache_idx] = block

    def _prepare_metadata_before_decode_dllm(self, task: "Task"):
        """Prepare metadata for DLLM decode step.

        DLLM decode uses decoding_start + block_length to determine cache requirements.
        Unlike normal decode, DLLM decode processes a full block of tokens per step.
        """
        task.new_cache_ids = []

        block_length = task.block_length
        decoding_start = task.decoding_start
        target_seq_len = decoding_start + block_length

        self.tid_to_cached_len[task.task_id] = target_seq_len
        num_target_blocks = ceil_div(target_seq_len, self.block_size)

        for idx in range(
            len(self.task_to_cache_ids[task.task_id]), num_target_blocks, 1
        ):
            cache_idx = self.get_free_cache_idx()
            self.task_to_cache_ids[task.task_id].add(cache_idx)
            task.new_cache_ids.append(cache_idx)

            # Create a placeholder TokenBlock if needed
            # DLLM decode doesn't use prefix_tokens for token tracking,
            # so we create placeholder blocks for metadata management only.
            if idx < len(task.token_blocks):
                block = task.token_blocks[idx]
            else:
                # Create a placeholder block for DLLM decode
                block = TokenBlock(
                    tokens=[0]
                    * self.block_size,  # Placeholder tokens (not used for DLLM)
                    blk_hash=None,
                    blk_size=self.block_size,
                    pre_blk_hash=(
                        task.token_blocks[-1].blk_hash
                        if task.token_blocks
                        else NONE_BLK_HASH
                    ),
                )
                task.token_blocks.append(block)

            block.cache_idx = cache_idx
            block.active_cnt += 1
            self.active_blocks[cache_idx] = block

    def update_prefix_caching_metadata(self, task: "Task", num_full_blocks):
        """Update metadata in task_hashed_block_cnt, task.token_blocks and hashed_block_pool"""
        if not self.enable_prefix_caching:
            return
        num_hashed_blocks = self.task_hashed_block_cnt[task.task_id]
        for idx in range(num_hashed_blocks, num_full_blocks, 1):
            block = task.token_blocks[idx]
            if idx == 0:
                # block是task的第一个tokenblock
                pre_blk_hash = NONE_BLK_HASH
            elif block.pre_blk_hash != NONE_BLK_HASH:
                # block被prefix caching击中，原本已生成pre_blk_hash和blk_hash
                pre_blk_hash = block.pre_blk_hash
            else:
                # block未被prefix caching击中
                pre_blk_hash = task.token_blocks[idx - 1].blk_hash
                assert (
                    pre_blk_hash is not None
                ), f"idx:{idx}, pre_block: {task.token_blocks[idx-1]} , task.token_blocks:{task.token_blocks}"
            block.pre_blk_hash = pre_blk_hash
            if len(block.tokens) != self.block_size:
                # decode
                block.tokens = task.prefix_tokens[
                    idx * self.block_size : (idx + 1) * self.block_size
                ]
            block = self.add_to_hashed_block_pool(block)
        self.task_hashed_block_cnt[task.task_id] = num_full_blocks

    def finalize_metadata_all_decode(self, task: "Task"):
        if task.task_id not in self.tid_to_cached_len:
            return
        for block in reversed(task.token_blocks):
            if block.cache_idx is None:
                continue
            block.active_cnt -= 1
            assert (
                block.active_cnt >= 0
            ), f"TokenBlock.activte_cnt ({block.active_cnt}) shouldn't smaller than 0."
            if block.active_cnt == 0:
                # 当block的活跃度为0时，将block从active_blocks中移除，并放入cached_idle_blocks的最右端
                self.active_blocks.pop(block.cache_idx)
                self.cached_idle_blocks[block.cache_idx] = block
        self.tid_to_cached_len.pop(task.task_id)
        self.task_to_cache_ids.pop(task.task_id)
        if task.task_id in self.task_hashed_block_cnt:
            self.task_hashed_block_cnt.pop(task.task_id)
