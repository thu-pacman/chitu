# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
统一的 Metadata 接口
"""

import msgpack
from typing import Optional, List, Dict, Set, Literal, Union
from dataclasses import dataclass

from chitu.task import (
    Task,
    PackedTasks,
    PackedTasksBase,
    TaskPool,
    TaskType,
    TaskDecodeType,
    SerializedPackedTasksPayloadType,
)


@dataclass
class MetadataConfig:
    """配置要传输的字段"""

    # 基础字段（总是传输）
    include_task_ids: bool = True
    include_payload_type: bool = True

    # Prefill 相关字段
    include_tokens: bool = False  # token 序列
    include_params: bool = False  # SampleParams
    include_consumed_tokens: bool = False  # 已消费的 token 数
    include_chunk_size: bool = False  # prefill chunk size

    # Decode 相关字段
    include_decode_status: bool = False  # decode 状态
    include_last_tokens: bool = False  # 上一个生成的 token (for DP+PP)

    # 输出相关字段
    include_return_logprobs: bool = False

    # 调度相关字段
    include_sched_group_id: bool = False

    # 测试相关字段
    include_test_flag: bool = False

    # 辅助信息
    include_slot_idx: bool = False

    @classmethod
    def for_prefill_full(cls) -> "MetadataConfig":
        """Prefill 首次传输配置"""
        return cls(
            include_task_ids=True,
            include_payload_type=True,
            include_tokens=True,
            include_params=True,
            include_consumed_tokens=True,
            include_chunk_size=True,
            include_return_logprobs=True,
            include_sched_group_id=True,
            include_test_flag=True,
            include_slot_idx=True,
        )

    @classmethod
    def for_prefill_incremental(cls) -> "MetadataConfig":
        """Prefill 增量传输配置（chunk prefill）"""
        return cls(
            include_task_ids=True,
            include_payload_type=True,
            # 说明：
            # chunk prefill 期间可能与“新任务首包”混在同一个 step 中（混合 batch）。
            # 这里保持 include_tokens/include_params=True，实际是否发送由 serializer
            # 对每个 task 的 is_new_task + consumed_req_tokens==0 决定，既正确也不冗余。
            include_tokens=True,
            include_params=True,
            include_consumed_tokens=True,
            include_chunk_size=True,
            include_slot_idx=True,
        )

    @classmethod
    def for_decode_minimal(cls) -> "MetadataConfig":
        """Decode 最小传输配置"""
        return cls(
            include_task_ids=True,
            include_payload_type=True,
            include_slot_idx=True,
        )

    @classmethod
    def for_decode_with_status(cls) -> "MetadataConfig":
        """Decode 带状态传输配置"""
        return cls(
            include_task_ids=True,
            include_payload_type=True,
            include_decode_status=True,
            include_slot_idx=True,
        )

    @classmethod
    def for_decode_with_tokens(cls, include_status: bool = True) -> "MetadataConfig":
        """Decode 带 tokens 传输配置（DP+PP）"""
        return cls(
            include_task_ids=True,
            include_payload_type=True,
            include_decode_status=include_status,
            include_last_tokens=True,
            include_slot_idx=True,
        )

    # ========== 针对各并行方案优化的配置 ==========

    @classmethod
    def for_tp_dispatch(cls) -> "MetadataConfig":
        """TP 专用配置（最精简）

        TP 特点：
        - 单机内通信，所有 ranks 共享 TaskPool
        - 只需要同步 task_ids 和 payload_type
        - 不需要传输任何 Task 详细信息
        """
        return cls(
            include_task_ids=True,
            include_payload_type=True,
            include_slot_idx=True,
            # 其他所有字段都不需要
            include_tokens=False,
            include_params=False,
            include_consumed_tokens=False,
            include_chunk_size=False,
            include_decode_status=False,
            include_last_tokens=False,
            include_return_logprobs=False,
            include_sched_group_id=False,
            include_test_flag=False,
        )

    @classmethod
    def for_pp_prefill_first(cls) -> "MetadataConfig":
        """PP Prefill 首次传输配置

        PP 特点：
        - 跨节点通信，不共享 TaskPool
        - 首次需要传输完整的 Task 信息
        """
        return cls.for_prefill_full()  # 与原有的 full 配置相同

    @classmethod
    def for_pp_prefill_chunk(cls) -> "MetadataConfig":
        """PP Prefill chunk 传输配置（已消费部分 tokens）

        只需要传输进度信息，不需要重复传输 tokens 和 params
        """
        return cls.for_prefill_incremental()  # 与原有的 incremental 配置相同

    @classmethod
    def for_pp_decode(cls) -> "MetadataConfig":
        """PP Decode 传输配置

        PP Decode 需要 decode_status，但不需要 last_token
        （last_token 通过 hidden states 传递）
        """
        return cls.for_decode_with_status()  # 与原有的配置相同

    @classmethod
    def for_dp_prefill_first(cls) -> "MetadataConfig":
        """DP Prefill 首次传输配置

        与 PP 相同，需要完整信息
        """
        return cls.for_prefill_full()

    @classmethod
    def for_dp_prefill_chunk(cls) -> "MetadataConfig":
        """DP Prefill chunk 传输配置

        与 PP 相同，只需进度信息
        """
        return cls.for_prefill_incremental()

    @classmethod
    def for_dp_decode(cls) -> "MetadataConfig":
        """DP Decode 传输配置（单独 DP，无 PP）

        需要 decode_status，不需要 last_token
        """
        return cls(
            include_task_ids=True,
            include_payload_type=True,
            include_decode_status=True,
            # DP 不使用 slot_idx（没有 PP stages）
            include_slot_idx=False,
        )

    @classmethod
    def for_dp_pp_decode(cls) -> "MetadataConfig":
        """DP+PP Decode 传输配置

        需要 decode_status 和 last_token（PP 需要）
        """
        return cls.for_decode_with_tokens(include_status=True)

    @classmethod
    def for_known_task_prefill(cls) -> "MetadataConfig":
        """已知任务的 Prefill 增量传输配置（去重优化）

        当接收方已经有该任务的完整信息时，只需要传：
        - task_id（用于识别任务）
        - consumed_tokens 和 chunk_size（进度信息）
        """
        return cls(
            include_task_ids=True,
            include_payload_type=True,
            include_consumed_tokens=True,
            include_chunk_size=True,
            include_slot_idx=True,
            # 不需要传输（接收方已有）
            include_tokens=False,
            include_params=False,
            include_return_logprobs=False,
            include_sched_group_id=False,
            include_test_flag=False,
        )

    @classmethod
    def for_known_task_decode(cls) -> "MetadataConfig":
        """已知任务的 Decode 最小传输配置（去重优化）

        当接收方已经有该任务的完整信息时，只需要传：
        - task_id（用于识别任务）
        - decode_status（可能变化的状态）
        """
        return cls(
            include_task_ids=True,
            include_payload_type=True,
            include_decode_status=True,
            include_slot_idx=True,
            # 不需要传输（接收方已有）
            include_tokens=False,
            include_params=False,
            include_consumed_tokens=False,
            include_chunk_size=False,
            include_return_logprobs=False,
            include_last_tokens=False,
        )


class MetadataSerializer:
    """统一的 Metadata 序列化器

    用于 PP 和 DP 的 ZMQ 通信
    TP 也使用统一的 msgpack 序列化方案

    去重优化：
    - 跟踪已传输的任务，对于已知任务只传增量信息
    - 跟踪已接收的任务，用于判断是否需要创建新 Task
    - TP 场景：跟踪 task_ids，如果相同则只传变化的字段
    """

    def __init__(self, enable_dedup: bool = True):
        """
        Args:
            enable_dedup: 是否启用去重优化（默认启用）
        """
        self.enable_dedup = enable_dedup

        # 已传输完整信息的任务集合（发送端使用）
        self._transmitted_tasks: Set[str] = set()

        # 已接收完整信息的任务集合（接收端使用）
        self._received_tasks: Set[str] = set()

        # TP 场景去重：上次传输的 task_ids（发送端使用）
        self._last_tp_task_ids: Optional[tuple] = None

        # TP 场景去重：上次接收的完整信息（接收端使用）
        self._last_tp_received: Optional[Dict] = None

    def serialize_metadata(
        self,
        tasks: Union[PackedTasks, PackedTasksBase],
        config: Optional[MetadataConfig] = None,
        slot_idx: Optional[int] = None,
        auto_dedup: bool = True,
        output_format: Literal["packed_tasks", "packed_tasks_base"] = "packed_tasks",
        force_include_last_tokens: bool = False,
    ) -> bytes:
        """序列化 metadata

        Args:
            tasks: PackedTasks 或 PackedTasksBase 对象
            config: 配置要传输的字段，如果为 None 则自动选择
            slot_idx: slot 索引（如果需要）
            auto_dedup: 是否自动去重（默认 True）
            output_format:
                - packed_tasks: msgpack 字典（PP/DP 用，包含完整 Task 字段）
                - packed_tasks_base: msgpack 字典（TP 用，仅包含 PackedTasksBase 字段）
            force_include_last_tokens: 强制包含 last_tokens（用于 DP+PP 场景）

        Returns:
            序列化后的 bytes
        """
        # TP 格式：msgpack 序列化 PackedTasksBase 基础字段
        if output_format == "packed_tasks_base":
            if not isinstance(tasks, PackedTasksBase):
                raise TypeError(
                    f"output_format='packed_tasks_base' requires PackedTasksBase, got {type(tasks)}"
                )

            # TP 去重优化：检查 task_ids 是否与上次相同
            current_task_ids = tuple(tasks.task_ids) if tasks.task_ids else ()
            is_same_tasks = (
                self.enable_dedup
                and auto_dedup
                and current_task_ids == self._last_tp_task_ids
                and len(current_task_ids) > 0
            )

            if is_same_tasks:
                # task_ids 相同，使用精简格式（只传变化的字段）
                # TP 场景：worker ranks 共享 TaskPool，不需要传完整 tokens
                # 但需要 token_lens 用于 prepare_cache_prefill 计算 delta_seq_len
                task_type_name = tasks.task_type.name if tasks.task_type else ""
                msg_dict: Dict[str, any] = {  # type: ignore[misc]
                    "format": "tp_minimal",
                    "payload_type": tasks.payload_type.name,
                    "task_type": task_type_name,
                    "num_tasks": tasks.num_tasks,
                    "num_tokens": tasks.num_tokens,
                    "task_ids": tasks.task_ids,
                    # 以下字段每个 step 可能变化，必须传输
                    "has_outputs": tasks.has_outputs,
                    "has_model_run": tasks.has_model_run,
                    # token_lens 用于 prepare_cache_prefill（不用传完整 tokens）
                    "token_lens": (
                        [len(t) for t in tasks.tokens] if tasks.tokens else []
                    ),
                }
            else:
                # 首次传输或 task_ids 变化，使用完整格式
                task_type_name = tasks.task_type.name if tasks.task_type else ""
                msg_dict = {
                    "format": "tp_full",
                    "payload_type": tasks.payload_type.name,
                    "task_type": task_type_name,
                    "num_tasks": tasks.num_tasks,
                    "num_tokens": tasks.num_tokens,
                    "task_ids": tasks.task_ids,
                    "req_ids": tasks.req_ids,
                    "tokens": tasks.tokens,
                    "has_outputs": tasks.has_outputs,
                    "has_model_run": tasks.has_model_run,
                }
                # 更新去重状态
                self._last_tp_task_ids = current_task_ids

            if slot_idx is not None:
                msg_dict["slot_idx"] = slot_idx
            return msgpack.packb(msg_dict, use_bin_type=True)  # type: ignore[return-value]

        if not isinstance(tasks, PackedTasks):
            raise TypeError(
                f"output_format='packed_tasks' requires PackedTasks, got {type(tasks)}"
            )

        # 去重检测：检查哪些任务是新任务，哪些是已知任务
        new_task_ids = []
        known_task_ids = []

        if self.enable_dedup and auto_dedup and tasks.tasks:
            for task in tasks.tasks:
                if task.task_id in self._transmitted_tasks:
                    known_task_ids.append(task.task_id)
                else:
                    new_task_ids.append(task.task_id)

        # 选择配置
        if config is None:
            config = self._auto_select_config_with_dedup(
                tasks, new_task_ids, known_task_ids, force_include_last_tokens
            )

        # 构建消息字典
        msg_dict: Dict[str, any] = {}  # type: ignore[misc]

        # 基础信息
        if config.include_payload_type:
            msg_dict["payload_type"] = tasks.payload_type.name

        if config.include_task_ids:
            msg_dict["task_ids"] = tasks.task_ids

        msg_dict["task_type"] = tasks.task_type.name if tasks.task_type else ""

        # 去重标记：记录哪些任务是新任务
        if self.enable_dedup and auto_dedup:
            msg_dict["new_task_ids"] = new_task_ids

        # 任务列表
        tasks_data = []
        for task in tasks.tasks:
            task_data: Dict[str, any] = {"task_id": task.task_id}  # type: ignore[misc]

            # 判断是否是新任务
            is_new_task = task.task_id in new_task_ids

            # Prefill 相关字段
            if config.include_tokens:
                # 只在首次 prefill 且是新任务时传输 tokens
                if (
                    is_new_task
                    and task.task_type == TaskType.Prefill
                    and task.consumed_req_tokens == 0
                ):
                    task_data["tokens"] = list(task.prefix_tokens)
                # 已知任务不传 tokens

            if config.include_params:
                # 只对新任务传输 params
                if is_new_task:
                    task_data["params"] = {
                        "temperature": task.params.temperature,
                        "top_p": task.params.top_p,
                        "top_k": task.params.top_k,
                        "frequency_penalty": task.params.frequency_penalty,
                    }
                # 已知任务不传 params

            if config.include_consumed_tokens:
                task_data["consumed_req_tokens"] = task.consumed_req_tokens

            if config.include_chunk_size and task.prefill_chunk_size is not None:
                task_data["prefill_chunk_size"] = task.prefill_chunk_size

            # Decode 相关字段
            if config.include_decode_status:
                task_data["decode_status"] = task._decode_status.value

            if config.include_last_tokens:
                task_data["last_token"] = task.next_token

            # 输出相关字段（只对新任务）
            if config.include_return_logprobs and is_new_task:
                task_data["return_logprobs"] = task.return_logprobs

            # 调度相关字段（只对新任务）
            if (
                config.include_sched_group_id
                and is_new_task
                and task.sched_group_id is not None
            ):
                task_data["sched_group_id"] = task.sched_group_id

            # 测试相关字段（只对新任务）
            if config.include_test_flag and is_new_task:
                task_data["_test_flag"] = task._test_flag

            tasks_data.append(task_data)

        msg_dict["tasks"] = tasks_data

        # 辅助信息
        if config.include_slot_idx and slot_idx is not None:
            msg_dict["slot_idx"] = slot_idx

        # 更新已传输任务集合
        if self.enable_dedup and auto_dedup:
            for tid in new_task_ids:
                self._transmitted_tasks.add(tid)

        # 序列化
        return msgpack.packb(msg_dict, use_bin_type=True)  # type: ignore[return-value]

    def deserialize_metadata(
        self,
        data: bytes,
        require_task_creation: bool = True,
        output_format: Literal["packed_tasks", "packed_tasks_base"] = "packed_tasks",
    ) -> tuple[
        SerializedPackedTasksPayloadType,
        Union[PackedTasks, PackedTasksBase],
        Optional[int],
    ]:
        """反序列化 metadata

        Args:
            data: 序列化的 bytes
            require_task_creation: 是否需要创建新的 Task 对象
                                   (Prefill 时为 True, Decode 时为 False)

        Returns:
            (payload_type, PackedTasks, slot_idx)
        """
        msg_dict = msgpack.unpackb(data, raw=False)

        # TP 格式：还原 PackedTasksBase
        if output_format == "packed_tasks_base":
            fmt = msg_dict.get("format")
            if fmt not in ("tp_full", "tp_minimal"):
                raise ValueError(f"Unexpected packed_tasks_base format: {fmt}")

            payload_type = SerializedPackedTasksPayloadType[msg_dict["payload_type"]]
            task_type = TaskType[msg_dict["task_type"]]
            slot_idx = msg_dict.get("slot_idx")

            if fmt == "tp_minimal":
                # 精简格式：从 token_lens 重建 tokens（用于 prepare_cache_prefill）
                # 实际 token 内容不重要，只需要长度正确
                token_lens = msg_dict.get("token_lens", [])
                tokens = [[0] * length for length in token_lens]
                req_ids = msg_dict["task_ids"]
            else:
                # 完整格式：直接从消息中获取
                tokens = msg_dict["tokens"]
                req_ids = msg_dict["req_ids"]
                # 保存完整信息用于后续精简格式
                self._last_tp_received = {
                    "tokens": tokens,
                    "req_ids": req_ids,
                }

            packed_tasks_base = PackedTasksBase(
                num_tasks=msg_dict["num_tasks"],
                task_ids=msg_dict["task_ids"],
                req_ids=req_ids,
                task_type=task_type,
                tokens=tokens,
                payload_type=payload_type,
                num_tokens=msg_dict["num_tokens"],
                has_outputs=msg_dict["has_outputs"],
                has_model_run=msg_dict["has_model_run"],
            )
            return payload_type, packed_tasks_base, slot_idx

        # 解析基础信息
        payload_type = SerializedPackedTasksPayloadType[msg_dict["payload_type"]]
        task_type = TaskType[msg_dict["task_type"]]
        tasks_data = msg_dict.get("tasks", [])
        slot_idx = msg_dict.get("slot_idx")

        # 去重信息：哪些是新任务
        new_task_ids = set(msg_dict.get("new_task_ids", []))

        # 处理空任务
        if not tasks_data:
            if task_type in [TaskType.Prefill, TaskType.EmptyPrefill]:
                empty_type = TaskType.EmptyPrefill
            elif task_type in [TaskType.Decode, TaskType.EmptyDecode]:
                empty_type = TaskType.EmptyDecode
            else:
                empty_type = None
            return payload_type, PackedTasks([], empty_task_type=empty_type), slot_idx

        # 重建或更新 Task 对象
        task_list = []
        for task_data in tasks_data:
            task_id = task_data["task_id"]

            # 判断是否是新任务（根据去重标记或本地状态）
            is_new_task = task_id in new_task_ids or (
                self.enable_dedup and task_id not in self._received_tasks
            )

            if require_task_creation:
                # Prefill: 创建或获取 Task
                if task_id in TaskPool.pool:
                    # 已存在的任务
                    task = TaskPool.pool[task_id]
                    # 同步去重状态（防止状态不一致）
                    if self.enable_dedup and task_id not in self._received_tasks:
                        self._received_tasks.add(task_id)
                elif "params" in task_data:
                    # 新任务或去重状态不一致但消息包含完整信息：创建 Task
                    task = self._create_task_from_data(task_id, task_data)
                    TaskPool.add(task)
                    if self.enable_dedup:
                        self._received_tasks.add(task_id)
                else:
                    # 不在 TaskPool 且消息不包含完整信息，无法恢复
                    raise ValueError(
                        f"Task {task_id} not in TaskPool and message lacks full info. "
                        "Dedup state mismatch cannot be recovered."
                    )

                # 更新进度信息
                if "consumed_req_tokens" in task_data:
                    task.consumed_req_tokens = task_data["consumed_req_tokens"]
                if "prefill_chunk_size" in task_data:
                    task.set_prefill_chunk_size_for_one_step(
                        int(task_data["prefill_chunk_size"])
                    )
            else:
                # Decode: 从 TaskPool 获取已有 Task
                if task_id not in TaskPool.pool:
                    raise ValueError(f"Task {task_id} not found in TaskPool")
                task = TaskPool.pool[task_id]

                # 更新状态
                if "decode_status" in task_data:
                    task._decode_status = TaskDecodeType(
                        value=task_data["decode_status"]
                    )

                if "last_token" in task_data:
                    task.update_response_no_sync(task_data["last_token"])

            task_list.append(task)

        # 构建 PackedTasks
        packed_tasks = PackedTasks([], tasks=task_list)

        return payload_type, packed_tasks, slot_idx

    def _create_task_from_data(self, task_id: str, task_data: Dict) -> Task:
        """从序列化数据创建 Task 对象"""
        from chitu.task import SampleParams

        params_dict = task_data.get("params", {})
        params = SampleParams(
            temperature=params_dict.get("temperature", 1.0),
            top_p=params_dict.get("top_p", 0.9),
            top_k=params_dict.get("top_k", 50),
            frequency_penalty=params_dict.get("frequency_penalty", 0.0),
        )
        tokens = task_data.get("tokens", [])
        task = Task(task_id=task_id, req=None, params=params, prefix_tokens=tokens)  # type: ignore[arg-type]
        task.return_logprobs = task_data.get("return_logprobs", False)
        task._test_flag = task_data.get("_test_flag", False)
        task.sched_group_id = task_data.get("sched_group_id")
        return task

    def _auto_select_config_with_dedup(
        self,
        tasks: PackedTasks,
        new_task_ids: List[str],
        known_task_ids: List[str],
        force_include_last_tokens: bool = False,
    ) -> MetadataConfig:
        """根据 tasks 类型和去重状态自动选择配置

        Args:
            tasks: PackedTasks 对象
            new_task_ids: 新任务的 ID 列表（在 _transmitted_tasks 中不存在的任务）
            known_task_ids: 已知任务的 ID 列表（在 _transmitted_tasks 中存在的任务）
            force_include_last_tokens: 强制包含 last_tokens（用于 DP+PP 场景）

        配置选择逻辑：
        - Prefill:
          - all_known=True: 所有任务都已传输过完整信息，只需传进度
          - has_first_prefill=True: 有任务是首包（consumed==0），需要完整信息
          - 其他: 混合 batch 或后续 chunk，启用 tokens/params 选项让序列化器按需传输
        - Decode:
          - all_known=True: 只需传 task_ids 和 decode_status
          - force_include_last_tokens=True: 额外传 last_tokens（DP+PP 场景）
        """
        # 如果所有任务都是已知任务（已传输过完整信息），使用精简配置
        all_known = len(new_task_ids) == 0 and len(known_task_ids) > 0

        if tasks.task_type == TaskType.Prefill:
            if all_known:
                # 所有任务都已传输过完整信息，只需传进度信息
                return MetadataConfig.for_known_task_prefill()

            has_any_first_prefill = any(
                (t.consumed_req_tokens == 0) for t in (tasks.tasks or [])
            )
            if has_any_first_prefill:
                # 有任务是首包 prefill（consumed==0），需要携带完整的新任务信息
                return MetadataConfig.for_prefill_full()
            else:
                # 混合 batch 或后续 chunk：启用 tokens/params 选项
                # 实际是否传输由序列化器根据 is_new_task 和 consumed_req_tokens 决定
                return MetadataConfig.for_prefill_incremental()

        elif tasks.task_type == TaskType.Decode:
            # Decode 阶段所有任务应该都是已知的（在 Prefill 阶段已创建）
            # 如果有"新任务"，说明是首次传输给该接收方
            if force_include_last_tokens:
                # DP+PP 场景：需要传 last_tokens（PP 后续 stage 需要）
                return MetadataConfig.for_decode_with_tokens(include_status=True)
            else:
                # 只传 task_ids 和 decode_status（精简配置）
                return MetadataConfig.for_decode_with_status()
        else:
            return MetadataConfig()

    def clear_task(self, task_id: str) -> None:
        """清理单个任务的去重状态

        当任务结束（EndTask）时调用此方法
        """
        self._transmitted_tasks.discard(task_id)
        self._received_tasks.discard(task_id)

        # 清理 TP 去重状态（如果该任务在缓存中）
        if self._last_tp_task_ids is not None and task_id in self._last_tp_task_ids:
            self._last_tp_task_ids = None
            self._last_tp_received = None

    def clear_tasks(self, task_ids: List[str]) -> None:
        """清理多个任务的去重状态

        Args:
            task_ids: 要清理的任务 ID 列表
        """
        for tid in task_ids:
            self._transmitted_tasks.discard(tid)
            self._received_tasks.discard(tid)

        # 清理 TP 去重状态（如果有任务在缓存中）
        if self._last_tp_task_ids is not None:
            task_ids_set = set(task_ids)
            if any(tid in task_ids_set for tid in self._last_tp_task_ids):
                self._last_tp_task_ids = None
                self._last_tp_received = None

    def clear_tp_dedup_state(self) -> None:
        """清理 TP 去重状态

        当 TP 的 task_ids 发生变化时调用此方法
        """
        self._last_tp_task_ids = None
        self._last_tp_received = None

    def sync_with_task_pool(self) -> int:
        """同步去重状态与 TaskPool，清理已不存在的任务

        用于定期清理，防止去重状态泄漏。
        建议在每次调度循环结束时调用。

        Returns:
            清理的任务数量
        """
        cleaned_count = 0

        # 清理 _transmitted_tasks 中不在 TaskPool 的任务
        stale_transmitted = [
            tid for tid in self._transmitted_tasks if tid not in TaskPool.pool
        ]
        for tid in stale_transmitted:
            self._transmitted_tasks.discard(tid)
            cleaned_count += 1

        # 清理 _received_tasks 中不在 TaskPool 的任务
        stale_received = [
            tid for tid in self._received_tasks if tid not in TaskPool.pool
        ]
        for tid in stale_received:
            self._received_tasks.discard(tid)
            cleaned_count += 1

        # 清理 TP 去重状态（如果缓存的任务已不存在）
        if self._last_tp_task_ids is not None:
            if any(tid not in TaskPool.pool for tid in self._last_tp_task_ids):
                self._last_tp_task_ids = None
                self._last_tp_received = None

        return cleaned_count

    def get_dedup_stats(self) -> Dict[str, any]:  # type: ignore[misc]
        """获取去重状态统计信息（用于调试）

        Returns:
            包含已传输和已接收任务数量的字典
        """
        return {
            "transmitted_tasks": len(self._transmitted_tasks),
            "received_tasks": len(self._received_tasks),
            "tp_dedup_active": self._last_tp_task_ids is not None,
            "tp_cached_task_count": (
                len(self._last_tp_task_ids) if self._last_tp_task_ids else 0
            ),
        }
