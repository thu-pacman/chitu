# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
统一的 Metadata 接口
"""

import msgpack
from typing import Optional, List, Dict, Set, Literal, Union, Any
from dataclasses import dataclass
from logging import getLogger

from chitu.global_vars import get_global_args
from chitu.task import (
    Task,
    PackedTasks,
    PackedTasksBase,
    TaskPool,
    TaskType,
    SerializedPackedTasksPayloadType,
    SampleParams,
    is_normal_payload,
)

logger = getLogger(__name__)


@dataclass
class MetadataConfig:
    """配置要传输的字段"""

    # 基础字段（总是传输）
    include_task_ids: bool = True
    include_payload_type: bool = True
    include_task_type: bool = True

    # 固定字段（只用传输一次）
    include_tokens: bool = False  # token 序列
    include_sample_params: bool = (
        False  # 采样相关内容（温度，topk，topp，frequency prnalty）
    )
    include_return_params: bool = False  # 返回相关内容（包含logprobs和test flag）

    # Prefill 相关字段
    include_consumed_tokens: bool = False  # 已消费的输入 token 数
    include_chunk_size: bool = False  # 下一次 prefill 消费的输入 token 数

    # PackedTasks 相关字段
    include_tokens_short: bool = (
        False  # TP 只需要传输 token 长度，不需要完整 token 序列
    )

    # PD
    include_prompt_len: bool = False
    include_next_token: bool = False
    include_pd_prefill_engine_rank: bool = False
    include_boot_ids: bool = False

    # 辅助信息
    include_has_outputs: bool = False  # 状态信息
    include_slot_idx: bool = False  # PP+skew 用来分配任务所属的 kvcache slot

    @property
    def include_task(self):
        """每个 task 单独的信息，会分开存储在各个 task 中，包含固定字段和 prefill 相关字段"""
        return (
            self.include_tokens
            or self.include_sample_params
            or self.include_return_params
            or self.include_consumed_tokens
            or self.include_chunk_size
            or self.include_prompt_len
            or self.include_pd_prefill_engine_rank
        )

    @classmethod
    def for_special(cls) -> "MetadataConfig":
        """特殊 payload 传输配置：仅包含最基础的字段"""
        return cls(
            include_task_ids=True,
            include_payload_type=True,
            include_task_type=True,
        )

    @classmethod
    def for_prefill(cls) -> "MetadataConfig":
        """Prefill 传输配置"""
        return cls(
            include_task_ids=True,
            include_payload_type=True,
            include_task_type=True,
            include_tokens=True,
            include_sample_params=True,
            include_return_params=True,
            include_consumed_tokens=True,
            include_chunk_size=True,
            include_slot_idx=True,
        )

    @classmethod
    def for_decode_minimal(cls) -> "MetadataConfig":
        """Decode 最小传输配置（一般不使用）"""
        return cls(
            include_task_ids=True,
            include_payload_type=True,
            include_task_type=True,
            include_slot_idx=True,
        )

    @classmethod
    def for_decode_with_status(cls) -> "MetadataConfig":
        """Decode 带状态传输配置"""
        return cls(
            include_task_ids=True,
            include_payload_type=True,
            include_task_type=True,
            include_has_outputs=True,
            include_slot_idx=True,
        )

    # ========== 针对各并行方案优化的配置 ==========

    @classmethod
    def for_tp_prefill(cls) -> "MetadataConfig":
        """TP 专用配置（最精简）

        TP 特点：
        - 单机内通信，所有 ranks 共享 TaskPool
        - 只需要同步 task_ids 和 payload_type
        - 不需要传输任何 Task 详细信息
        - 不需要完整的 token 信息，只需要传递每个 task 的 token 长度
        """
        return cls(
            include_task_ids=True,
            include_payload_type=True,
            include_task_type=True,
            include_tokens_short=True,
            include_has_outputs=True,
            include_slot_idx=True,
        )

    @classmethod
    def for_tp_decode(cls) -> "MetadataConfig":
        """TP 专用配置（最精简）

        TP 特点：
        - 单机内通信，所有 ranks 共享 TaskPool
        - 只需要同步 task_ids 和 payload_type
        - 不需要传输任何 Task 详细信息
        """
        return cls(
            include_task_ids=True,
            include_payload_type=True,
            include_task_type=True,
            include_has_outputs=True,
            include_slot_idx=True,
        )

    @classmethod
    def for_pd_decode_rank(cls) -> "MetadataConfig":
        """PD 对 decode rank 通信专用配置

        PD 分离中 decode rank 的特点：
        - 需要创建任务，且任务创建时的状态为 Decode
        - 需要额外传输 prompt_len 与 pd_prefill_engine_rank
        - 其余信息与通常的 decode 相同
        """
        return cls(
            include_task_ids=True,
            include_payload_type=True,
            include_task_type=True,
            include_has_outputs=True,
            include_slot_idx=True,
            include_sample_params=True,
            include_return_params=True,
            include_prompt_len=True,
            include_pd_prefill_engine_rank=True,
            include_next_token=True,
        )

    @classmethod
    def for_pd_dp_decode_rank(cls) -> "MetadataConfig":
        """PD 对 decode rank 通信专用配置（DP 版）

        比常规的 PD 分离配置多传输一个 boot_ids，用于接收 bootstrap 信息
        """
        return cls(
            include_task_ids=True,
            include_payload_type=True,
            include_task_type=True,
            include_has_outputs=True,
            include_slot_idx=True,
            include_sample_params=True,
            include_return_params=True,
            include_prompt_len=True,
            include_pd_prefill_engine_rank=True,
            include_next_token=True,
            include_boot_ids=True,
            include_tokens=True,
        )

    # 目前未使用，保留
    # 目前的通信中 PP 和 DP 需要传输的信息是相同的，故不做区分
    @classmethod
    def for_pp_prefill(cls) -> "MetadataConfig":
        return cls.for_prefill()

    @classmethod
    def for_pp_decode(cls) -> "MetadataConfig":
        return cls.for_decode_with_status()

    @classmethod
    def for_dp_prefill(cls) -> "MetadataConfig":
        return cls.for_prefill()

    @classmethod
    def for_dp_decode(cls) -> "MetadataConfig":
        return cls.for_decode_with_status()


class MetadataSerializer:
    """统一的 Metadata 序列化器

    用于 PP 和 DP 的 ZMQ 通信
    TP 也使用统一的 msgpack 序列化方案

    去重优化：
    - 跟踪已传输的任务，对于已知任务只传增量信息
    - 跟踪已接收的任务，用于判断是否需要创建新 Task
    - TP 场景：跟踪 task_ids，如果相同则只传变化的字段
    """

    def __init__(
        self,
        enable_dedup: bool = True,
        mode: Literal["DP", "PP", "TP", "none"] = "none",
    ):
        """
        Args:
            enable_dedup: 是否启用去重优化（默认启用）
        """
        self.enable_dedup = enable_dedup
        self.mode = mode
        self.transmitted_task_ids: Dict[int, Set] = {}

    def get_recv_type(self, tasks: Union[PackedTasks, PackedTasksBase]):
        if self.mode == "TP":
            return "PackedTasksBase"
        return tasks.__class__.__name__

    def serialize_metadata(
        self,
        tasks: Union[PackedTasks, PackedTasksBase],
        config: Optional[MetadataConfig] = None,
        slot_idx: Optional[int] = None,
        auto_dedup: bool = True,
        target_rank: int = -1,
    ) -> bytes:
        assert isinstance(
            tasks, PackedTasksBase
        ), "Input tasks should be PackedTasksBase or PackedTasks"

        # 去重检测：检查哪些任务是新任务
        new_task_ids = []
        has_dedup = (
            self.enable_dedup
            and auto_dedup
            and isinstance(tasks, PackedTasks)
            and isinstance(tasks.task_ids, list)
            and tasks.task_type != TaskType.Special
        )

        if has_dedup:
            if not self.transmitted_task_ids.get(target_rank):
                self.transmitted_task_ids[target_rank] = set()
            transmitted_task_ids = self.transmitted_task_ids[target_rank]
            new_task_ids = [
                task_id
                for task_id in tasks.task_ids
                if task_id not in transmitted_task_ids
            ]
            transmitted_task_ids.update(new_task_ids)
        elif tasks.payload_type == SerializedPackedTasksPayloadType.EndTask:
            if self.transmitted_task_ids.get(target_rank):
                self.transmitted_task_ids[target_rank] -= set(tasks.task_ids)

        # 选择配置
        if config is None:
            config = self._auto_select_config_with_dedup(tasks, new_task_ids)

        msg_dict: Dict[str, Any] = {}

        # 固定传输字段
        msg_dict["format"] = self.get_recv_type(tasks)
        if config.include_payload_type:
            msg_dict["payload_type"] = tasks.payload_type.name
        if config.include_task_type:
            msg_dict["task_type"] = tasks.task_type.name
        if config.include_task_ids:
            msg_dict["task_ids"] = tasks.task_ids
            msg_dict["num_tasks"] = tasks.num_tasks

        # 去重标记：记录哪些任务是新任务
        if self.enable_dedup and auto_dedup:
            msg_dict["new_task_ids"] = new_task_ids

        # TP 采用 short 版 token 信息：仅包含 token 长度
        if config.include_tokens_short:
            msg_dict["token_lengths"] = [len(tokens) for tokens in tasks.tokens]
            msg_dict["num_tokens"] = tasks.num_tokens

        # Decode 状态信息
        if config.include_has_outputs:
            msg_dict["has_outputs"] = tasks.has_outputs

        # Task 信息
        new_task_ids_set = set(new_task_ids)
        if config.include_task and isinstance(tasks, PackedTasks):
            tasks_data = []
            for task in tasks.tasks:
                task_data: Dict[str, Any] = {"task_id": task.task_id}

                # 判断是否是新任务
                is_new_task = task.task_id in new_task_ids_set

                # 新任务（首次传输）相关字段
                if is_new_task:
                    # 只在新任务时传输 tokens
                    if config.include_tokens:
                        task_data["tokens"] = list(task.prefix_tokens)
                        task_data["standard_tokens"] = task._test_standard_tokens
                        task_data["grammar_str"] = task.grammar_str
                    # 只对新任务传输 sample_params
                    if config.include_sample_params:
                        task_data["sample_params"] = {
                            "temperature": task.sample_params.temperature,
                            "top_p": task.sample_params.top_p,
                            "top_k": task.sample_params.top_k,
                            "frequency_penalty": task.sample_params.frequency_penalty,
                        }
                    # 输出相关字段（只对新任务）
                    if config.include_return_params:
                        task_data["return_logprobs"] = task.return_logprobs
                        task_data["_test_flag"] = task._test_flag
                    # prompt_len
                    if config.include_prompt_len:
                        task_data["prompt_len"] = task.prompt_len
                    # pd prefill rank
                    if config.include_pd_prefill_engine_rank:
                        task_data["pd_prefill_engine_rank"] = (
                            task.pd_prefill_engine_rank
                        )

                # Prefill 相关字段
                if config.include_consumed_tokens:
                    task_data["consumed_req_tokens"] = task.consumed_req_tokens

                if config.include_chunk_size and task.prefill_chunk_size is not None:
                    task_data["prefill_chunk_size"] = task.prefill_chunk_size

                # Decode 相关字段
                if config.include_next_token:
                    task_data["next_token"] = task.next_token

                tasks_data.append(task_data)

            msg_dict["tasks"] = tasks_data

        # 辅助信息
        if config.include_slot_idx and slot_idx is not None:
            msg_dict["slot_idx"] = slot_idx
        if config.include_boot_ids and len(new_task_ids) > 0:
            # Decode task meta：首次下发时发送 MsgPackableTask
            # 让 worker rank 本地 TaskPool 可以构造 PackedTasks，并在收到 bootstrap 时发送 TransferInfo
            msg_dict["boot_ids"] = new_task_ids
            logger.debug(
                f"[PD_TRACE][dp.send_decode_bootstrap] to_rank={int(target_rank)} "
                f"scheduled_task_ids_len={len(tasks.task_ids)} bootstrap_tasks={new_task_ids}"
            )

        if tasks.new_cache_ids_list:
            msg_dict["new_cache_ids_list"] = tasks.new_cache_ids_list
        if tasks.inc_hit_tokens_list:
            msg_dict["inc_hit_tokens_list"] = tasks.inc_hit_tokens_list
        if tasks.prefix_lens:
            msg_dict["prefix_lens"] = tasks.prefix_lens

        # 序列化
        return msgpack.packb(msg_dict, use_bin_type=True)

    def deserialize_metadata(
        self,
        data: bytes,
    ) -> tuple[
        SerializedPackedTasksPayloadType,
        Union[PackedTasks, PackedTasksBase],
        Optional[int],
        Dict[str, Any],
    ]:
        extra_info = dict()
        msg_dict = msgpack.unpackb(data, raw=False)
        msg_format = msg_dict.get("format", "error no format")
        assert msg_format in (
            "PackedTasks",
            "PackedTasksBase",
        ), f"Unsupport message format: {msg_format}"

        payload_type = SerializedPackedTasksPayloadType[
            msg_dict.get("payload_type", "NoneType")
        ]
        if msg_format == "PackedTasksBase" or not is_normal_payload(payload_type):
            task_type = TaskType[msg_dict.get("task_type", "Special")]
            slot_idx = msg_dict.get("slot_idx")
            num_tasks = msg_dict.get("num_tasks", 0)
            # token_lens 和 num_tokens 默认采用 decode 配置（所有 task 的 token 长度为1）
            num_tokens = msg_dict.get("num_tokens", num_tasks)
            token_lens = msg_dict.get("token_lengths", [1 for _ in range(num_tasks)])
            tokens = [[0] * length for length in token_lens]

            packed_tasks_base = PackedTasksBase(
                num_tasks=num_tasks,
                task_ids=msg_dict.get("task_ids", []),
                task_type=task_type,
                tokens=tokens,
                payload_type=payload_type,
                new_cache_ids_list=msg_dict.get("new_cache_ids_list", []),
                inc_hit_tokens_list=msg_dict.get("inc_hit_tokens_list", []),
                prefix_lens=msg_dict.get("prefix_lens", []),
                num_tokens=num_tokens,
                has_outputs=msg_dict.get("has_outputs", []),
            )

            return payload_type, packed_tasks_base, slot_idx, extra_info

        # 解析基础信息
        task_type = TaskType[msg_dict.get("task_type", "Special")]
        tasks_data = msg_dict.get("tasks", [])
        task_ids = msg_dict.get("task_ids", [])
        slot_idx = msg_dict.get("slot_idx")
        has_outputs = msg_dict.get("has_outputs")

        # 处理空任务
        if len(task_ids) == 0:
            return (
                payload_type,
                PackedTasks([], task_type=task_type),
                slot_idx,
                extra_info,
            )

        # 重建或更新 Task 对象
        task_list = []
        for task_data in tasks_data:
            task_id = task_data["task_id"]

            if task_id in TaskPool.pool:
                task = TaskPool.pool[task_id]
            else:
                task = self._create_task_from_data(
                    task_id, task_data, task_type=task_type
                )
                TaskPool.add(task)

            # Prefill
            if "consumed_req_tokens" in task_data:
                task.consumed_req_tokens = task_data["consumed_req_tokens"]
            if "prefill_chunk_size" in task_data:
                task.set_prefill_chunk_size_for_one_step(
                    int(task_data["prefill_chunk_size"])
                )
            # Decode
            if "next_token" in task_data:
                task.next_token = task_data["next_token"]

            task_list.append(task)

        # Prefill：含 task data，采用更快的 PackedTasks 构建方法，避免重复索引
        # Decode：不含 task data，直接从 task_ids 构建
        if len(task_list) > 0:
            packed_tasks = PackedTasks([], tasks=task_list)
        else:
            packed_tasks = PackedTasks(task_ids)

        # 使用接收的状态信息，覆盖从本地读取的状态信息
        if has_outputs:
            packed_tasks.has_outputs = has_outputs

        packed_tasks.new_cache_ids_list = msg_dict.get("new_cache_ids_list", [])
        packed_tasks.inc_hit_tokens_list = msg_dict.get("inc_hit_tokens_list", [])
        packed_tasks.prefix_lens = msg_dict.get("prefix_lens", [])

        boot_ids = msg_dict.get("boot_ids", None)
        if boot_ids:
            extra_info["boot_ids"] = boot_ids

        return payload_type, packed_tasks, slot_idx, extra_info

    def _create_task_from_data(
        self, task_id: str, task_data: Dict, task_type: TaskType = TaskType.Prefill
    ) -> Task:
        """从序列化数据创建 Task 对象"""

        sample_params_dict = task_data.get("sample_params", {})
        sample_params = SampleParams(
            temperature=sample_params_dict.get("temperature"),
            top_p=sample_params_dict.get("top_p"),
            top_k=sample_params_dict.get("top_k"),
            frequency_penalty=sample_params_dict.get("frequency_penalty"),
        )
        tokens = task_data.get("tokens")
        grammar_str = task_data.get("grammar_str", "")
        prompt_len = task_data.get("prompt_len")
        task = Task(
            task_id=task_id,
            req=None,
            sample_params=sample_params,
            prefix_tokens=tokens,
            grammar_str=grammar_str,
            prompt_len=prompt_len,
        )
        task.return_logprobs = task_data.get("return_logprobs")
        task._test_flag = task_data.get("_test_flag")
        task._test_standard_tokens = task_data.get("standard_tokens")
        task.pd_prefill_engine_rank = task_data.get("pd_prefill_engine_rank")
        if task_type in (TaskType.Prefill, TaskType.Decode):
            task.task_type = task_type
        return task

    def _auto_select_config_with_dedup(
        self,
        tasks: Union[PackedTasks, PackedTasksBase],
        new_task_ids: List[str],
    ) -> MetadataConfig:
        """根据 tasks 类型和去重状态自动选择配置

        Args:
            tasks: PackedTasks 对象
            new_task_ids: 新任务的 ID 列表（在 _transmitted_tasks 中不存在的任务）
            force_include_last_tokens: 强制包含 last_tokens（用于 DP+PP 场景）

        配置选择逻辑：
        - Prefill:
          - all_known=True: 所有任务都已传输过完整信息，只需传进度
          - has_first_prefill=True: 有任务是首包（consumed==0），需要完整信息
          - 其他: 混合 batch 或后续 chunk，启用 tokens/params 选项让序列化器按需传输
        - Decode:
          - 不需要传输 task 相关信息
          - 只需要传输 has_outputs，和 TP 统一
        """
        if self.mode == "TP" or type(tasks) == PackedTasksBase:
            # PackedTasksBase 特判：只能使用 special 和 TP 配置，和 TP 传输放一起
            if tasks.task_type == TaskType.Prefill:
                return MetadataConfig.for_tp_prefill()
            elif tasks.task_type == TaskType.Decode:
                return MetadataConfig.for_tp_decode()
            else:
                return MetadataConfig.for_special()

        pd_enabled = (
            get_global_args().dp_config.router.pd_disaggregation.enabled
            if getattr(get_global_args(), "dp_config", None) is not None
            else False
        )
        if tasks.task_type == TaskType.Prefill:
            return MetadataConfig.for_prefill()
        elif tasks.task_type == TaskType.Decode:
            if pd_enabled:
                if self.mode == "DP":
                    return MetadataConfig.for_pd_dp_decode_rank()
                return MetadataConfig.for_pd_decode_rank()
            else:
                return MetadataConfig.for_decode_with_status()
        else:
            return MetadataConfig.for_special()
