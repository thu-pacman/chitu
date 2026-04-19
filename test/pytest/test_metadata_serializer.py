# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
MetadataSerializer 综合单元测试

测试目标：
1. 覆盖 TP、PP、DP 各种场景下的 metadata 序列化/反序列化
2. 验证 MetadataConfig 配置的正确性
3. 验证去重功能的正确性（PP/DP dedup 和 TP dedup）
4. 验证特殊 payload 的处理
5. 覆盖 TP+PP chunked prefill 组合场景
6. 验证自动配置选择逻辑 (_auto_select_config_with_dedup)

运行方式:
    pytest test/pytest/test_metadata_serializer_comprehensive.py -v -s

性能测试:
    pytest test/pytest/test_metadata_serializer_comprehensive.py -v -s --warmup-round=5 --timing-round=20

注意：
- task_id 必须是 hex 格式（如 "00000001"）
- 某些测试需要 global_args 初始化
"""

import pytest
import msgpack

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def configured_packed_tasks_base():
    """Ensure PackedTasksBase is configured"""
    from chitu.task import PackedTasksBase

    if not PackedTasksBase.configured:
        PackedTasksBase.configure(max_num_tasks=32)


@pytest.fixture(autouse=True)
def cleanup_task_pool():
    """Clean up TaskPool after each test"""
    yield
    try:
        from chitu.task import TaskPool

        task_ids = list(TaskPool.pool.keys())
        for tid in task_ids:
            TaskPool.remove(tid)
    except Exception:
        pass


@pytest.fixture
def ensure_global_args():
    """Ensure global_args is initialized with minimal config for Task creation.

    If global_args is already set (by another test), we don't re-set it.
    This avoids AssertionError from set_global_args when running in a shared pytest session.
    """
    from chitu.global_vars import get_global_args, set_global_args

    # Check if already initialized
    try:
        get_global_args()
        return  # Already initialized, don't try to re-set
    except AssertionError:
        pass  # Not initialized, we can set it

    from omegaconf import OmegaConf

    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "op_impl": "cuda",
                    "tp_size": 1,
                    "pp_size": 1,
                    "dp_size": 1,
                    "max_seq_len": 4096,
                    "max_batch_size": 32,
                    "mtp_size": 1,
                    "schedule_overlap": False,
                }
            }
        )
    )


@pytest.fixture
def sample_params():
    """Create test SampleParams"""
    from chitu.task import SampleParams

    return SampleParams(
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        frequency_penalty=0.0,
    )


@pytest.fixture
def prefill_task_factory(sample_params, ensure_global_args):
    """Factory for creating Prefill tasks (requires global_args)"""
    from chitu.task import Task, TaskPool, TaskType

    def _create(task_id: str, tokens: list, consumed: int = 0):
        t = Task(
            task_id=task_id, req=None, sample_params=sample_params, prefix_tokens=tokens
        )
        t.task_type = TaskType.Prefill
        t.consumed_req_tokens = consumed
        TaskPool.add(t)
        return t

    return _create


@pytest.fixture
def decode_task_factory(sample_params, ensure_global_args):
    """Factory for creating Decode tasks (requires global_args)"""
    from chitu.task import Task, TaskPool, TaskType

    def _create(task_id: str, tokens: list):
        t = Task(
            task_id=task_id, req=None, sample_params=sample_params, prefix_tokens=tokens
        )
        t.task_type = TaskType.Decode
        t.consumed_req_tokens = len(tokens)
        TaskPool.add(t)
        return t

    return _create


# ============================================================================
# Test: TP Dispatch (PackedTasksBase msgpack format) with Benchmark
# ============================================================================


class TestTPDispatch:
    """Test TP metadata dispatch using PackedTasksBase msgpack format"""

    @pytest.mark.parametrize("num_tasks", [2, 8, 16])
    def test_prefill_roundtrip(self, num_tasks, prefill_task_factory, record_benchmark):
        """TP Prefill: PackedTasksBase can correctly roundtrip"""
        from chitu.metadata_serializer import MetadataSerializer
        from chitu.task import (
            PackedTasks,
            PackedTasksBase,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        tasks_list = [
            prefill_task_factory(f"{i + 1:08x}", [1, 2, 3, 4, 5], consumed=0)
            for i in range(num_tasks)
        ]
        base = PackedTasks([], tasks=tasks_list)

        serializer = MetadataSerializer(mode="TP")

        def roundtrip():
            data = serializer.serialize_metadata(base)
            return serializer.deserialize_metadata(data)

        payload_type, out, slot_idx = record_benchmark.run(
            roundtrip,
            num_tasks=num_tasks,
            impl="tp_prefill",
        )

        assert type(out) == PackedTasksBase
        assert payload_type == SerializedPackedTasksPayloadType.Prefill
        assert out.num_tasks == num_tasks
        assert all(
            len(send_tokens) == len(recv_tokens)
            for send_tokens, recv_tokens in zip(base.tokens, out.tokens)
        )

    @pytest.mark.parametrize("num_tasks", [2, 8, 16])
    def test_decode_roundtrip(self, num_tasks, decode_task_factory, record_benchmark):
        """TP Decode: PackedTasksBase can correctly roundtrip"""
        from chitu.metadata_serializer import MetadataSerializer
        from chitu.task import (
            PackedTasks,
            PackedTasksBase,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        tasks_list = [
            decode_task_factory(f"{i + 1:08x}", [1, 2, 3, 4, 5])
            for i in range(num_tasks)
        ]
        base = PackedTasks([], tasks=tasks_list)

        serializer = MetadataSerializer(mode="TP")

        def roundtrip():
            data = serializer.serialize_metadata(base)
            return serializer.deserialize_metadata(data)

        payload_type, out, _ = record_benchmark.run(
            roundtrip,
            num_tasks=num_tasks,
            impl="tp_decode",
        )

        assert type(out) == PackedTasksBase
        assert payload_type == SerializedPackedTasksPayloadType.Decode
        assert out.num_tasks == num_tasks
        assert out.task_type == TaskType.Decode
        assert all(
            len(send_tokens) == len(recv_tokens)
            for send_tokens, recv_tokens in zip(base.tokens, out.tokens)
        )
        assert out.prefix_lens == base.prefix_lens

    def test_empty_prefill(self, configured_packed_tasks_base, record_benchmark):
        """TP Empty Prefill"""
        from chitu.metadata_serializer import MetadataSerializer
        from chitu.task import (
            PackedTasks,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        base = PackedTasks([], task_type=TaskType.Prefill)

        serializer = MetadataSerializer(mode="TP")

        def roundtrip():
            data = serializer.serialize_metadata(base)
            return serializer.deserialize_metadata(data)

        payload_type, out, _ = record_benchmark.run(
            roundtrip,
            num_tasks=0,
            impl="tp_empty_prefill",
        )

        assert payload_type == SerializedPackedTasksPayloadType.Prefill
        assert out.num_tasks == 0

    def test_empty_decode(self, configured_packed_tasks_base, record_benchmark):
        """TP Empty Decode"""
        from chitu.metadata_serializer import MetadataSerializer
        from chitu.task import (
            PackedTasks,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        base = PackedTasks([], task_type=TaskType.Decode)

        serializer = MetadataSerializer(mode="TP")

        def roundtrip():
            data = serializer.serialize_metadata(base)
            return serializer.deserialize_metadata(data)

        payload_type, out, _ = record_benchmark.run(
            roundtrip,
            num_tasks=0,
            impl="tp_empty_decode",
        )

        assert payload_type == SerializedPackedTasksPayloadType.Decode
        assert out.num_tasks == 0

    def test_slot_idx_preserved(self, prefill_task_factory, record_benchmark):
        """TP: slot_idx should be preserved"""
        from chitu.metadata_serializer import MetadataSerializer
        from chitu.task import (
            PackedTasks,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        base = PackedTasks(
            [], tasks=[prefill_task_factory("00000001", [1, 2, 3], consumed=0)]
        )

        serializer = MetadataSerializer(mode="TP")

        def roundtrip():
            data = serializer.serialize_metadata(base, slot_idx=42)
            return serializer.deserialize_metadata(data)

        payload_type, out, slot_idx = record_benchmark.run(
            roundtrip,
            num_tasks=1,
            impl="tp_slot_idx",
        )

        assert payload_type == SerializedPackedTasksPayloadType.Prefill
        assert slot_idx == 42


# ============================================================================
# Test: PP/DP Dispatch - Empty Tasks
# ============================================================================


class TestEmptyTasksDispatch:
    """Test empty task serialization/deserialization"""

    def test_empty_prefill_roundtrip(self, record_benchmark):
        """PP/DP Empty Prefill should be handled correctly"""
        from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
        from chitu.task import PackedTasks, TaskType

        serializer = MetadataSerializer(mode="PP")
        empty = PackedTasks([], task_type=TaskType.Prefill)
        config = MetadataConfig.for_prefill()

        def roundtrip():
            data = serializer.serialize_metadata(empty, config=config)
            return serializer.deserialize_metadata(data)

        payload_type, out, _ = record_benchmark.run(
            roundtrip,
            num_tasks=0,
            impl="pp_empty_prefill",
        )

        assert out.task_type == TaskType.Prefill
        assert out.num_tasks == 0

    def test_empty_decode_roundtrip(self, record_benchmark):
        """PP/DP Empty Decode should be handled correctly"""
        from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
        from chitu.task import PackedTasks, TaskType

        serializer = MetadataSerializer(mode="PP")
        empty = PackedTasks([], task_type=TaskType.Decode)
        config = MetadataConfig.for_decode_minimal()

        def roundtrip():
            data = serializer.serialize_metadata(empty, config=config)
            return serializer.deserialize_metadata(data)

        payload_type, out, _ = record_benchmark.run(
            roundtrip,
            num_tasks=0,
            impl="pp_empty_decode",
        )

        assert out.task_type == TaskType.Decode
        assert out.num_tasks == 0


# ============================================================================
# Test: PP Dispatch with Task (requires global_args)
# ============================================================================


class TestPPDispatchWithTask:
    """Test PP metadata dispatch (requires Task objects)"""

    def test_prefill_first_creates_task(self, prefill_task_factory, record_benchmark):
        """PP Prefill first: should create new Task"""
        from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
        from chitu.task import PackedTasks

        task = prefill_task_factory("0000000a", [1, 2, 3, 4, 5], consumed=0)
        packed = PackedTasks([], tasks=[task])
        config = MetadataConfig.for_pp_prefill()

        # Verify correctness with fresh serializer
        verify_serializer = MetadataSerializer(mode="PP")
        verify_data = verify_serializer.serialize_metadata(packed, config=config)
        msg = msgpack.unpackb(verify_data, raw=False)

        assert "tasks" in msg
        assert len(msg["tasks"]) == 1
        task_data = msg["tasks"][0]
        assert "sample_params" in task_data
        assert "tokens" in task_data

        # Benchmark with separate serializers
        def benchmark_fn():
            s = MetadataSerializer(mode="PP")
            return s.serialize_metadata(packed, config=config)

        record_benchmark.run(benchmark_fn, num_tasks=1, impl="pp_prefill_first")

    def test_prefill_chunk_no_redundant_data(
        self, prefill_task_factory, record_benchmark
    ):
        """PP Prefill chunk: should not transmit redundant data"""
        from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
        from chitu.task import PackedTasks

        task = prefill_task_factory("0000000b", [1, 2, 3, 4, 5, 6], consumed=3)

        serializer = MetadataSerializer(mode="PP", enable_dedup=True)
        packed = PackedTasks([], tasks=[task])

        # First transmission
        serializer.serialize_metadata(packed, config=MetadataConfig.for_pp_prefill())

        # Chunk transmission (task already known)
        data = record_benchmark.run(
            lambda: serializer.serialize_metadata(
                packed, config=MetadataConfig.for_pp_prefill()
            ),
            num_tasks=1,
            impl="pp_prefill_chunk",
        )
        msg = msgpack.unpackb(data, raw=False)

        # Known task should not transmit sample_params
        task_data = msg["tasks"][0]
        assert (
            "sample_params" not in task_data or task_data.get("sample_params") is None
        )

    def test_pp_decode_minimal(self, decode_task_factory, record_benchmark):
        """PP Decode: should use minimal config"""
        from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
        from chitu.task import PackedTasks

        task = decode_task_factory("0000000c", [1, 2, 3])

        serializer = MetadataSerializer(mode="PP")
        packed = PackedTasks([], tasks=[task])
        config = MetadataConfig.for_pp_decode()

        data = record_benchmark.run(
            lambda: serializer.serialize_metadata(packed, config=config),
            num_tasks=1,
            impl="pp_decode",
        )
        msg = msgpack.unpackb(data, raw=False)

        # PP Decode should not include sample_params or tokens
        assert msg.get("tasks") is None


# ============================================================================
# Test: DP Dispatch with Task (requires global_args)
# ============================================================================


class TestDPDispatchWithTask:
    """Test DP metadata dispatch (requires Task objects)"""

    def test_dp_prefill_first(self, prefill_task_factory, record_benchmark):
        """DP Prefill first: should include full info"""
        from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
        from chitu.task import PackedTasks

        task = prefill_task_factory("0000000d", [10, 20, 30], consumed=0)
        packed = PackedTasks([], tasks=[task])
        config = MetadataConfig.for_dp_prefill()

        # Verify correctness with fresh serializer
        verify_serializer = MetadataSerializer(mode="DP")
        verify_data = verify_serializer.serialize_metadata(packed, config=config)
        msg = msgpack.unpackb(verify_data, raw=False)

        assert "tasks" in msg
        task_data = msg["tasks"][0]
        assert "sample_params" in task_data
        assert "tokens" in task_data

        # Benchmark with separate serializers
        def benchmark_fn():
            s = MetadataSerializer()
            return s.serialize_metadata(packed, config=config)

        record_benchmark.run(benchmark_fn, num_tasks=1, impl="dp_prefill_first")


# ============================================================================
# Test: Deduplication (PP/DP) with Benchmark
# ============================================================================


class TestDeduplication:
    """Test deduplication functionality (PP/DP)"""

    def test_first_transmission_full_info(self, prefill_task_factory, record_benchmark):
        """First transmission should include full info"""
        from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
        from chitu.task import PackedTasks

        task = prefill_task_factory("000000f0", [1, 2, 3, 4, 5], consumed=0)
        packed = PackedTasks([], tasks=[task])
        config = MetadataConfig.for_prefill()

        # Verify correctness with fresh serializer
        verify_serializer = MetadataSerializer(mode="PP", enable_dedup=True)
        verify_data = verify_serializer.serialize_metadata(packed, config=config)
        msg = msgpack.unpackb(verify_data, raw=False)

        # First transmission, task in new_task_ids
        assert "new_task_ids" in msg
        assert "000000f0" in msg["new_task_ids"]

        # Should include full info
        task_data = msg["tasks"][0]
        assert "sample_params" in task_data
        assert "tokens" in task_data

        # Benchmark with separate serializers
        def benchmark_fn():
            s = MetadataSerializer(enable_dedup=True)
            return s.serialize_metadata(packed, config=config)

        record_benchmark.run(benchmark_fn, num_tasks=1, impl="dedup_first")

    def test_subsequent_transmission_minimal(
        self, prefill_task_factory, record_benchmark
    ):
        """Subsequent transmission should be minimal"""
        from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
        from chitu.task import PackedTasks

        task = prefill_task_factory("000000f1", [1, 2, 3, 4, 5, 6], consumed=0)

        serializer = MetadataSerializer(mode="PP", enable_dedup=True)
        packed = PackedTasks([], tasks=[task])

        # First transmission
        serializer.serialize_metadata(packed, config=MetadataConfig.for_prefill())

        # Simulate chunk prefill
        task.consumed_req_tokens = 3

        # Subsequent transmission - benchmark this
        data = record_benchmark.run(
            lambda: serializer.serialize_metadata(
                packed, config=MetadataConfig.for_prefill()
            ),
            num_tasks=1,
            impl="dedup_subsequent",
        )
        msg = msgpack.unpackb(data, raw=False)

        # Task not in new_task_ids
        assert "000000f1" not in msg.get("new_task_ids", [])

        # Known task should not include sample_params
        task_data = msg["tasks"][0]
        assert (
            "sample_params" not in task_data or task_data.get("sample_params") is None
        )

    def test_mixed_batch_new_and_known(self, prefill_task_factory, record_benchmark):
        """Mixed batch: new and known tasks"""
        from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
        from chitu.task import PackedTasks

        known_task = prefill_task_factory("000000f2", [1, 2, 3, 4, 5, 6], consumed=3)
        new_task = prefill_task_factory("000000f3", [7, 8, 9], consumed=0)
        config_full = MetadataConfig.for_prefill()
        config_incr = MetadataConfig.for_prefill()

        # Verify correctness with fresh serializer
        verify_serializer = MetadataSerializer(mode="PP", enable_dedup=True)
        verify_serializer.serialize_metadata(
            PackedTasks([], tasks=[known_task]),
            config=config_full,
        )
        packed = PackedTasks([], tasks=[known_task, new_task])
        verify_data = verify_serializer.serialize_metadata(packed, config=config_incr)
        msg = msgpack.unpackb(verify_data, raw=False)

        # Verify new_task_ids
        assert "000000f3" in msg["new_task_ids"]
        assert "000000f2" not in msg["new_task_ids"]

        # Verify data
        task_data_map = {d["task_id"]: d for d in msg["tasks"]}

        # Known task should not have sample_params
        assert (
            "sample_params" not in task_data_map["000000f2"]
            or task_data_map["000000f2"].get("sample_params") is None
        )

        # New task should have sample_params
        assert "sample_params" in task_data_map["000000f3"]

        # Benchmark with separate serializers
        def benchmark_fn():
            s = MetadataSerializer(mode="PP", enable_dedup=True)
            s.serialize_metadata(
                PackedTasks([], tasks=[known_task]), config=config_full
            )
            return s.serialize_metadata(packed, config=config_incr)

        record_benchmark.run(benchmark_fn, num_tasks=2, impl="dedup_mixed")


# ============================================================================
# Test: Auto Config Selection (_auto_select_config_with_dedup)
# ============================================================================


class TestAutoConfigSelection:
    """Test automatic config selection with dedup"""

    def test_all_new_tasks_prefill_uses_full(
        self, prefill_task_factory, record_benchmark
    ):
        """All new tasks in Prefill should use full config"""
        from chitu.metadata_serializer import MetadataSerializer
        from chitu.task import PackedTasks

        task = prefill_task_factory("000000a0", [1, 2, 3, 4, 5], consumed=0)
        packed = PackedTasks([], tasks=[task])

        # Verify correctness with fresh serializer
        verify_serializer = MetadataSerializer(mode="PP", enable_dedup=True)
        verify_data = verify_serializer.serialize_metadata(packed, config=None)
        msg = msgpack.unpackb(verify_data, raw=False)

        # Should include full info for new task
        assert "000000a0" in msg.get("new_task_ids", [])
        task_data = msg["tasks"][0]
        assert "sample_params" in task_data
        assert "tokens" in task_data

        # Benchmark with separate serializers
        def benchmark_fn():
            s = MetadataSerializer(mode="PP", enable_dedup=True)
            return s.serialize_metadata(packed, config=None)

        record_benchmark.run(benchmark_fn, num_tasks=1, impl="auto_new_prefill")

    def test_all_known_tasks_decode_uses_minimal(
        self, decode_task_factory, record_benchmark
    ):
        """All known tasks in Decode should use minimal config"""
        from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
        from chitu.task import PackedTasks

        task = decode_task_factory("000000a1", [1, 2, 3])

        serializer = MetadataSerializer(enable_dedup=True)
        packed = PackedTasks([], tasks=[task])

        # First make task known
        serializer.serialize_metadata(packed, config=MetadataConfig.for_prefill())

        # Decode with auto selection - benchmark this
        data = record_benchmark.run(
            lambda: serializer.serialize_metadata(packed, config=None),
            num_tasks=1,
            impl="auto_known_decode",
        )
        msg = msgpack.unpackb(data, raw=False)

        # Should use minimal config for known task
        assert msg.get("tasks") is None


# ============================================================================
# Test: TP+PP Chunked Prefill Scenario with Benchmark
# ============================================================================


class TestTPPPChunkedPrefill:
    """Test TP+PP chunked prefill scenario (tp_size=2, pp_size=2)"""

    def test_chunked_prefill_with_multiple_requests(
        self, configured_packed_tasks_base, record_benchmark
    ):
        """Test chunked prefill with multiple concurrent requests"""
        from chitu.metadata_serializer import MetadataSerializer
        from chitu.task import (
            PackedTasksBase,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        # First request starts
        base1 = PackedTasksBase(
            num_tasks=1,
            task_ids=["00000001"],
            task_type=TaskType.Prefill,
            tokens=[[1, 2, 3, 4, 5]],
            payload_type=SerializedPackedTasksPayloadType.Prefill,
            num_tokens=5,
            has_outputs=[1],
        )

        # Second request joins (new task_ids -> tp_full)
        base2 = PackedTasksBase(
            num_tasks=2,
            task_ids=["00000001", "00000002"],
            task_type=TaskType.Prefill,
            tokens=[[6, 7, 8], [1, 2, 3, 4]],
            payload_type=SerializedPackedTasksPayloadType.Prefill,
            num_tokens=7,
            has_outputs=[1, 1],
        )

        # Verify correctness with fresh serializer
        verify_serializer = MetadataSerializer(mode="TP")
        verify_serializer.serialize_metadata(base1)
        verify_data2 = verify_serializer.serialize_metadata(base2)
        msg2 = msgpack.unpackb(verify_data2, raw=False)

        # Benchmark with separate serializers
        def benchmark_fn():
            s = MetadataSerializer(mode="TP")
            s.serialize_metadata(base1)
            return s.serialize_metadata(base2)

        record_benchmark.run(benchmark_fn, num_tasks=2, impl="tp_pp_new_join")

    def test_slot_idx_for_pp(self, prefill_task_factory, record_benchmark):
        """Test slot_idx handling for PP"""
        from chitu.metadata_serializer import MetadataSerializer
        from chitu.task import (
            PackedTasks,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        serializer = MetadataSerializer(mode="TP")
        base = PackedTasks(
            [], tasks=[prefill_task_factory("00000001", [1, 2, 3], consumed=0)]
        )

        # PP uses slot_idx for pipeline scheduling - benchmark roundtrip
        def roundtrip():
            data = serializer.serialize_metadata(base, slot_idx=1)
            return serializer.deserialize_metadata(data)

        _, _, recv_slot_idx = record_benchmark.run(
            roundtrip,
            num_tasks=1,
            impl="pp_slot_idx",
        )

        assert recv_slot_idx == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
