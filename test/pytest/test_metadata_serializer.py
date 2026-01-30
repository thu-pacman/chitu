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
                    "max_reqs": 32,
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
        t = Task(task_id=task_id, req=None, params=sample_params, prefix_tokens=tokens)
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
        t = Task(task_id=task_id, req=None, params=sample_params, prefix_tokens=tokens)
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
    def test_prefill_roundtrip(
        self, num_tasks, configured_packed_tasks_base, record_benchmark
    ):
        """TP Prefill: PackedTasksBase can correctly roundtrip"""
        from chitu.metadata_serializer import MetadataSerializer
        from chitu.task import (
            PackedTasksBase,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        task_ids = [f"{i + 1:08x}" for i in range(num_tasks)]
        tokens = [[j for j in range(10)] for _ in range(num_tasks)]

        base = PackedTasksBase(
            num_tasks=num_tasks,
            task_ids=task_ids,
            req_ids=task_ids,
            task_type=TaskType.Prefill,
            tokens=tokens,
            payload_type=SerializedPackedTasksPayloadType.Prefill,
            num_tokens=num_tasks * 10,
            has_outputs=[1] * num_tasks,
            has_model_run=[],
        )

        serializer = MetadataSerializer()

        def roundtrip():
            data = serializer.serialize_metadata(
                base, output_format="packed_tasks_base"
            )
            return serializer.deserialize_metadata(
                data, require_task_creation=False, output_format="packed_tasks_base"
            )

        payload_type, out, slot_idx = record_benchmark.run(
            roundtrip,
            num_tasks=num_tasks,
            impl="tp_prefill",
        )

        assert payload_type == SerializedPackedTasksPayloadType.Prefill
        assert out.num_tasks == num_tasks

    @pytest.mark.parametrize("num_tasks", [2, 8, 16])
    def test_decode_roundtrip(
        self, num_tasks, configured_packed_tasks_base, record_benchmark
    ):
        """TP Decode: PackedTasksBase can correctly roundtrip"""
        from chitu.metadata_serializer import MetadataSerializer
        from chitu.task import (
            PackedTasksBase,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        task_ids = [f"{i + 1:08x}" for i in range(num_tasks)]
        tokens = [[100 + i] for i in range(num_tasks)]

        base = PackedTasksBase(
            num_tasks=num_tasks,
            task_ids=task_ids,
            req_ids=task_ids,
            task_type=TaskType.Decode,
            tokens=tokens,
            payload_type=SerializedPackedTasksPayloadType.Decode,
            num_tokens=num_tasks,
            has_outputs=[1] * num_tasks,
            has_model_run=[1] * num_tasks,
        )

        serializer = MetadataSerializer()

        def roundtrip():
            data = serializer.serialize_metadata(
                base, output_format="packed_tasks_base"
            )
            return serializer.deserialize_metadata(
                data, require_task_creation=False, output_format="packed_tasks_base"
            )

        payload_type, out, _ = record_benchmark.run(
            roundtrip,
            num_tasks=num_tasks,
            impl="tp_decode",
        )

        assert payload_type == SerializedPackedTasksPayloadType.Decode
        assert out.num_tasks == num_tasks
        assert out.task_type == TaskType.Decode

    def test_empty_prefill(self, configured_packed_tasks_base, record_benchmark):
        """TP Empty Prefill"""
        from chitu.metadata_serializer import MetadataSerializer
        from chitu.task import (
            PackedTasksBase,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        base = PackedTasksBase(
            num_tasks=0,
            task_ids=[],
            req_ids=[],
            task_type=TaskType.Prefill,
            tokens=[],
            payload_type=SerializedPackedTasksPayloadType.Prefill,
            num_tokens=0,
            has_outputs=[],
            has_model_run=[],
        )

        serializer = MetadataSerializer()

        def roundtrip():
            data = serializer.serialize_metadata(
                base, output_format="packed_tasks_base"
            )
            return serializer.deserialize_metadata(
                data, require_task_creation=False, output_format="packed_tasks_base"
            )

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
            PackedTasksBase,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        base = PackedTasksBase(
            num_tasks=0,
            task_ids=[],
            req_ids=[],
            task_type=TaskType.Decode,
            tokens=[],
            payload_type=SerializedPackedTasksPayloadType.Decode,
            num_tokens=0,
            has_outputs=[],
            has_model_run=[],
        )

        serializer = MetadataSerializer()

        def roundtrip():
            data = serializer.serialize_metadata(
                base, output_format="packed_tasks_base"
            )
            return serializer.deserialize_metadata(
                data, require_task_creation=False, output_format="packed_tasks_base"
            )

        payload_type, out, _ = record_benchmark.run(
            roundtrip,
            num_tasks=0,
            impl="tp_empty_decode",
        )

        assert payload_type == SerializedPackedTasksPayloadType.Decode
        assert out.num_tasks == 0

    def test_slot_idx_preserved(self, configured_packed_tasks_base, record_benchmark):
        """TP: slot_idx should be preserved"""
        from chitu.metadata_serializer import MetadataSerializer
        from chitu.task import (
            PackedTasksBase,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        base = PackedTasksBase(
            num_tasks=1,
            task_ids=["0000000a"],
            req_ids=["0000000a"],
            task_type=TaskType.Prefill,
            tokens=[[1, 2]],
            payload_type=SerializedPackedTasksPayloadType.Prefill,
            num_tokens=2,
            has_outputs=[1],
            has_model_run=[],
        )

        serializer = MetadataSerializer()

        def roundtrip():
            data = serializer.serialize_metadata(
                base, slot_idx=42, output_format="packed_tasks_base"
            )
            return serializer.deserialize_metadata(
                data, require_task_creation=False, output_format="packed_tasks_base"
            )

        payload_type, out, slot_idx = record_benchmark.run(
            roundtrip,
            num_tasks=1,
            impl="tp_slot_idx",
        )

        assert payload_type == SerializedPackedTasksPayloadType.Prefill
        assert slot_idx == 42


# ============================================================================
# Test: TP Deduplication (tp_full / tp_minimal formats) with Benchmark
# ============================================================================


class TestTPDeduplication:
    """Test TP-specific deduplication using tp_full/tp_minimal formats"""

    @pytest.mark.parametrize("num_tasks", [4, 8, 16])
    def test_first_transmission_uses_tp_full(
        self, num_tasks, configured_packed_tasks_base, record_benchmark
    ):
        """First TP transmission should use tp_full format"""
        from chitu.metadata_serializer import MetadataSerializer
        from chitu.task import (
            PackedTasksBase,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        task_ids = [f"{i + 1:08x}" for i in range(num_tasks)]
        tokens = [[100 + i] for i in range(num_tasks)]

        base = PackedTasksBase(
            num_tasks=num_tasks,
            task_ids=task_ids,
            req_ids=task_ids,
            task_type=TaskType.Decode,
            tokens=tokens,
            payload_type=SerializedPackedTasksPayloadType.Decode,
            num_tokens=num_tasks,
            has_outputs=[1] * num_tasks,
            has_model_run=[1] * num_tasks,
        )

        # First verify correctness with fresh serializer
        verify_serializer = MetadataSerializer()
        verify_data = verify_serializer.serialize_metadata(
            base, output_format="packed_tasks_base"
        )
        msg = msgpack.unpackb(verify_data, raw=False)
        assert msg.get("format") == "tp_full"
        assert "tokens" in msg
        assert "req_ids" in msg

        # Then benchmark (uses separate serializer each call)
        def benchmark_fn():
            s = MetadataSerializer()
            return s.serialize_metadata(base, output_format="packed_tasks_base")

        record_benchmark.run(benchmark_fn, num_tasks=num_tasks, impl="tp_full")

    @pytest.mark.parametrize("num_tasks", [4, 8, 16])
    def test_repeated_transmission_uses_tp_minimal(
        self, num_tasks, configured_packed_tasks_base, record_benchmark
    ):
        """Repeated TP transmission with same task_ids should use tp_minimal"""
        from chitu.metadata_serializer import MetadataSerializer
        from chitu.task import (
            PackedTasksBase,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        task_ids = [f"{i + 1:08x}" for i in range(num_tasks)]
        tokens = [[100 + i] for i in range(num_tasks)]

        base = PackedTasksBase(
            num_tasks=num_tasks,
            task_ids=task_ids,
            req_ids=task_ids,
            task_type=TaskType.Decode,
            tokens=tokens,
            payload_type=SerializedPackedTasksPayloadType.Decode,
            num_tokens=num_tasks,
            has_outputs=[1] * num_tasks,
            has_model_run=[1] * num_tasks,
        )

        serializer = MetadataSerializer()

        # First transmission (tp_full)
        data1 = serializer.serialize_metadata(base, output_format="packed_tasks_base")

        # Repeated transmission (tp_minimal) - benchmark this
        data2 = record_benchmark.run(
            lambda: serializer.serialize_metadata(
                base, output_format="packed_tasks_base"
            ),
            num_tasks=num_tasks,
            impl="tp_minimal",
        )

        msg = msgpack.unpackb(data2, raw=False)
        assert msg.get("format") == "tp_minimal"
        # tp_minimal should not include tokens and req_ids
        assert "tokens" not in msg
        assert "req_ids" not in msg
        # tp_minimal should be smaller
        assert len(data2) < len(data1)

    def test_tp_minimal_deserialize_uses_cached_data(
        self, configured_packed_tasks_base, record_benchmark
    ):
        """TP minimal deserialize should use cached tokens and req_ids"""
        from chitu.metadata_serializer import MetadataSerializer
        from chitu.task import (
            PackedTasksBase,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        base = PackedTasksBase(
            num_tasks=2,
            task_ids=["00000001", "00000002"],
            req_ids=["00000001", "00000002"],
            task_type=TaskType.Decode,
            tokens=[[100], [200]],
            payload_type=SerializedPackedTasksPayloadType.Decode,
            num_tokens=2,
            has_outputs=[1, 1],
            has_model_run=[1, 1],
        )

        sender = MetadataSerializer()
        receiver = MetadataSerializer()

        # First transmission
        data1 = sender.serialize_metadata(base, output_format="packed_tasks_base")
        _, out1, _ = receiver.deserialize_metadata(
            data1, require_task_creation=False, output_format="packed_tasks_base"
        )

        # Second transmission (tp_minimal)
        data2 = sender.serialize_metadata(base, output_format="packed_tasks_base")

        _, out2, _ = record_benchmark.run(
            lambda: receiver.deserialize_metadata(
                data2, require_task_creation=False, output_format="packed_tasks_base"
            ),
            num_tasks=2,
            impl="tp_minimal_deser",
        )

        # Both should have same tokens (receiver uses cache for tp_minimal)
        assert out2.num_tasks == 2
        assert len(out2.tokens) == 2

    def test_tp_dedup_resets_on_task_change(
        self, configured_packed_tasks_base, record_benchmark
    ):
        """TP dedup should reset when task_ids change"""
        from chitu.metadata_serializer import MetadataSerializer
        from chitu.task import (
            PackedTasksBase,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        # First batch
        base1 = PackedTasksBase(
            num_tasks=2,
            task_ids=["00000001", "00000002"],
            req_ids=["00000001", "00000002"],
            task_type=TaskType.Decode,
            tokens=[[100], [200]],
            payload_type=SerializedPackedTasksPayloadType.Decode,
            num_tokens=2,
            has_outputs=[1, 1],
            has_model_run=[1, 1],
        )

        # Second batch with different tasks
        base2 = PackedTasksBase(
            num_tasks=2,
            task_ids=["00000003", "00000004"],
            req_ids=["00000003", "00000004"],
            task_type=TaskType.Decode,
            tokens=[[300], [400]],
            payload_type=SerializedPackedTasksPayloadType.Decode,
            num_tokens=2,
            has_outputs=[1, 1],
            has_model_run=[1, 1],
        )

        # Verify correctness with fresh serializer
        verify_serializer = MetadataSerializer()
        verify_serializer.serialize_metadata(base1, output_format="packed_tasks_base")
        data2 = verify_serializer.serialize_metadata(
            base2, output_format="packed_tasks_base"
        )
        msg = msgpack.unpackb(data2, raw=False)
        # Should use tp_full because task_ids changed
        assert msg.get("format") == "tp_full"

        # Benchmark with separate serializers
        def benchmark_fn():
            s = MetadataSerializer()
            s.serialize_metadata(base1, output_format="packed_tasks_base")
            return s.serialize_metadata(base2, output_format="packed_tasks_base")

        record_benchmark.run(benchmark_fn, num_tasks=2, impl="tp_task_change")

    def test_clear_tp_dedup_state(self, configured_packed_tasks_base, record_benchmark):
        """clear_tp_dedup_state should reset TP dedup tracking"""
        from chitu.metadata_serializer import MetadataSerializer
        from chitu.task import (
            PackedTasksBase,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        base = PackedTasksBase(
            num_tasks=2,
            task_ids=["00000001", "00000002"],
            req_ids=["00000001", "00000002"],
            task_type=TaskType.Decode,
            tokens=[[100], [200]],
            payload_type=SerializedPackedTasksPayloadType.Decode,
            num_tokens=2,
            has_outputs=[1, 1],
            has_model_run=[1, 1],
        )

        # Verify correctness with fresh serializer
        verify_serializer = MetadataSerializer()
        verify_serializer.serialize_metadata(base, output_format="packed_tasks_base")
        verify_serializer.clear_tp_dedup_state()
        data = verify_serializer.serialize_metadata(
            base, output_format="packed_tasks_base"
        )
        msg = msgpack.unpackb(data, raw=False)
        # Same tasks should use tp_full again after clear
        assert msg.get("format") == "tp_full"

        # Benchmark with separate serializers
        def benchmark_fn():
            s = MetadataSerializer()
            s.serialize_metadata(base, output_format="packed_tasks_base")
            s.clear_tp_dedup_state()
            return s.serialize_metadata(base, output_format="packed_tasks_base")

        record_benchmark.run(benchmark_fn, num_tasks=2, impl="tp_after_clear")


# ============================================================================
# Test: PP/DP Dispatch - Empty Tasks
# ============================================================================


class TestEmptyTasksDispatch:
    """Test empty task serialization/deserialization"""

    def test_empty_prefill_roundtrip(self, record_benchmark):
        """PP/DP Empty Prefill should be handled correctly"""
        from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
        from chitu.task import PackedTasks, TaskType

        serializer = MetadataSerializer()
        empty = PackedTasks([], task_type=TaskType.Prefill)
        config = MetadataConfig.for_prefill_full()

        def roundtrip():
            data = serializer.serialize_metadata(empty, config=config)
            return serializer.deserialize_metadata(data, require_task_creation=True)

        payload_type, out, _ = record_benchmark.run(
            roundtrip,
            num_tasks=0,
            impl="pp_empty_prefill",
        )

        assert out.task_type == TaskType.Prefill

    def test_empty_decode_roundtrip(self, record_benchmark):
        """PP/DP Empty Decode should be handled correctly"""
        from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
        from chitu.task import PackedTasks, TaskType

        serializer = MetadataSerializer()
        empty = PackedTasks([], task_type=TaskType.Decode)
        config = MetadataConfig.for_decode_minimal()

        def roundtrip():
            data = serializer.serialize_metadata(empty, config=config)
            return serializer.deserialize_metadata(data, require_task_creation=False)

        payload_type, out, _ = record_benchmark.run(
            roundtrip,
            num_tasks=0,
            impl="pp_empty_decode",
        )

        assert out.task_type == TaskType.Decode


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
        config = MetadataConfig.for_pp_prefill_first()

        # Verify correctness with fresh serializer
        verify_serializer = MetadataSerializer()
        verify_data = verify_serializer.serialize_metadata(packed, config=config)
        msg = msgpack.unpackb(verify_data, raw=False)

        assert "tasks" in msg
        assert len(msg["tasks"]) == 1
        task_data = msg["tasks"][0]
        assert "params" in task_data
        assert "tokens" in task_data

        # Benchmark with separate serializers
        def benchmark_fn():
            s = MetadataSerializer()
            return s.serialize_metadata(packed, config=config)

        record_benchmark.run(benchmark_fn, num_tasks=1, impl="pp_prefill_first")

    def test_prefill_chunk_no_redundant_data(
        self, prefill_task_factory, record_benchmark
    ):
        """PP Prefill chunk: should not transmit redundant data"""
        from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
        from chitu.task import PackedTasks

        task = prefill_task_factory("0000000b", [1, 2, 3, 4, 5, 6], consumed=3)

        serializer = MetadataSerializer(enable_dedup=True)
        packed = PackedTasks([], tasks=[task])

        # First transmission
        serializer.serialize_metadata(
            packed, config=MetadataConfig.for_pp_prefill_first()
        )

        # Chunk transmission (task already known)
        data = record_benchmark.run(
            lambda: serializer.serialize_metadata(
                packed, config=MetadataConfig.for_pp_prefill_chunk()
            ),
            num_tasks=1,
            impl="pp_prefill_chunk",
        )
        msg = msgpack.unpackb(data, raw=False)

        # Known task should not transmit params
        task_data = msg["tasks"][0]
        assert "params" not in task_data or task_data.get("params") is None

    def test_pp_decode_minimal(self, decode_task_factory, record_benchmark):
        """PP Decode: should use minimal config"""
        from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
        from chitu.task import PackedTasks

        task = decode_task_factory("0000000c", [1, 2, 3])

        serializer = MetadataSerializer()
        packed = PackedTasks([], tasks=[task])
        config = MetadataConfig.for_pp_decode()

        data = record_benchmark.run(
            lambda: serializer.serialize_metadata(packed, config=config),
            num_tasks=1,
            impl="pp_decode",
        )
        msg = msgpack.unpackb(data, raw=False)

        task_data = msg["tasks"][0]
        # PP Decode should not include params or tokens
        assert "params" not in task_data or task_data.get("params") is None


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
        config = MetadataConfig.for_dp_prefill_first()

        # Verify correctness with fresh serializer
        verify_serializer = MetadataSerializer()
        verify_data = verify_serializer.serialize_metadata(packed, config=config)
        msg = msgpack.unpackb(verify_data, raw=False)

        assert "tasks" in msg
        task_data = msg["tasks"][0]
        assert "params" in task_data
        assert "tokens" in task_data

        # Benchmark with separate serializers
        def benchmark_fn():
            s = MetadataSerializer()
            return s.serialize_metadata(packed, config=config)

        record_benchmark.run(benchmark_fn, num_tasks=1, impl="dp_prefill_first")

    def test_dp_pp_decode_includes_last_tokens(
        self, decode_task_factory, record_benchmark
    ):
        """DP+PP Decode: should include last_tokens"""
        from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
        from chitu.task import PackedTasks

        task = decode_task_factory("0000000e", [1, 2, 3])
        task.update_response_no_sync(100)  # Set last token

        serializer = MetadataSerializer()
        packed = PackedTasks([], tasks=[task])
        config = MetadataConfig.for_dp_pp_decode()

        data = record_benchmark.run(
            lambda: serializer.serialize_metadata(packed, config=config),
            num_tasks=1,
            impl="dp_pp_decode",
        )
        msg = msgpack.unpackb(data, raw=False)

        task_data = msg["tasks"][0]
        assert "last_token" in task_data


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
        config = MetadataConfig.for_prefill_full()

        # Verify correctness with fresh serializer
        verify_serializer = MetadataSerializer(enable_dedup=True)
        verify_data = verify_serializer.serialize_metadata(packed, config=config)
        msg = msgpack.unpackb(verify_data, raw=False)

        # First transmission, task in new_task_ids
        assert "new_task_ids" in msg
        assert "000000f0" in msg["new_task_ids"]

        # Should include full info
        task_data = msg["tasks"][0]
        assert "params" in task_data
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

        serializer = MetadataSerializer(enable_dedup=True)
        packed = PackedTasks([], tasks=[task])

        # First transmission
        serializer.serialize_metadata(packed, config=MetadataConfig.for_prefill_full())

        # Simulate chunk prefill
        task.consumed_req_tokens = 3

        # Subsequent transmission - benchmark this
        data = record_benchmark.run(
            lambda: serializer.serialize_metadata(
                packed, config=MetadataConfig.for_prefill_incremental()
            ),
            num_tasks=1,
            impl="dedup_subsequent",
        )
        msg = msgpack.unpackb(data, raw=False)

        # Task not in new_task_ids
        assert "000000f1" not in msg.get("new_task_ids", [])

        # Known task should not include params
        task_data = msg["tasks"][0]
        assert "params" not in task_data or task_data.get("params") is None

    def test_mixed_batch_new_and_known(self, prefill_task_factory, record_benchmark):
        """Mixed batch: new and known tasks"""
        from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
        from chitu.task import PackedTasks

        known_task = prefill_task_factory("000000f2", [1, 2, 3, 4, 5, 6], consumed=3)
        new_task = prefill_task_factory("000000f3", [7, 8, 9], consumed=0)
        config_full = MetadataConfig.for_prefill_full()
        config_incr = MetadataConfig.for_prefill_incremental()

        # Verify correctness with fresh serializer
        verify_serializer = MetadataSerializer(enable_dedup=True)
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

        # Known task should not have params
        assert (
            "params" not in task_data_map["000000f2"]
            or task_data_map["000000f2"].get("params") is None
        )

        # New task should have params
        assert "params" in task_data_map["000000f3"]

        # Benchmark with separate serializers
        def benchmark_fn():
            s = MetadataSerializer(enable_dedup=True)
            s.serialize_metadata(
                PackedTasks([], tasks=[known_task]), config=config_full
            )
            return s.serialize_metadata(packed, config=config_incr)

        record_benchmark.run(benchmark_fn, num_tasks=2, impl="dedup_mixed")

    def test_clear_tasks_resets_state(self, prefill_task_factory, record_benchmark):
        """clear_tasks should reset dedup state"""
        from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
        from chitu.task import PackedTasks

        task = prefill_task_factory("000000f4", [1, 2, 3], consumed=0)
        packed = PackedTasks([], tasks=[task])
        config = MetadataConfig.for_prefill_full()

        # Verify correctness with fresh serializer
        verify_serializer = MetadataSerializer(enable_dedup=True)
        verify_serializer.serialize_metadata(packed, config=config)
        verify_serializer.clear_tasks(["000000f4"])
        verify_data = verify_serializer.serialize_metadata(packed, config=config)
        msg = msgpack.unpackb(verify_data, raw=False)

        assert "000000f4" in msg["new_task_ids"]

        # Benchmark with separate serializers
        def benchmark_fn():
            s = MetadataSerializer(enable_dedup=True)
            s.serialize_metadata(packed, config=config)
            s.clear_tasks(["000000f4"])
            return s.serialize_metadata(packed, config=config)

        record_benchmark.run(benchmark_fn, num_tasks=1, impl="dedup_after_clear")


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
        verify_serializer = MetadataSerializer(enable_dedup=True)
        verify_data = verify_serializer.serialize_metadata(packed, config=None)
        msg = msgpack.unpackb(verify_data, raw=False)

        # Should include full info for new task
        assert "000000a0" in msg.get("new_task_ids", [])
        task_data = msg["tasks"][0]
        assert "params" in task_data
        assert "tokens" in task_data

        # Benchmark with separate serializers
        def benchmark_fn():
            s = MetadataSerializer(enable_dedup=True)
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
        serializer.serialize_metadata(packed, config=MetadataConfig.for_prefill_full())

        # Decode with auto selection - benchmark this
        data = record_benchmark.run(
            lambda: serializer.serialize_metadata(packed, config=None),
            num_tasks=1,
            impl="auto_known_decode",
        )
        msg = msgpack.unpackb(data, raw=False)

        # Should use minimal config for known task
        task_data = msg["tasks"][0]
        assert "params" not in task_data or task_data.get("params") is None
        assert "tokens" not in task_data or task_data.get("tokens") is None

    def test_force_include_last_tokens(self, decode_task_factory, record_benchmark):
        """force_include_last_tokens should include last_tokens"""
        from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
        from chitu.task import PackedTasks

        task = decode_task_factory("000000a2", [1, 2, 3])
        task.update_response_no_sync(100)

        serializer = MetadataSerializer(enable_dedup=True)
        packed = PackedTasks([], tasks=[task])

        # First make task known
        serializer.serialize_metadata(packed, config=MetadataConfig.for_prefill_full())

        # Use force_include_last_tokens - benchmark this
        data = record_benchmark.run(
            lambda: serializer.serialize_metadata(
                packed, config=None, force_include_last_tokens=True
            ),
            num_tasks=1,
            impl="force_last_tokens",
        )
        msg = msgpack.unpackb(data, raw=False)

        task_data = msg["tasks"][0]
        assert "last_token" in task_data


# ============================================================================
# Test: TP+PP Chunked Prefill Scenario with Benchmark
# ============================================================================


class TestTPPPChunkedPrefill:
    """Test TP+PP chunked prefill scenario (tp_size=2, pp_size=2)"""

    @pytest.mark.parametrize("chunk_size", [10, 50, 100])
    def test_chunked_prefill_metadata_flow(
        self, chunk_size, configured_packed_tasks_base, record_benchmark
    ):
        """Simulate TP+PP chunked prefill metadata flow"""
        from chitu.metadata_serializer import MetadataSerializer
        from chitu.task import (
            PackedTasksBase,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        serializer = MetadataSerializer()

        # Chunk 1: first tokens
        chunk1 = PackedTasksBase(
            num_tasks=1,
            task_ids=["00000001"],
            req_ids=["00000001"],
            task_type=TaskType.Prefill,
            tokens=[[i for i in range(chunk_size)]],
            payload_type=SerializedPackedTasksPayloadType.Prefill,
            num_tokens=chunk_size,
            has_outputs=[1],
            has_model_run=[],
        )

        data1 = serializer.serialize_metadata(chunk1, output_format="packed_tasks_base")
        msg1 = msgpack.unpackb(data1, raw=False)
        assert msg1.get("format") == "tp_full"

        # Chunk 2: next tokens (same task_id) - benchmark this
        chunk2 = PackedTasksBase(
            num_tasks=1,
            task_ids=["00000001"],
            req_ids=["00000001"],
            task_type=TaskType.Prefill,
            tokens=[[chunk_size + i for i in range(chunk_size)]],
            payload_type=SerializedPackedTasksPayloadType.Prefill,
            num_tokens=chunk_size,
            has_outputs=[1],
            has_model_run=[],
        )

        data2 = record_benchmark.run(
            lambda: serializer.serialize_metadata(
                chunk2, output_format="packed_tasks_base"
            ),
            chunk_size=chunk_size,
            impl="tp_pp_chunk2",
        )
        msg2 = msgpack.unpackb(data2, raw=False)
        # Same task_id, should use tp_minimal
        assert msg2.get("format") == "tp_minimal"

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
            req_ids=["00000001"],
            task_type=TaskType.Prefill,
            tokens=[[1, 2, 3, 4, 5]],
            payload_type=SerializedPackedTasksPayloadType.Prefill,
            num_tokens=5,
            has_outputs=[1],
            has_model_run=[],
        )

        # Second request joins (new task_ids -> tp_full)
        base2 = PackedTasksBase(
            num_tasks=2,
            task_ids=["00000001", "00000002"],
            req_ids=["00000001", "00000002"],
            task_type=TaskType.Prefill,
            tokens=[[6, 7, 8], [1, 2, 3, 4]],
            payload_type=SerializedPackedTasksPayloadType.Prefill,
            num_tokens=7,
            has_outputs=[1, 1],
            has_model_run=[],
        )

        # Verify correctness with fresh serializer
        verify_serializer = MetadataSerializer()
        verify_serializer.serialize_metadata(base1, output_format="packed_tasks_base")
        verify_data2 = verify_serializer.serialize_metadata(
            base2, output_format="packed_tasks_base"
        )
        msg2 = msgpack.unpackb(verify_data2, raw=False)
        # Task set changed, should use tp_full
        assert msg2.get("format") == "tp_full"

        # Benchmark with separate serializers
        def benchmark_fn():
            s = MetadataSerializer()
            s.serialize_metadata(base1, output_format="packed_tasks_base")
            return s.serialize_metadata(base2, output_format="packed_tasks_base")

        record_benchmark.run(benchmark_fn, num_tasks=2, impl="tp_pp_new_join")

    def test_slot_idx_for_pp(self, configured_packed_tasks_base, record_benchmark):
        """Test slot_idx handling for PP"""
        from chitu.metadata_serializer import MetadataSerializer
        from chitu.task import (
            PackedTasksBase,
            TaskType,
            SerializedPackedTasksPayloadType,
        )

        serializer = MetadataSerializer()

        base = PackedTasksBase(
            num_tasks=1,
            task_ids=["00000001"],
            req_ids=["00000001"],
            task_type=TaskType.Prefill,
            tokens=[[1, 2, 3]],
            payload_type=SerializedPackedTasksPayloadType.Prefill,
            num_tokens=3,
            has_outputs=[1],
            has_model_run=[],
        )

        # PP uses slot_idx for pipeline scheduling - benchmark roundtrip
        def roundtrip():
            data = serializer.serialize_metadata(
                base, slot_idx=1, output_format="packed_tasks_base"
            )
            return serializer.deserialize_metadata(
                data, require_task_creation=False, output_format="packed_tasks_base"
            )

        _, _, recv_slot_idx = record_benchmark.run(
            roundtrip,
            num_tasks=1,
            impl="pp_slot_idx",
        )

        assert recv_slot_idx == 1


# ============================================================================
# Test: Sync with TaskPool
# ============================================================================


class TestSyncWithTaskPool:
    """Test sync_with_task_pool functionality"""

    def test_sync_cleans_stale_entries(self, prefill_task_factory, record_benchmark):
        """sync_with_task_pool should clean stale entries"""
        from chitu.metadata_serializer import MetadataSerializer, MetadataConfig
        from chitu.task import PackedTasks, TaskPool

        task1 = prefill_task_factory("000000c0", [1, 2, 3], consumed=0)
        task2 = prefill_task_factory("000000c1", [4, 5, 6], consumed=0)

        serializer = MetadataSerializer(enable_dedup=True)

        # Transmit both tasks
        packed = PackedTasks([], tasks=[task1, task2])
        serializer.serialize_metadata(packed, config=MetadataConfig.for_prefill_full())

        # Get initial dedup stats
        stats = serializer.get_dedup_stats()
        initial_transmitted = stats["transmitted_tasks"]  # Fixed key name

        # Remove task1 from TaskPool (simulate task completion)
        TaskPool.remove("000000c0")

        # Sync with TaskPool - benchmark this
        record_benchmark.run(
            lambda: serializer.sync_with_task_pool(),
            num_tasks=2,
            impl="sync_task_pool",
        )

        # Check that stale entry was cleaned
        stats = serializer.get_dedup_stats()
        assert stats["transmitted_tasks"] < initial_transmitted  # Fixed key name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
