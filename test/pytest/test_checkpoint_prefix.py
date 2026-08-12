import pytest

from chitu.checkpoint_prefix import (
    CheckpointPrefix,
    CheckpointPrefixError,
    as_checkpoint_prefix,
)
from chitu.quantization.utils import (
    get_backend_from_checkpoint_prefix,
    get_quant_from_checkpoint_prefix,
    get_quant_kwargs_from_checkpoint_prefix,
)


def assert_message_contains_all(error: pytest.ExceptionInfo, *parts: str):
    message = str(error.value)
    for part in parts:
        assert part in message


def test_checkpoint_prefix_single_path():
    prefix = CheckpointPrefix("model.layers.0")

    assert prefix.paths == frozenset({"model.layers.0"})
    assert prefix.is_single()
    assert prefix.single() == "model.layers.0"
    assert str(prefix) == "model.layers.0"


def test_checkpoint_prefix_multi_path_uses_set_semantics():
    prefix = CheckpointPrefix(["a.b", "a.b", ".c.d."])

    assert sorted(prefix.paths) == ["a.b", "c.d"]
    assert not prefix.is_single()


def test_as_checkpoint_prefix_returns_existing_instance():
    prefix = CheckpointPrefix("a")

    assert as_checkpoint_prefix(prefix) is prefix


def test_checkpoint_prefix_join_suffix_to_each_path():
    prefix = CheckpointPrefix(["a", "b"])

    assert sorted((prefix / "c").paths) == ["a.c", "b.c"]


def test_checkpoint_prefix_join_cartesian_product():
    left = CheckpointPrefix(["a", "b"])
    right = CheckpointPrefix(["x", "y"])

    assert sorted((left / right).paths) == ["a.x", "a.y", "b.x", "b.y"]


def test_checkpoint_prefix_join_accepts_leading_dot_suffix_and_empty_prefix():
    assert (CheckpointPrefix("root") / ".child.").paths == frozenset({"root.child"})
    assert (CheckpointPrefix("") / "layers").paths == frozenset({"layers"})


def test_checkpoint_prefix_has_no_len():
    prefix = CheckpointPrefix(["a", "b"])

    with pytest.raises(TypeError):
        len(prefix)


def test_checkpoint_prefix_str_multi_path_raises_with_all_paths():
    prefix = CheckpointPrefix(["a", "b"])

    with pytest.raises(CheckpointPrefixError) as error:
        str(prefix)

    assert_message_contains_all(error, "single checkpoint path", "a", "b")


def test_get_quant_from_checkpoint_prefix_single_path_keeps_existing_behavior():
    rules = [{"regex": "q_proj", "type": "fp8"}]

    assert get_quant_from_checkpoint_prefix("layers.0.self_attn.q_proj", rules) == "fp8"
    assert get_quant_from_checkpoint_prefix("layers.0.self_attn.o_proj", rules) is None


def test_get_quant_from_checkpoint_prefix_multi_path_consistent():
    prefix = CheckpointPrefix.merged(
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
    )
    rules = [
        {"regex": "q_proj", "type": "fp8"},
        {"regex": "k_proj", "type": "fp8"},
    ]

    assert get_quant_from_checkpoint_prefix(prefix, rules) == "fp8"


def test_get_quant_from_checkpoint_prefix_multi_path_quantized_and_unquantized_raises():
    prefix = CheckpointPrefix.merged(
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.o_proj",
    )
    rules = [{"regex": "q_proj", "type": "fp8"}]

    with pytest.raises(CheckpointPrefixError) as error:
        get_quant_from_checkpoint_prefix(prefix, rules)

    assert_message_contains_all(
        error,
        "quantization",
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.o_proj",
        "'fp8'",
        "None",
    )


def test_get_quant_from_checkpoint_prefix_multi_path_different_quant_raises():
    prefix = CheckpointPrefix.merged(
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
    )
    rules = [
        {"regex": "q_proj", "type": "fp8"},
        {"regex": "k_proj", "type": "int8"},
    ]

    with pytest.raises(CheckpointPrefixError) as error:
        get_quant_from_checkpoint_prefix(prefix, rules)

    assert_message_contains_all(
        error, "quantization", "q_proj", "k_proj", "'fp8'", "'int8'"
    )


def test_get_quant_from_checkpoint_prefix_multi_path_common_prefix_rule_matches_all_paths():
    prefix = CheckpointPrefix.merged("a.b.c", "a.b.d")
    rules = [{"regex": r"a\.b", "type": "fp8"}]

    assert get_quant_from_checkpoint_prefix(prefix, rules) == "fp8"


def test_get_quant_from_checkpoint_prefix_multi_path_partial_prefix_rule_raises():
    prefix = CheckpointPrefix.merged("a.b.c", "x.y.z")
    rules = [{"regex": r"a\.b", "type": "fp8"}]

    with pytest.raises(CheckpointPrefixError) as error:
        get_quant_from_checkpoint_prefix(prefix, rules)

    assert_message_contains_all(
        error, "quantization", "a.b.c", "x.y.z", "'fp8'", "None"
    )
    prefix = CheckpointPrefix.merged(
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
    )
    rules = [
        {"regex": "gate_proj", "type": "blockfp4", "kwargs": {"block_size": 128}},
        {"regex": "up_proj", "type": "blockfp4", "kwargs": {"block_size": 128}},
    ]

    assert get_quant_kwargs_from_checkpoint_prefix(prefix, rules) == {"block_size": 128}


def test_get_quant_kwargs_from_checkpoint_prefix_multi_path_different_kwargs_raises():
    prefix = CheckpointPrefix.merged(
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
    )
    rules = [
        {"regex": "gate_proj", "type": "blockfp4", "kwargs": {"block_size": 128}},
        {"regex": "up_proj", "type": "blockfp4", "kwargs": {"block_size": 64}},
    ]

    with pytest.raises(CheckpointPrefixError) as error:
        get_quant_kwargs_from_checkpoint_prefix(prefix, rules)

    assert_message_contains_all(
        error, "quant kwargs", "gate_proj", "up_proj", "128", "64"
    )


def test_get_backend_from_checkpoint_prefix_multi_path_consistent():
    prefix = CheckpointPrefix.merged(
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
    )
    rules = [
        {"regex": "gate_proj", "backend": "triton"},
        {"regex": "up_proj", "backend": "triton"},
    ]

    assert get_backend_from_checkpoint_prefix(prefix, rules) == "triton"


def test_get_backend_from_checkpoint_prefix_multi_path_different_backend_raises():
    prefix = CheckpointPrefix.merged(
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
    )
    rules = [
        {"regex": "gate_proj", "backend": "triton"},
        {"regex": "up_proj", "backend": "cpuinfer"},
    ]

    with pytest.raises(CheckpointPrefixError) as error:
        get_backend_from_checkpoint_prefix(prefix, rules)

    assert_message_contains_all(
        error, "backend", "gate_proj", "up_proj", "triton", "cpuinfer"
    )
