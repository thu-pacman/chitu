# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Optional, Any
from omegaconf import ListConfig
from chitu.global_vars import get_global_args
from chitu.checkpoint_prefix import (
    CheckpointPrefix,
    CheckpointPrefixError,
    as_checkpoint_prefix,
)


def _get_consistent_checkpoint_prefix_value(
    checkpoint_prefix: str | CheckpointPrefix,
    *,
    query_name: str,
    getter,
):
    checkpoint_prefix = as_checkpoint_prefix(checkpoint_prefix)
    values = {path: getter(path) for path in checkpoint_prefix.paths}
    expected_value = next(iter(values.values()))
    if all(value == expected_value for value in values.values()):
        return expected_value

    details = ", ".join(f"{path} -> {value!r}" for path, value in values.items())
    raise CheckpointPrefixError(
        f"Inconsistent checkpoint-prefix {query_name} for merged paths: {details}. "
        "Merged checkpoint tensors must use consistent quantization/backend rules."
    )


def _match_rule_for_checkpoint_prefix(checkpoint_prefix: str, rules):
    decorated_checkpoint_prefix = f".{checkpoint_prefix}."  # compatible with prefix.{name}.suffix and prefix.{name} and {name}.suffix
    for rule in rules:
        pattern = rule.get("regex")
        if not pattern:
            continue
        if not re.search(pattern, decorated_checkpoint_prefix):
            continue

        layers = rule.get("layers")
        if layers:
            match = re.search(r"layers\.(\d+)\.", decorated_checkpoint_prefix)
            if not match:
                continue
            layer_id = int(match.group(1))
            if layer_id not in layers:
                continue

        return rule
    return None


def _get_quant_kwargs_from_single_checkpoint_prefix(
    checkpoint_prefix: str, rules: Optional[list | ListConfig] = None
) -> dict[str, Any]:
    if not rules:
        rules = get_global_args().models.quant_config.rules

    rule = _match_rule_for_checkpoint_prefix(checkpoint_prefix, rules)
    if rule is None:
        return {}
    return dict(rule.get("kwargs") or {})


def get_quant_kwargs_from_checkpoint_prefix(
    checkpoint_prefix: str | CheckpointPrefix, rules: Optional[list | ListConfig] = None
) -> dict[str, Any]:
    if not rules:
        rules = get_global_args().models.quant_config.rules

    return _get_consistent_checkpoint_prefix_value(
        checkpoint_prefix,
        query_name="quant kwargs",
        getter=lambda path: _get_quant_kwargs_from_single_checkpoint_prefix(
            path, rules
        ),
    )


def _get_quant_from_single_checkpoint_prefix(
    checkpoint_prefix: str, rules={}
) -> Optional[str]:
    if not rules:
        rules = get_global_args().models.quant_config.rules

    rule = _match_rule_for_checkpoint_prefix(checkpoint_prefix, rules)
    if rule is None:
        return None
    return rule.get("type")


def get_quant_from_checkpoint_prefix(
    checkpoint_prefix: str | CheckpointPrefix, rules={}
) -> Optional[str]:
    if not rules:
        rules = get_global_args().models.quant_config.rules

    return _get_consistent_checkpoint_prefix_value(
        checkpoint_prefix,
        query_name="quantization",
        getter=lambda path: _get_quant_from_single_checkpoint_prefix(path, rules),
    )


def _get_backend_from_single_checkpoint_prefix(checkpoint_prefix: str, rules={}) -> str:
    if not rules:
        rules = get_global_args().models.backend_config.rules

    rule = _match_rule_for_checkpoint_prefix(checkpoint_prefix, rules)
    if rule is None:
        return "default"
    return rule["backend"]


def get_backend_from_checkpoint_prefix(
    checkpoint_prefix: str | CheckpointPrefix, rules={}
) -> str:
    if not rules:
        rules = get_global_args().models.backend_config.rules

    return _get_consistent_checkpoint_prefix_value(
        checkpoint_prefix,
        query_name="backend",
        getter=lambda path: _get_backend_from_single_checkpoint_prefix(path, rules),
    )


def get_layer_id_from_checkpoint_prefix(checkpoint_prefix: str, rules={}) -> int:
    if not rules:
        rules = get_global_args().models.quant_config.rules
    checkpoint_prefix = f".{checkpoint_prefix}."  # compatible with prefix.{name}.suffix and prefix.{name} and {name}.suffix
    for rule in rules:
        pattern = rule.get("regex")
        if pattern and re.search(pattern, checkpoint_prefix):
            layers = rule.get("layers")
            if layers:
                match = re.search(r"layers\.(\d+)\.", checkpoint_prefix)
                if match:
                    layer_id = int(match.group(1))
                    if layer_id in layers:
                        return layer_id
            return 0
    return -1


def collect_layers_by_type(type_list: list[str] | ListConfig, rules) -> list[int]:
    layer_set = set()
    for rule in rules:
        if rule.get("type") in type_list:
            layers = rule.get("layers")
            if layers:
                layer_set.update(layers)
    return sorted(layer_set)
