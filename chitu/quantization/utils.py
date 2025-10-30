# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Optional, Any
from chitu.global_vars import get_global_args


def get_quant_kwargs_from_checkpoint_prefix(
    checkpoint_prefix: str, rules: Optional[list] = None
) -> dict[str, Any]:
    if not rules:
        rules = get_global_args().models.quant_config.rules

    for rule in rules:
        pattern = rule.get("regex")
        if not pattern:
            continue
        if not re.search(pattern, checkpoint_prefix):
            continue

        layers = rule.get("layers")
        if layers:
            m = re.search(r"layers\.(\d+)\.", checkpoint_prefix)
            if not m:
                continue
            layer_id = int(m.group(1))
            if layer_id not in layers:
                continue

        return dict(rule.get("kwargs") or {})

    return {}


def get_quant_from_checkpoint_prefix(checkpoint_prefix: str, rules={}) -> Optional[str]:
    if not rules:
        rules = get_global_args().models.quant_config.rules
    for rule in rules:
        pattern = rule.get("regex")
        if pattern and re.search(pattern, checkpoint_prefix):
            layers = rule.get("layers")
            if layers:
                match = re.search(r"layers\.(\d+)\.", checkpoint_prefix)
                if match:
                    layer_id = int(match.group(1))
                    if layer_id not in layers:
                        continue
            return rule["type"]
    return None


def get_backend_from_checkpoint_prefix(checkpoint_prefix: str, rules={}) -> str:
    if not rules:
        rules = get_global_args().models.backend_config.rules
    for rule in rules:
        pattern = rule.get("regex")
        if pattern and re.search(pattern, checkpoint_prefix):
            layers = rule.get("layers")
            if layers:
                match = re.search(r"layers\.(\d+)\.", checkpoint_prefix)
                if match:
                    layer_id = int(match.group(1))
                    if layer_id not in layers:
                        continue
            return rule["backend"]
    return "default"


def get_layer_id_from_checkpoint_prefix(checkpoint_prefix: str, rules={}) -> int:
    if not rules:
        rules = get_global_args().models.quant_config.rules
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


def collect_layers_by_type(type_list: list[str], rules) -> list[int]:
    layer_set = set()
    for rule in rules:
        if rule.get("type") in type_list:
            layers = rule.get("layers")
            if layers:
                layer_set.update(layers)
    return sorted(layer_set)
