from typing import Optional
import re
from typing import List
from typing import Optional, Union, List, Any
from chitu.global_vars import get_global_args


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


def get_backend_from_checkpoint_prefix(
    checkpoint_prefix: str, rules={}
) -> Optional[str]:
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


def collect_layers_by_type(type_list: List[str], rules) -> List[int]:
    layer_set = set()
    for rule in rules:
        if rule.get("type") in type_list:
            layers = rule.get("layers")
            if layers:
                layer_set.update(layers)
    return sorted(layer_set)
