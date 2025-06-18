from typing import Optional
import re


def get_quant_from_checkpoint_prefix(checkpoint_prefix: str, rules) -> Optional[str]:
    for rule in rules:
        pattern = rule.get("regex")
        if pattern and re.search(pattern, checkpoint_prefix):
            return rule.type
    return None
