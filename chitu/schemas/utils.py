# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional
import re
from logging import getLogger
import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


logger = getLogger(__name__)


class ModelConfigResolver:
    """
    Model configuration resolver for dynamically reading configuration values from model's config.json files.

    Main features:
    1. Supports $(config.json:field_name) syntax to reference values from config.json
    2. Supports nested field access, such as "model.num_layers"
    3. Automatically caches loaded configuration files

    Usage examples:
    - In model configuration files: n_heads: "$(config.json:head_dim)"
    - This will read the head_dim field value from the model's config.json file
    - If head_dim = 128 in config.json, n_heads will eventually be set to 128

    Syntax format:
    - $(config.json:field_name) - read single field
    - $(config.json:model.num_layers) - read nested field

    Notes:
    - Requires valid ckpt_dir path
    - config.json file must exist in ckpt_dir directory
    - If field doesn't exist or parsing fails, original value is kept and warning is logged
    """

    def __init__(self):
        self._config_cache: dict[str, dict[str, Any]] = {}

    def resolve_config_value(self, value: Any, ckpt_dir: Optional[str] = None) -> Any:
        match = re.match(r"^\$\(config\.json:([^)]+)\)$", str(value))
        if not match:
            return value

        field_name = match.group(1)

        if not ckpt_dir:
            logger.warning(
                f"Cannot resolve config value '{value}': ckpt_dir not provided"
            )
            return value

        try:
            config_data = self._load_config_json(ckpt_dir)
            if config_data is None:
                logger.warning(f"Cannot load config file: {ckpt_dir}/config.json")
                return value

            # Support nested fields, such as "model.num_layers"
            field_parts = field_name.split(".")
            result = config_data

            for part in field_parts:
                if isinstance(result, dict) and part in result:
                    result = result[part]
                else:
                    logger.warning(
                        f"Config field '{field_name}' does not exist in config.json"
                    )
                    return value

            logger.info(f"Read config from config.json: {field_name} = {result}")
            return result

        except Exception as e:
            logger.warning(f"Error parsing config value '{value}': {e}")
            return value

    def _load_config_json(self, ckpt_dir: str) -> Optional[dict[str, Any]]:

        config_path = Path(ckpt_dir) / "config.json"

        if str(config_path) in self._config_cache:
            return self._config_cache[str(config_path)]

        if not config_path.exists():
            logger.warning(f"Config file does not exist: {config_path}")
            return None

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                self._config_cache[str(config_path)] = config_data
                return config_data
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")
            return None

    def process_config_dict(
        self, config_dict: Any, ckpt_dir: Optional[str] = None
    ) -> dict[str, Any]:
        if isinstance(config_dict, DictConfig):
            config_dict = OmegaConf.to_container(config_dict, resolve=True)

        processed_dict: dict[str, Any] = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                processed_dict[key] = self.process_config_dict(value, ckpt_dir)
            elif isinstance(value, list):
                processed_dict[key] = [
                    (
                        self.process_config_dict(item, ckpt_dir)
                        if isinstance(item, dict)
                        else self.resolve_config_value(item, ckpt_dir)
                    )
                    for item in value
                ]
            else:
                processed_dict[key] = self.resolve_config_value(value, ckpt_dir)

        return processed_dict
