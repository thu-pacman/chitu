# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from hydra.core.config_store import ConfigStore
from chitu.schemas.serve_config import ServeConfig
from chitu.schemas.serve_config_rules import ServeConfigRules

cs = ConfigStore.instance()
cs.store(name="serve_config_schema", node=ServeConfig)

__all__ = ["ServeConfig", "ServeConfigRules"]
