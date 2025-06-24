from hydra.core.config_store import ConfigStore
from chitu.schemas.serve_config import ServeConfig

cs = ConfigStore.instance()
cs.store(name="serve_config_schema", node=ServeConfig)

__all__ = [
    "ServeConfig",
]
