import pytest
from omegaconf import OmegaConf
from chitu.schemas import ServeConfig


def test_type_validation():
    config = ServeConfig()
    omega_config = OmegaConf.structured(config)

    with pytest.raises(Exception):
        omega_config.serve.port = "invalid_port_string"

    with pytest.raises(Exception):
        omega_config.infer.do_load = "not_a_boolean"

    with pytest.raises(Exception):
        omega_config.infer.seed = "not_a_float"
