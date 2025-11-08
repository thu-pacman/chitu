import pytest
import tempfile
import json
from pathlib import Path
from omegaconf import OmegaConf
from chitu.schemas import ServeConfig, ServeConfigRules
from chitu.schemas.utils import ModelConfigResolver


def test_type_validation():
    config = ServeConfig()
    omega_config = OmegaConf.structured(config)

    with pytest.raises(Exception):
        omega_config.serve.port = "invalid_port_string"

    with pytest.raises(Exception):
        omega_config.debug.skip_model_load = "not_a_boolean"

    with pytest.raises(Exception):
        omega_config.infer.seed = "not_a_float"


class TestServeConfigRules:
    @pytest.fixture
    def callback(self):
        return ServeConfigRules()

    @pytest.fixture
    def config(self):
        return OmegaConf.create(
            {
                "models": {
                    "name": "Qwen3-32B",
                    "type": "hf-llama",
                    "tokenizer_type": "hf",
                    "n_heads": 64,
                    "n_kv_heads": 8,
                },
                "serve": {"port": 21002},
                "infer": {
                    "num_blocks": 2,
                    "attn_type": "flash_attn",
                    "op_impl": "torch",
                    "bind_process_to_cpu": "auto",
                    "bind_thread_to_cpu": "physical_core",
                },
                "scheduler": {"type": "prefill_first"},
            }
        )

    def test_valid_num_blocks(self, callback, config):
        callback.on_job_start(config=config)

    def test_invalid_num_blocks(self, callback, config):
        config.infer.num_blocks = -2
        with pytest.raises(SystemExit) as exc_info:
            callback.on_job_start(config=config)
        assert exc_info.value.code == 1

    def test_invalid_attn_type(self, callback, config):
        config.infer.attn_type = "not_in_valid_range"
        with pytest.raises(SystemExit) as exc_info:
            callback.on_job_start(config=config)
        assert exc_info.value.code == 1

    def test_resolve_config_value(self, callback, config):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_data = {
                "head_dim": 128,
            }

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f)

            config.models.ckpt_dir = temp_dir
            config.models.n_heads = "$(config.json:head_dim)"
            model_resolver = ModelConfigResolver()
            config.models = model_resolver.process_config_dict(
                config.models, config.models.ckpt_dir
            )
            assert config.models.n_heads != 64
            assert config.models.n_heads == 128
