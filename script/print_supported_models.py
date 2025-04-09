import os
import glob
import hydra

from chitu.utils import get_config_dir_path


def print_processed_configs(config_dir: str):
    with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        print("")
        print("Supported models:")
        for model_file in glob.iglob(os.path.join(config_dir, "*.yaml")):
            model_cfg_name = os.path.basename(model_file).replace(".yaml", "")
            cfg = hydra.compose(config_name=model_cfg_name)
            print(f"- {model_cfg_name} ({cfg.source})")
            print(
                f"  Usage: Append `models={model_cfg_name}` command line argument when starting Chitu"
            )


if __name__ == "__main__":
    print_processed_configs(os.path.join(get_config_dir_path(), "models"))
