import os
import hydra
import safetensors.torch
import torch.distributed
from omegaconf import DictConfig

from chitu.chitu_main import chitu_init
from chitu.backend import Backend
from chitu.utils import get_config_dir_path


@hydra.main(
    version_base=None,
    config_path=get_config_dir_path(),
    config_name=os.getenv("CONFIG_NAME", "serve_config"),
)
def main(args: DictConfig):
    target_dir = os.getenv("PREPROCESS_AND_SAVE_DIR")
    os.makedirs(target_dir, exist_ok=True)

    chitu_init(args)

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        os.system(f"cp -r {args.models.ckpt_dir}/*.json {target_dir}/")
        os.system(f"cp -r {args.models.tokenizer_path}/*.json {target_dir}/")

    safetensors.torch.save_file(
        Backend.model.state_dict(), target_dir + f"/model.rank{rank}.safetensors"
    )


if __name__ == "__main__":
    main()
