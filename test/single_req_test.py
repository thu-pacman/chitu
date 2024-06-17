from cinfer.task import UserRequest, TaskPool, PrefillTask
from cinfer.model import Backend
from cinfer.cinfer_main import cinfer_init, cinfer_run
from cinfer.global_vars import set_global_variables
from faker import Faker
import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base=None, config_path="../example/configs", config_name="serve_config"
)
def main(args: DictConfig):
    set_global_variables()
    Backend.build(args.model)
    cinfer_init(args)
    fake = Faker()
    for i in range(5):
        msg = ""
        for j in range(10):
            msg += fake.sentence()
        req = UserRequest(f"{msg}", f"request_{i}")
        TaskPool.add(PrefillTask(f"prefill_{req.request_id}", req, req.message))
    while len(TaskPool.pool) > 0:
        cinfer_run()


if __name__ == "__main__":
    main()
