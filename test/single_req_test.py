from cinfer.task import UserRequest, TaskPool, PrefillTask
from cinfer.model import Backend
from cinfer.cinfer_main import cinfer_init, cinfer_run
from cinfer.global_vars import set_global_variables, get_timers
from faker import Faker
import hydra
from omegaconf import DictConfig

from logging import getLogger

logger = getLogger(__name__)


def gen_reqs(num_reqs, prompt_len):
    fake = Faker()
    reqs = []
    for i in range(num_reqs):
        msg = ""
        for j in range(prompt_len):
            msg += fake.word() + " "
        req = UserRequest(f"{msg}", f"request_{i}")
        reqs.append(req)
    return reqs


@hydra.main(
    version_base=None, config_path="../example/configs", config_name="serve_config"
)
def main(args: DictConfig):

    set_global_variables()
    Backend.build(args.model)
    cinfer_init(args)

    reqs = gen_reqs(num_reqs=5, prompt_len=100)
    for req in reqs:
        TaskPool.add(PrefillTask(f"prefill_{req.request_id}", req, req.message))

    while len(TaskPool.pool) > 0:
        cinfer_run()

    for req in reqs:
        logger.info(f"Response: {req.response}")

    timers = get_timers()
    timers.log(
        [
            "cache_prepare",
            "cache_finalize_prefill",
            "cache_finalize_decode",
            "prefill",
            "decode",
        ]
    )


if __name__ == "__main__":
    main()
