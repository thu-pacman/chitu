from .common import *
from copy import deepcopy
import queue
import datetime
from torch import multiprocessing as mp

logger = logging.getLogger("Executor")


class Executor:
    engine_registry: EngineRegistry
    requests: Sequence[Request]

    def __init__(self):
        self.engine_registry = EngineRegistry()

    def copy_fully_redundant_engine(self):
        cnt = 0
        for name, engines in self.engine_registry.engines.items():
            for engine in engines.values():
                # if engine is not redundant at all
                if not any(v[0] for v in engine.input_signature.values()):
                    self.engine_registry.register(
                        name, build_fully_redundant_engine(engine)
                    )
                    cnt += 1
                    break
        logger.info(f"Add {cnt} fully redundant engines.")

    def add_engine(self, name: str, engine: Engine):
        self.engine_registry.register(name, engine)

    def set_requests(self, requests):
        self.requests = deepcopy(requests)

    def set_tasks(self, stage, batch_ids, signature):
        logger.debug(f"Run {stage=} {batch_ids=} {signature=}")
        batch = [self.requests[v] for v in batch_ids]
        outputs = self.engine_registry.run(stage, batch, signature)
        output_signature = self.engine_registry.get_output_signature(stage, signature)
        # Store outputs into buffer
        for k, v in outputs.items():
            if output_signature[k][0] or len(batch_ids) == 1:
                # if redundant or single request
                # if not isinstance(v, (dict, list)):
                #     v.fingerprint = self.get_random_fingerprint()
                for i in batch_ids:
                    # buffers[(i, k)] = v
                    self.requests[i].inputs[k] = v
            elif torch.is_tensor(
                v
            ):  # non-redundent, parition the first dim with batches
                assert v.shape[0] % len(batch_ids) == 0
                assert v.shape[0] // len(batch_ids) in [
                    1,
                    2,
                ], "Strict for diffusion"
                chunk_size = v.shape[0] // len(batch_ids)
                for i, batch_id in enumerate(batch_ids):
                    v_single = v[i * chunk_size : (i + 1) * chunk_size]
                    # v_single.fingerprint = self.get_random_fingerprint()
                    # buffers[(i, k)] = v_single
                    self.requests[batch_id].inputs[k] = v_single
            else:
                for i, batch_id in enumerate(batch_ids):
                    self.requests[batch_id].inputs[k] = [v[i]]


class ExecutorConsumer:
    engine_registry: EngineRegistry
    requests: Sequence[Request]
    queue: mp.JoinableQueue

    def __init__(
        self, queue, register_engines: Callable[["ExecutorConsumer", bool], None]
    ):
        self.queue = queue
        self.engine_registry = EngineRegistry()

        register_engines(self, dry_run_only=False)

        # Sync with producer
        self.queue.put("Init")
        self.queue.join()
        logger.info("ExecutorConsumer initilized")

    def copy_fully_redundant_engine(self):
        cnt = 0
        for name, engines in self.engine_registry.engines.items():
            for engine in engines.values():
                # if engine is not redundant at all
                if not any(v[0] for v in engine.input_signature.values()):
                    self.engine_registry.register(
                        name, build_fully_redundant_engine(engine)
                    )
                    cnt += 1
                    break
        logger.info(f"Add {cnt} fully redundant engines.")

    def add_engine(self, name: str, engine: Engine):
        self.engine_registry.register(name, engine)

    def dispacth(self):
        cnt = 0
        finish = False
        # with torch.profiler.record_function(f'dispatch'):
        while True:
            try:
                msg = self.queue.get(block=False)
                logger.debug("Consume msg: %s", msg)
                if msg[0] == "tasks":
                    self.set_tasks(*msg[1:])
                elif msg[0] == "requests":
                    self.set_requests(*msg[1:])
                elif msg[0] == "finish":
                    finish = True
                else:
                    assert False, "Unknown command"
                cnt += 1
            except queue.Empty:
                if cnt > 0:
                    torch.cuda.synchronize()
                    logger.debug("Task done %s times", cnt)
                for i in range(cnt):
                    self.queue.task_done()
                cnt = 0
                if finish:
                    break
                pass
        if False:
            for i, req in enumerate(self.requests):
                image = req.inputs["_output"][0]
                fn = f"output/out_{datetime.datetime.now().strftime('%m%d-%H%M%S')}_{i}.png"
                image.save(fn)
                logger.info(f"Save image to {fn}")
        logger.info("Finished")

    def set_requests(self, requests):
        logger.info(f"set_requsts {len(requests)}")
        self.requests = requests

    def set_tasks(self, stage, batch_ids, signature, cached_stage="0_unet_part1"):
        # with torch.profiler.record_function(f"set_tasks {stage}"):
        logger.debug(
            "Run stage=%s batch_ids=%s signature=%s", stage, batch_ids, signature
        )
        batch = [self.requests[v] for v in batch_ids]
        # TODO: refactor cache
        if stage == cached_stage and hasattr(self, "cache"):
            outputs = self.cache
        else:
            outputs = self.engine_registry.run(stage, batch, signature)
            logger.info("Set cache for %s", stage)
        if stage == cached_stage and not hasattr(self, "cache"):
            self.cache = outputs
        output_signature = self.engine_registry.get_output_signature(stage, signature)

        for k, v in outputs.items():
            if output_signature[k][0] or len(batch_ids) == 1:
                for i in batch_ids:
                    self.requests[i].inputs[k] = v
            elif torch.is_tensor(
                v
            ):  # non-redundent, parition the first dim with batches
                assert v.shape[0] % len(batch_ids) == 0
                assert v.shape[0] // len(batch_ids) in [
                    1,
                    2,
                ], "Strict for diffusion"
                chunk_size = v.shape[0] // len(batch_ids)
                for i, batch_id in enumerate(batch_ids):
                    v_single = v[i * chunk_size : (i + 1) * chunk_size]
                    self.requests[batch_id].inputs[k] = v_single
            else:
                for i, batch_id in enumerate(batch_ids):
                    self.requests[batch_id].inputs[k] = [v[i]]
