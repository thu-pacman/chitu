import logging
from .common import *
from torch import multiprocessing as mp
import random
import torchperf
import copy

logger = logging.getLogger("Scheduler")


class ExecutorProducer:
    queue: mp.JoinableQueue

    def __init__(self, queue):
        self.queue = queue

    def initialize(self):
        # Sync with consumer
        msg = self.queue.get(True)
        assert msg == "Init"
        self.queue.task_done()
        logger.info("ExecutorProducer synced and initilized")

    def set_requests(self, requests):
        # self.requests = deepcopy(requests)
        self.queue.put(("requests", requests))

    def set_tasks(self, stage, batch_ids, signature):
        self.queue.put(("tasks", stage, batch_ids, signature))

    def finish(self):
        self.queue.put(("finish",))
        self.join()

    def join(self):
        self.queue.join()


class Scheduler:
    engine_registry: EngineRegistry
    pipeline_registry: dict[str, list[str]]
    executor: ExecutorProducer

    def __init__(
        self,
        q: mp.JoinableQueue = None,
        register_engines: Callable[["Scheduler", bool], None] = None,
    ):
        # self.pipeline_registry = {"unet": ["unet_head"]}
        self.pipeline_registry = {
            "unet": ["unet_head", "unet_body"],
            "case_edit": ["0_unet_part1", "1_controlnet", "2_unet_part2"],
        }
        self.engine_registry = EngineRegistry()
        random.seed(0)

        register_engines(self, dry_run_only=True)

        # Async executor
        self.queue = q
        if self.queue is not None:
            self.executor = ExecutorProducer(q)
            self.executor.initialize()  # Sync with consumer

    def join(self):
        self.executor.join()

    def finish(self):
        self.executor.finish()

    def get_random_fingerprint(self):
        return random.randint(0, 2147483647)

    def add_engine(self, name: str, engine: Engine):
        self.engine_registry.register(name, engine)

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

    @staticmethod
    def calculate_fingerprint(tensor: torch.Tensor) -> Fingerprint:
        return get_fingerprint(tensor)

    def set_hash_fingerprint(self, requests: Sequence[Request]):
        for req in requests:
            for k, v in req.inputs.items():
                if torch.is_tensor(v):
                    v.fingerprint = self.calculate_fingerprint(v)

    def get_largest_batch(self, tasks: list[Request], task_ids, redundent_inputs, rag):
        def equal_input(a, b) -> bool:
            assert torch.is_tensor(a) == torch.is_tensor(b)
            return get_fingerprint(a) == get_fingerprint(b)

        batch_ids = [task_ids[0]]
        remaining_ids = []
        base_inputs = tasks[task_ids[0]].inputs
        for tid in task_ids[1:]:
            for input_name in redundent_inputs:
                if not equal_input(
                    base_inputs[input_name], tasks[tid].inputs[input_name]
                ):
                    remaining_ids.append(tid)
                    break
            else:
                batch_ids.append(tid)

        return batch_ids, remaining_ids

    def estimate_time(self, stage: str, tasks, task_ids, redundent_inputs, rag):
        time_overhead = 0.1
        # TODO: implement performance model
        if len(task_ids) > 1:
            if len(redundent_inputs) > 1 and redundent_inputs[0] == "sample":
                return len(task_ids) / 3 + time_overhead
            if len(redundent_inputs) > 1:
                return len(task_ids) / 2 + time_overhead
        return len(task_ids) + time_overhead

    def get_stage_input_names(self, stage: str) -> list[str]:
        return self.engine_registry.get_input_names(stage)

    def get_stage_output_names(self, stage: str) -> list[str]:
        return self.engine_registry.get_output_names(stage)

    def dp(self, stage: str, tasks: list[Request]):
        plans: dict[TaskIdBitset, list[list[int]]] = {0: []}
        signatures: dict[TaskIdBitset, list[Signature]] = {0: []}
        times: dict[TaskIdBitset, float] = {0: 0}
        for t in tasks:
            assert t.pipeline_name == tasks[0].pipeline_name

        def set_to_bitset(vs) -> int:
            return sum(1 << v for v in vs)

        def dfs(stage: str, task_ids: list[int]) -> float:
            tasks_bitset = set_to_bitset(task_ids)
            if tasks_bitset in times:
                return times[tasks_bitset]
            min_time = float("inf")
            all_input_names = self.get_stage_input_names(stage)
            # for redundant_inputs in powerset(all_input_names):
            #     # TODO: rag should be a boolean for every input?
            #     # for rag in [True, False]:
            #     for rag in [False]:
            #         signature = build_signature_from_list(
            #             all_input_names, redundant_inputs, rag
            #         )

            # Iterate over engines instead of properties
            for signature in self.engine_registry.get_engine_signatures(stage):
                redundant_inputs = [k for k, v in signature.items() if v[0]]
                rag = [k for k, v in signature.items() if v[1]]
                # skip non-existing engines
                if not self.engine_registry.exist(stage, signature):
                    continue
                batch_ids, remaining_ids = self.get_largest_batch(
                    tasks, task_ids, redundant_inputs, rag
                )
                logger.debug(
                    f"get_largest_batch {signature=} {batch_ids=} {remaining_ids=}"
                )
                est_time = self.estimate_time(
                    stage, tasks, batch_ids, redundant_inputs, rag
                )
                tot_time = est_time + dfs(stage, remaining_ids)
                if tot_time < min_time:
                    min_time = tot_time
                    full_signature = [signature] + signatures[
                        set_to_bitset(remaining_ids)
                    ]
                    plan = [batch_ids] + plans[set_to_bitset(remaining_ids)]
                del tot_time
            assert min_time < float("inf")
            times[tasks_bitset] = min_time
            signatures[tasks_bitset] = full_signature
            plans[tasks_bitset] = plan
            logger.debug(f"DP State: {min_time=} {plan=}")
            return min_time

        full_tasks = list(range(len(tasks)))
        full_bitset = set_to_bitset(full_tasks)
        dfs(stage, full_tasks)
        return times[full_bitset], plans[full_bitset], signatures[full_bitset]

    def schedule(self, stage: str, requests: list[Request]):
        time, plan, signatures = self.dp(stage, requests)
        return time, plan, signatures

    def run(
        self, requests: Sequence[Request]
    ) -> list[tuple[str, list[tuple[int, Signature]]]]:
        # buffers = {}  # store intermediate results
        stages = self.pipeline_registry[requests[0].pipeline_name]
        # Initialize fingerprints
        self.set_hash_fingerprint(requests)
        plan = []

        for stage in stages:
            # tasks = self.construct_tasks(stage, requests, buffers)
            time, id_batches, signatures = self.schedule(stage, requests)
            plan.append((stage, list(zip(id_batches, signatures))))
            for batch_ids, signature in zip(id_batches, signatures):
                logger.debug(f"Run {stage=} {batch_ids=} {signature=}")
                batch = [requests[i] for i in batch_ids]
                outputs = self.engine_registry.run(stage, batch, signature)

                output_signature = self.engine_registry.get_output_signature(
                    stage, signature
                )
                # logger.debug(f"==== stage output: {tensors_to_shapes(outputs).items()}")
                # Store outputs into buffer
                for k, v in outputs.items():
                    if output_signature[k][0] or len(batch_ids) == 1:
                        # if redundant or single request
                        if not isinstance(v, (dict, list)):
                            v.fingerprint = self.get_random_fingerprint()
                        for i in batch_ids:
                            # buffers[(i, k)] = v
                            requests[i].inputs[k] = v
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
                            v_single.fingerprint = self.get_random_fingerprint()
                            # buffers[(i, k)] = v_single
                            requests[batch_id].inputs[k] = v_single
                    else:
                        for i, batch_id in enumerate(batch_ids):
                            requests[batch_id].inputs[k] = [v[i]]
        return plan

    def async_run(
        self,
        requests: Sequence[Request],
        force_sync: bool = False,
        disable_execution: bool = False,
    ) -> list[tuple[str, list[tuple[int, Signature]]]]:
        stages = self.pipeline_registry[requests[0].pipeline_name]
        self.set_hash_fingerprint(requests)
        plan = []
        # with torchperf.time_with_sync("Scheduler schedulinig time", cuda_sync=False):

        # Direct use of `requests` results in RuntimeError: dictionary changed size during iteration
        self.executor.set_requests(copy.deepcopy(requests))
        for stage in stages:
            logger.debug(f"Start schedule stage=%s: requests=%s", stage, requests)
            time, id_batches, signatures = self.schedule(stage, requests)
            plan.append((stage, list(zip(id_batches, signatures))))
            for batch_ids, signature in zip(id_batches, signatures):
                if not disable_execution:
                    self.executor.set_tasks(stage, batch_ids, signature)
                if force_sync:
                    self.join()
                engine = self.engine_registry.get_engine(stage, signature)
                batch = [requests[i] for i in batch_ids]
                dry_run_outputs = engine.dry_run(batch)
                output_signature = engine.output_signature
                logger.debug(
                    "dry_run output stage=%s: dry_run_outputs=%s output_signature=%s",
                    stage,
                    dry_run_outputs,
                    output_signature,
                )
                # Store outputs into buffer
                for i, output_dict in enumerate(dry_run_outputs):
                    for k, v in output_dict.items():
                        batch[i].inputs[k] = v
        return plan
