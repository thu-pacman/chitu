from typing import Callable, Any, TypeVar, Sequence
from itertools import chain, combinations, product
from dataclasses import dataclass
import torch
from torchperf.utils import shapes_to_tensors, tensors_to_shapes
import logging
from uniserve.transform.consistency import Condition

logging.basicConfig(
    level=logging.INFO,
    format="[%(pathname)s:%(lineno)d %(name)s] %(levelname)s - %(message)s",
    force=True,
)

Task = str
Signature = dict[str, tuple[bool, bool]]  #  {input_name:[redundancy, ragged]}
# SignatureKey = frozenset[tuple[str, tuple[bool, bool]]]  #  {input_name:[redundancy, ragged]}
SignatureKey = frozenset  #  {input_name:[redundancy, ragged]}
TaskIdBitset = int
Fingerprint = int


@dataclass
class FakeTensor:
    shape: tuple[int]
    fingerprint: int


@dataclass
class dNode:
    name: str
    belongs_to: str
    merge_to: str
    condition: Condition
    node: torch.fx.Node


@dataclass
class dGraph:
    name: str
    graph: torch.fx.Graph
    num_nodes: int

    def __init__(self, name: str, condition: Condition):
        self.name = name
        self.num_nodes = 0
        self.graph = torch.fx.Graph()

    def __repr__(self):
        ret = f"dGraph("
        ret += f"Name: {self.name}\n"
        ret += f"Graph: {self.graph}\n"
        ret += f"Num Nodes: {self.num_nodes}\n"
        return ret + ")"

    def __getitem__(self, name):
        return self.graph.nodes[name]

    def insert_node_before(self, node: torch.fx.Node, new_node: torch.fx.Node):
        pass

    def insert_node_after(self, node: torch.fx.Node, new_node: torch.fx.Node):
        pass


@dataclass
class Request:
    inputs: dict[str, torch.Tensor | Any]
    pipeline_name: str

    # def __init__(self, pipeline_name, inputs):
    #     self.pipeline_name = pipeline_name
    #     self.inputs = inputs
    def __repr__(self):
        ret = f"Request("
        ret += f"Pipeline: {self.pipeline_name}\n"
        ret += "Inputs:\n"
        for k, v in self.inputs.items():
            ret += f"  {k}: {tensors_to_shapes(v)}"
            if hasattr(v, "fingerprint"):
                ret += f", fg {v.fingerprint}"
            if torch.is_tensor(v):
                ret += f", sum {v.sum()}"
            if isinstance(v, (list, tuple)):  # Dump fingerprint for list and tuple
                ret += f", fg ["
                for vv in v:
                    ret += (
                        f"{vv.fingerprint}," if hasattr(vv, "fingerprint") else "None,"
                    )
                ret += f"]"
            ret += f"\n"
        return ret + ")"

    # redirect get to inputs dict
    def __getitem__(self, name):
        return self.inputs[name]


@dataclass
class Engine:
    input_signature: Signature
    output_signature: Signature
    run: Callable
    dry_run: Callable = None


class EngineRegistry:
    engines: dict[str, dict[SignatureKey, Engine]]
    input_names: dict[str, list[str]]
    output_names: dict[str, list[str]]

    def __init__(self):
        self.engines = {}
        self.input_names = {}
        self.output_names = {}

    def register(self, name: str, engine: Engine):
        if name not in self.engines:
            self.engines[name] = {}
            self.input_names[name] = list(engine.input_signature)
            self.output_names[name] = list(engine.output_signature)
        signature = frozenset(engine.input_signature.items())
        assert signature not in self.engines[name]
        self.engines[name][signature] = engine

    # def __getitem__(self, name, signature:dict[str]):
    #     return self.engines[name][0]
    def exist(self, name, signature: Signature):
        if name not in self.engines:
            return False
        return frozenset(signature.items()) in self.engines[name]

    def get_input_names(self, stage: str) -> list[str]:
        return self.input_names[stage]

    def get_output_names(self, stage: str) -> list[str]:
        return self.output_names[stage]

    def get_engine(self, stage: str, signature: Signature) -> Engine:
        return self.engines[stage][frozenset(signature.items())]

    def get_output_signature(self, stage: str, signature: Signature) -> Signature:
        return self.engines[stage][frozenset(signature.items())].output_signature

    def get_engine_signatures(self, stage: str) -> Sequence[Signature]:
        return (engine.input_signature for engine in self.engines[stage].values())

    def run(
        self, stage: str, requests: list[Request], signature: Signature
    ) -> dict[str, torch.Tensor]:
        engine = self.engines[stage][frozenset(signature.items())]
        inputs = {}
        # construct batched inputs conforming engine input format
        # 1. filter out unused tensors
        # 2. concat inputs into a batch
        for name, (red, rag) in signature.items():
            if red:  # keep only one tensor for redundant input
                inputs[name] = requests[0].inputs[name]
            elif torch.is_tensor(requests[0].inputs[name]):
                inputs[name] = torch.concat(
                    [req.inputs[name] for req in requests], dim=0
                )
            elif isinstance(requests[0].inputs[name], list):
                inputs[name] = list(chain(*(req.inputs[name] for req in requests)))
            else:
                for req in requests:
                    assert req.inputs[name] == requests[0].inputs[name]
                inputs[name] = requests[0].inputs[name]
        raw_outputs = engine.run(**inputs)

        # Wrap outputs into a tuple
        if torch.is_tensor(raw_outputs):
            assert len(self.get_output_names(stage)) == 1
            raw_outputs = (raw_outputs,)
        assert isinstance(raw_outputs, (tuple, list))
        assert len(raw_outputs) == len(self.get_output_names(stage))
        outputs = {
            k: v
            for k, v in zip(self.get_output_names(stage), raw_outputs)
            if not k.startswith("__")  # Skip ununsed outputs
        }
        return outputs


def build_fully_redundant_engine(engine):
    def build_all_redudant_signature(sig: Signature) -> Signature:
        for k, v in sig.items():
            assert not v[0]  # no redundanct version
        return {k: (True, v[1]) for k, v in sig.items()}

    return Engine(
        input_signature=build_all_redudant_signature(engine.input_signature),
        output_signature=build_all_redudant_signature(engine.output_signature),
        run=engine.run,
        dry_run=engine.dry_run,
    )


def get_fingerprint(obj: Any) -> Fingerprint:
    if isinstance(obj, Fingerprint):
        return obj
    elif hasattr(obj, "fingerprint"):
        return obj.fingerprint
    elif isinstance(obj, (tuple, list)):
        a = tuple(get_fingerprint(v) for v in obj)
        return hash(a)
    elif torch.is_tensor(obj):
        obj.fingerprint = obj.sum().item()
        return obj.fingerprint
    elif isinstance(obj, (str,)):
        return hash(obj)
    else:
        assert False, f"No fingerprint for {obj.__class__} object: {obj}"


def fingerprint_and(*fg_or_obj_with_fg: tuple[Fingerprint | Any]) -> Fingerprint:
    # return hash(a+10007)*(b+10003)%1000000007
    a = tuple(get_fingerprint(v) for v in fg_or_obj_with_fg)
    return hash(a)
