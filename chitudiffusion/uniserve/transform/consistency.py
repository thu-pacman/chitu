import torch
import torch.fx
from torch.fx import Node, Proxy
import torch.nn.functional as F
import dataclasses
from copy import deepcopy
import operator
import inspect
import sympy
import os
import inspect
import re


from typing import Union, Dict


def reduce_any(a, fn):
    """
    Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys.
    """
    if isinstance(a, tuple):
        return any(reduce_any(elem, fn) for elem in a)
        # Support NamedTuple (if it has `_fields`) by repacking into original type.
        # return t if not hasattr(a, "_fields") else type(a)(*t)
    elif isinstance(a, list):
        return any(reduce_any(elem, fn) for elem in a)
    elif isinstance(a, dict):
        return any(reduce_any(v, fn) for k, v in a.items())
    elif isinstance(a, slice):
        return False
        # return slice(
        #     reduce_any(a.start, fn), reduce_any(a.stop, fn), reduce_any(a.step, fn)
        # )
    else:
        return fn(a)


@dataclasses.dataclass
class RedundancyDim:
    shape: tuple[int]

    def __init__(self, shape):
        self.shape = tuple(shape)

    def dim(self):
        return len(self.shape)

    def __getitem__(self, key):
        return self.shape[key]

    def __len__(self):
        return len(self.shape)

    def copy(self):
        return deepcopy(self)

    def __and__(lhs, rhs):
        n = max(lhs.dim(), rhs.dim())
        ret = []
        for i in range(n):
            a = lhs[i] if i < lhs.dim() else 1
            b = rhs[i] if i < rhs.dim() else 1
            ret.append(a & b)
        return RedundancyDim(ret)


@dataclasses.dataclass
class Condition:
    shape: list[sympy.core.basic.Basic]

    def __init__(self, shape):
        self.shape = list(shape)

    def dim(self):
        return len(self.shape)

    def __getitem__(self, key):
        return self.shape[key]

    def __len__(self):
        return len(self.shape)

    def copy(self):
        return deepcopy(self)

    def __and__(lhs, rhs):
        n = max(lhs.dim(), rhs.dim())
        ret = []
        for i in range(n):
            a = lhs[i] if i < lhs.dim() else True
            b = rhs[i] if i < rhs.dim() else True
            ret.append(a & b)
        return Condition(ret)


class ConsistencyProp:
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self.env: Dict[str, Condition] = {}
        self.val = {}

    def propagate(self, *args):
        # for v in args:
        # assert isinstance(v, (Condition)), f"{v} has type {v.__class__}"
        # args_iter = iter(args)
        ids_count = 0

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: self.env[n.name])

        def load_val(a):
            return torch.fx.graph.map_arg(a, lambda n: self.val[n.name])

        def fetch_attr(target: str):
            target_atoms = target.split(".")
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(
                        f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}"
                    )
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        # argsv_iter = iter(
        #     [
        #         torch.randn(2, 4, 32, 32),
        #         torch.tensor(1),
        #         torch.randn(2, 77, 2048),
        #         torch.randn(2, 1280),
        #         torch.randn(2, 6),
        #     ]
        #     + [torch.randn(1) for _ in range(10)]
        # )  # example tensor to verified the validation of the unet
        for node in self.graph.nodes:
            args = load_arg(node.args)
            args_val = load_val(node.args)
            kwargs = load_arg(node.kwargs)
            kwargs_val = load_val(node.kwargs)

            if node.op == "get_attr":
                val = fetch_attr(node.target)
                result = Condition([False] * val.dim()) if torch.is_tensor(val) else val
            elif node.op == "placeholder":
                if "tensor_meta" in node.meta:
                    tensor_meta = node.meta["tensor_meta"]
                    shape = (
                        tensor_meta.shape
                        if hasattr(tensor_meta, "shape")
                        else tensor_meta
                    )
                    if "parameters" not in node.name:
                        # automatically generate input condition symbols
                        result = Condition(
                            [sympy.symbols(f"b{ids_count}")]
                            + [False] * (len(shape) - 1)
                        )
                        print(f"assign {node.name} to condition {result}")
                        if "residuals" not in node.name:
                            ids_count += 1
                    else:
                        result = Condition([False] * len(shape))
                    val = torch.zeros(
                        shape,
                        dtype=(
                            tensor_meta.dtype
                            if hasattr(tensor_meta, "dtype")
                            else torch.float32
                        ),
                    )
                elif hasattr(node, "target") and isinstance(node.target, str):
                    try:
                        val = fetch_attr(node.target)
                        result = (
                            Condition([False] * val.dim())
                            if torch.is_tensor(val)
                            else val
                        )
                    except (RuntimeError, AttributeError):
                        raise RuntimeError(
                            f"Cannot determine shape for placeholder node {node.name}. "
                            f"Node has no tensor_meta and fetch_attr failed for target: {node.target}"
                        )
                else:
                    raise RuntimeError(
                        f"Cannot determine shape for placeholder node {node.name}. "
                        f"Node has no tensor_meta and no target attribute. "
                        f"Consider running ShapeProp before calling propagate."
                    )
            elif node.op == "call_function":
                if node.name == "_log_api_usage_once":
                    self.env[node.name] = Condition([False])
                    self.val[node.name] = torch.zeros(0)
                    continue
                val = node.target(*args_val, **kwargs_val)
                if node.target in [operator.add, operator.mul, operator.truediv]:
                    a = args[0]
                    b = args[1]
                    if isinstance(args[0], (float, int)):
                        a = Condition([True] * 1)
                    if isinstance(args[1], (float, int)):
                        b = Condition([True] * 1)
                    result = a & b
                elif node.target in [F.linear]:
                    result = Condition(args[0][:-1] + [False])
                elif node.target in [F.scaled_dot_product_attention]:
                    result = args[0] & args[1] & args[2]
                elif node.target in [F.conv2d]:
                    assert len(args[0]) == 4
                    # todo: Other dimensions should depend on inputs
                    result = Condition([args[0][0]] + [False] * 3)
                elif node.target in [
                    torch.exp,
                    torch.sin,
                    torch.cos,
                    F.interpolate,
                    F.gelu,
                    F.group_norm,
                    F.layer_norm,
                    F.dropout,
                ]:
                    result = args[0]
                elif node.target in [torch.cat, torch.concat]:
                    # todo
                    cat_args = load_arg(node.args[0])
                    result = cat_args[0]
                    for item in cat_args:
                        result = result & item
                elif node.target in [operator.getitem]:  # Use None to insert a dim of 1
                    if isinstance(args[1], int):
                        result = args[0]
                    else:
                        idx = [args[1]] if isinstance(args[1], type(None)) else args[1]
                        temp_list = []
                        dim = 0
                        for idx, v in enumerate(idx):
                            if v == None:
                                temp_list.append(True)
                            else:
                                temp_list.append(args[0][dim])
                                dim += 1
                        result = Condition(temp_list)
                        if node.name == "getitem":  # HACK for the first getitem
                            result = args[0]
                elif node.target in [F.relu, F.silu]:
                    result = args[0]
                elif node.target in [torch.arange]:
                    result = Condition([False])
                else:
                    assert (
                        False
                    ), f"Unsupported call function node {node.target}. {args=} {kwargs=} {node.args=} {node.kwargs=}"
            elif node.op == "call_method":
                val = getattr(torch.Tensor, node.target)(*args_val, **kwargs_val)
                if node.target in ["to", "float", "contiguous"]:
                    result = args[0]
                elif node.target in ["expand", "chunk"]:  # need speical opt
                    result = args[0]
                    # TODO chunk result is tuple, should add new expressions.
                elif node.target in ["flatten"]:
                    result = Condition([args[0][0]])
                elif node.target in ["mul", "add", "div"]:
                    a = args[0]
                    b = args[1]
                    if isinstance(args[0], (float, int)):
                        a = Condition([True] * 1)
                    if isinstance(args[1], (float, int)):
                        b = Condition([True] * 1)
                    result = a & b
                elif node.target in ["transpose"]:
                    # result = args[0]
                    print(args)
                    lt = []
                    for id in range(len(args[0])):
                        lt.append(args[0][id])
                    lt[args[1]] = args[0][args[2]]
                    lt[args[2]] = args[0][args[1]]
                    result = Condition(lt)
                elif node.target in [
                    "reshape",
                    "view",
                ]:  # conservative strategy, will ruin the consistency
                    if isinstance(args[1], tuple):
                        result = Condition([args[0][0]] + [False] * (len(args[1]) - 1))
                    else:
                        result = Condition([args[0][0]] + [False] * (len(args) - 2))
                elif node.target in ["permute"]:
                    temp_list = []
                    for idx in args:
                        if isinstance(idx, int):
                            temp_list.append(args[0][idx])
                    result = Condition(temp_list)
                else:
                    assert (
                        False
                    ), f"Unsupported call method node {node.target}. {args=} {kwargs=} {node.args=} {node.kwargs=} {node.target.__class__}"
            elif node.op == "call_module":
                val = self.modules[node.target](*args_val, **kwargs_val)
                ops_consist = ["norm", "dropout", "act"]

                for strs in ops_consist:
                    pattern = re.compile(rf".*{strs}.*")
                    if pattern.match(node.target.__str__()):
                        result = args[0]
                        break
                else:
                    if isinstance(self.modules[node.target], torch.nn.Linear):
                        # print("prop linear")
                        result = Condition(args[0][:-1] + [False])
                    elif isinstance(self.modules[node.target], torch.nn.Conv2d):
                        # print("prop conv2d")
                        assert len(args[0]) == 4
                        result = Condition([args[0][0]] + [False] * 3)
                    elif re.compile(rf".*sub.*").match(node.target.__str__()):
                        # print("prop sub")
                        result = args[0]
                    elif re.compile(rf".*ff.*").match(node.target.__str__()):
                        # print("prop ff")
                        result = args[0]
                    elif re.compile(rf".*transformer.*").match(
                        node.target.__str__()
                    ):  # not quite sure
                        # print("prop transformer")
                        result = args[0] & kwargs["encoder_hidden_states"]
                    else:
                        assert (
                            False
                        ), f"Unsupported call module node {node.target.__str__()}, {args=} {kwargs=} {node.args=} {node.kwargs=} {val.shape=} "
            print(
                (
                    f"{{{node.name}: {node.target.__str__()}}}: {result=} {args=} {kwargs=} {node.args=} {node.kwargs=} "
                    + (
                        f"val.shape={val.shape}"
                        if hasattr(val, "shape")
                        else f"val(all_items)={val}"
                    )
                )
            )
            self.env[node.name] = result
            self.val[node.name] = val
