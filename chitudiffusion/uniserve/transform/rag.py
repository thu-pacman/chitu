import torch
import torch.fx
from torch.fx import Node, Proxy
import dataclasses
from copy import deepcopy
import operator
import inspect

import uniserve.layers as unn
import uniserve.models as umodel

from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.transformers.transformer_2d import (
    Transformer2DModel,
    BasicTransformerBlock,
)

from typing import Union


@dataclasses.dataclass
class RaggedDim:
    def __str__(self) -> str:
        return "R"

    def __repr__(self) -> str:
        return self.__str__()


@dataclasses.dataclass
class RaggedShape:
    shape: tuple[Union[int, RaggedDim]]
    is_shape: bool = False
    rag_division_ratio: Union[int, None] = None

    def dim(self):
        return len(self.shape)

    def isRagged(self) -> bool:
        return len(self.get_ragged_dims()) > 0

    def get_ragged_dims(self):
        return [i for i, v in enumerate(self.shape) if isinstance(v, RaggedDim)]

    def __getitem__(self, key):
        return self.shape[key]

    def __len__(self):
        return len(self.shape)

    def copy(self):
        return deepcopy(self)


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


def get_info(node: Node) -> RaggedShape:
    return node.info


def set_info(node: Node, attr: RaggedShape):
    node.info = attr


def check_args_have_ragged(args=None, kwargs=None):
    def fn(v):
        result = False
        if isinstance(v, Node):
            t = get_info(v)
            if isinstance(t, list):
                for item in t:
                    result = result | item.isRagged()
            else:
                result = result | t.isRagged()
        return result

    return reduce_any((args, kwargs), fn)


def get_concrete_shape(node: Node):
    tensor_meta = node.meta["tensor_meta"]
    return tensor_meta.shape


def get_concrete_output_shape(node: Node, i):
    return node.meta["tensor_meta"][i].shape


# TODO: set correct regular shape
# TODO: set rag division ratio
class RagProp(torch.fx.Interpreter):
    def __init__(self, gm):
        super().__init__(gm)

    def propagate(self, shapes: list[list[Union[int, RaggedDim, None]]]):
        return super().run(*shapes)

    def init_input_ragged_shape(
        self, ragged_shape: RaggedShape, concrete_shape: list[int]
    ):
        assert not hasattr(self, "ragged_shape")
        # Assumptions: ragged dims will not be reordered
        # ragged_shape[i] = [length of 1st ragged dim, 2nd ragged dim, ..., i-th ragged dim]
        # record compressed shape of each ragged dim for each level of compression
        self.ragged_shape: list[list[int]] = [[], [], []]

        n_ragged = len(ragged_shape.get_ragged_dims())
        assert n_ragged == 2

        for i in ragged_shape.get_ragged_dims():
            self.ragged_shape[2].append(concrete_shape[i])
        self.ragged_shape[1] = [self.ragged_shape[2][0] * self.ragged_shape[2][1]]

    def amend_ragged_shape_by_concrete_shape(
        self, result: RaggedShape, concrete_shape: list[int]
    ):
        result.rag_division_ratio = None
        if result.isRagged():
            input_ragged_shape = self.ragged_shape[len(result.get_ragged_dims())]
        cnt = 0
        for i, v in enumerate(result.shape):
            if isinstance(v, RaggedDim):
                assert (
                    input_ragged_shape[cnt] % concrete_shape[i] == 0
                ), f"{input_ragged_shape[cnt]=}  {concrete_shape[i]=}"
                ratio = input_ragged_shape[cnt] // concrete_shape[i]
                assert (
                    result.rag_division_ratio is None
                    or result.rag_division_ratio == ratio
                )
                result.rag_division_ratio = ratio
                cnt += 1
            else:
                result.shape[i] = concrete_shape[i]
        return result

    def run_node(self, n: Node):
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        if n.name == "_log_api_usage_once":
            return None
        with self._set_current_node(n):
            assert isinstance(n.args, tuple)
            assert isinstance(n.kwargs, dict)
            if (
                check_args_have_ragged(n.args, n.kwargs)
                or "tensor_meta" not in n.meta  # size nodes do not have meta data
                or n.target
                in ["output"]  # output nodes have a tuple of nodes as output
                or n.op in ["placeholder"]
                or n.target == "chunk"
            ):
                result = getattr(self, n.op)(n, n.target, n.args, n.kwargs)
            else:
                result = RaggedShape(shape=list(get_concrete_shape(n)))
            assert isinstance(result, (RaggedShape, list))
            # amend by concrete shapes
            if isinstance(result, RaggedShape):
                if (
                    not result.is_shape
                    and not (
                        n.op == "placeholder"
                        and n.name.startswith("s")  # is dynamic shape placeholder
                    )
                    and not (n.op == "call_function" and n.target == torch.sym_sum)
                    and "tensor_meta" in n.meta
                ):
                    concrete_shape = get_concrete_shape(n)
                    result = self.amend_ragged_shape_by_concrete_shape(
                        result, concrete_shape
                    )
            elif isinstance(result, list):
                for i in range(len(result)):
                    concrete_shape = get_concrete_output_shape(n, i)
                    result[i] = self.amend_ragged_shape_by_concrete_shape(
                        result[i], concrete_shape
                    )
            set_info(n, result)
        return result

    # ======================================================
    # ============= execute different commands =============
    # ======================================================

    def placeholder(self, node: Node, target, args: tuple, kwargs: dict):
        t = RaggedShape(shape=next(self.args_iter), is_shape=False)
        if t.isRagged() and len(t.shape) > 1:
            self.init_input_ragged_shape(t, get_concrete_shape(node))
        elif node.name.startswith("s"):
            t.is_shape = True
        return t

    def call_function(self, node: Node, target, args: tuple, kwargs: dict):
        # These functions are prodcued in call_method
        assert node.target not in [
            torch._C._VariableFunctions.permute,
            torch._C._VariableFunctions.reshape,
        ]

        if len(node.args) == 0:
            result = RaggedShape(shape=self.get_meta(node.args[0]).shape)
        elif node.target == "output":
            return RaggedShape(shape=list())
        elif node.target in [
            torch._C._VariableFunctions.cat,
            torch._C._VariableFunctions.concat,
        ]:
            # TODO: realize concat
            # exit()
            result = deepcopy(get_info(node.args[0][0]))
        elif node.target in [torch.sym_sum]:
            result = deepcopy(get_info(node.args[0][1]))
        elif node.target == operator.mul:
            if isinstance(node.args[0], Node):
                result = deepcopy(get_info(node.args[0]))
            else:
                result = deepcopy(get_info(node.args[1]))
        else:
            attr = deepcopy(get_info(node.args[0]))
            if node.target == operator.getitem:
                # call_function  getitem_120  <built-in function getitem> (size_27, 3)
                if isinstance(attr, list):
                    result = attr[node.args[1]]
                elif attr.is_shape:
                    result = RaggedShape(
                        shape=[attr.shape[node.args[1]]], is_shape=True
                    )
                else:
                    result = attr
            else:
                result = attr
        return result

    def call_method(self, node: Node, target, args: tuple, kwargs: dict):
        if node.target == "reshape" or node.target == "view":
            # call_method reshape_21 reshape (linear_21, 2, getitem_119, getitem_120, 640)
            result = []
            for v in node.args[1:]:
                if isinstance(v, Node):
                    d = get_info(v)
                    assert d.is_shape == True
                    assert len(d.shape) == 1
                    result.append(d.shape[0])
                elif v == -1:
                    result.append(RaggedDim())
                else:
                    result.append(v)
            result = RaggedShape(result)
        elif node.target == "transpose":
            attr = get_info(node.args[0])
            dim0 = node.args[1]
            dim1 = node.args[2]
            new_shape = list(attr.shape)
            new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
            result = RaggedShape(
                new_shape,
                is_shape=attr.is_shape,
                rag_division_ratio=attr.rag_division_ratio,
            )
        elif node.target == "permute":
            # call_method permute_4 permute (l__self___down_blocks_2_attentions_0_norm, 0, 2, 3, 1)
            attr = get_info(node.args[0])
            index = node.args[1:]
            # Index can be in a list or as several seperate args
            if isinstance(index[0], (list, tuple)):
                assert len(index) == 1
                index = index[0]
            result = RaggedShape(
                shape=list(attr.shape[index[i]] for i in range(attr.dim()))
            )
        elif node.target == "chunk":
            # 根据 args，result为chunk以后的结构，kwargs里面有被切分的dim为度
            # args[0]是被chunk的tensor/meta, args[1]是chunk数, kwargs["dim"]为在哪个维度分割
            input_ragged = deepcopy(get_info(node.args[0]))
            num_chunks = node.args[1]
            dim = kwargs.get("dim", -1)  # 默认为-1，如果未指定
            if dim < 0:
                dim += len(input_ragged.shape)
            # 计算chunk后每一份的shape
            out = []
            shape = list(input_ragged.shape)
            chunk_size = (shape[dim] + num_chunks - 1) // num_chunks  # 类似torch.chunk行为
            for i in range(num_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, shape[dim])
                new_shape = shape.copy()
                new_shape[dim] = end - start if end > start else 0
                out.append(
                    RaggedShape(
                        new_shape,
                        is_shape=input_ragged.is_shape,
                        rag_division_ratio=input_ragged.rag_division_ratio,
                    )
                )
            result = out
        elif node.target == "size" or node.target == "add":
            # call_method  size_29 size (add_11,)
            result = deepcopy(get_info(node.args[0]))
            # result.is_shape = True
        else:
            assert (
                not check_args_have_ragged(node.args[1:], node.kwargs)
                or node.target == "mul"
            )
            result = deepcopy(get_info(node.args[0]))
        return result

    def call_module(self, node: Node, target, args: tuple, kwargs: dict):
        # TODO: deal with up/down-sample
        assert not check_args_have_ragged(node.args[1:], node.kwargs)
        result = deepcopy(get_info(node.args[0]))
        return result

    def output(self, node: Node, target, args: tuple, kwargs: dict):
        # if isinstance(args[0], Node):
        #     ret = [deepcopy(get_info(args[0]))]
        # else:
        assert isinstance(args[0], (list, tuple))
        ret = [deepcopy(get_info(n)) for n in args[0]]
        return ret


def prepare_fx_input(args, kwargs):
    ret = []
    if args is not None:
        for v in args:
            if v is not None:
                ret.append(v)
    if kwargs is not None:
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                ret.append(v)
            elif isinstance(v, dict):
                ret += prepare_fx_input(None, v)
            elif v is None:
                continue
            else:
                print(f"{k=} {v=} ignored in fx input")
    return ret


def fx_shape_inference(gm, fx_input: list[torch.Tensor]):
    torch.fx.passes.shape_prop.ShapeProp(gm).propagate(*fx_input)
    return gm


def rag_inference(
    gm: torch.fx.GraphModule, shapes: list[list[Union[int, RaggedDim, None]]]
):
    return RagProp(gm).propagate(shapes)


# Deprecated
def regular_and_rag_shape_inference(
    gm: torch.fx.GraphModule,
    args: list[torch.Tensor],
    kwargs: dict[str, : list[torch.Tensor]],
    ragged_dims: list[list[int]],
):
    fx_input = prepare_fx_input(args, kwargs)

    # infer the numberr of dynamic dim placeholders
    num_ragged_dims = 0
    for node in gm.graph.nodes:
        if node.op == "placeholder" and node.target.startswith("s"):
            num_ragged_dims += 1
    assert num_ragged_dims >= len(ragged_dims)
    if num_ragged_dims != len(ragged_dims):
        print(
            f"Warning: setting {len(ragged_dims)} ragged dimensions but there are {num_ragged_dims} dynamic dims in the fx graph"
        )

    # regular inference: create `num_ragged_dims` new 0 for dynamic shape
    # since dynamo generates graph with `num_rag_dims` extra placeholders
    fx_shape_inference(gm, [0] * num_ragged_dims + fx_input)

    # Rag inference: create `num_ragged_dims` new empty shapes for dynamic shape
    shapes = [[]] * num_ragged_dims + [list(t.shape) for t in fx_input]
    for i, dim in ragged_dims:
        shapes[i + num_ragged_dims][dim] = RaggedDim()
    rag_inference(gm, shapes)

    # torchperf.torch_dynamo.draw_simple_graph(gm_unet, "test.svg")


def regular_and_rag_shape_inference_with_fx_inputs(
    gm: torch.fx.GraphModule,
    fx_inputs: list[torch.Tensor],
    ragged_dims: dict[torch.Tensor, list[int]],
):
    assert isinstance(ragged_dims, dict)
    fx_concrete = []
    for item in fx_inputs:
        if isinstance(item, torch.SymInt):
            fx_concrete.append(int(item))
        else:
            fx_concrete.append(item)
    fx_shape_inference(gm, fx_concrete)
    # exit()
    # Rag inference: create `num_ragged_dims` new empty shapes for dynamic shape
    shapes = [list(t.shape) if torch.is_tensor(t) else list() for t in fx_inputs]
    for i, tensor in enumerate(fx_inputs):
        if torch.is_tensor(tensor) and tensor in ragged_dims:
            for dim in ragged_dims[tensor]:
                shapes[i][dim] = RaggedDim()
        if isinstance(tensor, torch.SymInt):
            shapes[i].append(RaggedDim())
    # print("Ragged shape inference shapes:")
    # for idx, s in enumerate(shapes):
    #     print(f"  input {idx}: shape={s}")
    rag_inference(gm, shapes)


# def check_ragged_dim(shape:RaggedShape, ragged_dims: list[int]):
#     for d in ragged_dims:
#         if shape.shape[i] != RaggedDim():
#             return False
#     return True


class RagTransformer(torch.fx.Transformer):
    def __init__(self, module):
        super().__init__(module)
        # 禁用垃圾回收，保留所有节点的 proxy 以便后续访问
        self.garbage_collect_values = False
        # initialize index proxies
        self.indices = {}  # (name, divisor) : proxy
        for dims in [1, 2]:
            for device in ["cuda", "cpu"]:
                name = f"idx{dims}d_{device}"
                self.indices[(name, 1)] = Proxy(
                    self.new_graph.placeholder(
                        name, default_value=inspect.Signature.empty
                    ),
                    self.tracer,
                )
        # cumulative index for FA2
        name = f"cum_idx1d_cuda"
        self.indices[(name, 1)] = Proxy(
            self.new_graph.placeholder(name, default_value=inspect.Signature.empty),
            self.tracer,
        )
        name = f"prompt_cum_idx1d_cuda"
        self.indices[(name, 1)] = Proxy(
            self.new_graph.placeholder(name, default_value=inspect.Signature.empty),
            self.tracer,
        )

    def divide_index2d(self, idx, ratio, device=None):
        return idx // self.index_2d_divisor(ratio, device=device)

    def index_2d_divisor(self, v, device=None):
        return torch.tensor([[v], [v], [v**2]], dtype=torch.int64, device=device)

    def get_index(self, dims: int, divisor: int, device: str, cumulative=False):
        assert isinstance(dims, int)
        assert isinstance(divisor, int)
        assert isinstance(device, str)
        name = f"idx{dims}d_{device}"
        if cumulative:
            name = "cum_" + name
        key = (name, divisor)
        if key not in self.indices:
            if divisor > 1:  # create new index by division
                if dims == 2:
                    self.indices[key] = self.divide_index2d(
                        self.indices[(name, 1)], divisor, device
                    )
                else:
                    self.indices[key] = self.indices[(name, 1)] // divisor
        return self.indices[key]

    def run_node(self, n: Node) -> torch.fx.Proxy:
        with self._set_current_node(n):
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            rshape = get_info(n)
            if isinstance(rshape, (list, tuple)):  # For output node
                rshape = rshape[0]
            if isinstance(rshape, RaggedShape):
                if rshape.is_shape:  # shape operations are static
                    return None
                elif (
                    check_args_have_ragged(n.args, n.kwargs)
                    or rshape.isRagged()
                    or n.target == "view"
                    or n.target == "transpose"
                ):
                    return getattr(self, n.op)(n, n.target, args, kwargs)
            # Output or Non-rag operations
            return getattr(super(), n.op)(n.target, args, kwargs)

    def placeholder(self, n: Node, target, args, kwargs) -> torch.fx.Proxy:
        assert isinstance(target, str)
        default_value = next(iter(args)) if args else inspect.Signature.empty
        t = Proxy(
            self.new_graph.placeholder(target, default_value=default_value), self.tracer
        )
        shape = get_info(n)
        if len(shape) == 4:
            assert shape.get_ragged_dims() == [2, 3]
            idx2d_cpu = self.get_index(2, shape.rag_division_ratio, "cpu")
            t = torch.ops.uniserve.ragged_nchw_to_nhwc(t, shape[1], idx2d_cpu)
        return t

    def call_function(self, n: Node, target, args, kwargs):
        if target == torch.sigmoid:
            return torch.neg(*args, **kwargs)
        elif target == torch.nn.functional.interpolate:
            assert len(args) == 1
            in_shape = get_info(n.args[0])
            out_shape = get_info(n)
            x = args[0]
            in_index = self.get_index(2, in_shape.rag_division_ratio, "cpu")
            out_index = self.get_index(2, out_shape.rag_division_ratio, "cpu")
            x = torch.ops.uniserve.ragged_nhwc_to_nchw(x, in_shape[1], in_index)
            x = torch.ops.uniserve.ragged_nchw_interpolate(
                x, in_shape[1], in_index, **kwargs
            )
            x = torch.ops.uniserve.ragged_nchw_to_nhwc(x, out_shape[1], out_index)
            return x
        elif target == torch.nn.functional.scaled_dot_product_attention:
            q, k, v = args[:3]
            # q = self.env[n.args[0].args[0]]
            # k = self.env[n.args[1].args[0]]
            # v = self.env[n.args[2].args[0]]
            q_info = get_info(n.args[0])
            k_info = get_info(n.args[1])
            v_info = get_info(n.args[2])

            q_view_node = n.args[0].args[0]
            view_args = q_view_node.args[1:]
            num_heads = view_args[2]
            head_dim = view_args[3]
            hidden_dim = num_heads * head_dim
            print(f"{num_heads=} {head_dim=} {hidden_dim=}")
            # 使用 reshape 而不是 view，因为 transpose 跳过后 tensor 可能不连续
            q = q.reshape(-1, num_heads, head_dim)
            k = k.reshape(-1, num_heads, head_dim)
            v = v.reshape(-1, num_heads, head_dim)
            # print(f"{self.env[n.args[0].args[0]]=} {self.env[n.args[1].args[0]]=} {self.env[n.args[2].args[0]]=}")
            # print(f"{get_info(n.args[0].args[0])=} {get_info(n.args[1].args[0])=} {get_info(n.args[2].args[0])=}")
            print(f"{n.name} qkv_info: {q_info}, {k_info}, {v_info}")

            cu_seqlen_q = self.get_index(  # sequence cum index cuda
                1,
                get_info(n.args[0]).rag_division_ratio,
                "cuda",
                cumulative=True,
            )
            cu_seqlen_k = self.indices[
                ("prompt_cum_idx1d_cuda", 1)
            ]  # prompt cum index cuda
            # Calculate max_len as integer
            divisor = get_info(n.args[0]).rag_division_ratio
            max_len = 128 * 128 // (divisor**2)
            # return torch.ops.uniserve.flashattn_varlen_fwd(
            #     q, k, v, cu_seqlen_q, cu_seqlen_q,
            # )
            is_self_attention = (
                q_info.isRagged() and k_info.isRagged() and v_info.isRagged()
            )
            print(f"{n.name} is_self_attention: {is_self_attention}")
            if is_self_attention:
                tensor_out = torch.ops.uniserve.flashattn_varlen_fwd(
                    q,
                    k,
                    v,
                    cu_seqlen_q,
                    cu_seqlen_q,
                    max_len,
                    max_len,
                )
                return tensor_out.reshape(-1, hidden_dim)
            else:
                tensor_out = torch.ops.uniserve.flashattn_varlen_fwd(
                    q,
                    k,
                    v,
                    cu_seqlen_q,
                    cu_seqlen_k,
                    max_len,
                    77,
                )
                return tensor_out.reshape(-1, hidden_dim)
        else:
            print(f"Non-replaced call_function: {target}")
        return super().call_function(target, args, kwargs)

    def call_method(self, n: Node, target, args, kwargs):
        if target in ["permute", "reshape", "transpose", "view"]:
            tensor, *args_tail = args
            return tensor
        if target in ["add"]:
            # print(f"{n.name} add args  {n.args[0].name}:{get_info(n.args[0])}, {n.args[1].name}:{get_info(n.args[1])}")
            args0_is_ragged = get_info(n.args[0]).isRagged()
            args1_is_ragged = get_info(n.args[1]).isRagged()
            # print(f"{args0_is_ragged=} {args1_is_ragged=} {args0_is_ragged ^ args1_is_ragged=}")
            if args0_is_ragged and not args1_is_ragged:
                print(
                    "add two different ragged type tensors, use addB_jr_rr to replace."
                )
                divisor_in = get_info(n.args[0]).rag_division_ratio
                b_tensor = self.env[n.args[1].args[0]]
                print(f"{get_info(n.args[1].args[0])=}")
                return torch.ops.uniserve.addB_jr_rr(
                    args[0], self.get_index(2, divisor_in, "cuda"), b_tensor
                )
        return super().call_method(target, args, kwargs)

    def add_module(self, old_target, new_mod):
        self.tracer.root.add_module("u_" + old_target, new_mod)

    def call_module(self, n: Node, target, args, kwargs):
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        # assumption: current datalayout is [nhw, c] or [nSeq, c]
        divisor_in = get_info(n.args[0]).rag_division_ratio
        divisor_out = get_info(n).rag_division_ratio
        if isinstance(submod, torch.nn.Conv2d):
            in_channels = get_info(n.args[0])[1]
            new_mod = unn.RaggedNhwcConv2d(submod)
            # add new submodules to the global module self.tracer.root
            self.tracer.root.add_module("u_" + target, new_mod)
            return new_mod(
                args[0],
                self.get_index(2, divisor_in, "cuda"),
                self.get_index(2, divisor_in, "cpu"),
                self.get_index(2, divisor_out, "cuda"),
                self.get_index(2, divisor_out, "cpu"),
            )
        elif isinstance(submod, ResnetBlock2D):
            new_mod = umodel.RaggedResnetBlock2D_nhwc(submod)
            self.add_module(target, new_mod)

            assert len(args) == 2
            assert (
                divisor_in == divisor_out
            ), "TODO: support ResNet with down/up sampling"
            return self.tracer.call_module(
                new_mod,
                new_mod.forward,
                (
                    args[0],
                    self.get_index(1, divisor_in**2, "cuda", True),
                    self.get_index(2, divisor_in, "cuda"),
                    self.get_index(2, divisor_in, "cpu"),
                    args[1],  # temb
                ),
                kwargs,  # {'scale':}
            )
        elif isinstance(submod, BasicTransformerBlock):
            new_mod = umodel.RaggedTransformerBlock_nhwc(
                submod, get_info(n.args[0]).rag_division_ratio
            )
            self.add_module(target, new_mod)
            assert len(args) == 1
            return self.tracer.call_module(
                new_mod,
                new_mod.forward,
                (
                    args[0],
                    self.get_index(  # sequence cum index cuda
                        1,
                        get_info(n.args[0]).rag_division_ratio,
                        "cuda",
                        cumulative=True,
                    ),
                    self.indices[("prompt_cum_idx1d_cuda", 1)],  # prompt cum index cuda
                    self.get_index(1, get_info(n.args[0]).rag_division_ratio, "cpu"),
                ),
                kwargs,
            )
        elif isinstance(submod, torch.nn.modules.normalization.GroupNorm):
            new_mod = unn.RaggedNhwcGroupNorm(submod)
            self.add_module(target, new_mod)
            assert len(args) == 1 and len(kwargs) == 0
            return self.tracer.call_module(
                new_mod,
                new_mod.forward,
                (
                    *args,
                    self.get_index(
                        1, get_info(n.args[0]).rag_division_ratio ** 2, "cuda", True
                    ),
                    self.get_index(2, get_info(n.args[0]).rag_division_ratio, "cpu"),
                ),
                kwargs,
            )
        elif isinstance(submod, torch.nn.Linear):
            # TODO: Check the last dim is not ragged
            None

        return super().call_module(target, args, kwargs)

    def get_attr(self, n: Node, target, args, kwargs):
        return super().get_attr(target, args, kwargs)

    def output(self, n: Node, target, args, kwargs):
        rshape: RaggedShape = get_info(n)[0]
        print("output", rshape)
        assert len(args) == 1
        if len(rshape.get_ragged_dims()) == 2:
            args = (
                (
                    torch.ops.uniserve.ragged_nhwc_to_nchw(
                        args[0][0],
                        rshape[1],
                        self.get_index(2, rshape.rag_division_ratio, "cpu"),
                    ),
                ),
            )
        return super().output(target, args, kwargs)
