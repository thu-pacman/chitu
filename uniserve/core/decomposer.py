import torch.fx
import torch
import builtins
from uniserve.transform.consistency import ConsistencyProp
from uniserve.core.common import dNode, dGraph
from torch.fx.passes import split_utils


class Decomposer:
    threshold: int
    tmp_counts: dict[str, int]
    fuse_edge: dict[str, list[str]]
    in_dim: dict[str, int]
    merge_to: dict[str, str]

    def __init__(self, threshold: int = 10, name: str = "split_graph"):
        self.threshold = threshold
        self.tmp_counts = {}
        self.fuse_edge = {}
        self.in_dim = {}
        self.merge_to = {}
        self.name = name

    def init_params(self, gm: torch.fx.GraphModule, fx_args: list[torch.Tensor]):
        print(f"len(fx_args)={len(fx_args)}")
        iter_fx_args = iter(fx_args)
        new_fx_args = []
        for node in gm.graph.nodes:
            if node.name == "_log_api_usage_once":
                gm.graph.erase_node(node)
            if node.op != "placeholder":
                continue
            is_plain_faketensor_placeholder = not isinstance(
                node.meta["example_value"], torch.nn.Parameter
            )
            value = next(iter_fx_args)
            if not is_plain_faketensor_placeholder:
                # Replace the original placeholder node with a get_attr node at the same position, keeping the name the same
                orignal_name = node.name
                gm.register_parameter(node.name, value)
                with gm.graph.inserting_before(node):
                    get_attr_node = gm.graph.get_attr(node.name)
                get_attr_node.name = orignal_name  # keep the name the same
                # get_attr_node.tag = node.tag
                node.replace_all_uses_with(get_attr_node)
                gm.graph.erase_node(node)
            else:
                new_fx_args.append(value)
        # gm.recompile()
        return gm, new_fx_args

    def fuse_graph_topo_sort(self, gm: torch.fx.GraphModule):
        for node in gm.graph.nodes:
            if node.tag in ["True", "False"]:
                continue
            for inp_node in node.all_input_nodes:
                if inp_node.tag in ["True", "False"]:
                    continue
                if self.fuse_edge.get(inp_node.tag, None) is None:
                    self.fuse_edge[inp_node.tag] = []
                self.fuse_edge[inp_node.tag].append(node.tag)
        print(f"fuse_edge: {self.fuse_edge}")
        for tag, edges in self.fuse_edge.items():
            seen = set()
            deduped = []
            for item in edges:
                if item not in seen and item != tag:
                    deduped.append(item)
                    seen.add(item)
            self.fuse_edge[tag] = deduped
            if self.in_dim.get(tag, None) is None:
                self.in_dim[tag] = 0
            for item in self.fuse_edge[tag]:
                print(f"edge {tag} -> {item}")
                if self.in_dim.get(item, None) is None:
                    self.in_dim[item] = 0
                self.in_dim[item] += 1

        queue = []
        qhead = 0
        for item in self.in_dim.keys():
            if self.in_dim[item] == 0:
                queue.append(item)
        while qhead < len(queue):
            item = queue[qhead]
            qhead += 1
            for edge in self.fuse_edge[item]:
                self.in_dim[edge] -= 1
                if self.in_dim[edge] == 0:
                    queue.append(edge)
        print(f"queue: {queue}")
        for item in reversed(queue):
            if self.tmp_counts.get(item) < self.threshold:
                connected = self.fuse_edge.get(item, [])
                if connected:
                    min_index = len(queue)
                    merge_to_tag = None
                    for conn in connected:
                        if conn in queue:
                            temp = self.merge_to[conn]
                            idx = queue.index(temp)
                            if idx < min_index:
                                min_index = idx
                                merge_to_tag = temp
                    if merge_to_tag is not None:
                        self.merge_to[item] = merge_to_tag
                    else:
                        raise RuntimeError(
                            f"Node '{item}' has no connected nodes to merge with."
                        )
            else:
                self.merge_to[item] = item
        print(f"merge_to: {self.merge_to}")

    def notify_graph(self, prop: ConsistencyProp, gm: torch.fx.GraphModule):
        for node in gm.graph.nodes:
            cond = prop.env.get(node.name, None)
            if (
                hasattr(cond, "shape")
                and isinstance(cond.shape, (list, tuple))
                and len(cond.shape) > 0
            ):
                s = str(cond.shape[0])
            else:
                s = str(cond)
            if s not in ["True", "False"]:
                # 按出现符号的顺序，命名为dGraph{i}
                if not hasattr(self, "_symbol_order"):
                    self._symbol_order = {}
                    self._symbol_count = 0
                sym = s
                if sym not in self._symbol_order:
                    name = f"dGraph{self._symbol_count}"
                    self._symbol_order[sym] = name
                    self._symbol_count += 1
                key = sym
                if self.tmp_counts.get(key) != None:
                    self.tmp_counts[key] += 1
                else:
                    self.tmp_counts[key] = 1
            else:
                key = s
            node.tag = key
            node.merge_to = "None"
        for key in self.tmp_counts.keys():
            print(f"{key=} {self.tmp_counts[key]=}")

        self.fuse_graph_topo_sort(gm)
        print(f"merge_to: {self.merge_to}")
        for node in reversed(gm.graph.nodes):
            if node.name == "_log_api_usage_once":
                continue
            if self.tmp_counts.get(node.tag, 0) < self.threshold:
                if self.merge_to.get(node.tag, None) is not None:
                    old_tag = node.tag
                    new_tag = self.merge_to[self.merge_to[old_tag]]
                    node.tag = new_tag
                    self.merge_to[old_tag] = new_tag
                    self.tmp_counts[new_tag] += 1
                    self.tmp_counts[old_tag] -= 1
                    if self.tmp_counts[old_tag] == 0:
                        del self.tmp_counts[old_tag]
                else:
                    print(
                        f"trying merge None node {node.name} {node.tag} {node.merge_to}"
                    )
                    node.tag = node.merge_to
                    self.tmp_counts[node.tag] += 1
            for inp_node in node.all_input_nodes:
                print(f"inp_node {inp_node.name} {inp_node.tag} {inp_node.merge_to}")
                if self.tmp_counts.get(inp_node.tag, None) is None:
                    print(
                        f"trying merge None inp_node {inp_node.name} {inp_node.tag} {node.tag}"
                    )
                    inp_node.merge_to = node.tag

        print("After merge:")
        tag_list = []
        for key in self.tmp_counts.keys():
            print(f"{key}({self._symbol_order[key]}): {self.tmp_counts[key]=}")
            tag_list.append(self._symbol_order[key])
        for node in gm.graph.nodes:
            if node.name == "_log_api_usage_once":
                continue
            node.tag = self._symbol_order[node.tag]
        print("succesfully notify graph. tag_list: ", tag_list)
        return gm, tag_list

    def split_build_dGraph(self, gm: torch.fx.GraphModule, tag_list: list[str]):
        # for node in gm.graph.nodes:
        #     if node.op == "placeholder" and node.meta.get("value", None) is None:
        #         print(f"placeholder {node.name} has no value")
        #     print(f"{node.name=} {node.tag=}")
        #     assert node.tag in tag_list, f"node {node.name} has tag {node.tag} not in tag_list {tag_list}"
        # gm._disable_recompile = True
        try:
            split_gm = split_utils.split_by_tags(gm, tags=tag_list)
        except SyntaxError as e:
            print(f"SyntaxError during split: {e}")
            # 尝试打印生成的代码
            import traceback

            traceback.print_exc()
            raise
        split_gm.to_folder(self.name)
        # for tag in tag_list:
        #     subgraph = split_gm.__getattr__(tag)
        #     subgraph.graph.print_tabular()
        #     subgraph.to_folder(f"./test_{tag}_transformed")
