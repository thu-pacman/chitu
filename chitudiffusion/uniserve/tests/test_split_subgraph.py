import torch
import torchperf
from torch.fx.passes import split_utils


class SimpleModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)
        self.linear3 = torch.nn.Linear(20, 10)

    def forward(self, in1, in2):
        r1 = self.linear1(in1)
        r2 = self.linear2(in2)
        r3 = torch.cat([r1, r2])
        return self.linear3(r3)


val: dict = {}

if __name__ == "__main__":
    model = SimpleModule()
    a = torch.randn(10)
    b = torch.randn(10)
    args = [a, b]
    gm, fx_args = torchperf.torch_dynamo.get_dynamo_graph_modules_and_args(
        model, args, kwargs={}, full_graph=True, dynamic=True
    )
    gm[0].graph.print_tabular()
    original_result = model(a, b)
    # print(f"{len(fx_args[0])=}")
    for node in gm[0].graph.nodes:
        for inp_node in node.all_input_nodes:
            print(f"{inp_node.name=}")
        if (
            "l_1" in node.name
            or "r1" in node.name
            or "r2" in node.name
            or "r3" in node.name
        ):
            node.tag = "a"
        else:
            node.tag = "b"

    # gm[0].graph.print_tabular()
    iter_fx_args = iter(fx_args[0])
    placeholders = [n for n in gm[0].graph.nodes if n.op == "placeholder"]
    for placeholder in placeholders:
        # print(f"{placeholder.name=}{placeholder.meta['example_value']=}")
        example_value = placeholder.meta["example_value"]
        is_plain_faketensor_placeholder = not isinstance(
            example_value, torch.nn.Parameter
        )
        value = next(iter_fx_args)
        # print(f"{value.shape=}, {type(value)=}")
        # Replace placeholder as get_attr node
        # If this placeholder is not plain input (i.e., is a Parameter/buffer),
        # replace it in the graph with a get_attr node.
        if not is_plain_faketensor_placeholder:
            # Save the old placeholder node to remove after replacement
            old_ph = placeholder
            gm[0].register_parameter(placeholder.name, value)
            get_attr_node = gm[0].graph.get_attr(placeholder.name)
            placeholder.replace_all_uses_with(get_attr_node)
            gm[0].graph.erase_node(placeholder)
    gm[0].recompile()
    print(
        "===============================================After recompiling:================================================"
    )
    # gm[0].graph.print_tabular()
    tags = ["a", "b"]
    split_gm = split_utils.split_by_tags(gm[0], tags=tags)
    split_gm.graph.print_tabular()
    # for tag in tags:
    # split_gm.__getattr__(tag).graph.print_tabular()
    # split_gm.a.to_folder("./a_transformed")
    # split_gm.b.to_folder("./b_transformed")
    output1 = split_gm.a(a, b)
    if isinstance(split_gm.a.graph.output_node().args[0], tuple):
        for i in range(len(output1)):
            print(
                f"{type(split_gm.a.graph.output_node().args[0][i])=}, {split_gm.a.graph.output_node().args[0][i]} reuslt:{output1[i]}"
            )
            val[split_gm.a.graph.output_node().args[0][i].name] = output1[i]
    else:
        print(
            f"{type(split_gm.a.graph.output_node().args[0])=}, {split_gm.a.graph.output_node().args[0]} reuslt:{output1}"
        )
        val[split_gm.a.graph.output_node().args[0].name] = output1
    args = []
    # 遍历 placeholder，根据 val 创建 args，传入计算得到结果，并且和拆分前计算图结果对比
    new_args = []
    print(val)
    for placeholder in split_gm.b.graph.nodes:
        if placeholder.op == "placeholder":
            print(f"{placeholder.name=}")
            key = placeholder.name
            if key in val:
                new_args.append(val[key])
            elif key in locals():
                new_args.append(locals()[key])
            elif key in globals():
                new_args.append(globals()[key])
            else:
                # fallback: use None (should not happen, just as a guarantee)
                new_args.append(None)
    print(f"{new_args=}")
    output2 = split_gm.b(*new_args)

    # output2 可能是元组或单个 tensor
    if isinstance(output2, tuple):
        for i, o in enumerate(output2):
            print(f"{split_gm.b.graph.output_node().args[0][i]} reuslt:{o}")
            # 对比和 original_result
            if hasattr(original_result, "__getitem__"):
                assert torch.allclose(o, original_result[i]), f"Mismatch at output {i}"
            else:
                assert torch.allclose(o, original_result), "Mismatch in single output"
    else:
        print(f"{split_gm.b.graph.output_node().args[0]} reuslt:{output2}")
        assert torch.allclose(output2, original_result), "Mismatch in single output"

    print("Consistency check passed: split result matches original result")
