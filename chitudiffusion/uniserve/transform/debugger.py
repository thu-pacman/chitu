from typing import Any, Dict, Tuple
import torch
import torch.fx
from torch.fx import Node, Proxy


class Debugger(torch.fx.Interpreter):
    """Save intermediate results in self.env"""

    def __init__(self, mod: torch.nn.Module):
        super().__init__(mod, garbage_collect_values=False)

    def run_node(self, n: Node) -> Any:
        ret = super().run_node(n)
        # shape = ret.shape if torch.is_tensor(ret) else ret
        # print(n.name, shape)
        # if (
        #     n.op == "call_module"
        #     and (n.name.startswith("u_") or n.name.startswith("l_"))
        #     and "resnets" in n.name
        # ):
        #     print(n.name, ret)
        return ret
