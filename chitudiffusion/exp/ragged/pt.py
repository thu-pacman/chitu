import torch

torch.set_default_device("cuda")
a = torch.nested.nested_tensor([torch.randn((1, 2, 5)), torch.randn((1, 3, 5))])

# print(f'{torch.matmul(a, a.T)}')
# print(f'{torch.bmm(a, a.T)}')
print(f"{a}")


@torch.compile()
def foo():
    return torch.nn.functional.softmax(a, dim=-1)


print(f"{torch.nn.functional.softmax(a, dim = -1)}")
print(f"{torch.nn.functional.softmax(a, dim = -2)}")
print(f"{torch.nn.functional.softmax(a, dim = -3)}")
# print(foo())
# print(f'{torch.nn.functional.softmax(a, dim = 0)}')
# print(f'{torch.nn.functional.softmax(a, dim = -4)}') # error
# print(f'{a@a.T=}')
#
