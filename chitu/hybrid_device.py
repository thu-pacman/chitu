import torch


class CPUParameter(torch.nn.Parameter):
    """
    A torch.nn.Parameter that always stay on CPU.
    """

    pass
