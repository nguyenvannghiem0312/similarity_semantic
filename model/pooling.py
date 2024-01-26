import torch
from torch import Tensor
from torch import nn


class Pooling(nn.Module):
    def __init__(self,
                 method = "mean"):
        super(Pooling, self).__init__()
        assert method in ["mean", "max", "cls"]
        self.method = method

    def forward(self, inputs_tensor):
        if self.method == "cls":
            return 
        if self.method == "mean":
            return torch.mean(inputs_tensor, dim=1)
        if self.method == "max":
            return torch.max(inputs_tensor, dim=1)
