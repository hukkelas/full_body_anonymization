from typing import Dict
import torch
import torch.nn as nn
from fba import utils

class Sequential(nn.Sequential) :

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for module in self:
            x = module(x)
        return x

class Module(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def extra_repr(self):
        num_params = utils.num_parameters(self) / 10**6
        return f"Num params: {num_params:.3f}M"
