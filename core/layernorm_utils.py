import torch
from torch import nn
from torch.nn import Parameter
import math

class ScaleNorm(nn.Module):
    """ScaleNorm"""
    def __init__(self, input_dim, eps=1e-5):
        super(ScaleNorm, self).__init__()
        """
        scale = sqrt(dim)
        """
        scale = math.sqrt(input_dim * 1.0)
        self.scale = Parameter(torch.tensor(scale), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class FixNorm(nn.Module):
    def __init__(self, input_dim, eps=1e-5):
        super(FixNorm, self).__init__()
        """
        scale = sqrt(dim)
        """
        scale = math.sqrt(input_dim * 1.0)
        self.scale = Parameter(torch.tensor(scale), requires_grad=True)
        self.weight = Parameter(torch.randn(input_dim), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        norm = self.scale / (torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps) * torch.norm(self.weight.data).clamp(min=self.eps))
        return x * self.weight * norm