import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import abstractmethod

class BaseEncoder(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    @abstractmethod
    def forward(self, x: torch.Tensor):
        raise NotImplementedError


class ConstantEncoderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, T):
        return torch.stack([x for _ in range(T)])

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.mean(grad_output, dim=0)
        return grad_x, None


class ConstantEncoder(BaseEncoder):
    def __init__(self, T):
        super().__init__(T)

    def forward(self, x):
        return ConstantEncoderFunction.apply(x, self.T)


class PoissonEncoderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, T):
        return torch.stack([torch.rand_like(x).le(x).to(x).float() for _ in range(T)])

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.mean(grad_output, dim=0)
        return grad_x, None

class PoissonEncoder(BaseEncoder):
    def __init__(self, T):
        super().__init__(T)

    def forward(self, x: torch.Tensor):
        return PoissonEncoderFunction.apply(x, self.T)