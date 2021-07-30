from abc import abstractmethod
from typing import Callable
import torch
import torch.nn as nn
from spikingjelly_lite.clock_driven import surrogate
import math
import torch.nn.functional as F


class BaseNeuron(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0, rand_init=False, surrogate_function=surrogate.Sigmoid()):
        super().__init__()
        self.v_threshold = v_threshold # nn.Parameter(torch.tensor([v_threshold]))
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function
        self.rand_init = rand_init

    def generate_init_v(self, dv):
        if not self.rand_init:
            self.v_init = 0
        else:
            self.v_init = torch.rand_like(dv).to(dv)

    @abstractmethod
    def neuronal_charge(self, dv: torch.Tensor, v):
        raise NotImplementedError

    def _neuronal_charge(self, input: torch.Tensor, t):
        # t = 0, 1, 2, ..., T-1
        if t == 0:
            self.h[t] = self.neuronal_charge(input[t], self.v_init)
        else:
            self.h[t] = self.neuronal_charge(input[t], self.v[t-1])
    
    def _neuronal_fire(self, t):
        # t = 0, 1, 2, ..., T-1
        self.s[t] = self.surrogate_function(self.h[t] - self.v_threshold)
        # print(self.s[t])

    def _neuronal_reset(self, t):
        # t = 0, 1, 2, ..., T-1
        if self.v_reset is None:
            self.v[t] = self.h[t] - self.s[t] * self.v_threshold
        else:
            self.v[t] = (1 - self.s[t]) * self.h[t] #+ self.s[t] * self.v_reset


    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}'


    def forward(self, input: torch.Tensor):
        self.h = {} # 刚刚充电后电位
        self.s = {} # 脉冲
        self.v = {} # 脉冲发放后残留的电位
        T = input.shape[0]

        self.generate_init_v(input[0])
        for t in range(T):
            self._neuronal_charge(input, t)
            self._neuronal_fire(t)
            self._neuronal_reset(t)
        return torch.stack(tuple(self.s.values()))


class IFNeuron(BaseNeuron):
    def __init__(self, v_threshold=1.0, v_reset=0.0, rand_init=False, surrogate_function=surrogate.Sigmoid()):
        super().__init__(v_threshold, v_reset, rand_init, surrogate_function)

    def neuronal_charge(self, dv: torch.Tensor, v):
        h = v + dv
        return h


class LIFNeuron(BaseNeuron):
    def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0, rand_init=False, surrogate_function=surrogate.Sigmoid()):
        super().__init__(v_threshold, v_reset, rand_init, surrogate_function)
        self.tau = tau

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, tau={self.tau}'

    def neuronal_charge(self, dv: torch.Tensor, v):
        if self.v_reset is None:
            h = v + (dv - v) / self.tau
        else:
            h = v + (dv - (v - self.v_reset)) / self.tau
        return h
