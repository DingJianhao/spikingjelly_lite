import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import abstractmethod
from spikingjelly_lite.clock_driven import surrogate

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


class Vit(BaseEncoder):
    def __init__(self, T, surrogate_function=surrogate.Sigmoid()):
        super().__init__(T)
        self.v_reset = None
        self.surrogate_function = surrogate_function
        self.rand_init = True
        self.T = T
        self.v_threshold = 1.0 #nn.Parameter(torch.tensor([v_threshold]))

        self.salt_noise_prop = 0.004 / 10 # 有待优化
        self.salt_noise_intensity = 1

        self.spatial_gaussian_noise_intensity = 0.02
        self.threshold_noise_intensity = 0.015
        self.mul_noise_intensity = 0.02

    def generate_init_v(self, dv):
        self.v_init = torch.rand_like(dv).to(dv)

    def _neuronal_charge(self, input: torch.Tensor, t):
        # t = 0, 1, 2, ..., T-1
        i = input[t] + self.spatial_gaussian_noise_intensity * self.spatial_gaussian_noise
        if t == 0:
            self.h[t] = self.neuronal_charge(i, self.v_init)
        else:
            self.h[t] = self.neuronal_charge(i, self.v[t-1])

    def _neuronal_fire(self, t):
        # t = 0, 1, 2, ..., T-1
        self.s[t] = self.surrogate_function(self.h[t] - self.v_threshold - self.threshold_noise)

    def _neuronal_reset(self, t):
        # t = 0, 1, 2, ..., T-1
        self.v[t] = self.h[t] - self.s[t] * (self.v_threshold + self.threshold_noise)

    def neuronal_charge(self, dv: torch.Tensor, v):
        # dv : B,1,h,w
        nonlinear = self.nonlinear_intensity(dv) # IF 非线性响应
        nonlinear_max = torch.max(nonlinear)
        nonlinear = nonlinear * (1 - self.salt_noise_mask) + \
                    self.salt_noise_mask * self.salt_noise_intensity * nonlinear_max # 传输信道带来的椒盐噪声 (可以优化)
        h = v + nonlinear * self.mul_noise # 器件失配噪声（乘性）
        h += 0.01 * nonlinear_max * torch.randn_like(nonlinear).to(nonlinear) # 加性噪声(暗电流)
        return h

    def nonlinear_intensity(self, x, a0=0.01881594, a1=0.31948065, a2=-0.01242004):
        return a0 * torch.exp(x / a1) + a2
        # 0.131892 * x
        # 0.00069294 0.13804802 0.02963751
        # 0.00141151 0.15306994 0.02891575
        # 0.00153569 0.154876   0.02111664
        # 0.01881594 0.31948065 -0.01242004

    def forward(self, input: torch.Tensor):
        self.h = {} # 刚刚充电后电位
        self.s = {} # 脉冲
        self.v = {} # 脉冲发放后残留的电位
        input = ConstantEncoderFunction.apply(input, self.T)
        merge = (input[:, :, 0] + input[:, :, 1] + input[:, :, 2]) / 3
        merge = torch.unsqueeze(merge, dim=2)

        self.generate_init_v(merge[0])
        self.salt_noise_mask = (torch.rand_like(merge[0]).to(merge[0]) < self.salt_noise_prop).float()
        self.mul_noise = (1 + torch.randn_like(merge[0]).to(merge[0]) * self.mul_noise_intensity)
        self.threshold_noise = torch.randn_like(merge[0]).to(merge[0]) * self.threshold_noise_intensity
        #(torch.rand_like(merge[0]).to(merge[0]) - 0.5) * 2
        self.spatial_gaussian_noise = torch.randn_like(merge[0]).to(merge[0])
        for t in range(self.T):
            self._neuronal_charge(merge, t)
            self._neuronal_fire(t)
            self._neuronal_reset(t)
        return torch.stack(tuple(self.s.values()))