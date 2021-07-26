import torch
import torch.nn as nn

class NeuronPipe(nn.Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, input: torch.Tensor):
        y_shape = [input.shape[0], input.shape[1]]
        y_seq = self.module(input.flatten(0, 1))
        y_shape.extend(y_seq.shape[1:])
        return y_seq.reshape(y_shape)

class FiringRate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor):
        return torch.mean(input,dim=0)