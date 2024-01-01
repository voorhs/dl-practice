from torch import nn
from functions import ReLUFunction, LinearFunction, LogSoftmaxFunction, DropoutFunction, CrossEntropyFunction, SiLUFunction, HardswishFunction
import math
import torch


class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

        self.fn = ReLUFunction.apply

    def forward(self, input):
        return self.fn(input)
    


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()
        
        self.fn = LinearFunction.apply

    def reset_parameters(self):
        std = math.sqrt(2 / self.in_features)
        nn.init.normal_(self.weight, mean=0, std=std)
        nn.init.normal_(self.bias, mean=0, std=std)

    def forward(self, inp):
        return self.fn(inp, self.weight, self.bias)


class LogSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fn = LogSoftmaxFunction.apply

    def forward(self, input):
        return self.fn(input)
    

class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        
        self.p = p
        self.fn = DropoutFunction.apply

    def forward(self, input):
        return self.fn(input, self.p, self.training)


class CrossEntropy(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        
        self.reduction = reduction
        self.fn = CrossEntropyFunction.apply

    def forward(self, activations, target):
        return self.fn(activations, target, self.reduction)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

        self.fn = SiLUFunction.apply

    def forward(self, input):
        return self.fn(input)


class Hardswish(nn.Module):
    def __init__(self):
        super().__init__()

        self.fn = HardswishFunction.apply

    def forward(self, input):
        return self.fn(input)