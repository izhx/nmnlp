from typing import Tuple, Callable
import math

import torch
import torch.nn as nn


class NonLinear(nn.Module):
    """
    A linear with activation.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 activation: Callable = nn.GELU()):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        return self.activation(x)


class Biaffine(nn.Module):
    """
    A biaffine module.
    Input: tensors of sizes (b x n1 x d1) and (b x n2 x d2)
    Output: tensor of size (b x n1 x n2 x O)
    """

    def __init__(self, in1_features: int, in2_features: int, out_features: int,
                 bias: Tuple[bool] = (False, False)):
        super().__init__()
        self.bias = bias
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in1_features + int(bias[0]), (
            in2_features + int(bias[1])) * out_features))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(0))
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        (b, n1, _), (_, n2, d2), o = input1.shape, input2.shape, self.out_features

        if self.bias[0]:
            input1 = torch.cat([input1, input1.new_ones((b, n1, 1))], -1)
        if self.bias[1]:
            input2 = torch.cat([input2, input2.new_ones((b, n2, 1))], -1)
            d2 += 1
        # (b, n1, d1) * (d1, o*d2) -> (b, n1, o*d2) -> (b, n1*o, d2)
        lin = input1.matmul(self.weight).reshape(b, n1*o, d2)
        # (b, n1*o, d2) * (b, d2, n2) -> (b, n1*o, n2)
        output = lin.bmm(input2.transpose(1, 2))
        # (b, n1*O, n2) -> (b, n1, n2, O)
        output = output.view(b, n1, o, n2).transpose(2, 3)

        return output  # einsum will cause cuda out of memory, fuck
