""" Some drop out class.
"""

import torch


class WordDropout(torch.nn.Dropout):
    """ mask whole -1 dim array. """

    def forward(self, x: torch.Tensor):  # pylint:disable=arguments-differ
        if not self.training or self.p == 0:
            return x
        mask = torch.rand(*x.shape[:-1], 1, device=x.device) < self.p
        return x.masked_fill_(mask, 0) if self.inplace else x.masked_fill(mask, 0)


class LockedDropout(torch.nn.Dropout):
    """ batch dim share mask. """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__(p, inplace)
        self.q = 1 - p  # pylint:disable=invalid-name

    def forward(self, x: torch.Tensor):  # pylint:disable=arguments-differ
        if not self.training or self.p == 0:
            return x
        mask = torch.rand(1, *x.shape[1:], device=x.device).bernoulli_(
            p=self.q).div_(self.q).expand_as(x)
        return x.mul_(mask) if self.inplace else x.mul(mask)
