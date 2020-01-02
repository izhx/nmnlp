import torch


class WordDropout(torch.nn.Dropout):
    def forward(self, x: torch.Tensor, replacement=None):
        if not self.training or self.p == 0:
            return x

        masksize = *x.shape[:-1], 1
        dropmask = torch.rand(*masksize, device=x.device) < self.p

        if replacement is None:
            x = x.masked_fill(dropmask, 0)
        else:
            x = x.masked_fill(dropmask, replacement)

        return x
