"""
a
"""

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LstmEncoder(torch.nn.Module):
    """
    基本的双向LSTM
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 200,
                 num_layers: int = 3,
                 bias: bool = True,
                 batch_first: bool = True,
                 dropout: float = 0,
                 bidirectional: bool = False):
        super().__init__()
        self.encoder = torch.nn.LSTM(input_size, hidden_size, num_layers,
                                     bias, batch_first, dropout, bidirectional)
        self.output_dim = hidden_size * 2 if bidirectional else hidden_size

    def forward(self, inputs, seq_lens=None, hx=None):  # pylint:disable=arguments-differ
        inputs = pack_padded_sequence(inputs, seq_lens, batch_first=True)
        feat, _ = self.encoder(inputs, hx=hx)  # -> [N,L,C]
        feat, _ = pad_packed_sequence(feat, batch_first=True)
        return feat
