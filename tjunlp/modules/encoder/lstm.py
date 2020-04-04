"""
a
"""

import math
import warnings
import numbers
from typing import List, Tuple, Optional

import torch
from torch.nn import Module, Parameter, ParameterList, Embedding, init, _VF
from torch.nn.modules.rnn import apply_permutation, LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class LstmEncoder(Module):
    """
    封装LSTM
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 200,
                 num_layers: int = 3,
                 bias: bool = True,
                 batch_first: bool = True,
                 dropout: float = 0,
                 bidirectional: bool = False,
                 **kwargs):
        super().__init__()
        self.lstm = LSTM(input_size, hidden_size, num_layers, bias, batch_first,
                         dropout, bidirectional)
        self.output_dim = hidden_size * 2 if bidirectional else hidden_size

    def forward(self, inputs, seq_lens=None, hx=None, **kwargs):  # pylint:disable=arguments-differ
        inputs = pack_padded_sequence(inputs, seq_lens, batch_first=self.lstm.batch_first)
        feat, _ = self.lstm(inputs, hx=hx)  # -> [N,L,C]
        feat, _ = pad_packed_sequence(feat, batch_first=self.lstm.batch_firs)
        return feat


class PGLSTM(Module):
    """
    PGN LSTM, modified from `RNNbase` and `LSTM` in `torch.nn.modules.rnn`.
    """
    def __init__(self, num_domains: int, domain_dim: int,
                 input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = True,
                 dropout: float = 0., bidirectional: bool = False,
                 group_dims: List[int] = None):
        """
        You can assign dims for all groups of all directions of all layers, or
        laryer shared, layer and direction share.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        gate_size = 4 * hidden_size

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        # the language embedding in the paper. I(M)
        self.domain_embedding = Embedding(num_domains, domain_dim)

        if group_dims is not None:
            if len(group_dims) == 1:
                group_dims = [[group_dims * 4] * num_directions] * num_layers
            elif len(group_dims) == num_layers:
                group_dims = [[[d] * 4] * num_directions for d in group_dims]
            # elif len(group_dims) == num_layers * 4:
                # group_dims = [[group_dims[i * 4: i * 4 + 4]] * num_directions for i in range(num_layers)]
            else:
                # we don't recommened assign all dims manually.
                raise ValueError("please check dims of parameter groups!")

            # the P(M', M) that controlled parameter sharings
            self.controller = ParameterList([Parameter(torch.Tensor(
                m_prime, domain_dim)) for m_prime in torch.tensor(
                    group_dims).view(-1)])
        else:
            group_dims = [[[domain_dim] * 4] * num_directions] * num_layers
            self.register_parameter('controller', None)

        self._flat_weights_names = []
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                m1, m2, m3, m4 = group_dims[layer][direction]

                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size, m1))
                w_hh = Parameter(torch.Tensor(gate_size, hidden_size, m2))
                b_ih = Parameter(torch.Tensor(gate_size, m3))
                # Second bias vector included for CuDNN compatibility. Only one
                # bias vector is needed in standard definition.
                b_hh = Parameter(torch.Tensor(gate_size, m4))
                layer_params = (w_ih, w_hh, b_ih, b_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._flat_weights_names.extend(param_names)
                self._all_weights.append(param_names)

        # the W(Pdec, M or M'), we haven't change the name.
        self._flat_weights = [getattr(self, weight) for weight in self._flat_weights_names]
        self.reset_parameters()

    def check_hidden_size(self, hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
        # type: (torch.Tensor, Tuple[int, int, int], str) -> None
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

    def check_forward_args(self, input, hidden, batch_sizes):
        # type: (torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]) -> None
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden[0], expected_hidden_size,
                               'Expected hidden[0] size {}, got {}')
        self.check_hidden_size(hidden[1], expected_hidden_size,
                               'Expected hidden[1] size {}, got {}')

    def check_input(self, input, batch_sizes):
        # type: (torch.Tensor, Optional[torch.Tensor]) -> None
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

    def get_expected_hidden_size(self, input, batch_sizes):
        # type: (torch.Tensor, Optional[torch.Tensor]) -> Tuple[int, int, int]
        if batch_sizes is not None:
            mini_batch = batch_sizes[0]
            mini_batch = int(mini_batch)
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        return expected_hidden_size

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def permute_hidden(self, hx, permutation):
        # type: (Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]
        if permutation is None:
            return hx
        return apply_permutation(hx[0], permutation), apply_permutation(hx[1], permutation)

    def forward(self, input: torch.Tensor, hx=None, domain_id=-1):  # noqa: F811
        """ one domain one time.
        """
        if 0 <= domain_id < self.domain_embedding.num_embeddings:
            domain_emb = self.domain_embedding(torch.tensor(domain_id, device=input.device))
        else:
            raise ValueError(f"invalid domain id <{domain_id}>")

        # generate weights using domain embedding
        if self.controller is None:
            flat_weights = [w.matmul(domain_emb).squeeze(-1) for w in self._flat_weights]
        else:
            flat_weights = [w.matmul(p).matmul(domain_emb).squeeze(-1) for p, w in zip(
                self.controller, self._flat_weights)]

        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = _VF.lstm(input, hx, flat_weights, self.bias, self.num_layers,
                              self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.lstm(input, batch_sizes, hx, flat_weights, self.bias,
                              self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)
