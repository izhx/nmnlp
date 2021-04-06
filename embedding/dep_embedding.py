"""
DepSAWR embedding
"""
from typing import List
from itertools import chain

import os
import json
import codecs

import torch
from torch.nn import Module, Embedding, Dropout
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..modules.augmented_lstm import BiAugmentedLstm
from ..modules.dropout import WordDropout
from ..modules.linear import NonLinear
from ..common.util import printf

NUM_UPOSTAG = 18  # 17 + pad


class DepSAWR(Module):
    """

    """
    def __init__(self,
                 word_dim: int,
                 upostag: List,
                 pos_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 arc_dim: int,
                 rel_dim: int,
                 dropout: float = 0.5,
                 pretrained_path: str = None,
                 requires_grad: bool = False):
        super().__init__()
        self.word_dim = word_dim
        self.arc_dim = arc_dim
        self.rel_dim = rel_dim
        feature_dim = word_dim + pos_dim
        self.pos_embedding = Embedding(len(upostag), pos_dim, padding_idx=0)
        self.lstm = BiAugmentedLstm(feature_dim,
                                    hidden_size,
                                    num_layers,
                                    recurrent_dropout_probability=dropout,
                                    bidirectional=True)
        self.arc_mlp = NonLinear(hidden_size * 2, arc_dim + rel_dim)
        self.rel_mlp = NonLinear(hidden_size * 2, arc_dim + rel_dim)
        self.word_dropout = WordDropout(dropout)
        self.dropout = Dropout(dropout)

        self.dims = [arc_dim + rel_dim] * 2 + [hidden_size * 2] * 3
        self.upostag = upostag

        if pretrained_path:
            self.load_state_dict(torch.load(pretrained_path))
            for param in self.parameters():
                param.requires_grad = requires_grad

    def forward(self,
                word_embedding: torch.FloatTensor,
                pos_ids: torch.LongTensor,
                lengths: torch.LongTensor,
                modules: torch.nn.ModuleList = None,
                **kwargs):
        pos_embedding = self.pos_embedding(pos_ids)
        embedding = torch.cat((word_embedding, pos_embedding), dim=-1)
        embedding = self.word_dropout(embedding)

        inputs = pack_padded_sequence(embedding, lengths, True, True)
        output_sequence, _, layer_outputs = self.lstm(inputs)
        outputs, _ = pad_packed_sequence(output_sequence, True)

        outputs = self.dropout(outputs)
        outputs = self.arc_mlp(outputs), self.rel_mlp(outputs)

        if modules:
            outputs = [m(o) for m, o in zip(modules, chain(outputs, layer_outputs))]

        return outputs

    @classmethod
    def from_pretrained(cls, name_or_path, requires_grad: bool = False):
        with codecs.open(f"{name_or_path}/config.json",
                         mode='r',
                         encoding='UTF-8') as file:
            config = json.load(file)
        config['requires_grad'] = requires_grad
        printf(f"Model loaded from <{name_or_path}>")

        return cls(pretrained_path=f"{name_or_path}/model.bin", **config)

    def save(self, dir_path):
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        config = {
            'word_dim': self.word_dim,
            'pos_dim': self.pos_embedding.embedding_dim,
            'hidden_size': self.lstm.hidden_size,
            'num_layers': self.lstm.num_layers,
            'arc_dim': self.arc_dim,
            'rel_dim': self.rel_dim,
            'dropout': self.dropout.p,
            'upostag': self.upostag
        }

        with codecs.open(f"{dir_path}/config.json", mode='w',
                         encoding='UTF-8') as file:
            json.dump(config, file)

        torch.save(self.state_dict(), f"{dir_path}/model.bin")
        printf(f"Model saved at <{dir_path}>")
