"""
Pack bert.
"""
from typing import Dict, Tuple, Any

import torch
from torch.nn.functional import embedding_bag
from torch.nn import Module, ModuleList
from transformers import AutoConfig, AutoModel, BertModel

from ..linear import NonLinear
from ..fusion import ScalarMixWithDropout

CLS = 101
SEP = 102


class BertEmbedding(Module):
    """
    bert embedding fix word pieces.
    """

    def __init__(self,
                 name_or_path: str,
                 freeze: str = 'all',
                 layer_num: int = 1,  # 从最后一层开始，往前取几层
                 transform_dim: int = 0,  # 每层降维到多少
                 scalar_mix: Dict[str, Any] = None,
                 word_piece: str = 'first',  # 需要保证input ids为第一个
                 **kwargs):
        super().__init__()
        config = AutoConfig.from_pretrained(name_or_path)
        config.output_hidden_states = True
        self.bert = AutoModel.from_config(config)

        self.layer_num = layer_num
        self.output_dim = config.hidden_size
        self.index = 2 if isinstance(self.bert, BertModel) else 1

        if freeze == 'all':
            for param in self.bert.parameters():
                param.requires_grad = False

        if transform_dim > 0:
            self.word_transform = ModuleList([NonLinear(
                self.output_dim, transform_dim) for _ in range(self.layer_num)])
            self.output_dim = transform_dim
        else:
            self.word_transform = None

        if word_piece == 'first':
            self.word_piece = None
        else:  # mean of pieces
            offset = torch.tensor([0], dtype=torch.long)
            # self.register_buffer("offset", offset)
            self.word_piece = lambda x: embedding_bag(
                x, self.bert.embeddings.word_embeddings.weight, offset.to(x.device))

        self.scalar_mix = None if scalar_mix is None else ScalarMixWithDropout(
            layer_num, **scalar_mix)

    def forward(self,  # pylint:disable=arguments-differ
                input_ids: torch.Tensor,
                word_pieces: Dict[Tuple[int], torch.LongTensor] = None,
                attention_mask: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        if self.word_piece is not None and word_pieces is not None:
            for (s, w), pieces in word_pieces.items():
                inputs_embeds[s, w, :] = self.word_piece(pieces)

        hidden_states = self.bert(attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        hidden_states = hidden_states[self.index][-self.layer_num:]

        if self.word_transform is not None:
            for i in range(self.layer_num):
                hidden_states[i] = self.word_transform[i](hidden_states[i])

        if self.scalar_mix is not None:
            hidden_states = self.scalar_mix(hidden_states, attention_mask)

        return hidden_states
