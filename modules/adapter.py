"""
Adapter with transformers and PyTorch.

Parameter-Efficient Transfer Learning for NLPParameter-Efficient Transfer Learning for NLP
https://arxiv.org/abs/1902.00751
https://github.com/google-research/adapter-bert
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import embedding_bag, linear

from transformers.modeling_bert import BertModel


class Adapter(nn.Module):
    def __init__(self, in_features, bottleneck_size, external_param=False):
        super().__init__()
        self.in_features = in_features
        self.bottleneck_size = bottleneck_size
        self.act_fn = nn.GELU()
        if external_param:
            self.params = [None, None, None, None]
        else:
            self.params = nn.ParameterList([
                nn.Parameter(torch.Tensor(bottleneck_size, in_features)),
                nn.Parameter(torch.Tensor(bottleneck_size)),
                nn.Parameter(torch.Tensor(in_features, bottleneck_size)),
                nn.Parameter(torch.Tensor(in_features))
            ])
            self.reset_parameters()

    def reset_parameters(self):
        def init_linear(w, b):
            init.kaiming_uniform_(w, a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(b, -bound, bound)

        init_linear(self.params[0], self.params[1])
        init_linear(self.params[2], self.params[3])

    def forward(self, hidden_states: torch.Tensor):
        x = linear(hidden_states, self.params[0], self.params[1])
        x = self.act_fn(x)
        x = linear(x, self.params[2], self.params[3])
        x = x + hidden_states
        return x


class AdapterBertOutput(nn.Module):
    """
    替代BertOutput和BertSelfOutput
    """
    def __init__(self, base, adapter_forward):
        super().__init__()
        self.base = base
        self.adapter_forward = adapter_forward

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.base.dense(hidden_states)
        hidden_states = self.base.dropout(hidden_states)
        hidden_states = self.adapter_forward(hidden_states)
        hidden_states = self.base.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AdapterBertModel(nn.Module):
    def __init__(self,
                 name_or_path: str,
                 freeze: str = 'all',
                 word_piece: str = 'first',  # 需要保证input ids为第一个
                 adapter_size: int = 128,
                 external_param: bool = False,
                 **kwargs):
        super().__init__()
        self.bert = BertModel.from_pretrained(name_or_path)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.adapters = nn.ModuleList([
            Adapter(self.bert.config.hidden_size, adapter_size, external_param)
            for _ in range(self.bert.config.num_hidden_layers * 2)
        ])

        for i, layer in enumerate(self.bert.encoder.layer):
            layer.output = AdapterBertOutput(
                layer.output, self.adapters[i * 2].forward)
            layer.attention.output = AdapterBertOutput(
                layer.attention.output, self.adapters[i * 2 + 1].forward)

        self.output_dim = self.bert.config.hidden_size

        if word_piece == 'first':
            self.word_piece = None
        else:  # mean of pieces
            offset = torch.tensor([0], dtype=torch.long)
            # self.register_buffer("offset", offset)
            self.word_piece = lambda x: embedding_bag(
                x, self.bert.embeddings.word_embeddings.weight, offset.to(x.device))

    def forward(self,  # pylint:disable=arguments-differ
                input_ids: torch.Tensor,
                word_pieces: Dict[Tuple[int], torch.LongTensor] = None,
                mask: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        if self.word_piece is not None and word_pieces is not None:
            for (s, w), pieces in word_pieces.items():
                inputs_embeds[s, w, :] = self.word_piece(pieces)

        attention_mask = None if mask is None else mask.float()
        bert_output = self.bert(attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        return bert_output[0]
