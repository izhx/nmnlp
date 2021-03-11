"""
Adapter with transformers and PyTorch.

Parameter-Efficient Transfer Learning for NLPParameter-Efficient Transfer Learning for NLP
https://arxiv.org/abs/1902.00751
https://github.com/google-research/adapter-bert
"""

from typing import List, Union

import torch
import torch.nn as nn
from torch.nn.functional import embedding_bag, linear

from transformers.models.bert.modeling_bert import BertModel


def set_requires_grad(module: nn.Module, status: bool = False):
    for param in module.parameters():
        param.requires_grad = status


def batch_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """ batched linear forward """
    y = torch.einsum("bth,boh->bto", x, w)
    y = y + b.unsqueeze(1)
    return y


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
                nn.Parameter(torch.zeros(bottleneck_size)),
                nn.Parameter(torch.Tensor(in_features, bottleneck_size)),
                nn.Parameter(torch.zeros(in_features))
            ])
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.params[0], std=1e-3)
        nn.init.normal_(self.params[2], std=1e-3)

    def forward(self, hidden_states: torch.Tensor):
        linear_forward = batch_linear if self.params[0].dim() == 3 else linear
        x = linear_forward(hidden_states, self.params[0], self.params[1])
        x = self.act_fn(x)
        x = linear_forward(x, self.params[2], self.params[3])
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
                 name_or_path_or_model: Union[str, BertModel],
                 adapter_size: int = 128,
                 adapter_num: int = 12,
                 external_param: Union[bool, List[bool]] = False,
                 word_piece: str = 'first',  # 需要保证input ids为第一个
                 # mix_out: bool = False,
                 **kwargs):
        super().__init__()
        if isinstance(name_or_path_or_model, str):
            self.bert = BertModel.from_pretrained(name_or_path_or_model)
        else:
            self.bert = name_or_path_or_model

        set_requires_grad(self.bert, False)

        if isinstance(external_param, bool):
            param_place = [external_param for _ in range(adapter_num)]
        elif isinstance(external_param, list):
            param_place = [False for _ in range(adapter_num)]
            for i, e in enumerate(external_param, 1):
                param_place[-i] = e
        else:
            raise ValueError("wrong type of external_param!")

        self.adapters = nn.ModuleList([nn.ModuleList([
                Adapter(self.bert.config.hidden_size, adapter_size, param_place[i]),
                Adapter(self.bert.config.hidden_size, adapter_size, param_place[i])
            ]) for i in range(adapter_num)
        ])

        # if mix_out:
        #     self.weights = nn.Parameter(torch.ones(adapter_num))
        # else:
        #     self.weights = None

        for i, adapters in enumerate(self.adapters, 1):
            layer = self.bert.encoder.layer[-i]
            layer.output = AdapterBertOutput(layer.output, adapters[0].forward)
            set_requires_grad(layer.output.base.LayerNorm, True)
            layer.attention.output = AdapterBertOutput(layer.attention.output, adapters[1].forward)
            set_requires_grad(layer.attention.output.base.LayerNorm, True)

        self.output_dim = self.bert.config.hidden_size
        self.word_piece = word_piece

    def forward(
        self, input_ids: torch.Tensor, mask: torch.Tensor = None,
        word_pieces=None, lengths=None, **kwargs
    ) -> torch.Tensor:
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        if word_pieces is not None:
            inputs_embeds = self.merge_piece_emb(word_pieces, inputs_embeds)
        attention_mask = None if mask is None else mask.float()

        bert_output = self.bert(attention_mask=attention_mask, inputs_embeds=inputs_embeds)

        bert_representation = self.merge_piece_out(word_pieces, bert_output[0], lengths)
        return bert_representation

    def merge_piece_emb(self, pieces, inputs):
        if self.word_piece == 'embedding':
            offset = torch.tensor([0], dtype=torch.long)
            for (s, w), p in pieces.items():
                inputs[s, w, :] = embedding_bag(
                    p, self.bert.embeddings.word_embeddings.weight,
                    offset.to(p.device))
        return inputs

    def merge_piece_out(self, pieces, output, lengths: torch.LongTensor):
        """
        piece: [ {start: l_span, ...}, {start: l_span, ...}, ..., [len1, ...]]
        """
        if self.word_piece != 'output':
            return output

        representations, pad_len = list(), lengths.max().item()
        for i, (out, ps) in enumerate(zip(output, pieces)):
            origin_len, bert_len = lengths[i].item(), out.size(0)
            j, rep_i = 0, list()
            while j < bert_len:
                out_j = out[j]
                if j in ps:
                    out_j = out[j: j + ps[j]].mean(dim=0)
                    j += ps[j]
                else:
                    j += 1
                rep_i.append(out_j)
                if len(rep_i) == origin_len:
                    break
            while len(rep_i) < pad_len:
                rep_i.append(torch.zeros_like(rep_i[0]))
            representations.append(torch.stack(rep_i))
        representations = torch.stack(representations)
        return representations
