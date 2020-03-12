"""
Pack bert.
"""
from typing import List, Union, Dict, Tuple

import torch
from transformers import BertModel, BertTokenizer


CLS = 101
SEP = 102

HIDDEN_STATES_INDEX = 2


class BertEmbedding(torch.nn.Module):
    """
    bert embedding fix word pieces.
    """

    def __init__(self,
                 name_or_path: str,
                 freeze: str = 'all',
                 layer_num: int = 1,  # 从最后一层开始，往前取几层
                 **kwargs):
        super().__init__()
        self.bert = BertModel.from_pretrained(name_or_path)
        self.bert.encoder.output_hidden_states = True
        if freeze == 'all':
            for param in self.bert.parameters():
                param.requires_grad = False
        self.output_dim = self.bert.config.hidden_size
        self.layer_num = layer_num
        self.piece_transorm = lambda x: sum(x) / len(x)

    def forward(self,  # pylint:disable=arguments-differ
                input_ids: torch.Tensor,
                word_pieces: Dict[Tuple[int], torch.LongTensor] = None,
                token_type_ids: torch.Tensor = None,
                position_ids: torch.Tensor = None,
                head_mask: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
        if word_pieces is not None:
            for (s, w), piece in word_pieces.items():
                piece_embeds = self.bert.embeddings.word_embeddings(piece)
                inputs_embeds[s, w, :] = self.piece_transorm(piece_embeds)

        hidden_states = self.bert(
            attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, head_mask=head_mask,
            inputs_embeds=inputs_embeds)[HIDDEN_STATES_INDEX]
        return hidden_states[-self.layer_num:]
