"""
Pack bert.
"""
from typing import List, Union, Dict, Tuple

import torch
from transformers import BertModel, BertTokenizer

_BERT_BASE_UNCASED = '/home/xps/Desktop/workspace/model_weight/bert-base-uncased/'
_BERT_MULTILINGUAL = '/home/xps/Desktop/workspace/model_weight/bert-multilingual/'

CLS = 101
SEP = 102

HIDDEN_STATES_INDEX = 2


def get_pretrained(name_or_path: str, the_class: Union[BertModel, BertTokenizer]):
    """
    return pretrained bert model or tokenizer.
    """
    if name_or_path == 'bert_multi':
        return the_class.from_pretrained(_BERT_MULTILINGUAL)
    if name_or_path == 'bert':
        return the_class.from_pretrained(_BERT_BASE_UNCASED)
    return the_class.from_pretrained(name_or_path)


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
        self.bert = get_pretrained(name_or_path, BertModel)
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
