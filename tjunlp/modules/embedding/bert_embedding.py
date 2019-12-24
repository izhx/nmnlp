"""
Pack bert.
"""
from typing import List
import torch
from transformers import BertModel

BERT_BASE_UNCASED = '/home/xps/Desktop/workspace/model_weight/bert-base-uncased/'
BERT_MULTILINGUAL = '/home/xps/Desktop/workspace/model_weight/bert-multilingual/'

CLS = 101
SEP = 102


class PackedBertEmbedding(torch.nn.Module):
    """
    Packed Bert, remove unnecessary code.
    """

    def __init__(self, name, freeze):
        super().__init__()
        if name == 'bert_multi':
            bert = BertModel.from_pretrained(BERT_MULTILINGUAL)
        else:
            bert = BertModel.from_pretrained(BERT_BASE_UNCASED)

        if freeze == 'all':
            for param in bert.parameters():
                param.requires_grad = False

        self.embeddings = bert.embeddings
        self.encoder = bert.encoder
        self.num_hidden_layers = bert.config.num_hidden_layers
        self.output_dim = bert.config.hidden_size

    def forward(self, input_ids: torch.Tensor,  # pylint:disable=arguments-differ
                seq_lens: List[int],
                head_mask: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        for i, s in enumerate(seq_lens):
            input_ids[i, 0], input_ids[i, s] = CLS, SEP
        if head_mask is None:
            head_mask = [None] * self.num_hidden_layers
        if attention_mask is not None:
            input_shape = input_ids.size()
            if attention_mask.dim() == 3:
                extended_attention_mask = attention_mask[:, None, :, :]
            elif attention_mask.dim() == 2:
                extended_attention_mask = attention_mask[:, None, None, :]
            else:
                raise ValueError(f"Wrong shape for input_ids (shape {input_shape}"
                                 f") or attention_mask (shape {attention_mask.shape})")

            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (
                1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        outputs = self.embeddings(input_ids=input_ids)
        outputs = self.encoder(outputs, extended_attention_mask, head_mask)
        return outputs[0]
