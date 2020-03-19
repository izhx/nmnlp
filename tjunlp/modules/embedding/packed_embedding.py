"""
Packed embedding with encoder.
"""

from typing import Dict, Any

import torch

from tjunlp.core.vocabulary import Vocabulary, PRETRAIN_POSTFIX
from .embedding import Embedding
from ..encoder import build_encoder
from ..fusion import Fusion


class DeepEmbedding(torch.nn.Module):
    """
    Construct the embedding may have encoder.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 encoder: Dict[str, Any] = None,
                 padding_idx: int = 0,
                 max_norm: float = None,
                 norm_type: float = 2,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 _weight: torch.Tensor = None):
        super(DeepEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
            scale_grad_by_freq, sparse, _weight)
        if encoder is None:
            self.encoder = None
            self.output_dim = embedding_dim
        else:
            self.encoder = build_encoder(embedding_dim, **encoder)
            self.output_dim = self.encoder.output_dim

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:  # pylint:disable=arguments-differ
        input_ids = self.embedding(input_ids)
        if self.encoder is not None:
            input_ids = self.encoder(input_ids, **kwargs)
        return input_ids


class PreTrainEmbedding(torch.nn.Module):
    """
    Embedding module for the insufficient pretrain vector.
    """

    def __init__(self, vocab: Vocabulary,
                 pretrained_file: str,
                 vocab_namespace: str,
                 embedding_dim: int,
                 fusion_method: str = 'cat',
                 padding_index: int = 0,
                 trainable: bool = False,  # whether free the pretrained
                 max_norm: float = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 **kwargs: Any):
        super().__init__()
        self.key_pretrained = vocab_namespace + PRETRAIN_POSTFIX
        self.output_dim = embedding_dim * (2 if fusion_method == 'cat' else 1)
        self.trainable = Embedding(
            len(vocab[vocab_namespace]), embedding_dim, padding_index=padding_index,
            max_norm=max_norm, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)
        embedding = Embedding.from_pretrain(
            vocab, pretrained_file, vocab_namespace + PRETRAIN_POSTFIX,
            padding_index, trainable, max_norm, norm_type, scale_grad_by_freq, sparse)
        if embedding.output_dim != embedding_dim:
            self.pretrained = torch.nn.ModuleList([embedding, torch.nn.Linear(
                embedding.output_dim, embedding_dim, False)])
        else:
            self.pretrained = embedding
        self.fusion = Fusion(fusion_method, 2 if fusion_method == 'mix' else -1)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        embedding_trainable = self.trainable(input_ids)
        embedding_pretrained = self.pretrained(kwargs[self.key_pretrained])
        embedding = self.fusion(embedding_trainable, embedding_pretrained)
        return embedding
