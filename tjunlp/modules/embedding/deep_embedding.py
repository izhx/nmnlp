"""
Packed embedding with encoder.
"""

from typing import Dict, Any

import torch

from ..encoder import build_encoder


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

    def forward(self, input_ids: torch.Tensor, **kwargs):  # pylint:disable=arguments-differ
        input_ids = self.embedding(input_ids)
        if self.encoder is not None:
            input_ids = self.encoder(input_ids, **kwargs)
        return input_ids
