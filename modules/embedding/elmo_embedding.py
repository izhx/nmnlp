"""
Elmo embedding
"""

from typing import List

import torch

from allennlp.modules.elmo import Elmo, batch_to_ids


class ElmoEmbedding(torch.nn.Module):
    """
    Packed elmo.
    """
    def __init__(self,
                 name_or_path: str,
                 num_output_representations: int = 1,
                 requires_grad: bool = False,
                 do_layer_norm: bool = False,
                 dropout: float = 0.5,
                 keep_sentence_boundaries: bool = False,
                 scalar_mix_parameters: List[float] = None,
                 module: torch.nn.Module = None):
        super().__init__()
        self.elmo = Elmo(name_or_path + '_options.json',
                         name_or_path + '_weights.hdf5',
                         num_output_representations,
                         requires_grad,
                         do_layer_norm,
                         dropout,
                         keep_sentence_boundaries=keep_sentence_boundaries,
                         scalar_mix_parameters=scalar_mix_parameters,
                         module=module)

    def forward(self, sentences: List, **kwargs):
        """
        sentences: ['aaaa', 'ohhhhhhhh']
        """
        character_ids = batch_to_ids(sentences)
        output = self.elmo(character_ids)
        return output['elmo_representations']
