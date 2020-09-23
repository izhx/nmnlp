"""
Elmo embedding
"""

from typing import List

import torch

try:
    from allennlp.modules.elmo import Elmo, batch_to_ids
except ModuleNotFoundError:
    import warnings
    warnings.warn('没有发现AllenNLP，Elmo将无法使用')


from .scir_elmo import Embedder


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
                 module: torch.nn.Module = None,
                 **kwargs):
        super().__init__()
        if 'scir' in name_or_path.lower():  # 垃圾scir elmo，一坨屎山
            self.elmo = Embedder(name_or_path)
            self.get_output = lambda x: x
            self.batch_to_ids = self.elmo.sents2elmo
        else:
            self.elmo = Elmo(name_or_path + 'options.json',
                             name_or_path + 'weights.hdf5',
                             num_output_representations,
                             requires_grad,
                             do_layer_norm,
                             dropout,
                             keep_sentence_boundaries=keep_sentence_boundaries,
                             scalar_mix_parameters=scalar_mix_parameters,
                             module=module)
            self.get_output = lambda x: x['elmo_representations'][0]
            self.batch_to_ids = batch_to_ids
        self.output_dim = self.elmo.get_output_dim()

    def forward(self, input_ids: torch.Tensor, sentences: List[List[str]], **kwargs):
        character_ids = self.batch_to_ids(sentences).to(input_ids.device)
        elmo_output = self.elmo(character_ids)  # 出来的mask和之前生成的一样的，没用
        return self.get_output(elmo_output)
